#!/usr/bin/env python3
"""
backend.py - Multi-Backend Inference Abstraction Layer

This module provides a unified interface for different inference backends,
enabling the pipeline to work with ONNX Runtime, llama.cpp, PyTorch, or vLLM.

Usage:
    from pipeline_utils.backend import get_backend, BackendType, DeviceType
    
    backend = get_backend(
        BackendType.ONNX_RUNTIME,
        BackendConfig(device=DeviceType.NPU),
    )
    backend.load(model_path)
    output = backend.generate("Hello", max_tokens=50)
"""

from abc import ABC, abstractmethod
from contextlib import nullcontext
from enum import Enum
from dataclasses import dataclass, field, replace
from typing import Optional, List, Dict, Any, Union, Sequence, Tuple, Callable
import importlib
import inspect
import json
import os
import re
import sys
import warnings

import numpy as np

from .kv_cache import KVCacheManager


# =============================================================================
# ENUMS
# =============================================================================

class BackendType(Enum):
    """Supported inference backends."""
    ONNX_RUNTIME = "onnx_runtime"   # Current implementation (ONNX models)
    LLAMA_CPP = "llama_cpp"         # llama.cpp via llama-cpp-python (GGUF)
    TORCH = "torch"                 # Direct PyTorch (SafeTensors)
    VLLM = "vllm"                   # High-throughput server mode


class DeviceType(Enum):
    """Target device for inference."""
    CPU = "cpu"
    GPU = "gpu"       # CUDA/ROCm/MPS
    NPU = "npu"       # VitisAI/OpenVINO/CoreML
    AUTO = "auto"     # Best available


class ModelFormat(Enum):
    """Model file formats."""
    ONNX = "onnx"
    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    PYTORCH_BIN = "pytorch_bin"


# Preference order for OnnxRuntimeBackend. AUTO is detected from this table.
DEVICE_PROVIDER_MAP = {
    DeviceType.NPU: ["VitisAIExecutionProvider", "RyzenAIExecutionProvider", "CPUExecutionProvider"],
    DeviceType.GPU: ["ROCmExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"],
    DeviceType.CPU: ["CPUExecutionProvider"],
    DeviceType.AUTO: None,  # auto-detect
}

_AUTO_DEVICE_ORDER = (DeviceType.NPU, DeviceType.GPU, DeviceType.CPU)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BackendConfig:
    """Configuration for inference backends."""
    device: DeviceType = DeviceType.AUTO
    
    # GPU offloading (for hybrid execution)
    n_gpu_layers: int = -1  # -1 = all, 0 = none, N = first N layers
    
    # Context and generation
    n_ctx: int = 4096
    n_batch: int = 512
    
    # Threading
    n_threads: int = 0  # 0 = auto
    
    # Quantization (for llama.cpp)
    quantization: Optional[str] = None  # Q4_K_M, Q8_0, etc.
    
    # ONNX-specific. None uses DEVICE_PROVIDER_MAP for ``device``.
    onnx_providers: Optional[List[str]] = None
    onnx_optimization_level: str = "DISABLE"
    # Drop names this ONNX Runtime build cannot load before creating a session.
    use_available_providers_only: bool = True
    
    # Memory
    low_memory: bool = False

    # Hybrid execution. ``auto`` becomes hybrid when prefill and decode
    # devices differ, which is the default, or when ``npu_layer_range`` is set.
    execution_policy: str = "auto"  # monolithic, sequential, hybrid
    prefill_device: DeviceType = DeviceType.NPU
    decode_device: DeviceType = DeviceType.GPU
    npu_layer_range: Optional[Tuple[int, int]] = None


def _coerce_device(device: Union[DeviceType, str]) -> DeviceType:
    """Accept a DeviceType or its value, such as ``\"npu\"``."""
    if isinstance(device, DeviceType):
        return device
    text = str(device).split(".")[-1].strip().lower()
    try:
        return DeviceType(text)
    except ValueError as exc:
        known = ", ".join(item.value for item in DeviceType)
        raise ValueError(f"Unknown device '{device}'. Expected one of: {known}") from exc


def _available_execution_providers() -> List[str]:
    """Providers this ONNX Runtime build can actually run."""
    import onnxruntime as ort
    return list(ort.get_available_providers())


def _raw_providers_for_device(device: Union[DeviceType, str]) -> List[str]:
    """Device preference order, including providers this process cannot run."""
    kind = _coerce_device(device)
    if kind not in DEVICE_PROVIDER_MAP:
        raise ValueError(f"No provider map entry for {kind}")
    mapped = DEVICE_PROVIDER_MAP[kind]
    if mapped is not None:
        return list(mapped)
    selected: List[str] = []
    for candidate in _AUTO_DEVICE_ORDER:
        for provider in DEVICE_PROVIDER_MAP[candidate]:
            if provider == "CPUExecutionProvider" or provider in selected:
                continue
            selected.append(provider)
    selected.append("CPUExecutionProvider")
    return selected


def filter_onnx_providers(
    providers: Optional[Sequence[str]],
    available: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Keep requested providers that this ONNX Runtime build can load.

    ``available`` defaults to ``ort.get_available_providers()``. Order is
    preserved. An empty intersection becomes ``CPUExecutionProvider``. The
    second list is the provider names that were removed.
    """
    if available is None:
        available = _available_execution_providers()
    installed = set(available)
    kept: List[str] = []
    dropped: List[str] = []
    for provider in providers or []:
        name = str(provider)
        if name in installed:
            kept.append(name)
        else:
            dropped.append(name)
    if not kept:
        kept = ["CPUExecutionProvider"]
    return kept, dropped


def _warn_dropped_providers(dropped: Sequence[str], kept: Sequence[str]) -> None:
    """Tell the caller which names will not be passed to ONNX Runtime."""
    if not dropped:
        return
    warnings.warn(
        "Dropped unavailable execution providers: "
        + ", ".join(dropped)
        + ". Using "
        + ", ".join(kept)
        + ".",
        RuntimeWarning,
        stacklevel=3,
    )


def _filter_provider_map(
    mapping: Dict[str, List[str]],
    available: Sequence[str],
) -> Tuple[Dict[str, List[str]], List[str]]:
    """Filter every phase or component list with ``filter_onnx_providers``."""
    filtered: Dict[str, List[str]] = {}
    dropped: List[str] = []
    for name, providers in mapping.items():
        kept, gone = filter_onnx_providers(providers, available)
        filtered[name] = kept
        dropped.extend(gone)
    return filtered, dropped


def providers_for_device(
    device: Union[DeviceType, str],
    available: Optional[Sequence[str]] = None,
    filter_to_available: bool = True,
) -> List[str]:
    """
    Provider list for one device.

    Names follow ``DEVICE_PROVIDER_MAP`` and stay in that order. A provider
    this process cannot run is left out unless ``filter_to_available`` is
    false. ``AUTO`` walks NPU, then GPU, then CPU and keeps the installed
    accelerators, with CPU last.
    """
    raw = _raw_providers_for_device(device)
    if not filter_to_available:
        return raw
    if available is None:
        available = _available_execution_providers()
    kept, _dropped = filter_onnx_providers(raw, available)
    return kept


def resolve_onnx_providers(
    config: "BackendConfig",
    available: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Providers a backend should pass to ONNX Runtime.

    An ``onnx_providers`` list is the caller override. Otherwise the device
    map selects the list. When ``use_available_providers_only`` is true, names
    missing from ``ort.get_available_providers()`` are removed. An override
    whose names are all missing raises. When the flag is false, the requested
    list is returned unchanged.
    """
    if not config.use_available_providers_only:
        explicit = list(config.onnx_providers or [])
        if explicit:
            return explicit
        return _raw_providers_for_device(config.device)
    if available is None:
        available = _available_execution_providers()
    explicit = list(config.onnx_providers or [])
    if explicit:
        installed = set(available)
        if not any(provider in installed for provider in explicit):
            raise RuntimeError(
                f"None of the requested providers {explicit} are available. "
                f"Available: {list(available)}"
            )
        kept, _dropped = filter_onnx_providers(explicit, available)
        return kept
    return providers_for_device(config.device, available)


_EXECUTION_POLICY_ALIASES = {
    "auto": "auto",
    "monolithic": "monolithic",
    "monolithic_only": "monolithic",
    "sequential": "sequential",
    "split": "sequential",
    "split_cpu_memory_optimized": "sequential",
    "hybrid": "hybrid",
    "hybrid_npu_igpu": "hybrid",
    "hybrid_npu_igpu_phase_split": "hybrid",
    "hybrid_npu_moe_routing": "hybrid",
}


@dataclass
class HybridExecutionPlan:
    """
    Where each ONNX phase and component should run.

    ``policy`` is ``monolithic``, ``sequential``, or ``hybrid``. A hybrid
    plan puts prefill on ``prefill_device`` and decode on ``decode_device``.
    Layers inside ``npu_layer_range`` stay on the NPU and the other layers
    stay on CPU. The MoE router stays on CPU and experts stay on the NPU.
    """

    policy: str
    phase_providers: Dict[str, List[str]]
    component_providers: Dict[str, List[str]]
    npu_layer_range: Optional[Tuple[int, int]]
    prefill_device: DeviceType
    decode_device: DeviceType

    @property
    def persist_sessions(self) -> bool:
        """Hybrid and sequential loads recreate a session with that step's providers."""
        return self.policy == "monolithic"


def _coerce_layer_range(value: Optional[Sequence[int]]) -> Optional[Tuple[int, int]]:
    """Accept a half-open ``(start, end)`` span of transformer layers."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(
            "npu_layer_range must be (start, end), a half-open layer span"
        )
    start = int(value[0])
    end = int(value[1])
    if start < 0 or end <= start:
        raise ValueError(
            f"npu_layer_range must satisfy 0 <= start < end, got {(start, end)}"
        )
    return (start, end)


def resolve_execution_policy(config: "BackendConfig") -> str:
    """
    Resolve ``auto`` to monolithic, sequential, or hybrid.

    Hybrid is selected when prefill and decode target different devices or
    an NPU layer span is set. Sequential is selected for a low-memory load
    whose phase devices match. Anything else stays one device.
    """
    text = str(config.execution_policy or "auto").strip().lower().replace("-", "_")
    if text not in _EXECUTION_POLICY_ALIASES:
        allowed = "auto, monolithic, sequential, hybrid"
        raise ValueError(
            f"Unknown execution policy '{config.execution_policy}'. Expected one of: {allowed}"
        )
    chosen = _EXECUTION_POLICY_ALIASES[text]
    layer_range = _coerce_layer_range(config.npu_layer_range)
    if chosen == "monolithic" and layer_range is not None:
        raise ValueError(
            "npu_layer_range applies to hybrid or sequential execution, not monolithic"
        )
    if chosen != "auto":
        return chosen
    prefill = _coerce_device(config.prefill_device)
    decode = _coerce_device(config.decode_device)
    if layer_range is not None or prefill != decode:
        return "hybrid"
    if config.low_memory:
        return "sequential"
    return "monolithic"


def _hybrid_component_providers(
    config: "BackendConfig",
    available: Sequence[str],
    filter_to_available: bool = True,
) -> Tuple[Dict[str, List[str]], Optional[Tuple[int, int]]]:
    """Layer-range and MoE pins. Exact layer names win over the ``layer_*`` fallback."""
    components: Dict[str, List[str]] = {}
    span = _coerce_layer_range(config.npu_layer_range)
    cpu = providers_for_device(
        DeviceType.CPU, available, filter_to_available=filter_to_available
    )
    npu = providers_for_device(
        DeviceType.NPU, available, filter_to_available=filter_to_available
    )
    if span is not None:
        components["layer_*"] = list(cpu)
        start, end = span
        for index in range(start, end):
            components[f"layer_{index}"] = list(npu)
    for name in ("router", "moe_router", "router_*"):
        components[name] = list(cpu)
    for name in ("expert", "experts", "moe_expert", "expert_*", "experts_*", "moe_expert_*"):
        components[name] = list(npu)
    return components, span


def build_execution_plan(
    config: "BackendConfig",
    available: Optional[Sequence[str]] = None,
) -> HybridExecutionPlan:
    """
    Build the hybrid placement for one backend config.

    A monolithic plan leaves the provider maps empty. Hybrid and sequential
    plans record prefill, decode, the NPU layer span, and the MoE pins.
    """
    filter_to_available = config.use_available_providers_only
    if available is None and filter_to_available:
        available = _available_execution_providers()
    policy = resolve_execution_policy(config)
    prefill = _coerce_device(config.prefill_device)
    decode = _coerce_device(config.decode_device)
    if policy == "monolithic":
        return HybridExecutionPlan(
            policy=policy,
            phase_providers={},
            component_providers={},
            npu_layer_range=None,
            prefill_device=prefill,
            decode_device=decode,
        )
    installed = list(available or [])
    components, span = _hybrid_component_providers(
        config, installed, filter_to_available=filter_to_available
    )
    return HybridExecutionPlan(
        policy=policy,
        phase_providers={
            "prefill": providers_for_device(
                prefill, installed, filter_to_available=filter_to_available
            ),
            "decode": providers_for_device(
                decode, installed, filter_to_available=filter_to_available
            ),
        },
        component_providers=components,
        npu_layer_range=span,
        prefill_device=prefill,
        decode_device=decode,
    )


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================

class InferenceBackend(ABC):
    """
    Abstract base class for inference backends.
    
    All backends must implement load(), generate(), and cleanup().
    This enables swapping between ONNX, llama.cpp, PyTorch, etc.
    """
    
    def __init__(self, config: Optional[BackendConfig] = None):
        self.config = config or BackendConfig()
        self.model = None
        self.is_loaded = False
        self.model_path: Optional[str] = None
        self.kv_cache: Optional[KVCacheManager] = None
        self.warmed_up = False
    
    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Return the backend type enum."""
        pass
    
    @property
    @abstractmethod
    def supported_formats(self) -> List[ModelFormat]:
        """Return list of supported model formats."""
        pass
    
    @abstractmethod
    def load(self, model_path: str) -> None:
        """
        Load model from path.
        
        Args:
            model_path: Path to model file or directory
        """
        pass
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 50,
        stream_callback: Optional[Callable[[str, int], None]] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            stream_callback: Called with each new token's text and its
                0-based index as soon as that token is sampled. None
                leaves generation silent.
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        pass
    
    def cleanup(self) -> None:
        """Release the model. The KV cache object stays attached."""
        self.model = None
        self.is_loaded = False
        self.warmed_up = False

    def warmup(self, iterations: int = 2) -> Dict[str, Any]:
        """
        Pay the first-run compilation cost before timing.

        On an NPU that cost is graph compilation, XCLBIN generation, memory
        planning, operator fallback, and cache initialization. Each iteration
        resets the cache and generates two tokens from a fixed prompt, so
        prefill and one decode both run. The runs are not a latency result.
        The cache is cleared again afterward. Zero iterations leave the
        warmed flag unchanged and do not run the model.
        """
        if isinstance(iterations, bool) or not isinstance(iterations, int):
            raise TypeError(f"warmup iterations must be an int, got {iterations!r}")
        if iterations < 0:
            raise ValueError(f"warmup iterations must be >= 0, got {iterations}")
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        if iterations == 0:
            return {"iterations": 0, "warmed": bool(self.warmed_up)}
        for _ in range(iterations):
            self.reset_cache()
            self._warmup_once()
        self.reset_cache()
        self.warmed_up = True
        return {"iterations": iterations, "warmed": True}

    def _warmup_once(self) -> None:
        """One prefill plus one decode token. Subclasses may compile directly."""
        self.generate(_WARMUP_PROMPT, max_tokens=_WARMUP_NEW_TOKENS, temperature=0.0)

    def reset_cache(self) -> None:
        """Clear the attached KV cache. An unset cache is left unset."""
        cache = self.kv_cache
        if cache is not None and hasattr(cache, "reset"):
            cache.reset()
        pipeline = getattr(self, "_pipeline", None)
        pipeline_cache = getattr(pipeline, "kv_cache", None) if pipeline is not None else None
        if (
            pipeline_cache is not None
            and pipeline_cache is not cache
            and hasattr(pipeline_cache, "reset")
        ):
            pipeline_cache.reset()

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Report the attached KV cache.

        An unset cache reports ``attached`` false and zero length. An
        attached cache returns its memory report plus ``attached`` true.
        """
        cache = self.kv_cache
        if cache is None or not hasattr(cache, "get_memory_usage"):
            return {
                "attached": False,
                "sequence_length": 0,
                "current_seq_len": 0,
                "num_layers": 0,
                "total_bytes": 0,
                "total_mb": 0.0,
            }
        stats = dict(cache.get_memory_usage())
        stats["attached"] = True
        return stats

    def _kv_usage_fields(self) -> Dict[str, Any]:
        cache = self.kv_cache
        if cache is None:
            return {"kv_cache_attached": False, "kv_sequence_length": 0}
        length = getattr(cache, "get_sequence_length", None)
        return {
            "kv_cache_attached": True,
            "kv_sequence_length": int(length()) if callable(length) else 0,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(loaded={self.is_loaded}, device={self.config.device.value})"


# =============================================================================
# ONNX RUNTIME BACKEND
# =============================================================================

_OPTIMIZATION_LEVELS = {
    "DISABLE": "ORT_DISABLE_ALL",
    "BASIC": "ORT_ENABLE_BASIC",
    "EXTENDED": "ORT_ENABLE_EXTENDED",
    "ALL": "ORT_ENABLE_ALL",
}


def _sequential_pipeline_class():
    """
    Import SequentialModelPipeline when a split model is loaded.

    The pipeline module imports this package, so the import stays inside
    load() and this helper. A missing module path is retried from the
    MASTER_POC directory that sits next to pipeline_utils.
    """
    module_name = "grok_inference_split_model_pipeline"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent not in sys.path:
            sys.path.insert(0, parent)
        module = importlib.import_module(module_name)
    return module.SequentialModelPipeline


def _load_tokenizer(model_path: Optional[str]):
    """Load a Hugging Face tokenizer that sits beside a monolithic ONNX file."""
    if not model_path:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    candidates: List[str] = []
    if os.path.isdir(model_path):
        candidates.append(model_path)
    else:
        parent = os.path.dirname(os.path.abspath(model_path))
        candidates.append(parent)
        info_path = os.path.join(parent, "model_info.json")
        if os.path.isfile(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                payload = {}
            original = payload.get("original_model")
            if isinstance(original, str) and original:
                candidates.append(original if os.path.isdir(original) else os.path.dirname(original))
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen or not os.path.isdir(candidate):
            continue
        seen.add(candidate)
        try:
            return AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
        except Exception:
            continue
    return None


def _prompt_token_ids(tokenizer: Any, prompt: str) -> np.ndarray:
    """Tokenize a prompt to int64 input ids of shape [batch, seq]."""
    encoded = tokenizer(prompt, return_tensors="np")
    if isinstance(encoded, dict) or hasattr(encoded, "keys"):
        if "input_ids" not in encoded:
            raise RuntimeError("Tokenizer result has no input_ids")
        ids = np.asarray(encoded["input_ids"])
    else:
        ids = np.asarray(encoded)
    if ids.ndim == 1:
        ids = ids.reshape(1, -1)
    if ids.ndim != 2:
        raise RuntimeError(f"Tokenizer input_ids must be 2D, got shape {ids.shape}")
    return np.ascontiguousarray(ids, dtype=np.int64)


_WARMUP_PROMPT = "Hello"
_WARMUP_NEW_TOKENS = 2


def _synthetic_onnx_feeds(
    inputs: Sequence[Any],
    seq: int,
    past_len: int,
    batch: int = 1,
) -> Dict[str, np.ndarray]:
    """Concrete feeds for one warmup run when no tokenizer is available."""
    past_names = [node.name for node in inputs if "past" in node.name.lower()]
    token_ids = np.ones((batch, seq), dtype=np.int64)
    position = np.arange(past_len, past_len + seq, dtype=np.int64).reshape(1, -1)
    attention_length = past_len + seq if past_names else seq
    feeds: Dict[str, np.ndarray] = {}
    for node in inputs:
        name = node.name
        lower = name.lower()
        if "input_ids" in lower or lower in ("input", "tokens", "ids"):
            feeds[name] = token_ids
        elif "attention" in lower and "mask" in lower:
            feeds[name] = np.ones((batch, attention_length), dtype=np.int64)
        elif "position" in lower:
            feeds[name] = position
        elif name in past_names:
            array = _empty_kv(node, batch)
            if past_len > 0 and array.ndim >= 3:
                shape = list(array.shape)
                shape[2] = past_len
                array = np.zeros(shape, dtype=array.dtype)
            feeds[name] = array
        else:
            raise RuntimeError(
                f"Monolithic ONNX input '{name}' is not input_ids, "
                "an attention mask, position ids, or past KV"
            )
    return feeds


def _emit_stream_token(
    callback: Optional[Callable[[str, int], None]],
    text: str,
    index: int,
) -> None:
    """Report one generated piece. A missing callback does nothing."""
    if callback is None:
        return
    callback(text, int(index))


def _accepts_parameter(func: Any, name: str) -> bool:
    """True when ``func`` has an explicit parameter of this name."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    return name in signature.parameters


def _stream_llama_piece(chunk: Any) -> str:
    """Text from one llama.cpp stream chunk."""
    if isinstance(chunk, str):
        return chunk
    if not isinstance(chunk, dict):
        return ""
    choices = chunk.get("choices") or []
    if choices and isinstance(choices[0], dict):
        return str(choices[0].get("text") or "")
    if "text" in chunk:
        return str(chunk.get("text") or "")
    return ""


def _generate_with_pipeline(
    pipeline: Any,
    prompt: str,
    max_tokens: int,
    stream_callback: Optional[Callable[[str, int], None]],
) -> str:
    """
    Run a split pipeline.

    A pipeline that accepts ``stream_callback`` emits each token itself.
    Any other pipeline still returns its text, and this emits that text
    once at index 0 when a callback was requested.
    """
    generate = pipeline.generate
    if stream_callback is not None and _accepts_parameter(generate, "stream_callback"):
        return generate(
            prompt,
            max_new_tokens=max_tokens,
            stream_callback=stream_callback,
        )
    text = generate(prompt, max_new_tokens=max_tokens)
    if stream_callback is not None and text:
        stream_callback(text, 0)
    return text


def _decode_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
    ids = [int(token) for token in np.asarray(token_ids).reshape(-1)]
    text = tokenizer.decode(ids)
    if isinstance(text, str):
        return text
    return str(text)


def _sample_next_token(row: Any, temperature: float, top_p: float) -> int:
    """
    Pick one token from a logits row.

    A temperature of 0 selects the maximum logit. A positive temperature
    scales the logits, applies nucleus filtering when top_p is below 1,
    and draws one id.
    """
    scores = np.asarray(row, dtype=np.float64).reshape(-1)
    if scores.size < 1:
        raise RuntimeError("Logits row is empty")
    if temperature is None or float(temperature) <= 0:
        return int(np.argmax(scores))
    scaled = scores / float(temperature)
    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0:
        return int(np.argmax(scores))
    probs = probs / total
    if top_p is not None and 0 < float(top_p) < 1:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        keep_count = int(np.searchsorted(cumulative, float(top_p), side="left")) + 1
        keep_count = max(1, min(keep_count, int(order.size)))
        masked = np.zeros_like(probs)
        masked[order[:keep_count]] = probs[order[:keep_count]]
        masked_total = float(np.sum(masked))
        if masked_total <= 0:
            return int(np.argmax(scores))
        probs = masked / masked_total
    probs = np.clip(probs, 0, None)
    probs = probs / float(np.sum(probs))
    return int(np.random.choice(probs.size, p=probs))


def _kv_kind(name: str) -> str:
    lower = name.lower()
    if re.search(r"(?:^|[._])value$", lower):
        return "value"
    if re.search(r"(?:^|[._])key$", lower):
        return "key"
    return ""


def _kv_layer(name: str) -> Optional[int]:
    numbers = re.findall(r"\d+", name)
    if not numbers:
        return None
    return int(numbers[-1])


def _pair_present_to_past(present_name: str, past_names: Sequence[str]) -> Optional[str]:
    """Match a present output to the past input that should store it."""
    folded = (
        present_name
        .replace("present_key_values", "past_key_values")
        .replace("present", "past")
    )
    if folded in past_names:
        return folded
    kind = _kv_kind(present_name)
    layer = _kv_layer(present_name)
    matches = [
        past for past in past_names
        if _kv_kind(past) == kind and _kv_layer(past) == layer
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _node_dtype(node: Any) -> np.dtype:
    type_name = str(getattr(node, "type", "") or "").lower()
    if "float16" in type_name:
        return np.dtype(np.float16)
    if "bfloat16" in type_name:
        return np.dtype(np.float32)
    return np.dtype(np.float32)


def _empty_kv(node: Any, batch: int) -> np.ndarray:
    """A zero past cache with sequence length 0 for the first prefill step."""
    raw = list(getattr(node, "shape", None) or [])
    shape: List[int] = []
    for axis, dim in enumerate(raw):
        if isinstance(dim, (int, np.integer)) and int(dim) > 0:
            shape.append(int(dim))
        elif axis == 0:
            shape.append(int(batch))
        elif axis == 2:
            shape.append(0)
        else:
            shape.append(1)
    if len(shape) < 4:
        shape = [int(batch), 1, 0, 1]
    elif len(shape) > 4:
        shape = shape[:4]
    return np.zeros(shape, dtype=_node_dtype(node))


def _past_length(past_state: Dict[str, np.ndarray]) -> int:
    if not past_state:
        return 0
    array = next(iter(past_state.values()))
    if array.ndim < 3:
        return 0
    return int(array.shape[2])


def _select_logits(output_names: Sequence[str], values: Sequence[Any]) -> np.ndarray:
    paired = list(zip(output_names, values)) if output_names else [
        (str(index), value) for index, value in enumerate(values)
    ]
    for name, value in paired:
        if "logit" in str(name).lower():
            return np.asarray(value)
    for _, value in paired:
        array = np.asarray(value)
        if array.ndim >= 2:
            return array
    raise RuntimeError("ONNX session did not return logits")


def _logits_row(logits: np.ndarray) -> np.ndarray:
    if logits.ndim == 3:
        return logits[0, -1]
    if logits.ndim == 2:
        return logits[0]
    if logits.ndim == 1:
        return logits
    raise RuntimeError(f"Logits must be 1D, 2D, or 3D, got shape {logits.shape}")


def _store_presents(
    output_names: Sequence[str],
    values: Sequence[Any],
    past_names: Sequence[str],
    past_state: Dict[str, np.ndarray],
) -> None:
    """Write present KV into the past slots for the next decode step."""
    for name, value in zip(output_names, values):
        if "present" not in str(name).lower():
            continue
        past_name = _pair_present_to_past(str(name), past_names)
        if past_name is None:
            continue
        array = np.asarray(value)
        previous = past_state.get(past_name)
        if (
            previous is not None
            and array.ndim >= 3
            and previous.ndim >= 3
            and int(array.shape[2]) < int(previous.shape[2])
        ):
            array = np.concatenate([previous, array], axis=2)
        past_state[past_name] = array


def _host_kv(tensor: Any) -> np.ndarray:
    """NumPy view of a cache tensor, including a device buffer."""
    if isinstance(tensor, np.ndarray):
        return tensor
    numpy = getattr(tensor, "numpy", None)
    if callable(numpy):
        return np.asarray(numpy())
    to_numpy = getattr(tensor, "to_numpy", None)
    if callable(to_numpy):
        return np.asarray(to_numpy())
    return np.asarray(tensor)


def _valid_kv_prefix(cache: KVCacheManager, layer: int, kind: str) -> np.ndarray:
    """
    Feed only the tokens the cache has accepted.

    A static buffer is allocated at ``max_seq_len``. Length zero is a
    sequence axis of 0, not that full buffer.
    """
    key, value = cache.get_past_tensors(layer)
    array = _host_kv(key if kind == "key" else value)
    length = int(cache.layer_length(layer))
    if array.ndim >= 3 and int(array.shape[2]) != length:
        array = np.ascontiguousarray(array[:, :, :length, ...])
    return array


def _write_presents_to_cache(
    cache: KVCacheManager,
    output_names: Sequence[str],
    values: Sequence[Any],
    past_names: Sequence[str],
) -> None:
    """Store full present tensors with ``update_from_present``."""
    pending: Dict[int, Dict[str, np.ndarray]] = {}
    for name, value in zip(output_names, values):
        if "present" not in str(name).lower():
            continue
        past_name = _pair_present_to_past(str(name), past_names)
        if past_name is None:
            continue
        layer = _kv_layer(past_name)
        kind = _kv_kind(past_name)
        if layer is None or kind not in ("key", "value"):
            continue
        pending.setdefault(layer, {})[kind] = np.asarray(value)
    for layer, pair in pending.items():
        if "key" not in pair or "value" not in pair:
            continue
        cache.update_from_present(layer, pair["key"], pair["value"])


class OnnxRuntimeBackend(InferenceBackend):
    """
    ONNX Runtime backend for split or whole ONNX models.

    A model directory is a split model. ``load`` builds a
    ``SequentialModelPipeline`` and ``generate`` runs that pipeline.
    A single ``.onnx`` file stays one ``InferenceSession``. ``generate``
    tokenizes the prompt and runs prefill, then one decode step per new token.
    """

    def __init__(self, config: Optional[BackendConfig] = None):
        super().__init__(config)
        self._is_split = False
        self._pipeline = None
        self._tokenizer = None
        self.providers = resolve_onnx_providers(self.config)
        self.dropped_providers: List[str] = []
        self.execution_plan = build_execution_plan(self.config)

    @property
    def backend_type(self) -> BackendType:
        return BackendType.ONNX_RUNTIME

    @property
    def supported_formats(self) -> List[ModelFormat]:
        return [ModelFormat.ONNX]

    def load(self, model_path: str) -> None:
        """Load a split model directory or one ONNX file."""
        self.warmed_up = False
        self.model_path = model_path
        self._is_split = os.path.isdir(model_path)
        self._pipeline = None
        self.providers = resolve_onnx_providers(self.config)
        self.execution_plan = build_execution_plan(self.config)
        self._apply_available_providers()
        if self._is_split:
            pipeline_cls = _sequential_pipeline_class()
            plan = self.execution_plan
            persist = (not self.config.low_memory) and plan.persist_sessions
            self._pipeline = pipeline_cls(
                model_path,
                providers=list(self.providers),
                verbose=False,
                persist_sessions=persist,
                phase_providers=dict(plan.phase_providers) or None,
                component_providers=dict(plan.component_providers) or None,
            )
            self.model = getattr(self._pipeline, "session_manager", None) or self._pipeline
            self._tokenizer = getattr(self._pipeline, "tokenizer", None)
            self._share_pipeline_cache(self._pipeline)
        else:
            import onnxruntime as ort

            options = ort.SessionOptions()
            level_name = _OPTIMIZATION_LEVELS.get(
                str(self.config.onnx_optimization_level).upper()
            )
            level = getattr(ort.GraphOptimizationLevel, level_name, None) if level_name else None
            if level is not None:
                options.graph_optimization_level = level
            if self.config.n_threads > 0:
                options.intra_op_num_threads = int(self.config.n_threads)
            self.model = ort.InferenceSession(
                model_path,
                sess_options=options,
                providers=list(self.providers),
            )
            self._tokenizer = _load_tokenizer(model_path)
            if self.execution_plan.policy != "monolithic":
                warnings.warn(
                    "A single ONNX file uses one session. Prefill, decode, "
                    "layer-range, and MoE placement apply to a split model directory.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self.is_loaded = True

    def _apply_available_providers(self) -> None:
        """
        Check ``ort.get_available_providers()`` before a session is created.

        Phase and component lists are filtered with the same set. Dropped
        names are kept on ``dropped_providers`` and reported once. With
        ``use_available_providers_only`` off, the requested lists pass through.
        """
        if not self.config.use_available_providers_only:
            self.dropped_providers = []
            return
        available = _available_execution_providers()
        requested = (
            list(self.config.onnx_providers)
            if self.config.onnx_providers
            else _raw_providers_for_device(self.config.device)
        )
        _kept, requested_dropped = filter_onnx_providers(requested, available)
        plan = self.execution_plan
        if plan.policy != "monolithic":
            for device in (plan.prefill_device, plan.decode_device):
                _phase_kept, phase_gone = filter_onnx_providers(
                    _raw_providers_for_device(device), available
                )
                requested_dropped.extend(phase_gone)
            if plan.component_providers:
                for device in (DeviceType.NPU, DeviceType.CPU):
                    _pin_kept, pin_gone = filter_onnx_providers(
                        _raw_providers_for_device(device), available
                    )
                    requested_dropped.extend(pin_gone)
        self.providers, session_dropped = filter_onnx_providers(self.providers, available)
        phase, phase_dropped = _filter_provider_map(
            self.execution_plan.phase_providers, available
        )
        components, component_dropped = _filter_provider_map(
            self.execution_plan.component_providers, available
        )
        self.execution_plan = replace(
            self.execution_plan,
            phase_providers=phase,
            component_providers=components,
        )
        dropped: List[str] = []
        for name in (
            list(requested_dropped)
            + phase_dropped
            + component_dropped
            + session_dropped
        ):
            if name not in dropped:
                dropped.append(name)
        self.dropped_providers = dropped
        _warn_dropped_providers(dropped, self.providers)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        stream_callback: Optional[Callable[[str, int], None]] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text from a loaded ONNX model.

        A split model returns the text from ``SequentialModelPipeline.generate``.
        A monolithic session returns the decoded prompt plus the new tokens.
        ``stream_callback`` receives each new token as soon as it is sampled.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        self.reset_cache()
        if self._is_split:
            if self._pipeline is None:
                raise RuntimeError(
                    "Split ONNX model has no SequentialModelPipeline. "
                    "Call load() on a model directory."
                )
            return _generate_with_pipeline(
                self._pipeline, prompt, max_tokens, stream_callback
            )
        return self._monolithic_generate(
            prompt, max_tokens, temperature, top_p, stream_callback
        )

    def _monolithic_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream_callback: Optional[Callable[[str, int], None]] = None,
    ) -> str:
        """
        Prefill one ONNX session, then decode one token at a time.

        Inputs named like input ids, an attention mask, position ids, and
        past key/value tensors are filled from the tokenizer and the KV
        state. Present outputs become the past state of the next step.
        A session without past inputs receives the full token sequence on
        every step.
        """
        if self._tokenizer is None:
            self._tokenizer = _load_tokenizer(self.model_path)
        if self._tokenizer is None:
            raise RuntimeError(
                "Monolithic ONNX generation needs a tokenizer next to the model "
                "or assigned to OnnxRuntimeBackend._tokenizer."
            )
        steps = int(max_tokens)
        if steps < 0:
            raise ValueError(f"max_tokens must be >= 0, got {max_tokens}")
        session = self.model
        inputs = list(session.get_inputs())
        outputs = list(session.get_outputs())
        output_names = [node.name for node in outputs]
        input_names = [node.name for node in inputs]
        past_names = [name for name in input_names if "past" in name.lower()]
        prompt_ids = _prompt_token_ids(self._tokenizer, prompt)
        if int(prompt_ids.shape[1]) < 1:
            raise ValueError("Prompt tokenized to zero tokens")
        batch = int(prompt_ids.shape[0])
        self._ensure_monolithic_cache(inputs, past_names, batch)
        past_state = {
            node.name: _empty_kv(node, batch)
            for node in inputs
            if node.name in past_names and self.kv_cache is None
        }
        generated: List[int] = []
        eos = getattr(self._tokenizer, "eos_token_id", None)
        history = prompt_ids
        for step in range(steps):
            if self.kv_cache is not None and past_names:
                past_len = int(self.kv_cache.get_sequence_length())
            else:
                past_len = _past_length(past_state)
            if past_names:
                if step == 0:
                    token_ids = prompt_ids
                else:
                    token_ids = np.array([[generated[-1]]], dtype=np.int64)
                position = np.arange(
                    past_len,
                    past_len + int(token_ids.shape[1]),
                    dtype=np.int64,
                ).reshape(1, -1)
            else:
                if generated:
                    history = np.concatenate(
                        [prompt_ids, np.array([generated], dtype=np.int64)],
                        axis=1,
                    )
                token_ids = history
                position = np.arange(int(token_ids.shape[1]), dtype=np.int64).reshape(1, -1)
            attention_length = past_len + int(token_ids.shape[1]) if past_names else int(token_ids.shape[1])
            feeds: Dict[str, np.ndarray] = {}
            for node in inputs:
                name = node.name
                lower = name.lower()
                if "input_ids" in lower or lower in ("input", "tokens", "ids"):
                    feeds[name] = token_ids
                elif "attention" in lower and "mask" in lower:
                    feeds[name] = np.ones((batch, attention_length), dtype=np.int64)
                elif "position" in lower:
                    feeds[name] = position
                elif name in past_names and self.kv_cache is not None:
                    layer = _kv_layer(name)
                    kind = _kv_kind(name)
                    if layer is None or kind not in ("key", "value"):
                        raise RuntimeError(
                            f"Monolithic ONNX past input '{name}' has no layer or KV kind"
                        )
                    feeds[name] = _valid_kv_prefix(self.kv_cache, layer, kind)
                elif name in past_state:
                    feeds[name] = past_state[name]
                else:
                    raise RuntimeError(
                        f"Monolithic ONNX input '{name}' is not input_ids, "
                        "an attention mask, position ids, or past KV"
                    )
            values = session.run(output_names or None, feeds)
            if not values:
                raise RuntimeError("ONNX session returned no outputs")
            next_id = _sample_next_token(
                _logits_row(_select_logits(output_names, values)),
                temperature,
                top_p,
            )
            generated.append(next_id)
            _emit_stream_token(
                stream_callback,
                _decode_ids(self._tokenizer, [next_id]),
                len(generated) - 1,
            )
            if past_names and self.kv_cache is not None:
                _write_presents_to_cache(
                    self.kv_cache, output_names, values, past_names
                )
            elif past_names:
                _store_presents(output_names, values, past_names, past_state)
            if eos is not None and next_id == int(eos):
                break
        if generated:
            full_ids = np.concatenate(
                [prompt_ids, np.array([generated], dtype=np.int64)],
                axis=1,
            )
        else:
            full_ids = prompt_ids
        return _decode_ids(self._tokenizer, full_ids[0])

    def get_memory_usage(self) -> Dict[str, Any]:
        manager = None
        if self._pipeline is not None:
            manager = getattr(self._pipeline, "session_manager", None)
        source = manager if manager is not None else self.model
        if hasattr(source, "get_memory_usage"):
            usage = dict(source.get_memory_usage())
            usage["split"] = self._is_split
            usage["providers"] = list(self.providers)
            usage["execution_policy"] = self.execution_plan.policy
            usage.update(self._kv_usage_fields())
            usage["warmed_up"] = self.warmed_up
            return usage
        return {
            "backend": "onnx_runtime",
            "loaded": self.is_loaded,
            "split": self._is_split,
            "providers": list(self.providers),
            "execution_policy": self.execution_plan.policy,
            "warmed_up": self.warmed_up,
            **self._kv_usage_fields(),
        }

    def _warmup_once(self) -> None:
        """Compile prefill and decode. A session with no tokenizer uses synthetic feeds."""
        if not self._is_split and self._tokenizer is None:
            loaded = _load_tokenizer(self.model_path)
            if loaded is not None:
                self._tokenizer = loaded
        if self._is_split or self._tokenizer is not None:
            self.generate(_WARMUP_PROMPT, max_tokens=_WARMUP_NEW_TOKENS, temperature=0.0)
            return
        self._warmup_anonymous_session()

    def _warmup_anonymous_session(self) -> None:
        """Run two concrete shapes so an NPU compiles prefill and the next step."""
        session = self.model
        if session is None or not hasattr(session, "run"):
            raise RuntimeError("Monolithic ONNX warmup needs a loaded InferenceSession.")
        inputs = list(session.get_inputs()) if hasattr(session, "get_inputs") else []
        if not inputs:
            session.run(None, {})
            return
        past_names = [node.name for node in inputs if "past" in node.name.lower()]
        shapes = [(1, 0), (1, 1)] if past_names else [(1, 0), (2, 0)]
        for seq, past_len in shapes:
            session.run(None, _synthetic_onnx_feeds(inputs, seq, past_len))

    def _share_pipeline_cache(self, pipeline: Any) -> None:
        """
        Use one KV cache for this backend and a split pipeline.

        A cache already set on the backend is installed on the pipeline.
        Otherwise the pipeline cache is adopted.
        """
        pipeline_cache = getattr(pipeline, "kv_cache", None)
        if self.kv_cache is None:
            if pipeline_cache is not None:
                self.kv_cache = pipeline_cache
            return
        if pipeline_cache is not self.kv_cache:
            pipeline.kv_cache = self.kv_cache

    def _ensure_monolithic_cache(
        self,
        inputs: Sequence[Any],
        past_names: Sequence[str],
        batch: int,
    ) -> None:
        """Build a CPU KV cache from past-input shapes when one is unset."""
        if self.kv_cache is not None or not past_names:
            return
        grouped: Dict[int, Any] = {}
        for node in inputs:
            if node.name not in past_names:
                continue
            layer = _kv_layer(node.name)
            kind = _kv_kind(node.name)
            if layer is None or kind not in ("key", "value"):
                continue
            grouped.setdefault(layer, node)
        if not grouped:
            return
        sample = _empty_kv(next(iter(grouped.values())), batch)
        limit = int(self.config.n_ctx) if int(self.config.n_ctx) > 0 else 4096
        self.kv_cache = KVCacheManager(
            num_layers=max(grouped) + 1,
            num_kv_heads=int(sample.shape[1]),
            head_dim=int(sample.shape[3]),
            dtype=sample.dtype,
            batch_size=int(sample.shape[0]),
            max_seq_len=limit,
        )

    def cleanup(self) -> None:
        """Unload a split pipeline, then drop the session."""
        pipeline = self._pipeline
        self._pipeline = None
        if pipeline is not None and hasattr(pipeline, "cleanup"):
            pipeline.cleanup()
        super().cleanup()


# =============================================================================
# LLAMA.CPP BACKEND
# =============================================================================

class LlamaCppBackend(InferenceBackend):
    """
    llama.cpp backend via llama-cpp-python bindings.
    
    Supports GGUF format models with GPU offloading.
    """
    
    @property
    def backend_type(self) -> BackendType:
        return BackendType.LLAMA_CPP
    
    @property
    def supported_formats(self) -> List[ModelFormat]:
        return [ModelFormat.GGUF]
    
    def load(self, model_path: str) -> None:
        """Load GGUF model."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for LlamaCppBackend. "
                "Install with: pip install llama-cpp-python"
            )
        
        self.model_path = model_path
        
        # Determine GPU layers based on device config
        n_gpu_layers = 0
        if self.config.device == DeviceType.GPU:
            n_gpu_layers = self.config.n_gpu_layers if self.config.n_gpu_layers >= 0 else 999
        elif self.config.device == DeviceType.AUTO:
            n_gpu_layers = self.config.n_gpu_layers if self.config.n_gpu_layers >= 0 else -1
        
        self.warmed_up = False
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=self.config.n_ctx,
            n_batch=self.config.n_batch,
            n_threads=self.config.n_threads if self.config.n_threads > 0 else None,
            verbose=False,
        )
        
        self.is_loaded = True
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 50,
        stream_callback: Optional[Callable[[str, int], None]] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text using llama.cpp. A callback receives each streamed piece."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if stream_callback is None:
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                echo=False,
            )
            return output["choices"][0]["text"]

        try:
            stream = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                echo=False,
                stream=True,
            )
        except TypeError:
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                echo=False,
            )
            text = output["choices"][0]["text"]
            if text:
                stream_callback(text, 0)
            return text

        pieces: List[str] = []
        index = 0
        for chunk in stream:
            text = _stream_llama_piece(chunk)
            if not text:
                continue
            stream_callback(text, index)
            pieces.append(text)
            index += 1
        return "".join(pieces)

    def reset_cache(self) -> None:
        """Clear the KV cache and llama.cpp's own state when it has one."""
        super().reset_cache()
        model = self.model
        reset = getattr(model, "reset", None)
        if callable(reset):
            reset()
    
    def get_memory_usage(self) -> Dict[str, Any]:
        # llama.cpp doesn't expose direct memory stats
        return {
            "backend": "llama_cpp",
            "n_gpu_layers": self.config.n_gpu_layers,
            "n_ctx": self.config.n_ctx,
            "warmed_up": self.warmed_up,
            **self._kv_usage_fields(),
        }


def _pretrained_source(model_path: str) -> str:
    """Use a weight file's directory as the Hugging Face load path."""
    if os.path.isdir(model_path):
        return model_path
    parent = os.path.dirname(os.path.abspath(model_path))
    return parent or model_path


def _import_torch_stack():
    """Import torch and transformers, or raise an install hint."""
    try:
        torch_mod = importlib.import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "torch is required for TorchBackend. "
            "Install with: pip install torch"
        ) from exc
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise ImportError(
            "transformers is required for TorchBackend. "
            "Install with: pip install transformers"
        ) from exc
    if not hasattr(transformers, "AutoModelForCausalLM") or not hasattr(transformers, "AutoTokenizer"):
        raise ImportError(
            "transformers.AutoModelForCausalLM and AutoTokenizer are required "
            "for TorchBackend. Install with: pip install transformers"
        )
    return torch_mod, transformers


def _import_vllm():
    """Import vLLM, or raise an install hint."""
    try:
        return importlib.import_module("vllm")
    except ImportError as exc:
        raise ImportError(
            "vllm is required for VLLMBackend. "
            "Install with: pip install vllm"
        ) from exc


def _call_filtered(func: Any, kwargs: Dict[str, Any]) -> Any:
    """Call ``func`` with the kwargs its signature accepts."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**kwargs)
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return func(**kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in parameters}
    return func(**accepted)


def _accelerator_kind(torch_mod: Any) -> Optional[str]:
    """The first live accelerator this PyTorch build exposes."""
    if torch_mod is None:
        return None
    cuda = getattr(torch_mod, "cuda", None)
    is_cuda = getattr(cuda, "is_available", None)
    if callable(is_cuda) and is_cuda():
        return "cuda"
    backends = getattr(torch_mod, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    is_mps = getattr(mps, "is_available", None)
    if callable(is_mps) and is_mps():
        return "mps"
    npu = getattr(torch_mod, "npu", None)
    is_npu = getattr(npu, "is_available", None)
    if callable(is_npu) and is_npu():
        return "npu"
    return None


def _torch_device_map(device: Union[DeviceType, str], torch_mod: Any) -> str:
    """Map DeviceType onto a ``from_pretrained`` device_map value."""
    kind = _accelerator_kind(torch_mod)
    selected = _coerce_device(device)
    if selected == DeviceType.CPU:
        return "cpu"
    if selected == DeviceType.GPU:
        if kind == "cuda":
            return "cuda"
        if kind == "mps":
            return "mps"
        warnings.warn(
            "TorchBackend placed the model on CPU because this PyTorch build "
            "has no CUDA or MPS device.",
            RuntimeWarning,
            stacklevel=3,
        )
        return "cpu"
    if selected == DeviceType.NPU:
        if kind == "npu":
            return "npu"
        warnings.warn(
            "TorchBackend placed the model on CPU because this PyTorch build "
            "has no NPU device.",
            RuntimeWarning,
            stacklevel=3,
        )
        return "cpu"
    if kind == "cuda":
        return "auto"
    if kind in ("mps", "npu"):
        return kind
    return "cpu"


_TORCH_DTYPE_NAMES = {
    "float32": "float32",
    "fp32": "float32",
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
}


def _torch_from_pretrained_kwargs(
    config: BackendConfig,
    torch_mod: Any,
    device_map: str,
) -> Dict[str, Any]:
    """dtype, bitsandbytes flags, and device_map for one Torch load."""
    kwargs: Dict[str, Any] = {
        "low_cpu_mem_usage": bool(config.low_memory),
        "device_map": device_map,
    }
    quant = str(config.quantization or "").strip().lower()
    if quant in ("int8", "8bit", "load_in_8bit"):
        kwargs["load_in_8bit"] = True
    elif quant in ("int4", "4bit", "load_in_4bit"):
        kwargs["load_in_4bit"] = True
    elif quant in _TORCH_DTYPE_NAMES:
        kwargs["torch_dtype"] = getattr(torch_mod, _TORCH_DTYPE_NAMES[quant])
    elif quant:
        warnings.warn(
            f"TorchBackend does not apply quantization {config.quantization!r}. "
            "Use fp16, bf16, fp32, int8, or int4.",
            RuntimeWarning,
            stacklevel=3,
        )
    elif device_map in ("cuda", "npu", "mps", "auto"):
        dtype = getattr(torch_mod, "float16", None)
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
    return kwargs


def _tensor_to_id_list(value: Any) -> List[int]:
    """Flatten a tensor or array of token ids."""
    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "numpy"):
        current = current.numpy()
    return [int(token) for token in np.asarray(current).reshape(-1)]


def _ids_tensor(torch_mod: Any, token_ids: Sequence[int], device: str) -> Any:
    """A [1, seq] long tensor on ``device``."""
    column = torch_mod.tensor(
        [[int(token) for token in token_ids]],
        dtype=getattr(torch_mod, "long", None),
    )
    if device and hasattr(column, "to"):
        column = column.to(device)
    return column


def _torch_inference(torch_mod: Any):
    """``torch.no_grad`` when the build provides it."""
    no_grad = getattr(torch_mod, "no_grad", None)
    if not callable(no_grad):
        return nullcontext()
    return no_grad()


def _torch_forward(model: Any, input_ids: Any, attention_mask: Any, past: Any) -> Any:
    """One causal-LM step. A model without cache kwargs receives ids only."""
    kwargs: Dict[str, Any] = {"input_ids": input_ids, "use_cache": True}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if past is not None:
        kwargs["past_key_values"] = past
    try:
        return model(**kwargs)
    except TypeError:
        return model(input_ids)


def _row_from_torch_logits(outputs: Any) -> np.ndarray:
    """The last-position logits row as a 1D numpy array."""
    logits = getattr(outputs, "logits", None)
    if logits is None and isinstance(outputs, (tuple, list)) and outputs:
        logits = outputs[0]
    if logits is None:
        raise RuntimeError("Torch model returned no logits")
    if hasattr(logits, "detach"):
        logits = logits.detach()
    if hasattr(logits, "float"):
        logits = logits.float()
    if hasattr(logits, "cpu"):
        logits = logits.cpu()
    if hasattr(logits, "numpy"):
        logits = logits.numpy()
    array = np.asarray(logits)
    if array.ndim >= 2:
        array = array.reshape(-1, array.shape[-1])[-1]
    return np.asarray(array, dtype=np.float64).reshape(-1)


def _vllm_device(device: Union[DeviceType, str]) -> Optional[str]:
    """Device string for ``vllm.LLM``. AUTO leaves the choice to vLLM."""
    selected = _coerce_device(device)
    if selected == DeviceType.CPU:
        return "cpu"
    if selected == DeviceType.GPU:
        return "cuda"
    if selected == DeviceType.NPU:
        warnings.warn(
            "VLLMBackend is serving this NPU request on CPU.",
            RuntimeWarning,
            stacklevel=3,
        )
        return "cpu"
    return None


_VLLM_DTYPES = {
    "float32": "float32",
    "fp32": "float32",
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "auto": "auto",
}
_VLLM_QUANTS = {
    "awq": "awq",
    "gptq": "gptq",
    "squeezellm": "squeezellm",
    "fp8": "fp8",
}


def _vllm_precision(quantization: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(dtype, quantization)`` for one vLLM load."""
    name = str(quantization or "").strip().lower()
    if not name:
        return None, None
    if name in _VLLM_QUANTS:
        return None, _VLLM_QUANTS[name]
    if name in _VLLM_DTYPES:
        return _VLLM_DTYPES[name], None
    warnings.warn(
        f"VLLMBackend does not apply quantization {quantization!r}. "
        "Use fp16, bf16, fp32, awq, gptq, or fp8.",
        RuntimeWarning,
        stacklevel=3,
    )
    return None, None


def _vllm_sampling_params(vllm_mod: Any, max_tokens: int, temperature: float, top_p: float) -> Any:
    """Build SamplingParams, keeping only fields this vLLM build accepts."""
    kwargs = {
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    params_cls = getattr(vllm_mod, "SamplingParams", None)
    if params_cls is None:
        return kwargs
    return _call_filtered(params_cls, kwargs)


def _vllm_completion(outputs: Any) -> Tuple[str, List[int]]:
    """Completion text and token ids from one ``LLM.generate`` result."""
    if outputs is None:
        return "", []
    first = outputs[0] if isinstance(outputs, (list, tuple)) and outputs else outputs
    if isinstance(first, str):
        return first, []
    if isinstance(first, dict):
        text = str(first.get("text") or "")
        token_ids = [int(token) for token in list(first.get("token_ids") or [])]
        return text, token_ids
    completions = getattr(first, "outputs", None)
    if completions:
        completion = completions[0]
        if isinstance(completion, dict):
            text = str(completion.get("text") or "")
            token_ids = [int(token) for token in list(completion.get("token_ids") or [])]
            return text, token_ids
        text = str(getattr(completion, "text", "") or "")
        token_ids = [int(token) for token in list(getattr(completion, "token_ids", None) or [])]
        return text, token_ids
    text = str(getattr(first, "text", "") or "")
    token_ids = [int(token) for token in list(getattr(first, "token_ids", None) or [])]
    return text, token_ids


# =============================================================================
# PYTORCH BACKEND
# =============================================================================

class TorchBackend(InferenceBackend):
    """
    Direct PyTorch backend for SafeTensors and pytorch_model.bin.

    ``load`` builds a causal LM with ``from_pretrained``. ``generate``
    runs one cached forward per new token and samples with the same
    temperature and top-p rules as the ONNX loop.
    """

    def __init__(self, config: Optional[BackendConfig] = None):
        super().__init__(config)
        self._tokenizer = None
        self._past_key_values = None
        self._torch = None
        self._device = self.config.device.value

    @property
    def backend_type(self) -> BackendType:
        return BackendType.TORCH

    @property
    def supported_formats(self) -> List[ModelFormat]:
        return [ModelFormat.SAFETENSORS, ModelFormat.PYTORCH_BIN]

    def load(self, model_path: str) -> None:
        """Load a Hugging Face directory or a single weight file."""
        torch_mod, transformers = _import_torch_stack()
        source = _pretrained_source(model_path)
        self.warmed_up = False
        self.model_path = model_path
        self._past_key_values = None
        if self.config.n_threads > 0 and callable(getattr(torch_mod, "set_num_threads", None)):
            torch_mod.set_num_threads(int(self.config.n_threads))
        self._device = _torch_device_map(self.config.device, torch_mod)
        self._torch = torch_mod
        model = transformers.AutoModelForCausalLM.from_pretrained(
            source,
            **_torch_from_pretrained_kwargs(self.config, torch_mod, self._device),
        )
        if hasattr(model, "eval"):
            model.eval()
        self.model = model
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(source)
        self.is_loaded = True

    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        stream_callback: Optional[Callable[[str, int], None]] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text with a loaded PyTorch model.

        The return value is the decoded prompt plus the new tokens.
        ``stream_callback`` receives each new token as soon as it is sampled.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        steps = int(max_tokens)
        if steps < 0:
            raise ValueError(f"max_tokens must be >= 0, got {max_tokens}")
        self.reset_cache()
        if self._tokenizer is None:
            raise RuntimeError(
                "Torch generation needs a tokenizer. Call load() on a model directory."
            )
        torch_mod = self._torch
        if torch_mod is None:
            torch_mod, _transformers = _import_torch_stack()
            self._torch = torch_mod
        encoded = self._tokenizer(prompt, return_tensors="pt")
        if not isinstance(encoded, dict) and not hasattr(encoded, "keys"):
            raise RuntimeError("Tokenizer result has no input_ids")
        if "input_ids" not in encoded:
            raise RuntimeError("Tokenizer result has no input_ids")
        all_ids = _tensor_to_id_list(encoded["input_ids"])
        if not all_ids:
            raise ValueError("Prompt tokenized to zero tokens")
        eos = getattr(self._tokenizer, "eos_token_id", None)
        past = None
        model = self.model
        with _torch_inference(torch_mod):
            for index in range(steps):
                if past is None:
                    step_ids = _ids_tensor(torch_mod, all_ids, self._device)
                else:
                    step_ids = _ids_tensor(torch_mod, [all_ids[-1]], self._device)
                step_mask = _ids_tensor(torch_mod, [1] * len(all_ids), self._device)
                outputs = _torch_forward(model, step_ids, step_mask, past)
                past = getattr(outputs, "past_key_values", None)
                self._past_key_values = past
                next_id = _sample_next_token(
                    _row_from_torch_logits(outputs),
                    temperature,
                    top_p,
                )
                all_ids.append(next_id)
                _emit_stream_token(
                    stream_callback,
                    _decode_ids(self._tokenizer, [next_id]),
                    index,
                )
                if eos is not None and next_id == int(eos):
                    break
        return _decode_ids(self._tokenizer, all_ids)

    def reset_cache(self) -> None:
        """Clear the attached KV cache and the PyTorch past state."""
        super().reset_cache()
        self._past_key_values = None

    def cleanup(self) -> None:
        """Drop the module and the tokenizer."""
        self._tokenizer = None
        self._past_key_values = None
        self._torch = None
        super().cleanup()

    def get_memory_usage(self) -> Dict[str, Any]:
        return {
            "backend": "torch",
            "device": self._device,
            "loaded": self.is_loaded,
            "n_ctx": self.config.n_ctx,
            "warmed_up": self.warmed_up,
            **self._kv_usage_fields(),
        }


# =============================================================================
# VLLM BACKEND
# =============================================================================

class VLLMBackend(InferenceBackend):
    """
    vLLM backend for high-throughput Hugging Face checkpoints.

    The same SafeTensors directory stays on ``TorchBackend`` when the
    format is detected automatically. Construct this backend with
    ``BackendType.VLLM`` when the server-style engine should run it.
    """

    def __init__(self, config: Optional[BackendConfig] = None):
        super().__init__(config)
        self._device = self.config.device.value

    @property
    def backend_type(self) -> BackendType:
        return BackendType.VLLM

    @property
    def supported_formats(self) -> List[ModelFormat]:
        return [ModelFormat.SAFETENSORS, ModelFormat.PYTORCH_BIN]

    def load(self, model_path: str) -> None:
        """Load a checkpoint into ``vllm.LLM``."""
        vllm_mod = _import_vllm()
        llm_cls = getattr(vllm_mod, "LLM", None)
        if llm_cls is None:
            raise ImportError(
                "vllm.LLM is required for VLLMBackend. "
                "Install with: pip install vllm"
            )
        source = _pretrained_source(model_path)
        self.warmed_up = False
        self.model_path = model_path
        device = _vllm_device(self.config.device)
        self._device = device or self.config.device.value
        dtype, quant = _vllm_precision(self.config.quantization)
        kwargs: Dict[str, Any] = {
            "model": source,
            "tensor_parallel_size": 1,
        }
        if int(self.config.n_ctx) > 0:
            kwargs["max_model_len"] = int(self.config.n_ctx)
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        if quant is not None:
            kwargs["quantization"] = quant
        if self.config.low_memory:
            kwargs["gpu_memory_utilization"] = 0.5
        self.model = _call_filtered(llm_cls, kwargs)
        self.is_loaded = True

    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        stream_callback: Optional[Callable[[str, int], None]] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate a completion with vLLM.

        The return value is the new text only. A callback receives one
        decoded piece per completion token id. Without token ids, the
        callback receives the full completion once at index 0.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        steps = int(max_tokens)
        if steps < 0:
            raise ValueError(f"max_tokens must be >= 0, got {max_tokens}")
        self.reset_cache()
        if steps == 0:
            return ""
        vllm_mod = _import_vllm()
        params = _vllm_sampling_params(vllm_mod, steps, temperature, top_p)
        outputs = self.model.generate([prompt], params)
        text, token_ids = _vllm_completion(outputs)
        if stream_callback is not None:
            tokenizer = None
            getter = getattr(self.model, "get_tokenizer", None)
            if token_ids and callable(getter):
                tokenizer = getter()
            if token_ids and tokenizer is not None and hasattr(tokenizer, "decode"):
                for index, token_id in enumerate(token_ids):
                    stream_callback(_decode_ids(tokenizer, [token_id]), index)
            elif text:
                stream_callback(text, 0)
        return text

    def get_memory_usage(self) -> Dict[str, Any]:
        return {
            "backend": "vllm",
            "device": self._device,
            "loaded": self.is_loaded,
            "n_ctx": self.config.n_ctx,
            "warmed_up": self.warmed_up,
            **self._kv_usage_fields(),
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_backend(
    backend_type: BackendType,
    config: Optional[BackendConfig] = None
) -> InferenceBackend:
    """
    Factory function to get appropriate backend.
    
    Args:
        backend_type: Type of backend to create
        config: Optional configuration
        
    Returns:
        Configured InferenceBackend instance
    """
    backends = {
        BackendType.ONNX_RUNTIME: OnnxRuntimeBackend,
        BackendType.LLAMA_CPP: LlamaCppBackend,
        BackendType.TORCH: TorchBackend,
        BackendType.VLLM: VLLMBackend,
    }
    
    if backend_type not in backends:
        raise ValueError(f"Unsupported backend: {backend_type}. Available: {list(backends.keys())}")
    
    return backends[backend_type](config)


def detect_model_format(model_path: str) -> ModelFormat:
    """
    Detect model format from path.
    
    Args:
        model_path: Path to model file or directory
        
    Returns:
        Detected ModelFormat
    """
    if os.path.isdir(model_path):
        # Check for split ONNX
        if os.path.exists(os.path.join(model_path, "model_info.json")):
            return ModelFormat.ONNX
        # Check for SafeTensors
        if any(f.endswith(".safetensors") for f in os.listdir(model_path)):
            return ModelFormat.SAFETENSORS
    else:
        ext = os.path.splitext(model_path)[1].lower()
        if ext == ".onnx":
            return ModelFormat.ONNX
        elif ext == ".gguf":
            return ModelFormat.GGUF
        elif ext == ".safetensors":
            return ModelFormat.SAFETENSORS
        elif ext == ".bin":
            return ModelFormat.PYTORCH_BIN
    
    raise ValueError(f"Cannot detect format for: {model_path}")


def get_backend_for_model(
    model_path: str,
    config: Optional[BackendConfig] = None
) -> InferenceBackend:
    """
    Automatically select and configure backend based on model format.
    
    Args:
        model_path: Path to model
        config: Optional configuration
        
    Returns:
        Appropriate backend for the model format
    """
    fmt = detect_model_format(model_path)
    
    format_to_backend = {
        ModelFormat.ONNX: BackendType.ONNX_RUNTIME,
        ModelFormat.GGUF: BackendType.LLAMA_CPP,
        ModelFormat.SAFETENSORS: BackendType.TORCH,
        ModelFormat.PYTORCH_BIN: BackendType.TORCH,
    }
    
    backend_type = format_to_backend.get(fmt)
    if not backend_type:
        raise ValueError(f"No backend available for format: {fmt}")
    
    return get_backend(backend_type, config)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "BackendType",
    "DeviceType",
    "ModelFormat",
    # Classes
    "BackendConfig",
    "InferenceBackend",
    "OnnxRuntimeBackend",
    "LlamaCppBackend",
    "TorchBackend",
    "VLLMBackend",
    # Functions
    "get_backend",
    "detect_model_format",
    "get_backend_for_model",
    "providers_for_device",
    "filter_onnx_providers",
    "resolve_onnx_providers",
    "resolve_execution_policy",
    "build_execution_plan",
    # Provider map and hybrid plan
    "DEVICE_PROVIDER_MAP",
    "HybridExecutionPlan",
]
