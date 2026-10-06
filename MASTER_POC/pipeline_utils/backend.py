#!/usr/bin/env python3
"""
backend.py - Multi-Backend Inference Abstraction Layer

This module provides a unified interface for different inference backends,
enabling the pipeline to work with ONNX Runtime, llama.cpp, or PyTorch.

Usage:
    from pipeline_utils.backend import get_backend, BackendType, DeviceType
    
    backend = get_backend(BackendType.ONNX_RUNTIME)
    backend.load(model_path, device=DeviceType.GPU)
    output = backend.generate("Hello", max_tokens=50)
"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Sequence
import importlib
import json
import os
import re
import sys

import numpy as np


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
    
    # ONNX-specific
    onnx_providers: List[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    onnx_optimization_level: str = "DISABLE"
    
    # Memory
    low_memory: bool = False


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
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
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
        """Release resources. Override if needed."""
        self.model = None
        self.is_loaded = False
    
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

    @property
    def backend_type(self) -> BackendType:
        return BackendType.ONNX_RUNTIME

    @property
    def supported_formats(self) -> List[ModelFormat]:
        return [ModelFormat.ONNX]

    def load(self, model_path: str) -> None:
        """Load a split model directory or one ONNX file."""
        self.model_path = model_path
        self._is_split = os.path.isdir(model_path)
        self._pipeline = None
        if self._is_split:
            pipeline_cls = _sequential_pipeline_class()
            self._pipeline = pipeline_cls(
                model_path,
                providers=list(self.config.onnx_providers),
                verbose=False,
                persist_sessions=not self.config.low_memory,
            )
            self.model = getattr(self._pipeline, "session_manager", None) or self._pipeline
            self._tokenizer = getattr(self._pipeline, "tokenizer", None)
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
                providers=self.config.onnx_providers,
            )
            self._tokenizer = _load_tokenizer(model_path)
        self.is_loaded = True

    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text from a loaded ONNX model.

        A split model returns the text from ``SequentialModelPipeline.generate``.
        A monolithic session returns the decoded prompt plus the new tokens.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        if self._is_split:
            if self._pipeline is None:
                raise RuntimeError(
                    "Split ONNX model has no SequentialModelPipeline. "
                    "Call load() on a model directory."
                )
            return self._pipeline.generate(prompt, max_new_tokens=max_tokens)
        return self._monolithic_generate(prompt, max_tokens, temperature, top_p)

    def _monolithic_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
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
        past_state = {
            node.name: _empty_kv(node, batch)
            for node in inputs
            if node.name in past_names
        }
        generated: List[int] = []
        eos = getattr(self._tokenizer, "eos_token_id", None)
        history = prompt_ids
        for step in range(steps):
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
            if past_names:
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
            return usage
        return {
            "backend": "onnx_runtime",
            "loaded": self.is_loaded,
            "split": self._is_split,
        }

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
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text using llama.cpp."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=False,
        )
        
        return output["choices"][0]["text"]
    
    def get_memory_usage(self) -> Dict[str, Any]:
        # llama.cpp doesn't expose direct memory stats
        return {
            "backend": "llama_cpp",
            "n_gpu_layers": self.config.n_gpu_layers,
            "n_ctx": self.config.n_ctx,
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
        # SafeTensors/PyTorch would need torch backend (not implemented yet)
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
    # Functions
    "get_backend",
    "detect_model_format",
    "get_backend_for_model",
]
