"""
ONNX model inference utilities.

This module provides the production inference engine for split and merged
ONNX models. It uses pipeline_utils for session lifetime, KV cache, I/O
binding, and configuration loading.

Supports:
- SEQUENTIAL mode: load and release each component (memory-efficient)
- PERSISTENT mode: load every session once (NPU and GPU)
- IO_BOUND mode: persistent sessions plus device buffers

Usage:
    from chunking_utils.onnx.inference import SplitModelInferenceEngine, InferenceConfig

    engine = SplitModelInferenceEngine(
        model_dir="./split_output/my_model",
        config=InferenceConfig(
            providers=["VitisAIExecutionProvider"],
            mode=InferenceMode.PERSISTENT,
        ),
    )
"""

from __future__ import annotations

import gc
import json
import os
import re
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from pipeline_utils.session_manager import (
        SessionManager,
        SessionConfig,
        get_provider_for_device,
    )
    from pipeline_utils.kv_cache import KVCacheManager, KVUpdateMode
    from pipeline_utils.config_loader import load_model_config, ModelConfig
    from pipeline_utils.dtypes import get_numpy_dtype, safe_cast
    from pipeline_utils.io_binding import DeviceBuffer, IOBindingManager
    HAS_PIPELINE_UTILS = True
except ImportError:
    HAS_PIPELINE_UTILS = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

from .static_shapes import allow_shape_check_bypass

try:
    from transformers import AutoTokenizer
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False


_PROVIDER_ALIASES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "dml": "DmlExecutionProvider",
    "directml": "DmlExecutionProvider",
    "vitisai": "VitisAIExecutionProvider",
    "ryzenai": "VitisAIExecutionProvider",
    "azure": "AzureExecutionProvider",
}
_OPTIMIZATION_LEVELS = ("DISABLE", "BASIC", "EXTENDED", "ALL")
_DEVICES = {
    "auto": "auto",
    "cpu": "cpu",
    "gpu": "gpu",
    "igpu": "igpu",
    "npu": "npu",
    "rdna": "igpu",
    "xdna": "npu",
}
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
)


class InferenceMode(Enum):
    """Execution mode for the inference engine."""

    SEQUENTIAL = "sequential"
    PERSISTENT = "persistent"
    IO_BOUND = "io_bound"


def _real_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean, got {value!r}")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return value


def _non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer, got {value!r}")
    return value


def _non_negative_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a non-negative number, got {value!r}")
    number = float(value)
    if not np.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be a non-negative number, got {value!r}")
    return number


def _optional_text(value: Any, label: str) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string, got {value!r}")
    return value.strip()


def _coerce_mode(value: Any) -> InferenceMode:
    if isinstance(value, InferenceMode):
        return value
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise ValueError(f"mode must be sequential, persistent, or io_bound, got {value!r}")
    token = value.strip().lower()
    for mode in InferenceMode:
        if token == mode.value:
            return mode
    raise ValueError(f"mode must be sequential, persistent, or io_bound, got {value!r}")


def _coerce_device(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise ValueError(f"device must be auto, npu, igpu, gpu, or cpu, got {value!r}")
    key = value.strip().lower()
    if key not in _DEVICES:
        raise ValueError(f"device must be auto, npu, igpu, gpu, or cpu, got {value!r}")
    return _DEVICES[key]


def _coerce_providers(providers: Any) -> List[str]:
    if isinstance(providers, (str, bytes)) or not isinstance(providers, (list, tuple)):
        raise ValueError("providers must be a list of provider names")
    if len(providers) == 0:
        raise ValueError("providers must not be empty")
    normalized: List[str] = []
    for entry in providers:
        if isinstance(entry, bool) or not isinstance(entry, str) or not entry.strip():
            raise ValueError(f"provider must be a non-empty name, got {entry!r}")
        token = entry.strip()
        key = token.lower()
        if key in _PROVIDER_ALIASES:
            name = _PROVIDER_ALIASES[key]
        elif token.endswith("ExecutionProvider"):
            name = token
        else:
            raise ValueError(f"provider {token!r} is not an ONNX Runtime execution provider")
        if name not in normalized:
            normalized.append(name)
    return normalized


_DEVICE_PROVIDER_LISTS = {
    "cpu": ["CPUExecutionProvider"],
    "gpu": ["CUDAExecutionProvider", "ROCmExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"],
    "igpu": ["DmlExecutionProvider", "ROCmExecutionProvider", "CPUExecutionProvider"],
    "npu": ["VitisAIExecutionProvider", "RyzenAIExecutionProvider", "CPUExecutionProvider"],
}
_LAYER_PROVIDER_KEY = re.compile(r"^layer_(\d+|\*)$")
_HYBRID_POLICIES = {
    "hybrid",
    "hybrid_npu_igpu",
    "phase_split",
    "moe_routing",
    "speculative",
}


class _ProviderPlacement:
    """Phase and role provider lists derived from an existing plan."""

    def __init__(self, phases: Dict[str, List[str]], components: Dict[str, List[str]]):
        self.phases = phases
        self.components = components


def _device_key(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    raw = getattr(value, "value", value)
    text = str(raw).split(".")[-1].strip().lower()
    return text or None


def _coerce_provider_map(value: Any, label: str) -> Dict[str, List[str]]:
    if value is None:
        return {}
    if isinstance(value, (str, bytes)) or not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping of names to providers, got {value!r}")
    copied: Dict[str, List[str]] = {}
    for key, providers in value.items():
        if isinstance(key, bool) or not isinstance(key, str) or not key.strip():
            raise ValueError(f"{label} key must be a non-empty string, got {key!r}")
        copied[key.strip()] = _coerce_providers(providers)
    return copied


def _providers_from_step(step: Any) -> List[str]:
    provider = getattr(step, "provider", None)
    if isinstance(provider, str) and provider.strip():
        return _coerce_providers([provider.strip()])
    device = _device_key(getattr(step, "device", None)) or _device_key(
        getattr(step, "device_type", None)
    )
    if device in _DEVICE_PROVIDER_LISTS:
        return list(_DEVICE_PROVIDER_LISTS[device])
    return ["CPUExecutionProvider"]


def _policy_text(plan: Any) -> str:
    policy = getattr(plan, "policy", None)
    raw = getattr(policy, "value", policy)
    return str(raw or "").strip().lower().replace("-", "_")


def _read_execution_plan(plan: Any, phases: Dict[str, List[str]], components: Dict[str, List[str]]) -> None:
    if isinstance(plan, (str, bytes, bool, int, float, list, tuple)):
        raise ValueError(f"execution_plan must be an execution plan, got {plan!r}")
    phase_map = getattr(plan, "phase_providers", None)
    component_map = getattr(plan, "component_providers", None)
    steps = getattr(plan, "steps", None)
    has_maps = isinstance(phase_map, dict) or isinstance(component_map, dict)
    has_steps = isinstance(steps, list)
    if not has_maps and not has_steps and not hasattr(plan, "prefill_device"):
        raise ValueError(
            "execution_plan must be a HybridExecutionPlan or an ExecutionPlan"
        )
    if isinstance(phase_map, dict) and phase_map:
        phases.update(_coerce_provider_map(phase_map, "phase_providers"))
    if isinstance(component_map, dict) and component_map:
        components.update(_coerce_provider_map(component_map, "component_providers"))
    if has_steps:
        for step in steps:
            phase = str(getattr(step, "phase", "decode") or "decode").strip().lower()
            providers = _providers_from_step(step)
            name = getattr(step, "component_name", None)
            if phase in ("router", "moe_router"):
                components.setdefault("router", providers)
                if isinstance(name, str) and name.strip():
                    components.setdefault(name.strip(), providers)
            elif phase in ("expert", "experts"):
                components.setdefault("expert", providers)
                if isinstance(name, str) and name.strip():
                    components.setdefault(name.strip(), providers)
            elif phase in ("prefill", "decode") and phase not in phases:
                phases[phase] = providers
    prefill = _device_key(getattr(plan, "prefill_device", None))
    decode = _device_key(getattr(plan, "decode_device", None))
    hybrid = _policy_text(plan) in _HYBRID_POLICIES or (
        prefill is not None and decode is not None and prefill != decode
    )
    if hybrid:
        if prefill in _DEVICE_PROVIDER_LISTS and "prefill" not in phases:
            phases["prefill"] = list(_DEVICE_PROVIDER_LISTS[prefill])
        if decode in _DEVICE_PROVIDER_LISTS and "decode" not in phases:
            phases["decode"] = list(_DEVICE_PROVIDER_LISTS[decode])


def _build_placement(
    phase_providers: Any,
    component_providers: Any,
    execution_plan: Any,
) -> Optional[_ProviderPlacement]:
    """
    Provider lists for prefill, decode, the MoE router, and experts.

    Layer-index keys are ignored. A sequential layer series on the NPU is
    not a placement. The two ExecutionPlan classes stay distinct; this view
    only reads their fields.
    """
    phases: Dict[str, List[str]] = {}
    components: Dict[str, List[str]] = {}
    if execution_plan is not None:
        _read_execution_plan(execution_plan, phases, components)
    if phase_providers:
        phases.update(_coerce_provider_map(phase_providers, "phase_providers"))
    if component_providers:
        components.update(_coerce_provider_map(component_providers, "component_providers"))
    components = {
        key: value
        for key, value in components.items()
        if not _LAYER_PROVIDER_KEY.match(key)
    }
    if not phases and not components:
        return None
    return _ProviderPlacement(phases, components)


def _component_role(name: str) -> Optional[str]:
    """Router or expert. A dense gate projection is neither."""
    text = name.lower().replace("-", "_")
    if "gate_proj" in text:
        return None
    if "expert" in text:
        return "expert"
    if (
        "router" in text
        or "mlp.gate" in text
        or "/mlp/gate" in text
        or ".gate" in text
        or "/gate" in text
        or text.endswith("gate")
    ):
        return "router"
    return None


def _is_bound_activation(value: Any) -> bool:
    """A device buffer stays on device. NumPy values are host tensors."""
    if isinstance(value, np.ndarray):
        return False
    if HAS_PIPELINE_UTILS and isinstance(value, DeviceBuffer):
        return True
    return callable(getattr(value, "to_numpy", None)) and hasattr(value, "ort_value")


def _as_host(value: Any) -> Any:
    """Copy one tensor to NumPy. Device buffers stay untouched until this call."""
    if isinstance(value, np.ndarray):
        return value
    to_numpy = getattr(value, "to_numpy", None)
    if callable(to_numpy):
        return to_numpy()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        return numpy()
    return np.asarray(value)


def _coerce_stop_ids(value: Any) -> List[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("stop_token_ids must be a list of token ids")
    ids: List[int] = []
    for entry in value:
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise ValueError(f"stop token id must be an integer, got {entry!r}")
        ids.append(entry)
    return ids


def _read_model_info(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model_info.json is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("model_info.json must be a JSON object")
    return payload


def _layer_template(name: str, layer: int) -> str:
    """Replace one delimited layer index with the cache placeholder."""
    return re.sub(rf"(?<!\d){int(layer)}(?!\d)", "{layer}", name, count=1)


def _named_format(names: Any, kind: str, layer: int) -> Optional[str]:
    if not isinstance(names, (list, tuple)):
        return None
    for name in names:
        if not isinstance(name, str):
            continue
        lower = name.lower()
        if kind == "key" and "key" in lower and "value" not in lower:
            return _layer_template(name, layer)
        if kind == "value" and "value" in lower and "key" not in lower:
            return _layer_template(name, layer)
    return None


def _has_tokenizer_files(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    return any(os.path.isfile(os.path.join(path, name)) for name in _TOKENIZER_FILES)


def _sample_token(row: Any, temperature: float) -> int:
    """Pick one id. A temperature of zero selects the largest logit."""
    scores = np.asarray(row, dtype=np.float64).reshape(-1)
    if scores.size < 1:
        raise RuntimeError("logits row is empty")
    if temperature <= 0:
        return int(np.argmax(scores))
    scaled = scores / float(temperature)
    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0:
        return int(np.argmax(scores))
    probs = probs / total
    return int(np.random.choice(probs.size, p=probs))


@dataclass
class InferenceConfig:
    """Configuration for the split model inference engine."""

    providers: List[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    mode: InferenceMode = InferenceMode.SEQUENTIAL
    device: str = "auto"
    optimization_level: str = "DISABLE"
    disable_shape_check: bool = True
    max_new_tokens: int = 50
    temperature: float = 0.0
    stop_token_ids: List[int] = field(default_factory=list)
    warmup_iterations: int = 0
    stream_callback: Optional[Callable[[int, str, float], None]] = None
    use_io_binding: bool = False
    io_binding_device: str = "cpu"
    kv_cache_dtype: Optional[str] = None
    static_kv_cache: bool = False
    max_seq_len: int = 4096
    tokenizer_path: Optional[str] = None
    trust_remote_code: bool = True
    phase_providers: Optional[Dict[str, List[str]]] = None
    component_providers: Optional[Dict[str, List[str]]] = None
    execution_plan: Optional[Any] = None

    def __post_init__(self) -> None:
        self.mode = _coerce_mode(self.mode)
        self.device = _coerce_device(self.device)
        if self.device != "auto" and HAS_PIPELINE_UTILS:
            # get_provider_for_device has no igpu category and would return CPU.
            lookup = "gpu" if self.device == "igpu" else self.device
            self.providers = get_provider_for_device(lookup)
        self.providers = _coerce_providers(self.providers)
        self.disable_shape_check = _real_bool(self.disable_shape_check, "disable_shape_check")
        self.use_io_binding = _real_bool(self.use_io_binding, "use_io_binding")
        self.static_kv_cache = _real_bool(self.static_kv_cache, "static_kv_cache")
        self.trust_remote_code = _real_bool(self.trust_remote_code, "trust_remote_code")
        if self.mode is InferenceMode.IO_BOUND:
            self.use_io_binding = True
        self.max_new_tokens = _positive_int(self.max_new_tokens, "max_new_tokens")
        self.warmup_iterations = _non_negative_int(self.warmup_iterations, "warmup_iterations")
        self.max_seq_len = _positive_int(self.max_seq_len, "max_seq_len")
        self.temperature = _non_negative_number(self.temperature, "temperature")
        self.stop_token_ids = _coerce_stop_ids(self.stop_token_ids)
        if self.stream_callback is not None and not callable(self.stream_callback):
            raise ValueError("stream_callback must be callable")
        level = self.optimization_level
        if isinstance(level, bool) or not isinstance(level, str) or not level.strip():
            raise ValueError(f"optimization_level must be DISABLE, BASIC, EXTENDED, or ALL, got {level!r}")
        level = level.strip().upper()
        if level not in _OPTIMIZATION_LEVELS:
            raise ValueError(f"optimization_level must be DISABLE, BASIC, EXTENDED, or ALL, got {level!r}")
        self.optimization_level = level
        device_name = self.io_binding_device
        if isinstance(device_name, bool) or not isinstance(device_name, str) or not device_name.strip():
            raise ValueError(f"io_binding_device must be a non-empty string, got {device_name!r}")
        self.io_binding_device = device_name.strip().lower()
        self.kv_cache_dtype = _optional_text(self.kv_cache_dtype, "kv_cache_dtype")
        self.tokenizer_path = _optional_text(self.tokenizer_path, "tokenizer_path")
        if self.phase_providers is not None and not isinstance(self.phase_providers, dict):
            raise ValueError(
                f"phase_providers must be a mapping, got {self.phase_providers!r}"
            )
        if self.component_providers is not None and not isinstance(self.component_providers, dict):
            raise ValueError(
                f"component_providers must be a mapping, got {self.component_providers!r}"
            )
        self._placement = _build_placement(
            self.phase_providers,
            self.component_providers,
            self.execution_plan,
        )


@dataclass
class InferenceResult:
    """Result of a generation run."""

    text: str
    token_ids: List[int]
    num_tokens_generated: int
    total_time_s: float
    prefill_time_s: float
    decode_time_s: float
    tokens_per_second: float
    time_to_first_token_s: float
    peak_memory_mb: float = 0.0


class SplitModelInferenceEngine:
    """
    Production inference engine for split ONNX models.

    Sequential mode loads a component for one run and then releases it.
    Persistent mode keeps every session resident. IO_BOUND mode keeps
    those sessions and runs them through IOBindingManager.
    """

    def __init__(
        self,
        model_dir: str,
        config: Optional[InferenceConfig] = None,
        tokenizer_path: Optional[str] = None,
        verbose: bool = True,
    ):
        if not HAS_ORT:
            raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")
        if not HAS_PIPELINE_UTILS:
            raise ImportError(
                "pipeline_utils is required. Ensure pipeline_utils/ is in your Python path."
            )
        if isinstance(model_dir, bool) or not isinstance(model_dir, str) or not model_dir.strip():
            raise ValueError(f"model_dir must be a non-empty path, got {model_dir!r}")
        if config is not None and not isinstance(config, InferenceConfig):
            raise ValueError("config must be an InferenceConfig")
        tokenizer_path = _optional_text(tokenizer_path, "tokenizer_path")
        verbose = _real_bool(verbose, "verbose")

        self.model_dir = model_dir.strip()
        self.config = config or InferenceConfig()
        self.verbose = verbose
        self._resident_sessions: Dict[str, Any] = {}
        self._last_result: Optional[InferenceResult] = None
        self._stats: Dict[str, float] = {}
        self._kv_key_fmt: Optional[str] = None
        self._kv_val_fmt: Optional[str] = None
        self._past_key_fmt: Optional[str] = None
        self._past_val_fmt: Optional[str] = None
        self._placement = getattr(self.config, "_placement", None)
        self._phase_sessions: Dict[Tuple[str, str, Tuple[str, ...]], Any] = {}
        self._provider_trace: List[Tuple[str, str, List[str]]] = []

        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"model path not found: {self.model_dir}")

        self.is_monolith = os.path.isfile(self.model_dir)
        if self.is_monolith:
            self.model_path: Optional[str] = self.model_dir
            self.model_dir = os.path.dirname(self.model_dir) or "."
            self.model_info = {"num_layers": 32}
            info_path = os.path.join(self.model_dir, "model_info.json")
            if os.path.isfile(info_path):
                self.model_info.update(_read_model_info(info_path))
            self.num_layers = _layer_count(self.model_info, required=False)
        else:
            if not os.path.isdir(self.model_dir):
                raise ValueError(f"model_dir must be a file or directory, got {self.model_dir!r}")
            self.model_path = None
            info_path = os.path.join(self.model_dir, "model_info.json")
            if not os.path.isfile(info_path):
                raise FileNotFoundError(f"model_info.json not found in {self.model_dir}")
            self.model_info = _read_model_info(info_path)
            self.num_layers = _layer_count(self.model_info, required=True)
        components = self.model_info.get("components", {})
        if components is None:
            components = {}
        if not isinstance(components, dict):
            raise ValueError("model_info components must be an object")
        self.model_info["components"] = components

        search_path = tokenizer_path or self.config.tokenizer_path or self.model_dir
        try:
            self.model_config = load_model_config(self.model_dir, search_path)
            self.num_layers = _positive_int(self.model_config.num_layers, "num_layers")
            self._log(
                f"[Config] Loaded: {self.model_config.num_layers} layers, "
                f"{self.model_config.num_kv_heads} KV heads, "
                f"{self.model_config.head_dim} head dim"
            )
        except FileNotFoundError:
            warnings.warn("No config found, using defaults")
            self.model_config = ModelConfig(
                num_layers=self.num_layers,
                num_kv_heads=8,
                head_dim=128,
                source_file="defaults",
            )

        if self.config.kv_cache_dtype:
            kv_dtype = get_numpy_dtype(self.config.kv_cache_dtype)
        elif self.model_config.dtype is not None:
            kv_dtype = self.model_config.dtype
        else:
            kv_dtype = np.float16
        self._static_window_layers = self._detect_static_window_layers()
        cache_limit = (
            self.config.max_seq_len
            if (self.config.static_kv_cache or self._static_window_layers)
            else None
        )
        kv_update_mode = KVUpdateMode.CONCAT_SWAP if self._static_window_layers else None
        self.kv_cache = KVCacheManager(
            num_layers=self.model_config.num_layers,
            num_kv_heads=self.model_config.num_kv_heads,
            head_dim=self.model_config.head_dim,
            dtype=kv_dtype,
            batch_size=1,
            max_seq_len=cache_limit,
            update_mode=kv_update_mode,
        )

        self.session_manager: Optional[SessionManager] = None
        self.io_manager: Optional[IOBindingManager] = None
        if (
            self._placement is None
            and self.config.mode in (InferenceMode.PERSISTENT, InferenceMode.IO_BOUND)
        ):
            self._init_persistent_sessions()
        if self.config.use_io_binding and self.config.mode is not InferenceMode.SEQUENTIAL:
            self._init_io_binding()

        self.tokenizer = None
        self._load_tokenizer(tokenizer_path)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _load_tokenizer(self, tokenizer_path: Optional[str]) -> None:
        if not HAS_TOKENIZER:
            return
        explicit = tokenizer_path or self.config.tokenizer_path
        tok_path = explicit or self.model_dir
        if explicit is None and not _has_tokenizer_files(tok_path):
            self._log("[Tokenizer] No local tokenizer files")
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tok_path,
                trust_remote_code=self.config.trust_remote_code,
            )
        except Exception as exc:
            warnings.warn(f"Could not load tokenizer: {exc}")

    def _init_persistent_sessions(self) -> None:
        """Load every component session for persistent and IO_BOUND modes."""
        self._log("[Init] Loading persistent sessions...")
        session_config = SessionConfig(
            providers=list(self.config.providers),
            optimization_level=self.config.optimization_level,
            disable_shape_check=self.config.disable_shape_check,
        )
        self.session_manager = SessionManager(
            model_dir=self.model_dir,
            config=session_config,
            model_info=self.model_info,
        )
        self.session_manager.load_all()
        memory = self.session_manager.get_memory_usage()
        self._log(f"[Init] Sessions loaded: {float(memory.get('total_mb', 0.0)):.1f} MB")

    def _init_io_binding(self) -> None:
        """Allocate the device buffer manager used by IO_BOUND runs."""
        self._log("[Init] Initializing I/O binding...")
        self.io_manager = IOBindingManager(
            device_type=self.config.io_binding_device,
            device_id=0,
            auto_reallocate=True,
            max_seq_len=self.config.max_seq_len,
        )

    def warmup(self, iterations: Optional[int] = None) -> None:
        """
        Run short generations so an NPU can compile and cache each graph.

        A failed iteration is logged and the remaining iterations still run.
        """
        count = self.config.warmup_iterations if iterations is None else iterations
        count = _non_negative_int(count, "warmup_iterations")
        if count <= 0:
            return
        self._log(f"[Warmup] Running {count} warmup iterations...")
        for index in range(count):
            start = time.perf_counter()
            try:
                self.generate("warmup", max_new_tokens=1)
            except Exception as exc:
                self._log(f"  Warmup {index + 1}/{count} failed: {exc}")
                continue
            self._log(f"  Warmup {index + 1}/{count}: {time.perf_counter() - start:.3f}s")
        self._log("[Warmup] Complete")

    def _detect_static_window_layers(self) -> bool:
        """True when split layer components declare an ``attention_bias`` input."""
        if self.is_monolith:
            return False
        component = self.model_info.get("components", {}).get("layer_0", {})
        if not isinstance(component, dict):
            return False
        inputs = component.get("input_names")
        if not isinstance(inputs, list):
            return False
        return "attention_bias" in inputs

    def _detect_kv_format(self) -> Tuple[str, str, str, str]:
        """Return present key, present value, past key, and past value formats."""
        if (
            self._kv_key_fmt
            and self._kv_val_fmt
            and self._past_key_fmt
            and self._past_val_fmt
        ):
            return (
                self._kv_key_fmt,
                self._kv_val_fmt,
                self._past_key_fmt,
                self._past_val_fmt,
            )
        component = self.model_info.get("components", {}).get("layer_0", {})
        if not isinstance(component, dict):
            component = {}
        present_key = _named_format(component.get("output_names"), "key", 0)
        present_value = _named_format(component.get("output_names"), "value", 0)
        past_key = _named_format(component.get("input_names"), "key", 0)
        past_value = _named_format(component.get("input_names"), "value", 0)
        self._kv_key_fmt = present_key or "present.{layer}.key"
        self._kv_val_fmt = present_value or "present.{layer}.value"
        self._past_key_fmt = past_key or "past_key_values.{layer}.key"
        self._past_val_fmt = past_value or "past_key_values.{layer}.value"
        self._log(f"[KV] Detected format: {self._kv_key_fmt}")
        return (
            self._kv_key_fmt,
            self._kv_val_fmt,
            self._past_key_fmt,
            self._past_val_fmt,
        )

    def _component_path(self, component_name: str) -> str:
        if self.is_monolith and component_name == "monolith":
            if not self.model_path:
                raise RuntimeError("monolith model path is missing")
            return self.model_path
        component = self.model_info.get("components", {}).get(component_name)
        if not isinstance(component, dict):
            raise RuntimeError(f"component {component_name} is not in model_info")
        filename = component.get("file", "")
        if isinstance(filename, bool) or not isinstance(filename, str) or not filename.strip():
            raise ValueError(f"component {component_name} is missing a file name")
        return os.path.join(self.model_dir, filename.strip())

    def _providers_for(self, component_name: str, phase: str) -> List[str]:
        """
        Providers for one component in prefill or decode.

        A router stays on its component list and an expert stays on its
        component list. Every other component uses the phase list. Layer
        indexes are not a placement.
        """
        placement = self._placement
        if placement is None:
            return list(self.config.providers)
        role = _component_role(component_name)
        candidates = [component_name]
        if role == "router":
            candidates.extend(["router", "moe_router", "router_*", "moe_router_*"])
        elif role == "expert":
            candidates.extend(
                ["expert", "experts", "moe_expert", "expert_*", "experts_*", "moe_expert_*"]
            )
        for key in candidates:
            providers = placement.components.get(key)
            if providers:
                return list(providers)
        phase_providers = placement.phases.get(phase)
        if phase_providers:
            return list(phase_providers)
        return list(self.config.providers)

    def _open_session(self, component_name: str, providers: Optional[List[str]] = None) -> Any:
        filepath = self._component_path(component_name)
        chosen = list(self.config.providers if providers is None else providers)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"{component_name} file not found: {filepath}")
        options = ort.SessionOptions()
        options.graph_optimization_level = {
            "DISABLE": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }[self.config.optimization_level]
        if allow_shape_check_bypass(
            chosen,
            self.config.disable_shape_check,
            static_kv_cache=self.config.static_kv_cache,
        ):
            options.add_session_config_entry("session.disable_node_shape_check", "1")
        try:
            return ort.InferenceSession(
                filepath,
                sess_options=options,
                providers=chosen,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load {component_name}: {exc}") from exc

    def _get_session(self, component_name: str, phase: str = "decode") -> Tuple[Any, bool]:
        """Return the session and whether this call owns its lifetime."""
        if self._placement is not None:
            providers = self._providers_for(component_name, phase)
            self._provider_trace.append((component_name, phase, list(providers)))
            key = (component_name, phase, tuple(providers))
            resident = self._phase_sessions.get(key)
            if resident is not None:
                return resident, False
            session = self._open_session(component_name, providers)
            if self.config.mode is not InferenceMode.SEQUENTIAL:
                self._phase_sessions[key] = session
                return session, False
            return session, True
        if self.session_manager is not None and not (
            self.is_monolith and component_name == "monolith"
        ):
            session = self.session_manager.get_session(component_name)
            if session is None:
                raise RuntimeError(f"Failed to load {component_name}")
            return session, False
        resident = self._resident_sessions.get(component_name)
        if resident is not None:
            return resident, False
        session = self._open_session(component_name)
        if self.config.mode is not InferenceMode.SEQUENTIAL:
            self._resident_sessions[component_name] = session
            return session, False
        return session, True

    def _component_names(self, session: Any, component_name: str) -> Tuple[List[str], List[str]]:
        if self.is_monolith and component_name == "monolith":
            return (
                [item.name for item in session.get_inputs()],
                [item.name for item in session.get_outputs()],
            )
        component = self.model_info.get("components", {}).get(component_name, {})
        if not isinstance(component, dict):
            component = {}
        inputs = component.get("input_names")
        outputs = component.get("output_names")
        if not isinstance(inputs, list) or not inputs:
            inputs = [item.name for item in session.get_inputs()]
        if not isinstance(outputs, list) or not outputs:
            outputs = [item.name for item in session.get_outputs()]
        return list(inputs), list(outputs)

    def _cast_input(self, name: str, value: Any, expected: Dict[str, str]) -> Any:
        if _is_bound_activation(value):
            return value
        type_name = expected.get(name)
        if not type_name:
            return value
        try:
            return safe_cast(value, get_numpy_dtype(type_name), warn_on_precision_loss=False)
        except (ValueError, KeyError, TypeError):
            return value

    def _run_component(
        self,
        component_name: str,
        registry: Dict[str, Any],
        phase: str = "decode",
    ) -> None:
        session, owned = self._get_session(component_name, phase)
        try:
            input_names, output_names = self._component_names(session, component_name)
            expected = {item.name: item.type for item in session.get_inputs()}
            session_inputs = {}
            for name in input_names:
                if name in registry:
                    session_inputs[name] = self._cast_input(name, registry[name], expected)
            if not session_inputs:
                raise RuntimeError(f"{component_name} received no inputs")
            if self.io_manager is not None:
                device_runner = getattr(self.io_manager, "run_with_device_buffers", None)
                use_device = callable(device_runner) and (
                    self.config.mode is InferenceMode.IO_BOUND or self.config.use_io_binding
                )
                if use_device:
                    bound = device_runner(session, session_inputs, output_names)
                else:
                    bound = self.io_manager.run_with_binding(session, session_inputs)
                for name in output_names:
                    if name in bound:
                        registry[name] = bound[name]
                return
            outputs = session.run(output_names or None, session_inputs)
            actual = output_names or [item.name for item in session.get_outputs()]
            for name, value in zip(actual, outputs):
                registry[name] = value
        finally:
            if owned:
                del session
                gc.collect()

    def _execution_order(self) -> List[str]:
        if self.is_monolith:
            return ["monolith"]
        components = self.model_info.get("components", {})
        order: List[str] = []
        for name in ("other", "embedding"):
            if name in components:
                order.append(name)
        for index in range(self.num_layers):
            order.append(f"layer_{index}")
        roles = [
            name
            for name in components
            if name not in order and name not in ("norm", "lm_head") and _component_role(name)
        ]
        roles.sort(key=lambda name: (0 if _component_role(name) == "router" else 1, name))
        order.extend(roles)
        for name in ("norm", "lm_head"):
            if name in components:
                order.append(name)
        if not order:
            raise RuntimeError("model_info has no components to run")
        return order

    def _logits_from(self, registry: Dict[str, Any]) -> Any:
        if "logits" in registry:
            return registry["logits"]
        for name, value in registry.items():
            if "logits" in str(name).lower():
                return value
        for value in registry.values():
            if hasattr(value, "ndim") and value.ndim == 3:
                return value
        raise RuntimeError("logits output was not produced")

    def _peak_memory_mb(self) -> float:
        if self.session_manager is None:
            return 0.0
        usage = self.session_manager.get_memory_usage()
        total = usage.get("total_mb", 0.0) if isinstance(usage, dict) else 0.0
        try:
            return float(total)
        except (TypeError, ValueError):
            return 0.0

    def _run_static_window_step(
        self,
        token_ids: np.ndarray,
        registry: Dict[str, Any],
        present_key: str,
        present_value: str,
        past_key: str,
        past_value: str,
        phase: str,
    ) -> None:
        """Execute one static-window forward pass across all components and swap KV buffers."""
        q_len = int(token_ids.shape[1])
        registry["input_ids"] = token_ids
        registry["position_ids"] = self.kv_cache.position_ids(query_len=q_len)
        registry["attention_bias"] = self.kv_cache.attention_bias(
            query_len=q_len,
            align="right",
        )
        registry["attention_mask"] = self.kv_cache.attention_mask()
        for layer_idx in range(self.num_layers):
            k_past, v_past = self.kv_cache.get_past_tensors(layer_idx)
            registry[past_key.format(layer=layer_idx)] = k_past
            registry[past_value.format(layer=layer_idx)] = v_past
        for component_name in self._execution_order():
            self._run_component(component_name, registry, phase)
            if component_name.startswith("layer_"):
                layer_idx = int(component_name.split("_", 1)[1])
                pk_name = present_key.format(layer=layer_idx)
                pv_name = present_value.format(layer=layer_idx)
                if pk_name in registry and pv_name in registry:
                    shadow_k = self.kv_cache._cache[layer_idx]["key_shadow"]
                    shadow_v = self.kv_cache._cache[layer_idx]["value_shadow"]
                    np.copyto(_as_host(shadow_k), _as_host(registry[pk_name]))
                    np.copyto(_as_host(shadow_v), _as_host(registry[pv_name]))
                    past_valid = self.kv_cache.layer_length(layer_idx)
                    self.kv_cache.commit_present_swap(
                        layer_idx,
                        past_valid + q_len,
                        rolling=True,
                        query_len=q_len,
                    )

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        stream_callback: Optional[Callable[[int, str, float], None]] = None,
    ) -> str:
        """
        Generate text from a prompt.

        The return value is the decoded prompt plus the new tokens.
        get_result returns the same text with token ids and timing.
        """
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer loaded. Provide tokenizer_path.")
        if isinstance(prompt, bool) or not isinstance(prompt, str):
            raise ValueError(f"prompt must be a string, got {prompt!r}")
        if max_new_tokens is None:
            max_tokens = self.config.max_new_tokens
        else:
            max_tokens = _positive_int(max_new_tokens, "max_new_tokens")
        callback = stream_callback or self.config.stream_callback
        if callback is not None and not callable(callback):
            raise ValueError("stream_callback must be callable")

        inputs = self.tokenizer(prompt, return_tensors="np")
        # transformers BatchEncoding is a mapping, not a dict.
        if not (isinstance(inputs, dict) or hasattr(inputs, "keys")) or "input_ids" not in inputs:
            raise RuntimeError("tokenizer did not return input_ids")
        prompt_ids = np.asarray(inputs["input_ids"])
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.reshape(1, -1)
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] < 1 or prompt_ids.shape[1] < 1:
            raise RuntimeError("tokenizer returned no tokens")
        prompt_ids = np.asarray(prompt_ids, dtype=np.int64)
        prompt_len = int(prompt_ids.shape[1])

        self.kv_cache.reset()
        present_key, present_value, past_key, past_value = self._detect_kv_format()
        registry: Dict[str, Any] = {}
        generated_ids: List[int] = []
        last_token_id: Optional[int] = None
        total_start = time.perf_counter()
        prefill_time = 0.0
        decode_time = 0.0
        first_token_time = 0.0
        eos_id = getattr(self.tokenizer, "eos_token_id", None)

        for token_idx in range(max_tokens):
            token_start = time.perf_counter()
            if self._static_window_layers:
                if token_idx == 0:
                    for p_idx in range(prompt_len - 1):
                        self._run_static_window_step(
                            prompt_ids[:, p_idx : p_idx + 1],
                            registry,
                            present_key,
                            present_value,
                            past_key,
                            past_value,
                            "prefill",
                        )
                    self._run_static_window_step(
                        prompt_ids[:, prompt_len - 1 : prompt_len],
                        registry,
                        present_key,
                        present_value,
                        past_key,
                        past_value,
                        "prefill",
                    )
                else:
                    self._run_static_window_step(
                        np.array([[last_token_id]], dtype=np.int64),
                        registry,
                        present_key,
                        present_value,
                        past_key,
                        past_value,
                        "decode",
                    )
            else:
                if token_idx == 0:
                    current_ids = prompt_ids
                    position_ids = np.arange(prompt_len, dtype=np.int64).reshape(1, -1)
                    attention_mask = np.ones((1, prompt_len), dtype=np.int64)
                else:
                    current_ids = np.array([[last_token_id]], dtype=np.int64)
                    past_len = int(self.kv_cache.get_sequence_length())
                    position_ids = np.array([[past_len]], dtype=np.int64)
                    attention_mask = np.ones((1, past_len + 1), dtype=np.int64)
                phase = "prefill" if token_idx == 0 else "decode"
                registry["input_ids"] = current_ids
                registry["position_ids"] = position_ids
                registry["attention_mask"] = attention_mask
                self.kv_cache.inject_into_registry(
                    registry,
                    key_format=past_key,
                    value_format=past_value,
                )
                for component_name in self._execution_order():
                    self._run_component(component_name, registry, phase)
                    if self.is_monolith or component_name.startswith("layer_"):
                        self.kv_cache.extract_from_registry(
                            registry,
                            present_key_format=present_key,
                            present_value_format=present_value,
                        )
            logits = _as_host(self._logits_from(registry))
            last_token_id = _sample_token(np.asarray(logits)[:, -1, :], self.config.temperature)
            generated_ids.append(last_token_id)
            elapsed = time.perf_counter() - token_start
            if token_idx == 0:
                prefill_time = elapsed
                first_token_time = time.perf_counter() - total_start
            else:
                decode_time += elapsed
            if callback is not None:
                callback(token_idx, self.tokenizer.decode([last_token_id]), elapsed)
            if eos_id is not None and last_token_id == eos_id:
                break
            if isinstance(eos_id, (list, tuple)) and last_token_id in eos_id:
                break
            if last_token_id in self.config.stop_token_ids:
                break

        total_time = time.perf_counter() - total_start
        tokens_per_second = len(generated_ids) / decode_time if decode_time > 0 else 0.0
        full_ids = np.concatenate(
            [prompt_ids, np.array([generated_ids], dtype=np.int64)],
            axis=1,
        )
        text = self.tokenizer.decode(full_ids[0].tolist())
        self._stats = {
            "total_time_s": total_time,
            "prefill_time_s": prefill_time,
            "decode_time_s": decode_time,
            "ttft_s": first_token_time,
            "tokens_per_second": tokens_per_second,
            "num_tokens": float(len(generated_ids)),
        }
        self._last_result = InferenceResult(
            text=text,
            token_ids=list(generated_ids),
            num_tokens_generated=len(generated_ids),
            total_time_s=total_time,
            prefill_time_s=prefill_time,
            decode_time_s=decode_time,
            tokens_per_second=tokens_per_second,
            time_to_first_token_s=first_token_time,
            peak_memory_mb=self._peak_memory_mb(),
        )
        return text

    def get_stats(self) -> Dict[str, float]:
        """Timing statistics from the last generation run."""
        return dict(self._stats)

    def get_result(self) -> InferenceResult:
        """Structured result from the last generation run."""
        if self._last_result is None:
            stats = self._stats
            return InferenceResult(
                text="",
                token_ids=[],
                num_tokens_generated=int(stats.get("num_tokens", 0)),
                total_time_s=float(stats.get("total_time_s", 0.0)),
                prefill_time_s=float(stats.get("prefill_time_s", 0.0)),
                decode_time_s=float(stats.get("decode_time_s", 0.0)),
                tokens_per_second=float(stats.get("tokens_per_second", 0.0)),
                time_to_first_token_s=float(stats.get("ttft_s", 0.0)),
            )
        result = self._last_result
        return InferenceResult(
            text=result.text,
            token_ids=list(result.token_ids),
            num_tokens_generated=result.num_tokens_generated,
            total_time_s=result.total_time_s,
            prefill_time_s=result.prefill_time_s,
            decode_time_s=result.decode_time_s,
            tokens_per_second=result.tokens_per_second,
            time_to_first_token_s=result.time_to_first_token_s,
            peak_memory_mb=result.peak_memory_mb,
        )

    def cleanup(self) -> None:
        """Release sessions and device buffers."""
        if self.session_manager is not None:
            self.session_manager.unload_all()
        self._resident_sessions.clear()
        self._phase_sessions.clear()
        if self.io_manager is not None:
            self.io_manager.deallocate_all()
        gc.collect()

    def __repr__(self) -> str:
        return (
            f"SplitModelInferenceEngine(mode={self.config.mode.value}, "
            f"providers={self.config.providers}, layers={self.num_layers})"
        )


def _layer_count(info: Dict[str, Any], required: bool) -> int:
    if "num_layers" not in info:
        if required:
            raise ValueError("model_info.json is missing num_layers")
        return 32
    return _positive_int(info["num_layers"], "num_layers")


def run_inference(
    output_dir: str,
    prompt: str = "Test prompt.",
    config: Optional[InferenceConfig] = None,
    max_new_tokens: int = 10,
) -> str:
    """
    Run sequential inference on a split ONNX model and return the decoded text.

    This replaces the merge-and-validate stub. The engine is closed before
    the text is returned.
    """
    if config is not None and not isinstance(config, InferenceConfig):
        raise ValueError("config must be an InferenceConfig")
    if isinstance(prompt, bool) or not isinstance(prompt, str):
        raise ValueError(f"prompt must be a string, got {prompt!r}")
    max_new_tokens = _positive_int(max_new_tokens, "max_new_tokens")
    engine = SplitModelInferenceEngine(output_dir, config=config, verbose=False)
    try:
        return engine.generate(prompt, max_new_tokens=max_new_tokens)
    finally:
        engine.cleanup()


def create_inference_engine(
    model_dir: str,
    providers: Optional[List[str]] = None,
    persist: bool = False,
    tokenizer_path: Optional[str] = None,
) -> SplitModelInferenceEngine:
    """Create a sequential or persistent inference engine."""
    persist = _real_bool(persist, "persist")
    config = InferenceConfig(
        providers=["CPUExecutionProvider"] if providers is None else providers,
        mode=InferenceMode.PERSISTENT if persist else InferenceMode.SEQUENTIAL,
        tokenizer_path=tokenizer_path,
    )
    return SplitModelInferenceEngine(
        model_dir,
        config=config,
        tokenizer_path=tokenizer_path,
    )


__all__ = [
    "SplitModelInferenceEngine",
    "InferenceConfig",
    "InferenceMode",
    "InferenceResult",
    "run_inference",
    "create_inference_engine",
]
