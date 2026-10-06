"""
Shared data types for model chunking operations.

This module defines the core data types used across both ONNX and SafeTensors
splitting operations, providing a unified interface for configuration and results.
"""

import os
import re
from enum import Enum
from typing import Dict, Iterable, List, Any, Optional
from dataclasses import dataclass, field


class ModelFormat(str, Enum):
    """
    Canonical model file formats for chunking and inference.

    ONNX, SafeTensors, GGUF, and PyTorch ``.bin`` checkpoints share this enum.
    """
    ONNX = "onnx"
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    PYTORCH_BIN = "pytorch_bin"


class TensorDtype(str, Enum):
    """
    Supported tensor data types with memory implications.
    
    Memory per parameter:
    - FLOAT32: 4 bytes
    - FLOAT16: 2 bytes
    - BFLOAT16: 2 bytes
    - INT8: 1 byte
    - INT4: 0.5 bytes
    - AUTO: Preserve original dtype
    """
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT8 = "int8"
    INT4 = "int4"
    AUTO = "auto"


class MemoryMode(str, Enum):
    """
    Memory loading strategies for model splitting.
    
    - EAGER: Load entire model into RAM (current behavior, fastest but high memory)
    - STREAMING: Process one tensor at a time (lowest memory, recommended for large models)
    - LAZY: Memory-map files for on-demand loading
    - AUTO: Use HuggingFace's device_map="auto" for optimal device placement
    """
    EAGER = "eager"
    STREAMING = "streaming"
    LAZY = "lazy"
    AUTO = "auto"


class ComponentType(str, Enum):
    """
    Types of model components that can be extracted during splitting.

    MoE router and expert, Gated DeltaNet, sparse attention, the multi-token
    prediction head, and n-gram embeddings are the hybrid Qwen3.8 components.
    """
    EMBEDDING = "embedding"
    LAYER = "layer"
    NORM = "norm"
    LM_HEAD = "lm_head"
    MOE_ROUTER = "moe_router"
    MOE_EXPERT = "moe_expert"
    GATED_DELTANET = "gated_deltanet"
    SPARSE_ATTENTION = "sparse_attention"
    MTP_HEAD = "mtp_head"
    NGRAM_EMBEDDING = "ngram_embedding"
    OTHER = "other"


# Components that live on one transformer layer. A layer id is part of the name.
_LAYER_SCOPED_COMPONENTS = {
    ComponentType.LAYER,
    ComponentType.MOE_ROUTER,
    ComponentType.MOE_EXPERT,
    ComponentType.GATED_DELTANET,
    ComponentType.SPARSE_ATTENTION,
}


class DeviceType(str, Enum):
    """
    Inference device for a split model.

    IPU is the AMD iGPU target. AUTO picks NPU, then iGPU, then GPU, then CPU.
    """
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    IPU = "ipu"
    AUTO = "auto"


class ExecutionPolicy(str, Enum):
    """
    How a split model may be executed.

    AUTO behaves as one monolithic graph. Sequential and persistent layer
    splits are rejected on the NPU. Hybrid NPU and iGPU, phase split, and
    MoE routing are the NPU exceptions. Speculative draft and verify is a
    phase schedule, not a layer series.
    """
    AUTO = "auto"
    MONOLITHIC = "monolithic"
    SEQUENTIAL = "sequential"
    PERSISTENT = "persistent"
    HYBRID_NPU_IGPU = "hybrid_npu_igpu"
    PHASE_SPLIT = "phase_split"
    MOE_ROUTING = "moe_routing"
    SPECULATIVE = "speculative"


# Ahead-of-time accelerators. A sequential layer series on these providers
# recompiles or context-switches between graphs.
_NPU_PROVIDER_MARKERS = (
    "vitisai",
    "ryzenai",
    "openvino",
    "coreml",
)


# Ordered ONNX Runtime fallback for each device. AUTO is resolved later.
# NPU prefers VitisAI, then RyzenAI. IPU is the AMD iGPU order.
PROVIDER_FALLBACK = {
    DeviceType.CPU: ["CPUExecutionProvider"],
    DeviceType.GPU: [
        "ROCmExecutionProvider",
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ],
    DeviceType.NPU: [
        "VitisAIExecutionProvider",
        "RyzenAIExecutionProvider",
        "CPUExecutionProvider",
    ],
    DeviceType.IPU: [
        "DmlExecutionProvider",
        "ROCmExecutionProvider",
        "MIGraphXExecutionProvider",
        "CPUExecutionProvider",
    ],
    DeviceType.AUTO: None,
}


def _coerce_device_type(device: Any) -> DeviceType:
    """Accept a DeviceType or its value, such as npu or ipu."""
    if isinstance(device, DeviceType):
        return device
    text = str(device).split(".")[-1].strip().lower()
    try:
        return DeviceType(text)
    except ValueError as exc:
        known = ", ".join(item.value for item in DeviceType)
        raise ValueError(
            f"Unknown target_device {device!r}. Expected one of: {known}"
        ) from exc


def _coerce_execution_policy(policy: Any) -> ExecutionPolicy:
    """Accept an ExecutionPolicy or its value."""
    if isinstance(policy, ExecutionPolicy):
        return policy
    text = str(policy).split(".")[-1].strip().lower()
    try:
        return ExecutionPolicy(text)
    except ValueError as exc:
        known = ", ".join(item.value for item in ExecutionPolicy)
        raise ValueError(
            f"Unknown execution_policy {policy!r}. Expected one of: {known}"
        ) from exc


def _providers_include_npu(providers: Iterable[str]) -> bool:
    """True when any provider name is an ahead-of-time NPU."""
    for name in providers:
        lowered = str(name).lower()
        if any(marker in lowered for marker in _NPU_PROVIDER_MARKERS):
            return True
    return False


def _auto_provider_fallback() -> List[str]:
    """NPU, then AMD iGPU, then discrete GPU, with CPU last."""
    selected: List[str] = []
    for kind in (DeviceType.NPU, DeviceType.IPU, DeviceType.GPU):
        for name in PROVIDER_FALLBACK[kind]:
            if name == "CPUExecutionProvider" or name in selected:
                continue
            selected.append(name)
    selected.append("CPUExecutionProvider")
    return selected


# Planning names shared with model-config quantization. Sub-byte layouts
# stay on TensorDtype.INT4; the packing name records AWQ, GPTQ, or MatMulNBits.
_QUANT_METHOD_ALIASES = {
    "awq": "awq",
    "gptq": "gptq",
    "matmulnbits": "matmul_nbits",
    "matmul_nbits": "matmul_nbits",
    "rtn": "rtn",
    "qdq": "qdq",
    "quark": "quark",
    "bitsandbytes": "bitsandbytes",
    "bnb": "bitsandbytes",
}
_QUANT_PACKING_ALIASES = {
    "awq": "awq",
    "awq_int4": "awq",
    "awq_w4a16": "awq",
    "gptq": "gptq",
    "gptq_int4": "gptq",
    "matmulnbits": "matmul_nbits",
    "matmul_nbits": "matmul_nbits",
    "blocked": "blocked",
    "int4_blockwise": "blocked",
    "qdq": "qdq",
    "qoperator": "qoperator",
    "gemm": "gemm",
    "gemv": "gemv",
}
_METHOD_DEFAULT_PACKING = {
    "awq": "awq",
    "gptq": "gptq",
    "matmul_nbits": "matmul_nbits",
}
_TENSOR_DTYPE_ALIASES = {
    "float32": TensorDtype.FLOAT32,
    "fp32": TensorDtype.FLOAT32,
    "float": TensorDtype.FLOAT32,
    "float16": TensorDtype.FLOAT16,
    "fp16": TensorDtype.FLOAT16,
    "half": TensorDtype.FLOAT16,
    "bfloat16": TensorDtype.BFLOAT16,
    "bf16": TensorDtype.BFLOAT16,
    "int8": TensorDtype.INT8,
    "uint8": TensorDtype.INT8,
    "int4": TensorDtype.INT4,
    "uint4": TensorDtype.INT4,
    "int4x2": TensorDtype.INT4,
    "nf4": TensorDtype.INT4,
    "fp4": TensorDtype.INT4,
    "auto": TensorDtype.AUTO,
}
# Activation and index tensors use ONNX names that are not weight dtypes.
_INDEX_DTYPE_ALIASES = {
    "int64": "int64",
    "int32": "int32",
    "int16": "int16",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "bool": "bool",
    "boolean": "bool",
    "float64": "float64",
    "double": "float64",
    "fp64": "float64",
}
_LAYER_FILE_NAME = re.compile(r"^layer_(\d+)$")
_MODEL_FILE_SUFFIXES = {
    ".onnx",
    ".safetensors",
    ".gguf",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
}


def _normalize_quant_token(value: Any) -> str:
    """Lowercase a planning name and drop torch. prefixes and separators."""
    text = str(value).strip().lower().replace(" ", "").replace("-", "_")
    if text.startswith("torch."):
        text = text[6:]
    return text


def _coerce_quant_method(value: Any) -> Optional[str]:
    """Accept an AWQ, GPTQ, MatMulNBits, or related method name."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(f"quantization method must be a string, got {value!r}")
    text = _normalize_quant_token(value)
    if not text:
        return None
    if text not in _QUANT_METHOD_ALIASES:
        known = ", ".join(sorted(set(_QUANT_METHOD_ALIASES.values())))
        raise ValueError(
            f"Unknown quantization method {value!r}. Expected one of: {known}"
        )
    return _QUANT_METHOD_ALIASES[text]


def _coerce_quant_dtype(value: Any, field_name: str) -> Optional[TensorDtype]:
    """Accept a TensorDtype or a short name such as fp16 or int4."""
    if value is None:
        return None
    if isinstance(value, TensorDtype):
        return value
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(f"{field_name} must be a TensorDtype, got {value!r}")
    text = _normalize_quant_token(value)
    if text not in _TENSOR_DTYPE_ALIASES:
        known = ", ".join(item.value for item in TensorDtype)
        raise ValueError(
            f"Unknown {field_name} {value!r}. Expected one of: {known}"
        )
    return _TENSOR_DTYPE_ALIASES[text]


def _coerce_group_size(value: Any) -> Optional[int]:
    """None stays unset. A positive int is a group. -1 is per-channel."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"group_size must be an integer, got {value!r}")
    if value == 0 or value < -1:
        raise ValueError(
            f"group_size must be positive or -1 for per-channel, got {value}"
        )
    return value


def _coerce_packing_format(value: Any) -> Optional[str]:
    """Accept AWQ, GPTQ, MatMulNBits, blocked, QDQ, GEMM, or GEMV packing."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(f"packing_format must be a string, got {value!r}")
    text = _normalize_quant_token(value)
    if not text:
        return None
    if text not in _QUANT_PACKING_ALIASES:
        known = ", ".join(sorted(set(_QUANT_PACKING_ALIASES.values())))
        raise ValueError(
            f"Unknown packing_format {value!r}. Expected one of: {known}"
        )
    return _QUANT_PACKING_ALIASES[text]


def _from_record(cls: Any, data: Any, label: str, drop: Iterable[str] = ()) -> Any:
    """Build a dataclass from a JSON object. Unknown keys and a non-dict raise."""
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a dict, got {data!r}")
    payload = dict(data)
    for key in drop:
        payload.pop(key, None)
    try:
        return cls(**payload)
    except TypeError as exc:
        raise ValueError(f"{label} fields are incomplete, got {data!r}") from exc


def _copy_json_tree(value: Any) -> Any:
    """Copy JSON data so a loaded document does not share lists with the caller."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_copy_json_tree(item) for item in value]
    if isinstance(value, dict):
        copied: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"JSON key must be a string, got {key!r}")
            copied[key] = _copy_json_tree(item)
        return copied
    raise ValueError(f"value is not JSON data, got {value!r}")


@dataclass
class QuantizationConfig:
    """
    Weight, activation, and KV quantization for an AMD NPU split.

    ``group_size`` -1 is per-channel. ``packing_format`` names AWQ, GPTQ, or
    MatMulNBits when the method does not already imply that layout. INT4 and
    INT8 stay on ``TensorDtype``; the packing name records how those values
    are stored.
    """
    method: Optional[str] = None
    weight_dtype: Optional[TensorDtype] = None
    activation_dtype: Optional[TensorDtype] = None
    kv_cache_dtype: Optional[TensorDtype] = None
    group_size: Optional[int] = None
    has_zero_point: bool = False
    packing_format: Optional[str] = None

    def __post_init__(self) -> None:
        self.method = _coerce_quant_method(self.method)
        self.weight_dtype = _coerce_quant_dtype(self.weight_dtype, "weight_dtype")
        self.activation_dtype = _coerce_quant_dtype(
            self.activation_dtype, "activation_dtype"
        )
        self.kv_cache_dtype = _coerce_quant_dtype(self.kv_cache_dtype, "kv_cache_dtype")
        self.group_size = _coerce_group_size(self.group_size)
        if not isinstance(self.has_zero_point, bool):
            raise ValueError(
                f"has_zero_point must be a boolean, got {self.has_zero_point!r}"
            )
        self.packing_format = _coerce_packing_format(self.packing_format)
        if self.packing_format is None and self.method in _METHOD_DEFAULT_PACKING:
            self.packing_format = _METHOD_DEFAULT_PACKING[self.method]

    def to_dict(self) -> Dict[str, Any]:
        """Copy the record. Dtypes are stored as their string values."""
        return {
            "method": self.method,
            "weight_dtype": None if self.weight_dtype is None else self.weight_dtype.value,
            "activation_dtype": (
                None if self.activation_dtype is None else self.activation_dtype.value
            ),
            "kv_cache_dtype": (
                None if self.kv_cache_dtype is None else self.kv_cache_dtype.value
            ),
            "group_size": self.group_size,
            "has_zero_point": self.has_zero_point,
            "packing_format": self.packing_format,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantizationConfig":
        """Build a record from a JSON object. Dtype names are coerced."""
        return _from_record(cls, data, "quantization")


def _coerce_quantization(value: Any) -> QuantizationConfig:
    """Copy a QuantizationConfig, or build one from a dict."""
    if isinstance(value, QuantizationConfig):
        return QuantizationConfig(
            method=value.method,
            weight_dtype=value.weight_dtype,
            activation_dtype=value.activation_dtype,
            kv_cache_dtype=value.kv_cache_dtype,
            group_size=value.group_size,
            has_zero_point=value.has_zero_point,
            packing_format=value.packing_format,
        )
    if isinstance(value, dict):
        return QuantizationConfig(**value)
    raise ValueError(
        f"quantization must be a QuantizationConfig or a dict, got {value!r}"
    )


class KVCachePolicy(str, Enum):
    """
    How a split plans to keep past tokens.

    This is the planning name. The runtime cache manager still uses
    dense_contiguous and paged_blocks for the same two layouts.
    """
    DENSE = "dense"
    SLIDING_WINDOW = "sliding_window"
    PAGED = "paged"
    HYBRID_SPARSE = "hybrid_sparse"


_KV_POLICY_ALIASES = {
    "dense": KVCachePolicy.DENSE,
    "dense_contiguous": KVCachePolicy.DENSE,
    "contiguous": KVCachePolicy.DENSE,
    "sliding_window": KVCachePolicy.SLIDING_WINDOW,
    "sliding": KVCachePolicy.SLIDING_WINDOW,
    "window": KVCachePolicy.SLIDING_WINDOW,
    "paged": KVCachePolicy.PAGED,
    "paged_blocks": KVCachePolicy.PAGED,
    "paged_attention": KVCachePolicy.PAGED,
    "blocks": KVCachePolicy.PAGED,
    "hybrid_sparse": KVCachePolicy.HYBRID_SPARSE,
    "hybrid": KVCachePolicy.HYBRID_SPARSE,
    "sparse": KVCachePolicy.HYBRID_SPARSE,
}
_KV_LAYOUT_ALIASES = {
    "bnsd": "BNSD",
    "bnsh": "BNSD",
    "batch_heads_seq_dim": "BNSD",
    "batch_num_seq_dim": "BNSD",
}


def _coerce_kv_policy(value: Any) -> KVCachePolicy:
    """Accept a planning policy, or a runtime name such as paged_blocks."""
    if isinstance(value, KVCachePolicy):
        return value
    raw = getattr(value, "value", value)
    if isinstance(raw, bool) or not isinstance(raw, str):
        raise ValueError(f"KV cache policy must be a KVCachePolicy, got {value!r}")
    text = raw.strip().lower().replace("-", "_").replace(" ", "_")
    text = text.split(".")[-1]
    if text not in _KV_POLICY_ALIASES:
        known = ", ".join(item.value for item in KVCachePolicy)
        raise ValueError(
            f"Unknown KV cache policy {value!r}. Expected one of: {known}"
        )
    return _KV_POLICY_ALIASES[text]


def _coerce_kv_layout(value: Any) -> str:
    """Store BNSD for the layout the cache allocates. Other names stay uppercase."""
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise ValueError(f"layout must be a non-empty string, got {value!r}")
    token = value.strip().lower().replace("-", "_").replace(" ", "_")
    if token in _KV_LAYOUT_ALIASES:
        return _KV_LAYOUT_ALIASES[token]
    compact = token.replace("_", "")
    if compact.isalpha():
        return token.upper()
    raise ValueError(f"layout {value!r} is not a recognized kv cache layout")


def _coerce_non_negative_int(value: Any, field_name: str) -> int:
    """Reject a bool or a negative count. Zero stays unset."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative, got {value}")
    return value


def _coerce_page_size(value: Any) -> int:
    """A page holds at least one token."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"page_size must be an integer, got {value!r}")
    if value < 1:
        raise ValueError(f"page_size must be positive, got {value}")
    return value


def _coerce_sliding_window(value: Any) -> Optional[int]:
    """None stays unset. A window holds at least one token."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"sliding_window must be an integer, got {value!r}")
    if value < 1:
        raise ValueError(f"sliding_window must be positive, got {value}")
    return value


@dataclass
class KVCacheConfig:
    """
    Capacity and layout for the KV cache attached to a split.

    ``max_seq_len`` 0 means the capacity is unset. ``layout`` BNSD is batch,
    heads, sequence, head dim. A sliding-window policy requires
    ``sliding_window``. ``page_size`` is the paged block length.
    """
    dtype: TensorDtype = TensorDtype.FLOAT16
    layout: str = "BNSD"
    max_seq_len: int = 0
    policy: KVCachePolicy = KVCachePolicy.DENSE
    page_size: int = 1
    sliding_window: Optional[int] = None

    def __post_init__(self) -> None:
        coerced = _coerce_quant_dtype(self.dtype, "dtype")
        if coerced is None:
            raise ValueError("dtype is required")
        self.dtype = coerced
        self.layout = _coerce_kv_layout(self.layout)
        self.max_seq_len = _coerce_non_negative_int(self.max_seq_len, "max_seq_len")
        self.policy = _coerce_kv_policy(self.policy)
        self.page_size = _coerce_page_size(self.page_size)
        self.sliding_window = _coerce_sliding_window(self.sliding_window)
        if self.policy is KVCachePolicy.SLIDING_WINDOW and self.sliding_window is None:
            raise ValueError("sliding_window policy requires a positive sliding_window")

    def to_dict(self) -> Dict[str, Any]:
        """Copy the record. The dtype and policy are stored as their string values."""
        return {
            "dtype": self.dtype.value,
            "layout": self.layout,
            "max_seq_len": self.max_seq_len,
            "policy": self.policy.value,
            "page_size": self.page_size,
            "sliding_window": self.sliding_window,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KVCacheConfig":
        """Build a record from a JSON object. Policy names are coerced."""
        return _from_record(cls, data, "kv cache")


def _canonical_dtype_name(value: Any) -> str:
    """Map fp16 and float16 onto one name. int64 stays an index dtype."""
    if isinstance(value, TensorDtype):
        return value.value
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise ValueError(f"dtype must be a dtype name, got {value!r}")
    text = _normalize_quant_token(value)
    if text in _TENSOR_DTYPE_ALIASES:
        return _TENSOR_DTYPE_ALIASES[text].value
    if text in _INDEX_DTYPE_ALIASES:
        return _INDEX_DTYPE_ALIASES[text]
    known = ", ".join(item.value for item in TensorDtype)
    raise ValueError(f"Unknown dtype {value!r}. Expected one of: {known}")


def _coerce_required_dtype(value: Any) -> TensorDtype:
    """A split dtype is required. fp16 is stored as float16."""
    coerced = _coerce_quant_dtype(value, "dtype")
    if coerced is None:
        raise ValueError("dtype is required")
    return coerced


def _coerce_memory_mode(value: Any) -> MemoryMode:
    """Accept a MemoryMode or its value, such as streaming or eager."""
    if isinstance(value, MemoryMode):
        return value
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(f"memory_mode must be a MemoryMode, got {value!r}")
    text = value.strip().lower()
    try:
        return MemoryMode(text)
    except ValueError as exc:
        known = ", ".join(item.value for item in MemoryMode)
        raise ValueError(
            f"Unknown memory_mode {value!r}. Expected one of: {known}"
        ) from exc


def _model_path_must_exist(model: str) -> bool:
    """A filesystem model must exist. A Hugging Face id such as org/name does not."""
    if os.path.isabs(model) or model.startswith(".") or model.startswith("~"):
        return True
    return os.path.splitext(model)[1].lower() in _MODEL_FILE_SUFFIXES


def _coerce_kv_cache(value: Any) -> KVCacheConfig:
    """Copy a KVCacheConfig, or build one from a dict."""
    if isinstance(value, KVCacheConfig):
        return KVCacheConfig(
            dtype=value.dtype,
            layout=value.layout,
            max_seq_len=value.max_seq_len,
            policy=value.policy,
            page_size=value.page_size,
            sliding_window=value.sliding_window,
        )
    if isinstance(value, dict):
        return KVCacheConfig(**value)
    raise ValueError(
        f"kv_cache must be a KVCacheConfig or a dict, got {value!r}"
    )


@dataclass
class SplitConfig:
    """
    Configuration for model splitting operations.
    
    This is a unified configuration that works for both ONNX and SafeTensors
    model splitting. Some fields may only be relevant for specific formats.
    
    Attributes:
        model: Path to the model file/directory or HuggingFace model ID.
        output: Base directory for split output files.
        split_only: If True, skip inference validation after splitting.
        
        # Memory efficiency options
        dtype: Target tensor dtype (AUTO preserves original).
        memory_mode: Loading strategy (STREAMING for low-memory environments).
        device: Torch or runtime device string, such as cpu or cuda:0.
        providers: ONNX Runtime provider fallback, first choice first.
        target_device: CPU, GPU, NPU, IPU, or AUTO.
        execution_policy: Monolithic, sequential, persistent, hybrid, or speculative.
        quantization: Weight, activation, and KV scheme, including group size and packing.
        kv_cache: Cache dtype, layout, capacity, policy, page size, and sliding window.
        low_cpu_mem_usage: Use HuggingFace's low-memory loading.
        
        # HuggingFace integration
        trust_remote_code: Allow custom code from model repos.
        use_accelerate: Enable accelerate library for device_map.
        
        # Inference testing
        context: Maximum context length for tokenization.
        batch_size: Batch size for inference testing.
        prompt: Test prompt for inference validation.
        max_new_tokens: Maximum new tokens to generate.
        verbose: Enable verbose output during splitting.
    """
    model: str
    output: str
    split_only: bool = False
    
    # Memory efficiency options
    dtype: TensorDtype = TensorDtype.AUTO
    memory_mode: MemoryMode = MemoryMode.STREAMING  # Default to memory-efficient
    device: str = "cpu"
    providers: List[str] = field(default_factory=list)
    target_device: DeviceType = DeviceType.AUTO
    execution_policy: ExecutionPolicy = ExecutionPolicy.AUTO
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)
    low_cpu_mem_usage: bool = True
    
    # HuggingFace integration
    trust_remote_code: bool = True
    use_accelerate: bool = False
    
    # Inference testing
    context: int = 1024
    batch_size: int = 1
    prompt: str = "Test prompt."
    max_new_tokens: int = 50
    verbose: bool = False

    def __post_init__(self) -> None:
        self.dtype = _coerce_required_dtype(self.dtype)
        self.memory_mode = _coerce_memory_mode(self.memory_mode)
        self.target_device = _coerce_device_type(self.target_device)
        self.execution_policy = _coerce_execution_policy(self.execution_policy)
        if self.providers is None:
            self.providers = []
        if isinstance(self.providers, (str, bytes)) or not isinstance(self.providers, list):
            raise ValueError(
                f"providers must be a list of provider names, got {self.providers!r}"
            )
        cleaned: List[str] = []
        for item in self.providers:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"provider name must be a non-empty string, got {item!r}"
                )
            cleaned.append(item.strip())
        self.providers = cleaned
        self.quantization = _coerce_quantization(self.quantization)
        self.kv_cache = _coerce_kv_cache(self.kv_cache)
        self._reject_npu_sequential_split()

    def _reject_npu_sequential_split(self) -> None:
        """
        Refuse sequential and persistent layer splits on an NPU.

        AUTO and monolithic stay valid. Hybrid phase split, MoE routing, and
        speculative draft and verify are not layer series.
        """
        if self.execution_policy not in (
            ExecutionPolicy.SEQUENTIAL,
            ExecutionPolicy.PERSISTENT,
        ):
            return
        on_npu = (
            self.target_device is DeviceType.NPU
            or _providers_include_npu(self.resolved_providers())
        )
        if on_npu:
            raise ValueError(
                f"execution_policy {self.execution_policy.value} is rejected "
                "on an NPU provider. Sequential layer splitting on NPU is unsafe."
            )

    def validate(self) -> None:
        """
        Reject a missing model file, an invalid memory mode, an inconsistent
        dtype, or a sequential NPU policy.

        A Hugging Face model id is not a file. ``dtype`` AUTO does not have to
        match the activation dtype.
        """
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError(f"model must be a non-empty string, got {self.model!r}")
        if not isinstance(self.output, str) or not self.output.strip():
            raise ValueError(f"output must be a non-empty string, got {self.output!r}")
        if not isinstance(self.dtype, TensorDtype):
            raise ValueError(f"Unknown dtype {self.dtype!r}")
        if not isinstance(self.memory_mode, MemoryMode):
            known = ", ".join(item.value for item in MemoryMode)
            raise ValueError(
                f"Unknown memory_mode {self.memory_mode!r}. Expected one of: {known}"
            )
        if _model_path_must_exist(self.model) and not os.path.exists(self.model):
            raise ValueError(f"missing file {self.model}")
        activation = self.quantization.activation_dtype
        if (
            self.dtype is not TensorDtype.AUTO
            and activation is not None
            and self.dtype is not activation
        ):
            raise ValueError(
                f"inconsistent dtype: split dtype {self.dtype.value} and "
                f"activation dtype {activation.value}"
            )
        quant_kv = self.quantization.kv_cache_dtype
        if quant_kv is not None and quant_kv is not self.kv_cache.dtype:
            raise ValueError(
                f"inconsistent dtype: quantization kv cache {quant_kv.value} and "
                f"kv cache {self.kv_cache.dtype.value}"
            )
        self._reject_npu_sequential_split()

    def to_dict(self) -> Dict[str, Any]:
        """Copy the split config. Enums are stored as their string values."""
        return {
            "model": self.model,
            "output": self.output,
            "split_only": self.split_only,
            "dtype": self.dtype.value,
            "memory_mode": self.memory_mode.value,
            "device": self.device,
            "providers": list(self.providers),
            "target_device": self.target_device.value,
            "execution_policy": self.execution_policy.value,
            "quantization": self.quantization.to_dict(),
            "kv_cache": self.kv_cache.to_dict(),
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "trust_remote_code": self.trust_remote_code,
            "use_accelerate": self.use_accelerate,
            "context": self.context,
            "batch_size": self.batch_size,
            "prompt": self.prompt,
            "max_new_tokens": self.max_new_tokens,
            "verbose": self.verbose,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SplitConfig":
        """Build a split config from a JSON object."""
        return _from_record(cls, data, "split config")

    def resolved_providers(self) -> List[str]:
        """
        Provider fallback order.

        An explicit ``providers`` list is kept. An empty list uses the order
        for ``target_device``. AUTO tries the NPU, then the AMD iGPU, then
        a discrete GPU, and CPU last.
        """
        if self.providers:
            return list(self.providers)
        mapped = PROVIDER_FALLBACK[self.target_device]
        if mapped is None:
            return _auto_provider_fallback()
        return list(mapped)


from .architecture import ArchitectureType  # Re-export for convenience


def _copy_dim_list(shape: Any, name: str) -> List[Optional[int]]:
    """Copy one shape. None is a symbolic axis. Every other axis is a non-negative int."""
    if isinstance(shape, (str, bytes)) or not isinstance(shape, list):
        raise ValueError(f"{name} must be a list of dimensions, got {shape!r}")
    dims: List[Optional[int]] = []
    for dim in shape:
        if dim is None:
            dims.append(None)
            continue
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            raise ValueError(f"{name} dimension must be a non-negative integer or None, got {dim!r}")
        dims.append(dim)
    return dims


def _boundary_name_is_stateful(name: str) -> bool:
    """Past and present KV tensors carry state across steps."""
    lowered = name.lower()
    return (
        lowered == "past_key_values"
        or lowered.startswith("past_key_values.")
        or lowered == "present"
        or lowered.startswith("present.")
    )


@dataclass
class BoundaryTensor:
    """
    One tensor that crosses a component boundary.

    ``hidden_states``, ``attention_mask``, and ``position_ids`` are activations.
    ``past_key_values.*`` and ``present.*`` are stateful. ``None`` in ``shape``
    is a symbolic axis such as the sequence length.
    """
    name: str
    dtype: str
    shape: List[Optional[int]]
    producer_component: str
    consumer_components: List[str]
    is_stateful: bool = False
    device_preference: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"boundary tensor name must be a non-empty string, got {self.name!r}")
        self.name = self.name.strip()
        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise ValueError(f"boundary tensor dtype must be a non-empty string, got {self.dtype!r}")
        self.dtype = self.dtype.strip()
        self.shape = _copy_dim_list(self.shape, f"shape of {self.name!r}")
        if not isinstance(self.producer_component, str) or not self.producer_component.strip():
            raise ValueError(
                f"producer_component must be a non-empty string, got {self.producer_component!r}"
            )
        self.producer_component = self.producer_component.strip()
        self.consumer_components = _copy_name_list(
            self.consumer_components, "consumer_components"
        )
        if not isinstance(self.is_stateful, bool):
            raise ValueError(f"is_stateful must be a bool, got {self.is_stateful!r}")
        if _boundary_name_is_stateful(self.name):
            self.is_stateful = True
        self.device_preference = _coerce_component_device(self.device_preference)

    def to_dict(self) -> Dict[str, Any]:
        """Copy the boundary so IO binding can allocate without sharing this object."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "producer_component": self.producer_component,
            "consumer_components": list(self.consumer_components),
            "is_stateful": self.is_stateful,
            "device_preference": self.device_preference,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BoundaryTensor":
        """Build a boundary tensor from a JSON object. A symbolic axis stays None."""
        return _from_record(cls, data, "boundary tensor")


def _coerce_boundary_tensor(value: Any) -> BoundaryTensor:
    """Accept a BoundaryTensor or a mapping with the same fields."""
    if isinstance(value, BoundaryTensor):
        return value
    if isinstance(value, dict):
        try:
            return BoundaryTensor(**value)
        except TypeError as exc:
            raise ValueError(f"boundary tensor fields are incomplete, got {value!r}") from exc
    raise ValueError(f"boundary tensor must be a BoundaryTensor, got {value!r}")


def _coerce_optional_provider(value: Any) -> Optional[str]:
    """None stays unset. A provider name is stripped."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(f"provider must be a string, got {value!r}")
    text = value.strip()
    if not text:
        return None
    return text


def _coerce_plan_device(value: Any, field_name: str) -> Optional[str]:
    """None stays unset. xdna is the NPU and rdna is the iGPU."""
    try:
        return _coerce_component_device(value)
    except ValueError as exc:
        raise ValueError(str(exc).replace("device_preference", field_name)) from exc


def _coerce_step_phase(value: Any) -> str:
    """A step always has a phase. Draft and verify use the scheduler names."""
    if value is None or value == "":
        raise ValueError("phase is required")
    phase = _coerce_component_phase(value)
    if phase is None:
        raise ValueError("phase is required")
    return phase


def _step_uses_npu(step: "ExecutionStep") -> bool:
    """True when the step is placed on an ahead-of-time NPU."""
    if step.device == "npu":
        return True
    if step.provider and _providers_include_npu([step.provider]):
        return True
    return False


def _reject_plan_cycle(steps: List["ExecutionStep"]) -> None:
    """Reject a dependency that is missing or that loops."""
    names = [step.component_name for step in steps]
    present = set(names)
    children: Dict[str, List[str]] = {name: [] for name in names}
    indegree = {name: 0 for name in names}
    for step in steps:
        for dependency in step.depends_on:
            if dependency not in present:
                raise ValueError(
                    f"Step {step.component_name!r} depends on {dependency!r}, "
                    "which is not in the plan."
                )
            children[dependency].append(step.component_name)
            indegree[step.component_name] += 1
    ready = [name for name in names if indegree[name] == 0]
    seen = 0
    while ready:
        name = ready.pop(0)
        seen += 1
        for child in children[name]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if seen != len(names):
        raise ValueError("execution plan has a dependency cycle")


@dataclass
class ExecutionStep:
    """
    One component in a split execution plan.

    ``phase`` defaults to decode. ``depends_on`` names earlier components.
    ``resident`` keeps that component loaded. ``device`` uses the AMD names,
    so xdna is the NPU and rdna is the iGPU.
    """
    component_name: str
    provider: Optional[str] = None
    device: Optional[str] = None
    phase: str = "decode"
    depends_on: List[str] = field(default_factory=list)
    resident: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.component_name, str) or not self.component_name.strip():
            raise ValueError(
                f"component_name must be a non-empty string, got {self.component_name!r}"
            )
        self.component_name = self.component_name.strip()
        self.provider = _coerce_optional_provider(self.provider)
        self.device = _coerce_plan_device(self.device, "device")
        self.phase = _coerce_step_phase(self.phase)
        self.depends_on = _copy_name_list(self.depends_on, "depends_on")
        if self.component_name in self.depends_on:
            raise ValueError(
                f"Step {self.component_name!r} depends on itself"
            )
        if not isinstance(self.resident, bool):
            raise ValueError(f"resident must be a boolean, got {self.resident!r}")

    def to_dict(self) -> Dict[str, Any]:
        """Copy the step so a scheduler can read it without sharing this object."""
        return {
            "component_name": self.component_name,
            "provider": self.provider,
            "device": self.device,
            "phase": self.phase,
            "depends_on": list(self.depends_on),
            "resident": self.resident,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionStep":
        """Build one plan step from a JSON object."""
        return _from_record(cls, data, "execution step")


def _coerce_execution_step(value: Any) -> ExecutionStep:
    """Copy an ExecutionStep, or build one from a dict."""
    if isinstance(value, ExecutionStep):
        return ExecutionStep(
            component_name=value.component_name,
            provider=value.provider,
            device=value.device,
            phase=value.phase,
            depends_on=list(value.depends_on),
            resident=value.resident,
        )
    if isinstance(value, dict):
        return ExecutionStep(**value)
    raise ValueError(
        f"execution step must be an ExecutionStep or a dict, got {value!r}"
    )


@dataclass
class ExecutionPlan:
    """
    How a split model runs on AMD hardware.

    Prefill can stay on the NPU while decode stays on the iGPU. ``policy``
    AUTO with more than one NPU step is rejected, and a sequential or
    persistent policy is rejected when any step is on the NPU.
    """
    steps: List[ExecutionStep] = field(default_factory=list)
    policy: ExecutionPolicy = ExecutionPolicy.AUTO
    prefill_device: Optional[str] = None
    decode_device: Optional[str] = None
    static_shapes: bool = True
    warmup_iterations: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.steps, (str, bytes)) or not isinstance(self.steps, list):
            raise ValueError(f"steps must be a list, got {self.steps!r}")
        cleaned: List[ExecutionStep] = []
        seen = set()
        for item in self.steps:
            step = _coerce_execution_step(item)
            if step.component_name in seen:
                raise ValueError(
                    f"execution plan repeats component {step.component_name!r}"
                )
            seen.add(step.component_name)
            cleaned.append(step)
        self.steps = cleaned
        _reject_plan_cycle(self.steps)
        self.policy = _coerce_execution_policy(self.policy)
        self.prefill_device = _coerce_plan_device(self.prefill_device, "prefill_device")
        self.decode_device = _coerce_plan_device(self.decode_device, "decode_device")
        if not isinstance(self.static_shapes, bool):
            raise ValueError(
                f"static_shapes must be a boolean, got {self.static_shapes!r}"
            )
        self.warmup_iterations = _as_non_negative_int(
            self.warmup_iterations, "warmup_iterations"
        )
        self._reject_unsafe_npu_schedule()
        self._require_policy_phases()

    def validate(self) -> None:
        """
        Reject a missing dependency, a cycle, or an unsafe NPU schedule.

        An empty plan is a valid AUTO schedule. Hybrid, MoE, and speculative
        plans still require the phases they name.
        """
        if isinstance(self.steps, (str, bytes)) or not isinstance(self.steps, list):
            raise ValueError(f"steps must be a list, got {self.steps!r}")
        seen = set()
        for step in self.steps:
            if not isinstance(step, ExecutionStep):
                raise ValueError(
                    f"execution step must be an ExecutionStep, got {step!r}"
                )
            if step.component_name in seen:
                raise ValueError(
                    f"execution plan repeats component {step.component_name!r}"
                )
            seen.add(step.component_name)
            _coerce_step_phase(step.phase)
            _coerce_plan_device(step.device, "device")
        _reject_plan_cycle(self.steps)
        if not isinstance(self.policy, ExecutionPolicy):
            raise ValueError(
                f"Unknown execution_policy {self.policy!r}. "
                f"Expected one of: {', '.join(item.value for item in ExecutionPolicy)}"
            )
        _coerce_plan_device(self.prefill_device, "prefill_device")
        _coerce_plan_device(self.decode_device, "decode_device")
        if not isinstance(self.static_shapes, bool):
            raise ValueError(
                f"static_shapes must be a boolean, got {self.static_shapes!r}"
            )
        _as_non_negative_int(self.warmup_iterations, "warmup_iterations")
        self._reject_unsafe_npu_schedule()
        self._require_policy_phases()

    def _reject_unsafe_npu_schedule(self) -> None:
        """Refuse a sequential layer series on an NPU."""
        npu_steps = [step for step in self.steps if _step_uses_npu(step)]
        if self.policy in (ExecutionPolicy.SEQUENTIAL, ExecutionPolicy.PERSISTENT) and npu_steps:
            raise ValueError(
                f"execution_policy {self.policy.value} is rejected on an NPU provider. "
                "Sequential layer splitting on NPU is unsafe."
            )
        if (
            self.policy in (ExecutionPolicy.AUTO, ExecutionPolicy.MONOLITHIC)
            and len(npu_steps) > 1
        ):
            raise ValueError(
                f"execution_policy {self.policy.value} is rejected when more than one "
                "step is on an NPU provider."
            )

    def _require_policy_phases(self) -> None:
        """Hybrid, MoE, and speculative plans name the phases they run."""
        phases = {step.phase for step in self.steps}
        if self.policy in (ExecutionPolicy.HYBRID_NPU_IGPU, ExecutionPolicy.PHASE_SPLIT):
            if "prefill" not in phases or "decode" not in phases:
                raise ValueError(
                    f"{self.policy.value} requires a prefill step and a decode step."
                )
        if self.policy is ExecutionPolicy.MOE_ROUTING:
            if "router" not in phases or "expert" not in phases:
                raise ValueError(
                    f"{self.policy.value} requires a router step and an expert step."
                )
        if self.policy is ExecutionPolicy.SPECULATIVE:
            if "speculative_draft" not in phases or "verification" not in phases:
                raise ValueError(
                    f"{self.policy.value} requires a speculative_draft step and a verification step."
                )

    def to_dict(self) -> Dict[str, Any]:
        """Copy the plan. Each step is copied on its own."""
        return {
            "steps": [step.to_dict() for step in self.steps],
            "policy": self.policy.value,
            "prefill_device": self.prefill_device,
            "decode_device": self.decode_device,
            "static_shapes": self.static_shapes,
            "warmup_iterations": self.warmup_iterations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionPlan":
        """Build an execution plan from execution_plan.json."""
        return _from_record(cls, data, "execution plan")


def _coerce_execution_plan(value: Any) -> ExecutionPlan:
    """Copy an ExecutionPlan, or build one from a dict."""
    if isinstance(value, ExecutionPlan):
        return ExecutionPlan(
            steps=list(value.steps),
            policy=value.policy,
            prefill_device=value.prefill_device,
            decode_device=value.decode_device,
            static_shapes=value.static_shapes,
            warmup_iterations=value.warmup_iterations,
        )
    if isinstance(value, dict):
        return ExecutionPlan(**value)
    raise ValueError(
        f"execution_plan must be an ExecutionPlan or a dict, got {value!r}"
    )


@dataclass
class SplitResult:
    """
    Result of a model splitting operation.

    This is a unified result type that works for both ONNX and SafeTensors
    splitting operations.

    Attributes:
        output_dir: Directory where split files were saved.
        num_layers: Number of transformer layers in the model.
        component_files: Mapping of component names to file paths.
        model_info: Additional model metadata and information.
        format: The format that was split (safetensors or onnx).
        boundary_tensors: Tensors passed between components, including KV state.
        execution_plan: How those components run, including device and phase.
    """
    output_dir: str
    num_layers: int
    component_files: Dict[str, str]
    model_info: Dict[str, Any]
    format: ModelFormat = ModelFormat.SAFETENSORS
    boundary_tensors: List[BoundaryTensor] = field(default_factory=list)
    execution_plan: ExecutionPlan = field(default_factory=ExecutionPlan)

    def __post_init__(self) -> None:
        if not isinstance(self.format, ModelFormat):
            text = str(self.format).split(".")[-1].strip().lower()
            try:
                self.format = ModelFormat(text)
            except ValueError as exc:
                known = ", ".join(item.value for item in ModelFormat)
                raise ValueError(
                    f"Unknown format {self.format!r}. Expected one of: {known}"
                ) from exc
        if isinstance(self.boundary_tensors, (str, bytes)) or not isinstance(self.boundary_tensors, list):
            raise ValueError(
                f"boundary_tensors must be a list, got {self.boundary_tensors!r}"
            )
        cleaned: List[BoundaryTensor] = []
        seen = set()
        for item in self.boundary_tensors:
            tensor = _coerce_boundary_tensor(item)
            if tensor.name in seen:
                raise ValueError(f"boundary_tensors contains a duplicate name {tensor.name!r}")
            seen.add(tensor.name)
            cleaned.append(tensor)
        self.boundary_tensors = cleaned
        self.execution_plan = _coerce_execution_plan(self.execution_plan)

    def validate(self) -> None:
        """
        Reject an empty component map, a missing file, an invalid layer id,
        a dependency that is not in the map, or an inconsistent boundary dtype.
        """
        if not isinstance(self.component_files, dict) or not self.component_files:
            raise ValueError("component_files is empty")
        if not isinstance(self.output_dir, str) or not os.path.isdir(self.output_dir):
            raise ValueError(f"missing directory {self.output_dir!r}")
        if isinstance(self.num_layers, bool) or not isinstance(self.num_layers, int) or self.num_layers < 0:
            raise ValueError(
                f"num_layers must be a non-negative integer, got {self.num_layers!r}"
            )
        names = set()
        layer_ids: List[int] = []
        for name, path in self.component_files.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"component name must be a non-empty string, got {name!r}")
            if not isinstance(path, str) or not path.strip() or not os.path.isfile(path):
                raise ValueError(f"missing file {path!r} for component {name!r}")
            names.add(name)
            match = _LAYER_FILE_NAME.match(name)
            if match:
                layer_ids.append(int(match.group(1)))
        for layer_id in layer_ids:
            if layer_id >= self.num_layers:
                raise ValueError(
                    f"invalid layer id {layer_id} for num_layers {self.num_layers}"
                )
        for index in range(self.num_layers):
            layer_name = f"layer_{index}"
            if layer_name not in names:
                raise ValueError(f"missing component {layer_name}")
        for tensor in self.boundary_tensors:
            _canonical_dtype_name(tensor.dtype)
            if tensor.producer_component not in names:
                raise ValueError(
                    f"boundary {tensor.name!r} depends on {tensor.producer_component!r}, "
                    "which is not in component_files"
                )
            for consumer in tensor.consumer_components:
                if consumer not in names:
                    raise ValueError(
                        f"boundary {tensor.name!r} depends on {consumer!r}, "
                        "which is not in component_files"
                    )
        for step in self.execution_plan.steps:
            if step.component_name not in names:
                raise ValueError(
                    f"Step {step.component_name!r} is not in component_files"
                )
            for dependency in step.depends_on:
                if dependency not in names:
                    raise ValueError(
                        f"Step {step.component_name!r} depends on {dependency!r}, "
                        "which is not in component_files"
                    )
        self.execution_plan.validate()

    def to_dict(self) -> Dict[str, Any]:
        """Copy the split result. This object is the model_info.json document."""
        return {
            "output_dir": self.output_dir,
            "num_layers": self.num_layers,
            "component_files": dict(self.component_files),
            "model_info": _copy_json_tree(self.model_info),
            "format": self.format.value,
            "boundary_tensors": [tensor.to_dict() for tensor in self.boundary_tensors],
            "execution_plan": self.execution_plan.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SplitResult":
        """Build a split result from model_info.json."""
        if not isinstance(data, dict):
            raise ValueError(f"split result must be a dict, got {data!r}")
        payload = dict(data)
        if "model_info" in payload:
            payload["model_info"] = _copy_json_tree(payload["model_info"])
        if isinstance(payload.get("component_files"), dict):
            payload["component_files"] = dict(payload["component_files"])
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ValueError(
                f"split result fields are incomplete, got {data!r}"
            ) from exc

    def boundary_by_name(self) -> Dict[str, BoundaryTensor]:
        """Lookup used when IO binding allocates one buffer per boundary name."""
        return {tensor.name: tensor for tensor in self.boundary_tensors}


# Phases a split component may be assigned. A layer index is not a phase.
# Aliases collapse onto the names the session scheduler already accepts.
_COMPONENT_PHASES = frozenset({
    "prefill",
    "decode",
    "router",
    "expert",
    "kv_cache",
    "speculative_draft",
    "verification",
    "norm",
    "lm_head",
    "embedding",
})

_COMPONENT_PHASE_ALIASES = {
    "prefill": "prefill",
    "decode": "decode",
    "router": "router",
    "moe_router": "router",
    "moe": "router",
    "expert": "expert",
    "expert_execution": "expert",
    "experts": "expert",
    "kv_cache": "kv_cache",
    "kv_cache_management": "kv_cache",
    "kv": "kv_cache",
    "speculative_draft": "speculative_draft",
    "speculative": "speculative_draft",
    "draft": "speculative_draft",
    "verification": "verification",
    "verify": "verification",
    "norm": "norm",
    "lm_head": "lm_head",
    "embedding": "embedding",
}

_COMPONENT_RESIDENT_POLICIES = frozenset({"auto", "resident", "on_demand", "evict"})

_COMPONENT_DEVICE_ALIASES = {
    "auto": "auto",
    "npu": "npu",
    "xdna": "npu",
    "igpu": "igpu",
    "rdna": "igpu",
    "gpu": "gpu",
    "cpu": "cpu",
}


def _coerce_component_type(value: Any) -> ComponentType:
    """Accept a ComponentType or its value, such as layer or lm_head."""
    if isinstance(value, ComponentType):
        return value
    text = str(value).split(".")[-1].strip().lower()
    try:
        return ComponentType(text)
    except ValueError as exc:
        known = ", ".join(item.value for item in ComponentType)
        raise ValueError(
            f"Unknown component_type {value!r}. Expected one of: {known}"
        ) from exc


def _as_non_negative_int(value: Any, name: str) -> int:
    """Reject bools and negatives. A count or an order is a real integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _copy_name_list(value: Any, name: str) -> List[str]:
    """Copy a list of tensor, file, or component names."""
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise ValueError(f"{name} must be a list of names, got {value!r}")
    cleaned: List[str] = []
    seen = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name} entry must be a non-empty string, got {item!r}")
        text = item.strip()
        if text in seen:
            raise ValueError(f"{name} contains a duplicate name {text!r}")
        seen.add(text)
        cleaned.append(text)
    return cleaned


def _copy_type_map(value: Any, names: List[str], field_name: str, names_field: str) -> Dict[str, str]:
    """Copy dtype names. Every key must already be listed in the name list."""
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dict, got {value!r}")
    allowed = set(names)
    cleaned: Dict[str, str] = {}
    for key, dtype in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{field_name} key must be a non-empty string, got {key!r}")
        text = key.strip()
        if text not in allowed:
            raise ValueError(f"{field_name} key {text!r} is not in {names_field}")
        if not isinstance(dtype, str) or not dtype.strip():
            raise ValueError(
                f"{field_name}[{text!r}] must be a non-empty type name, got {dtype!r}"
            )
        cleaned[text] = dtype.strip()
    return cleaned


def _copy_shape_map(
    value: Any,
    names: List[str],
    field_name: str,
    names_field: str,
) -> Dict[str, List[Optional[int]]]:
    """Copy shapes. None is a symbolic axis. Every other axis is a non-negative int."""
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dict, got {value!r}")
    allowed = set(names)
    cleaned: Dict[str, List[Optional[int]]] = {}
    for key, shape in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{field_name} key must be a non-empty string, got {key!r}")
        text = key.strip()
        if text not in allowed:
            raise ValueError(f"{field_name} key {text!r} is not in {names_field}")
        if isinstance(shape, (str, bytes)) or not isinstance(shape, list):
            raise ValueError(f"{field_name}[{text!r}] must be a list of dimensions, got {shape!r}")
        dims: List[Optional[int]] = []
        for dim in shape:
            if dim is None:
                dims.append(None)
                continue
            dims.append(_as_non_negative_int(dim, f"{field_name}[{text!r}] dimension"))
        cleaned[text] = dims
    return cleaned


def _coerce_component_phase(phase: Any) -> Optional[str]:
    """None stays unset. Draft and verify collapse onto the scheduler names."""
    if phase is None or phase == "":
        return None
    text = str(getattr(phase, "value", phase)).strip().lower().replace("-", "_").replace(" ", "_")
    text = _COMPONENT_PHASE_ALIASES.get(text, text)
    if text not in _COMPONENT_PHASES:
        allowed = ", ".join(sorted(_COMPONENT_PHASES))
        raise ValueError(f"Unknown phase {phase!r}. Expected one of: {allowed}")
    return text


def _coerce_component_device(preference: Any) -> Optional[str]:
    """None uses the caller provider list. xdna is the NPU and rdna is the iGPU."""
    if preference is None or preference == "":
        return None
    text = str(preference).strip().lower()
    if text not in _COMPONENT_DEVICE_ALIASES:
        allowed = ", ".join(sorted({"auto", "npu", "igpu", "gpu", "cpu"}))
        raise ValueError(
            f"Unknown device_preference {preference!r}. Expected one of: {allowed}"
        )
    return _COMPONENT_DEVICE_ALIASES[text]


def _coerce_component_resident(policy: Any) -> str:
    """Normalize residency. transient is the on-demand policy."""
    text = str(policy if policy is not None else "auto").strip().lower().replace("-", "_")
    if text == "transient":
        text = "on_demand"
    if text not in _COMPONENT_RESIDENT_POLICIES:
        allowed = ", ".join(sorted(_COMPONENT_RESIDENT_POLICIES))
        raise ValueError(
            f"Unknown resident_policy {policy!r}. Expected one of: {allowed}"
        )
    return text


@dataclass
class ComponentInfo:
    """
    Execution contract for one split component.

    The type, layer, file, tensor count, and size identify the artifact.
    Names, dtypes, and shapes describe the boundary. External data, dependencies,
    order, device, residency, and phase describe how it runs. An omitted phase
    stays unset, including for a layer.
    """
    component_type: ComponentType
    layer_id: Optional[int] = None
    file_path: Optional[str] = None
    num_tensors: int = 0
    size_bytes: int = 0

    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    input_types: Dict[str, str] = field(default_factory=dict)
    output_types: Dict[str, str] = field(default_factory=dict)
    input_shapes: Dict[str, List[Optional[int]]] = field(default_factory=dict)
    output_shapes: Dict[str, List[Optional[int]]] = field(default_factory=dict)

    external_data_files: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    execution_order: int = 0
    device_preference: Optional[str] = None
    resident_policy: str = "auto"
    phase: Optional[str] = None

    def __post_init__(self) -> None:
        self.component_type = _coerce_component_type(self.component_type)
        if self.layer_id is not None:
            self.layer_id = _as_non_negative_int(self.layer_id, "layer_id")
        if self.file_path is not None:
            if not isinstance(self.file_path, str) or not self.file_path.strip():
                raise ValueError(f"file_path must be a non-empty string, got {self.file_path!r}")
            self.file_path = self.file_path.strip()
        self.num_tensors = _as_non_negative_int(self.num_tensors, "num_tensors")
        self.size_bytes = _as_non_negative_int(self.size_bytes, "size_bytes")
        self.input_names = _copy_name_list(self.input_names, "input_names")
        self.output_names = _copy_name_list(self.output_names, "output_names")
        self.input_types = _copy_type_map(
            self.input_types, self.input_names, "input_types", "input_names"
        )
        self.output_types = _copy_type_map(
            self.output_types, self.output_names, "output_types", "output_names"
        )
        self.input_shapes = _copy_shape_map(
            self.input_shapes, self.input_names, "input_shapes", "input_names"
        )
        self.output_shapes = _copy_shape_map(
            self.output_shapes, self.output_names, "output_shapes", "output_names"
        )
        self.external_data_files = _copy_name_list(
            self.external_data_files, "external_data_files"
        )
        self.depends_on = _copy_name_list(self.depends_on, "depends_on")
        if self.name in self.depends_on:
            raise ValueError(f"depends_on cannot include this component {self.name!r}")
        self.execution_order = _as_non_negative_int(self.execution_order, "execution_order")
        self.device_preference = _coerce_component_device(self.device_preference)
        self.resident_policy = _coerce_component_resident(self.resident_policy)
        self.phase = _coerce_component_phase(self.phase)

    def validate(self) -> None:
        """
        Reject an invalid layer id, a missing file, a missing dependency,
        or an input dtype that disagrees with the output dtype.
        """
        if self.component_type is ComponentType.LAYER:
            if (
                isinstance(self.layer_id, bool)
                or not isinstance(self.layer_id, int)
                or self.layer_id < 0
            ):
                raise ValueError(f"invalid layer id {self.layer_id!r}")
        elif (
            self.layer_id is not None
            and self.component_type not in _LAYER_SCOPED_COMPONENTS
        ):
            raise ValueError(
                f"invalid layer id {self.layer_id!r} for {self.component_type.value}"
            )
        if self.file_path is not None and not os.path.isfile(self.file_path):
            raise ValueError(f"missing file {self.file_path}")
        for path in self.external_data_files:
            if not os.path.isfile(path):
                raise ValueError(f"missing file {path}")
        if self.name in self.depends_on:
            raise ValueError(f"depends_on cannot include this component {self.name!r}")
        canonical: Dict[str, str] = {}
        for mapping in (self.input_types, self.output_types):
            if not isinstance(mapping, dict):
                raise ValueError(f"dtype map must be a dict, got {mapping!r}")
            for name, dtype in mapping.items():
                stored = _canonical_dtype_name(dtype)
                previous = canonical.get(name)
                if previous is not None and previous != stored:
                    raise ValueError(
                        f"inconsistent dtype for {name!r}: {previous} and {stored}"
                    )
                canonical[name] = stored

    @property
    def name(self) -> str:
        """Generate a canonical name for this component."""
        if self.component_type is ComponentType.LAYER or (
            self.component_type in _LAYER_SCOPED_COMPONENTS and self.layer_id is not None
        ):
            return f"{self.component_type.value}_{self.layer_id}"
        return self.component_type.value

    def to_dict(self) -> Dict[str, Any]:
        """Copy the execution contract. Callers can store this in model info."""
        return {
            "name": self.name,
            "component_type": self.component_type.value,
            "layer_id": self.layer_id,
            "file_path": self.file_path,
            "num_tensors": self.num_tensors,
            "size_bytes": self.size_bytes,
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "input_types": dict(self.input_types),
            "output_types": dict(self.output_types),
            "input_shapes": {key: list(shape) for key, shape in self.input_shapes.items()},
            "output_shapes": {key: list(shape) for key, shape in self.output_shapes.items()},
            "external_data_files": list(self.external_data_files),
            "depends_on": list(self.depends_on),
            "execution_order": self.execution_order,
            "device_preference": self.device_preference,
            "resident_policy": self.resident_policy,
            "phase": self.phase,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComponentInfo":
        """Build a component from component_manifest.json. The derived name is ignored."""
        return _from_record(cls, data, "component", drop=("name",))


@dataclass
class MergeResult:
    """
    Result of a model merging operation.
    
    Attributes:
        success: Whether the merge completed successfully.
        output_path: Path to the merged model file (if applicable).
        num_components: Number of components that were merged.
        total_tensors: Total number of tensors in the merged model.
        error: Error message if merge failed.
    """
    success: bool
    output_path: Optional[str] = None
    num_components: int = 0
    total_tensors: int = 0
    error: Optional[str] = None
