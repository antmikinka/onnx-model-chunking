"""
Shared data types for model chunking operations.

This module defines the core data types used across both ONNX and SafeTensors
splitting operations, providing a unified interface for configuration and results.
"""

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
    """
    EMBEDDING = "embedding"
    LAYER = "layer"
    NORM = "norm"
    LM_HEAD = "lm_head"
    OTHER = "other"


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
    """
    output_dir: str
    num_layers: int
    component_files: Dict[str, str]
    model_info: Dict[str, Any]
    format: ModelFormat = ModelFormat.SAFETENSORS
    boundary_tensors: List[BoundaryTensor] = field(default_factory=list)

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

    @property
    def name(self) -> str:
        """Generate a canonical name for this component."""
        if self.component_type == ComponentType.LAYER:
            return f"layer_{self.layer_id}"
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
