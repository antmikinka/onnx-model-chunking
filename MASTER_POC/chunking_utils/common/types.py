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
    """
    output_dir: str
    num_layers: int
    component_files: Dict[str, str]
    model_info: Dict[str, Any]
    format: ModelFormat = ModelFormat.SAFETENSORS


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
