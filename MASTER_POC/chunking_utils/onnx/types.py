"""
ONNX-specific data types for model chunking operations.

This module defines data types specific to ONNX model splitting,
including ComponentInfo for tracking extracted model components.
"""

from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field

from ..common.types import (
    ModelFormat,
    SplitResult,
    _boundary_name_is_stateful,
    _coerce_component_device,
    _coerce_execution_policy,
    _coerce_group_size,
    _coerce_kv_layout,
    _coerce_optional_provider,
    _coerce_plan_device,
    _coerce_quant_dtype,
    _coerce_quant_method,
    _copy_dim_list,
    normalize_cache_role,
)

# Bit width implied by a canonical weight dtype. auto has no implied width.
_WEIGHT_DTYPE_BITS = {
    "float32": 32,
    "float16": 16,
    "bfloat16": 16,
    "int8": 8,
    "int4": 4,
}
_OPEN_WEIGHT_BITS = frozenset({2, 3, 4, 8, 16, 32})


@dataclass
class BoundaryTensorInfo:
    """
    One tensor that enters or leaves an ONNX component.

    ``shape`` uses None for a symbolic axis. A past or present name is
    stateful, carries a KV cache role, and prefers the NPU when no device
    is set. Which component consumes or produces the tensor is the list
    that holds this record.
    """
    name: str
    dtype: str
    shape: List[Optional[int]]
    is_stateful: bool = False
    cache_role: Optional[str] = None  # "past_key", "past_value", "present_key", etc.
    device_preference: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(
                f"boundary tensor name must be a non-empty string, got {self.name!r}"
            )
        self.name = self.name.strip()
        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise ValueError(
                f"boundary tensor dtype must be a non-empty string, got {self.dtype!r}"
            )
        self.dtype = self.dtype.strip()
        self.shape = _copy_dim_list(self.shape, f"shape of {self.name!r}")
        if not isinstance(self.is_stateful, bool):
            raise ValueError(f"is_stateful must be a bool, got {self.is_stateful!r}")
        if _boundary_name_is_stateful(self.name):
            self.is_stateful = True
        self.cache_role = normalize_cache_role(self.name, self.cache_role)
        self.device_preference = _coerce_component_device(self.device_preference)
        if self.device_preference is None and self.is_stateful:
            self.device_preference = "npu"

    def to_dict(self) -> Dict[str, Any]:
        """Copy the record so a caller can serialize it without sharing the shape."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "is_stateful": self.is_stateful,
            "cache_role": self.cache_role,
            "device_preference": self.device_preference,
        }


def _coerce_boundary_tensor_info(value: Any) -> BoundaryTensorInfo:
    """Accept a BoundaryTensorInfo or a mapping with the same fields."""
    if isinstance(value, BoundaryTensorInfo):
        return value
    if isinstance(value, dict):
        try:
            return BoundaryTensorInfo(**value)
        except TypeError as exc:
            raise ValueError(
                f"boundary tensor fields are incomplete, got {value!r}"
            ) from exc
    raise ValueError(f"boundary tensor must be a BoundaryTensorInfo, got {value!r}")


def _copy_boundary_list(value: Any, field_name: str) -> List[BoundaryTensorInfo]:
    """Copy one boundary list. A string is not a list. Duplicate names raise."""
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list, got {value!r}")
    cleaned: List[BoundaryTensorInfo] = []
    seen = set()
    for item in value:
        tensor = _coerce_boundary_tensor_info(item)
        if tensor.name in seen:
            raise ValueError(f"duplicate {field_name} name {tensor.name!r}")
        seen.add(tensor.name)
        cleaned.append(
            BoundaryTensorInfo(
                name=tensor.name,
                dtype=tensor.dtype,
                shape=list(tensor.shape),
                is_stateful=tensor.is_stateful,
                cache_role=tensor.cache_role,
                device_preference=tensor.device_preference,
            )
        )
    return cleaned


@dataclass
class ComponentInfo:
    """
    Information about a semantic ONNX model component.
    
    Tracks metadata about extracted model components during the splitting
    process, including node indices, initializer names, and I/O boundaries.
    
    Attributes:
        component_type: Type of component ('embedding', 'layer', 'norm', 'lm_head', 'other').
        layer_id: For layer components, the layer index (0-indexed).
        node_indices: Indices of nodes belonging to this component in the original graph.
        node_names: Names of nodes belonging to this component.
        initializer_names: Names of initializers (weights) used by this component.
        input_names: Input tensor names for this component.
        output_names: Output tensor names for this component.
        boundary_inputs: Tensors this component consumes, with dtype, shape, and device.
        boundary_outputs: Tensors this component produces, with dtype, shape, and device.
    """
    component_type: str  # 'embedding', 'layer', 'norm', 'lm_head', 'other'
    layer_id: Optional[int] = None  # For layer components
    node_indices: List[int] = field(default_factory=list)
    node_names: List[str] = field(default_factory=list)
    initializer_names: Set[str] = field(default_factory=set)
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    boundary_inputs: List[BoundaryTensorInfo] = field(default_factory=list)
    boundary_outputs: List[BoundaryTensorInfo] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Copy boundary lists so two components do not share one factory list."""
        self.boundary_inputs = _copy_boundary_list(self.boundary_inputs, "boundary_inputs")
        self.boundary_outputs = _copy_boundary_list(self.boundary_outputs, "boundary_outputs")

    @property
    def name(self) -> str:
        """Generate a canonical name for this component."""
        if self.component_type == "layer":
            return f"layer_{self.layer_id}"
        return self.component_type
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes in this component."""
        return len(self.node_indices)
    
    @property
    def num_initializers(self) -> int:
        """Number of initializers in this component."""
        return len(self.initializer_names)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.component_type,
            "layer_id": self.layer_id,
            "num_nodes": self.num_nodes,
            "num_initializers": self.num_initializers,
            "input_names": self.input_names,
            "output_names": self.output_names,
            "boundary_inputs": [item.to_dict() for item in self.boundary_inputs],
            "boundary_outputs": [item.to_dict() for item in self.boundary_outputs],
        }


def _coerce_info_dtype(value: Any, field_name: str) -> str:
    """Store a canonical dtype name. None is not a dtype."""
    if value is None:
        raise ValueError(f"{field_name} must be a string, got {value!r}")
    return _coerce_quant_dtype(value, field_name).value


def _coerce_weight_bits(value: Any, weight_dtype: str) -> int:
    """A bool is not a bit count. The count must match the weight dtype."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"weight_bits must be an integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"weight_bits must be positive, got {value}")
    expected = _WEIGHT_DTYPE_BITS.get(weight_dtype)
    if expected is not None and value != expected:
        raise ValueError(
            f"weight_bits {value} does not match weight_dtype {weight_dtype!r}"
        )
    if expected is None and value not in _OPEN_WEIGHT_BITS:
        known = ", ".join(str(item) for item in sorted(_OPEN_WEIGHT_BITS))
        raise ValueError(f"weight_bits must be one of {known}, got {value}")
    return value


@dataclass
class QuantizationInfo:
    """
    Quantization contract for one ONNX split.

    ``method`` accepts MatMulNBits, AWQ, and GPTQ. ``weight_dtype`` and
    ``activation_dtype`` are canonical names such as float16 or int4.
    ``group_size`` -1 is per-channel. ``weight_bits`` matches the weight dtype.
    """
    method: Optional[str] = None  # "MatMulNBits", "AWQ", "GPTQ"
    weight_bits: int = 16
    weight_dtype: str = "float16"
    activation_dtype: str = "float16"
    group_size: Optional[int] = None
    has_zero_point: bool = False

    def __post_init__(self) -> None:
        self.method = _coerce_quant_method(self.method)
        self.weight_dtype = _coerce_info_dtype(self.weight_dtype, "weight_dtype")
        self.activation_dtype = _coerce_info_dtype(
            self.activation_dtype, "activation_dtype"
        )
        self.weight_bits = _coerce_weight_bits(self.weight_bits, self.weight_dtype)
        self.group_size = _coerce_group_size(self.group_size)
        if not isinstance(self.has_zero_point, bool):
            raise ValueError(
                f"has_zero_point must be a boolean, got {self.has_zero_point!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Copy the record so a caller can serialize it without sharing this object."""
        return {
            "method": self.method,
            "weight_bits": self.weight_bits,
            "weight_dtype": self.weight_dtype,
            "activation_dtype": self.activation_dtype,
            "group_size": self.group_size,
            "has_zero_point": self.has_zero_point,
        }


def _coerce_positive_int(value: Any, field_name: str) -> int:
    """A bool is not a count. Zero is not a cache dimension."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{field_name} must be positive, got {value}")
    return value


def _coerce_kv_pattern(value: Any, field_name: str, family: str, role: str) -> str:
    """A past or present pattern names one layer with {layer}, then key or value."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")
    text = value.strip()
    if text.count("{layer}") != 1:
        raise ValueError(
            f"{field_name} must name the layer once with {{layer}}, got {text!r}"
        )
    lowered = text.lower()
    prefixes = (
        ("past_key_values.", "past.")
        if family == "past"
        else ("present_key_values.", "present.")
    )
    if not lowered.startswith(prefixes):
        expected = " or ".join(prefixes)
        raise ValueError(f"{field_name} must start with {expected}, got {text!r}")
    tail = lowered.split("{layer}", 1)[1]
    if role == "key":
        if ".value" in tail or ".key" not in tail:
            raise ValueError(f"{field_name} must name a key, got {text!r}")
    elif ".key" in tail or ".value" not in tail:
        raise ValueError(f"{field_name} must name a value, got {text!r}")
    return text


@dataclass
class KVCacheContract:
    """
    Past and present KV tensors for one ONNX split.

    Layout BNSD is batch, heads, sequence, head dim. Each pattern contains
    ``{layer}`` once. Past names start with ``past_key_values.`` and present
    names start with ``present.``.
    """
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    layout: str  # "BNSD"
    max_seq_len: int
    past_key_pattern: str
    past_value_pattern: str
    present_key_pattern: str
    present_value_pattern: str

    def __post_init__(self) -> None:
        self.num_layers = _coerce_positive_int(self.num_layers, "num_layers")
        self.num_kv_heads = _coerce_positive_int(self.num_kv_heads, "num_kv_heads")
        self.head_dim = _coerce_positive_int(self.head_dim, "head_dim")
        self.dtype = _coerce_info_dtype(self.dtype, "dtype")
        self.layout = _coerce_kv_layout(self.layout)
        self.max_seq_len = _coerce_positive_int(self.max_seq_len, "max_seq_len")
        self.past_key_pattern = _coerce_kv_pattern(
            self.past_key_pattern, "past_key_pattern", "past", "key"
        )
        self.past_value_pattern = _coerce_kv_pattern(
            self.past_value_pattern, "past_value_pattern", "past", "value"
        )
        self.present_key_pattern = _coerce_kv_pattern(
            self.present_key_pattern, "present_key_pattern", "present", "key"
        )
        self.present_value_pattern = _coerce_kv_pattern(
            self.present_value_pattern, "present_value_pattern", "present", "value"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Copy the contract so a caller can serialize it without sharing this object."""
        return {
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "dtype": self.dtype,
            "layout": self.layout,
            "max_seq_len": self.max_seq_len,
            "past_key_pattern": self.past_key_pattern,
            "past_value_pattern": self.past_value_pattern,
            "present_key_pattern": self.present_key_pattern,
            "present_value_pattern": self.present_value_pattern,
        }


def _coerce_kv_cache_contract(value: Any) -> Optional["KVCacheContract"]:
    """None stays unset. A KVCacheContract is copied. A dict builds one."""
    if value is None:
        return None
    if isinstance(value, KVCacheContract):
        return KVCacheContract(
            num_layers=value.num_layers,
            num_kv_heads=value.num_kv_heads,
            head_dim=value.head_dim,
            dtype=value.dtype,
            layout=value.layout,
            max_seq_len=value.max_seq_len,
            past_key_pattern=value.past_key_pattern,
            past_value_pattern=value.past_value_pattern,
            present_key_pattern=value.present_key_pattern,
            present_value_pattern=value.present_value_pattern,
        )
    if isinstance(value, dict):
        try:
            return KVCacheContract(**value)
        except TypeError as exc:
            raise ValueError(f"kv cache fields are incomplete, got {value!r}") from exc
    raise ValueError(f"kv_cache must be a KVCacheContract or a dict, got {value!r}")


def _coerce_quantization_info(value: Any) -> Optional[QuantizationInfo]:
    """None stays unset. A QuantizationInfo is copied. A dict builds one."""
    if value is None:
        return None
    if isinstance(value, QuantizationInfo):
        return QuantizationInfo(
            method=value.method,
            weight_bits=value.weight_bits,
            weight_dtype=value.weight_dtype,
            activation_dtype=value.activation_dtype,
            group_size=value.group_size,
            has_zero_point=value.has_zero_point,
        )
    if isinstance(value, dict):
        try:
            return QuantizationInfo(**value)
        except TypeError as exc:
            raise ValueError(
                f"quantization fields are incomplete, got {value!r}"
            ) from exc
    raise ValueError(
        f"quantization must be a QuantizationInfo or a dict, got {value!r}"
    )


# Metadata generation written into model_info.json by OnnxSplitResult.
SCHEMA_VERSION = "2.0.0"


def _coerce_schema_version(value: Any) -> str:
    """A schema version is major.minor.patch and matches this metadata generation."""
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise ValueError(f"schema_version must be a non-empty string, got {value!r}")
    text = value.strip()
    parts = text.split(".")
    if (
        len(parts) != 3
        or any(
            not part.isdigit() or (len(part) > 1 and part.startswith("0"))
            for part in parts
        )
    ):
        raise ValueError(f"schema_version must be major.minor.patch, got {text!r}")
    if text != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version {text!r} does not match {SCHEMA_VERSION}"
        )
    return text


@dataclass
class OnnxSplitResult(SplitResult):
    """
    Result of an ONNX model splitting operation.
    
    Extends the base SplitResult with ONNX-specific information.
    
    Attributes:
        output_dir: Directory where split files were saved.
        num_layers: Number of transformer layers in the model.
        component_files: Mapping of component names to file paths.
        model_info: Additional model metadata and information.
        format: Always ModelFormat.ONNX for this class.
        ir_version: ONNX IR version of the original model.
        producer: Producer name from the original model.
        components: Dictionary of ComponentInfo objects.
        quantization: MatMulNBits, AWQ, or GPTQ contract for this split.
        kv_cache: Past and present key and value patterns for this split.
        schema_version: Metadata generation for model_info.json. The current
            generation is 2.0.0.
    """
    ir_version: int = 0
    producer: str = ""
    components: Dict[str, ComponentInfo] = field(default_factory=dict)
    quantization: Optional[QuantizationInfo] = None
    kv_cache: Optional[KVCacheContract] = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        """Validate boundary tensors, keep this result on ONNX, and copy contracts."""
        super().__post_init__()
        self.format = ModelFormat.ONNX
        self.quantization = _coerce_quantization_info(self.quantization)
        self.kv_cache = _coerce_kv_cache_contract(self.kv_cache)
        if self.kv_cache is not None and self.kv_cache.num_layers != self.num_layers:
            raise ValueError(
                f"kv_cache num_layers {self.kv_cache.num_layers} does not match "
                f"split num_layers {self.num_layers!r}"
            )
        self.schema_version = _coerce_schema_version(self.schema_version)

    def to_dict(self) -> Dict[str, Any]:
        """Copy the split result and include the quantization, KV cache, and schema version."""
        recorded = super().to_dict()
        recorded["quantization"] = (
            None if self.quantization is None else self.quantization.to_dict()
        )
        recorded["kv_cache"] = None if self.kv_cache is None else self.kv_cache.to_dict()
        recorded["schema_version"] = self.schema_version
        return recorded


@dataclass
class OnnxMergeResult:
    """
    Result of an ONNX model merging operation.
    
    Attributes:
        success: Whether the merge completed successfully.
        output_path: Path to the merged model file (if saved).
        num_nodes: Total number of nodes in the merged model.
        num_initializers: Total number of initializers in the merged model.
        error: Error message if merge failed.
        psnr: Peak signal-to-noise ratio in decibels when numerical
            verification ran. An exact match is infinity.
    """
    success: bool
    output_path: Optional[str] = None
    num_nodes: int = 0
    num_initializers: int = 0
    error: Optional[str] = None
    psnr: Optional[float] = None


# Phases an ONNX placement step may name. Router aliases collapse onto router.
_ONNX_STEP_PHASES = frozenset({"prefill", "decode", "router", "expert"})
_ONNX_STEP_PHASE_ALIASES = {
    "prefill": "prefill",
    "decode": "decode",
    "router": "router",
    "moe_router": "router",
    "moe": "router",
    "expert": "expert",
    "expert_execution": "expert",
    "experts": "expert",
}


def _coerce_onnx_step_phase(value: Any) -> str:
    """A placement step is prefill, decode, router, or expert."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"phase must be a non-empty string, got {value!r}")
    text = value.strip().lower().replace("-", "_").replace(" ", "_")
    text = _ONNX_STEP_PHASE_ALIASES.get(text, text)
    if text not in _ONNX_STEP_PHASES:
        allowed = ", ".join(sorted(_ONNX_STEP_PHASES))
        raise ValueError(f"Unknown phase {value!r}. Expected one of: {allowed}")
    return text


def _copy_execution_step(value: Any) -> "ExecutionStep":
    """Copy an ExecutionStep, or build one from a dict. Extra keys are an error."""
    if isinstance(value, ExecutionStep):
        return ExecutionStep(
            component_name=value.component_name,
            provider=value.provider,
            device=value.device,
            phase=value.phase,
            resident=value.resident,
        )
    if isinstance(value, dict):
        try:
            return ExecutionStep(**value)
        except TypeError as exc:
            raise ValueError(
                f"execution step fields are incomplete, got {value!r}"
            ) from exc
    raise ValueError(
        f"execution step must be an ExecutionStep or a dict, got {value!r}"
    )


@dataclass
class ExecutionStep:
    """
    Where one ONNX component runs.

    ``phase`` is prefill, decode, router, or expert, and defaults to decode.
    ``device`` uses the AMD names, so xdna is the NPU and rdna is the iGPU.
    ``resident`` keeps the component loaded.
    """
    component_name: str
    provider: Optional[str] = None
    device: Optional[str] = None
    phase: str = "decode"  # prefill, decode, router, expert
    resident: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.component_name, str) or not self.component_name.strip():
            raise ValueError(
                f"component_name must be a non-empty string, got {self.component_name!r}"
            )
        self.component_name = self.component_name.strip()
        self.provider = _coerce_optional_provider(self.provider)
        self.device = _coerce_plan_device(self.device, "device")
        self.phase = _coerce_onnx_step_phase(self.phase)
        if not isinstance(self.resident, bool):
            raise ValueError(f"resident must be a boolean, got {self.resident!r}")

    def to_dict(self) -> Dict[str, Any]:
        """Copy the step so a caller can serialize it without sharing this object."""
        return {
            "component_name": self.component_name,
            "provider": self.provider,
            "device": self.device,
            "phase": self.phase,
            "resident": self.resident,
        }


@dataclass
class ExecutionPlan:
    """
    How ONNX components run on AMD hardware.

    ``policy`` is a canonical name such as auto or hybrid_npu_igpu.
    ``npu_safe`` records whether the graph may run on the NPU. Prefill and
    decode can name different devices.
    """
    steps: List[ExecutionStep] = field(default_factory=list)
    policy: str = "auto"
    npu_safe: bool = False
    prefill_device: Optional[str] = None
    decode_device: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.steps, (str, bytes)) or not isinstance(self.steps, list):
            raise ValueError(f"steps must be a list, got {self.steps!r}")
        cleaned: List[ExecutionStep] = []
        seen = set()
        for item in self.steps:
            step = _copy_execution_step(item)
            if step.component_name in seen:
                raise ValueError(
                    f"execution plan repeats component {step.component_name!r}"
                )
            seen.add(step.component_name)
            cleaned.append(step)
        self.steps = cleaned
        self.policy = _coerce_execution_policy(self.policy).value
        if not isinstance(self.npu_safe, bool):
            raise ValueError(f"npu_safe must be a boolean, got {self.npu_safe!r}")
        self.prefill_device = _coerce_plan_device(self.prefill_device, "prefill_device")
        self.decode_device = _coerce_plan_device(self.decode_device, "decode_device")

    def to_dict(self) -> Dict[str, Any]:
        """Copy the plan. Each step is copied on its own."""
        return {
            "steps": [step.to_dict() for step in self.steps],
            "policy": self.policy,
            "npu_safe": self.npu_safe,
            "prefill_device": self.prefill_device,
            "decode_device": self.decode_device,
        }
