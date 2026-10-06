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
    _copy_dim_list,
    normalize_cache_role,
)


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
    """
    ir_version: int = 0
    producer: str = ""
    components: Dict[str, ComponentInfo] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate boundary tensors, then keep this result on ONNX."""
        super().__post_init__()
        self.format = ModelFormat.ONNX


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
