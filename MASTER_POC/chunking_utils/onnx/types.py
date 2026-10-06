"""
ONNX-specific data types for model chunking operations.

This module defines data types specific to ONNX model splitting,
including ComponentInfo for tracking extracted model components.
"""

from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field

from ..common.types import SplitResult, ModelFormat


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
    """
    component_type: str  # 'embedding', 'layer', 'norm', 'lm_head', 'other'
    layer_id: Optional[int] = None  # For layer components
    node_indices: List[int] = field(default_factory=list)
    node_names: List[str] = field(default_factory=list)
    initializer_names: Set[str] = field(default_factory=set)
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    
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
