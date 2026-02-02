"""
Shared data types for model chunking operations.

This module defines the core data types used across both ONNX and SafeTensors
splitting operations, providing a unified interface for configuration and results.
"""

from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


class ModelFormat(str, Enum):
    """
    Supported model formats for chunking operations.
    """
    SAFETENSORS = "safetensors"
    ONNX = "onnx"
    PYTORCH = "pytorch"  # .bin files


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
        device: Target device ('cpu', 'cuda', 'cuda:0', etc.).
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


@dataclass
class ComponentInfo:
    """
    Information about a semantic model component.
    
    This is a base class that can be extended by format-specific implementations.
    Used to track metadata about extracted model components during splitting.
    
    Attributes:
        component_type: Type of component (embedding, layer, norm, lm_head, other).
        layer_id: For layer components, the layer index (0-indexed).
        file_path: Path to the saved component file.
        num_tensors: Number of tensors/weights in this component.
        size_bytes: Approximate size in bytes.
    """
    component_type: ComponentType
    layer_id: Optional[int] = None
    file_path: Optional[str] = None
    num_tensors: int = 0
    size_bytes: int = 0
    
    @property
    def name(self) -> str:
        """Generate a canonical name for this component."""
        if self.component_type == ComponentType.LAYER:
            return f"layer_{self.layer_id}"
        return self.component_type.value


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
