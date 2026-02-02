"""
Chunking Utils - Modular Model Splitting for ONNX and SafeTensors

A unified package for splitting large language models into per-component
files for efficient deployment, debugging, and modular inference.

Supported Formats:
    - SafeTensors (.safetensors) - HuggingFace model format
    - ONNX (.onnx) - Open Neural Network Exchange format

Supported Architectures:
    - LLaMA, Mistral, Phi, Gemma (LLaMA-style)
    - Qwen, Qwen2, Qwen3
    - GPT-2

Basic Usage:
    >>> from chunking_utils import split_safetensors, split_onnx
    
    # Split a HuggingFace model
    >>> result = split_safetensors(
    ...     model="Qwen/Qwen2.5-0.5B",
    ...     output="./output"
    ... )
    >>> print(f"Split into {result.num_layers} layers")
    
    # Split an ONNX model
    >>> result = split_onnx(
    ...     model="./model.onnx",
    ...     output="./output"
    ... )

Advanced Usage:
    >>> from chunking_utils import SplitConfig, split_safetensors
    
    >>> config = SplitConfig(
    ...     model="Qwen/Qwen2.5-0.5B",
    ...     output="./output",
    ...     split_only=True,  # Skip inference validation
    ... )
    >>> result = split_safetensors(config=config)

Merging:
    >>> from chunking_utils import merge_safetensors, merge_onnx
    
    # Merge SafeTensors back to state dict
    >>> state_dict = merge_safetensors("./output/Qwen--Qwen2.5-0.5B")
    
    # Merge ONNX back to single model
    >>> merged_model = merge_onnx("./output/model")
"""

# ==============================================================================
# Common Types and Utilities
# ==============================================================================

from .common.types import (
    ModelFormat,
    TensorDtype,
    MemoryMode,
    ComponentType,
    SplitConfig,
    SplitResult,
    MergeResult,
)

from .common.architecture import (
    ArchitectureType,
    ArchitecturePrefixes,
    get_prefixes,
    get_prefixes_tuple,
    architecture_from_string,
    SAFETENSORS_PREFIXES,
    ONNX_PREFIXES,
    LLAMA_STYLE_ARCHITECTURES,
)

from .common.path_utils import (
    is_local_path,
    sanitize_model_name,
)

# ==============================================================================
# SafeTensors Operations
# ==============================================================================

from .safetensors import (
    split_safetensors,
    merge_safetensors,
    load_state_dict,
    download_model,
    stream_tensors,
    get_safetensors_files,
    find_complete_config,
    ensure_complete_config,
    detect_architecture_from_weights,
    infer as infer_safetensors,
)

# ==============================================================================
# ONNX Operations
# ==============================================================================

from .onnx import (
    split_onnx,
    merge_onnx,
    load_onnx_model,
    resolve_model_path,
    extract_components,
    create_component_onnx,
    detect_architecture_from_onnx,
    ComponentInfo,
    OnnxSplitResult,
    run_inference as infer_onnx,
)

# ==============================================================================
# Convenience Functions
# ==============================================================================

def auto_detect_format(model_path: str) -> ModelFormat:
    """
    Auto-detect the model format from the path.
    
    Args:
        model_path: Path to model file or directory.
    
    Returns:
        Detected ModelFormat.
    """
    import os
    
    if model_path.endswith(".onnx"):
        return ModelFormat.ONNX
    
    if os.path.isdir(model_path):
        # Check for ONNX files first
        for f in os.listdir(model_path):
            if f.endswith(".onnx"):
                return ModelFormat.ONNX
        # Default to SafeTensors for directories
        return ModelFormat.SAFETENSORS
    
    # Check extension
    if model_path.endswith(".safetensors"):
        return ModelFormat.SAFETENSORS
    if model_path.endswith(".bin"):
        return ModelFormat.PYTORCH
    
    # Default to SafeTensors (works for HuggingFace model IDs)
    return ModelFormat.SAFETENSORS


def split_model(
    model: str,
    output: str,
    format: ModelFormat = None,
    **kwargs
) -> SplitResult:
    """
    Split a model using the appropriate splitter based on format.
    
    Auto-detects format if not specified.
    
    Args:
        model: Path to model or HuggingFace model ID.
        output: Output directory path.
        format: Model format (auto-detected if None).
        **kwargs: Additional arguments passed to the splitter.
    
    Returns:
        SplitResult with split information.
    
    Example:
        >>> result = split_model("./model.onnx", "./output")
        >>> result = split_model("Qwen/Qwen2.5-0.5B", "./output")
    """
    if format is None:
        format = auto_detect_format(model)
    
    if format == ModelFormat.ONNX:
        return split_onnx(model=model, output=output, **kwargs)
    else:
        return split_safetensors(model=model, output=output, **kwargs)


# ==============================================================================
# Public API
# ==============================================================================

__all__ = [
    # Types
    "ModelFormat",
    "ComponentType",
    "SplitConfig",
    "SplitResult",
    "MergeResult",
    "ComponentInfo",
    "OnnxSplitResult",
    
    # Architecture
    "ArchitectureType",
    "ArchitecturePrefixes",
    "get_prefixes",
    "get_prefixes_tuple",
    "architecture_from_string",
    "SAFETENSORS_PREFIXES",
    "ONNX_PREFIXES",
    "LLAMA_STYLE_ARCHITECTURES",
    
    # Path utilities
    "is_local_path",
    "sanitize_model_name",
    
    # SafeTensors operations
    "split_safetensors",
    "merge_safetensors",
    "load_state_dict",
    "download_model",
    "find_complete_config",
    "ensure_complete_config",
    "detect_architecture_from_weights",
    "infer_safetensors",
    
    # ONNX operations
    "split_onnx",
    "merge_onnx",
    "load_onnx_model",
    "resolve_model_path",
    "extract_components",
    "create_component_onnx",
    "detect_architecture_from_onnx",
    "infer_onnx",
    
    # Convenience
    "auto_detect_format",
    "split_model",
]

__version__ = "1.0.0"
