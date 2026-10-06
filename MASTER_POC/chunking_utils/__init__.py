"""
Chunking Utils - Modular Model Splitting for ONNX and SafeTensors

A unified package for splitting large language models into per-component
files for efficient deployment, debugging, and modular inference.

Supported Formats:
    - ONNX (.onnx)
    - SafeTensors (.safetensors)
    - GGUF (.gguf)
    - PyTorch .bin (pytorch_bin)

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

import importlib

# ==============================================================================
# Common Types and Utilities
# ==============================================================================

from .common.types import (
    ModelFormat,
    TensorDtype,
    MemoryMode,
    ComponentType,
    DeviceType,
    ExecutionPolicy,
    PROVIDER_FALLBACK,
    SplitConfig,
    QuantizationConfig,
    KVCachePolicy,
    KVCacheConfig,
    ExecutionStep,
    ExecutionPlan,
    SplitResult,
    BoundaryTensor,
    MergeResult,
)

from .common.architecture import (
    ArchitectureType,
    ArchitecturePrefixes,
    get_prefixes,
    get_prefixes_tuple,
    architecture_from_string,
    EXACT_MAP,
    ArchitectureDescriptor,
    ARCHITECTURE_DESCRIPTORS,
    get_architecture_descriptor,
    QUANTIZED_OP_PATTERNS,
    quantized_op_scheme,
    SAFETENSORS_PREFIXES,
    ONNX_PREFIXES,
    LLAMA_STYLE_ARCHITECTURES,
)

from .common.path_utils import (
    is_local_path,
    sanitize_model_name,
)
from .common.path_manager import (
    find_onnx_model,
    find_external_data,
    find_external_data_files,
    validate_split_model,
    validate_split_model_dir,
    get_cache_dir,
    get_npu_cache_dir,
    resolve_npu_artifact_cache,
)

# SafeTensors and ONNX splitters import torch and onnx. They load on first use
# so ModelFormat can be imported without those packages.
_LAZY_EXPORTS = {
    "split_safetensors": (".safetensors", "split_safetensors"),
    "merge_safetensors": (".safetensors", "merge_safetensors"),
    "load_state_dict": (".safetensors", "load_state_dict"),
    "download_model": (".safetensors", "download_model"),
    "stream_tensors": (".safetensors", "stream_tensors"),
    "get_safetensors_files": (".safetensors", "get_safetensors_files"),
    "find_complete_config": (".safetensors", "find_complete_config"),
    "ensure_complete_config": (".safetensors", "ensure_complete_config"),
    "detect_architecture_from_weights": (".safetensors", "detect_architecture_from_weights"),
    "infer_safetensors": (".safetensors", "infer"),
    "split_onnx": (".onnx", "split_onnx"),
    "merge_onnx": (".onnx", "merge_onnx"),
    "load_onnx_model": (".onnx", "load_onnx_model"),
    "resolve_model_path": (".onnx", "resolve_model_path"),
    "extract_components": (".onnx", "extract_components"),
    "create_component_onnx": (".onnx", "create_component_onnx"),
    "detect_architecture_from_onnx": (".onnx", "detect_architecture_from_onnx"),
    "ComponentInfo": (".onnx", "ComponentInfo"),
    "OnnxSplitResult": (".onnx", "OnnxSplitResult"),
    "infer_onnx": (".onnx", "run_inference"),
}


def __getattr__(name):
    """Import a splitter symbol the first time a caller asks for it."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    module = importlib.import_module(module_name, __package__)
    value = getattr(module, attr)
    globals()[name] = value
    return value


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
    if model_path.endswith(".gguf"):
        return ModelFormat.GGUF
    if model_path.endswith(".safetensors"):
        return ModelFormat.SAFETENSORS
    if model_path.endswith(".bin"):
        return ModelFormat.PYTORCH_BIN
    
    if os.path.isdir(model_path):
        names = os.listdir(model_path)
        if any(name.endswith(".onnx") for name in names):
            return ModelFormat.ONNX
        if any(name.endswith(".gguf") for name in names):
            return ModelFormat.GGUF
        return ModelFormat.SAFETENSORS
    
    # HuggingFace model IDs have no local extension.
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
        return __getattr__("split_onnx")(model=model, output=output, **kwargs)
    if format in (ModelFormat.SAFETENSORS, ModelFormat.PYTORCH_BIN):
        return __getattr__("split_safetensors")(model=model, output=output, **kwargs)
    raise ValueError(
        f"No splitter for format {format.value}. Supported split formats are onnx and safetensors."
    )


# ==============================================================================
# Public API
# ==============================================================================

__all__ = [
    # Types
    "ModelFormat",
    "ComponentType",
    "DeviceType",
    "ExecutionPolicy",
    "PROVIDER_FALLBACK",
    "SplitConfig",
    "QuantizationConfig",
    "KVCachePolicy",
    "KVCacheConfig",
    "ExecutionStep",
    "ExecutionPlan",
    "SplitResult",
    "BoundaryTensor",
    "MergeResult",
    "ComponentInfo",
    "OnnxSplitResult",
    
    # Architecture
    "ArchitectureType",
    "ArchitecturePrefixes",
    "get_prefixes",
    "get_prefixes_tuple",
    "architecture_from_string",
    "EXACT_MAP",
    "ArchitectureDescriptor",
    "ARCHITECTURE_DESCRIPTORS",
    "get_architecture_descriptor",
    "QUANTIZED_OP_PATTERNS",
    "quantized_op_scheme",
    "SAFETENSORS_PREFIXES",
    "ONNX_PREFIXES",
    "LLAMA_STYLE_ARCHITECTURES",
    
    # Path utilities
    "is_local_path",
    "sanitize_model_name",
    "find_onnx_model",
    "find_external_data",
    "find_external_data_files",
    "validate_split_model",
    "validate_split_model_dir",
    "get_cache_dir",
    "get_npu_cache_dir",
    "resolve_npu_artifact_cache",
    
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
