"""
Common utilities for chunking_utils module.

This module contains shared utilities used across both ONNX and SafeTensors formats.
"""

from .path_utils import is_local_path, sanitize_model_name
from .architecture import (
    ArchitectureType,
    ArchitecturePrefixes,
    SAFETENSORS_PREFIXES,
    ONNX_PREFIXES,
    get_prefixes,
    LLAMA_STYLE_ARCHITECTURES,
    architecture_from_string,
)
from .types import (
    ModelFormat,
    TensorDtype,
    MemoryMode,
    ComponentType,
    SplitConfig,
    SplitResult,
    MergeResult,
)

_HF_EXPORTS = (
    "TORCH_DTYPE_MAP",
    "get_torch_dtype",
    "detect_architecture_hf",
    "get_model_config",
    "estimate_model_memory",
    "check_dependencies",
)


def __getattr__(name):
    """Load HuggingFace helpers on first use so torch is not required to import types."""
    if name in _HF_EXPORTS:
        from . import hf_integration

        value = getattr(hf_integration, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Path utilities
    "is_local_path",
    "sanitize_model_name",
    # Architecture
    "ArchitectureType",
    "ArchitecturePrefixes",
    "SAFETENSORS_PREFIXES",
    "ONNX_PREFIXES",
    "get_prefixes",
    "LLAMA_STYLE_ARCHITECTURES",
    "architecture_from_string",
    # Types
    "ModelFormat",
    "TensorDtype",
    "MemoryMode",
    "ComponentType",
    "SplitConfig",
    "SplitResult",
    "MergeResult",
    # HuggingFace integration
    "TORCH_DTYPE_MAP",
    "get_torch_dtype",
    "detect_architecture_hf",
    "get_model_config",
    "estimate_model_memory",
    "check_dependencies",
]

