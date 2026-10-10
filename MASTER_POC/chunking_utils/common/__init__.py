"""
Common utilities for chunking_utils module.

This module contains shared utilities used across both ONNX and SafeTensors formats.
"""

from .path_utils import is_local_path, sanitize_model_name
from .path_manager import (
    find_onnx_model,
    find_external_data,
    find_external_data_files,
    validate_split_model,
    validate_split_model_dir,
    get_cache_dir,
    get_npu_cache_dir,
    resolve_npu_artifact_cache,
)
from .architecture import (
    ArchitectureType,
    ArchitecturePrefixes,
    SAFETENSORS_PREFIXES,
    ONNX_PREFIXES,
    get_prefixes,
    LLAMA_STYLE_ARCHITECTURES,
    architecture_from_string,
    EXACT_MAP,
    ArchitectureDescriptor,
    ARCHITECTURE_DESCRIPTORS,
    get_architecture_descriptor,
    QUANTIZED_OP_PATTERNS,
    quantized_op_scheme,
)
from .architecture_detector import detect_architecture
from .graph_topology import (
    GraphTopology,
    NodeGroup,
    analyze_graph_topology,
)
from .types import (
    ModelFormat,
    TensorDtype,
    MemoryMode,
    ComponentType,
    DeviceType,
    ExecutionPolicy,
    PROVIDER_FALLBACK,
    SplitConfig,
    QuantizationConfig,
    QuantizationInfo,
    KVCachePolicy,
    KVCacheConfig,
    KVCacheContract,
    ExecutionStep,
    ExecutionPlan,
    SplitResult,
    BoundaryTensor,
    ComponentInfo,
    MergeResult,
)

_HF_EXPORTS = (
    "TORCH_DTYPE_MAP",
    "get_torch_dtype",
    "ONNX_DTYPE_MAP",
    "get_onnx_dtype",
    "detect_architecture_hf",
    "get_model_config",
    "get_split_model_config",
    "estimate_model_memory",
    "estimate_kv_cache_memory",
    "recommend_execution_strategy",
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
    "find_onnx_model",
    "find_external_data",
    "find_external_data_files",
    "validate_split_model",
    "validate_split_model_dir",
    "get_cache_dir",
    "get_npu_cache_dir",
    "resolve_npu_artifact_cache",
    # Architecture
    "ArchitectureType",
    "ArchitecturePrefixes",
    "SAFETENSORS_PREFIXES",
    "ONNX_PREFIXES",
    "get_prefixes",
    "LLAMA_STYLE_ARCHITECTURES",
    "architecture_from_string",
    "EXACT_MAP",
    "ArchitectureDescriptor",
    "ARCHITECTURE_DESCRIPTORS",
    "get_architecture_descriptor",
    "QUANTIZED_OP_PATTERNS",
    "quantized_op_scheme",
    "detect_architecture",
    "GraphTopology",
    "NodeGroup",
    "analyze_graph_topology",
    # Types
    "ModelFormat",
    "TensorDtype",
    "MemoryMode",
    "ComponentType",
    "DeviceType",
    "ExecutionPolicy",
    "PROVIDER_FALLBACK",
    "SplitConfig",
    "QuantizationConfig",
    "QuantizationInfo",
    "KVCachePolicy",
    "KVCacheConfig",
    "KVCacheContract",
    "ExecutionStep",
    "ExecutionPlan",
    "SplitResult",
    "BoundaryTensor",
    "ComponentInfo",
    "MergeResult",
    # HuggingFace integration
    "TORCH_DTYPE_MAP",
    "get_torch_dtype",
    "ONNX_DTYPE_MAP",
    "get_onnx_dtype",
    "detect_architecture_hf",
    "get_model_config",
    "get_split_model_config",
    "estimate_model_memory",
    "estimate_kv_cache_memory",
    "recommend_execution_strategy",
    "check_dependencies",
]

