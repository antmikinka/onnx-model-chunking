#!/usr/bin/env python3
"""
pipeline_utils - Modular Utilities for ONNX Inference Pipelines

This package provides reusable components for ONNX model inference:
- dtypes: Comprehensive ONNX data type handling
- kv_cache: KV cache management for incremental decoding
- config_loader: Model configuration loading and validation
- session_manager: ONNX session lifecycle and persistence
- io_binding: Zero-copy inference with device-resident buffers
- npu_config: NPU configuration and management (AMD XDNA/Ryzen AI)

Usage:
    from pipeline_utils import KVCacheManager, safe_cast, load_model_config
    from pipeline_utils import SessionManager, SessionConfig
    from pipeline_utils import IOBindingManager, MemoryMode
    from pipeline_utils.dtypes import ONNX_DTYPE_MAP, get_numpy_dtype
    from pipeline_utils.npu_config import NPUConfig, NPUOffloadConfig, is_npu_available
"""

from .dtypes import (
    ONNX_DTYPE_MAP,
    SHORT_DTYPE_MAP,
    SPECIAL_TYPES,
    get_numpy_dtype,
    safe_cast,
    get_dtype_info,
    supports_bfloat16,
    get_bfloat16_dtype,
)

from .kv_cache import KVCacheManager

from .config_loader import (
    ModelConfig,
    load_model_config,
    get_dtype_from_config,
)

from .session_manager import (
    SessionManager,
    SessionConfig,
    SessionInfo,
    ExecutionPolicy,
    ExecutionStep,
    ExecutionPlan,
    ProviderProfile,
    TensorIO,
    IOMetadata,
    PreparedIO,
    StaticBufferAllocation,
    get_available_providers,
    provider_profile_for,
    primary_provider_name,
    filter_session_providers,
    get_provider_for_device,
    provider_is_npu,
    providers_target_npu,
    is_sequential_layer_split,
    validate_npu_execution_policy,
    SUPPORTED_PROVIDERS,
    PROVIDER_CATEGORIES,
)

from .io_binding import (
    IOBindingManager,
    HeterogeneousIOBindingManager,
    AsyncIOBindingManager,
    DeviceBuffer,
    WeightDoubleBuffer,
    SharedBuffer,
    MemoryMode,
    MemoryPlacement,
    SharedAllocationKind,
    StaticKVPool,
    StaticShapeError,
    get_supported_devices,
    get_onnx_element_type,
    create_ortvalue_from_numpy,
    ort_memory_candidates,
    allocate_os_shared_memory,
    NUMPY_TO_ONNX_ELEMENT_TYPE,
    VALID_ORTVALUE_DEVICES,
    DEFAULT_NPU_MAX_SEQ_LEN,
)

from .backend import (
    BackendType,
    DeviceType,
    ModelFormat,
    BackendConfig,
    InferenceBackend,
    OnnxRuntimeBackend,
    LlamaCppBackend,
    get_backend,
    detect_model_format,
    get_backend_for_model,
)

from .dtypes import GGUF_DTYPE_MAP, AWQ_DTYPE_MAP

# NPU configuration module (P1.4)
from .npu_config import (
    NPUConfig,
    NPUOffloadConfig,
    NPUCapabilities,
    NPUCompilationMode,
    NPUOffloadStrategy,
    is_npu_available,
    get_npu_capabilities,
    validate_npu_config,
    get_recommended_npu_config,
)


__version__ = "2.2.0"

__all__ = [
    # dtypes exports
    "ONNX_DTYPE_MAP",
    "SHORT_DTYPE_MAP",
    "SPECIAL_TYPES",
    "GGUF_DTYPE_MAP",
    "AWQ_DTYPE_MAP",
    "get_numpy_dtype",
    "safe_cast",
    "get_dtype_info",
    "supports_bfloat16",
    "get_bfloat16_dtype",
    # kv_cache exports
    "KVCacheManager",
    # config_loader exports
    "ModelConfig",
    "load_model_config",
    "get_dtype_from_config",
    # session_manager exports
    "SessionManager",
    "SessionConfig",
    "SessionInfo",
    "ExecutionPolicy",
    "ExecutionStep",
    "ExecutionPlan",
    "ProviderProfile",
    "TensorIO",
    "IOMetadata",
    "PreparedIO",
    "StaticBufferAllocation",
    "get_available_providers",
    "provider_profile_for",
    "primary_provider_name",
    "filter_session_providers",
    "get_provider_for_device",
    "provider_is_npu",
    "providers_target_npu",
    "is_sequential_layer_split",
    "validate_npu_execution_policy",
    "SUPPORTED_PROVIDERS",
    "PROVIDER_CATEGORIES",
    # io_binding exports
    "IOBindingManager",
    "HeterogeneousIOBindingManager",
    "AsyncIOBindingManager",
    "DeviceBuffer",
    "WeightDoubleBuffer",
    "SharedBuffer",
    "MemoryMode",
    "MemoryPlacement",
    "SharedAllocationKind",
    "StaticKVPool",
    "StaticShapeError",
    "get_supported_devices",
    "get_onnx_element_type",
    "create_ortvalue_from_numpy",
    "ort_memory_candidates",
    "allocate_os_shared_memory",
    "NUMPY_TO_ONNX_ELEMENT_TYPE",
    "VALID_ORTVALUE_DEVICES",
    "DEFAULT_NPU_MAX_SEQ_LEN",
    # backend exports
    "BackendType",
    "DeviceType",
    "ModelFormat",
    "BackendConfig",
    "InferenceBackend",
    "OnnxRuntimeBackend",
    "LlamaCppBackend",
    "get_backend",
    "detect_model_format",
    "get_backend_for_model",
    # npu_config exports (P1.4)
    "NPUConfig",
    "NPUOffloadConfig",
    "NPUCapabilities",
    "NPUCompilationMode",
    "NPUOffloadStrategy",
    "is_npu_available",
    "get_npu_capabilities",
    "validate_npu_config",
    "get_recommended_npu_config",
]
