#!/usr/bin/env python3
"""
session_manager.py - ONNX Session Lifecycle Management

This module provides session persistence and management for ONNX Runtime.
Key feature: Load all sessions once at startup and keep them resident,
avoiding the performance penalty of repeated load/unload cycles.

This is critical for NPU performance where weight locality in tile SRAM
is essential for optimal throughput.

Usage:
    from pipeline_utils.session_manager import SessionManager, SessionConfig
    
    config = SessionConfig(providers=['VitisAIExecutionProvider', 'CPUExecutionProvider'])
    manager = SessionManager(model_dir, config)
    manager.load_all()  # Load all sessions once
    
    session = manager.get_session("layer_0")  # Get pre-loaded session
"""

import os
import json
import re
import statistics
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Any, Tuple, Union
import numpy as np
import onnxruntime as ort

from chunking_utils.common.types import ExecutionPolicy


# =============================================================================
# PROVIDER CONSTANTS
# =============================================================================

# All supported ONNX Runtime execution providers
SUPPORTED_PROVIDERS = [
    'VitisAIExecutionProvider',    # AMD XDNA NPU (Ryzen AI 1.7+)
    'RyzenAIExecutionProvider',    # AMD Ryzen AI
    'CUDAExecutionProvider',       # NVIDIA GPU
    'ROCmExecutionProvider',       # AMD GPU
    'DmlExecutionProvider',        # DirectML (Windows NPU/GPU)
    'CoreMLExecutionProvider',     # Apple Neural Engine
    'TensorrtExecutionProvider',   # NVIDIA TensorRT
    'OpenVINOExecutionProvider',   # Intel NPU/GPU
    'MIGraphXExecutionProvider',   # AMD MIGraphX
    'CPUExecutionProvider',        # Universal fallback
]

# Provider categories for documentation.
# DirectML stays in the NPU category for device selection, but it is also the
# iGPU side of hybrid dispatch, so the monolithic guard does not treat it as
# an ahead-of-time NPU.
PROVIDER_CATEGORIES = {
    'gpu': ['CUDAExecutionProvider', 'ROCmExecutionProvider', 'TensorrtExecutionProvider', 'MIGraphXExecutionProvider'],
    'npu': ['RyzenAIExecutionProvider', 'VitisAIExecutionProvider', 'OpenVINOExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider'],
    'cpu': ['CPUExecutionProvider']
}

# Ahead-of-time accelerators. A layer-by-layer session on these providers
# recompiles or context-switches between graphs.
NPU_PROVIDER_MARKERS = (
    "vitisai",
    "ryzenai",
    "openvino",
    "coreml",
)

GPU_PROVIDER_MARKERS = (
    "cuda",
    "rocm",
    "tensorrt",
    "migraphx",
    "dml",
    "directml",
)

_LAYER_COMPONENT = re.compile(r"(^|_)layer_\d+$")

NPU_SEQUENTIAL_SPLIT_ERROR = (
    "Execution policy 'sequential' unloads each NPU session. "
    "Ahead-of-time compilation then runs again on the next load. "
    "Keep layer sessions resident and run them in order. "
    "A layer can use VitisAI while lm_head stays on DirectML and embeddings stay on CPU."
)


HYBRID_NPU_POLICIES = frozenset({
    ExecutionPolicy.HYBRID_NPU_IGPU,
    ExecutionPolicy.PHASE_SPLIT,
    ExecutionPolicy.MOE_ROUTING,
})


def _provider_name(provider: Any) -> str:
    """Return the provider name from a string or an (name, options) pair."""
    if isinstance(provider, (list, tuple)) and provider:
        provider = provider[0]
    return str(provider)


def provider_is_npu(provider: Any) -> bool:
    """True for VitisAI, RyzenAI, OpenVINO, and CoreML provider names."""
    name = _provider_name(provider).lower()
    return any(marker in name for marker in NPU_PROVIDER_MARKERS)


def provider_is_gpu(provider: Any) -> bool:
    """True for CUDA, ROCm, TensorRT, MIGraphX, and DirectML."""
    name = _provider_name(provider).lower()
    return any(marker in name for marker in GPU_PROVIDER_MARKERS)


def providers_target_npu(providers: Optional[Iterable[Any]]) -> bool:
    """True when any configured provider is an ahead-of-time NPU."""
    return any(provider_is_npu(provider) for provider in (providers or ()))


def is_sequential_layer_split(component_names: Optional[Iterable[str]]) -> bool:
    """
    True when the model is a layer-by-layer session series.

    Two or more components named ``layer_<index>`` (or ``*_layer_<index>``)
    are the sequential split MASTER_REFERENCE forbids on an NPU.
    """
    names = list(component_names or ())
    layer_count = sum(1 for name in names if _LAYER_COMPONENT.search(str(name)))
    return layer_count >= 2


def coerce_execution_policy(policy: Union[ExecutionPolicy, str]) -> ExecutionPolicy:
    """Accept an enum value or its string value."""
    if isinstance(policy, ExecutionPolicy):
        return policy
    try:
        return ExecutionPolicy(str(policy))
    except ValueError as exc:
        allowed = ", ".join(item.value for item in ExecutionPolicy)
        raise ValueError(
            f"Unknown execution policy '{policy}'. Expected one of: {allowed}"
        ) from exc


MEMORY_POLICIES = frozenset({
    "auto",
    "minimal",
    "persistent",
    "io_bound",
    "npu_static",
})


def _coerce_memory_policy(policy: Any) -> str:
    """Normalize a memory policy. ``resident`` is the persistent mode."""
    text = str(policy if policy is not None else "auto").strip().lower().replace("-", "_")
    if text == "resident":
        text = "persistent"
    if text not in MEMORY_POLICIES:
        allowed = ", ".join(sorted(MEMORY_POLICIES))
        raise ValueError(
            f"Unknown memory policy '{policy}'. Expected one of: {allowed}"
        )
    return text


def validate_npu_execution_policy(
    execution_policy: Union[ExecutionPolicy, str],
    providers: Optional[Sequence[Any]],
    component_names: Optional[Iterable[str]] = None,
) -> ExecutionPolicy:
    """
    Reject an NPU policy that unloads every session between steps.

    A resident layer series is allowed. Layers, embeddings, and lm_head may
    use different providers. Phase-split and MoE routing stay allowed. CPU
    and GPU splits stay allowed. A policy that does not match the selected
    device warns.
    """
    policy = coerce_execution_policy(execution_policy)
    provider_list = list(providers or ())
    names = list(component_names or ())
    targets_npu = providers_target_npu(provider_list)
    targets_gpu = any(provider_is_gpu(provider) for provider in provider_list)
    layer_split = is_sequential_layer_split(names)
    # SEQUENTIAL unloads the session after each step, so the next token
    # compiles the graph again. PERSISTENT keeps the compiled session.
    if targets_npu and policy == ExecutionPolicy.SEQUENTIAL:
        raise RuntimeError(NPU_SEQUENTIAL_SPLIT_ERROR)
    # Speculative draft and verify is a phase schedule, not a layer series.
    if targets_npu and layer_split and policy == ExecutionPolicy.SPECULATIVE:
        raise RuntimeError(NPU_SEQUENTIAL_SPLIT_ERROR)

    if policy in HYBRID_NPU_POLICIES and not targets_npu:
        warnings.warn(
            f"Execution policy '{policy.value}' is an NPU hybrid strategy, "
            "but the provider list has no VitisAI, RyzenAI, OpenVINO, or "
            "CoreML provider.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif policy == ExecutionPolicy.PERSISTENT and not targets_gpu and not targets_npu:
        warnings.warn(
            "Execution policy 'persistent' expects a GPU provider "
            "(CUDA, ROCm, TensorRT, MIGraphX, or DirectML).",
            RuntimeWarning,
            stacklevel=2,
        )
    elif (
        policy == ExecutionPolicy.SEQUENTIAL
        and targets_gpu
        and not targets_npu
    ):
        warnings.warn(
            "Execution policy 'sequential' unloads each "
            "layer. A GPU target should use 'persistent'.",
            RuntimeWarning,
            stacklevel=2,
        )
    return policy


# =============================================================================
# PROVIDER PROFILES
# =============================================================================

# Keys ONNX Runtime accepts on each execution provider. Anything else stays
# on the profile for session options and is not passed into InferenceSession.
_EP_OPTION_KEYS = {
    "cpu": frozenset(),
    "generic": frozenset(),
    "cuda": frozenset({
        "device_id",
        "arena_extend_strategy",
        "gpu_mem_limit",
        "cudnn_conv_algo_search",
        "do_copy_in_default_stream",
        "enable_cuda_graph",
        "cudnn_conv_use_max_workspace",
    }),
    "rocm": frozenset({
        "device_id",
        "arena_extend_strategy",
        "gpu_mem_limit",
        "miopen_conv_exhaustive_search",
        "do_copy_in_default_stream",
        "tunable_op_enable",
        "tunable_op_tuning_enable",
        "enable_hip_graph",
    }),
    "migraphx": frozenset({
        "device_id",
        "gpu_mem_limit",
        "arena_extend_strategy",
    }),
    "directml": frozenset({
        "device_id",
        "disable_metacommands",
    }),
    "vitisai": frozenset({
        "config_file",
        "cacheDir",
        "cacheKey",
        "target",
    }),
    "ryzenai": frozenset({
        "cacheDir",
        "cacheKey",
        "target",
    }),
    "tensorrt": frozenset({
        "device_id",
        "trt_engine_cache_enable",
        "trt_engine_cache_path",
        "trt_max_workspace_size",
    }),
    "openvino": frozenset({
        "device_type",
        "cache_dir",
        "num_of_threads",
    }),
    "coreml": frozenset({
        "ModelFormat",
        "MLComputeUnits",
        "RequireStaticInputShapes",
    }),
}

_STATIC_SHAPE_DEVICES = frozenset({"vitisai", "ryzenai", "openvino", "coreml"})

_NPU_PERFORMANCE_MODE = {
    "default": "balanced",
    "fast": "low_latency",
    "optimized": "high_performance",
}

_GPU_MEMORY_DEVICES = frozenset({"cuda", "rocm", "migraphx", "tensorrt"})

_npu_static_shape_warning_sent = False


def _warn_static_npu_shapes(provider_name: str) -> None:
    """Tell the caller once that an AOT NPU profile keeps shape checking on."""
    global _npu_static_shape_warning_sent
    if _npu_static_shape_warning_sent:
        return
    _npu_static_shape_warning_sent = True
    warnings.warn(
        f"{provider_name} keeps ONNX shape checking enabled. "
        "AOT NPU compilation requires static shapes.",
        RuntimeWarning,
        stacklevel=3,
    )


@dataclass
class ProviderProfile:
    """
    Session and execution-provider settings for one device.

    ``provider_options`` holds both session hints and EP keys. Only the keys
    in ``_EP_OPTION_KEYS`` are forwarded to ONNX Runtime.
    """

    name: str
    device_type: str
    optimization_level: str
    enable_graph_cache: bool
    enable_compile_cache: bool
    cache_dir: Optional[str]
    device_id: int
    memory_limit_mb: Optional[int]
    provider_options: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "ProviderProfile":
        """Return a detached copy so overrides do not change the template."""
        return ProviderProfile(
            name=self.name,
            device_type=self.device_type,
            optimization_level=self.optimization_level,
            enable_graph_cache=self.enable_graph_cache,
            enable_compile_cache=self.enable_compile_cache,
            cache_dir=self.cache_dir,
            device_id=self.device_id,
            memory_limit_mb=self.memory_limit_mb,
            provider_options=dict(self.provider_options),
        )


def _cpu_profile() -> ProviderProfile:
    threads = os.cpu_count() or 1
    return ProviderProfile(
        name="CPUExecutionProvider",
        device_type="cpu",
        optimization_level="DISABLE",
        enable_graph_cache=False,
        enable_compile_cache=False,
        cache_dir=None,
        device_id=0,
        memory_limit_mb=None,
        provider_options={
            "intra_op_num_threads": threads,
            "inter_op_num_threads": 1,
            "enable_cpu_mem_arena": True,
            "arena_extend_strategy": "kSameAsRequested",
            "allow_spinning": False,
        },
    )


def _cuda_profile() -> ProviderProfile:
    return ProviderProfile(
        name="CUDAExecutionProvider",
        device_type="cuda",
        optimization_level="EXTENDED",
        enable_graph_cache=False,
        enable_compile_cache=False,
        cache_dir=None,
        device_id=0,
        memory_limit_mb=None,
        provider_options={
            "device_id": "0",
            "arena_extend_strategy": "kSameAsRequested",
            "do_copy_in_default_stream": "1",
            "cudnn_conv_algo_search": "HEURISTIC",
            "cudnn_conv_use_max_workspace": "0",
            "enable_cuda_graph": "0",
        },
    )


def _rocm_profile() -> ProviderProfile:
    return ProviderProfile(
        name="ROCmExecutionProvider",
        device_type="rocm",
        optimization_level="EXTENDED",
        enable_graph_cache=False,
        enable_compile_cache=False,
        cache_dir=None,
        device_id=0,
        memory_limit_mb=None,
        provider_options={
            "device_id": "0",
            "arena_extend_strategy": "kSameAsRequested",
            "do_copy_in_default_stream": "1",
            "miopen_conv_exhaustive_search": "0",
            "tunable_op_enable": "1",
            "tunable_op_tuning_enable": "0",
            "enable_hip_graph": "0",
        },
    )


def _migraphx_profile() -> ProviderProfile:
    return ProviderProfile(
        name="MIGraphXExecutionProvider",
        device_type="migraphx",
        optimization_level="EXTENDED",
        enable_graph_cache=False,
        enable_compile_cache=False,
        cache_dir=None,
        device_id=0,
        memory_limit_mb=None,
        provider_options={
            "device_id": "0",
            "arena_extend_strategy": "kSameAsRequested",
        },
    )


def _directml_profile() -> ProviderProfile:
    return ProviderProfile(
        name="DmlExecutionProvider",
        device_type="directml",
        optimization_level="EXTENDED",
        enable_graph_cache=True,
        enable_compile_cache=False,
        cache_dir=None,
        device_id=0,
        memory_limit_mb=None,
        provider_options={
            "device_id": "0",
            # Metacommands are the Windows GPU/NPU fast path. Leaving them
            # enabled is the graph-cache behavior for DirectML.
            "disable_metacommands": "0",
        },
    )


def _npu_profile(name: str, device_type: str) -> ProviderProfile:
    return ProviderProfile(
        name=name,
        device_type=device_type,
        optimization_level="DISABLE",
        enable_graph_cache=True,
        enable_compile_cache=True,
        cache_dir=None,
        device_id=0,
        memory_limit_mb=None,
        provider_options={
            "target": "X1",
            "cacheKey": device_type,
            "performance_mode": "balanced",
        },
    )


def _tensorrt_profile() -> ProviderProfile:
    return ProviderProfile(
        name="TensorrtExecutionProvider",
        device_type="tensorrt",
        optimization_level="EXTENDED",
        enable_graph_cache=False,
        enable_compile_cache=True,
        cache_dir=None,
        device_id=0,
        memory_limit_mb=None,
        provider_options={
            "device_id": "0",
            "trt_engine_cache_enable": "1",
        },
    )


def _openvino_profile() -> ProviderProfile:
    return ProviderProfile(
        name="OpenVINOExecutionProvider",
        device_type="openvino",
        optimization_level="DISABLE",
        enable_graph_cache=True,
        enable_compile_cache=True,
        cache_dir=None,
        device_id=0,
        memory_limit_mb=None,
        provider_options={
            "device_type": "NPU",
            "num_of_threads": "1",
        },
    )


def _coreml_profile() -> ProviderProfile:
    return ProviderProfile(
        name="CoreMLExecutionProvider",
        device_type="coreml",
        optimization_level="DISABLE",
        enable_graph_cache=False,
        enable_compile_cache=False,
        cache_dir=None,
        device_id=0,
        memory_limit_mb=None,
        provider_options={
            "ModelFormat": "MLProgram",
            "MLComputeUnits": "ALL",
            "RequireStaticInputShapes": "1",
        },
    )


def _generic_profile(name: str) -> ProviderProfile:
    return ProviderProfile(
        name=name,
        device_type="generic",
        optimization_level="DISABLE",
        enable_graph_cache=False,
        enable_compile_cache=False,
        cache_dir=None,
        device_id=0,
        memory_limit_mb=None,
        provider_options={},
    )


_PROFILE_BUILDERS = (
    ("vitisai", lambda: _npu_profile("VitisAIExecutionProvider", "vitisai")),
    ("ryzenai", lambda: _npu_profile("RyzenAIExecutionProvider", "ryzenai")),
    ("openvino", _openvino_profile),
    ("coreml", _coreml_profile),
    ("tensorrt", _tensorrt_profile),
    ("cuda", _cuda_profile),
    ("migraphx", _migraphx_profile),
    ("rocm", _rocm_profile),
    ("directml", _directml_profile),
    ("dml", _directml_profile),
    ("cpu", _cpu_profile),
)


def primary_provider_name(providers: Optional[Sequence[Any]]) -> str:
    """Return the first provider. That entry is the selected target."""
    for provider in providers or ():
        return _provider_name(provider)
    return "CPUExecutionProvider"


def filter_session_providers(
    providers: Optional[Sequence[Any]],
    available: Optional[Sequence[Any]] = None,
) -> Tuple[List[Any], List[str]]:
    """
    Keep requested providers that this ONNX Runtime build can load.

    ``available`` defaults to ``ort.get_available_providers()``. Order is
    preserved, and a ``(name, options)`` tuple stays intact. An empty
    intersection becomes ``CPUExecutionProvider``. The second list is the
    provider names that were removed.
    """
    if available is None:
        available = ort.get_available_providers()
    available_names = {_provider_name(provider) for provider in available}
    kept: List[Any] = []
    dropped: List[str] = []
    for provider in providers or []:
        name = _provider_name(provider)
        if name in available_names:
            kept.append(provider)
        else:
            dropped.append(name)
    if not kept:
        kept = ["CPUExecutionProvider"]
    return kept, dropped


def provider_profile_for(provider: Any, **overrides: Any) -> ProviderProfile:
    """
    Build the profile for one provider name.

    Keyword arguments replace fields on the copy. ``provider_options`` is
    merged rather than replaced.
    """
    name = _provider_name(provider)
    lowered = name.lower()
    builder = None
    for marker, candidate in _PROFILE_BUILDERS:
        if marker in lowered:
            builder = candidate
            break
    profile = builder() if builder is not None else _generic_profile(name)
    profile.name = name
    for key, value in overrides.items():
        if value is None:
            continue
        if key == "provider_options" and isinstance(value, dict):
            profile.provider_options.update(value)
            continue
        if hasattr(profile, key):
            setattr(profile, key, value)
    embedded = overrides.get("provider_options")
    if (
        "device_id" not in overrides
        and isinstance(embedded, dict)
        and embedded.get("device_id") is not None
    ):
        profile.device_id = int(embedded["device_id"])
    if "device_id" in profile.provider_options:
        profile.provider_options["device_id"] = str(profile.device_id)
    return profile


def ep_provider_options(profile: ProviderProfile) -> Dict[str, str]:
    """
    Return the provider-options dict ONNX Runtime will accept.

    Session hints such as CPU thread counts and ``performance_mode`` stay on
    the profile. They are not included here.
    """
    allowed = _EP_OPTION_KEYS.get(profile.device_type, frozenset())
    source: Dict[str, Any] = dict(profile.provider_options)
    if profile.device_type in _GPU_MEMORY_DEVICES or profile.device_type == "directml":
        source.setdefault("device_id", str(profile.device_id))
    if profile.memory_limit_mb and profile.device_type in _GPU_MEMORY_DEVICES:
        source["gpu_mem_limit"] = str(int(profile.memory_limit_mb) * 1024 * 1024)
    if profile.cache_dir and profile.enable_compile_cache:
        if profile.device_type in {"vitisai", "ryzenai"}:
            source.setdefault("cacheDir", profile.cache_dir)
            source.setdefault("cacheKey", profile.device_type)
        elif profile.device_type == "openvino":
            source.setdefault("cache_dir", profile.cache_dir)
        elif profile.device_type == "tensorrt":
            source["trt_engine_cache_enable"] = "1"
            source["trt_engine_cache_path"] = profile.cache_dir
    elif not profile.enable_compile_cache:
        for key in (
            "cacheDir",
            "cacheKey",
            "cache_dir",
            "trt_engine_cache_enable",
            "trt_engine_cache_path",
        ):
            source.pop(key, None)
    options: Dict[str, str] = {}
    for key, value in source.items():
        if key not in allowed or value is None:
            continue
        options[key] = str(value)
    return options


_CACHE_EP_KEYS = frozenset({
    "cacheDir",
    "cacheKey",
    "cache_dir",
    "trt_engine_cache_enable",
    "trt_engine_cache_path",
})


@dataclass
class ProviderCacheSpec:
    """
    Compile cache and ORT profile locations for one provider.

    ``logical_options`` is the stable view (``cache_dir`` and ``enable_cache``).
    ``provider_options`` is that view translated into the keys the selected
    execution provider accepts. VitisAI and RyzenAI receive ``cacheDir`` and
    ``cacheKey``. The XCLBIN and VitisAI directories sit under ``cache_dir``.
    """

    enabled: bool
    cache_dir: Optional[str]
    xclbin_dir: Optional[str]
    vitisai_dir: Optional[str]
    profile_dir: Optional[str]
    profiling: bool
    logical_options: Dict[str, Any] = field(default_factory=dict)
    provider_options: Dict[str, str] = field(default_factory=dict)

    def ensure_directories(self) -> None:
        """Create the compile-cache and profile directories this spec names."""
        if self.enabled and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            if self.xclbin_dir:
                os.makedirs(self.xclbin_dir, exist_ok=True)
            if self.vitisai_dir:
                os.makedirs(self.vitisai_dir, exist_ok=True)
        if self.profiling and self.profile_dir:
            os.makedirs(self.profile_dir, exist_ok=True)


def provider_cache_spec(
    profile: ProviderProfile,
    *,
    enabled: bool,
    profiling: bool = False,
    profile_dir: Optional[str] = None,
) -> ProviderCacheSpec:
    """
    Describe where this provider should reuse compiled artifacts.

    Caching is active only when the caller enabled it and the profile has a
    directory. The logical options stay ``cache_dir`` and ``enable_cache``.
    The forwarded options are whatever ``ep_provider_options`` keeps for a
    compile cache.
    """
    active = bool(enabled and profile.enable_compile_cache and profile.cache_dir)
    cache_dir = profile.cache_dir if active else None
    npu = profile.device_type in {"vitisai", "ryzenai"}
    xclbin_dir = os.path.join(cache_dir, "xclbin") if cache_dir and npu else None
    vitisai_dir = os.path.join(cache_dir, "vitisai") if cache_dir and npu else None
    resolved_profile = profile_dir if profiling else None
    if profiling and not resolved_profile and cache_dir:
        resolved_profile = os.path.join(cache_dir, "profiles")
    forwarded = {
        key: value
        for key, value in ep_provider_options(profile).items()
        if key in _CACHE_EP_KEYS
    }
    if not active:
        forwarded = {}
    return ProviderCacheSpec(
        enabled=bool(enabled),
        cache_dir=cache_dir,
        xclbin_dir=xclbin_dir,
        vitisai_dir=vitisai_dir,
        profile_dir=resolved_profile,
        profiling=bool(profiling),
        logical_options={
            "cache_dir": cache_dir,
            "enable_cache": active,
        },
        provider_options=forwarded,
    )


def _apply_cache_switch(
    profile: ProviderProfile,
    *,
    enabled: bool,
    cache_dir: Optional[str],
    overwrite: bool,
) -> None:
    """Turn compile caching on or off before provider options are built."""
    if not enabled:
        profile.enable_compile_cache = False
        profile.cache_dir = None
        return
    if not cache_dir:
        return
    if overwrite or not profile.cache_dir:
        profile.cache_dir = cache_dir


@dataclass
class ProviderAssignment:
    """Provider names and the matching option dicts for one session."""

    providers: List[str]
    provider_options: List[Dict[str, str]]
    profile: ProviderProfile
    dropped: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.providers) != len(self.provider_options):
            raise ValueError(
                "providers and provider_options must be the same length, "
                f"got {len(self.providers)} and {len(self.provider_options)}"
            )


def _normalize_provider_request(
    requested: Sequence[Any],
    provider_options: Optional[Sequence[Any]],
) -> List[Tuple[str, Dict[str, Any]]]:
    """Pair each provider name with the option dict the caller attached."""
    entries: List[Tuple[str, Dict[str, Any]]] = []
    for entry in requested:
        name = _provider_name(entry)
        embedded: Dict[str, Any] = {}
        if (
            isinstance(entry, (list, tuple))
            and len(entry) > 1
            and isinstance(entry[1], dict)
        ):
            embedded = dict(entry[1])
        entries.append((name, embedded))
    if provider_options is None:
        return entries
    if len(provider_options) != len(entries):
        raise ValueError(
            "providers and provider_options must be the same length, "
            f"got {len(entries)} and {len(provider_options)}"
        )
    merged: List[Tuple[str, Dict[str, Any]]] = []
    for (name, embedded), extra in zip(entries, provider_options):
        if extra is None:
            extra = {}
        if not isinstance(extra, dict):
            raise TypeError(
                f"provider_options for '{name}' must be a dict, "
                f"got {type(extra).__name__}"
            )
        combined = dict(embedded)
        combined.update(extra)
        merged.append((name, combined))
    return merged


def build_provider_assignment(
    requested: Sequence[Any],
    provider_options: Optional[Sequence[Any]] = None,
    *,
    available: Optional[Sequence[Any]] = None,
    cache_dir: Optional[str] = None,
    enable_provider_cache: bool = True,
    append_cpu: bool = False,
    npu_graph_capture: bool = True,
    npu_compilation_mode: str = "default",
) -> ProviderAssignment:
    """
    Turn a provider request into the two lists InferenceSession accepts.

    ``providers`` is a list of names. ``provider_options`` is the same length.
    A provider missing from ``available`` is dropped. An empty result becomes
    CPU. ``append_cpu`` adds that fallback when the request named one device.
    """
    entries = _normalize_provider_request(list(requested or []), provider_options)
    if not entries:
        entries = [("CPUExecutionProvider", {})]
    available_names = None
    if available is not None:
        available_names = {_provider_name(provider) for provider in available}
    kept: List[Tuple[str, Dict[str, Any]]] = []
    dropped: List[str] = []
    for name, options in entries:
        if available_names is not None and name not in available_names:
            dropped.append(name)
            continue
        kept.append((name, options))
    if not kept:
        kept = [("CPUExecutionProvider", {})]
    if append_cpu and all(name != "CPUExecutionProvider" for name, _options in kept):
        kept.append(("CPUExecutionProvider", {}))

    names: List[str] = []
    ep_options: List[Dict[str, str]] = []
    profiles: List[ProviderProfile] = []
    for name, embedded in kept:
        kwargs: Dict[str, Any] = {}
        if cache_dir and enable_provider_cache:
            kwargs["cache_dir"] = cache_dir
        if embedded:
            kwargs["provider_options"] = embedded
        profile = provider_profile_for(name, **kwargs)
        _apply_cache_switch(
            profile,
            enabled=enable_provider_cache,
            cache_dir=cache_dir,
            overwrite=False,
        )
        if profile.device_type in {"vitisai", "ryzenai"}:
            if not npu_graph_capture:
                profile.enable_graph_cache = False
            if "performance_mode" not in embedded:
                profile.provider_options["performance_mode"] = _NPU_PERFORMANCE_MODE.get(
                    str(npu_compilation_mode),
                    "balanced",
                )
        names.append(profile.name)
        ep_options.append(ep_provider_options(profile))
        profiles.append(profile)
    return ProviderAssignment(
        providers=names,
        provider_options=ep_options,
        profile=profiles[0],
        dropped=dropped,
    )


_LLM_EAGER_TARGET = "RyzenAI_llm_eager"
_EMBEDDING_COMPONENT_NAMES = frozenset({
    "embedding",
    "embeddings",
    "embed",
    "tok_embeddings",
})


def _is_embedding_component(name: str) -> bool:
    """True for the token embedding graph. Its table does not fit in VAIP."""
    text = str(name).strip().lower().replace("-", "_")
    return text in _EMBEDDING_COMPONENT_NAMES or text.startswith("embed")


def _discover_overlay_xclbin() -> str:
    """
    Return a Strix overlay path when one is installed.

    The name is still written when the file is absent. Vitis then opens the
    session and leaves unmatched nodes on CPU.
    """
    roots: List[str] = []
    env_root = os.environ.get("RYZEN_AI_INSTALLATION_PATH", "").strip()
    if env_root:
        roots.append(env_root)
    program = os.environ.get("ProgramFiles", r"C:\Program Files")
    for version in ("1.8.0", "1.7.0", "1.6.0", "1.5.1"):
        roots.append(os.path.join(program, "RyzenAI", version))
    relatives = (
        os.path.join(
            "voe-4.0-win_amd64", "xclbins", "strix", "AMD_AIE2P_4x4_Overlay.xclbin"
        ),
        os.path.join(
            "voe-4.0-win_amd64", "xclbins", "strix", "AMD_AIE2P_Nx4_Overlay.xclbin"
        ),
    )
    for root in roots:
        for relative in relatives:
            candidate = os.path.join(root, relative)
            if os.path.isfile(candidate):
                return candidate
    return "AMD_AIE2P_4x4_Overlay.xclbin"


def llm_vaip_config_document() -> Dict[str, Any]:
    """VAIP passes that can claim a com.microsoft MatMulNBits node."""
    return {
        "passes": [
            {"name": "init", "plugin": "vaip-pass_init"},
            {
                "name": "fuse_MatMulNBits",
                "plugin": "vaip-pass_matmul_nbits",
                "enable_gc": True,
                "disabled": False,
            },
            {
                "name": "fuse_MLP",
                "plugin": "vaip-pass_mladf_mlp",
                "enable_gc": True,
                "disabled": False,
            },
            {
                "name": "fuse_SSMLP",
                "plugin": "vaip-pass_ssmlp",
                "enable_gc": True,
                "disabled": False,
            },
        ],
        "target": _LLM_EAGER_TARGET,
        "targets": [
            {
                "name": _LLM_EAGER_TARGET,
                "xclbin": _discover_overlay_xclbin().replace("\\", "/"),
                "pass": ["fuse_MatMulNBits", "fuse_SSMLP", "fuse_MLP"],
                "provider_options": {
                    "enable_preemption": "0",
                    "enable_txn_elf": "1",
                    "is_preemptible": "0",
                    "dd_use_elf_flow": "1",
                    "enable_cache_file_io_in_mem": "0",
                },
            }
        ],
    }


def ensure_llm_vaip_config(directory: str) -> str:
    """Write the LLM eager VAIP config into ``directory`` and return its path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "vaip_llm_eager.json")
    text = json.dumps(llm_vaip_config_document(), indent=2) + "\n"
    existing = None
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                existing = handle.read()
        except OSError:
            existing = None
    if existing != text:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    return path


def _first_explicit_options(
    provider_options: Optional[Sequence[Any]],
) -> Dict[str, Any]:
    """Return the option dict the caller attached to the first provider."""
    if not provider_options:
        return {}
    first = provider_options[0]
    if isinstance(first, dict):
        return dict(first)
    return {}


def specialize_component_assignment(
    name: str,
    assignment: ProviderAssignment,
    *,
    pinned: bool,
    explicit_options: Optional[Dict[str, Any]] = None,
    config_dir: Optional[str] = None,
) -> ProviderAssignment:
    """
    Place one component on the provider that can run it.

    An unpinned embedding stays on CPU. The fp32 table exceeds the VAIP
    protobuf cap. A layer assigned to VitisAI or RyzenAI requests
    ``RyzenAI_llm_eager`` and the MatMulNBits fuse passes. An explicit
    target, config file, or cache key is left alone. ``lm_head`` and any
    other pinned provider, including DirectML, stay as assigned.
    """
    explicit = dict(explicit_options or {})
    providers = list(assignment.providers)
    if not providers:
        return assignment
    if _is_embedding_component(name) and not pinned and provider_is_npu(providers[0]):
        cpu = provider_profile_for("CPUExecutionProvider")
        return ProviderAssignment(
            providers=["CPUExecutionProvider"],
            provider_options=[{}],
            profile=cpu,
            dropped=list(assignment.dropped),
        )
    if not _LAYER_COMPONENT.search(str(name)):
        return assignment
    if not provider_is_npu(providers[0]):
        return assignment
    profile = assignment.profile.copy()
    if profile.device_type not in {"vitisai", "ryzenai"}:
        return assignment
    if "target" not in explicit:
        profile.provider_options["target"] = _LLM_EAGER_TARGET
    if "cacheKey" not in explicit:
        profile.provider_options["cacheKey"] = str(name)
    if (
        profile.device_type == "vitisai"
        and "config_file" not in explicit
        and config_dir
    ):
        profile.provider_options["config_file"] = ensure_llm_vaip_config(config_dir)
    options = [dict(item) for item in assignment.provider_options]
    options[0] = ep_provider_options(profile)
    return ProviderAssignment(
        providers=providers,
        provider_options=options,
        profile=profile,
        dropped=list(assignment.dropped),
    )


@dataclass
class WarmupResult:
    """One warmup pass. These runs pay for compilation and are not a benchmark."""

    component_name: str
    iterations: int
    elapsed_ms: float


@dataclass
class BenchmarkResult:
    """
    Steady-state latency for one component.

    ``first_run_ms`` is the cold compilation sample. It is recorded and left
    out of the measured statistics when ``exclude_first_run`` discards it.
    """

    component_name: str
    exclude_first_run: bool
    warmup_iterations: int
    measured_iterations: int
    cold_run_excluded: bool
    first_run_ms: Optional[float]
    latencies_ms: List[float]
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float


def _concrete_ort_shape(shape: Any) -> Optional[Tuple[int, ...]]:
    """Return a fully static shape, or None when any axis is symbolic."""
    if shape is None:
        return None
    dims: List[int] = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            dims.append(dim)
            continue
        return None
    return tuple(dims) if dims else None


def _validate_dummy_inputs(session: Any, dummy_inputs: Dict[str, np.ndarray]) -> None:
    """Reject a warmup feed that would recompile or crash a static NPU graph."""
    if not dummy_inputs:
        raise ValueError("dummy_inputs is empty")
    getter = getattr(session, "get_inputs", None)
    if getter is None:
        return
    required = list(getter())
    missing = [item.name for item in required if item.name not in dummy_inputs]
    if missing:
        raise ValueError(f"dummy_inputs is missing session inputs: {missing}")
    for item in required:
        value = dummy_inputs[item.name]
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"dummy input '{item.name}' must be a numpy array, "
                f"got {type(value).__name__}"
            )
        expected = _concrete_ort_shape(getattr(item, "shape", None))
        if expected is not None and tuple(value.shape) != expected:
            raise ValueError(
                f"dummy input '{item.name}' has shape {tuple(value.shape)}, "
                f"but the session shape is {expected}. "
                "A static NPU graph cannot change shape during warmup."
            )


def _latency_stats(samples_ms: Sequence[float]) -> Tuple[float, float, float, float, float]:
    """Mean, median, min, max, and population stdev in milliseconds."""
    values = [float(sample) for sample in samples_ms]
    if not values:
        raise ValueError("measured_iterations produced no samples")
    spread = statistics.pstdev(values) if len(values) > 1 else 0.0
    return (
        statistics.mean(values),
        statistics.median(values),
        min(values),
        max(values),
        spread,
    )


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SessionConfig:
    """
    Configuration for ONNX Runtime session creation.

    Attributes:
        providers: Execution providers in priority order. The first entry
            selects the ProviderProfile.
        profile: Explicit profile. When omitted, the first provider selects one.
        optimization_level: Graph optimization level (DISABLE/BASIC/EXTENDED/ALL).
            None uses the selected profile's level.
        inter_op_threads: Number of threads for parallel op execution.
            0 uses the profile value.
        intra_op_threads: Number of threads within a single op.
            0 uses the profile value.
        memory_limit_mb: Optional memory limit for the active device allocator.
        device_id: Device index passed to the execution provider. 0 is the
            default device. An explicit profile whose device is already
            non-zero keeps that id when this field is left at 0.
        provider_options: One option dict per provider, in the same order as
            ``providers``. None leaves each provider on its profile.
        use_available_providers_only: Drop providers this process cannot load.
        cache_dir: Compile-cache directory. The manager uses
            ``<model_dir>/.ort_cache`` when this is omitted and caching is on.
        warmup: When true, a loaded component is not treated as compiled
            until ``warmup_component`` has run.
        static_shapes: Keep ONNX shape checking on and freeze symbolic
            batch and sequence axes for IO binding.
        memory_policy: auto, minimal, persistent, io_bound, or npu_static.
            auto uses npu_static on an NPU target and persistent otherwise.
        enable_provider_cache: Reuse compiled graphs across loads. VitisAI and
            RyzenAI receive ``cacheDir`` and ``cacheKey`` when this is true
            and a directory is set.
        enable_profiling: Enable ONNX Runtime profiling.
        profile_dir: Directory for the ORT profile trace. When omitted and
            profiling is on, traces go under ``<cache_dir>/profiles``.
        graph_optimization_level: Enum for optimization level
        disable_shape_check: Disable node shape checking for dynamic shapes.
            AOT NPU profiles keep shape checking on.
        execution_policy: How this model may be split across sessions.
            ``auto`` resolves to monolithic execution, which still rejects a
            sequential NPU layer split. Phase-split and MoE routing are the
            allowed NPU exceptions.

    NPU-Specific Options (for VitisAIExecutionProvider):
        npu_graph_capture: Enable NPU graph capture for faster inference (default: True)
            When enabled, the NPU graph is captured and cached for reuse,
            reducing compilation overhead on subsequent runs.
        npu_weight_preload: Preload weights to NPU tile SRAM (default: True)
            When enabled, model weights are preloaded into the NPU's on-chip
            SRAM for faster access during inference.
        npu_compilation_mode: NPU compilation strategy (default: "default")
            Options:
            - "default": Balanced compilation for general workloads
            - "fast": Faster compilation, potentially lower optimization
            - "optimized": Slower compilation, maximum optimization
    """
    providers: List[str] = field(default_factory=lambda: ['CPUExecutionProvider'])
    execution_policy: Union[ExecutionPolicy, str] = "auto"
    profile: Optional[ProviderProfile] = None
    optimization_level: Optional[str] = None  # None = use the provider profile
    inter_op_threads: int = 0  # 0 = profile default
    intra_op_threads: int = 0  # 0 = profile default
    memory_limit_mb: Optional[int] = None
    device_id: int = 0
    provider_options: Optional[List[Dict[str, Any]]] = None
    use_available_providers_only: bool = True
    cache_dir: Optional[str] = None
    warmup: bool = False
    static_shapes: bool = True
    memory_policy: str = "auto"
    enable_provider_cache: bool = True
    enable_profiling: bool = False
    profile_dir: Optional[str] = None
    disable_shape_check: bool = True  # Ignored when static_shapes is set

    def __post_init__(self) -> None:
        if self.device_id is None:
            self.device_id = 0
        else:
            self.device_id = int(self.device_id)
        self.memory_policy = _coerce_memory_policy(self.memory_policy)
        if self.provider_options is not None:
            if len(self.provider_options) != len(self.providers):
                raise ValueError(
                    "provider_options and providers must be the same length, "
                    f"got {len(self.provider_options)} and {len(self.providers)}"
                )
            self.provider_options = [
                dict(options or {}) for options in self.provider_options
            ]

    def _device_id_override(self) -> Optional[int]:
        """
        Device index to write onto a profile.

        The default 0 does not replace an explicit profile that already
        names another device.
        """
        if int(self.device_id) != 0:
            return int(self.device_id)
        if self.profile is not None and int(self.profile.device_id) != 0:
            return None
        return int(self.device_id)

    def _provider_option_list(self) -> List[Dict[str, Any]]:
        """Option dicts aligned with ``providers``. Missing entries are empty."""
        if not self.provider_options:
            return [{} for _ in self.providers]
        if len(self.provider_options) != len(self.providers):
            raise ValueError(
                "provider_options and providers must be the same length, "
                f"got {len(self.provider_options)} and {len(self.providers)}"
            )
        return [dict(options or {}) for options in self.provider_options]

    # NPU-specific options (P1.2: NPU Session Options)
    npu_graph_capture: bool = True
    npu_weight_preload: bool = True
    npu_compilation_mode: str = "default"  # "default", "fast", "optimized"
    
    def active_profile(self, cache_dir: Optional[str] = None) -> ProviderProfile:
        """
        Return the profile for the selected provider.

        An explicit ``profile`` wins. Otherwise the first provider is the
        selection. Session-level device, memory, cache, and optimization
        fields replace the matching profile values when they are set.
        """
        if self.profile is not None:
            profile = self.profile.copy()
        else:
            profile = provider_profile_for(primary_provider_name(self.providers))
        if self.optimization_level:
            profile.optimization_level = self.optimization_level
        if self.memory_limit_mb is not None:
            profile.memory_limit_mb = self.memory_limit_mb
        chosen_device = self._device_id_override()
        if chosen_device is not None:
            profile.device_id = chosen_device
            if "device_id" in profile.provider_options:
                profile.provider_options["device_id"] = str(chosen_device)
        _apply_cache_switch(
            profile,
            enabled=self.enable_provider_cache,
            cache_dir=self.cache_dir or cache_dir,
            overwrite=False,
        )
        if profile.device_type in {"vitisai", "ryzenai"}:
            if not self.npu_graph_capture:
                profile.enable_graph_cache = False
            profile.provider_options["performance_mode"] = _NPU_PERFORMANCE_MODE.get(
                str(self.npu_compilation_mode),
                "balanced",
            )
        if profile.memory_limit_mb and profile.device_type in _GPU_MEMORY_DEVICES:
            profile.provider_options["gpu_mem_limit"] = str(
                int(profile.memory_limit_mb) * 1024 * 1024
            )
        return profile

    def resolved_optimization_level(self) -> str:
        """Profile level, unless this config sets one explicitly."""
        return self.active_profile().optimization_level

    @staticmethod
    def _optimization_enum(level: str) -> ort.GraphOptimizationLevel:
        """Convert a profile or config level string to the ORT enum."""
        mapping = {
            "DISABLE": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        return mapping.get(
            str(level).upper(),
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        )

    def get_optimization_level_enum(self) -> ort.GraphOptimizationLevel:
        """Convert the resolved optimization level to an ONNX Runtime enum."""
        return self._optimization_enum(self.resolved_optimization_level())

    def ort_providers(self, cache_dir: Optional[str] = None) -> List[Any]:
        """
        Build the provider list passed to InferenceSession.

        Each accelerator gets its own option dict. CPU stays a bare name.
        """
        prepared: List[Any] = []
        extras = self._provider_option_list()
        for index, provider in enumerate(self.providers):
            profile = provider_profile_for(provider)
            chosen_device = self._device_id_override()
            if chosen_device is not None:
                profile.device_id = chosen_device
                if "device_id" in profile.provider_options:
                    profile.provider_options["device_id"] = str(chosen_device)
            if self.memory_limit_mb is not None:
                profile.memory_limit_mb = self.memory_limit_mb
            extra = extras[index]
            if extra:
                if extra.get("device_id") is not None:
                    profile.device_id = int(extra["device_id"])
                profile.provider_options.update(extra)
                if "device_id" in profile.provider_options:
                    profile.provider_options["device_id"] = str(profile.device_id)
            _apply_cache_switch(
                profile,
                enabled=self.enable_provider_cache,
                cache_dir=self.cache_dir or cache_dir,
                overwrite=True,
            )
            if profile.device_type in {"vitisai", "ryzenai"} and not self.npu_graph_capture:
                profile.enable_graph_cache = False
            options = ep_provider_options(profile)
            provider_name = _provider_name(provider)
            if options:
                prepared.append((provider_name, options))
            else:
                prepared.append(provider_name)
        return prepared
    
    def create_session_options(
        self,
        cache_dir: Optional[str] = None,
        profile: Optional[ProviderProfile] = None,
        providers: Optional[Sequence[Any]] = None,
    ) -> ort.SessionOptions:
        """
        Create SessionOptions from the selected provider profile.

        VitisAI still receives graph capture, weight preload, and compilation
        mode. AOT NPU profiles keep shape checking enabled. Pass ``profile``
        and ``providers`` to build options for one component instead of the
        manager-wide provider list.

        Returns:
            Configured ort.SessionOptions object

        Example:
            >>> config = SessionConfig(
            ...     providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
            ...     npu_graph_capture=True,
            ...     npu_weight_preload=True
            ... )
            >>> opts = config.create_session_options()
            # opts now contains vitisai.* session config entries
        """
        component_profile = profile is not None
        if profile is None:
            profile = self.active_profile(cache_dir=cache_dir)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = self._optimization_enum(profile.optimization_level)

        if component_profile:
            intra = int(profile.provider_options.get("intra_op_num_threads") or 0)
            inter = int(profile.provider_options.get("inter_op_num_threads") or 0)
        else:
            intra = self.intra_op_threads or int(
                profile.provider_options.get("intra_op_num_threads") or 0
            )
            inter = self.inter_op_threads or int(
                profile.provider_options.get("inter_op_num_threads") or 0
            )
        if inter > 0:
            opts.inter_op_num_threads = inter
        if intra > 0:
            opts.intra_op_num_threads = intra

        if profile.device_type == "cpu":
            opts.enable_cpu_mem_arena = bool(
                profile.provider_options.get("enable_cpu_mem_arena", True)
            )
            spinning = bool(profile.provider_options.get("allow_spinning", False))
            spin_value = "1" if spinning else "0"
            opts.add_session_config_entry("session.intra_op.allow_spinning", spin_value)
            opts.add_session_config_entry("session.inter_op.allow_spinning", spin_value)

        profile_prefix = self._profile_file_prefix(profile)
        if self.enable_profiling:
            opts.enable_profiling = True
            if profile_prefix:
                opts.profile_file_prefix = profile_prefix

        if profile.device_type in _STATIC_SHAPE_DEVICES and self.disable_shape_check:
            _warn_static_npu_shapes(profile.name)
        elif not self.static_shapes and self.disable_shape_check:
            opts.add_session_config_entry("session.disable_node_shape_check", "1")

        selected_providers = self.providers if providers is None else providers
        self._apply_vitisai_profile(opts, profile, selected_providers)
        return opts

    def _apply_vitisai_profile(
        self,
        opts: ort.SessionOptions,
        profile: ProviderProfile,
        providers: Optional[Sequence[Any]] = None,
    ) -> None:
        """Write VitisAI session keys when that provider is selected or listed."""
        selected = self.providers if providers is None else providers
        uses_vitisai = profile.device_type == "vitisai" or any(
            "vitisai" in _provider_name(provider).lower() for provider in selected
        )
        if not uses_vitisai:
            return
        opts.add_session_config_entry(
            "vitisai.graph_capture",
            "1" if profile.enable_graph_cache else "0",
        )
        opts.add_session_config_entry(
            "vitisai.weight_preload",
            "1" if self.npu_weight_preload else "0",
        )
        opts.add_session_config_entry(
            "vitisai.compilation_mode",
            self.npu_compilation_mode,
        )
        if profile.memory_limit_mb is not None:
            opts.add_session_config_entry(
                "vitisai.memory_limit_mb",
                str(int(profile.memory_limit_mb)),
            )

    def artifact_cache(self, cache_dir: Optional[str] = None) -> ProviderCacheSpec:
        """
        Return the compile-cache description for the selected provider.

        The logical options are ``cache_dir`` and ``enable_cache``. The
        provider-options dict is the translation for that provider.
        """
        return provider_cache_spec(
            self.active_profile(cache_dir=cache_dir),
            enabled=self.enable_provider_cache,
            profiling=self.enable_profiling,
            profile_dir=self.profile_dir,
        )

    def _profile_file_prefix(self, profile: ProviderProfile) -> Optional[str]:
        """Directory prefix for an ORT profile trace, or None when unused."""
        if not self.enable_profiling:
            return None
        directory = self.profile_dir
        if not directory and self.enable_provider_cache and profile.cache_dir:
            directory = os.path.join(profile.cache_dir, "profiles")
        if not directory:
            return None
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, "ort_profile")

    def resolved_execution_policy(self) -> ExecutionPolicy:
        """
        Return the policy the NPU guard uses.

        ``auto`` stays stored as auto and behaves as monolithic execution.
        The sequential unload policy is still rejected on an NPU. A resident
        layer series is not.
        """
        policy = coerce_execution_policy(self.execution_policy)
        self.execution_policy = policy
        if policy is ExecutionPolicy.AUTO:
            return ExecutionPolicy.MONOLITHIC
        return policy

    def resolved_memory_policy(self) -> str:
        """
        Return the memory mode for this config.

        ``auto`` uses a static NPU pool when an NPU provider is selected and
        persistent residency otherwise.
        """
        policy = _coerce_memory_policy(self.memory_policy)
        self.memory_policy = policy
        if policy != "auto":
            return policy
        if providers_target_npu(self.providers):
            return "npu_static"
        return "persistent"

    def validate_execution_policy(
        self,
        component_names: Optional[Iterable[str]] = None,
    ) -> ExecutionPolicy:
        """
        Raise if this config would unload an NPU session between steps.

        ``component_names`` comes from the discovered model. A persistent
        layer series is accepted. The sequential unload policy is rejected
        on an NPU provider.
        """
        return validate_npu_execution_policy(
            self.resolved_execution_policy(),
            self.providers,
            component_names,
        )


@dataclass
class SessionInfo:
    """
    Information about a loaded or loadable ONNX session.
    
    Attributes:
        name: Component name (e.g., "layer_0", "embedding")
        file_path: Absolute path to ONNX file
        file_size_bytes: Size of ONNX file on disk
        is_loaded: Whether session is currently loaded
        input_names: List of input tensor names
        output_names: List of output tensor names
        input_types: Dict of input name to ONNX type string
        output_types: Dict of output name to ONNX type string
        provider_used: Actual provider being used (only when loaded)
        phase: Scheduler phase. Prefill, decode, router, expert, kv_cache,
            speculative_draft, and verification are the AMD phases.
        execution_group: Sessions that stay together across a phase load.
        device_preference: npu, igpu, gpu, cpu, or auto. None uses the manager list.
        resident_policy: auto, resident, on_demand, or evict. auto follows the phase.
        execution_order: Position in ``get_execution_order``.
        device_affinity: Device this component runs on (npu, igpu, gpu, cpu).
        estimated_runtime_memory_mb: Five-term estimate, filled by the memory plan.
        compiled: True after a warmup has compiled the graph.
        warmed_up: True after ``warmup_component`` or ``warmup_session``.
        device_type: Device the plan assigned (npu, igpu, gpu, cpu)
        resident: Whether this session is kept loaded across phases
        scheduled_provider: Provider the plan resolved for this step
        input_shapes: Declared input shapes, including symbolic dimensions
        output_shapes: Declared output shapes, including symbolic dimensions
        input_dtypes: Dtype name for each input
        output_dtypes: Dtype name for each output
    """
    name: str
    file_path: str
    file_size_bytes: int = 0
    is_loaded: bool = False
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    input_types: Dict[str, str] = field(default_factory=dict)
    output_types: Dict[str, str] = field(default_factory=dict)
    input_shapes: Dict[str, Tuple[Any, ...]] = field(default_factory=dict)
    output_shapes: Dict[str, Tuple[Any, ...]] = field(default_factory=dict)
    input_dtypes: Dict[str, str] = field(default_factory=dict)
    output_dtypes: Dict[str, str] = field(default_factory=dict)
    provider_used: Optional[str] = None
    phase: Optional[str] = None
    execution_order: int = 0
    execution_group: Optional[str] = None
    device_preference: Optional[str] = None
    device_affinity: Optional[str] = None
    resident_policy: str = "auto"
    estimated_runtime_memory_mb: float = 0.0
    compiled: bool = False
    warmed_up: bool = False
    device_type: Optional[str] = None
    resident: bool = False
    scheduled_provider: Optional[str] = None

    def resolved_resident_policy(self) -> str:
        """
        Return the residency this session should use.

        ``auto`` follows the phase. Prefill, decode, the router, and the KV
        cache stay resident. Experts, speculative drafts, and verification
        load for that phase and can be evicted when another phase runs.
        """
        policy = _coerce_resident_policy(self.resident_policy)
        if policy != "auto":
            return policy
        if self.phase in PHASE_RESIDENT_DEFAULT:
            return PHASE_RESIDENT_DEFAULT[self.phase]
        if self.resident:
            return "resident"
        return "on_demand"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
            "file_size_mb": round(self.file_size_bytes / (1024 * 1024), 2),
            "is_loaded": self.is_loaded,
            "input_names": self.input_names,
            "output_names": self.output_names,
            "input_types": self.input_types,
            "output_types": self.output_types,
            "input_shapes": {key: list(value) for key, value in self.input_shapes.items()},
            "output_shapes": {key: list(value) for key, value in self.output_shapes.items()},
            "input_dtypes": self.input_dtypes,
            "output_dtypes": self.output_dtypes,
            "provider_used": self.provider_used,
            "phase": self.phase,
            "execution_order": self.execution_order,
            "execution_group": self.execution_group,
            "device_preference": self.device_preference,
            "device_affinity": self.device_affinity,
            "resident_policy": self.resident_policy,
            "estimated_runtime_memory_mb": self.estimated_runtime_memory_mb,
            "compiled": self.compiled,
            "warmed_up": self.warmed_up,
            "device_type": self.device_type,
            "resident": self.resident,
            "scheduled_provider": self.scheduled_provider,
        }


# Phases a heterogeneous schedule can assign. Embedding, norm, and lm_head
# stay because the token path uses them. The AMD schedule is phase-based:
# prefill, decode, the MoE router, expert execution, the KV cache, a
# speculative draft, and verification. Aliases collapse onto these names.
EXECUTION_PHASES = frozenset({
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

PHASE_ALIASES = {
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

RESIDENT_POLICIES = frozenset({"auto", "resident", "on_demand", "evict"})

# auto follows the phase. Context that the next token still needs stays
# loaded. Experts, drafts, and verification are loaded for that phase only.
PHASE_RESIDENT_DEFAULT = {
    "prefill": "resident",
    "decode": "resident",
    "router": "resident",
    "expert": "on_demand",
    "kv_cache": "resident",
    "speculative_draft": "on_demand",
    "verification": "on_demand",
    "norm": "resident",
    "lm_head": "resident",
    "embedding": "resident",
}

DEVICE_PREFERENCES = frozenset({"auto", "npu", "igpu", "gpu", "cpu"})

_DEVICE_PREFERENCE_ALIASES = {
    "auto": "auto",
    "npu": "npu",
    "xdna": "npu",
    "igpu": "igpu",
    "rdna": "igpu",
    "gpu": "gpu",
    "cpu": "cpu",
}

_DEVICE_PROVIDER_ORDER = {
    "npu": [
        "RyzenAIExecutionProvider",
        "VitisAIExecutionProvider",
        "OpenVINOExecutionProvider",
        "CoreMLExecutionProvider",
    ],
    "igpu": [
        "DmlExecutionProvider",
        "ROCmExecutionProvider",
        "MIGraphXExecutionProvider",
    ],
    "gpu": [
        "CUDAExecutionProvider",
        "TensorrtExecutionProvider",
        "ROCmExecutionProvider",
        "MIGraphXExecutionProvider",
    ],
    "cpu": ["CPUExecutionProvider"],
}

_PHASE_NAME_PREFIXES = (
    "speculative_draft",
    "kv_cache",
    "verification",
    "prefill",
    "decode",
    "router",
    "expert",
    "embedding",
    "lm_head",
    "norm",
    "draft",
)

_IGPU_PROVIDER_MARKERS = ("rocm", "migraphx", "dml", "directml")
_DISCRETE_GPU_MARKERS = ("cuda", "tensorrt")


def device_type_for_provider(provider: Any) -> str:
    """Return npu, igpu, gpu, cpu, or generic for a provider name."""
    name = _provider_name(provider).lower()
    if any(marker in name for marker in NPU_PROVIDER_MARKERS):
        return "npu"
    if any(marker in name for marker in _IGPU_PROVIDER_MARKERS):
        return "igpu"
    if any(marker in name for marker in _DISCRETE_GPU_MARKERS):
        return "gpu"
    if "cpu" in name:
        return "cpu"
    return "generic"


def _coerce_phase(phase: Any) -> str:
    """Normalize a phase name and reject ones the scheduler does not know."""
    value = getattr(phase, "value", phase)
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    text = PHASE_ALIASES.get(text, text)
    if text not in EXECUTION_PHASES:
        allowed = ", ".join(sorted(EXECUTION_PHASES))
        raise ValueError(
            f"Unknown execution phase '{phase}'. Expected one of: {allowed}"
        )
    return text


def _coerce_resident_policy(policy: Any) -> str:
    """Normalize a residency policy."""
    text = str(policy if policy is not None else "auto").strip().lower().replace("-", "_")
    if text == "transient":
        text = "on_demand"
    if text not in RESIDENT_POLICIES:
        allowed = ", ".join(sorted(RESIDENT_POLICIES))
        raise ValueError(
            f"Unknown resident_policy '{policy}'. Expected one of: {allowed}"
        )
    return text


def _coerce_device_preference(preference: Any) -> Optional[str]:
    """Normalize a device preference. None means the manager provider list."""
    if preference is None or preference == "":
        return None
    text = str(preference).strip().lower()
    if text not in _DEVICE_PREFERENCE_ALIASES:
        allowed = ", ".join(sorted(DEVICE_PREFERENCES))
        raise ValueError(
            f"Unknown device_preference '{preference}'. Expected one of: {allowed}"
        )
    return _DEVICE_PREFERENCE_ALIASES[text]


def infer_component_phase(name: str) -> Optional[str]:
    """
    Read a phase from a component name.

    ``expert_0`` is the expert phase. ``layer_0`` is not a phase: sequential
    layer names stay unassigned so they are not treated as an NPU schedule.
    """
    text = str(name).strip().lower().replace("-", "_")
    if text in PHASE_ALIASES:
        return PHASE_ALIASES[text]
    for prefix in _PHASE_NAME_PREFIXES:
        if text.startswith(prefix + "_"):
            return _coerce_phase(prefix)
    return None


def _step_targets_npu(step: "ExecutionStep") -> bool:
    """True when the step is placed on an ahead-of-time NPU."""
    return step.device_type == "npu" or provider_is_npu(step.provider)


@dataclass
class ExecutionStep:
    """
    One component in a heterogeneous run.

    ``phase`` is prefill, decode, router, expert, kv_cache, speculative_draft,
    verification, norm, lm_head, or embedding. ``moe_router`` and
    ``expert_execution`` are accepted aliases. ``depends_on`` names earlier
    steps. ``resident`` keeps the session loaded. ``resident_policy`` overrides
    that when it is not ``auto``.
    """

    component_name: str
    provider: str
    device_type: str
    phase: str
    depends_on: List[str] = field(default_factory=list)
    output_buffers: List[str] = field(default_factory=list)
    resident: bool = True
    provider_options: Optional[Dict[str, Any]] = None
    execution_group: Optional[str] = None
    device_preference: Optional[str] = None
    resident_policy: str = "auto"

    def __post_init__(self) -> None:
        self.component_name = str(self.component_name)
        self.provider = _provider_name(self.provider)
        self.device_type = str(self.device_type).strip().lower()
        self.phase = _coerce_phase(self.phase)
        self.depends_on = [str(name) for name in self.depends_on]
        self.output_buffers = [str(name) for name in self.output_buffers]
        self.resident = bool(self.resident)
        if self.execution_group is None or str(self.execution_group).strip() == "":
            self.execution_group = None
        else:
            self.execution_group = str(self.execution_group)
        self.device_preference = _coerce_device_preference(self.device_preference)
        self.resident_policy = _coerce_resident_policy(self.resident_policy)
        if self.provider_options is None:
            self.provider_options = {}
        elif isinstance(self.provider_options, dict):
            self.provider_options = dict(self.provider_options)
        else:
            raise TypeError(
                "ExecutionStep.provider_options must be a dict, "
                f"got {type(self.provider_options).__name__}"
            )


@dataclass
class ExecutionPlan:
    """
    Ordered schedule for SessionManager.

    Steps form a dependency graph. ``ordered_steps`` is the run order.
    """

    steps: List[ExecutionStep]
    name: str = "model"

    def validate(self) -> None:
        """
        Reject an empty plan, a repeated component, a bad device placement,
        a missing dependency, or a cycle.
        """
        if not self.steps:
            raise ValueError("Execution plan has no steps")
        seen = set()
        for step in self.steps:
            if not step.component_name:
                raise ValueError("Execution step is missing a component name")
            if step.component_name in seen:
                raise ValueError(
                    f"Execution plan repeats component '{step.component_name}'"
                )
            seen.add(step.component_name)
            expected = device_type_for_provider(step.provider)
            if expected != "generic" and step.device_type != expected:
                raise ValueError(
                    f"Step '{step.component_name}' uses {step.provider}, "
                    f"which is a {expected} provider, not '{step.device_type}'."
                )
            deduped: List[str] = []
            for dependency in step.depends_on:
                if dependency not in deduped:
                    deduped.append(dependency)
            step.depends_on = deduped
            for buffer_name in step.output_buffers:
                if not buffer_name.strip():
                    raise ValueError(
                        f"Step '{step.component_name}' has an empty output buffer"
                    )
        self.ordered_steps()

    def ordered_steps(self) -> List[ExecutionStep]:
        """
        Return steps in dependency order.

        Independent steps keep their declaration order.
        """
        by_name = {step.component_name: step for step in self.steps}
        if len(by_name) != len(self.steps):
            raise ValueError("Execution plan repeats a component name")
        index = {
            step.component_name: position
            for position, step in enumerate(self.steps)
        }
        children: Dict[str, List[str]] = {name: [] for name in by_name}
        indegree = {name: 0 for name in by_name}
        for step in self.steps:
            for dependency in step.depends_on:
                if dependency not in by_name:
                    raise ValueError(
                        f"Step '{step.component_name}' depends on '{dependency}', "
                        "which is not in the plan."
                    )
                if dependency == step.component_name:
                    raise ValueError(
                        f"Step '{step.component_name}' depends on itself"
                    )
                children[dependency].append(step.component_name)
                indegree[step.component_name] += 1
        ready = sorted(
            [name for name, degree in indegree.items() if degree == 0],
            key=lambda name: index[name],
        )
        ordered: List[ExecutionStep] = []
        while ready:
            name = ready.pop(0)
            ordered.append(by_name[name])
            released = []
            for child in children[name]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    released.append(child)
            released.sort(key=lambda child: index[child])
            ready.extend(released)
            ready.sort(key=lambda child: index[child])
        if len(ordered) != len(by_name):
            raise ValueError("Execution plan has a dependency cycle")
        return ordered

    def steps_for_phase(self, phase: str) -> List[ExecutionStep]:
        """Return steps of one phase in run order."""
        selected = _coerce_phase(phase)
        return [step for step in self.ordered_steps() if step.phase == selected]


_BATCH_DIM_NAMES = frozenset({"batch", "batch_size", "n", "b"})
_SEQ_DIM_NAMES = frozenset({
    "seq",
    "sequence",
    "seq_len",
    "sequence_length",
    "total_sequence_length",
    "s",
})


def _dtype_name(value: Any) -> str:
    """Map an ONNX or model-config type string to a NumPy dtype name."""
    if value is None or value == "":
        return "float16"
    text = str(value).lower().replace("tensor(", "").replace(")", "")
    if "float16" in text or text in {"fp16", "f16"}:
        return "float16"
    if "bfloat16" in text or "bf16" in text:
        return "bfloat16"
    if "float64" in text or "double" in text:
        return "float64"
    if "float" in text or text in {"fp32", "f32"}:
        return "float32"
    if "int64" in text:
        return "int64"
    if "int32" in text or text == "int":
        return "int32"
    if "bool" in text:
        return "bool"
    return "float16"


def _as_shape(value: Any) -> Tuple[Any, ...]:
    """Normalize a model_info or ONNX shape to a tuple."""
    if value is None:
        return ()
    if isinstance(value, (str, int)):
        return (value,)
    return tuple(value)


def _copy_shape_map(raw: Any) -> Dict[str, Tuple[Any, ...]]:
    if not isinstance(raw, dict):
        return {}
    return {str(name): _as_shape(shape) for name, shape in raw.items()}


def _copy_dtype_map(raw: Any) -> Dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {str(name): _dtype_name(dtype) for name, dtype in raw.items()}


def _dynamic_axes(shape: Tuple[Any, ...]) -> Tuple[int, ...]:
    axes = []
    for index, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            continue
        axes.append(index)
    return tuple(axes)


def _dim_is_batch(dim: Any) -> bool:
    return isinstance(dim, str) and dim.strip().lower() in _BATCH_DIM_NAMES


def _dim_is_sequence(dim: Any) -> bool:
    if dim is None or (isinstance(dim, int) and dim <= 0):
        return True
    if not isinstance(dim, str):
        return False
    text = dim.strip().lower()
    return text in _SEQ_DIM_NAMES or "seq" in text


def resolve_static_shape(
    shape: Tuple[Any, ...],
    *,
    static: bool,
    batch_size: int,
    max_seq_len: int,
) -> Tuple[Optional[int], ...]:
    """
    Replace symbolic batch and sequence axes with concrete sizes.

    Positive integers stay as declared. With ``static`` false, symbolic
    axes stay unset so the caller can see that the tensor is not allocatable.
    """
    resolved: List[Optional[int]] = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
            continue
        if not static:
            resolved.append(None)
            continue
        if _dim_is_batch(dim):
            resolved.append(int(batch_size))
            continue
        if _dim_is_sequence(dim):
            resolved.append(int(max_seq_len))
            continue
        raise ValueError(
            f"Cannot freeze symbolic dimension '{dim}'. "
            "Use a positive size, or name it batch or seq."
        )
    return tuple(resolved)


@dataclass
class TensorIO:
    """One session input or output."""

    name: str
    shape: Tuple[Any, ...]
    dtype: str
    dynamic_axes: Tuple[int, ...] = ()


@dataclass
class IOMetadata:
    """Shapes and dtypes SessionManager can hand to an IO binding manager."""

    component_name: str
    inputs: List[TensorIO]
    outputs: List[TensorIO]
    phase: Optional[str] = None
    device_type: Optional[str] = None
    resident: bool = False


@dataclass
class PreparedIO:
    """Buffers allocated for one component, plus a binding when a session is loaded."""

    component_name: str
    metadata: IOMetadata
    buffers: Dict[str, Any]
    binding: Any = None


@dataclass
class StaticBufferAllocation:
    """Static KV pool, resident activations, and NPU-to-iGPU shared buffers."""

    max_seq_len: int
    batch_size: int
    kv_pool: Any
    resident_hidden: Dict[str, Any]
    bridges: Dict[str, Any]


_MIB = 1024 * 1024

# Planning factors used when the compiled binary or the provider workspace
# has not been measured. The floor matters most for an AOT NPU: the XCLBIN
# context and the runtime workspace stay large even when the ONNX file is small.
COMPILED_GRAPH_FACTOR = {
    "npu": 0.35,
    "tensorrt": 0.40,
    "migraphx": 0.30,
    "directml": 0.10,
    "cuda": 0.05,
    "rocm": 0.05,
    "gpu": 0.10,
    "igpu": 0.10,
    "cpu": 0.02,
}
COMPILED_GRAPH_FLOOR_MB = {
    "npu": 32.0,
    "tensorrt": 16.0,
    "migraphx": 16.0,
    "directml": 8.0,
    "cuda": 4.0,
    "rocm": 4.0,
    "gpu": 8.0,
    "igpu": 8.0,
    "cpu": 1.0,
}
PROVIDER_WORKSPACE_FACTOR = {
    "npu": 0.15,
    "tensorrt": 0.20,
    "migraphx": 0.15,
    "directml": 0.10,
    "cuda": 0.08,
    "rocm": 0.08,
    "gpu": 0.10,
    "igpu": 0.10,
    "cpu": 0.05,
}
PROVIDER_WORKSPACE_FLOOR_MB = {
    "npu": 48.0,
    "tensorrt": 32.0,
    "migraphx": 32.0,
    "directml": 16.0,
    "cuda": 16.0,
    "rocm": 16.0,
    "gpu": 16.0,
    "igpu": 16.0,
    "cpu": 8.0,
}

_DTYPE_BYTES = {
    "float16": 2.0,
    "bfloat16": 2.0,
    "float32": 4.0,
    "float64": 8.0,
    "int64": 8.0,
    "int32": 4.0,
    "int8": 1.0,
    "bool": 1.0,
}

_AUX_COMPONENT_NAMES = frozenset({
    "embedding",
    "embed",
    "tok_embeddings",
    "norm",
    "final_norm",
    "lm_head",
})


def _round_mb(value: float) -> float:
    return round(float(value), 6)


def _declared_mb(mapping: Dict[str, Any], mb_key: str, bytes_key: str) -> Optional[float]:
    """Return a caller-supplied size in mebibytes."""
    if mb_key in mapping and mapping[mb_key] is not None:
        return float(mapping[mb_key])
    if bytes_key in mapping and mapping[bytes_key] is not None:
        return float(mapping[bytes_key]) / _MIB
    return None


def _element_bytes(value: Any) -> float:
    """Bytes per element. int4 packs two values per byte."""
    text = str(value).lower() if value is not None else ""
    if "int4" in text or "uint4" in text:
        return 0.5
    return float(_DTYPE_BYTES.get(_dtype_name(value), 2.0))


def _is_aux_component(name: str) -> bool:
    return name in _AUX_COMPONENT_NAMES or name.startswith("embed")


def _provider_memory_class(provider: str, device_type: Optional[str]) -> str:
    text = provider.lower()
    if "tensorrt" in text:
        return "tensorrt"
    if "migraphx" in text:
        return "migraphx"
    if "cuda" in text:
        return "cuda"
    if "rocm" in text:
        return "rocm"
    if "dml" in text or "directml" in text:
        return "directml"
    if device_type == "igpu":
        return "igpu"
    if device_type == "gpu":
        return "gpu"
    return "cpu"


def _kv_bytes(architecture: Dict[str, Any], layers: int, batch: int, seq: int) -> float:
    """Key, value, and one int64 attention mask at the static sequence length."""
    if layers <= 0:
        return 0.0
    element = _element_bytes(architecture["dtype"])
    tensors = (
        batch
        * int(architecture["num_kv_heads"])
        * seq
        * int(architecture["head_dim"])
        * 2
        * layers
        * element
    )
    mask = batch * seq * 8
    return float(tensors + mask)


@dataclass
class RuntimeMemoryEstimate:
    """
    Runtime bytes a session keeps resident.

    ``file_size_bytes`` is only the ONNX file. This adds the compiled graph,
    the live activations, the static KV cache, and the provider workspace.
    """

    component_name: str
    weights_mb: float
    compiled_graph_mb: float
    activation_mb: float
    kv_cache_mb: float
    provider_workspace_mb: float

    @property
    def total_mb(self) -> float:
        return _round_mb(
            self.weights_mb
            + self.compiled_graph_mb
            + self.activation_mb
            + self.kv_cache_mb
            + self.provider_workspace_mb
        )


# =============================================================================
# SESSION MANAGER
# =============================================================================

class SessionManager:
    """
    Manages ONNX session lifecycle with persistence support.
    
    This class is designed for NPU optimization where keeping sessions
    (and their weights) resident in memory is critical for performance.
    
    Features:
    - Load all sessions once at startup
    - Get pre-loaded sessions by name
    - Track memory usage
    - Thread-safe access
    - Provider fallback handling
    - Reject the sequential unload policy on VitisAI, RyzenAI, OpenVINO,
      and CoreML. Resident layer sessions stay loaded and run in order.
      lm_head may use DirectML, and embeddings stay on CPU unless pinned.
    - Select a ProviderProfile from the first execution provider that is
      installed in this ONNX Runtime build
    - Run from an ExecutionPlan: dependency order, device, phase, and residency
    """
    
    def __init__(
        self,
        model_dir: str,
        config: Optional[SessionConfig] = None,
        model_info: Optional[Dict[str, Any]] = None,
        available_providers: Optional[Sequence[str]] = None,
    ):
        """
        Initialize the session manager.
        
        Args:
            model_dir: Directory containing split ONNX components
            config: Session configuration (uses defaults if None)
            model_info: Pre-loaded model_info.json dict (loads from file if None)
            available_providers: Provider names this process can run. The
                default is ``ort.get_available_providers()``.
        """
        self.model_dir = model_dir
        self.config = config or SessionConfig()
        self._available_providers = (
            list(available_providers) if available_providers is not None else None
        )
        
        # Load model info if not provided
        if model_info is None:
            info_path = os.path.join(model_dir, "model_info.json")
            if not os.path.exists(info_path):
                raise FileNotFoundError(f"model_info.json not found in {model_dir}")
            with open(info_path, 'r', encoding='utf-8') as f:
                self.model_info = json.load(f)
        else:
            self.model_info = model_info
        
        # Internal state
        self._sessions: Dict[str, ort.InferenceSession] = {}
        self._session_info: Dict[str, SessionInfo] = {}
        self._lock = threading.Lock()
        self._policy_signature: Optional[Tuple[Any, ...]] = None
        self.execution_plan: Optional[ExecutionPlan] = None
        self._resolved_providers: Dict[str, str] = {}
        self._static_batch: Optional[int] = None
        self._static_seq: Optional[int] = None
        self._warmed_sessions: set[str] = set()
        self._io_manager: Any = None
        
        # Pre-populate session info (without loading)
        self._discover_sessions()
        self._install_available_providers()
        self._enforce_execution_policy()

    def _install_available_providers(self) -> None:
        """
        Drop providers this ONNX Runtime build cannot load.

        The profile and the NPU policy then see the providers that can
        actually run. A later edit to ``config.providers`` is filtered again
        on the next load.
        """
        available = self._available_providers
        if available is None:
            available = ort.get_available_providers()
        previous = [_provider_name(provider) for provider in self.config.providers]
        option_list = self.config.provider_options
        if option_list is not None and len(option_list) != len(previous):
            raise ValueError(
                "provider_options and providers must be the same length, "
                f"got {len(option_list)} and {len(previous)}"
            )
        if not self.config.use_available_providers_only:
            self.provider_profile = self.config.active_profile(
                cache_dir=os.path.join(self.model_dir, ".ort_cache")
            )
            return
        kept, dropped = filter_session_providers(self.config.providers, available)
        kept_names = [_provider_name(provider) for provider in kept]
        if option_list is not None:
            available_names = {_provider_name(provider) for provider in available}
            kept_options = [
                options
                for provider, options in zip(self.config.providers, option_list)
                if _provider_name(provider) in available_names
            ]
            self.config.provider_options = kept_options or [{}]
        if dropped:
            warnings.warn(
                "Dropped unavailable execution providers: "
                + ", ".join(dropped)
                + ". Using "
                + ", ".join(kept_names)
                + ".",
                RuntimeWarning,
                stacklevel=4,
            )
        self.config.providers = kept
        self.provider_profile = self.config.active_profile(
            cache_dir=os.path.join(self.model_dir, ".ort_cache")
        )
        if previous != kept_names:
            self._policy_signature = None
    
    def _discover_sessions(self) -> None:
        """Discover all available sessions from model_info without loading them."""
        components = self.model_info.get("components", {})
        
        for name, comp_info in components.items():
            filename = comp_info.get("file", "")
            filepath = os.path.join(self.model_dir, filename)
            
            file_size = 0
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
            
            declared_phase = comp_info.get("phase")
            phase = (
                _coerce_phase(declared_phase)
                if declared_phase not in (None, "")
                else infer_component_phase(name)
            )
            preference = _coerce_device_preference(comp_info.get("device_preference"))
            group = comp_info.get("execution_group")
            if group is None or str(group).strip() == "":
                group = None
            else:
                group = str(group)
            device_type = None
            if "provider" in comp_info:
                device_type = device_type_for_provider(comp_info["provider"])
            if preference not in (None, "auto"):
                device_type = preference
            self._session_info[name] = SessionInfo(
                name=name,
                file_path=filepath,
                file_size_bytes=file_size,
                is_loaded=False,
                input_names=list(comp_info.get("input_names", [])),
                output_names=list(comp_info.get("output_names", [])),
                input_shapes=_copy_shape_map(comp_info.get("input_shapes")),
                output_shapes=_copy_shape_map(comp_info.get("output_shapes")),
                input_dtypes=_copy_dtype_map(comp_info.get("input_dtypes")),
                output_dtypes=_copy_dtype_map(comp_info.get("output_dtypes")),
                phase=phase,
                execution_group=group,
                device_preference=preference,
                resident_policy=_coerce_resident_policy(
                    comp_info.get("resident_policy", "auto")
                ),
                device_type=device_type,
                device_affinity=(
                    preference if preference not in (None, "auto") else device_type
                ),
            )
        self._assign_execution_order()
    
    def provider_assignment(
        self,
        name: str,
        providers: Optional[Sequence[Any]] = None,
        provider_options: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> ProviderAssignment:
        """
        Resolve the provider list and option dicts for one component.

        An explicit ``providers`` argument wins. Otherwise a plan step or the
        component record supplies the device. A component with neither keeps
        the manager list. Embeddings on that default list stay on CPU. A
        layer on VitisAI requests the LLM MatMul passes. ``lm_head`` keeps
        the provider recorded for it, including DirectML. The two returned
        lists have the same length.
        """
        if providers is None and name not in self._session_info:
            raise KeyError(f"Session '{name}' is not in this model")
        if self.config.use_available_providers_only:
            available = self._available_providers
            if available is None:
                available = ort.get_available_providers()
        else:
            available = None
        cache_dir = self.provider_profile.cache_dir
        pinned = True
        explicit_options: Dict[str, Any] = {}
        if providers is not None:
            explicit_options = _first_explicit_options(provider_options)
            assignment = build_provider_assignment(
                providers,
                provider_options,
                available=available,
                cache_dir=cache_dir,
                enable_provider_cache=self.config.enable_provider_cache,
                append_cpu=False,
                npu_graph_capture=self.config.npu_graph_capture,
                npu_compilation_mode=self.config.npu_compilation_mode,
            )
        else:
            request = self._component_provider_request(name)
            if request is None:
                pinned = False
                assignment = self._default_provider_assignment()
            else:
                requested, options, append_cpu = request
                explicit_options = _first_explicit_options(options)
                assignment = build_provider_assignment(
                    requested,
                    options,
                    available=available,
                    cache_dir=cache_dir,
                    enable_provider_cache=self.config.enable_provider_cache,
                    append_cpu=append_cpu,
                    npu_graph_capture=self.config.npu_graph_capture,
                    npu_compilation_mode=self.config.npu_compilation_mode,
                )
        config_dir = cache_dir or os.path.join(self.model_dir, ".ort_cache")
        assignment = specialize_component_assignment(
            name,
            assignment,
            pinned=pinned,
            explicit_options=explicit_options,
            config_dir=config_dir,
        )
        if assignment.dropped:
            warnings.warn(
                f"Component '{name}' dropped unavailable provider "
                + ", ".join(assignment.dropped)
                + ". Using "
                + ", ".join(assignment.providers)
                + ".",
                RuntimeWarning,
                stacklevel=3,
            )
        return assignment

    def _default_provider_assignment(self) -> ProviderAssignment:
        """The manager provider list, split into names and option dicts."""
        prepared = self.config.ort_providers(cache_dir=self.provider_profile.cache_dir)
        names: List[str] = []
        options: List[Dict[str, str]] = []
        for entry in prepared:
            if isinstance(entry, tuple):
                names.append(str(entry[0]))
                options.append(dict(entry[1]))
            else:
                names.append(str(entry))
                options.append({})
        if not names:
            names = ["CPUExecutionProvider"]
            options = [{}]
        return ProviderAssignment(
            providers=names,
            provider_options=options,
            profile=self.config.active_profile(cache_dir=self.provider_profile.cache_dir),
        )

    def _component_provider_request(
        self,
        name: str,
    ) -> Optional[Tuple[List[Any], Optional[List[Any]], bool]]:
        """
        Provider request for one component.

        The third value is true when a single device should keep CPU as a
        fallback. None means the component uses the manager list.
        """
        if self.execution_plan is not None:
            for step in self.execution_plan.steps:
                if step.component_name != name:
                    continue
                options = None
                if step.provider_options:
                    options = [dict(step.provider_options)]
                return [step.provider], options, True
        component = self.model_info.get("components", {}).get(name, {})
        raw_options = component.get("provider_options")
        if "providers" in component:
            requested = list(component["providers"])
            if raw_options is None:
                return requested, None, False
            if isinstance(raw_options, dict):
                padded = [raw_options] + [{}] * max(0, len(requested) - 1)
                return requested, padded, False
            return requested, list(raw_options), False
        if "provider" in component:
            options = None
            if isinstance(raw_options, dict):
                options = [raw_options]
            elif isinstance(raw_options, list):
                options = list(raw_options)
            return [component["provider"]], options, True
        info = self._session_info.get(name)
        if info is not None and info.device_preference not in (None, "auto"):
            return self._providers_for_preference(info.device_preference), None, True
        return None

    def _providers_for_preference(self, preference: str) -> List[str]:
        """
        Providers for a device preference, limited to ones this process can run.

        A provider that is both available and listed on the session config
        wins. Otherwise the first available provider for that device is used.
        CPU is appended when it can run.
        """
        selected = _coerce_device_preference(preference)
        if selected in (None, "auto"):
            return [_provider_name(provider) for provider in self.config.providers]
        available = self._available_providers
        if available is None:
            available = ort.get_available_providers()
        available_names = {_provider_name(provider) for provider in available}
        configured = {_provider_name(provider) for provider in self.config.providers}
        ordered = _DEVICE_PROVIDER_ORDER[selected]
        chosen = [
            name for name in ordered
            if name in available_names and name in configured
        ]
        if not chosen:
            chosen = [name for name in ordered if name in available_names]
        if not chosen:
            warnings.warn(
                f"No {selected} execution provider is available. "
                "Falling back to CPU.",
                RuntimeWarning,
                stacklevel=3,
            )
            if "CPUExecutionProvider" in available_names:
                return ["CPUExecutionProvider"]
            raise RuntimeError(f"No execution provider is available for '{selected}'")
        if (
            "CPUExecutionProvider" not in chosen
            and "CPUExecutionProvider" in available_names
        ):
            chosen.append("CPUExecutionProvider")
        return chosen

    def _load_single_session(
        self,
        name: str,
        providers: Optional[List[str]] = None,
        provider_options: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[ort.InferenceSession]:
        """
        Load a single ONNX session by component name.

        Args:
            name: Component name from model_info
            providers: Provider names for this component. The plan, the
                component record, or the manager list is used when omitted.
            provider_options: One option dict per provider. Session hints
                such as CPU thread counts stay on the profile. Execution-provider
                keys are forwarded to ONNX Runtime.

        Returns:
            Loaded InferenceSession or None if failed
        """
        if name not in self._session_info:
            return None
        
        info = self._session_info[name]
        assignment = self.provider_assignment(name, providers, provider_options)
        if assignment.providers:
            info.scheduled_provider = assignment.providers[0]
        
        if not os.path.exists(info.file_path):
            return None
        
        try:
            self._install_available_providers()
            self._enforce_execution_policy()
            self._prepare_profile_cache()
            component_providers = list(assignment.providers)
            component_provider_options = [
                dict(options) for options in assignment.provider_options
            ]
            opts = self.config.create_session_options(
                profile=assignment.profile,
                providers=component_providers,
            )
            session = ort.InferenceSession(
                info.file_path,
                sess_options=opts,
                providers=component_providers,
                provider_options=component_provider_options,
            )
            
            # Update session info with loaded details
            info.is_loaded = True
            info.input_names = [inp.name for inp in session.get_inputs()]
            info.output_names = [out.name for out in session.get_outputs()]
            info.input_types = {inp.name: inp.type for inp in session.get_inputs()}
            info.output_types = {out.name: out.type for out in session.get_outputs()}
            info.input_shapes = {inp.name: _as_shape(inp.shape) for inp in session.get_inputs()}
            info.output_shapes = {out.name: _as_shape(out.shape) for out in session.get_outputs()}
            info.input_dtypes = {inp.name: _dtype_name(inp.type) for inp in session.get_inputs()}
            info.output_dtypes = {out.name: _dtype_name(out.type) for out in session.get_outputs()}
            
            # Determine which provider is actually being used
            try:
                providers = session.get_providers()
                info.provider_used = providers[0] if providers else "Unknown"
            except Exception:
                info.provider_used = "Unknown"
            
            return session
            
        except Exception as e:
            print(f"  [SessionManager] Failed to load {name}: {e}")
            return None
    
    def load_all(self) -> Dict[str, SessionInfo]:
        """
        Load all ONNX sessions at once (eager loading).
        
        This is the recommended approach for NPU optimization.
        Loads all sessions and keeps them resident in memory.
        
        Returns:
            Dict of component name to SessionInfo
        """
        with self._lock:
            self._install_available_providers()
            self._enforce_execution_policy()
            loaded_count = 0
            failed_count = 0
            total_size = 0
            
            for name in self._session_info:
                if name not in self._sessions:
                    session = self._load_single_session(name)
                    if session:
                        self._sessions[name] = session
                        loaded_count += 1
                        total_size += self._session_info[name].file_size_bytes
                    else:
                        failed_count += 1
            
            print(f"  [SessionManager] Loaded {loaded_count} sessions "
                  f"({total_size / (1024*1024):.1f} MB), {failed_count} failed")
            
            return dict(self._session_info)
    
    def get_session(self, name: str) -> Optional[ort.InferenceSession]:
        """
        Get a session by component name.
        
        If session is not loaded and lazy loading is acceptable,
        it will be loaded on demand.
        
        Args:
            name: Component name (e.g., "layer_0", "embedding")
            
        Returns:
            Loaded InferenceSession or None
        """
        with self._lock:
            if name in self._sessions:
                return self._sessions[name]

            self._install_available_providers()
            self._enforce_execution_policy()
            # Lazy load if not found
            session = self._load_single_session(name)
            if session:
                self._sessions[name] = session
            return session

    def warmup_session(
        self,
        name: str,
        dummy_inputs: Dict[str, np.ndarray],
        iterations: int = 1,
    ) -> WarmupResult:
        """
        Execute a loaded session so the first-run cost is paid before timing.

        On an NPU that cost is graph compilation, XCLBIN generation, memory
        planning, operator fallback, and cache initialization. The runs are
        not a latency result. ``iterations`` defaults to one pass.
        """
        if iterations < 0:
            raise ValueError(f"warmup iterations must be >= 0, got {iterations}")
        session = self._require_session(name)
        _validate_dummy_inputs(session, dummy_inputs)
        if iterations == 0:
            return WarmupResult(name, 0, 0.0)
        started = time.perf_counter()
        for _ in range(iterations):
            session.run(None, dummy_inputs)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._warmed_sessions.add(name)
        self._sync_runtime_flags(name)
        return WarmupResult(name, iterations, elapsed_ms)

    def benchmark_session(
        self,
        name: str,
        dummy_inputs: Dict[str, np.ndarray],
        exclude_first_run: bool = True,
        warmup_iterations: int = 2,
        measured_iterations: int = 5,
    ) -> BenchmarkResult:
        """
        Measure steady-state latency after NPU first-run setup.

        Defaults match a fair comparison with a precompiled runtime:
        ``exclude_first_run=True``, ``warmup_iterations=2``,
        ``measured_iterations=5``. The cold run is stored in
        ``first_run_ms`` and is not part of the measured mean. Warmup runs
        are also left out. A session that was already warmed does not
        discard another sample.
        """
        if warmup_iterations < 0:
            raise ValueError(
                f"warmup_iterations must be >= 0, got {warmup_iterations}"
            )
        if measured_iterations < 1:
            raise ValueError(
                f"measured_iterations must be >= 1, got {measured_iterations}"
            )
        session = self._require_session(name)
        _validate_dummy_inputs(session, dummy_inputs)
        first_run_ms = None
        cold_run_excluded = False
        if exclude_first_run and name not in self._warmed_sessions:
            cold = self.warmup_session(name, dummy_inputs, iterations=1)
            first_run_ms = cold.elapsed_ms
            cold_run_excluded = True
        if warmup_iterations:
            self.warmup_session(name, dummy_inputs, iterations=warmup_iterations)
        latencies_ms: List[float] = []
        for _ in range(measured_iterations):
            started = time.perf_counter()
            session.run(None, dummy_inputs)
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
        self._warmed_sessions.add(name)
        self._sync_runtime_flags(name)
        mean_ms, median_ms, min_ms, max_ms, stdev_ms = _latency_stats(latencies_ms)
        return BenchmarkResult(
            component_name=name,
            exclude_first_run=exclude_first_run,
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
            cold_run_excluded=cold_run_excluded,
            first_run_ms=first_run_ms,
            latencies_ms=latencies_ms,
            mean_ms=mean_ms,
            median_ms=median_ms,
            min_ms=min_ms,
            max_ms=max_ms,
            stdev_ms=stdev_ms,
        )

    def _require_session(self, name: str) -> Any:
        """Return a loaded session, or raise when this component cannot run."""
        session = self.get_session(name)
        if session is None:
            raise RuntimeError(f"Session '{name}' is not loaded")
        return session
    
    def unload_session(self, name: str) -> bool:
        """
        Unload a specific session to free memory.
        
        Args:
            name: Component name to unload
            
        Returns:
            True if session was unloaded
        """
        with self._lock:
            return self._drop_session(name)

    def _drop_session(self, name: str) -> bool:
        """Unload one session. Caller holds ``self._lock``."""
        removed = name in self._sessions
        if removed:
            del self._sessions[name]
        info = self._session_info.get(name)
        if info is not None:
            info.is_loaded = False
            info.provider_used = None
            info.warmed_up = False
            info.compiled = False
        self._warmed_sessions.discard(name)
        return removed
    
    def unload_all(self) -> int:
        """
        Unload all sessions.
        
        Returns:
            Number of sessions unloaded
        """
        with self._lock:
            count = len(self._sessions)
            self._sessions.clear()
            self._warmed_sessions.clear()
            for info in self._session_info.values():
                info.is_loaded = False
                info.provider_used = None
                info.warmed_up = False
                info.compiled = False
            return count
    
    def get_session_info(self, name: str) -> Optional[SessionInfo]:
        """Get information about a session."""
        return self._session_info.get(name)
    
    def get_all_session_info(self) -> Dict[str, SessionInfo]:
        """Get information about all sessions."""
        return dict(self._session_info)
    
    def get_loaded_sessions(self) -> List[str]:
        """Get list of currently loaded session names."""
        with self._lock:
            return list(self._sessions.keys())
    
    def estimate_runtime_memory(self, name: str) -> RuntimeMemoryEstimate:
        """
        Estimate the memory one component keeps resident at runtime.

        The five terms are weights, the compiled graph, activations, the
        KV cache this component owns, and the provider workspace. A declared
        size or an on-disk external-data or compile-cache file replaces the
        heuristic for that term.
        """
        info = self._session_info.get(name)
        if info is None:
            raise KeyError(f"Session '{name}' is not in this model")
        component = self.model_info.get("components", {}).get(name, {})
        batch, seq = self._estimate_limits()
        profile = self._component_profile(name)
        device = self._memory_device(name, profile.name)
        weights_mb = self._weights_mb(info, component)
        return RuntimeMemoryEstimate(
            component_name=name,
            weights_mb=weights_mb,
            compiled_graph_mb=self._compiled_graph_mb(
                name, component, profile, device, weights_mb
            ),
            activation_mb=self._activation_mb(name, component, batch, seq),
            kv_cache_mb=self._component_kv_cache_mb(name, component, batch, seq),
            provider_workspace_mb=self._workspace_mb(
                component, profile, device, weights_mb
            ),
        )

    def estimate_runtime_memory_mb(self, name: str) -> float:
        """
        Return ``weights + compiled graph + activations + KV + workspace`` in MiB.

        Compare :meth:`estimate_model_runtime_memory_mb` with the NPU budget
        before choosing a monolithic session. File size alone is the weight
        term, and only when the weights are embedded in the ONNX file.
        """
        return self.estimate_runtime_memory(name).total_mb

    def estimate_model_runtime_memory_mb(self) -> float:
        """
        Estimate the whole model.

        Weights, compiled graphs, activations, and workspaces add across
        components. The KV pool is added once, because prefill and decode
        bind the same cache.
        """
        estimates = [
            self.estimate_runtime_memory(name) for name in self._session_info
        ]
        resident = sum(
            item.weights_mb
            + item.compiled_graph_mb
            + item.activation_mb
            + item.provider_workspace_mb
            for item in estimates
        )
        return _round_mb(resident + self._model_kv_cache_mb())

    def fits_npu_monolithic(self, budget_mb: Optional[float] = None) -> bool:
        """
        True when the model estimate fits in the NPU budget.

        ``budget_mb`` wins. Otherwise ``model_info['npu_memory_mb']`` or the
        active profile's ``memory_limit_mb`` is the budget. A model that does
        not fit has to use a hybrid phase split or MoE route instead of one
        monolithic NPU session.
        """
        return self.estimate_model_runtime_memory_mb() <= self._npu_budget_mb(budget_mb)

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.

        ``total_mb`` is the ONNX file size of loaded sessions.
        ``total_runtime_mb`` is the five-term model estimate.
        """
        with self._lock:
            per_session = {}
            total_bytes = 0
            loaded_count = 0
            
            for name, info in self._session_info.items():
                if info.is_loaded:
                    per_session[name] = info.file_size_bytes
                    total_bytes += info.file_size_bytes
                    loaded_count += 1
            runtime = {
                name: self.estimate_runtime_memory_mb(name)
                for name in self._session_info
            }
            
            return {
                "total_bytes": total_bytes,
                "total_mb": round(total_bytes / (1024 * 1024), 2),
                "loaded_count": loaded_count,
                "total_count": len(self._session_info),
                "per_session_bytes": per_session,
                "per_session_runtime_mb": runtime,
                "total_runtime_mb": self.estimate_model_runtime_memory_mb(),
            }
    
    def get_component_names(self) -> List[str]:
        """
        Get component names in execution order.

        An active execution plan supplies dependency order. Otherwise the
        prefix heuristic orders embedding, layers, norm, and lm_head.
        """
        if self.execution_plan is not None:
            return [
                step.component_name
                for step in self.execution_plan.ordered_steps()
            ]
        # Try to return in logical order: other, embedding, layers, norm, lm_head
        names = list(self._session_info.keys())
        ordered = []
        
        # Priority order
        for prefix in ["other", "embedding", "layer_", "norm", "lm_head"]:
            for name in names:
                if name.startswith(prefix) or name == prefix.rstrip("_"):
                    if name not in ordered:
                        ordered.append(name)
        
        # Add any remaining
        for name in names:
            if name not in ordered:
                ordered.append(name)
        
        return ordered

    def filter_available_providers(self) -> List[str]:
        """
        Return the providers this manager will pass to ONNX Runtime.

        When ``use_available_providers_only`` is true, names missing from
        this process are removed and CPU is kept as the fallback. When it
        is false, the configured list is returned unchanged.
        """
        if not self.config.use_available_providers_only:
            return [_provider_name(provider) for provider in self.config.providers]
        self._install_available_providers()
        return [_provider_name(provider) for provider in self.config.providers]

    def load_component(
        self,
        name: str,
        providers: Optional[List[str]] = None,
        provider_options: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[ort.InferenceSession]:
        """
        Load one component.

        ``providers`` and ``provider_options`` override the phase, the plan,
        and the manager list for this call. A component that is already
        loaded is returned as-is unless a provider list is passed.
        """
        if name not in self._session_info:
            raise KeyError(f"Session '{name}' is not in this model")
        with self._lock:
            if providers is None and name in self._sessions:
                self._sync_runtime_flags(name)
                return self._sessions[name]
            session = self._load_single_session(name, providers, provider_options)
            if session is not None:
                self._sessions[name] = session
            self._sync_runtime_flags(name)
            return session

    def warmup_component(
        self,
        name: str,
        dummy_inputs: Dict[str, np.ndarray],
    ) -> WarmupResult:
        """
        Compile and warm one loaded component.

        This is the single-pass form of ``warmup_session``. It marks the
        component warmed and compiled so a later benchmark can skip the
        cold run. ``config.warmup`` records that this step is required
        before the component is treated as ready.
        """
        return self.warmup_session(name, dummy_inputs, iterations=1)

    def get_execution_order(self) -> List[str]:
        """
        Return component names in run order and store that index.

        A loaded plan supplies dependency order. Otherwise embedding, layers,
        norm, and lm_head come first.
        """
        return self._assign_execution_order()

    def estimate_memory_plan(self) -> Dict[str, Any]:
        """
        Estimate runtime memory for every component in execution order.

        Each ``SessionInfo.estimated_runtime_memory_mb`` is updated. The
        model total counts the shared KV pool once. ``memory_policy`` is the
        resolved mode, not the raw ``auto`` value.
        """
        order = self.get_execution_order()
        components: Dict[str, Any] = {}
        for name in order:
            estimate = self.estimate_runtime_memory(name)
            info = self._session_info[name]
            info.estimated_runtime_memory_mb = estimate.total_mb
            self._sync_runtime_flags(name)
            components[name] = {
                "phase": info.phase,
                "execution_order": info.execution_order,
                "device_affinity": info.device_affinity,
                "resident_policy": info.resolved_resident_policy(),
                "estimated_runtime_memory_mb": estimate.total_mb,
                "weights_mb": estimate.weights_mb,
                "compiled_graph_mb": estimate.compiled_graph_mb,
                "activation_mb": estimate.activation_mb,
                "kv_cache_mb": estimate.kv_cache_mb,
                "provider_workspace_mb": estimate.provider_workspace_mb,
                "compiled": info.compiled,
                "warmed_up": info.warmed_up,
            }
        return {
            "memory_policy": self.config.resolved_memory_policy(),
            "execution_policy": self.config.resolved_execution_policy().value,
            "static_shapes": self.config.static_shapes,
            "warmup": self.config.warmup,
            "order": order,
            "components": components,
            "total_runtime_mb": self.estimate_model_runtime_memory_mb(),
            "kv_cache_mb": self._model_kv_cache_mb(),
        }

    def attach_io_binding_manager(self, io_manager: Any) -> Any:
        """
        Keep an IO binding manager and apply ``memory_policy`` to its mode.

        Later ``prepare_io_binding`` calls can omit the manager and use this
        one. Symbolic shapes are frozen when ``static_shapes`` is true.
        """
        if io_manager is None:
            raise ValueError("io_manager is required")
        self._io_manager = io_manager
        mode_name = self.config.resolved_memory_policy()
        mapping = {
            "minimal": "MINIMAL",
            "persistent": "PERSISTENT",
            "io_bound": "IO_BOUND",
            "npu_static": "NPU_STATIC",
        }
        if mode_name in mapping and (
            hasattr(io_manager, "memory_mode") or hasattr(io_manager, "mode")
        ):
            from .io_binding import MemoryMode
            resolved = getattr(MemoryMode, mapping[mode_name])
            if hasattr(io_manager, "memory_mode"):
                io_manager.memory_mode = resolved
            if hasattr(io_manager, "mode"):
                io_manager.mode = resolved
        return io_manager

    def _assign_execution_order(self) -> List[str]:
        """Write ``execution_order`` from the current component order."""
        names = self.get_component_names()
        for index, name in enumerate(names):
            info = self._session_info.get(name)
            if info is not None:
                info.execution_order = index
        return names

    def _sync_runtime_flags(self, name: str) -> None:
        """Copy warmup, compile, and device affinity onto SessionInfo."""
        info = self._session_info.get(name)
        if info is None:
            return
        info.warmed_up = name in self._warmed_sessions
        if info.warmed_up:
            info.compiled = True
        elif not info.is_loaded:
            info.compiled = False
        if info.device_preference not in (None, "auto"):
            info.device_affinity = info.device_preference
        elif info.device_type:
            info.device_affinity = info.device_type
        elif info.scheduled_provider:
            info.device_affinity = device_type_for_provider(info.scheduled_provider)

    def load_from_plan(self, plan: ExecutionPlan) -> Dict[str, SessionInfo]:
        """
        Schedule and load sessions from an execution plan.

        Resident steps are loaded in dependency order with that step's
        provider. Non-resident steps are unloaded. The plan's device and
        phase are recorded on each SessionInfo.
        """
        if not isinstance(plan, ExecutionPlan):
            raise TypeError("load_from_plan expects an ExecutionPlan")
        with self._lock:
            plan.validate()
            for step in plan.steps:
                if step.component_name not in self._session_info:
                    raise KeyError(
                        f"Execution plan names '{step.component_name}', "
                        "which is not in this model."
                    )
            self._install_available_providers()
            self._enforce_execution_policy()
            self._validate_plan_policy(plan)
            ordered = plan.ordered_steps()
            self.execution_plan = plan
            self._resolved_providers = {}
            loaded: Dict[str, SessionInfo] = {}
            for step in ordered:
                info = self._session_info[step.component_name]
                info.phase = step.phase
                info.device_type = step.device_type
                info.resident = step.resident
                if step.execution_group is not None:
                    info.execution_group = step.execution_group
                if step.device_preference is not None:
                    info.device_preference = step.device_preference
                if step.resident_policy != "auto":
                    info.resident_policy = step.resident_policy
                elif not step.resident:
                    info.resident_policy = "on_demand"
                if not step.resident:
                    assignment = self.provider_assignment(step.component_name)
                    info.scheduled_provider = (
                        assignment.providers[0] if assignment.providers else step.provider
                    )
                    self._resolved_providers[step.component_name] = info.scheduled_provider
                    self._drop_session(step.component_name)
                    self._sync_runtime_flags(step.component_name)
                    loaded[step.component_name] = info
                    continue
                session = self._load_single_session(step.component_name)
                self._resolved_providers[step.component_name] = (
                    info.scheduled_provider or step.provider
                )
                if session is not None:
                    self._sessions[step.component_name] = session
                self._sync_runtime_flags(step.component_name)
                loaded[step.component_name] = info
            self._assign_execution_order()
            return loaded

    def execution_order(self) -> List[ExecutionStep]:
        """Return the active plan in dependency order."""
        if self.execution_plan is None:
            raise RuntimeError("No execution plan is loaded")
        return self.execution_plan.ordered_steps()

    def sessions_for_phase(self, phase: str) -> List[ExecutionStep]:
        """Return active-plan steps for one phase, in run order."""
        if self.execution_plan is None:
            raise RuntimeError("No execution plan is loaded")
        return self.execution_plan.steps_for_phase(phase)

    def _sessions_in_phase(
        self,
        phase: str,
        execution_group: Optional[str] = None,
    ) -> List[str]:
        """Component names in one phase, in plan order then discovery order."""
        selected = _coerce_phase(phase)
        names: List[str] = []
        if self.execution_plan is not None:
            for step in self.execution_plan.ordered_steps():
                info = self._session_info.get(step.component_name)
                if info is None or info.phase != selected:
                    continue
                if execution_group is not None and info.execution_group != execution_group:
                    continue
                names.append(step.component_name)
        seen = set(names)
        for name, info in self._session_info.items():
            if name in seen or info.phase != selected:
                continue
            if execution_group is not None and info.execution_group != execution_group:
                continue
            names.append(name)
        return names

    def load_phase(
        self,
        phase: str,
        execution_group: Optional[str] = None,
    ) -> Dict[str, SessionInfo]:
        """
        Load every session that belongs to one phase.

        Resident sessions from other phases stay loaded. On-demand sessions
        from other phases are unloaded, which is how experts, drafts, and
        verification stay off the NPU until that phase runs. ``auto`` uses
        the phase default. An optional ``execution_group`` limits the load
        to one group inside the phase.
        """
        selected = _coerce_phase(phase)
        members = self._sessions_in_phase(selected, execution_group)
        if not members:
            group = f" in group '{execution_group}'" if execution_group else ""
            raise ValueError(f"No sessions are assigned to phase '{selected}'{group}")
        with self._lock:
            self._install_available_providers()
            self._enforce_execution_policy()
            member_set = set(members)
            for name, info in self._session_info.items():
                if name in member_set or info.phase is None:
                    continue
                if info.resolved_resident_policy() in {"on_demand", "evict"}:
                    self._drop_session(name)
                    info.resident = False
            loaded: Dict[str, SessionInfo] = {}
            loaded_count = 0
            for name in members:
                info = self._session_info[name]
                policy = info.resolved_resident_policy()
                if policy == "evict":
                    self._drop_session(name)
                    info.resident = False
                    self._sync_runtime_flags(name)
                    loaded[name] = info
                    continue
                info.resident = policy == "resident"
                if name in self._sessions:
                    if info.device_preference not in (None, "auto"):
                        info.device_type = info.device_preference
                    self._sync_runtime_flags(name)
                    loaded[name] = info
                    loaded_count += 1
                    continue
                session = self._load_single_session(name)
                if info.device_preference not in (None, "auto"):
                    info.device_type = info.device_preference
                elif info.scheduled_provider:
                    info.device_type = device_type_for_provider(info.scheduled_provider)
                if info.scheduled_provider:
                    self._resolved_providers[name] = info.scheduled_provider
                if session is not None:
                    self._sessions[name] = session
                    loaded_count += 1
                self._sync_runtime_flags(name)
                loaded[name] = info
            print(
                f"  [SessionManager] Phase '{selected}': "
                f"{loaded_count} loaded, {len(members)} scheduled"
            )
            return loaded

    def placement(self, component_name: str) -> ExecutionStep:
        """Return the plan step for a component."""
        if self.execution_plan is None:
            raise RuntimeError("No execution plan is loaded")
        for step in self.execution_plan.steps:
            if step.component_name == component_name:
                return step
        raise KeyError(f"Component '{component_name}' is not in the execution plan")

    def resolved_provider(self, component_name: str) -> str:
        """Return the provider that will run a planned step on this machine."""
        if component_name not in self._resolved_providers:
            raise KeyError(
                f"Component '{component_name}' has no resolved provider"
            )
        return self._resolved_providers[component_name]

    def _providers_for_step(self, step: ExecutionStep) -> List[Any]:
        """
        Build the ORT provider list for one step.

        Names and option dicts stay aligned. CPU is appended when the step
        names a single device. The manager provider list is left unchanged.
        """
        assignment = self.provider_assignment(step.component_name)
        prepared: List[Any] = []
        for provider_name, options in zip(assignment.providers, assignment.provider_options):
            if options:
                prepared.append((provider_name, dict(options)))
            else:
                prepared.append(provider_name)
        return prepared

    def _validate_plan_policy(self, plan: ExecutionPlan) -> None:
        """
        Check the plan shape against the configured execution policy.

        The sequential policy cannot unload an NPU step. A monolithic plan
        can keep several resident layer sessions on the NPU. Two non-layer
        NPU steps still need their own policy. Hybrid NPU and iGPU and phase
        split need prefill and decode. MoE routing needs a router and an
        expert. Speculative draft and verify needs both of those phases.
        """
        policy = self.config.resolved_execution_policy()
        phases = {step.phase for step in plan.steps}
        npu_steps = [step for step in plan.steps if _step_targets_npu(step)]
        layer_npu = [
            step for step in npu_steps if _LAYER_COMPONENT.search(step.component_name)
        ]
        if policy == ExecutionPolicy.SEQUENTIAL and npu_steps:
            raise RuntimeError(NPU_SEQUENTIAL_SPLIT_ERROR)
        if (
            policy == ExecutionPolicy.MONOLITHIC
            and len(npu_steps) > 1
            and len(layer_npu) != len(npu_steps)
        ):
            raise RuntimeError(NPU_SEQUENTIAL_SPLIT_ERROR)
        if policy in (
            ExecutionPolicy.HYBRID_NPU_IGPU,
            ExecutionPolicy.PHASE_SPLIT,
        ) and (
            "prefill" not in phases or "decode" not in phases
        ):
            raise ValueError(
                f"{policy.value} requires a prefill step and a decode step."
            )
        if policy == ExecutionPolicy.MOE_ROUTING and (
            "router" not in phases or "expert" not in phases
        ):
            raise ValueError(
                f"{policy.value} requires a router step and an expert step."
            )
        if policy == ExecutionPolicy.SPECULATIVE and (
            "speculative_draft" not in phases or "verification" not in phases
        ):
            raise ValueError(
                "speculative requires a speculative_draft step and a verification step."
            )

    def get_io_metadata(self, name: str) -> IOMetadata:
        """
        Return the input and output tensors for one component.

        A loaded session supplies the shapes. Otherwise they come from the
        component record in ``model_info``.
        """
        info = self._session_info.get(name)
        if info is None:
            raise KeyError(f"Session '{name}' is not in this model")
        component = self.model_info.get("components", {}).get(name, {})
        default_dtype = _dtype_name(
            component.get("dtype") or self.model_info.get("dtype") or "float16"
        )
        inputs = self._tensor_list(
            info.input_names,
            info.input_shapes or _copy_shape_map(component.get("input_shapes")),
            info.input_dtypes or _copy_dtype_map(component.get("input_dtypes")),
            default_dtype,
        )
        outputs = self._tensor_list(
            info.output_names,
            info.output_shapes or _copy_shape_map(component.get("output_shapes")),
            info.output_dtypes or _copy_dtype_map(component.get("output_dtypes")),
            default_dtype,
        )
        return IOMetadata(
            component_name=name,
            inputs=inputs,
            outputs=outputs,
            phase=info.phase,
            device_type=info.device_type,
            resident=info.resident,
        )

    def prepare_io_binding(
        self,
        component_name: str,
        io_manager: Any = None,
        static_shapes: Optional[bool] = None,
    ) -> PreparedIO:
        """
        Allocate this component's inputs and outputs on ``io_manager``.

        ``static_shapes`` replaces symbolic batch and sequence axes with the
        static batch and ``max_seq_len``. None uses ``config.static_shapes``.
        An omitted manager uses the one passed to ``attach_io_binding_manager``.
        An NPU output that a later iGPU step consumes is allocated once and
        reused. When the session is loaded, those buffers are bound to it.
        """
        if io_manager is None:
            io_manager = self._io_manager
        if io_manager is None:
            raise ValueError(
                "Pass io_manager or call attach_io_binding_manager first"
            )
        if static_shapes is None:
            static_shapes = self.config.static_shapes
        metadata = self.get_io_metadata(component_name)
        batch_size = self._static_batch or 1
        max_seq_len = self._sequence_limit(io_manager)
        buffers: Dict[str, Any] = {}
        for tensor in list(metadata.inputs) + list(metadata.outputs):
            if tensor.name in buffers:
                continue
            if not tensor.shape:
                raise ValueError(
                    f"Component '{component_name}' has no shape for '{tensor.name}'"
                )
            shape = resolve_static_shape(
                tensor.shape,
                static=static_shapes,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
            )
            if any(dim is None for dim in shape):
                raise ValueError(
                    f"Tensor '{tensor.name}' still has a dynamic dimension. "
                    "Pass static_shapes=True to freeze it for the NPU."
                )
            concrete = tuple(int(dim) for dim in shape if dim is not None)
            buffers[tensor.name] = self._place_tensor(
                io_manager,
                component_name,
                tensor.name,
                concrete,
                tensor.dtype,
            )
        binding = None
        session = self._sessions.get(component_name)
        if session is not None:
            binding = session.io_binding()
            self._bind_prepared_buffers(
                io_manager, binding, component_name, metadata, buffers
            )
        return PreparedIO(
            component_name=component_name,
            metadata=metadata,
            buffers=buffers,
            binding=binding,
        )

    def allocate_static_buffers(
        self,
        io_manager: Any,
        max_seq_len: int,
        batch_size: int = 1,
    ) -> StaticBufferAllocation:
        """
        Preallocate the NPU KV cache and the resident activations.

        KV tensors stay at ``max_seq_len``. Hidden states named by resident
        plan steps stay allocated. An NPU output consumed by an iGPU step
        is one shared buffer, so the decode input is the prefill output.
        """
        if max_seq_len <= 0 or batch_size <= 0:
            raise ValueError(
                f"max_seq_len and batch_size must be positive, "
                f"got {max_seq_len} and {batch_size}"
            )
        self._static_seq = int(max_seq_len)
        self._static_batch = int(batch_size)
        architecture = self._architecture()
        kv_target = self._static_kv_target(io_manager)
        kv_pool = kv_target.allocate_static_kv_cache(
            name="kv",
            num_layers=architecture["num_layers"],
            num_kv_heads=architecture["num_kv_heads"],
            head_dim=architecture["head_dim"],
            dtype=architecture["dtype"],
            batch_size=batch_size,
            max_seq_len=max_seq_len,
        )
        bridges: Dict[str, Any] = {}
        resident_hidden: Dict[str, Any] = {}
        for _producer, _consumer, buffer_name in self._npu_igpu_bridges():
            shape, dtype = self._activation_shape(
                buffer_name, batch_size, max_seq_len, architecture["hidden_size"]
            )
            bridges[buffer_name] = self._shared_or_device_buffer(
                io_manager, buffer_name, shape, dtype, shared=True
            )
        if self.execution_plan is not None:
            for step in self.execution_plan.steps:
                if not step.resident:
                    continue
                for buffer_name in step.output_buffers:
                    if buffer_name in bridges or buffer_name in resident_hidden:
                        continue
                    if buffer_name == "kv" or buffer_name.startswith("kv."):
                        continue
                    shape, dtype = self._activation_shape(
                        buffer_name,
                        batch_size,
                        max_seq_len,
                        architecture["hidden_size"],
                    )
                    resident_hidden[buffer_name] = self._shared_or_device_buffer(
                        io_manager,
                        buffer_name,
                        shape,
                        dtype,
                        shared=False,
                        device_type=step.device_type,
                    )
        return StaticBufferAllocation(
            max_seq_len=int(max_seq_len),
            batch_size=int(batch_size),
            kv_pool=kv_pool,
            resident_hidden=resident_hidden,
            bridges=bridges,
        )

    def _tensor_list(
        self,
        names: Sequence[str],
        shapes: Dict[str, Tuple[Any, ...]],
        dtypes: Dict[str, str],
        default_dtype: str,
    ) -> List[TensorIO]:
        selected = list(names) or list(shapes)
        tensors = []
        for tensor_name in selected:
            shape = shapes.get(tensor_name, ())
            tensors.append(TensorIO(
                name=tensor_name,
                shape=shape,
                dtype=dtypes.get(tensor_name, default_dtype),
                dynamic_axes=_dynamic_axes(shape),
            ))
        return tensors

    def _sequence_limit(self, io_manager: Any) -> int:
        if self._static_seq:
            return int(self._static_seq)
        prefill = getattr(io_manager, "prefill", None)
        limit = getattr(io_manager, "max_seq_len", None)
        if not limit and prefill is not None:
            limit = getattr(prefill, "max_seq_len", None)
        return int(limit or 4096)

    def _architecture(self) -> Dict[str, Any]:
        info = self.model_info
        num_layers = info.get("num_hidden_layers", info.get("num_layers"))
        if not num_layers:
            num_layers = sum(
                1 for name in self._session_info if _LAYER_COMPONENT.search(str(name))
            ) or 1
        heads = info.get("num_key_value_heads", info.get("num_kv_heads"))
        hidden = info.get("hidden_size")
        head_dim = info.get("head_dim")
        attention_heads = info.get("num_attention_heads")
        if not head_dim and hidden and attention_heads:
            if int(hidden) % int(attention_heads) != 0:
                raise ValueError(
                    f"hidden_size ({hidden}) is not divisible by "
                    f"num_attention_heads ({attention_heads})"
                )
            head_dim = int(hidden) // int(attention_heads)
        if not heads or not head_dim:
            raise ValueError(
                "model_info needs num_key_value_heads and head_dim, or "
                "hidden_size and num_attention_heads, before static KV allocation."
            )
        return {
            "num_layers": int(num_layers),
            "num_kv_heads": int(heads),
            "head_dim": int(head_dim),
            "hidden_size": int(hidden) if hidden else None,
            "dtype": _dtype_name(info.get("dtype") or info.get("torch_dtype") or "float16"),
        }

    def _try_architecture(self) -> Optional[Dict[str, Any]]:
        try:
            return self._architecture()
        except ValueError:
            return None

    def _estimate_limits(self) -> Tuple[int, int]:
        batch = self._static_batch or int(self.model_info.get("batch_size") or 1)
        seq = self._static_seq or self.model_info.get("max_seq_len") or (
            self.model_info.get("max_position_embeddings") or 4096
        )
        batch = int(batch)
        seq = int(seq)
        if batch <= 0 or seq <= 0:
            raise ValueError(
                f"batch_size and max_seq_len must be positive, got {batch} and {seq}"
            )
        return batch, seq

    def _component_provider(self, name: str) -> str:
        info = self._session_info[name]
        if info.scheduled_provider:
            return info.scheduled_provider
        if info.provider_used:
            return info.provider_used
        return primary_provider_name(self.config.providers)

    def _component_profile(self, name: str) -> ProviderProfile:
        provider = self._component_provider(name)
        if provider == self.provider_profile.name:
            return self.provider_profile
        profile = provider_profile_for(provider)
        if self.provider_profile.cache_dir:
            profile.cache_dir = self.provider_profile.cache_dir
        return profile

    def _memory_device(self, name: str, provider: str) -> str:
        info = self._session_info[name]
        if info.device_type == "npu" or provider_is_npu(provider):
            return "npu"
        if info.device_type == "cpu":
            return "cpu"
        return _provider_memory_class(provider, info.device_type)

    def _npu_budget_mb(self, budget_mb: Optional[float]) -> float:
        if budget_mb is not None:
            budget = float(budget_mb)
        elif self.model_info.get("npu_memory_mb"):
            budget = float(self.model_info["npu_memory_mb"])
        elif self.provider_profile.memory_limit_mb:
            budget = float(self.provider_profile.memory_limit_mb)
        else:
            raise ValueError(
                "Pass budget_mb or set model_info['npu_memory_mb'] "
                "to decide whether the model fits a monolithic NPU."
            )
        if budget <= 0:
            raise ValueError(f"NPU budget must be positive, got {budget}")
        return budget

    def _weights_mb(self, info: SessionInfo, component: Dict[str, Any]) -> float:
        declared = _declared_mb(component, "weights_mb", "weights_bytes")
        if declared is not None:
            return _round_mb(declared)
        external = self._external_weight_bytes(info, component)
        if external:
            return _round_mb(external / _MIB)
        return _round_mb(info.file_size_bytes / _MIB)

    def _external_weight_bytes(self, info: SessionInfo, component: Dict[str, Any]) -> int:
        declared = component.get("external_data_bytes")
        if declared is not None:
            return int(declared)
        paths = component.get("external_data") or []
        if isinstance(paths, str):
            paths = [paths]
        candidates = [str(path) for path in paths]
        if info.file_path:
            candidates.append(info.file_path + ".data")
            if info.file_path.endswith(".onnx"):
                candidates.append(info.file_path[:-5] + ".onnx.data")
        total = 0
        seen = set()
        for rel in candidates:
            path = rel if os.path.isabs(rel) else os.path.join(self.model_dir, rel)
            if path in seen or not os.path.isfile(path):
                continue
            seen.add(path)
            total += os.path.getsize(path)
        return total

    def _compiled_cache_bytes(self, name: str, profile: ProviderProfile) -> int:
        cache_dir = profile.cache_dir
        if not cache_dir:
            return 0
        root = os.path.join(cache_dir, name)
        if os.path.isfile(root):
            return os.path.getsize(root)
        if not os.path.isdir(root):
            return 0
        total = 0
        for dirpath, _dirs, files in os.walk(root):
            for filename in files:
                total += os.path.getsize(os.path.join(dirpath, filename))
        return total

    def _compiled_graph_mb(
        self,
        name: str,
        component: Dict[str, Any],
        profile: ProviderProfile,
        device: str,
        weights_mb: float,
    ) -> float:
        declared = _declared_mb(component, "compiled_graph_mb", "compiled_graph_bytes")
        if declared is not None:
            return _round_mb(declared)
        measured = self._compiled_cache_bytes(name, profile)
        if measured:
            return _round_mb(measured / _MIB)
        return _round_mb(max(
            COMPILED_GRAPH_FLOOR_MB[device],
            COMPILED_GRAPH_FACTOR[device] * weights_mb,
        ))

    def _workspace_mb(
        self,
        component: Dict[str, Any],
        profile: ProviderProfile,
        device: str,
        weights_mb: float,
    ) -> float:
        declared = _declared_mb(
            component, "provider_workspace_mb", "provider_workspace_bytes"
        )
        if declared is not None:
            return _round_mb(declared)
        factor = PROVIDER_WORKSPACE_FACTOR[device]
        floor = PROVIDER_WORKSPACE_FLOOR_MB[device]
        options = profile.provider_options
        if device == "cpu" and not bool(options.get("enable_cpu_mem_arena", True)):
            factor, floor = 0.01, 2.0
        if device == "cuda" and str(options.get("cudnn_conv_use_max_workspace", "0")) == "1":
            factor *= 2
        return _round_mb(max(floor, factor * weights_mb))

    def _activation_mb(
        self,
        name: str,
        component: Dict[str, Any],
        batch: int,
        seq: int,
    ) -> float:
        declared = _declared_mb(component, "activation_mb", "activation_bytes")
        if declared is not None:
            return _round_mb(declared)
        metadata = self.get_io_metadata(name)
        seen = set()
        total = 0.0
        counted = False
        for tensor in list(metadata.inputs) + list(metadata.outputs):
            if tensor.name in seen or not tensor.shape:
                continue
            seen.add(tensor.name)
            try:
                shape = resolve_static_shape(
                    tensor.shape,
                    static=True,
                    batch_size=batch,
                    max_seq_len=seq,
                )
            except ValueError:
                continue
            if any(dim is None or int(dim) <= 0 for dim in shape):
                continue
            nbytes = _element_bytes(tensor.dtype)
            for dim in shape:
                nbytes *= int(dim)
            total += nbytes
            counted = True
        if counted:
            return _round_mb(total / _MIB)
        hidden = self.model_info.get("hidden_size")
        if not hidden:
            return 0.0
        element = _element_bytes(self.model_info.get("dtype") or "float16")
        return _round_mb((2 * batch * seq * int(hidden) * element) / _MIB)

    def _kv_layers_for(self, name: str, total_layers: int) -> int:
        """
        Layers whose KV this component binds.

        A layer shard owns one layer. Prefill and decode each report the
        full pool, because both sessions bind it. The model total still
        counts that pool once.
        """
        component = self.model_info.get("components", {}).get(name, {})
        if "kv_layers" in component:
            return max(0, int(component["kv_layers"]))
        if component.get("num_layers") is not None:
            return max(0, int(component["num_layers"]))
        if _LAYER_COMPONENT.search(name):
            return 1
        if _is_aux_component(name):
            return 0
        if any(_LAYER_COMPONENT.search(other) for other in self._session_info):
            return 0
        info = self._session_info[name]
        if info.phase in {"prefill", "decode"}:
            return total_layers
        bodies = [
            other for other in self._session_info if not _is_aux_component(other)
        ]
        if len(bodies) == 1 and name in bodies:
            return total_layers
        return 0

    def _component_kv_cache_mb(
        self,
        name: str,
        component: Dict[str, Any],
        batch: int,
        seq: int,
    ) -> float:
        declared = _declared_mb(component, "kv_cache_mb", "kv_cache_bytes")
        if declared is not None:
            return _round_mb(declared)
        architecture = self._try_architecture()
        if architecture is None:
            return 0.0
        layers = self._kv_layers_for(name, architecture["num_layers"])
        return _round_mb(_kv_bytes(architecture, layers, batch, seq) / _MIB)

    def _model_kv_cache_mb(self) -> float:
        declared = _declared_mb(self.model_info, "kv_cache_mb", "kv_cache_bytes")
        if declared is not None:
            return _round_mb(declared)
        architecture = self._try_architecture()
        if architecture is None:
            return 0.0
        batch, seq = self._estimate_limits()
        return _round_mb(
            _kv_bytes(architecture, architecture["num_layers"], batch, seq) / _MIB
        )

    def _npu_igpu_bridges(self) -> List[Tuple[ExecutionStep, ExecutionStep, str]]:
        """Output buffers that leave an NPU step and enter an iGPU step."""
        if self.execution_plan is None:
            return []
        by_name = {step.component_name: step for step in self.execution_plan.steps}
        bridges = []
        for step in self.execution_plan.steps:
            if step.device_type not in {"igpu", "gpu"}:
                continue
            for dependency in step.depends_on:
                producer = by_name.get(dependency)
                if producer is None or not _step_targets_npu(producer):
                    continue
                for buffer_name in producer.output_buffers:
                    bridges.append((producer, step, buffer_name))
        return bridges

    def _activation_shape(
        self,
        buffer_name: str,
        batch_size: int,
        max_seq_len: int,
        hidden_size: Optional[int],
    ) -> Tuple[Tuple[int, ...], str]:
        """Use a declared tensor shape, otherwise a resident hidden state."""
        if self.execution_plan is not None:
            for step in self.execution_plan.steps:
                if buffer_name not in step.output_buffers and buffer_name not in (
                    tensor.name for tensor in self.get_io_metadata(step.component_name).outputs
                ):
                    continue
                metadata = self.get_io_metadata(step.component_name)
                for tensor in list(metadata.outputs) + list(metadata.inputs):
                    if tensor.name != buffer_name or not tensor.shape:
                        continue
                    shape = resolve_static_shape(
                        tensor.shape,
                        static=True,
                        batch_size=batch_size,
                        max_seq_len=max_seq_len,
                    )
                    return tuple(int(dim) for dim in shape if dim is not None), tensor.dtype
        if hidden_size:
            return (batch_size, max_seq_len, int(hidden_size)), _dtype_name(
                self.model_info.get("dtype") or "float16"
            )
        raise ValueError(
            f"No shape is declared for resident buffer '{buffer_name}'"
        )

    def _static_kv_target(self, io_manager: Any) -> Any:
        if hasattr(io_manager, "allocate_static_kv_cache"):
            return io_manager
        prefill = getattr(io_manager, "prefill", None)
        if prefill is not None and hasattr(prefill, "allocate_static_kv_cache"):
            return prefill
        raise TypeError(
            "allocate_static_buffers needs an IOBindingManager or a "
            "HeterogeneousIOBindingManager"
        )

    def _allocator_for(self, io_manager: Any, device_type: Optional[str]) -> Any:
        if device_type == "npu" and getattr(io_manager, "prefill", None) is not None:
            return io_manager.prefill
        if device_type in {"igpu", "gpu"} and getattr(io_manager, "decode", None) is not None:
            return io_manager.decode
        if hasattr(io_manager, "get_or_allocate"):
            return io_manager
        raise TypeError(
            "prepare_io_binding needs an IOBindingManager or a "
            "HeterogeneousIOBindingManager"
        )

    def _lookup_shared(self, io_manager: Any, name: str) -> Any:
        getter = getattr(io_manager, "get_shared_buffer", None)
        if getter is None:
            return None
        try:
            return getter(name)
        except KeyError:
            return None

    def _shared_or_device_buffer(
        self,
        io_manager: Any,
        name: str,
        shape: Tuple[int, ...],
        dtype: str,
        *,
        shared: bool,
        device_type: Optional[str] = None,
    ) -> Any:
        if shared and hasattr(io_manager, "allocate_shared_buffer"):
            existing = self._lookup_shared(io_manager, name)
            if existing is not None:
                return existing
            return io_manager.allocate_shared_buffer(name, shape, dtype)
        allocator = self._allocator_for(io_manager, device_type)
        existing = allocator.get_buffer(name) if hasattr(allocator, "get_buffer") else None
        if existing is not None:
            return existing
        return allocator.get_or_allocate(name, shape, dtype)

    def _shared_buffer_names(self) -> set:
        return {buffer_name for _producer, _consumer, buffer_name in self._npu_igpu_bridges()}

    def _place_tensor(
        self,
        io_manager: Any,
        component_name: str,
        tensor_name: str,
        shape: Tuple[int, ...],
        dtype: str,
    ) -> Any:
        info = self._session_info.get(component_name)
        device_type = info.device_type if info is not None else None
        if tensor_name in self._shared_buffer_names() and hasattr(io_manager, "allocate_shared_buffer"):
            return self._shared_or_device_buffer(
                io_manager, tensor_name, shape, dtype, shared=True
            )
        return self._shared_or_device_buffer(
            io_manager,
            tensor_name,
            shape,
            dtype,
            shared=False,
            device_type=device_type,
        )

    def _bind_prepared_buffers(
        self,
        io_manager: Any,
        binding: Any,
        component_name: str,
        metadata: IOMetadata,
        buffers: Dict[str, Any],
    ) -> None:
        shared_names = self._shared_buffer_names()
        input_names = {tensor.name for tensor in metadata.inputs}
        output_names = {tensor.name for tensor in metadata.outputs}
        for tensor_name, buffer in buffers.items():
            if tensor_name in shared_names and hasattr(io_manager, "bind_prefill_output"):
                if metadata.phase == "prefill" and tensor_name in output_names:
                    io_manager.bind_prefill_output(binding, tensor_name, tensor_name)
                    continue
                if metadata.phase == "decode" and tensor_name in input_names:
                    io_manager.bind_decode_input(binding, tensor_name, tensor_name)
                    continue
            ort_value = getattr(buffer, "ort_value", None)
            if ort_value is None:
                continue
            if tensor_name in input_names:
                binding.bind_ortvalue_input(tensor_name, ort_value)
            elif tensor_name in output_names:
                binding.bind_ortvalue_output(tensor_name, ort_value)

    def artifact_cache(self) -> ProviderCacheSpec:
        """
        Compile cache for the installed provider profile.

        An enabled cache with no configured directory uses
        ``<model_dir>/.ort_cache``, which ``_install_available_providers``
        already applied to ``provider_profile``.
        """
        profile = self.provider_profile.copy()
        if not self.config.enable_provider_cache:
            profile.enable_compile_cache = False
            profile.cache_dir = None
        return provider_cache_spec(
            profile,
            enabled=self.config.enable_provider_cache,
            profiling=self.config.enable_profiling,
            profile_dir=self.config.profile_dir,
        )

    def _prepare_profile_cache(self) -> None:
        """Create the compile-cache and profile directories for this load."""
        self.artifact_cache().ensure_directories()

    def _enforce_execution_policy(self) -> None:
        """
        Reject an NPU load that would unload each session.

        The same provider list and component set is checked once. A later
        change to ``config.providers`` or ``config.execution_policy`` is
        checked again on the next load. A resident layer series passes.
        """
        policy = self.config.resolved_execution_policy()
        signature = (
            policy,
            tuple(_provider_name(provider) for provider in self.config.providers),
            tuple(sorted(self._session_info)),
        )
        if signature == self._policy_signature:
            return
        self.config.validate_execution_policy(self._session_info.keys())
        self._policy_signature = signature

    def __len__(self) -> int:
        """Number of available sessions (loaded or not)."""
        return len(self._session_info)
    
    def __contains__(self, name: str) -> bool:
        """Check if a session name exists."""
        return name in self._session_info
    
    def __repr__(self) -> str:
        mem = self.get_memory_usage()
        return (f"SessionManager(sessions={len(self)}, "
                f"loaded={mem['loaded_count']}, "
                f"memory={mem['total_mb']}MB)")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_available_providers() -> List[str]:
    """
    Get list of execution providers available on this system.
    
    Returns:
        List of available provider names
    """
    return ort.get_available_providers()


def get_provider_for_device(device_type: str = "auto") -> List[str]:
    """
    Get recommended providers for a device type.
    
    Args:
        device_type: One of "auto", "npu", "gpu", "cpu"
        
    Returns:
        List of providers in priority order (always ends with CPU fallback)
    """
    available = get_available_providers()
    
    if device_type == "auto":
        # Return all available in preference order
        result = []
        for provider in SUPPORTED_PROVIDERS:
            if provider in available:
                result.append(provider)
        return result if result else ['CPUExecutionProvider']
    
    if device_type in PROVIDER_CATEGORIES:
        result = []
        for provider in PROVIDER_CATEGORIES[device_type]:
            if provider in available:
                result.append(provider)
        # Always add CPU fallback
        if 'CPUExecutionProvider' not in result:
            result.append('CPUExecutionProvider')
        return result
    
    # Unknown device type, return CPU
    return ['CPUExecutionProvider']


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Classes
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
    "RuntimeMemoryEstimate",
    "ProviderAssignment",
    "WarmupResult",
    "BenchmarkResult",
    # Functions
    "get_available_providers",
    "get_provider_for_device",
    "provider_is_npu",
    "providers_target_npu",
    "is_sequential_layer_split",
    "validate_npu_execution_policy",
    "provider_profile_for",
    "primary_provider_name",
    "filter_session_providers",
    "build_provider_assignment",
    "ep_provider_options",
    # Constants
    "SUPPORTED_PROVIDERS",
    "PROVIDER_CATEGORIES",
    "NPU_SEQUENTIAL_SPLIT_ERROR",
    "NPU_PROVIDER_MARKERS",
    "EXECUTION_PHASES",
    "MEMORY_POLICIES",
    "PHASE_ALIASES",
    "RESIDENT_POLICIES",
    "PHASE_RESIDENT_DEFAULT",
    "DEVICE_PREFERENCES",
    "infer_component_phase",
    "resolve_static_shape",
    "COMPILED_GRAPH_FACTOR",
    "COMPILED_GRAPH_FLOOR_MB",
    "PROVIDER_WORKSPACE_FACTOR",
    "PROVIDER_WORKSPACE_FLOOR_MB",
]
