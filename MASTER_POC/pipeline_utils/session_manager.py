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
import threading
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Any, Tuple, Union
import onnxruntime as ort


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
    "Sequential layer splitting on NPU causes AOT/context-switch penalties. "
    "Use monolithic, phase-split, or MoE expert-routing mode."
)


class ExecutionPolicy(Enum):
    """
    How a model is allowed to be divided across sessions.

    MONOLITHIC_ONLY is the only general NPU policy. The two hybrid values
    are the specialized exceptions: prefill on the NPU and decode on the
    iGPU, or MoE expert routing. The split values are for CPU and GPU.
    """

    MONOLITHIC_ONLY = "monolithic_only"
    SPLIT_CPU_MEMORY_OPTIMIZED = "split_cpu_memory_optimized"
    SPLIT_GPU_PERSISTENT = "split_gpu_persistent"
    HYBRID_NPU_IGPU_PHASE_SPLIT = "hybrid_npu_igpu_phase_split"
    HYBRID_NPU_MOE_ROUTING = "hybrid_npu_moe_routing"


SEQUENTIAL_LAYER_SPLIT_POLICIES = frozenset({
    ExecutionPolicy.SPLIT_CPU_MEMORY_OPTIMIZED,
    ExecutionPolicy.SPLIT_GPU_PERSISTENT,
})

HYBRID_NPU_POLICIES = frozenset({
    ExecutionPolicy.HYBRID_NPU_IGPU_PHASE_SPLIT,
    ExecutionPolicy.HYBRID_NPU_MOE_ROUTING,
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


def validate_npu_execution_policy(
    execution_policy: Union[ExecutionPolicy, str],
    providers: Optional[Sequence[Any]],
    component_names: Optional[Iterable[str]] = None,
) -> ExecutionPolicy:
    """
    Reject NPU sessions that would compile one graph per layer.

    Phase-split and MoE routing are allowed. CPU and GPU split policies are
    allowed. A policy that does not match the selected device warns.
    """
    policy = coerce_execution_policy(execution_policy)
    provider_list = list(providers or ())
    names = list(component_names or ())
    targets_npu = providers_target_npu(provider_list)
    targets_gpu = any(provider_is_gpu(provider) for provider in provider_list)
    layer_split = is_sequential_layer_split(names)
    # SPLIT_* is the explicit sequential/persistent layer policy. A layer
    # series under any other non-hybrid policy is the same shape in practice.
    sequential_split = policy in SEQUENTIAL_LAYER_SPLIT_POLICIES or (
        policy not in HYBRID_NPU_POLICIES and layer_split
    )

    if targets_npu and sequential_split and policy not in HYBRID_NPU_POLICIES:
        raise RuntimeError(NPU_SEQUENTIAL_SPLIT_ERROR)

    if policy in HYBRID_NPU_POLICIES and not targets_npu:
        warnings.warn(
            f"Execution policy '{policy.value}' is an NPU hybrid strategy, "
            "but the provider list has no VitisAI, RyzenAI, OpenVINO, or "
            "CoreML provider.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif policy == ExecutionPolicy.SPLIT_GPU_PERSISTENT and not targets_gpu:
        warnings.warn(
            "Execution policy 'split_gpu_persistent' expects a GPU provider "
            "(CUDA, ROCm, TensorRT, MIGraphX, or DirectML).",
            RuntimeWarning,
            stacklevel=2,
        )
    elif (
        policy == ExecutionPolicy.SPLIT_CPU_MEMORY_OPTIMIZED
        and targets_gpu
        and not targets_npu
    ):
        warnings.warn(
            "Execution policy 'split_cpu_memory_optimized' unloads each "
            "layer. A GPU target should use 'split_gpu_persistent'.",
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
    options: Dict[str, str] = {}
    for key, value in source.items():
        if key not in allowed or value is None:
            continue
        options[key] = str(value)
    return options


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
        device_id: Optional device index override for the selected profile.
        cache_dir: Optional compile-cache directory override.
        enable_profiling: Enable ONNX Runtime profiling
        graph_optimization_level: Enum for optimization level
        disable_shape_check: Disable node shape checking for dynamic shapes.
            AOT NPU profiles keep shape checking on.
        execution_policy: How this model may be split across sessions.
            NPU targets reject sequential layer splits. Phase-split and MoE
            routing are the allowed NPU exceptions.

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
    execution_policy: ExecutionPolicy = ExecutionPolicy.MONOLITHIC_ONLY
    profile: Optional[ProviderProfile] = None
    optimization_level: Optional[str] = None  # None = use the provider profile
    inter_op_threads: int = 0  # 0 = profile default
    intra_op_threads: int = 0  # 0 = profile default
    memory_limit_mb: Optional[int] = None
    device_id: Optional[int] = None
    cache_dir: Optional[str] = None
    enable_profiling: bool = False
    disable_shape_check: bool = True  # Ignored for AOT NPU profiles

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
        if self.device_id is not None:
            profile.device_id = int(self.device_id)
            if "device_id" in profile.provider_options:
                profile.provider_options["device_id"] = str(self.device_id)
        chosen_cache = self.cache_dir or cache_dir
        if chosen_cache and not profile.cache_dir:
            profile.cache_dir = chosen_cache
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
        for provider in self.providers:
            profile = provider_profile_for(provider)
            if self.device_id is not None:
                profile.device_id = int(self.device_id)
                if "device_id" in profile.provider_options:
                    profile.provider_options["device_id"] = str(self.device_id)
            if self.memory_limit_mb is not None:
                profile.memory_limit_mb = self.memory_limit_mb
            chosen_cache = self.cache_dir or cache_dir
            if chosen_cache:
                profile.cache_dir = chosen_cache
            if profile.device_type in {"vitisai", "ryzenai"} and not self.npu_graph_capture:
                profile.enable_graph_cache = False
            options = ep_provider_options(profile)
            provider_name = _provider_name(provider)
            if options:
                prepared.append((provider_name, options))
            else:
                prepared.append(provider_name)
        return prepared
    
    def create_session_options(self, cache_dir: Optional[str] = None) -> ort.SessionOptions:
        """
        Create SessionOptions from the selected provider profile.

        VitisAI still receives graph capture, weight preload, and compilation
        mode. AOT NPU profiles keep shape checking enabled.

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
        profile = self.active_profile(cache_dir=cache_dir)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = self._optimization_enum(profile.optimization_level)

        intra = self.intra_op_threads or int(profile.provider_options.get("intra_op_num_threads") or 0)
        inter = self.inter_op_threads or int(profile.provider_options.get("inter_op_num_threads") or 0)
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

        if self.enable_profiling:
            opts.enable_profiling = True

        if profile.device_type in _STATIC_SHAPE_DEVICES:
            if self.disable_shape_check:
                _warn_static_npu_shapes(profile.name)
        elif self.disable_shape_check:
            opts.add_session_config_entry("session.disable_node_shape_check", "1")

        self._apply_vitisai_profile(opts, profile)
        return opts

    def _apply_vitisai_profile(
        self,
        opts: ort.SessionOptions,
        profile: ProviderProfile,
    ) -> None:
        """Write VitisAI session keys when that provider is selected or listed."""
        uses_vitisai = profile.device_type == "vitisai" or any(
            "vitisai" in _provider_name(provider).lower() for provider in self.providers
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

    def resolved_execution_policy(self) -> ExecutionPolicy:
        """Return ``execution_policy`` as an enum, accepting its string value."""
        self.execution_policy = coerce_execution_policy(self.execution_policy)
        return self.execution_policy

    def validate_execution_policy(
        self,
        component_names: Optional[Iterable[str]] = None,
    ) -> ExecutionPolicy:
        """
        Raise if this config would layer-split an NPU.

        ``component_names`` comes from the discovered model. With no names,
        an explicit ``SPLIT_*`` policy is still rejected on an NPU provider.
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
    """
    name: str
    file_path: str
    file_size_bytes: int = 0
    is_loaded: bool = False
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    input_types: Dict[str, str] = field(default_factory=dict)
    output_types: Dict[str, str] = field(default_factory=dict)
    provider_used: Optional[str] = None
    
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
            "provider_used": self.provider_used,
        }


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
    - Reject layer-by-layer loads on VitisAI, RyzenAI, OpenVINO, and CoreML
      unless the policy is phase-split or MoE expert routing
    - Select a ProviderProfile from the first execution provider
    """
    
    def __init__(
        self,
        model_dir: str,
        config: Optional[SessionConfig] = None,
        model_info: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the session manager.
        
        Args:
            model_dir: Directory containing split ONNX components
            config: Session configuration (uses defaults if None)
            model_info: Pre-loaded model_info.json dict (loads from file if None)
        """
        self.model_dir = model_dir
        self.config = config or SessionConfig()
        self.provider_profile = self.config.active_profile(
            cache_dir=os.path.join(self.model_dir, ".ort_cache")
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
        
        # Pre-populate session info (without loading)
        self._discover_sessions()
        self._enforce_execution_policy()
    
    def _discover_sessions(self) -> None:
        """Discover all available sessions from model_info without loading them."""
        components = self.model_info.get("components", {})
        
        for name, comp_info in components.items():
            filename = comp_info.get("file", "")
            filepath = os.path.join(self.model_dir, filename)
            
            file_size = 0
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
            
            self._session_info[name] = SessionInfo(
                name=name,
                file_path=filepath,
                file_size_bytes=file_size,
                is_loaded=False,
                input_names=comp_info.get("input_names", []),
                output_names=comp_info.get("output_names", []),
            )
    
    def _load_single_session(self, name: str) -> Optional[ort.InferenceSession]:
        """
        Load a single ONNX session by component name.
        
        Args:
            name: Component name from model_info
            
        Returns:
            Loaded InferenceSession or None if failed
        """
        if name not in self._session_info:
            return None
        
        info = self._session_info[name]
        
        if not os.path.exists(info.file_path):
            return None
        
        try:
            self._enforce_execution_policy()
            self._prepare_profile_cache()
            opts = self.config.create_session_options(
                cache_dir=self.provider_profile.cache_dir
            )
            session = ort.InferenceSession(
                info.file_path,
                sess_options=opts,
                providers=self.config.ort_providers(
                    cache_dir=self.provider_profile.cache_dir
                ),
            )
            
            # Update session info with loaded details
            info.is_loaded = True
            info.input_names = [inp.name for inp in session.get_inputs()]
            info.output_names = [out.name for out in session.get_outputs()]
            info.input_types = {inp.name: inp.type for inp in session.get_inputs()}
            info.output_types = {out.name: out.type for out in session.get_outputs()}
            
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

            self._enforce_execution_policy()
            # Lazy load if not found
            session = self._load_single_session(name)
            if session:
                self._sessions[name] = session
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
            if name in self._sessions:
                del self._sessions[name]
                if name in self._session_info:
                    self._session_info[name].is_loaded = False
                    self._session_info[name].provider_used = None
                return True
            return False
    
    def unload_all(self) -> int:
        """
        Unload all sessions.
        
        Returns:
            Number of sessions unloaded
        """
        with self._lock:
            count = len(self._sessions)
            self._sessions.clear()
            for info in self._session_info.values():
                info.is_loaded = False
                info.provider_used = None
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
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dict with memory stats per session and total
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
            
            return {
                "total_bytes": total_bytes,
                "total_mb": round(total_bytes / (1024 * 1024), 2),
                "loaded_count": loaded_count,
                "total_count": len(self._session_info),
                "per_session_bytes": per_session,
            }
    
    def get_component_names(self) -> List[str]:
        """Get all component names in execution order."""
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
    
    def _prepare_profile_cache(self) -> None:
        """Create the compile-cache directory when this profile uses one."""
        profile = self.provider_profile
        if profile.cache_dir and (profile.enable_compile_cache or profile.enable_graph_cache):
            os.makedirs(profile.cache_dir, exist_ok=True)

    def _enforce_execution_policy(self) -> None:
        """
        Reject an NPU layer-by-layer load.

        The same provider list and component set is checked once. A later
        change to ``config.providers`` or ``config.execution_policy`` is
        checked again on the next load.
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
    "ProviderProfile",
    # Functions
    "get_available_providers",
    "get_provider_for_device",
    "provider_is_npu",
    "providers_target_npu",
    "is_sequential_layer_split",
    "validate_npu_execution_policy",
    "provider_profile_for",
    "primary_provider_name",
    "ep_provider_options",
    # Constants
    "SUPPORTED_PROVIDERS",
    "PROVIDER_CATEGORIES",
    "NPU_SEQUENTIAL_SPLIT_ERROR",
    "NPU_PROVIDER_MARKERS",
]
