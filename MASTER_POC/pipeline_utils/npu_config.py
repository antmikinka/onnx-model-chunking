#!/usr/bin/env python3
"""
npu_config.py - NPU Configuration and Management

This module provides comprehensive NPU configuration and management utilities
for AMD XDNA NPU (Ryzen AI) integration with ONNX Runtime.

Key Features:
- NPUConfig: Complete NPU configuration dataclass
- NPUOffloadConfig: Memory offloading strategy configuration
- NPU detection and validation utilities
- Provider capability checking

Usage:
    from pipeline_utils.npu_config import NPUConfig, NPUOffloadConfig, is_npu_available

    # Check if NPU is available
    if is_npu_available():
        config = NPUConfig(
            graph_capture=True,
            weight_preload=True,
            compilation_mode="optimized"
        )
        opts = config.create_session_options()
"""

import os
import json
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class NPUCompilationMode(str, Enum):
    """
    NPU compilation mode options.

    Members:
        DEFAULT: Balanced compilation for general workloads
        FAST: Prioritize compilation speed over optimization
        OPTIMIZED: Prioritize runtime performance (slower compilation)
    """
    DEFAULT = "default"
    FAST = "fast"
    OPTIMIZED = "optimized"


class NPUOffloadStrategy(str, Enum):
    """
    Memory offloading strategy options.

    Members:
        LAYER_BY_LAYER: Load/unload one layer at a time (minimal memory)
        PIPELINE_PARALLEL: Overlap compute and transfer
        STATIC_PARTITION: Pre-allocate memory partitions
    """
    LAYER_BY_LAYER = "layer_by_layer"
    PIPELINE_PARALLEL = "pipeline_parallel"
    STATIC_PARTITION = "static_partition"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class NPUConfig:
    """
    Comprehensive NPU configuration for ONNX Runtime VitisAI provider.

    This dataclass encapsulates all NPU-specific options available for
    optimizing inference on AMD XDNA NPU (Ryzen AI).

    Attributes:
        graph_capture: Enable graph capture for reuse (default: True)
            When enabled, the NPU computation graph is captured and cached
            after the first compilation. Subsequent runs reuse the cached
            graph, avoiding recompilation overhead.

            Recommended: True for inference, False for dynamic models

        weight_preload: Preload weights to NPU tile SRAM (default: True)
            When enabled, model weights are preloaded into the NPU's
            on-chip SRAM (tile memory). This dramatically reduces
            memory bandwidth bottlenecks during inference.

            Recommended: True when model fits in tile SRAM

        compilation_mode: NPU compilation strategy (default: "default")
            Controls the trade-off between compilation time and runtime
            performance. See NPUCompilationMode for options.

        enable_profiling: Enable NPU profiling (default: False)
            When enabled, collects performance metrics from the NPU.
            Useful for debugging and optimization.

        debug_mode: Enable debug logging (default: False)
            When enabled, outputs detailed NPU operation logs.

        memory_limit_mb: Memory limit for NPU allocations (default: None)
            Maximum memory (in MB) to allocate on NPU. If None, uses
            hardware default.

        num_cores: Number of NPU cores to use (default: -1 = all)
            Controls parallelism level. Use -1 for automatic selection.

    Example:
        >>> config = NPUConfig(
        ...     graph_capture=True,
        ...     weight_preload=True,
        ...     compilation_mode=NPUCompilationMode.OPTIMIZED
        ... )
        >>> opts = config.create_session_options()
    """
    graph_capture: bool = True
    weight_preload: bool = True
    compilation_mode: NPUCompilationMode = NPUCompilationMode.DEFAULT
    enable_profiling: bool = False
    debug_mode: bool = False
    memory_limit_mb: Optional[int] = None
    num_cores: int = -1  # -1 = use all available cores

    def create_session_options(self) -> Any:
        """
        Create ONNX Runtime SessionOptions with NPU configuration.

        Returns:
            ort.SessionOptions object configured for NPU

        Note:
            Requires onnxruntime to be installed. Returns None if
            onnxruntime is not available.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            print("[NPUConfig] Warning: onnxruntime not installed")
            return None

        opts = ort.SessionOptions()

        # Graph capture configuration
        opts.add_session_config_entry(
            "vitisai.graph_capture",
            "1" if self.graph_capture else "0"
        )

        # Weight preloading configuration
        opts.add_session_config_entry(
            "vitisai.weight_preload",
            "1" if self.weight_preload else "0"
        )

        # Compilation mode
        opts.add_session_config_entry(
            "vitisai.compilation_mode",
            str(self.compilation_mode.value)
        )

        # Profiling configuration
        opts.add_session_config_entry(
            "vitisai.enable_profiling",
            "1" if self.enable_profiling else "0"
        )

        # Debug mode
        opts.add_session_config_entry(
            "vitisai.debug_mode",
            "1" if self.debug_mode else "0"
        )

        # Memory limit
        if self.memory_limit_mb is not None:
            opts.add_session_config_entry(
                "vitisai.memory_limit_mb",
                str(self.memory_limit_mb)
            )

        # Number of cores
        if self.num_cores > 0:
            opts.add_session_config_entry(
                "vitisai.num_cores",
                str(self.num_cores)
            )

        return opts

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "graph_capture": self.graph_capture,
            "weight_preload": self.weight_preload,
            "compilation_mode": self.compilation_mode.value,
            "enable_profiling": self.enable_profiling,
            "debug_mode": self.debug_mode,
            "memory_limit_mb": self.memory_limit_mb,
            "num_cores": self.num_cores,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NPUConfig":
        """Create NPUConfig from dictionary."""
        # Handle compilation_mode enum conversion
        if "compilation_mode" in data:
            if isinstance(data["compilation_mode"], str):
                data["compilation_mode"] = NPUCompilationMode(data["compilation_mode"])

        return cls(**data)


@dataclass
class NPUOffloadConfig:
    """
    Configuration for NPU memory offloading strategies.

    When models exceed NPU on-chip memory (tile SRAM), this configuration
    controls how weights and activations are managed between CPU and NPU.

    Attributes:
        npu_memory_limit_mb: Maximum NPU memory usage (default: 4096)
            Limits how much model data resides in NPU memory.
            Typical values: 2048-8192 MB depending on NPU model.

        offload_strategy: Memory offloading approach (default: "layer_by_layer")
            - "layer_by_layer": Load one layer, compute, unload (minimal memory)
            - "pipeline_parallel": Overlap transfer and compute (better perf)
            - "static_partition": Pre-allocate fixed memory partitions

        prefetch_layers: Number of layers to prefetch ahead (default: 1)
            Controls how many future layers are loaded in advance.
            Higher values improve throughput but increase memory usage.

        enable_double_buffering: Use double buffering for transfers (default: True)
            When enabled, uses ping-pong buffers to overlap compute
            and data transfer, hiding latency.

        cpu_pinned_memory: Use CPU pinned memory for transfers (default: True)
            Pinned (page-locked) memory enables faster DMA transfers
            between CPU and NPU.

        async_transfer: Enable asynchronous transfers (default: True)
            When enabled, data transfers happen in background threads.

        stream_priority: Priority level for NPU streams (default: 0)
            Higher values = higher priority. Range: 0-10.
    """
    npu_memory_limit_mb: int = 4096
    offload_strategy: NPUOffloadStrategy = NPUOffloadStrategy.LAYER_BY_LAYER
    prefetch_layers: int = 1
    enable_double_buffering: bool = True
    cpu_pinned_memory: bool = True
    async_transfer: bool = True
    stream_priority: int = 0

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.npu_memory_limit_mb <= 0:
            raise ValueError("npu_memory_limit_mb must be positive")
        if self.prefetch_layers < 0:
            raise ValueError("prefetch_layers must be non-negative")
        if not (0 <= self.stream_priority <= 10):
            raise ValueError("stream_priority must be between 0 and 10")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "npu_memory_limit_mb": self.npu_memory_limit_mb,
            "offload_strategy": self.offload_strategy.value,
            "prefetch_layers": self.prefetch_layers,
            "enable_double_buffering": self.enable_double_buffering,
            "cpu_pinned_memory": self.cpu_pinned_memory,
            "async_transfer": self.async_transfer,
            "stream_priority": self.stream_priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NPUOffloadConfig":
        """Create NPUOffloadConfig from dictionary."""
        # Handle offload_strategy enum conversion
        if "offload_strategy" in data:
            if isinstance(data["offload_strategy"], str):
                data["offload_strategy"] = NPUOffloadStrategy(data["offload_strategy"])

        return cls(**data)


@dataclass
class NPUCapabilities:
    """
    Information about NPU hardware capabilities.

    Attributes:
        available: Whether NPU is available on this system
        driver_version: Version of NPU driver
        num_cores: Number of compute cores available
        memory_size_mb: Total NPU memory in MB
        tile_sram_mb: On-chip tile SRAM in MB
        supported_providers: List of supported ONNX providers
    """
    available: bool = False
    driver_version: Optional[str] = None
    num_cores: int = 0
    memory_size_mb: int = 0
    tile_sram_mb: int = 0
    supported_providers: List[str] = field(default_factory=list)

    def is_capable(self, model_size_mb: int) -> bool:
        """
        Check if NPU can handle a model of given size.

        Args:
            model_size_mb: Size of model in MB

        Returns:
            True if model fits in NPU memory
        """
        return self.available and (model_size_mb <= self.memory_size_mb)

    def fits_in_tile_sram(self, model_size_mb: int) -> bool:
        """
        Check if model fits entirely in tile SRAM.

        Args:
            model_size_mb: Size of model in MB

        Returns:
            True if model fits in tile SRAM
        """
        return self.available and (model_size_mb <= self.tile_sram_mb)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "available": self.available,
            "driver_version": self.driver_version,
            "num_cores": self.num_cores,
            "memory_size_mb": self.memory_size_mb,
            "tile_sram_mb": self.tile_sram_mb,
            "supported_providers": self.supported_providers,
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_npu_available() -> bool:
    """
    Check if AMD NPU (XDNA/Ryzen AI) is available on this system.

    Returns:
        True if NPU is detected and usable

    Example:
        >>> if is_npu_available():
        ...     print("NPU found!")
    """
    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        return "VitisAIExecutionProvider" in available_providers
    except ImportError:
        return False


def get_npu_capabilities() -> NPUCapabilities:
    """
    Query NPU hardware capabilities.

    Returns:
        NPUCapabilities object with hardware information

    Example:
        >>> caps = get_npu_capabilities()
        >>> if caps.available:
        ...     print(f"NPU: {caps.num_cores} cores, {caps.memory_size_mb}MB")
    """
    caps = NPUCapabilities()

    # Check if VitisAI provider is available
    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()

        if "VitisAIExecutionProvider" in available_providers:
            caps.available = True
            caps.supported_providers = available_providers
    except ImportError:
        return caps

    # Try to get detailed info from xclinfo or similar tools
    try:
        # Try xclinfo (Xilinx/AMD tool)
        result = subprocess.run(
            ["xclinfo"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Parse xclinfo output
            output = result.stdout
            caps.driver_version = _parse_xclinfo_driver_version(output)
            caps.num_cores = _parse_xclinfo_num_cores(output)
            caps.memory_size_mb = _parse_xclinfo_memory_size(output)
            caps.tile_sram_mb = _parse_xclinfo_tile_sram(output)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return caps


def _parse_xclinfo_driver_version(output: str) -> Optional[str]:
    """Parse driver version from xclinfo output."""
    for line in output.split("\n"):
        if "Version" in line and "xocl" in line.lower():
            parts = line.split(":")
            if len(parts) >= 2:
                return parts[1].strip()
    return None


def _parse_xclinfo_num_cores(output: str) -> int:
    """Parse number of cores from xclinfo output."""
    # This is a placeholder - actual parsing depends on xclinfo format
    return 0


def _parse_xclinfo_memory_size(output: str) -> int:
    """Parse memory size from xclinfo output."""
    for line in output.split("\n"):
        if "DDR" in line or "Memory" in line:
            # Parse memory size (format: "DDR[0]: 8192 MB")
            if "MB" in line:
                try:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        size_str = parts[1].strip().replace("MB", "").strip()
                        return int(size_str)
                except ValueError:
                    pass
    return 0


def _parse_xclinfo_tile_sram(output: str) -> int:
    """Parse tile SRAM size from xclinfo output."""
    # Tile SRAM is typically 16-32 MB for AMD XDNA NPUs
    # This is a reasonable default
    return 32


def validate_npu_config(config: NPUConfig, capabilities: NPUCapabilities) -> Tuple[bool, List[str]]:
    """
    Validate NPU configuration against hardware capabilities.

    Args:
        config: NPUConfig to validate
        capabilities: NPUCapabilities of the system

    Returns:
        Tuple of (is_valid, list_of_warnings)

    Example:
        >>> config = NPUConfig(memory_limit_mb=8192)
        >>> caps = get_npu_capabilities()
        >>> valid, warnings = validate_npu_config(config, caps)
        >>> if not valid:
        ...     print("Issues:", warnings)
    """
    warnings = []

    if not capabilities.available:
        return False, ["NPU not available on this system"]

    # Check memory limit
    if config.memory_limit_mb is not None:
        if config.memory_limit_mb > capabilities.memory_size_mb:
            warnings.append(
                f"memory_limit_mb ({config.memory_limit_mb}) exceeds "
                f"available NPU memory ({capabilities.memory_size_mb}MB)"
            )
        if config.memory_limit_mb > capabilities.tile_sram_mb * 2:
            warnings.append(
                f"memory_limit_mb ({config.memory_limit_mb}) may cause "
                f"offloading (tile SRAM: {capabilities.tile_sram_mb}MB)"
            )

    # Check compilation mode
    if config.compilation_mode == NPUCompilationMode.OPTIMIZED:
        warnings.append(
            "OPTIMIZED compilation mode may take longer to initialize"
        )

    return len(warnings) == 0, warnings


def get_recommended_npu_config(model_size_mb: float) -> NPUConfig:
    """
    Get recommended NPU configuration based on model size.

    Args:
        model_size_mb: Size of model in megabytes

    Returns:
        NPUConfig with recommended settings

    Example:
        >>> config = get_recommended_npu_config(model_size_mb=4000)
        >>> opts = config.create_session_options()
    """
    # Get actual capabilities if available
    caps = get_npu_capabilities()

    # Default recommendations
    config = NPUConfig()

    if caps.available:
        # Adjust based on whether model fits in tile SRAM
        if model_size_mb <= caps.tile_sram_mb:
            # Model fits entirely in SRAM - enable all optimizations
            config.graph_capture = True
            config.weight_preload = True
            config.compilation_mode = NPUCompilationMode.OPTIMIZED
        elif model_size_mb <= caps.memory_size_mb:
            # Model fits in NPU memory but not tile SRAM
            config.graph_capture = True
            config.weight_preload = False  # Can't preload all weights
            config.compilation_mode = NPUCompilationMode.DEFAULT
        else:
            # Model requires offloading
            config.graph_capture = True
            config.weight_preload = False
            config.compilation_mode = NPUCompilationMode.FAST
    else:
        # NPU not available - return safe defaults
        config.graph_capture = False
        config.weight_preload = False

    return config


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "NPUCompilationMode",
    "NPUOffloadStrategy",
    # Data classes
    "NPUConfig",
    "NPUOffloadConfig",
    "NPUCapabilities",
    # Utility functions
    "is_npu_available",
    "get_npu_capabilities",
    "validate_npu_config",
    "get_recommended_npu_config",
]
