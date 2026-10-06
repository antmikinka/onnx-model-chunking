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
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
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

# Provider categories for documentation
PROVIDER_CATEGORIES = {
    'gpu': ['CUDAExecutionProvider', 'ROCmExecutionProvider', 'TensorrtExecutionProvider', 'MIGraphXExecutionProvider'],
    'npu': ['RyzenAIExecutionProvider', 'VitisAIExecutionProvider', 'OpenVINOExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider'],
    'cpu': ['CPUExecutionProvider']
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SessionConfig:
    """
    Configuration for ONNX Runtime session creation.

    Attributes:
        providers: Execution providers in priority order
        optimization_level: Graph optimization level (DISABLE/BASIC/EXTENDED/ALL)
        inter_op_threads: Number of threads for parallel op execution
        intra_op_threads: Number of threads within a single op
        memory_limit_mb: Optional memory limit for arena allocator
        enable_profiling: Enable ONNX Runtime profiling
        graph_optimization_level: Enum for optimization level
        disable_shape_check: Disable node shape checking for dynamic shapes

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
    optimization_level: str = "DISABLE"  # DISABLE, BASIC, EXTENDED, ALL
    inter_op_threads: int = 0  # 0 = auto
    intra_op_threads: int = 0  # 0 = auto
    memory_limit_mb: Optional[int] = None
    enable_profiling: bool = False
    disable_shape_check: bool = True  # For dynamic shapes in LLMs

    # NPU-specific options (P1.2: NPU Session Options)
    npu_graph_capture: bool = True
    npu_weight_preload: bool = True
    npu_compilation_mode: str = "default"  # "default", "fast", "optimized"
    
    def get_optimization_level_enum(self) -> ort.GraphOptimizationLevel:
        """Convert string to ONNX Runtime enum."""
        mapping = {
            "DISABLE": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        return mapping.get(self.optimization_level.upper(), 
                          ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    
    def create_session_options(self) -> ort.SessionOptions:
        """
        Create configured SessionOptions object.

        Includes NPU-specific configuration when VitisAIExecutionProvider is used.

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
        opts = ort.SessionOptions()
        opts.graph_optimization_level = self.get_optimization_level_enum()

        if self.inter_op_threads > 0:
            opts.inter_op_num_threads = self.inter_op_threads
        if self.intra_op_threads > 0:
            opts.intra_op_num_threads = self.intra_op_threads

        if self.enable_profiling:
            opts.enable_profiling = True

        if self.disable_shape_check:
            opts.add_session_config_entry("session.disable_node_shape_check", "1")

        # P1.2: Add NPU-specific session configuration
        if "VitisAIExecutionProvider" in self.providers:
            self._add_npu_session_config(opts)

        return opts

    def _add_npu_session_config(self, opts: ort.SessionOptions) -> None:
        """
        Add NPU-specific session configuration entries.

        These options optimize VitisAI Execution Provider performance:

        1. vitisai.graph_capture: Enable graph capture for reuse
           - Reduces compilation overhead on subsequent runs
           - Recommended for inference workloads

        2. vitisai.weight_preload: Preload weights to tile SRAM
           - Keeps weights in on-chip memory for faster access
           - Critical for NPU performance

        3. vitisai.compilation_mode: Compilation strategy
           - "default": Balanced approach
           - "fast": Prioritize compilation speed
           - "optimized": Prioritize runtime performance

        Args:
            opts: SessionOptions object to configure
        """
        # Graph capture for faster inference
        opts.add_session_config_entry(
            "vitisai.graph_capture",
            "1" if self.npu_graph_capture else "0"
        )

        # Weight preloading to tile SRAM
        opts.add_session_config_entry(
            "vitisai.weight_preload",
            "1" if self.npu_weight_preload else "0"
        )

        # Compilation mode
        opts.add_session_config_entry(
            "vitisai.compilation_mode",
            self.npu_compilation_mode
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
        
        # Pre-populate session info (without loading)
        self._discover_sessions()
    
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
            opts = self.config.create_session_options()
            session = ort.InferenceSession(
                info.file_path,
                sess_options=opts,
                providers=self.config.providers
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
    # Functions
    "get_available_providers",
    "get_provider_for_device",
    # Constants
    "SUPPORTED_PROVIDERS",
    "PROVIDER_CATEGORIES",
]
