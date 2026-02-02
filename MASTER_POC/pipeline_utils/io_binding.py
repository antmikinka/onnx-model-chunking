#!/usr/bin/env python3
"""
io_binding.py - I/O Binding for Zero-Copy Inference

This module provides I/O binding utilities for ONNX Runtime that:
1. Keep tensors in device memory between operations (zero-copy)
2. Reduce CPU↔Device data transfer overhead
3. Support multiple device types (CPU, CUDA, DirectML, etc.)

IMPORTANT MEMORY TRADEOFF CLARIFICATION:
- I/O Binding does NOT reduce RAM usage
- It reduces DATA COPY latency between CPU and device
- For RAM reduction, use quantization, smaller batches, or sequential mode

Usage:
    from pipeline_utils.io_binding import IOBindingManager, MemoryMode
    
    # Create manager for CUDA device
    manager = IOBindingManager(device_type="cuda", device_id=0)
    
    # Pre-allocate buffers
    manager.allocate_buffer("hidden_states", (1, 512, 4096), np.float16)
    
    # Run with I/O binding
    outputs = manager.run_with_binding(session, {"input_ids": input_ids})
"""

import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

import onnxruntime as ort

# Import our dtype utilities
from .dtypes import get_numpy_dtype, ONNX_DTYPE_MAP


# =============================================================================
# MEMORY MODE ENUM
# =============================================================================

class MemoryMode(Enum):
    """
    Memory vs Speed tradeoff modes.
    
    Choose based on your constraints:
    
    MINIMAL:
        - Load one component at a time, unload after use
        - RAM: O(1 component) + activations
        - Speed: SLOW (reload each component per token)
        - Best for: Limited RAM, CPU inference
    
    PERSISTENT:
        - Load all components once, keep resident
        - RAM: O(all components) + activations  
        - Speed: FAST (no reload overhead)
        - Best for: Sufficient RAM, NPU/GPU inference
    
    IO_BOUND:
        - Persistent + I/O binding for device tensors
        - RAM: Same as PERSISTENT
        - Device Memory: Activations stay on device
        - Speed: FASTEST (no CPU-device copies)
        - Best for: NPU/GPU with device memory available
    """
    MINIMAL = "minimal"
    PERSISTENT = "persistent"
    IO_BOUND = "io_bound"
    
    @classmethod
    def get_description(cls, mode: 'MemoryMode') -> str:
        """Get human-readable description of memory mode."""
        descriptions = {
            cls.MINIMAL: "RAM-efficient: Load/unload per component (slow)",
            cls.PERSISTENT: "Balanced: All sessions resident (medium RAM, fast)",
            cls.IO_BOUND: "Throughput-optimized: Persistent + zero-copy (fast)",
        }
        return descriptions.get(mode, "Unknown mode")


# =============================================================================
# NUMPY TO ONNX ELEMENT TYPE MAPPING
# =============================================================================

# Map numpy dtypes to ONNX TensorProto.DataType values
# These are the integer constants used by ort.OrtValue.ortvalue_from_numpy
NUMPY_TO_ONNX_ELEMENT_TYPE = {
    np.float16: 10,     # FLOAT16
    np.float32: 1,      # FLOAT
    np.float64: 11,     # DOUBLE
    np.int8: 3,         # INT8
    np.int16: 5,        # INT16
    np.int32: 6,        # INT32
    np.int64: 7,        # INT64
    np.uint8: 2,        # UINT8
    np.uint16: 4,       # UINT16
    np.uint32: 12,      # UINT32
    np.uint64: 13,      # UINT64
    np.bool_: 9,        # BOOL
    np.complex64: 14,   # COMPLEX64
    np.complex128: 15,  # COMPLEX128
}

# String dtype names for display
ONNX_ELEMENT_TYPE_NAMES = {
    1: "FLOAT",
    2: "UINT8", 
    3: "INT8",
    4: "UINT16",
    5: "INT16",
    6: "INT32",
    7: "INT64",
    9: "BOOL",
    10: "FLOAT16",
    11: "DOUBLE",
    12: "UINT32",
    13: "UINT64",
    14: "COMPLEX64",
    15: "COMPLEX128",
    16: "BFLOAT16",
}


def get_onnx_element_type(dtype: np.dtype) -> int:
    """
    Convert numpy dtype to ONNX element type integer.
    
    Args:
        dtype: NumPy dtype
        
    Returns:
        ONNX TensorProto.DataType integer value
        
    Raises:
        ValueError: If dtype not supported
    """
    # Handle numpy dtype objects
    if hasattr(dtype, 'type'):
        dtype_type = dtype.type
    else:
        dtype_type = dtype
    
    # Check direct mapping
    for np_type, onnx_type in NUMPY_TO_ONNX_ELEMENT_TYPE.items():
        if dtype_type == np_type or np.dtype(np_type) == np.dtype(dtype):
            return onnx_type
    
    raise ValueError(f"Unsupported dtype for OrtValue: {dtype}")


# =============================================================================
# DEVICE BUFFER
# =============================================================================

@dataclass
class DeviceBuffer:
    """
    Represents a tensor buffer in device memory.
    
    This wraps an ort.OrtValue that lives on a specific device.
    The buffer can be reused across multiple inferences if the
    shape matches.
    
    Attributes:
        name: Tensor name (e.g., "hidden_states", "input_ids")
        shape: Tensor shape tuple
        dtype: NumPy dtype
        device_type: Device type string ("cpu", "cuda", "dml")
        device_id: Device ID (0, 1, etc.)
        ort_value: The underlying OrtValue (None if not allocated)
        is_dynamic: If True, shape can change between inferences
    """
    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype
    device_type: str = "cpu"
    device_id: int = 0
    ort_value: Optional[ort.OrtValue] = None
    is_dynamic: bool = False
    
    def __post_init__(self):
        """Convert dtype to numpy dtype if string."""
        if isinstance(self.dtype, str):
            self.dtype = np.dtype(self.dtype)
    
    @property
    def size_bytes(self) -> int:
        """Calculate buffer size in bytes."""
        if not self.shape:
            return 0
        num_elements = 1
        for dim in self.shape:
            num_elements *= dim
        return num_elements * np.dtype(self.dtype).itemsize
    
    @property
    def size_mb(self) -> float:
        """Calculate buffer size in megabytes."""
        return self.size_bytes / (1024 * 1024)
    
    @property
    def is_allocated(self) -> bool:
        """Check if buffer is currently allocated."""
        return self.ort_value is not None
    
    def matches_shape(self, shape: Tuple[int, ...]) -> bool:
        """Check if buffer shape matches requested shape."""
        return self.shape == tuple(shape)
    
    def to_numpy(self) -> np.ndarray:
        """
        Copy data from device to numpy array.
        
        Returns:
            NumPy array with buffer contents
            
        Raises:
            RuntimeError: If buffer not allocated
        """
        if self.ort_value is None:
            raise RuntimeError(f"Buffer '{self.name}' not allocated")
        return self.ort_value.numpy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "device_type": self.device_type,
            "device_id": self.device_id,
            "size_mb": round(self.size_mb, 3),
            "is_allocated": self.is_allocated,
            "is_dynamic": self.is_dynamic,
        }


# =============================================================================
# I/O BINDING MANAGER
# =============================================================================

class IOBindingManager:
    """
    Manages I/O binding for zero-copy inference.
    
    Key principle: Allocate device buffers once, reuse across inferences.
    Output of layer N becomes input of layer N+1 without CPU copies.
    
    Features:
    - Pre-allocate buffers in device memory
    - Reuse buffers when shapes match
    - Handle dynamic shapes by reallocating
    - Track memory usage
    - Thread-safe operations
    
    Example:
        manager = IOBindingManager(device_type="cuda", device_id=0)
        
        # Pre-allocate known buffers
        manager.allocate_buffer("hidden_states", (1, 512, 4096), np.float16)
        
        # Run inference with binding
        io_binding = manager.create_binding(session)
        manager.bind_inputs(io_binding, {"input_ids": input_array})
        manager.bind_outputs(io_binding, session)
        session.run_with_iobinding(io_binding)
        
        # Get outputs (still on device)
        hidden = manager.get_buffer("hidden_states")
    """
    
    def __init__(
        self,
        device_type: str = "cpu",
        device_id: int = 0,
        auto_reallocate: bool = True
    ):
        """
        Initialize the I/O binding manager.
        
        Args:
            device_type: Target device type ("cpu", "cuda", "dml", "coreml")
            device_id: Device ID for multi-GPU/NPU systems
            auto_reallocate: If True, automatically reallocate buffers on shape mismatch
        """
        self.device_type = device_type
        self.device_id = device_id
        self.auto_reallocate = auto_reallocate
        
        # Validate device type for OrtValue creation
        # OrtValue only supports: cpu, cuda, cuda_pinned
        # For other devices, data is created on CPU and copied during inference
        VALID_ORTVALUE_DEVICES = {"cpu", "cuda", "cuda_pinned"}
        if device_type in VALID_ORTVALUE_DEVICES:
            self._ortvalue_device = device_type
        else:
            # Fallback to CPU for OrtValue creation
            # The bind_output still uses the original device_type
            self._ortvalue_device = "cpu"
        
        self._buffers: Dict[str, DeviceBuffer] = {}
        self._lock = threading.RLock()  # Reentrant lock to prevent deadlock
    
    def _create_ort_value(
        self,
        data: np.ndarray,
        device_type: Optional[str] = None,
        device_id: Optional[int] = None
    ) -> ort.OrtValue:
        """
        Create an OrtValue from numpy array on target device.
        
        Args:
            data: NumPy array with data
            device_type: Override device type (defaults to manager's device)
            device_id: Override device ID
            
        Returns:
            OrtValue on target device
        """
        dev_type = device_type or self._ortvalue_device  # Use validated device
        dev_id = device_id if device_id is not None else self.device_id
        
        # Create OrtValue - copies data to device
        return ort.OrtValue.ortvalue_from_numpy(data, dev_type, dev_id)
    
    def _create_empty_ort_value(
        self,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        device_type: Optional[str] = None,
        device_id: Optional[int] = None
    ) -> ort.OrtValue:
        """
        Create an empty OrtValue buffer on target device.
        
        Args:
            shape: Tensor shape
            dtype: NumPy dtype
            device_type: Override device type
            device_id: Override device ID
            
        Returns:
            Empty OrtValue on target device
        """
        dev_type = device_type or self._ortvalue_device  # Use validated device
        dev_id = device_id if device_id is not None else self.device_id
        
        # Get ONNX element type
        element_type = get_onnx_element_type(dtype)
        
        # Create using shape (allocates on device without data copy)
        return ort.OrtValue.ortvalue_from_shape_and_type(
            list(shape), element_type, dev_type, dev_id
        )
    
    def allocate_buffer(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        data: Optional[np.ndarray] = None,
        is_dynamic: bool = False
    ) -> DeviceBuffer:
        """
        Allocate a buffer in device memory.
        
        Args:
            name: Buffer name for lookup
            shape: Tensor shape
            dtype: NumPy dtype
            data: Optional initial data (if None, creates empty buffer)
            is_dynamic: Mark as dynamic shape (may be reallocated)
            
        Returns:
            DeviceBuffer wrapping the OrtValue
        """
        with self._lock:
            # Convert dtype
            if isinstance(dtype, str):
                dtype = np.dtype(dtype)
            
            # Create OrtValue
            if data is not None:
                ort_value = self._create_ort_value(data)
            else:
                ort_value = self._create_empty_ort_value(shape, dtype)
            
            # Create buffer wrapper
            buffer = DeviceBuffer(
                name=name,
                shape=tuple(shape),
                dtype=dtype,
                device_type=self.device_type,
                device_id=self.device_id,
                ort_value=ort_value,
                is_dynamic=is_dynamic
            )
            
            self._buffers[name] = buffer
            return buffer
    
    def get_or_allocate(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        data: Optional[np.ndarray] = None
    ) -> DeviceBuffer:
        """
        Get existing buffer or allocate if needed.
        
        If buffer exists with matching shape, returns it.
        If shape differs and auto_reallocate is True, reallocates.
        
        Args:
            name: Buffer name
            shape: Required shape
            dtype: Required dtype
            data: Optional data to initialize with
            
        Returns:
            DeviceBuffer (existing or newly allocated)
        """
        with self._lock:
            if name in self._buffers:
                existing = self._buffers[name]
                if existing.matches_shape(shape):
                    # Update data if provided
                    if data is not None:
                        existing.ort_value = self._create_ort_value(data)
                    return existing
                elif self.auto_reallocate:
                    # Shape mismatch, reallocate
                    pass  # Fall through to allocation
                else:
                    raise ValueError(
                        f"Buffer '{name}' shape mismatch: "
                        f"existing {existing.shape} vs requested {shape}"
                    )
            
            # Allocate new buffer
            return self.allocate_buffer(name, shape, dtype, data, is_dynamic=True)
    
    def get_buffer(self, name: str) -> Optional[DeviceBuffer]:
        """Get buffer by name."""
        return self._buffers.get(name)
    
    def has_buffer(self, name: str) -> bool:
        """Check if buffer exists."""
        return name in self._buffers
    
    def update_buffer(self, name: str, data: np.ndarray) -> DeviceBuffer:
        """
        Update existing buffer with new data.
        
        Args:
            name: Buffer name
            data: New data (must match shape)
            
        Returns:
            Updated DeviceBuffer
        """
        with self._lock:
            if name not in self._buffers:
                # Create new buffer
                return self.allocate_buffer(name, data.shape, data.dtype, data)
            
            buffer = self._buffers[name]
            
            if not buffer.matches_shape(data.shape):
                if self.auto_reallocate:
                    return self.allocate_buffer(name, data.shape, data.dtype, data)
                else:
                    raise ValueError(f"Shape mismatch for buffer '{name}'")
            
            # Update OrtValue
            buffer.ort_value = self._create_ort_value(data)
            return buffer
    
    def create_binding(self, session: ort.InferenceSession) -> ort.IOBinding:
        """
        Create an IOBinding object for a session.
        
        Args:
            session: ONNX Runtime session
            
        Returns:
            IOBinding object for the session
        """
        return session.io_binding()
    
    def bind_inputs(
        self,
        io_binding: ort.IOBinding,
        inputs: Dict[str, Union[np.ndarray, str, DeviceBuffer]]
    ) -> None:
        """
        Bind inputs to the IOBinding.
        
        Args:
            io_binding: IOBinding object
            inputs: Dict mapping input names to:
                - np.ndarray: Will be copied to device
                - str: Name of existing buffer
                - DeviceBuffer: Direct binding
        """
        for name, value in inputs.items():
            if isinstance(value, np.ndarray):
                # Create OrtValue and bind
                ort_value = self._create_ort_value(value)
                io_binding.bind_ortvalue_input(name, ort_value)
            elif isinstance(value, str):
                # Look up buffer by name
                buffer = self._buffers.get(value)
                if buffer and buffer.ort_value:
                    io_binding.bind_ortvalue_input(name, buffer.ort_value)
                else:
                    raise ValueError(f"Buffer '{value}' not found")
            elif isinstance(value, DeviceBuffer):
                if value.ort_value:
                    io_binding.bind_ortvalue_input(name, value.ort_value)
                else:
                    raise ValueError(f"Buffer '{value.name}' not allocated")
            else:
                raise TypeError(f"Unsupported input type: {type(value)}")
    
    def bind_outputs(
        self,
        io_binding: ort.IOBinding,
        session: ort.InferenceSession,
        output_buffers: Optional[Dict[str, DeviceBuffer]] = None
    ) -> None:
        """
        Bind outputs to the IOBinding.
        
        If output_buffers provided, binds to those pre-allocated buffers.
        Otherwise, binds outputs to device memory (auto-allocated).
        
        Args:
            io_binding: IOBinding object
            session: Session to get output info from
            output_buffers: Optional dict of output name to DeviceBuffer
        """
        outputs = session.get_outputs()
        
        for output in outputs:
            name = output.name
            
            if output_buffers and name in output_buffers:
                buffer = output_buffers[name]
                if buffer.ort_value:
                    io_binding.bind_ortvalue_output(name, buffer.ort_value)
                else:
                    io_binding.bind_output(
                        name, 
                        device_type=self.device_type,
                        device_id=self.device_id
                    )
            else:
                # Bind to device memory (auto-allocate)
                io_binding.bind_output(
                    name,
                    device_type=self.device_type,
                    device_id=self.device_id
                )
    
    def get_bound_outputs(
        self,
        io_binding: ort.IOBinding,
        output_names: List[str]
    ) -> Dict[str, ort.OrtValue]:
        """
        Get output OrtValues after running with IOBinding.
        
        Args:
            io_binding: IOBinding after run_with_iobinding
            output_names: Names of outputs to retrieve
            
        Returns:
            Dict of output name to OrtValue
        """
        ort_outputs = io_binding.get_outputs()
        return {name: ort_val for name, ort_val in zip(output_names, ort_outputs)}
    
    def get_outputs_as_numpy(
        self,
        io_binding: ort.IOBinding,
        output_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Get outputs as numpy arrays (copies from device).
        
        Args:
            io_binding: IOBinding after run_with_iobinding
            output_names: Names of outputs to retrieve
            
        Returns:
            Dict of output name to numpy array
        """
        result = {}
        for name, ort_val in self.get_bound_outputs(io_binding, output_names).items():
            result[name] = ort_val.numpy()
        return result
    
    def run_with_binding(
        self,
        session: ort.InferenceSession,
        inputs: Dict[str, Union[np.ndarray, str, DeviceBuffer]]
    ) -> Dict[str, np.ndarray]:
        """
        Run inference with I/O binding (convenience method).
        
        Creates IOBinding, binds inputs/outputs, runs inference,
        and returns outputs as numpy arrays.
        
        Args:
            session: ONNX Runtime session to run
            inputs: Dict mapping input names to values
            
        Returns:
            Dict of output name to numpy array
        """
        io_binding = self.create_binding(session)
        self.bind_inputs(io_binding, inputs)
        self.bind_outputs(io_binding, session)
        session.run_with_iobinding(io_binding)
        output_names = [o.name for o in session.get_outputs()]
        return self.get_outputs_as_numpy(io_binding, output_names)
    
    def deallocate_buffer(self, name: str) -> bool:
        """
        Deallocate a buffer.
        
        Args:
            name: Buffer name
            
        Returns:
            True if buffer was deallocated
        """
        with self._lock:
            if name in self._buffers:
                del self._buffers[name]
                return True
            return False
    
    def deallocate_all(self) -> int:
        """
        Deallocate all buffers.
        
        Returns:
            Number of buffers deallocated
        """
        with self._lock:
            count = len(self._buffers)
            self._buffers.clear()
            return count
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dict with memory stats
        """
        with self._lock:
            total_bytes = 0
            per_buffer = {}
            
            for name, buffer in self._buffers.items():
                size = buffer.size_bytes
                total_bytes += size
                per_buffer[name] = {
                    "size_bytes": size,
                    "size_mb": buffer.size_mb,
                    "shape": buffer.shape,
                    "dtype": str(buffer.dtype),
                }
            
            return {
                "total_bytes": total_bytes,
                "total_mb": round(total_bytes / (1024 * 1024), 3),
                "buffer_count": len(self._buffers),
                "device_type": self.device_type,
                "device_id": self.device_id,
                "per_buffer": per_buffer,
            }
    
    def get_buffer_names(self) -> List[str]:
        """Get list of all buffer names."""
        return list(self._buffers.keys())
    
    def __len__(self) -> int:
        """Number of allocated buffers."""
        return len(self._buffers)
    
    def __contains__(self, name: str) -> bool:
        """Check if buffer exists."""
        return name in self._buffers
    
    def __repr__(self) -> str:
        mem = self.get_memory_usage()
        return (f"IOBindingManager(device={self.device_type}:{self.device_id}, "
                f"buffers={len(self)}, memory={mem['total_mb']}MB)")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_supported_devices() -> List[str]:
    """
    Get list of device types supported by current ONNX Runtime.
    
    Returns:
        List of device type strings
    """
    devices = ["cpu"]  # Always available
    
    providers = ort.get_available_providers()
    
    if "CUDAExecutionProvider" in providers:
        devices.append("cuda")
    if "DmlExecutionProvider" in providers:
        devices.append("dml")
    if "CoreMLExecutionProvider" in providers:
        devices.append("coreml")
    if "ROCMExecutionProvider" in providers:
        devices.append("rocm")
    
    return devices


def create_ortvalue_from_numpy(
    data: np.ndarray,
    device_type: str = "cpu",
    device_id: int = 0
) -> ort.OrtValue:
    """
    Create an OrtValue from numpy array.
    
    Convenience function for creating device tensors.
    
    Args:
        data: NumPy array
        device_type: Target device
        device_id: Device ID
        
    Returns:
        OrtValue on target device
    """
    return ort.OrtValue.ortvalue_from_numpy(data, device_type, device_id)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Classes
    "IOBindingManager",
    "DeviceBuffer",
    "MemoryMode",
    # Functions
    "get_supported_devices",
    "get_onnx_element_type",
    "create_ortvalue_from_numpy",
    # Constants
    "NUMPY_TO_ONNX_ELEMENT_TYPE",
    "ONNX_ELEMENT_TYPE_NAMES",
]
