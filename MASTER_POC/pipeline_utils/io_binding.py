#!/usr/bin/env python3
"""
io_binding.py - I/O Binding for Zero-Copy Inference

This module provides I/O binding utilities for ONNX Runtime that:
1. Keep tensors in device memory between operations (zero-copy)
2. Reduce CPU↔Device data transfer overhead
3. Support CPU, CUDA, and AMD heterogeneous targets
   (ROCm/HIP iGPU and VitisAI/RyzenAI XDNA NPU)

AMD memory placement:
- "rocm", "hip", "vitisai", and "ryzenai" are attempted as OrtValue
  devices. Upstream wheels that use vendor-aware devices are also
  tried ("gpu" or "npu" plus AMD's PCI vendor id 0x1002).
- If that build cannot place an OrtValue on the accelerator, allocation
  falls back to pinned host memory so the NPU DMA engine can read and
  write without a pageable CPU copy. The pinned candidates are
  "hip_pinned" and "rocm_pinned" (hipHostMalloc), then "cuda_pinned",
  then a HOST_ACCESSIBLE OrtMemoryInfo (HipPinned / CudaPinned).
- Pageable "cpu" memory is only the last candidate.

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
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Any, Union
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


class MemoryPlacement(Enum):
    """Where an OrtValue actually lives after device resolution."""

    DEVICE = "device"
    PINNED_HOST = "pinned_host"
    CPU = "cpu"


# PCI vendor ids from OrtDevice::VendorIds. Used when the wheel exposes
# generic "gpu" / "npu" devices instead of AMD's historical aliases.
AMD_PCI_VENDOR_ID = 0x1002
NVIDIA_PCI_VENDOR_ID = 0x10DE

# RDNA iGPU strings accepted by onnxruntime-rocm.
AMD_GPU_DEVICES = frozenset({"rocm", "hip"})
# XDNA NPU strings accepted by onnxruntime-vitisai / Ryzen AI wheels.
AMD_NPU_DEVICES = frozenset({"vitisai", "ryzenai"})
# Pinned host aliases. hip_pinned / rocm_pinned map to hipHostMalloc.
PINNED_HOST_DEVICES = frozenset({"cuda_pinned", "hip_pinned", "rocm_pinned"})

# OrtValue device strings this manager will attempt directly.
# AMD names stay in this set so they are never rewritten to "cpu" up front.
VALID_ORTVALUE_DEVICES = frozenset({
    "cpu",
    "cuda",
}) | AMD_GPU_DEVICES | AMD_NPU_DEVICES | PINNED_HOST_DEVICES


class _CapabilityGap(Exception):
    """The installed wheel cannot place an OrtValue with this memory info."""


def _amd_vendor_id() -> int:
    """Return AMD's PCI vendor id from the installed wheel when it exposes one."""
    enum = getattr(ort, "OrtDeviceVendorId", None)
    amd = getattr(enum, "AMD", None)
    if amd is not None:
        return int(amd)
    return AMD_PCI_VENDOR_ID


def _nvidia_vendor_id() -> int:
    """Return NVIDIA's PCI vendor id from the installed wheel when it exposes one."""
    enum = getattr(ort, "OrtDeviceVendorId", None)
    nvidia = getattr(enum, "NVIDIA", None)
    if nvidia is not None:
        return int(nvidia)
    return NVIDIA_PCI_VENDOR_ID


def _is_capability_gap(exc: BaseException) -> bool:
    """True when the failure means this build has no allocator for the device."""
    if isinstance(exc, _CapabilityGap):
        return True
    if isinstance(exc, TypeError):
        message = str(exc).lower()
        return any(
            token in message
            for token in ("argument", "positional", "unexpected", "keyword")
        )
    message = str(exc).lower()
    markers = (
        "unsupported device",
        "unknown device",
        "invalid device",
        "cannot place the ortvalue",
        "can't allocate memory",
        "cannot allocate memory",
        "no shared allocator",
        "appropriate package of onnxruntime",
        "please use the cann package",
        "please use the rocm package",
    )
    return any(marker in message for marker in markers)


@dataclass(frozen=True)
class OrtMemoryTarget:
    """One OrtValue memory-info attempt, in fallback order."""

    placement: MemoryPlacement
    device_type: str
    vendor_id: Optional[int] = None
    # Set when placement uses HOST_ACCESSIBLE OrtMemoryInfo (hipHostMalloc
    # equivalent on wheels that no longer accept "hip_pinned" / "cuda_pinned").
    memory_info_name: Optional[str] = None


def _dedupe_targets(targets: Sequence[OrtMemoryTarget]) -> List[OrtMemoryTarget]:
    seen = set()
    ordered: List[OrtMemoryTarget] = []
    for target in targets:
        key = (
            target.placement,
            target.device_type,
            target.vendor_id,
            target.memory_info_name,
        )
        if key in seen:
            continue
        seen.add(key)
        ordered.append(target)
    return ordered


def _pinned_string_targets(preferred: Optional[str] = None) -> List[OrtMemoryTarget]:
    """hipHostMalloc aliases first, then the portable cuda_pinned alias."""
    order = ["hip_pinned", "rocm_pinned", "cuda_pinned"]
    if preferred in order:
        order = [preferred] + [name for name in order if name != preferred]
    return [
        OrtMemoryTarget(MemoryPlacement.PINNED_HOST, name)
        for name in order
    ]


def _pinned_memory_info_targets(vendor_id: int) -> List[OrtMemoryTarget]:
    """HOST_ACCESSIBLE memory infos registered by AMD and CUDA execution providers."""
    names = ("HipPinned", "CudaPinned")
    nvidia = _nvidia_vendor_id()
    targets = []
    for name in names:
        info_vendor = vendor_id if name == "HipPinned" else nvidia
        targets.append(
            OrtMemoryTarget(
                MemoryPlacement.PINNED_HOST,
                name,
                vendor_id=info_vendor,
                memory_info_name=name,
            )
        )
    return targets


def ort_memory_candidates(device_type: str) -> List[OrtMemoryTarget]:
    """
    Ordered OrtValue placements for a logical device.

    The first candidate is the direct accelerator mapping. Pinned host
    memory follows when that mapping is unavailable. Pageable CPU is last.
    """
    key = (device_type or "cpu").lower()
    amd = _amd_vendor_id()
    cpu = [OrtMemoryTarget(MemoryPlacement.CPU, "cpu")]

    if key == "cpu":
        return cpu

    if key in AMD_GPU_DEVICES:
        direct = [
            OrtMemoryTarget(MemoryPlacement.DEVICE, key),
            OrtMemoryTarget(MemoryPlacement.DEVICE, "gpu", vendor_id=amd),
        ]
        pinned = _pinned_string_targets() + _pinned_memory_info_targets(amd)
        return _dedupe_targets(direct + pinned + cpu)

    if key in AMD_NPU_DEVICES:
        direct = [
            OrtMemoryTarget(MemoryPlacement.DEVICE, key),
            OrtMemoryTarget(MemoryPlacement.DEVICE, "npu", vendor_id=amd),
        ]
        pinned = _pinned_string_targets() + _pinned_memory_info_targets(amd)
        return _dedupe_targets(direct + pinned + cpu)

    if key == "cuda":
        direct = [OrtMemoryTarget(MemoryPlacement.DEVICE, "cuda")]
        pinned = [
            OrtMemoryTarget(MemoryPlacement.PINNED_HOST, "cuda_pinned"),
            OrtMemoryTarget(
                MemoryPlacement.PINNED_HOST,
                "CudaPinned",
                vendor_id=_nvidia_vendor_id(),
                memory_info_name="CudaPinned",
            ),
        ]
        return _dedupe_targets(direct + pinned + cpu)

    if key in PINNED_HOST_DEVICES:
        pinned = _pinned_string_targets(preferred=key) + _pinned_memory_info_targets(amd)
        return _dedupe_targets(pinned + cpu)

    # Future EP names are attempted as given, then pinned, then CPU.
    direct = [OrtMemoryTarget(MemoryPlacement.DEVICE, key)]
    pinned = _pinned_string_targets() + _pinned_memory_info_targets(amd)
    return _dedupe_targets(direct + pinned + cpu)


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
        device_type: Logical device ("cpu", "cuda", "rocm", "vitisai", ...)
        device_id: Device ID (0, 1, etc.)
        ort_value: The underlying OrtValue (None if not allocated)
        is_dynamic: If True, shape can change between inferences
        ortvalue_device: Memory-info string used to allocate ort_value
        memory_placement: device, pinned_host, or cpu
        vendor_id: PCI vendor id passed to OrtValue, when the wheel uses one
    """
    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype
    device_type: str = "cpu"
    device_id: int = 0
    ort_value: Optional[ort.OrtValue] = None
    is_dynamic: bool = False
    ortvalue_device: str = "cpu"
    memory_placement: MemoryPlacement = MemoryPlacement.CPU
    vendor_id: Optional[int] = None
    
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
            "ortvalue_device": self.ortvalue_device,
            "memory_placement": (
                self.memory_placement.value
                if isinstance(self.memory_placement, MemoryPlacement)
                else self.memory_placement
            ),
            "vendor_id": self.vendor_id,
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
        manager = IOBindingManager(device_type="vitisai", device_id=0)
        
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
            device_type: Logical device. AMD targets are "rocm", "hip",
                "vitisai", and "ryzenai". Also accepts "cpu", "cuda",
                "cuda_pinned", "hip_pinned", and "rocm_pinned".
            device_id: Device ID for multi-GPU/NPU systems
            auto_reallocate: If True, automatically reallocate buffers on shape mismatch
        """
        self.device_type = (device_type or "cpu").lower()
        self.device_id = device_id
        self.auto_reallocate = auto_reallocate

        # Preferred memory info is the caller's device. AMD names in
        # VALID_ORTVALUE_DEVICES are kept. The resolved placement is filled
        # in on the first allocation, after direct and pinned attempts.
        self._candidates = ort_memory_candidates(self.device_type)
        preferred = self._candidates[0]
        self._memory_target: Optional[OrtMemoryTarget] = None
        # Names in VALID_ORTVALUE_DEVICES, including rocm/hip/vitisai/ryzenai,
        # are preserved here. Pageable CPU is not substituted at construction.
        if self.device_type in VALID_ORTVALUE_DEVICES:
            self._ortvalue_device = self.device_type
        else:
            self._ortvalue_device = preferred.device_type
        self._ortvalue_vendor_id = preferred.vendor_id
        self._memory_placement = preferred.placement

        self._buffers: Dict[str, DeviceBuffer] = {}
        self._lock = threading.RLock()  # Reentrant lock to prevent deadlock
    
    @property
    def ortvalue_device(self) -> str:
        """Memory-info string of the preferred or resolved OrtValue device."""
        return self._ortvalue_device

    @property
    def memory_placement(self) -> MemoryPlacement:
        """device, pinned_host, or cpu. Preferred until the first allocation."""
        return self._memory_placement

    @property
    def ortvalue_vendor_id(self) -> Optional[int]:
        """PCI vendor id for the resolved memory info, if one is required."""
        return self._ortvalue_vendor_id

    @property
    def memory_resolved(self) -> bool:
        """True after the first allocation has selected a working memory info."""
        return self._memory_target is not None

    def _create_ort_value(
        self,
        data: np.ndarray,
        device_type: Optional[str] = None,
        device_id: Optional[int] = None
    ) -> ort.OrtValue:
        """
        Create an OrtValue from numpy array on the resolved device.

        Direct accelerator placement is tried first. Pinned host memory
        (hipHostMalloc aliases, then cuda_pinned) is used when this wheel
        cannot allocate an OrtValue on that device.
        """
        return self._allocate_ort_value(
            array=data,
            device_type=device_type,
            device_id=device_id,
        )

    def _create_empty_ort_value(
        self,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        device_type: Optional[str] = None,
        device_id: Optional[int] = None
    ) -> ort.OrtValue:
        """
        Create an empty OrtValue buffer on the resolved device.
        """
        return self._allocate_ort_value(
            shape=tuple(shape),
            dtype=dtype,
            device_type=device_type,
            device_id=device_id,
        )

    def _ensure_memory_target(self) -> OrtMemoryTarget:
        """Resolve memory info with a 1-element probe if allocation has not run."""
        if self._memory_target is None:
            self._allocate_ort_value(array=np.zeros((1,), dtype=np.float32))
        assert self._memory_target is not None
        return self._memory_target

    def _output_bind_device(self) -> str:
        """
        Device string for IOBinding.bind_output.

        String-based placements are passed through. HOST_ACCESSIBLE memory
        infos have no bind_output alias; preallocated OrtValues carry those.
        """
        target = self._ensure_memory_target()
        if target.memory_info_name:
            return "cpu"
        return target.device_type

    def _allocate_ort_value(
        self,
        array: Optional[np.ndarray] = None,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[np.dtype] = None,
        device_type: Optional[str] = None,
        device_id: Optional[int] = None,
    ) -> ort.OrtValue:
        """Walk the memory-info chain and cache the first placement that works."""
        requested = (device_type or self.device_type).lower()
        dev_id = self.device_id if device_id is None else device_id
        use_cache = requested == self.device_type and dev_id == self.device_id

        with self._lock:
            if use_cache and self._memory_target is not None:
                return self._place(self._memory_target, array, shape, dtype, dev_id)

            candidates = (
                self._candidates if use_cache else ort_memory_candidates(requested)
            )
            last_error: Optional[BaseException] = None
            for target in candidates:
                try:
                    value = self._place(target, array, shape, dtype, dev_id)
                except Exception as exc:
                    if target.placement == MemoryPlacement.DEVICE and not _is_capability_gap(exc):
                        raise
                    if target.placement == MemoryPlacement.CPU and not _is_capability_gap(exc):
                        raise
                    last_error = exc
                    continue

                if use_cache:
                    self._remember_target(target)
                return value

            raise RuntimeError(
                f"Unable to allocate an OrtValue for device '{requested}'. "
                f"Tried {[target.device_type for target in candidates]}. "
                f"Last error: {last_error}"
            )

    def _remember_target(self, target: OrtMemoryTarget) -> None:
        preferred = self._candidates[0]
        if (
            self._memory_target is None
            and target.placement != preferred.placement
        ):
            warnings.warn(
                f"IOBindingManager({self.device_type}) placed tensors on "
                f"{target.device_type} ({target.placement.value}) because "
                f"direct OrtValue allocation on {preferred.device_type} is "
                f"unavailable in this ONNX Runtime build.",
                RuntimeWarning,
                stacklevel=4,
            )
        self._memory_target = target
        self._ortvalue_device = target.device_type
        self._ortvalue_vendor_id = target.vendor_id
        self._memory_placement = target.placement

    def _place(
        self,
        target: OrtMemoryTarget,
        array: Optional[np.ndarray],
        shape: Optional[Tuple[int, ...]],
        dtype: Optional[np.dtype],
        device_id: int,
    ) -> ort.OrtValue:
        if target.memory_info_name:
            return self._place_host_accessible(target, array, shape, dtype, device_id)
        if array is not None:
            return self._from_numpy(array, target, device_id)
        if shape is None or dtype is None:
            raise ValueError("shape and dtype are required to allocate an empty OrtValue")
        element_type = get_onnx_element_type(dtype)
        return self._from_shape(list(shape), element_type, target, device_id)

    def _from_numpy(
        self,
        array: np.ndarray,
        target: OrtMemoryTarget,
        device_id: int,
    ) -> ort.OrtValue:
        factory = ort.OrtValue.ortvalue_from_numpy
        if target.vendor_id is None:
            return factory(array, target.device_type, device_id)
        try:
            return factory(array, target.device_type, device_id, target.vendor_id)
        except TypeError as exc:
            if _is_capability_gap(exc):
                raise _CapabilityGap(str(exc)) from exc
            raise

    def _from_shape(
        self,
        shape: List[int],
        element_type: int,
        target: OrtMemoryTarget,
        device_id: int,
    ) -> ort.OrtValue:
        factory = ort.OrtValue.ortvalue_from_shape_and_type
        if target.vendor_id is None:
            return factory(shape, element_type, target.device_type, device_id)
        try:
            return factory(shape, element_type, target.device_type, device_id, target.vendor_id)
        except TypeError as exc:
            if _is_capability_gap(exc):
                raise _CapabilityGap(str(exc)) from exc
            raise

    def _place_host_accessible(
        self,
        target: OrtMemoryTarget,
        array: Optional[np.ndarray],
        shape: Optional[Tuple[int, ...]],
        dtype: Optional[np.dtype],
        device_id: int,
    ) -> ort.OrtValue:
        """
        Allocate HOST_ACCESSIBLE memory.

        This is the hipHostMalloc / cudaHostAlloc equivalent on wheels that
        dropped the "hip_pinned" and "cuda_pinned" device strings. The
        execution provider must have registered a shared allocator for the
        memory-info name (HipPinned or CudaPinned).
        """
        try:
            from onnxruntime.capi import _pybind_state as capi
        except ImportError as exc:
            raise _CapabilityGap("onnxruntime C API is unavailable") from exc

        memory_types = getattr(capi, "OrtDeviceMemoryType", None)
        host_accessible = getattr(memory_types, "HOST_ACCESSIBLE", None)
        info_cls = getattr(capi, "OrtMemoryInfo", None)
        if host_accessible is None or info_cls is None or not hasattr(info_cls, "create_v2"):
            raise _CapabilityGap("HOST_ACCESSIBLE OrtMemoryInfo is unavailable")

        vendor = 0 if target.vendor_id is None else int(target.vendor_id)
        try:
            memory_info = info_cls.create_v2(
                target.memory_info_name,
                capi.OrtMemoryInfoDeviceType.CPU,
                vendor,
                int(device_id),
                host_accessible,
                0,
                capi.OrtAllocatorType.ORT_DEVICE_ALLOCATOR,
            )
        except Exception as exc:
            raise _CapabilityGap(str(exc)) from exc

        if array is not None:
            value = self._ortvalue_from_memory_info(
                list(array.shape),
                array.dtype,
                memory_info,
            )
            if not hasattr(value, "update_inplace"):
                raise _CapabilityGap("pinned OrtValue does not support update_inplace")
            value.update_inplace(np.ascontiguousarray(array))
            return value

        if shape is None or dtype is None:
            raise ValueError("shape and dtype are required to allocate an empty OrtValue")
        numpy_dtype = np.dtype(dtype)
        return self._ortvalue_from_memory_info(list(shape), numpy_dtype, memory_info)

    def _ortvalue_from_memory_info(
        self,
        shape: List[int],
        element_type: Any,
        memory_info: Any,
    ) -> ort.OrtValue:
        factory = ort.OrtValue.ortvalue_from_shape_and_type
        try:
            return factory(shape, element_type, memory_info=memory_info)
        except TypeError as exc:
            if "memory_info" not in str(exc):
                raise
        except Exception as exc:
            if _is_capability_gap(exc):
                raise _CapabilityGap(str(exc)) from exc
            raise

        try:
            from onnxruntime.capi import _pybind_state as capi
            raw = capi.OrtValue.ortvalue_from_shape_and_type_for_memory_info(
                shape, element_type, memory_info
            )
            return ort.OrtValue(raw)
        except Exception as exc:
            raise _CapabilityGap(str(exc)) from exc
    
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
                is_dynamic=is_dynamic,
                ortvalue_device=self._ortvalue_device,
                memory_placement=self._memory_placement,
                vendor_id=self._ortvalue_vendor_id,
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
                        device_type=self._output_bind_device(),
                        device_id=self.device_id
                    )
            else:
                # Bind to the resolved device (accelerator, pinned host, or CPU)
                io_binding.bind_output(
                    name,
                    device_type=self._output_bind_device(),
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
                "ortvalue_device": self._ortvalue_device,
                "memory_placement": self._memory_placement.value,
                "vendor_id": self._ortvalue_vendor_id,
                "memory_resolved": self._memory_target is not None,
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
                f"ortvalue={self._ortvalue_device}/{self._memory_placement.value}, "
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
        devices.append("cuda_pinned")
    if "DmlExecutionProvider" in providers:
        devices.append("dml")
    if "CoreMLExecutionProvider" in providers:
        devices.append("coreml")
    if "ROCMExecutionProvider" in providers or "MIGraphXExecutionProvider" in providers:
        devices.extend(["rocm", "hip", "hip_pinned", "rocm_pinned"])
    if "VitisAIExecutionProvider" in providers:
        devices.append("vitisai")
    if "RyzenAIExecutionProvider" in providers:
        devices.append("ryzenai")

    return devices


def create_ortvalue_from_numpy(
    data: np.ndarray,
    device_type: str = "cpu",
    device_id: int = 0
) -> ort.OrtValue:
    """
    Create an OrtValue from numpy array.

    Uses the same AMD device chain as IOBindingManager: direct placement,
    then pinned host memory, then pageable CPU.
    """
    manager = IOBindingManager(device_type=device_type, device_id=device_id)
    return manager._create_ort_value(data)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Classes
    "IOBindingManager",
    "DeviceBuffer",
    "MemoryMode",
    "MemoryPlacement",
    "OrtMemoryTarget",
    # Functions
    "get_supported_devices",
    "get_onnx_element_type",
    "create_ortvalue_from_numpy",
    "ort_memory_candidates",
    # Constants
    "NUMPY_TO_ONNX_ELEMENT_TYPE",
    "ONNX_ELEMENT_TYPE_NAMES",
    "VALID_ORTVALUE_DEVICES",
    "AMD_PCI_VENDOR_ID",
]
