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

Heterogeneous prefill/decode:
- IOBindingManager is one device. Splitting prefill onto the NPU and
  decode onto the iGPU needs one buffer both sessions bind.
- HeterogeneousIOBindingManager.allocate_shared_buffer() allocates that
  crossing buffer (KV cache, hidden states, logits) once: hipMallocManaged
  when the Ryzen AI driver provides it, otherwise page-locked host memory,
  otherwise one CPU OrtValue.
- bind_phase_bridge() binds that same address as the NPU prefill output
  and the iGPU decode input. Private weights stay on the prefill and
  decode managers.

NPU static KV cache:
- MemoryMode.NPU_STATIC disables OrtValue reallocation. XDNA XCLBIN
  graphs are compiled for one shape.
- allocate_static_kv_cache() allocates keys and values at max_seq_len
  (default 4096). attention_mask(valid_tokens) rewrites a fixed
  (batch, max_seq_len) mask: 1 for the written prefix, 0 for padding.

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

import ctypes
import os
import threading
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any, Union
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

    NPU_STATIC:
        - KV cache is allocated once at max sequence length
        - Sequence growth updates an attention mask, not the OrtValue shape
        - Required for XDNA XCLBIN graphs, which are compiled ahead of time
        - Best for: VitisAI / Ryzen AI prefill and decode
    """
    MINIMAL = "minimal"
    PERSISTENT = "persistent"
    IO_BOUND = "io_bound"
    NPU_STATIC = "npu_static"
    
    @classmethod
    def get_description(cls, mode: 'MemoryMode') -> str:
        """Get human-readable description of memory mode."""
        descriptions = {
            cls.MINIMAL: "RAM-efficient: Load/unload per component (slow)",
            cls.PERSISTENT: "Balanced: All sessions resident (medium RAM, fast)",
            cls.IO_BOUND: "Throughput-optimized: Persistent + zero-copy (fast)",
            cls.NPU_STATIC: "NPU AOT: KV pool at max sequence length, attention mask hides padding",
        }
        return descriptions.get(mode, "Unknown mode")


# Default context compiled into an XDNA KV pool when the caller does not
# pass a maximum. 4096 matches a typical Ryzen AI generation window.
DEFAULT_NPU_MAX_SEQ_LEN = 4096


class StaticShapeError(ValueError):
    """Resizing this OrtValue would reload or crash an AOT NPU graph."""


def kv_cache_shape(
    batch_size: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
    seq_axis: int = 2,
) -> Tuple[int, ...]:
    """Build a 4D KV shape with ``seq_len`` placed on ``seq_axis``."""
    if seq_axis < 0 or seq_axis > 3:
        raise ValueError(f"seq_axis must be in 0..3, got {seq_axis}")
    remaining = [batch_size, num_kv_heads, head_dim]
    shape: List[int] = []
    cursor = 0
    for axis in range(4):
        if axis == seq_axis:
            shape.append(seq_len)
        else:
            shape.append(remaining[cursor])
            cursor += 1
    return tuple(shape)


@dataclass
class StaticKVPool:
    """KV cache preallocated at the maximum sequence length an XCLBIN can run."""

    name: str
    max_seq_len: int
    batch_size: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: np.dtype
    seq_axis: int
    valid_seq_len: int
    key_names: List[str]
    value_names: List[str]
    mask_name: str

    def __post_init__(self) -> None:
        if isinstance(self.dtype, str):
            self.dtype = np.dtype(self.dtype)

    @property
    def kv_shape(self) -> Tuple[int, ...]:
        return kv_cache_shape(
            self.batch_size,
            self.num_kv_heads,
            self.max_seq_len,
            self.head_dim,
            self.seq_axis,
        )

    def padding_mask(self) -> np.ndarray:
        """
        Attention mask of shape ``(batch, max_seq_len)``.

        1 marks tokens the NPU may attend to. 0 marks padded or unwritten
        positions in the preallocated KV cache.
        """
        mask = np.zeros((self.batch_size, self.max_seq_len), dtype=np.int64)
        if self.valid_seq_len > 0:
            mask[:, : self.valid_seq_len] = 1
        return mask


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
    static_seq_axis: Optional[int] = None
    static_max_seq: Optional[int] = None
    
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

    def fits_static_request(self, shape: Tuple[int, ...]) -> bool:
        """
        True when ``shape`` is this buffer with a shorter sequence axis.

        The allocated OrtValue stays at ``static_max_seq``. A shorter request
        is the valid prefix, not a new allocation.
        """
        if self.static_seq_axis is None or self.static_max_seq is None:
            return False
        requested = tuple(shape)
        if len(requested) != len(self.shape):
            return False
        for index, (allocated, want) in enumerate(zip(self.shape, requested)):
            if index == self.static_seq_axis:
                if want <= 0 or want > allocated:
                    return False
            elif allocated != want:
                return False
        return True
    
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
            "static_seq_axis": self.static_seq_axis,
            "static_max_seq": self.static_max_seq,
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
        auto_reallocate: bool = True,
        memory_mode: Union[MemoryMode, str] = MemoryMode.IO_BOUND,
        max_seq_len: int = DEFAULT_NPU_MAX_SEQ_LEN,
    ):
        """
        Initialize the I/O binding manager.
        
        Args:
            device_type: Logical device. AMD targets are "rocm", "hip",
                "vitisai", and "ryzenai". Also accepts "cpu", "cuda",
                "cuda_pinned", "hip_pinned", and "rocm_pinned".
            device_id: Device ID for multi-GPU/NPU systems
            auto_reallocate: If True, automatically reallocate buffers on shape mismatch.
                Ignored for MemoryMode.NPU_STATIC.
            memory_mode: IO_BOUND allows dynamic shapes. NPU_STATIC keeps KV
                OrtValues at max_seq_len for an AOT XCLBIN.
            max_seq_len: Maximum sequence length used by allocate_static_kv_cache.
        """
        self.device_type = (device_type or "cpu").lower()
        self.device_id = device_id
        if isinstance(memory_mode, str):
            memory_mode = MemoryMode(memory_mode)
        self.memory_mode = memory_mode
        self.max_seq_len = int(max_seq_len)
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        # XDNA graphs are compiled for one shape. A later OrtValue resize
        # reloads that graph or faults the execution provider.
        self.auto_reallocate = (
            False if memory_mode == MemoryMode.NPU_STATIC else auto_reallocate
        )
        self._kv_pools: Dict[str, StaticKVPool] = {}
        self._buffer_pool: Dict[str, str] = {}

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
        A static KV buffer also matches a shorter sequence length: the
        OrtValue stays at max_seq_len and the attention mask marks the
        valid prefix.
        If shape differs and auto_reallocate is True, reallocates.
        MemoryMode.NPU_STATIC never reallocates.
        
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
                if existing.matches_shape(shape) or existing.fits_static_request(shape):
                    self._accept_static_shape(existing, shape, data)
                    return existing
                self._reject_or_reallocate(existing, shape)
            
            # Allocate new buffer. NPU_STATIC marks it fixed so the next
            # token cannot resize it.
            return self.allocate_buffer(
                name,
                shape,
                dtype,
                data,
                is_dynamic=self.auto_reallocate,
            )
    
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
                return self.allocate_buffer(
                    name,
                    data.shape,
                    data.dtype,
                    data,
                    is_dynamic=self.auto_reallocate,
                )
            
            buffer = self._buffers[name]
            if buffer.matches_shape(data.shape) or buffer.fits_static_request(data.shape):
                self._accept_static_shape(buffer, tuple(data.shape), data)
                return buffer
            self._reject_or_reallocate(buffer, tuple(data.shape))
            return self.allocate_buffer(name, data.shape, data.dtype, data)

    def _reject_or_reallocate(self, existing: DeviceBuffer, shape: Tuple[int, ...]) -> None:
        """Raise for NPU static buffers. Dynamic modes fall through and reallocate."""
        if existing.static_max_seq is not None or self.memory_mode == MemoryMode.NPU_STATIC or not self.auto_reallocate:
            raise StaticShapeError(
                f"Refusing to reallocate '{existing.name}' from {existing.shape} "
                f"to {tuple(shape)}. MemoryMode.{self.memory_mode.name} keeps the "
                f"OrtValue shape fixed for an AOT NPU graph. Grow the valid token "
                f"count with attention_mask() inside max_seq_len="
                f"{existing.static_max_seq or self.max_seq_len}."
            )

    def _accept_static_shape(
        self,
        existing: DeviceBuffer,
        shape: Tuple[int, ...],
        data: Optional[np.ndarray],
    ) -> None:
        """Write into the existing OrtValue. Do not allocate a replacement."""
        if data is not None:
            if existing.static_seq_axis is not None and not existing.matches_shape(shape):
                self._write_prefix(existing, data)
            elif self.memory_mode == MemoryMode.NPU_STATIC or existing.static_max_seq is not None:
                self._write_into_existing(existing, data)
            else:
                existing.ort_value = self._create_ort_value(data)
        if existing.static_seq_axis is not None:
            self._sync_valid_length(existing, int(shape[existing.static_seq_axis]))

    def _write_into_existing(self, buffer: DeviceBuffer, data: np.ndarray) -> None:
        if buffer.ort_value is None:
            raise RuntimeError(f"Buffer '{buffer.name}' is not allocated")
        host = buffer.ort_value.numpy()
        incoming = np.ascontiguousarray(data, dtype=host.dtype)
        if incoming.shape != host.shape:
            raise StaticShapeError(
                f"Cannot write shape {incoming.shape} into '{buffer.name}' "
                f"allocated as {host.shape}"
            )
        np.copyto(host, incoming)
        if int(host.ctypes.data) != int(buffer.ort_value.data_ptr()):
            buffer.ort_value.update_inplace(np.ascontiguousarray(host))

    def _write_prefix(self, buffer: DeviceBuffer, data: np.ndarray) -> None:
        """Copy a shorter sequence into the prefix of a max-length KV buffer."""
        if buffer.ort_value is None or buffer.static_seq_axis is None:
            raise RuntimeError(f"Buffer '{buffer.name}' is not a static KV buffer")
        host = buffer.ort_value.numpy()
        incoming = np.ascontiguousarray(data, dtype=host.dtype)
        axis = buffer.static_seq_axis
        if incoming.ndim != host.ndim:
            raise StaticShapeError(
                f"Rank mismatch for '{buffer.name}': allocated {host.shape}, "
                f"got {incoming.shape}"
            )
        for index, (allocated, requested) in enumerate(zip(host.shape, incoming.shape)):
            if index == axis:
                if requested <= 0 or requested > allocated:
                    raise StaticShapeError(
                        f"Sequence {requested} does not fit '{buffer.name}' "
                        f"capacity {allocated}"
                    )
            elif allocated != requested:
                raise StaticShapeError(
                    f"Axis {index} of '{buffer.name}' is fixed at {allocated}, got {requested}"
                )
        slices: List[slice] = [slice(None)] * host.ndim
        slices[axis] = slice(0, incoming.shape[axis])
        host[tuple(slices)] = incoming
        if int(host.ctypes.data) != int(buffer.ort_value.data_ptr()):
            buffer.ort_value.update_inplace(np.ascontiguousarray(host))

    def _sync_valid_length(self, buffer: DeviceBuffer, valid_seq_len: int) -> None:
        pool_name = self._buffer_pool.get(buffer.name)
        if pool_name is None:
            return
        pool = self._kv_pools[pool_name]
        if buffer.name == pool.mask_name:
            return
        self.set_valid_sequence_length(valid_seq_len, pool_name)

    def allocate_static_kv_cache(
        self,
        name: str = "kv",
        num_layers: int = 1,
        num_kv_heads: int = 1,
        head_dim: int = 1,
        dtype: np.dtype = np.float16,
        batch_size: int = 1,
        max_seq_len: Optional[int] = None,
        seq_axis: int = 2,
    ) -> StaticKVPool:
        """
        Allocate key and value OrtValues at the maximum sequence length.

        The sequence axis does not change during generation. Call
        ``attention_mask(valid_tokens)`` so the NPU ignores padded positions.
        """
        with self._lock:
            if name in self._kv_pools:
                raise ValueError(f"Static KV pool '{name}' already exists")
            if min(num_layers, num_kv_heads, head_dim, batch_size) <= 0:
                raise ValueError("KV pool dimensions must be positive")
            limit = self.max_seq_len if max_seq_len is None else int(max_seq_len)
            if limit <= 0:
                raise ValueError(f"max_seq_len must be positive, got {limit}")
            if isinstance(dtype, str):
                dtype = np.dtype(dtype)

            pool = StaticKVPool(
                name=name,
                max_seq_len=limit,
                batch_size=batch_size,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
                seq_axis=seq_axis,
                valid_seq_len=0,
                key_names=[],
                value_names=[],
                mask_name=f"{name}.attention_mask",
            )
            kv_shape = pool.kv_shape
            for layer in range(num_layers):
                key_name = f"{name}.layer{layer}.key"
                value_name = f"{name}.layer{layer}.value"
                for tensor_name in (key_name, value_name):
                    buffer = self.allocate_buffer(
                        tensor_name,
                        kv_shape,
                        dtype,
                        is_dynamic=False,
                    )
                    buffer.static_seq_axis = seq_axis
                    buffer.static_max_seq = limit
                    self._buffer_pool[tensor_name] = name
                pool.key_names.append(key_name)
                pool.value_names.append(value_name)

            mask = pool.padding_mask()
            mask_buffer = self.allocate_buffer(
                pool.mask_name,
                mask.shape,
                mask.dtype,
                data=mask,
                is_dynamic=False,
            )
            mask_buffer.static_seq_axis = 1
            mask_buffer.static_max_seq = limit
            self._buffer_pool[pool.mask_name] = name
            self._kv_pools[name] = pool
            return pool

    def set_valid_sequence_length(self, valid_seq_len: int, pool: str = "kv") -> np.ndarray:
        """
        Mark how many KV positions contain real tokens.

        Rewrites the preallocated attention mask in place. The KV OrtValues
        keep their maximum shape.
        """
        with self._lock:
            if pool not in self._kv_pools:
                raise KeyError(f"Static KV pool '{pool}' is not allocated")
            spec = self._kv_pools[pool]
            if valid_seq_len < 0 or valid_seq_len > spec.max_seq_len:
                raise StaticShapeError(
                    f"valid sequence length {valid_seq_len} is outside the "
                    f"preallocated KV pool '{pool}' of {spec.max_seq_len}. "
                    f"Refusing to reallocate the XCLBIN tensors."
                )
            spec.valid_seq_len = int(valid_seq_len)
            mask = spec.padding_mask()
            mask_buffer = self._buffers[spec.mask_name]
            self._write_into_existing(mask_buffer, mask)
            return mask

    def attention_mask(self, valid_seq_len: int, pool: str = "kv") -> np.ndarray:
        """
        Return the static attention mask for ``valid_seq_len`` tokens.

        Shape is ``(batch, max_seq_len)``. Ones cover the written prefix.
        Zeros tell the NPU to ignore the rest of the preallocated KV cache.
        """
        return self.set_valid_sequence_length(valid_seq_len, pool)

    def get_static_kv_pool(self, name: str = "kv") -> StaticKVPool:
        """Return a previously allocated static KV pool."""
        if name not in self._kv_pools:
            raise KeyError(f"Static KV pool '{name}' is not allocated")
        return self._kv_pools[name]
    
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
                self._buffer_pool.pop(name, None)
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
            self._kv_pools.clear()
            self._buffer_pool.clear()
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
                "memory_mode": self.memory_mode.value,
                "max_seq_len": self.max_seq_len,
                "auto_reallocate": self.auto_reallocate,
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
                f"mode={self.memory_mode.value}, "
                f"ortvalue={self._ortvalue_device}/{self._memory_placement.value}, "
                f"buffers={len(self)}, memory={mem['total_mb']}MB)")


# =============================================================================
# HETEROGENEOUS NPU / iGPU BRIDGE
# =============================================================================

class SharedAllocationKind(Enum):
    """How a buffer shared by the NPU and the iGPU was allocated."""

    UNIFIED = "unified"
    PINNED_HOST = "pinned_host"
    CPU = "cpu"


# hipMemAttachGlobal: managed memory is visible to every device in the process.
_HIP_MEM_ATTACH_GLOBAL = 0x01
# Portable, device-mapped, and coherent page-locked host memory.
_HIP_HOST_MALLOC_SHARED = 1 | 2 | 16
_HIP_LIB_NAMES = (
    "libamdhip64.so.6",
    "libamdhip64.so.5",
    "libamdhip64.so",
    "amdhip64.dll",
)
# Device strings that name private EP memory. A shared host pointer must
# not be bound under these names.
_PRIVATE_EP_DEVICES = AMD_NPU_DEVICES | AMD_GPU_DEVICES | frozenset({
    "cuda",
    "gpu",
    "npu",
})

_hip_library_lock = threading.Lock()
_hip_library_handle: Any = None
_hip_library_probed = False


@dataclass
class OsSharedAllocation:
    """One OS allocation whose address both the NPU and the iGPU can use."""

    ptr: int
    kind: SharedAllocationKind
    release: Callable[[], None]
    keepalive: Any = None


def _load_hip_library() -> Any:
    """Return the HIP runtime library, or None when this process has no APU driver."""
    global _hip_library_handle, _hip_library_probed
    with _hip_library_lock:
        if _hip_library_probed:
            return _hip_library_handle
        _hip_library_probed = True
        for name in _HIP_LIB_NAMES:
            try:
                library = ctypes.CDLL(name)
            except OSError:
                continue
            _configure_hip_library(library)
            _hip_library_handle = library
            return library
        return None


def _configure_hip_library(library: Any) -> None:
    if hasattr(library, "hipMallocManaged"):
        library.hipMallocManaged.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        library.hipMallocManaged.restype = ctypes.c_int
    if hasattr(library, "hipHostMalloc"):
        library.hipHostMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        library.hipHostMalloc.restype = ctypes.c_int
    if hasattr(library, "hipFree"):
        library.hipFree.argtypes = [ctypes.c_void_p]
        library.hipFree.restype = ctypes.c_int
    if hasattr(library, "hipHostFree"):
        library.hipHostFree.argtypes = [ctypes.c_void_p]
        library.hipHostFree.restype = ctypes.c_int


def _hip_allocate(
    library: Any,
    symbol: str,
    nbytes: int,
    flags: int,
    free_symbol: str,
    kind: SharedAllocationKind,
) -> Optional[OsSharedAllocation]:
    allocator = getattr(library, symbol, None)
    freer = getattr(library, free_symbol, None)
    if allocator is None or freer is None:
        return None
    holder = ctypes.c_void_p()
    status = allocator(
        ctypes.byref(holder),
        ctypes.c_size_t(nbytes),
        ctypes.c_uint(flags),
    )
    if status != 0 or not holder.value:
        return None
    address = int(holder.value)

    def release(addr: int = address, free_fn: Any = freer) -> None:
        free_fn(ctypes.c_void_p(addr))

    return OsSharedAllocation(ptr=address, kind=kind, release=release, keepalive=holder)


def allocate_os_shared_memory(nbytes: int) -> Optional[OsSharedAllocation]:
    """
    Allocate memory the NPU DMA engine and the iGPU can both address.

    Prefers hipMallocManaged (APU unified memory). Falls back to
    hipHostMalloc page-locked memory. Returns None when the HIP runtime
    is not installed; the caller then uses an OrtValue allocator.
    """
    if nbytes <= 0:
        return None
    library = _load_hip_library()
    if library is None:
        return None
    managed = _hip_allocate(
        library,
        "hipMallocManaged",
        nbytes,
        _HIP_MEM_ATTACH_GLOBAL,
        "hipFree",
        SharedAllocationKind.UNIFIED,
    )
    if managed is not None:
        return managed
    return _hip_allocate(
        library,
        "hipHostMalloc",
        nbytes,
        _HIP_HOST_MALLOC_SHARED,
        "hipHostFree",
        SharedAllocationKind.PINNED_HOST,
    )


def _array_from_pointer(
    ptr: int,
    shape: Tuple[int, ...],
    dtype: np.dtype,
) -> Tuple[np.ndarray, Any]:
    """Writable numpy view of an externally owned address. The view does not free it."""
    dtype = np.dtype(dtype)
    nbytes = int(dtype.itemsize)
    for dim in shape:
        nbytes *= int(dim)
    raw = (ctypes.c_byte * nbytes).from_address(int(ptr))
    array = np.ndarray(shape, dtype=dtype, buffer=raw)
    return array, raw


@dataclass
class SharedBuffer:
    """
    One tensor address shared by NPU prefill and iGPU decode.

    prefill binds it as an output. decode binds it as an input.
    Both bindings use this OrtValue and this data_ptr.
    """

    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype
    ort_value: Optional[ort.OrtValue]
    allocation_kind: SharedAllocationKind
    data_ptr: int
    nbytes: int
    npu_device_type: str
    igpu_device_type: str
    bind_device_type: str = "cpu"
    _release: Optional[Callable[[], None]] = None
    _keepalive: Any = None
    _closed: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.dtype, str):
            self.dtype = np.dtype(self.dtype)

    @property
    def crosses_devices(self) -> bool:
        """True when the allocation is unified or page-locked, not pageable CPU."""
        return self.allocation_kind in (
            SharedAllocationKind.UNIFIED,
            SharedAllocationKind.PINNED_HOST,
        )

    def to_numpy(self) -> np.ndarray:
        if self.ort_value is None:
            raise RuntimeError(f"Shared buffer '{self.name}' is closed")
        return self.ort_value.numpy()

    def close(self) -> None:
        """Drop the OrtValue alias, then free an OS allocation once."""
        if self._closed:
            return
        self._closed = True
        self.ort_value = None
        self._keepalive = None
        release = self._release
        self._release = None
        if release is not None:
            release()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "allocation_kind": self.allocation_kind.value,
            "data_ptr": self.data_ptr,
            "nbytes": self.nbytes,
            "npu_device_type": self.npu_device_type,
            "igpu_device_type": self.igpu_device_type,
            "bind_device_type": self.bind_device_type,
            "crosses_devices": self.crosses_devices,
            "closed": self._closed,
        }


class HeterogeneousIOBindingManager:
    """
    Prefill on the NPU and decode on the iGPU, with one shared KV address.

    ``prefill`` and ``decode`` are single-device managers for tensors that
    stay on one execution provider (weights, private activations).
    Tensors that cross the phase boundary are allocated only through
    ``allocate_shared_buffer``. Two private managers must not each allocate
    a copy of the KV cache.
    """

    def __init__(
        self,
        npu_device_type: str = "vitisai",
        npu_device_id: int = 0,
        igpu_device_type: str = "rocm",
        igpu_device_id: int = 0,
        max_seq_len: int = DEFAULT_NPU_MAX_SEQ_LEN,
        prefill_memory_mode: Union[MemoryMode, str] = MemoryMode.NPU_STATIC,
    ):
        self.npu_device_type = (npu_device_type or "vitisai").lower()
        self.npu_device_id = npu_device_id
        self.igpu_device_type = (igpu_device_type or "rocm").lower()
        self.igpu_device_id = igpu_device_id
        self.max_seq_len = int(max_seq_len)
        # Prefill runs on the NPU, whose XCLBIN cannot be reallocated per token.
        # Decode on the iGPU keeps dynamic shapes.
        self.prefill = IOBindingManager(
            self.npu_device_type,
            npu_device_id,
            memory_mode=prefill_memory_mode,
            max_seq_len=self.max_seq_len,
        )
        self.decode = IOBindingManager(self.igpu_device_type, igpu_device_id)
        self._shared: Dict[str, SharedBuffer] = {}
        self._lock = threading.RLock()

    def allocate_shared_buffer(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        data: Optional[np.ndarray] = None,
    ) -> SharedBuffer:
        """
        Allocate one buffer both the NPU prefill session and the iGPU decode
        session bind.

        Order:
        1. hipMallocManaged unified memory, when the Ryzen AI HIP driver is present
        2. hipHostMalloc page-locked host memory
        3. One pinned OrtValue (hip_pinned, rocm_pinned, cuda_pinned, or HOST_ACCESSIBLE)
        4. One pageable CPU OrtValue, reported as allocation_kind ``cpu``

        The OrtValue aliases the OS address. A device copy is discarded.
        """
        with self._lock:
            if name in self._shared:
                raise ValueError(f"Shared buffer '{name}' already exists")
            if isinstance(dtype, str):
                dtype = np.dtype(dtype)
            else:
                dtype = np.dtype(dtype)
            parsed_shape = tuple(int(dim) for dim in shape)
            if not parsed_shape or any(dim <= 0 for dim in parsed_shape):
                raise ValueError(f"Shared buffer '{name}' needs a positive shape, got {shape}")

            source: Optional[np.ndarray] = None
            if data is not None:
                source = np.ascontiguousarray(data, dtype=dtype)
                if tuple(source.shape) != parsed_shape:
                    raise ValueError(
                        f"Shared buffer '{name}' data shape {source.shape} "
                        f"does not match {parsed_shape}"
                    )

            nbytes = int(dtype.itemsize)
            for dim in parsed_shape:
                nbytes *= dim

            buffer = self._buffer_from_os_memory(name, parsed_shape, dtype, source, nbytes)
            if buffer is None:
                buffer = self._buffer_from_pinned_ortvalue(name, parsed_shape, dtype, source)
            if buffer.allocation_kind == SharedAllocationKind.CPU:
                warnings.warn(
                    f"Shared buffer '{name}' is one pageable CPU allocation at "
                    f"{buffer.data_ptr:#x}. hipMallocManaged and page-locked "
                    f"host memory are unavailable in this process. Prefill and "
                    f"decode still bind this same address.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self._shared[name] = buffer
            return buffer

    def _buffer_from_os_memory(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        data: Optional[np.ndarray],
        nbytes: int,
    ) -> Optional[SharedBuffer]:
        os_alloc = allocate_os_shared_memory(nbytes)
        if os_alloc is None:
            return None
        array = None
        raw = None
        ort_value = None
        try:
            array, raw = _array_from_pointer(os_alloc.ptr, shape, dtype)
            if data is not None:
                np.copyto(array, data)
            else:
                array.fill(0)
            # CPU OrtValue aliases the numpy view. A non-CPU device would copy
            # and drop the unified address.
            ort_value = ort.OrtValue.ortvalue_from_numpy(array, "cpu", 0)
            if int(ort_value.data_ptr()) != int(array.ctypes.data):
                ort_value = None
                array = None
                raw = None
                os_alloc.release()
                return None
        except Exception:
            ort_value = None
            array = None
            raw = None
            try:
                os_alloc.release()
            except Exception:
                pass
            return None
        return SharedBuffer(
            name=name,
            shape=shape,
            dtype=dtype,
            ort_value=ort_value,
            allocation_kind=os_alloc.kind,
            data_ptr=int(ort_value.data_ptr()),
            nbytes=nbytes,
            npu_device_type=self.npu_device_type,
            igpu_device_type=self.igpu_device_type,
            bind_device_type="cpu",
            _release=os_alloc.release,
            _keepalive=(array, raw, os_alloc.keepalive),
        )

    def _buffer_from_pinned_ortvalue(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        data: Optional[np.ndarray],
    ) -> SharedBuffer:
        source = data if data is not None else np.zeros(shape, dtype=dtype)
        pinned = IOBindingManager(device_type="hip_pinned", device_id=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ort_value = pinned._create_ort_value(source)
        if pinned.memory_placement == MemoryPlacement.PINNED_HOST:
            kind = SharedAllocationKind.PINNED_HOST
            bind_device = pinned.ortvalue_device
            if bind_device in _PRIVATE_EP_DEVICES or bind_device not in {
                "cpu",
                "cuda_pinned",
                "hip_pinned",
                "rocm_pinned",
            }:
                bind_device = "cpu"
        else:
            kind = SharedAllocationKind.CPU
            bind_device = "cpu"
        return SharedBuffer(
            name=name,
            shape=shape,
            dtype=dtype,
            ort_value=ort_value,
            allocation_kind=kind,
            data_ptr=int(ort_value.data_ptr()),
            nbytes=int(source.nbytes),
            npu_device_type=self.npu_device_type,
            igpu_device_type=self.igpu_device_type,
            bind_device_type=bind_device,
            _keepalive=(ort_value, source, pinned),
        )

    def get_shared_buffer(self, name: str) -> SharedBuffer:
        with self._lock:
            buffer = self._shared.get(name)
        if buffer is None:
            raise KeyError(f"Shared buffer '{name}' is not allocated")
        return buffer

    def bind_shared_pointer(
        self,
        io_binding: ort.IOBinding,
        tensor_name: str,
        buffer_name: str,
        *,
        as_output: bool,
        device_type: Optional[str] = None,
        device_id: int = 0,
    ) -> SharedBuffer:
        """
        Bind ``data_ptr`` of a shared buffer.

        ``device_type`` describes the memory (``cpu`` or a pinned-host alias).
        It is not the execution provider. ``vitisai`` and ``rocm`` name private
        device memory and are rejected here.
        """
        buffer = self.get_shared_buffer(buffer_name)
        if buffer.ort_value is None:
            raise RuntimeError(f"Shared buffer '{buffer_name}' is closed")
        bind_device = (device_type or buffer.bind_device_type).lower()
        if bind_device in _PRIVATE_EP_DEVICES:
            raise ValueError(
                f"Shared buffer '{buffer_name}' lives at {buffer.data_ptr:#x} "
                f"({buffer.allocation_kind.value}). Bind that address as "
                f"'{buffer.bind_device_type}', not as '{bind_device}'."
            )
        element_type = get_onnx_element_type(buffer.dtype)
        shape = list(buffer.shape)
        if as_output:
            io_binding.bind_output(
                tensor_name,
                device_type=bind_device,
                device_id=device_id,
                element_type=element_type,
                shape=shape,
                buffer_ptr=buffer.data_ptr,
            )
        else:
            io_binding.bind_input(
                tensor_name,
                bind_device,
                device_id,
                element_type,
                shape,
                buffer.data_ptr,
            )
        return buffer

    def bind_prefill_output(
        self,
        io_binding: ort.IOBinding,
        tensor_name: str,
        buffer_name: str,
    ) -> SharedBuffer:
        """Bind the shared address as an output of the NPU prefill session."""
        return self._bind_role(io_binding, tensor_name, buffer_name, as_output=True)

    def bind_decode_input(
        self,
        io_binding: ort.IOBinding,
        tensor_name: str,
        buffer_name: str,
    ) -> SharedBuffer:
        """Bind the shared address as an input of the iGPU decode session."""
        return self._bind_role(io_binding, tensor_name, buffer_name, as_output=False)

    def bind_phase_bridge(
        self,
        prefill_binding: ort.IOBinding,
        decode_binding: ort.IOBinding,
        tensor_name: str,
        buffer_name: str,
    ) -> SharedBuffer:
        """
        Bind one shared buffer as the prefill output and the decode input.

        Pinned OrtValues are bound as OrtValues so the pinned memory info is
        preserved. Unified and CPU buffers are bound by the same ``data_ptr``.
        """
        buffer = self.get_shared_buffer(buffer_name)
        if buffer.allocation_kind == SharedAllocationKind.PINNED_HOST:
            if buffer.ort_value is None:
                raise RuntimeError(f"Shared buffer '{buffer_name}' is closed")
            prefill_binding.bind_ortvalue_output(tensor_name, buffer.ort_value)
            decode_binding.bind_ortvalue_input(tensor_name, buffer.ort_value)
            return buffer
        self.bind_shared_pointer(prefill_binding, tensor_name, buffer_name, as_output=True)
        self.bind_shared_pointer(decode_binding, tensor_name, buffer_name, as_output=False)
        return buffer

    def _bind_role(
        self,
        io_binding: ort.IOBinding,
        tensor_name: str,
        buffer_name: str,
        *,
        as_output: bool,
    ) -> SharedBuffer:
        buffer = self.get_shared_buffer(buffer_name)
        if buffer.allocation_kind == SharedAllocationKind.PINNED_HOST:
            if buffer.ort_value is None:
                raise RuntimeError(f"Shared buffer '{buffer_name}' is closed")
            if as_output:
                io_binding.bind_ortvalue_output(tensor_name, buffer.ort_value)
            else:
                io_binding.bind_ortvalue_input(tensor_name, buffer.ort_value)
            return buffer
        return self.bind_shared_pointer(
            io_binding,
            tensor_name,
            buffer_name,
            as_output=as_output,
        )

    def deallocate_shared_buffer(self, name: str) -> bool:
        with self._lock:
            buffer = self._shared.pop(name, None)
        if buffer is None:
            return False
        buffer.close()
        return True

    def deallocate_shared(self) -> int:
        with self._lock:
            buffers = list(self._shared.values())
            self._shared.clear()
        for buffer in buffers:
            buffer.close()
        return len(buffers)

    def shared_memory_usage(self) -> Dict[str, Any]:
        with self._lock:
            buffers = dict(self._shared)
        total = sum(buffer.nbytes for buffer in buffers.values())
        return {
            "buffer_count": len(buffers),
            "total_bytes": total,
            "total_mb": round(total / (1024 * 1024), 3),
            "npu_device_type": self.npu_device_type,
            "igpu_device_type": self.igpu_device_type,
            "buffers": {name: buffer.to_dict() for name, buffer in buffers.items()},
        }

    def __repr__(self) -> str:
        usage = self.shared_memory_usage()
        return (
            f"HeterogeneousIOBindingManager(npu={self.npu_device_type}:{self.npu_device_id}, "
            f"igpu={self.igpu_device_type}:{self.igpu_device_id}, "
            f"shared={usage['buffer_count']}, shared_mb={usage['total_mb']})"
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Execution provider name -> device strings get_supported_devices() reports.
# VitisAI and RyzenAI are explicit entries. NPU discovery does not depend on
# the ROCm / MIGraphX iGPU check.
_PROVIDER_DEVICE_NAMES = {
    "CUDAExecutionProvider": ("cuda", "cuda_pinned"),
    "DmlExecutionProvider": ("dml",),
    "CoreMLExecutionProvider": ("coreml",),
    "ROCMExecutionProvider": ("rocm", "hip", "hip_pinned", "rocm_pinned"),
    "MIGraphXExecutionProvider": ("rocm", "hip", "hip_pinned", "rocm_pinned"),
    "VitisAIExecutionProvider": ("vitisai",),
    "RyzenAIExecutionProvider": ("ryzenai",),
}


def get_supported_devices(providers: Optional[Sequence[str]] = None) -> List[str]:
    """
    Device types the installed ONNX Runtime execution providers can target.

    ``VitisAIExecutionProvider`` appends ``vitisai``.
    ``RyzenAIExecutionProvider`` appends ``ryzenai``.
    CPU is always included. Pass ``providers`` to resolve a known list
    without querying the local wheel.
    """
    if providers is None:
        providers = ort.get_available_providers()

    available = {provider.lower() for provider in providers}
    devices = ["cpu"]
    seen = {"cpu"}

    def _add(device_name: str) -> None:
        if device_name not in seen:
            seen.add(device_name)
            devices.append(device_name)

    for provider_name, device_names in _PROVIDER_DEVICE_NAMES.items():
        if provider_name.lower() in available:
            for device_name in device_names:
                _add(device_name)

    # A versioned or prefixed NPU provider still contains these tokens.
    provider_text = " ".join(available)
    if "vitisai" in provider_text:
        _add("vitisai")
    if "ryzenai" in provider_text:
        _add("ryzenai")

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
    "HeterogeneousIOBindingManager",
    "DeviceBuffer",
    "SharedBuffer",
    "MemoryMode",
    "MemoryPlacement",
    "SharedAllocationKind",
    "StaticKVPool",
    "StaticShapeError",
    "OrtMemoryTarget",
    "OsSharedAllocation",
    # Functions
    "get_supported_devices",
    "get_onnx_element_type",
    "create_ortvalue_from_numpy",
    "ort_memory_candidates",
    "allocate_os_shared_memory",
    # Constants
    "NUMPY_TO_ONNX_ELEMENT_TYPE",
    "ONNX_ELEMENT_TYPE_NAMES",
    "VALID_ORTVALUE_DEVICES",
    "AMD_PCI_VENDOR_ID",
    "DEFAULT_NPU_MAX_SEQ_LEN",
]
