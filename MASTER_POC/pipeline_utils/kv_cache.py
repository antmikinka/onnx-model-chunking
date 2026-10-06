#!/usr/bin/env python3
"""
kv_cache.py - KV Cache Management Module

This module provides a KVCacheManager class for managing key-value cache state
during incremental transformer decoding.

CPU storage uses NumPy arrays. When ``max_seq_len`` is set, CPU and every
device backend preallocate one buffer of shape
``[batch, num_kv_heads, max_seq_len, head_dim]`` and write new tokens at
``current_seq_len`` instead of allocating a larger array per token.
NPU, iGPU, CUDA, pinned host, and hybrid NPU/iGPU storage keep that buffer
in one ort.OrtValue wrapped by a DeviceBuffer. The attention mask and
``sequence_length_scalar`` tell the graph how much of the buffer is valid.

Usage:
    from pipeline_utils.kv_cache import KVCacheManager, KVCacheBackend

    cache = KVCacheManager(
        num_layers=24,
        num_kv_heads=8,
        head_dim=128,
        dtype=np.float16,
        backend=KVCacheBackend.HYBRID_NPU_IGPU,
        max_seq_len=4096,
    )
    past_key, past_value = cache.get_past_tensors(0)
    past_key.ort_value  # bind this; do not copy to NumPy first
"""

from enum import Enum
from typing import Dict, Tuple, Optional, Any, List
import numpy as np

from .io_binding import (
    DEFAULT_NPU_MAX_SEQ_LEN,
    DeviceBuffer,
    HeterogeneousIOBindingManager,
    IOBindingManager,
    MemoryMode,
    MemoryPlacement,
    SharedBuffer,
)


class KVCacheBackend(Enum):
    """Where key and value tensors live between prefill and decode."""

    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    VITISAI = "vitisai"
    RYZENAI = "ryzenai"
    PINNED_HOST = "pinned_host"
    HYBRID_NPU_IGPU = "hybrid_npu_igpu"


_BACKEND_ALIASES = {
    "pinned": KVCacheBackend.PINNED_HOST,
    "hip_pinned": KVCacheBackend.PINNED_HOST,
    "cuda_pinned": KVCacheBackend.PINNED_HOST,
    "hybrid": KVCacheBackend.HYBRID_NPU_IGPU,
    "npu_igpu": KVCacheBackend.HYBRID_NPU_IGPU,
}

_BACKEND_DEVICE = {
    KVCacheBackend.CPU: "cpu",
    KVCacheBackend.CUDA: "cuda",
    KVCacheBackend.ROCM: "rocm",
    KVCacheBackend.VITISAI: "vitisai",
    KVCacheBackend.RYZENAI: "ryzenai",
    KVCacheBackend.PINNED_HOST: "hip_pinned",
    KVCacheBackend.HYBRID_NPU_IGPU: "vitisai",
}

_SEQ_AXIS = 2


def coerce_kv_backend(backend: Any) -> KVCacheBackend:
    """Accept the enum or its string value."""
    if isinstance(backend, KVCacheBackend):
        return backend
    text = str(backend).strip().lower().replace("-", "_")
    if text in _BACKEND_ALIASES:
        return _BACKEND_ALIASES[text]
    try:
        return KVCacheBackend(text)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in KVCacheBackend)
        raise ValueError(
            f"Unknown KV cache backend '{backend}'. Expected one of: {allowed}"
        ) from exc


def _tensor_shape(value: Any) -> Tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(
            f"KV tensor must be a NumPy array, OrtValue, or DeviceBuffer, "
            f"got {type(value).__name__}"
        )
    return tuple(int(dim) for dim in shape)


def _data_pointer(value: Any) -> Optional[int]:
    """Address of the storage, when the object exposes one."""
    if isinstance(value, np.ndarray):
        return int(value.ctypes.data)
    ort_value = getattr(value, "ort_value", None)
    if ort_value is not None and hasattr(ort_value, "data_ptr"):
        return int(ort_value.data_ptr())
    data_ptr = getattr(value, "data_ptr", None)
    if callable(data_ptr):
        return int(data_ptr())
    if isinstance(data_ptr, int):
        return data_ptr
    return None


def _as_numpy(value: Any) -> np.ndarray:
    """Host view or copy. Device backends use this only when storage differs."""
    if isinstance(value, np.ndarray):
        return value
    to_numpy = getattr(value, "to_numpy", None)
    if callable(to_numpy):
        return to_numpy()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        return numpy()
    raise TypeError(
        f"Cannot read KV data from {type(value).__name__}. "
        "Pass a NumPy array, OrtValue, or DeviceBuffer."
    )


def _placement(
    data: np.ndarray,
    position: int,
    seq_axis: int = _SEQ_AXIS,
) -> Tuple[Optional[np.ndarray], int, int]:
    """
    Decide which slice to write at ``position``.

    A present tensor longer than the tokens already stored is the full
    cache, so only the unseen tail is written and the length becomes that
    tensor's sequence axis. A shorter tensor is new tokens and is written
    at ``position``. An equal length adds nothing.
    """
    seq = int(data.shape[seq_axis])
    if seq > position:
        tail_slice = [slice(None)] * data.ndim
        tail_slice[seq_axis] = slice(position, seq)
        return data[tuple(tail_slice)], position, seq
    if seq == position:
        return None, position, position
    return data, position, position + seq


def _write_window(
    storage: Any,
    data: np.ndarray,
    start: int,
    seq_axis: int = _SEQ_AXIS,
) -> None:
    """
    Copy ``data`` into an existing buffer at sequence index ``start``.

    The allocated shape stays ``[batch, heads, max_seq_len, head_dim]``.
    A NumPy cache is updated in place. An OrtValue host view is written
    in place, and a host copy is pushed back with ``update_inplace``.
    """
    ort_value = getattr(storage, "ort_value", None)
    if isinstance(storage, np.ndarray):
        host = storage
        resident = None
    elif ort_value is not None:
        host = ort_value.numpy()
        resident = ort_value
    else:
        host = storage.numpy()
        resident = storage
    incoming = np.ascontiguousarray(data, dtype=host.dtype)
    if incoming.ndim != host.ndim:
        raise ValueError(
            f"KV rank mismatch: allocated {host.shape}, got {incoming.shape}"
        )
    if start < 0:
        raise ValueError(f"KV write index must be >= 0, got {start}")
    for index, (allocated, requested) in enumerate(zip(host.shape, incoming.shape)):
        if index == seq_axis:
            end = start + requested
            if requested <= 0 or end > allocated:
                raise ValueError(
                    f"Sequence window [{start}, {end}) does not fit "
                    f"max_seq_len capacity {allocated}"
                )
        elif allocated != requested:
            raise ValueError(
                f"KV axis {index} is fixed at {allocated}, got {requested}"
            )
    slices: List[slice] = [slice(None)] * host.ndim
    slices[seq_axis] = slice(start, start + int(incoming.shape[seq_axis]))
    host[tuple(slices)] = incoming
    if resident is not None and int(host.ctypes.data) != int(resident.data_ptr()):
        resident.update_inplace(np.ascontiguousarray(host))


class KVCacheManager:
    """
    Manages KV (Key-Value) cache state for incremental transformer decoding.

    In autoregressive generation with KV caching:
    - First iteration (prefill): Process full prompt, populate cache
    - Subsequent iterations (decode): Process only new token, use cached keys/values

    ``backend`` selects the memory. A cache with ``max_seq_len`` allocates
    ``[batch, heads, max_seq_len, head_dim]`` once. Decode writes the new
    token at ``[:, :, current_seq_len, :]`` and advances that length.
    CPU without ``max_seq_len`` still replaces NumPy arrays as the sequence
    grows. Device backends store an OrtValue inside a DeviceBuffer. Hybrid
    NPU/iGPU uses one shared allocation so prefill and decode bind the same
    pointer.

    Attributes:
        num_layers: Number of transformer layers
        num_kv_heads: Number of key-value heads per layer
        head_dim: Dimension of each head
        dtype: Data type for cache tensors
        batch_size: Batch size (default 1)
        backend: KVCacheBackend value
        max_seq_len: Preallocated sequence capacity, when the cache is static
        current_seq_len: Tokens written into the static buffer
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: Any = np.float16,
        batch_size: int = 1,
        backend: Any = KVCacheBackend.CPU,
        max_seq_len: Optional[int] = None,
        device_id: int = 0,
    ):
        """
        Initialize the KV cache manager.

        Args:
            num_layers: Number of transformer layers
            num_kv_heads: Number of key-value attention heads
            head_dim: Dimension of each attention head
            dtype: Numpy dtype for cache tensors (default: float16)
            batch_size: Batch size for inference (default: 1)
            backend: cpu, cuda, rocm, vitisai, ryzenai, pinned_host,
                or hybrid_npu_igpu
            max_seq_len: Preallocated sequence capacity. CPU with this set
                keeps one NumPy buffer and writes tokens into it. CPU
                without it grows by replacing arrays. Device backends
                default to the NPU window of 4096.
            device_id: Device index for the OrtValue allocation
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.batch_size = batch_size
        self.backend = coerce_kv_backend(backend)
        self.device_id = int(device_id)
        if max_seq_len is None:
            self.max_seq_len = (
                None
                if self.backend is KVCacheBackend.CPU
                else DEFAULT_NPU_MAX_SEQ_LEN
            )
        else:
            limit = int(max_seq_len)
            if limit <= 0:
                raise ValueError(f"max_seq_len must be positive, got {limit}")
            self.max_seq_len = limit
        self._static = self.max_seq_len is not None

        self._cache: Dict[int, Dict[str, Any]] = {}
        self._seq_len = 0
        self._layer_len: List[int] = [0] * self.num_layers
        self._binder: Optional[IOBindingManager] = None
        self._bridge: Optional[HeterogeneousIOBindingManager] = None
        self._shared: List[SharedBuffer] = []
        self.reset()

    @property
    def device_type(self) -> str:
        """OrtValue device name for this backend."""
        return _BACKEND_DEVICE[self.backend]

    @property
    def uses_device_memory(self) -> bool:
        """True when tensors are OrtValues rather than host NumPy arrays."""
        return self.backend is not KVCacheBackend.CPU

    @property
    def static(self) -> bool:
        """True when the sequence axis is preallocated at ``max_seq_len``."""
        return self._static

    def reset(self) -> None:
        """
        Reset the cache to empty state.

        A dynamic CPU cache creates empty tensors with shape
        ``[batch, heads, 0, head_dim]``. A static cache keeps the
        preallocated buffers and marks zero tokens valid, so the next
        prefill binds the same address.
        """
        self._seq_len = 0
        self._layer_len = [0] * self.num_layers
        if not self._static:
            self._cache.clear()
            for i in range(self.num_layers):
                self._cache[i] = {
                    "key": np.zeros(
                        (self.batch_size, self.num_kv_heads, 0, self.head_dim),
                        dtype=self.dtype,
                    ),
                    "value": np.zeros(
                        (self.batch_size, self.num_kv_heads, 0, self.head_dim),
                        dtype=self.dtype,
                    ),
                }
            return
        if self.backend is KVCacheBackend.CPU:
            shape = (
                self.batch_size,
                self.num_kv_heads,
                int(self.max_seq_len),
                self.head_dim,
            )
            if not self._cache:
                for i in range(self.num_layers):
                    self._cache[i] = {
                        "key": np.zeros(shape, dtype=self.dtype),
                        "value": np.zeros(shape, dtype=self.dtype),
                    }
            else:
                for i in range(self.num_layers):
                    self._cache[i]["key"].fill(0)
                    self._cache[i]["value"].fill(0)
            return
        if not self._cache:
            self._allocate_device_cache()
        if self._binder is not None:
            self._binder.set_valid_sequence_length(0, "kv")

    def _commit_length(self, layer_idx: int, new_len: int) -> None:
        """Publish one layer's valid prefix without reallocating."""
        if new_len < 0 or (
            self.max_seq_len is not None and new_len > int(self.max_seq_len)
        ):
            raise ValueError(
                f"Layer {layer_idx}: sequence length {new_len} exceeds "
                f"max_seq_len {self.max_seq_len}"
            )
        self._layer_len[layer_idx] = int(new_len)
        self._seq_len = int(new_len)
        if self._binder is not None:
            self._binder.set_valid_sequence_length(self._seq_len, "kv")

    def _allocate_device_cache(self) -> None:
        """Allocate one resident OrtValue per key and value."""
        if self.backend is KVCacheBackend.HYBRID_NPU_IGPU:
            self._allocate_hybrid_cache()
            return
        mode = (
            MemoryMode.NPU_STATIC
            if self.backend in {
                KVCacheBackend.VITISAI,
                KVCacheBackend.RYZENAI,
                KVCacheBackend.PINNED_HOST,
            }
            else MemoryMode.IO_BOUND
        )
        self._binder = IOBindingManager(
            self.device_type,
            self.device_id,
            memory_mode=mode,
            max_seq_len=int(self.max_seq_len),
        )
        pool = self._binder.allocate_static_kv_cache(
            name="kv",
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            seq_axis=_SEQ_AXIS,
        )
        for layer in range(self.num_layers):
            key = self._binder.get_buffer(pool.key_names[layer])
            value = self._binder.get_buffer(pool.value_names[layer])
            if key is None or value is None:
                raise RuntimeError(f"KV layer {layer} was not allocated")
            self._cache[layer] = {"key": key, "value": value}

    def _allocate_hybrid_cache(self) -> None:
        """One shared OrtValue per tensor for NPU prefill and iGPU decode."""
        self._bridge = HeterogeneousIOBindingManager(
            npu_device_type="vitisai",
            igpu_device_type="rocm",
            npu_device_id=self.device_id,
            igpu_device_id=self.device_id,
            max_seq_len=int(self.max_seq_len),
        )
        shape = (
            self.batch_size,
            self.num_kv_heads,
            int(self.max_seq_len),
            self.head_dim,
        )
        for layer in range(self.num_layers):
            slots = {}
            for kind in ("key", "value"):
                shared = self._bridge.allocate_shared_buffer(
                    f"kv.layer{layer}.{kind}",
                    shape,
                    self.dtype,
                )
                self._shared.append(shared)
                placement = (
                    MemoryPlacement.PINNED_HOST
                    if shared.crosses_devices
                    else MemoryPlacement.CPU
                )
                slots[kind] = DeviceBuffer(
                    name=shared.name,
                    shape=shape,
                    dtype=np.dtype(self.dtype),
                    device_type=self.device_type,
                    device_id=self.device_id,
                    ort_value=shared.ort_value,
                    is_dynamic=False,
                    ortvalue_device=shared.bind_device_type,
                    memory_placement=placement,
                    static_seq_axis=_SEQ_AXIS,
                    static_max_seq=int(self.max_seq_len),
                )
            self._cache[layer] = slots

    def get_sequence_length(self) -> int:
        """
        Get the current cached sequence length.

        A static cache reports the valid prefix. Its buffer stays at
        ``max_seq_len``. A dynamic CPU cache reports the NumPy sequence axis.

        Returns:
            Number of tokens currently in the cache (0 if empty)
        """
        if self._static:
            if self._binder is not None:
                return int(self._binder.get_static_kv_pool("kv").valid_seq_len)
            return int(self._seq_len)
        if 0 in self._cache and _tensor_shape(self._cache[0]["key"])[2] > 0:
            return int(_tensor_shape(self._cache[0]["key"])[2])
        return 0

    @property
    def current_seq_len(self) -> int:
        """Tokens written into the cache. The static buffer is not resized."""
        return self.get_sequence_length()

    def sequence_length_scalar(self) -> np.ndarray:
        """
        0-d int64 length for a graph that takes a sequence-length scalar.

        The value is ``current_seq_len``. Pair it with ``attention_mask``
        when the session needs both.
        """
        return np.array(self.current_seq_len, dtype=np.int64)

    def get_past_tensors(self, layer_idx: int) -> Tuple[Any, Any]:
        """
        Get the past key and value tensors for a specific layer.

        CPU returns NumPy arrays. Other backends return DeviceBuffer
        objects whose ``ort_value`` is the resident cache.

        Args:
            layer_idx: Index of the transformer layer (0-indexed)

        Returns:
            Tuple of (past_key, past_value)

        Raises:
            KeyError: If layer_idx is out of range
        """
        if layer_idx not in self._cache:
            raise KeyError(f"Layer {layer_idx} not in cache (have {self.num_layers} layers)")
        return self._cache[layer_idx]["key"], self._cache[layer_idx]["value"]

    def get_ort_values(self, layer_idx: int) -> Tuple[Any, Any]:
        """
        Return the OrtValues for one layer.

        CPU wraps the NumPy arrays in CPU OrtValues. Device backends
        return the resident values, which are the ones to bind.
        """
        key, value = self.get_past_tensors(layer_idx)
        return _ort_value_of(key), _ort_value_of(value)

    def attention_mask(self) -> np.ndarray:
        """
        Mask for the valid prefix.

        Static caches return shape ``(batch, max_seq_len)``. Ones cover
        ``current_seq_len`` and zeros cover the unused capacity, so the
        NPU graph keeps a fixed KV shape. A dynamic CPU cache returns a
        mask for the tokens stored so far. ``sequence_length_scalar`` is
        the same length as a 0-d int64.
        """
        valid = self.get_sequence_length()
        if self._binder is not None:
            return self._binder.attention_mask(valid, "kv")
        width = int(self.max_seq_len if self.max_seq_len is not None else valid)
        mask = np.zeros((self.batch_size, width), dtype=np.int64)
        if valid:
            mask[:, :valid] = 1
        return mask

    def update_from_present(
        self,
        layer_idx: int,
        present_key: Any,
        present_value: Any,
        validate_shape: bool = True,
        valid_seq_len: Optional[int] = None,
    ) -> None:
        """
        Update cache for a layer with new present key/value tensors.

        A dynamic CPU cache replaces the stored arrays. A static cache
        writes into the preallocated buffer at that layer's
        ``current_seq_len`` and leaves the buffer object in place. A
        present tensor longer than the tokens already stored contributes
        only its unseen tail. A shorter tensor is appended. If the caller
        passes the resident buffer with ``valid_seq_len``, only the valid
        length changes.

        Args:
            layer_idx: Index of the transformer layer
            present_key: The present.{layer_idx}.key output from the layer
            present_value: The present.{layer_idx}.value output from the layer
            validate_shape: If True, validate tensor shapes (default True)
            valid_seq_len: Prefix length when the caller passes the resident
                buffer itself. The resident buffer's sequence axis is the
                capacity, not the logical length.
        """
        if layer_idx not in self._cache:
            raise KeyError(f"Layer {layer_idx} not in cache (have {self.num_layers} layers)")

        key_shape = _tensor_shape(present_key)
        value_shape = _tensor_shape(present_value)
        if validate_shape:
            if key_shape != value_shape:
                raise ValueError(
                    f"Layer {layer_idx}: present_key shape {key_shape} != "
                    f"present_value shape {value_shape}"
                )
            if len(key_shape) != 4:
                raise ValueError(
                    f"Layer {layer_idx}: Expected 4D tensor [batch, heads, seq, head_dim], "
                    f"got shape {key_shape}"
                )
            if key_shape[1] != self.num_kv_heads:
                raise ValueError(
                    f"Layer {layer_idx}: Expected {self.num_kv_heads} KV heads, "
                    f"got {key_shape[1]}"
                )
            if key_shape[3] != self.head_dim:
                raise ValueError(
                    f"Layer {layer_idx}: Expected head_dim {self.head_dim}, "
                    f"got {key_shape[3]}"
                )

        if not self._static:
            self._cache[layer_idx]["key"] = present_key
            self._cache[layer_idx]["value"] = present_value
            return

        stored_key = self._cache[layer_idx]["key"]
        stored_value = self._cache[layer_idx]["value"]
        key_is_resident = _same_storage(stored_key, present_key)
        value_is_resident = _same_storage(stored_value, present_value)
        if key_is_resident and value_is_resident:
            if valid_seq_len is not None:
                self._commit_length(layer_idx, int(valid_seq_len))
            return

        position = self._layer_len[layer_idx]
        key_np = None if key_is_resident else _as_numpy(present_key)
        value_np = None if value_is_resident else _as_numpy(present_value)
        reference = key_np if key_np is not None else value_np
        _, _, new_len = _placement(reference, position)
        if new_len < 0 or (
            self.max_seq_len is not None and new_len > int(self.max_seq_len)
        ):
            raise ValueError(
                f"Layer {layer_idx}: sequence length {new_len} exceeds "
                f"max_seq_len {self.max_seq_len}"
            )
        if key_np is not None:
            window, start, _ = _placement(key_np, position)
            if window is not None:
                _write_window(stored_key, window, start)
        if value_np is not None:
            window, start, _ = _placement(value_np, position)
            if window is not None:
                _write_window(stored_value, window, start)
        self._commit_length(layer_idx, new_len)

    def inject_into_registry(
        self,
        registry: Dict[str, Any],
        key_format: str = "past_key_values.{layer}.key",
        value_format: str = "past_key_values.{layer}.value",
    ) -> None:
        """
        Inject all cache tensors into a tensor registry.

        CPU entries are NumPy arrays. Device entries are DeviceBuffers.
        Bind ``entry.ort_value`` rather than copying the tensor to host.

        Args:
            registry: Dictionary to inject cache tensors into
            key_format: Format string for key tensor names (use {layer} placeholder)
            value_format: Format string for value tensor names (use {layer} placeholder)
        """
        for i in range(self.num_layers):
            key_name = key_format.format(layer=i)
            value_name = value_format.format(layer=i)
            registry[key_name] = self._cache[i]["key"]
            registry[value_name] = self._cache[i]["value"]

    def extract_from_registry(
        self,
        registry: Dict[str, Any],
        present_key_format: str = "present.{layer}.key",
        present_value_format: str = "present.{layer}.value",
    ) -> int:
        """
        Extract present tensors from registry and update cache.

        This is a convenience method to update all layers' cache from
        a tensor registry after running the full model.

        Args:
            registry: Dictionary containing present.* tensors
            present_key_format: Format string for present key names
            present_value_format: Format string for present value names

        Returns:
            Number of layers successfully updated
        """
        updated = 0
        for i in range(self.num_layers):
            key_name = present_key_format.format(layer=i)
            value_name = present_value_format.format(layer=i)

            if key_name in registry and value_name in registry:
                self.update_from_present(i, registry[key_name], registry[value_name])
                updated += 1

        return updated

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage statistics for the cache.

        Device backends report the allocated OrtValue size, which stays at
        ``max_seq_len`` even when the valid prefix is shorter.

        Returns:
            Dictionary with memory stats (total_bytes, per_layer_bytes, etc.)
        """
        total_bytes = 0
        per_layer = []

        for i in range(self.num_layers):
            key_bytes = _nbytes(self._cache[i]["key"])
            value_bytes = _nbytes(self._cache[i]["value"])
            layer_bytes = key_bytes + value_bytes
            per_layer.append(layer_bytes)
            total_bytes += layer_bytes

        seq_len = self.get_sequence_length()

        return {
            "total_bytes": total_bytes,
            "total_mb": total_bytes / (1024 * 1024),
            "per_layer_bytes": per_layer,
            "sequence_length": seq_len,
            "num_layers": self.num_layers,
            "dtype": str(self.dtype),
            "backend": self.backend.value,
            "max_seq_len": self.max_seq_len,
            "current_seq_len": seq_len,
            "static": self._static,
            "storage": "device_buffer" if self.uses_device_memory else "numpy",
        }

    def __repr__(self) -> str:
        seq_len = self.get_sequence_length()
        mem = self.get_memory_usage()
        return (
            f"KVCacheManager(layers={self.num_layers}, heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, seq_len={seq_len}, "
            f"backend={self.backend.value}, memory={mem['total_mb']:.2f}MB)"
        )


def _nbytes(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    size_bytes = getattr(value, "size_bytes", None)
    if size_bytes is not None:
        return int(size_bytes)
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    return 0


def _ort_value_of(value: Any) -> Any:
    import onnxruntime as ort

    if isinstance(value, np.ndarray):
        return ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(value), "cpu", 0)
    ort_value = getattr(value, "ort_value", None)
    if ort_value is not None:
        return ort_value
    if hasattr(value, "data_ptr") and hasattr(value, "numpy"):
        return value
    raise TypeError(f"No OrtValue for KV tensor of type {type(value).__name__}")


def _same_storage(stored: Any, incoming: Any) -> bool:
    left = _data_pointer(stored)
    right = _data_pointer(incoming)
    return left is not None and left == right


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

__all__ = ["KVCacheManager", "KVCacheBackend", "coerce_kv_backend"]
