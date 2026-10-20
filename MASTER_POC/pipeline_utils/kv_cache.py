#!/usr/bin/env python3
"""
kv_cache.py - KV Cache Management Module

This module provides a KVCacheManager class for managing key-value cache state
during incremental transformer decoding.

CPU storage uses NumPy arrays. When ``max_seq_len`` is set, CPU, CUDA,
ROCm, and pinned host preallocate one buffer of shape
    ``[batch, num_kv_heads, max_seq_len, head_dim]`` and write new tokens at
``current_seq_len`` instead of allocating a larger array per token.
VitisAI, RyzenAI, and hybrid NPU/iGPU do not slot-write that buffer.
They concatenate the valid prefix with the new tokens into a second
buffer of the same shape, then swap the two pointers. The sequence axis
stays ``max_seq_len``, so the XCLBIN is not rebuilt. The attention mask
and ``sequence_length_scalar`` tell the graph how much of the active
buffer is valid.

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
    past_key, past_value = cache.get_device_tensors(0)
    cache.bind_cache_to_session(session, io_binding, layer_idx=0)

Hybrid stacks use one cache type per layer. Dense attention stays on
``DenseKVCache``. Gated DeltaNet and linear attention use
``RecurrentStateCache``. Sparse attention uses ``SparseKVCache``.
``HybridKVCacheManager`` mixes those with MoE router history and can hold
a draft cache beside a verify cache for speculative decoding.
"""

from enum import Enum
import re
import threading
from typing import Dict, Iterable, Tuple, Optional, Any, List, Sequence
import numpy as np
import onnxruntime as ort

from .io_binding import (
    DEFAULT_NPU_MAX_SEQ_LEN,
    DeviceBuffer,
    HeterogeneousIOBindingManager,
    IOBindingManager,
    MemoryMode,
    MemoryPlacement,
    SharedBuffer,
)


class KVCachePolicy(Enum):
    """How many past tokens stay resident, and how they are packed."""

    DENSE_CONTIGUOUS = "dense_contiguous"
    SLIDING_WINDOW = "sliding_window"
    PAGED_BLOCKS = "paged_blocks"
    HYBRID_SPARSE = "hybrid_sparse"


class KVCacheBackend(Enum):
    """Where key and value tensors live between prefill and decode."""

    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    VITISAI = "vitisai"
    RYZENAI = "ryzenai"
    PINNED_HOST = "pinned_host"
    HYBRID_NPU_IGPU = "hybrid_npu_igpu"


class KVUpdateMode(Enum):
    """How a new token becomes part of the resident cache.

    ``SLOT_WRITE`` stores the token at ``current_seq_len`` inside the live
    buffer. CPU, CUDA, ROCm, and pinned host can do that.

    ``CONCAT_SWAP`` is the spatial-NPU translation of the append in
    Stephen Panaro's ``gpt2.py``: ``torch.cat([old_k, new_k], dim=1)``,
    then install that tensor as the next cache. ``index_copy``,
    ``scatter``, and a slice assignment at a runtime index do not run on
    the ANE, and a VitisAI XCLBIN has the same problem. The sequence axis
    stays ``max_seq_len``. The concatenated prefix is written into the
    inactive buffer and the two pointers swap. The live buffer is not
    slot-updated, and past and present are never the same OrtValue.
    """

    SLOT_WRITE = "slot_write"
    CONCAT_SWAP = "concat_swap"


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

_SPATIAL_NPU_BACKENDS = frozenset({
    KVCacheBackend.VITISAI,
    KVCacheBackend.RYZENAI,
    KVCacheBackend.HYBRID_NPU_IGPU,
})

_UPDATE_MODE_ALIASES = {
    "slot": KVUpdateMode.SLOT_WRITE,
    "inplace": KVUpdateMode.SLOT_WRITE,
    "in_place": KVUpdateMode.SLOT_WRITE,
    "concat": KVUpdateMode.CONCAT_SWAP,
    "cat": KVUpdateMode.CONCAT_SWAP,
    "pointer_swap": KVUpdateMode.CONCAT_SWAP,
    "swap": KVUpdateMode.CONCAT_SWAP,
}

_SEQ_AXIS = 2

_POLICY_ALIASES = {
    "dense": KVCachePolicy.DENSE_CONTIGUOUS,
    "contiguous": KVCachePolicy.DENSE_CONTIGUOUS,
    "sliding": KVCachePolicy.SLIDING_WINDOW,
    "window": KVCachePolicy.SLIDING_WINDOW,
    "paged": KVCachePolicy.PAGED_BLOCKS,
    "paged_attention": KVCachePolicy.PAGED_BLOCKS,
    "blocks": KVCachePolicy.PAGED_BLOCKS,
    "hybrid": KVCachePolicy.HYBRID_SPARSE,
    "sparse": KVCachePolicy.HYBRID_SPARSE,
}


def coerce_kv_policy(policy: Any) -> KVCachePolicy:
    """Accept the enum or its string value."""
    if isinstance(policy, KVCachePolicy):
        return policy
    text = str(policy).strip().lower().replace("-", "_")
    if text in _POLICY_ALIASES:
        return _POLICY_ALIASES[text]
    try:
        return KVCachePolicy(text)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in KVCachePolicy)
        raise ValueError(
            f"Unknown KV cache policy '{policy}'. Expected one of: {allowed}"
        ) from exc


def default_kv_update_mode(backend: KVCacheBackend) -> KVUpdateMode:
    """Spatial NPUs concatenate and swap. Indexed devices slot-write."""
    if backend in _SPATIAL_NPU_BACKENDS:
        return KVUpdateMode.CONCAT_SWAP
    return KVUpdateMode.SLOT_WRITE


def coerce_kv_update_mode(mode: Any) -> KVUpdateMode:
    """Accept the enum or its string value."""
    if isinstance(mode, KVUpdateMode):
        return mode
    text = str(mode).strip().lower().replace("-", "_")
    if text in _UPDATE_MODE_ALIASES:
        return _UPDATE_MODE_ALIASES[text]
    try:
        return KVUpdateMode(text)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in KVUpdateMode)
        raise ValueError(
            f"Unknown KV update mode '{mode}'. Expected one of: {allowed}"
        ) from exc


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


class _PageState:
    """Host-side block table over a fixed KV allocation."""

    def __init__(self, physical_blocks: int):
        self.free = list(range(int(physical_blocks)))
        self.table: List[int] = []
        self.fill: List[int] = []
        self.write_pos = 0
        self.resident = 0

    def ensure(self, logical: int, max_seq_len: Optional[int]) -> int:
        while len(self.table) <= logical:
            self.table.append(-1)
            self.fill.append(0)
        if self.table[logical] < 0:
            if not self.free:
                raise ValueError(
                    f"Paged KV cache is full at max_seq_len {max_seq_len}"
                )
            self.table[logical] = int(self.free.pop(0))
        return int(self.table[logical])


def _resolve_layer_policies(
    policy: KVCachePolicy,
    layer_policies: Optional[Sequence[Any]],
    num_layers: int,
) -> List[KVCachePolicy]:
    if layer_policies is None:
        return [policy] * num_layers
    resolved = [coerce_kv_policy(item) for item in layer_policies]
    if len(resolved) != num_layers:
        raise ValueError(
            f"layer_policies has {len(resolved)} entries for {num_layers} layers"
        )
    return resolved


def _apply_policy_capacity(cache: "KVCacheManager") -> None:
    """Check policy arguments and give a CPU sliding window a fixed buffer."""
    if cache.block_size <= 0:
        raise ValueError(f"block_size must be positive, got {cache.block_size}")
    if cache.sink_size < 0:
        raise ValueError(f"sink_size must be >= 0, got {cache.sink_size}")
    if cache.retrieval_blocks < 0:
        raise ValueError(
            f"retrieval_blocks must be >= 0, got {cache.retrieval_blocks}"
        )
    kinds = set(cache._layer_policies)
    needs_window = (
        KVCachePolicy.SLIDING_WINDOW in kinds
        or KVCachePolicy.HYBRID_SPARSE in kinds
    )
    if needs_window and (cache.window_size is None or cache.window_size <= 0):
        raise ValueError(
            "window_size must be positive for sliding_window and hybrid_sparse"
        )
    needs_pool = (
        KVCachePolicy.PAGED_BLOCKS in kinds
        or KVCachePolicy.HYBRID_SPARSE in kinds
    )
    if needs_pool and cache.max_seq_len is None:
        raise ValueError("paged_blocks and hybrid_sparse require max_seq_len")
    if KVCachePolicy.SLIDING_WINDOW in kinds and cache.max_seq_len is None:
        if kinds != {KVCachePolicy.SLIDING_WINDOW}:
            raise ValueError("mixed layer_policies require max_seq_len")
        cache.max_seq_len = int(cache.window_size)
    if (
        cache.window_size is not None
        and cache.max_seq_len is not None
        and int(cache.window_size) > int(cache.max_seq_len)
    ):
        raise ValueError(
            f"window_size {cache.window_size} exceeds max_seq_len {cache.max_seq_len}"
        )
    if KVCachePolicy.HYBRID_SPARSE in kinds and cache.max_seq_len is not None:
        budget = (
            int(cache.sink_size)
            + int(cache.window_size)
            + int(cache.retrieval_blocks) * int(cache.block_size)
        )
        if budget > int(cache.max_seq_len):
            raise ValueError(
                f"hybrid_sparse keeps {budget} tokens, which exceeds "
                f"max_seq_len {cache.max_seq_len}"
            )
    if KVCachePolicy.PAGED_BLOCKS in kinds and cache.max_seq_len is not None:
        if int(cache.max_seq_len) // int(cache.block_size) < 1:
            raise ValueError(
                f"block_size {cache.block_size} exceeds max_seq_len {cache.max_seq_len}"
            )


def _read_prefix(storage: Any, length: int) -> np.ndarray:
    data = _as_numpy(storage)
    return np.array(data[:, :, :length, :], copy=True)


def _cat_seq(parts: Sequence[np.ndarray]) -> np.ndarray:
    present = [part for part in parts if part.shape[_SEQ_AXIS] > 0]
    if not present:
        return np.ascontiguousarray(parts[0])
    if len(present) == 1:
        return np.ascontiguousarray(present[0])
    return np.concatenate(present, axis=_SEQ_AXIS)


class KVCacheManager:
    """
    Manages KV (Key-Value) cache state for incremental transformer decoding.

    This is the dense-attention cache. ``entry()`` exposes each layer as a
    ``CacheEntry``. Recurrent, sparse, router, and speculative state live on
    ``RecurrentStateCache``, ``SparseKVCache``, and ``HybridKVCacheManager``.

    ``_lock`` is a ``threading.RLock``. ``reset``, ``update_from_present``,
    ``inject_into_registry``, and ``extract_from_registry`` hold it so
    prefill, decode, and registry copies do not interleave. The lock is
    reentrant because extract updates each layer through
    ``update_from_present``.

    In autoregressive generation with KV caching:
    - First iteration (prefill): Process full prompt, populate cache
    - Subsequent iterations (decode): Process only new token, use cached keys/values

    ``backend`` selects the memory. A cache with ``max_seq_len`` allocates
    ``[batch, heads, max_seq_len, head_dim]`` once. ``slot_write`` stores
    the new token at ``[:, :, current_seq_len, :]``. ``concat_swap``
    concatenates that token onto the valid prefix in the shadow buffer
    and exchanges the two pointers. CPU without ``max_seq_len`` still
    replaces NumPy arrays as the sequence grows. Device backends store an
    OrtValue inside a DeviceBuffer. Hybrid NPU/iGPU uses one shared
    allocation so prefill and decode bind the same pointer, plus a shadow
    when the update mode is ``concat_swap``.

    Attributes:
        num_layers: Number of transformer layers
        num_kv_heads: Number of key-value heads per layer
        head_dim: Dimension of each head
        dtype: Data type for cache tensors
        batch_size: Batch size (default 1)
        backend: KVCacheBackend value
        max_seq_len: Preallocated sequence capacity, when the cache is static
        current_seq_len: Write cursor. The next token is stored at this index.
            ``advance_sequence_length``, ``truncate_to_length``, and
            ``reserve`` move the cursor for a prefill/decode split.
        policy: KVCachePolicy. Dense keeps every token. Sliding window,
            paged blocks, and hybrid sparse keep a bounded working set
            inside the same allocation.
    """

    cache_type = "dense_kv"

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: Any = None,
        batch_size: int = 1,
        backend: Any = KVCacheBackend.CPU,
        max_seq_len: Optional[int] = None,
        device_id: int = 0,
        policy: Any = KVCachePolicy.DENSE_CONTIGUOUS,
        window_size: Optional[int] = None,
        block_size: int = 16,
        sink_size: int = 0,
        retrieval_blocks: int = 0,
        layer_policies: Optional[Sequence[Any]] = None,
        cache_dtype: Any = None,
        kv_quantization: Optional[str] = None,
        update_mode: Any = None,
    ):
        """
        Initialize the KV cache manager.

        Args:
            num_layers: Number of transformer layers
            num_kv_heads: Number of key-value attention heads
            head_dim: Dimension of each attention head
            dtype: Numpy dtype for cache tensors. Omitted values use
                ``KV_CACHE_DTYPE_POLICY['default']`` (float16).
                ``get_kv_cache_dtype`` selects NPU, ROCm, CPU, or a
                compressed scheme. Key and value can differ through
                ``get_kv_cache_dtypes``.
            batch_size: Batch size for inference (default: 1)
            backend: cpu, cuda, rocm, vitisai, ryzenai, pinned_host,
                or hybrid_npu_igpu
            max_seq_len: Preallocated sequence capacity. CPU with this set
                keeps one NumPy buffer and writes tokens into it. CPU
                without it grows by replacing arrays. Device backends
                default to the NPU window of 4096.
            device_id: Device index for the OrtValue allocation
            policy: dense_contiguous, sliding_window, paged_blocks,
                or hybrid_sparse. The default keeps the dense prefix.
            window_size: Tokens retained by sliding_window and the recent
                portion of hybrid_sparse
            block_size: Tokens per paged block and per retrieved block
            sink_size: Leading tokens hybrid_sparse never evicts
            retrieval_blocks: Extra blocks hybrid_sparse can pin
            layer_policies: Optional per-layer policies. Length must match
                ``num_layers``. Full-attention layers stay dense while
                sliding layers drop history.
            cache_dtype: Stored element type. Defaults from ``dtype`` or
                from ``kv_quantization``.
            kv_quantization: None, fp16, bf16, int8, fp8, int4, or fp4.
                None keeps the compute buffer as the stored form.
            update_mode: ``slot_write`` or ``concat_swap``. Omitted values
                use ``concat_swap`` for VitisAI, RyzenAI, and hybrid
                NPU/iGPU, and ``slot_write`` for every other backend.
        """
        dtype = _default_compute_dtype(dtype)
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.batch_size = batch_size
        self.backend = coerce_kv_backend(backend)
        self.update_mode = (
            default_kv_update_mode(self.backend)
            if update_mode is None
            else coerce_kv_update_mode(update_mode)
        )
        self.device_id = int(device_id)
        self.policy = coerce_kv_policy(policy)
        self.window_size = None if window_size is None else int(window_size)
        self.block_size = int(block_size)
        self.sink_size = int(sink_size)
        self.retrieval_blocks = int(retrieval_blocks)
        self._layer_policies = _resolve_layer_policies(
            self.policy, layer_policies, num_layers
        )
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
        _apply_policy_capacity(self)
        self._static = self.max_seq_len is not None

        self._cache: Dict[int, Dict[str, Any]] = {}
        self._seq_len = 0
        self._layer_len: List[int] = [0] * self.num_layers
        self._mask_layer = 0
        self._pinned: List[set] = [set() for _ in range(self.num_layers)]
        self._seen: List[int] = [0] * self.num_layers
        self._seen_published = 0
        self._sink_held: List[int] = [0] * self.num_layers
        self._retrieved_len: List[int] = [0] * self.num_layers
        self._physical_blocks = 0
        self._pages: List["_PageState"] = []
        self.kv_quantization = _coerce_kv_quantization(kv_quantization)
        self.cache_dtype = _resolve_cache_dtype(
            self.kv_quantization, cache_dtype, dtype
        )
        if _quantization_uses_scale(self.kv_quantization):
            self.key_scale = np.full((self.num_layers,), np.nan, dtype=np.float32)
            self.value_scale = np.full((self.num_layers,), np.nan, dtype=np.float32)
        else:
            self.key_scale = None
            self.value_scale = None
        self._compressed: Dict[int, Dict[str, np.ndarray]] = {}
        self._quant_valid: List[bool] = [False] * self.num_layers
        self._binder: Optional[IOBindingManager] = None
        self._bridge: Optional[HeterogeneousIOBindingManager] = None
        self._shared: List[SharedBuffer] = []
        self.phase: Optional[str] = None
        self.execution_phase: Optional[str] = None
        self.role = "target"
        # Reentrant so extract_from_registry can call update_from_present.
        # Acquire this before an IOBindingManager lock. The binder methods
        # used here take the IO lock and do not call back into the cache.
        self._lock = threading.RLock()
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
    def device_resident(self) -> bool:
        """True when the cache is bound to OrtValues from an IOBindingManager."""
        return self._binder is not None or bool(self._shared) or self.uses_device_memory

    @property
    def static(self) -> bool:
        """True when the sequence axis is preallocated at ``max_seq_len``."""
        return self._static

    def _concat_swap_active(self) -> bool:
        """True when publishes go to the shadow buffer instead of a slot write."""
        return self.update_mode is KVUpdateMode.CONCAT_SWAP and self._static

    def reset(self) -> None:
        """
        Reset the cache to empty state.

        A dynamic CPU cache creates empty tensors with shape
        ``[batch, heads, 0, head_dim]``. A static cache keeps the
        preallocated buffers and marks zero tokens valid, so the next
        prefill binds the same address.
        """
        with self._lock:
            self._seq_len = 0
            self._layer_len = [0] * self.num_layers
            self._mask_layer = 0
            self.phase = None
            self.execution_phase = None
            self._reset_policy_state()
            self._reset_quant_state()
            if self.device_resident:
                if not self._cache:
                    if self._binder is not None:
                        self._install_manager_cache(self._binder, int(self.max_seq_len))
                    else:
                        self._allocate_device_cache()
                if self._binder is not None:
                    self._binder.set_valid_sequence_length(0, "kv")
                return
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
                        if self._concat_swap_active():
                            self._cache[i]["key_shadow"] = np.zeros(shape, dtype=self.dtype)
                            self._cache[i]["value_shadow"] = np.zeros(shape, dtype=self.dtype)
                else:
                    for i in range(self.num_layers):
                        self._cache[i]["key"].fill(0)
                        self._cache[i]["value"].fill(0)
                        if self._concat_swap_active():
                            if "key_shadow" not in self._cache[i]:
                                self._cache[i]["key_shadow"] = np.zeros(shape, dtype=self.dtype)
                                self._cache[i]["value_shadow"] = np.zeros(shape, dtype=self.dtype)
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
        self._mask_layer = layer_idx
        if self._binder is None:
            return
        if self.policy_for(layer_idx) is KVCachePolicy.PAGED_BLOCKS:
            self._binder.get_static_kv_pool("kv").valid_seq_len = int(new_len)
            return
        self._binder.set_valid_sequence_length(self._seq_len, "kv")

    def _reset_policy_state(self) -> None:
        """Clear resident tokens. Pin requests stay, because they are config."""
        self._seen = [0] * self.num_layers
        self._seen_published = 0
        self._sink_held = [0] * self.num_layers
        self._retrieved_len = [0] * self.num_layers
        physical = 0
        if self.max_seq_len and self.block_size:
            physical = int(self.max_seq_len) // int(self.block_size)
        self._physical_blocks = physical
        self._pages = [_PageState(physical) for _ in range(self.num_layers)]

    def _reset_quant_state(self) -> None:
        """Drop compressed copies. The compute buffer stays allocated."""
        self._compressed = {}
        self._quant_valid = [False] * self.num_layers
        if self.key_scale is not None:
            self.key_scale.fill(np.nan)
            self.value_scale.fill(np.nan)

    def policy_for(self, layer_idx: int) -> KVCachePolicy:
        """Retention policy for one transformer layer."""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise KeyError(
                f"Layer {layer_idx} not in cache (have {self.num_layers} layers)"
            )
        return self._layer_policies[layer_idx]

    def layer_length(self, layer_idx: int) -> int:
        """Resident tokens in one layer."""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise KeyError(
                f"Layer {layer_idx} not in cache (have {self.num_layers} layers)"
            )
        if self.policy_for(layer_idx) is KVCachePolicy.PAGED_BLOCKS:
            return int(self._pages[layer_idx].resident)
        if self._static:
            return int(self._layer_len[layer_idx])
        if layer_idx not in self._cache:
            return 0
        return int(_tensor_shape(self._cache[layer_idx]["key"])[_SEQ_AXIS])

    def layer_seen_tokens(self, layer_idx: int) -> int:
        """Tokens accepted by one layer, including ones already evicted."""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise KeyError(
                f"Layer {layer_idx} not in cache (have {self.num_layers} layers)"
            )
        return int(self._seen[layer_idx])

    @property
    def seen_tokens(self) -> int:
        """Tokens accepted by the layer updated most recently."""
        return int(self._seen_published)

    def block_tables(self, layer_idx: int) -> np.ndarray:
        """
        Logical-to-physical block ids for one layer.

        Shape is ``(batch, logical_blocks)``. ``-1`` is an empty slot.
        Paged layers use this instead of assuming the sequence axis is
        one dense prefix.
        """
        if self.policy_for(layer_idx) is not KVCachePolicy.PAGED_BLOCKS:
            raise ValueError(
                f"Layer {layer_idx} uses {self.policy_for(layer_idx).value}, "
                "which has no block table"
            )
        logical = self._pages[layer_idx].table
        table = np.full((self.batch_size, len(logical)), -1, dtype=np.int32)
        if logical:
            table[:, :] = np.asarray(logical, dtype=np.int32)[None, :]
        return table

    def release_blocks(self, layer_idx: int, logical_block_ids: Iterable[int]) -> None:
        """
        Return paged blocks to the free list.

        The KV tensor is not reallocated. A later token can reuse the
        physical page.
        """
        if self.policy_for(layer_idx) is not KVCachePolicy.PAGED_BLOCKS:
            raise ValueError(
                f"release_blocks applies to paged_blocks, not "
                f"{self.policy_for(layer_idx).value}"
            )
        state = self._pages[layer_idx]
        for block_id in logical_block_ids:
            logical = int(block_id)
            if logical < 0 or logical >= len(state.table):
                raise ValueError(
                    f"Layer {layer_idx} has no logical block {logical}"
                )
            physical = state.table[logical]
            if physical < 0:
                raise ValueError(
                    f"Layer {layer_idx} logical block {logical} is already free"
                )
            state.resident -= int(state.fill[logical])
            state.fill[logical] = 0
            state.table[logical] = -1
            state.free.append(int(physical))
            state.free.sort()
        self._commit_length(layer_idx, state.resident)
        self._store_custom_mask(self._physical_mask(layer_idx))

    def pin_blocks(self, layer_idx: int, logical_block_ids: Iterable[int]) -> None:
        """
        Keep these logical blocks in a hybrid_sparse layer.

        The ids are applied on the next update that still contains those
        tokens, which is the prefill present for a fresh cache.
        """
        if self.policy_for(layer_idx) is not KVCachePolicy.HYBRID_SPARSE:
            raise ValueError(
                f"pin_blocks applies to hybrid_sparse, not "
                f"{self.policy_for(layer_idx).value}"
            )
        chosen = {int(block_id) for block_id in logical_block_ids}
        if any(block_id < 0 for block_id in chosen):
            raise ValueError("logical block ids must be >= 0")
        if len(chosen) > self.retrieval_blocks:
            raise ValueError(
                f"pin_blocks received {len(chosen)} blocks, retrieval_blocks "
                f"is {self.retrieval_blocks}"
            )
        self._pinned[layer_idx] = chosen

    def _layer_position(self, layer_idx: int) -> int:
        if layer_idx not in self._cache:
            return 0
        if self._static:
            return int(self._layer_len[layer_idx])
        return int(_tensor_shape(self._cache[layer_idx]["key"])[_SEQ_AXIS])

    def _same_resident_pool(self, io_manager: IOBindingManager, limit: int) -> bool:
        if self._binder is not io_manager or int(self.max_seq_len or -1) != int(limit):
            return False
        if not self._cache or 0 not in self._cache:
            return False
        resident = io_manager.get_buffer("kv.layer0.key")
        return resident is not None and resident is self._cache[0]["key"]

    def _snapshot_prefix(
        self,
        limit: int,
    ) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        rows: List[Tuple[int, np.ndarray, np.ndarray]] = []
        if not self._cache:
            return rows
        for layer in range(self.num_layers):
            if layer not in self._cache:
                continue
            position = self._layer_position(layer)
            if position <= 0:
                continue
            if position > limit:
                raise ValueError(
                    f"Layer {layer}: current_seq_len {position} exceeds "
                    f"max_seq_len {limit}"
                )
            key = self._cache[layer]["key"]
            value = self._cache[layer]["value"]
            rows.append(
                (
                    layer,
                    np.array(_as_numpy(key)[:, :, :position, :], copy=True),
                    np.array(_as_numpy(value)[:, :, :position, :], copy=True),
                )
            )
        return rows

    def _install_manager_cache(self, io_manager: IOBindingManager, limit: int) -> None:
        pool = io_manager.get_static_kv_pool("kv")
        installed: Dict[int, Dict[str, Any]] = {}
        for layer in range(self.num_layers):
            key = io_manager.get_buffer(pool.key_names[layer])
            value = io_manager.get_buffer(pool.value_names[layer])
            if (
                key is None
                or value is None
                or key.ort_value is None
                or value.ort_value is None
            ):
                raise RuntimeError(
                    f"KV layer {layer} has no OrtValue on the IOBindingManager"
                )
            installed[layer] = {"key": key, "value": value}
        self._cache = installed
        self._binder = io_manager
        self.max_seq_len = int(limit)
        self._static = True
        self._attach_shadows()

    def _mask_ort_value(self) -> ort.OrtValue:
        """Attention mask OrtValue owned by the IOBindingManager when one exists."""
        self.attention_mask()
        if self._binder is not None:
            buffer = self._binder.get_buffer("kv.attention_mask")
            if buffer is not None and buffer.ort_value is not None:
                return buffer.ort_value
        return self._refresh_host_ort("_mask_ort", self.attention_mask())

    def _scalar_ort_value(self) -> ort.OrtValue:
        """Resident 0-d int64 current_seq_len, updated in place."""
        scalar = np.ascontiguousarray(self.sequence_length_scalar())
        if self._binder is not None:
            buffer = self._binder.get_buffer("kv.sequence_length")
            if buffer is None or buffer.ort_value is None:
                buffer = self._binder.allocate_buffer(
                    "kv.sequence_length",
                    scalar.shape,
                    scalar.dtype,
                    data=scalar,
                    is_dynamic=False,
                )
            else:
                _refresh_ort_contents(buffer.ort_value, scalar)
            return buffer.ort_value
        return self._refresh_host_ort("_scalar_ort", scalar)

    def _refresh_host_ort(self, attr: str, data: np.ndarray) -> ort.OrtValue:
        incoming = np.ascontiguousarray(data)
        current = getattr(self, attr, None)
        current_shape = _ort_shape(current) if current is not None else None
        if current is not None and current_shape is not None and _same_scalar_storage(
            current_shape, tuple(incoming.shape)
        ):
            _refresh_ort_contents(current, incoming)
            return current
        current = ort.OrtValue.ortvalue_from_numpy(incoming, "cpu", 0)
        setattr(self, attr, current)
        return current

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
        self._attach_shadows()

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
                if self._concat_swap_active():
                    shadow = self._bridge.allocate_shared_buffer(
                        f"kv.layer{layer}.{kind}.shadow",
                        shape,
                        self.dtype,
                    )
                    self._shared.append(shadow)
                    slots[f"{kind}_shadow"] = DeviceBuffer(
                        name=shadow.name,
                        shape=shape,
                        dtype=np.dtype(self.dtype),
                        device_type=self.device_type,
                        device_id=self.device_id,
                        ort_value=shadow.ort_value,
                        is_dynamic=False,
                        ortvalue_device=shadow.bind_device_type,
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
            if (
                self.num_layers > 0
                and self.policy_for(self._mask_layer) is KVCachePolicy.PAGED_BLOCKS
            ):
                return int(self._seq_len)
            if self._binder is not None:
                return int(self._binder.get_static_kv_pool("kv").valid_seq_len)
            return int(self._seq_len)
        if 0 in self._cache and _tensor_shape(self._cache[0]["key"])[2] > 0:
            return int(_tensor_shape(self._cache[0]["key"])[2])
        return 0

    @property
    def current_seq_len(self) -> int:
        """
        Write cursor for the static buffer.

        The next token is stored at ``cache[:, :, current_seq_len, :]``.
        ``advance_sequence_length`` moves it forward after prefill or
        decode has written that many tokens. ``truncate_to_length``
        rewinds it. The buffer stays at ``max_seq_len``.
        """
        return self.get_sequence_length()

    def advance_sequence_length(self, num_tokens: int) -> int:
        """
        Move the write cursor forward by ``num_tokens``.

        Prefill writes the prompt into the static buffer, then advances
        by the prompt length. Decode writes one token at
        ``current_seq_len`` and advances by 1. Every layer shares that
        cursor. Returns the new length.

        Raises:
            ValueError: If ``num_tokens`` is negative, the cache is still
                dynamic, the new length exceeds ``max_seq_len``, or a
                layer would have to evict tokens to move the cursor.
        """
        with self._lock:
            count = int(num_tokens)
            if count < 0:
                raise ValueError(f"num_tokens must be >= 0, got {count}")
            self._require_static_cursor("advance_sequence_length")
            current = int(self.current_seq_len)
            new_len = current + count
            self._require_cursor_room(new_len)
            if count == 0 and all(
                int(self._layer_len[layer]) == new_len for layer in range(self.num_layers)
            ):
                return new_len
            self._set_shared_cursor(new_len)
            return new_len

    def truncate_to_length(self, length: int) -> int:
        """
        Rewind the write cursor to ``length``.

        Speculative rejection and a shorter prefill use this. The
        allocated buffer and its address stay in place. The attention
        mask covers only the retained prefix. Returns the new length.

        Raises:
            ValueError: If ``length`` is negative, past ``current_seq_len``,
                or past ``max_seq_len``, or the cache is still dynamic.
        """
        with self._lock:
            target = int(length)
            if target < 0:
                raise ValueError(f"length must be >= 0, got {target}")
            self._require_static_cursor("truncate_to_length")
            if target > int(self.max_seq_len):
                raise ValueError(
                    f"sequence length {target} exceeds max_seq_len {self.max_seq_len}"
                )
            current = int(self.current_seq_len)
            if target > current:
                raise ValueError(
                    f"truncate_to_length({target}) is past current_seq_len {current}"
                )
            self._require_cursor_room(target)
            if target == current and all(
                int(self._layer_len[layer]) == target for layer in range(self.num_layers)
            ):
                return target
            self._set_shared_cursor(target)
            return target

    def reserve(self, max_length: int) -> int:
        """
        Make sure the static buffer can hold ``max_length`` tokens.

        A dynamic CPU cache becomes static at that capacity and keeps
        tokens already stored. A cache whose ``max_seq_len`` is already
        at least ``max_length`` keeps that allocation, the write cursor,
        and any OrtValue address. Returns the reserved capacity.

        Raises:
            ValueError: If ``max_length`` is not positive, or it is larger
                than an existing ``max_seq_len``.
        """
        with self._lock:
            limit = int(max_length)
            if limit <= 0:
                raise ValueError(f"max_length must be positive, got {limit}")
            self._require_cursor_policy()
            if self.max_seq_len is not None and int(self.max_seq_len) < limit:
                raise ValueError(
                    f"reserve({limit}) exceeds max_seq_len {self.max_seq_len}"
                )
            if self.uses_device_memory or self._binder is not None:
                return int(self.max_seq_len)
            if self._static and self.max_seq_len is not None and self._cache:
                return int(self.max_seq_len)
            snapshot = self._snapshot_prefix(limit)
            shape = (
                self.batch_size,
                self.num_kv_heads,
                limit,
                self.head_dim,
            )
            self.max_seq_len = limit
            self._static = True
            self._cache = {}
            for layer in range(self.num_layers):
                self._cache[layer] = {
                    "key": np.zeros(shape, dtype=self.dtype),
                    "value": np.zeros(shape, dtype=self.dtype),
                }
            self._seq_len = 0
            self._layer_len = [0] * self.num_layers
            for layer, key_np, value_np in snapshot:
                self.update_from_present(layer, key_np, value_np)
            return int(self.max_seq_len)

    def preallocate(self, max_seq_len: int) -> int:
        """
        Allocate a fixed ``[batch, heads, max_seq_len, head_dim]`` buffer.

        A dynamic cache becomes static at that capacity. Later updates write
        into the same buffer. This is the public name for ``reserve``.
        """
        return self.reserve(max_seq_len)

    def _require_static_cursor(self, method: str) -> None:
        self._require_cursor_policy()
        if not self._static or self.max_seq_len is None:
            raise ValueError(
                f"{method} needs a static cache. Call reserve(max_length) "
                "before splitting prefill and decode."
            )

    def _require_cursor_policy(self) -> None:
        for layer in range(self.num_layers):
            policy = self.policy_for(layer)
            if policy in (KVCachePolicy.PAGED_BLOCKS, KVCachePolicy.HYBRID_SPARSE):
                raise ValueError(
                    f"Layer {layer} uses {policy.value}. The write cursor is a "
                    "dense or sliding prefix. Paged and hybrid sparse positions "
                    "move inside update_from_present."
                )

    def _require_cursor_room(self, new_len: int) -> None:
        if new_len > int(self.max_seq_len):
            raise ValueError(
                f"sequence length {new_len} exceeds max_seq_len {self.max_seq_len}"
            )
        for layer in range(self.num_layers):
            if self.policy_for(layer) is not KVCachePolicy.SLIDING_WINDOW:
                continue
            cap = int(self.window_size)
            if new_len > cap:
                raise ValueError(
                    f"Layer {layer}: length {new_len} exceeds the sliding_window "
                    f"cursor cap {cap}. Eviction stays inside update_from_present."
                )

    def _set_shared_cursor(self, new_len: int) -> None:
        """Publish one write cursor for every layer and refresh the mask."""
        for layer in range(self.num_layers):
            if (
                self.policy_for(layer) is KVCachePolicy.DENSE_CONTIGUOUS
                or self._seen[layer] <= new_len
            ):
                self._seen[layer] = new_len
            self._layer_len[layer] = new_len
            self._invalidate_quant(layer)
        self._seq_len = int(new_len)
        self._seen_published = max(self._seen) if self._seen else int(new_len)
        self._mask_layer = 0
        if self._binder is not None:
            self._binder.set_valid_sequence_length(int(new_len), "kv")

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

        A dynamic CPU cache returns NumPy arrays. A device backend, or a
        CPU cache after ``allocate_device_cache``, returns DeviceBuffer
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

    def entry(self, layer_idx: int) -> "CacheEntry":
        """
        One layer as a ``CacheEntry``.

        ``tensors`` holds the resident key and value. ``metadata`` records
        the valid length, policy, and backend. The tensors are the stored
        objects, so a device layer still points at its OrtValue.
        """
        key, value = self.get_past_tensors(layer_idx)
        return CacheEntry(
            layer_idx,
            self.cache_type,
            {"key": key, "value": value},
            {
                "seq_len": self.layer_length(layer_idx),
                "max_seq_len": self.max_seq_len,
                "policy": self.policy_for(layer_idx).value,
                "backend": self.backend.value,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "batch_size": self.batch_size,
            },
        )

    def entries(self) -> List["CacheEntry"]:
        """A ``CacheEntry`` for every layer, in layer order."""
        return [self.entry(layer) for layer in range(self.num_layers)]

    def get_ort_values(self, layer_idx: int) -> Tuple[Any, Any]:
        """
        Return the OrtValues for one layer.

        A device-resident cache returns the OrtValues already stored in
        the IOBindingManager. A dynamic CPU cache wraps its NumPy arrays,
        which allocates a new OrtValue.
        """
        if self.device_resident:
            return self.get_device_tensors(layer_idx)
        key, value = self.get_past_tensors(layer_idx)
        return _ort_value_of(key), _ort_value_of(value)

    def allocate_device_cache(
        self,
        io_manager: IOBindingManager,
        max_seq_len: int,
    ) -> IOBindingManager:
        """
        Keep this cache in ``io_manager`` at a fixed sequence length.

        The manager allocates ``[batch, heads, max_seq_len, head_dim]``
        once, or the cache adopts that pool when it already exists with
        the same layout. Tokens already stored on the host are copied
        into the prefix once. Later updates and session bindings use
        those OrtValues.

        Args:
            io_manager: The IOBindingManager that owns the session buffers
            max_seq_len: Preallocated sequence capacity

        Returns:
            The same ``io_manager``, with the KV pool installed

        Raises:
            TypeError: If ``io_manager`` is not an IOBindingManager
            ValueError: If the capacity is invalid or the manager's KV
                pool has a different layout
        """
        if not isinstance(io_manager, IOBindingManager):
            raise TypeError(
                "allocate_device_cache expects an IOBindingManager, "
                f"got {type(io_manager).__name__}"
            )
        limit = int(max_seq_len)
        if limit <= 0:
            raise ValueError(f"max_seq_len must be positive, got {limit}")
        if self._same_resident_pool(io_manager, limit):
            return io_manager

        snapshot = self._snapshot_prefix(limit)
        pool = _existing_kv_pool(io_manager)
        if pool is None:
            pool = io_manager.allocate_static_kv_cache(
                name="kv",
                num_layers=self.num_layers,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                dtype=self.dtype,
                batch_size=self.batch_size,
                max_seq_len=limit,
                seq_axis=_SEQ_AXIS,
            )
        else:
            _require_pool_match(pool, self, limit)

        self._binder = io_manager
        self._shared = []
        self.max_seq_len = limit
        self._static = True
        self._install_manager_cache(io_manager, limit)
        io_manager.set_valid_sequence_length(0, pool.name)
        self._seq_len = 0
        self._layer_len = [0] * self.num_layers
        for layer, key_np, value_np in snapshot:
            self.update_from_present(layer, key_np, value_np)
        return io_manager

    def get_device_tensors(self, layer_idx: int) -> Tuple[ort.OrtValue, ort.OrtValue]:
        """
        Return the resident key and value OrtValues for one layer.

        These are the objects to pass to ``bind_ortvalue_input`` and
        ``bind_ortvalue_output``. The call does not wrap a NumPy array.
        """
        if not self.device_resident:
            raise RuntimeError(
                "KV cache is a CPU dictionary. Call allocate_device_cache"
                "(io_manager, max_seq_len) before reading device tensors."
            )
        key, value = self.get_past_tensors(layer_idx)
        key_ort = getattr(key, "ort_value", None)
        value_ort = getattr(value, "ort_value", None)
        if key_ort is None or value_ort is None:
            raise RuntimeError(f"Layer {layer_idx} has no resident OrtValue")
        return key_ort, value_ort

    def _attach_shadows(self) -> None:
        """Pair each active KV buffer with an inactive one of the same shape."""
        if not self._concat_swap_active():
            return
        if self._binder is not None:
            pool = self._binder.ensure_kv_shadows("kv")
            for layer in range(self.num_layers):
                key_shadow = self._binder.get_buffer(pool.shadow_key_names[layer])
                value_shadow = self._binder.get_buffer(pool.shadow_value_names[layer])
                if key_shadow is None or value_shadow is None:
                    raise RuntimeError(f"KV layer {layer} is missing a shadow buffer")
                self._cache[layer]["key_shadow"] = key_shadow
                self._cache[layer]["value_shadow"] = value_shadow
            return
        for layer in range(self.num_layers):
            entry = self._cache.get(layer)
            if entry is None or "key_shadow" in entry:
                continue
            shape = _tensor_shape(entry["key"])
            entry["key_shadow"] = np.zeros(shape, dtype=self.dtype)
            entry["value_shadow"] = np.zeros(shape, dtype=self.dtype)

    def _shadow_pair(self, layer_idx: int) -> Tuple[Any, Any]:
        entry = self._cache[layer_idx]
        key = entry.get("key_shadow")
        value = entry.get("value_shadow")
        if key is None or value is None:
            raise RuntimeError(
                f"Layer {layer_idx} has no KV shadow buffer. concat_swap keeps "
                "one active buffer and one inactive buffer at max_seq_len."
            )
        return key, value

    def shadow_ort_values(self, layer_idx: int) -> Tuple[Any, Any]:
        """
        Return the inactive key and value for one layer.

        ``bind_cache_to_session`` binds these as the present outputs when
        the update mode is ``concat_swap``. After the step, call
        ``commit_present_swap`` so the next past input reads them.
        """
        key, value = self._shadow_pair(layer_idx)
        return _ort_value_of(key), _ort_value_of(value)

    def _swap_layer(self, layer_idx: int) -> None:
        """Exchange the active and shadow storage. The shape does not change."""
        entry = self._cache[layer_idx]
        active_key = entry["key"]
        active_value = entry["value"]
        shadow_key, shadow_value = self._shadow_pair(layer_idx)
        if isinstance(active_key, np.ndarray):
            entry["key"], entry["key_shadow"] = shadow_key, active_key
            entry["value"], entry["value_shadow"] = shadow_value, active_value
            return
        if active_key.ort_value is None or shadow_key.ort_value is None:
            raise RuntimeError(f"Layer {layer_idx} key buffer is not allocated")
        if active_value.ort_value is None or shadow_value.ort_value is None:
            raise RuntimeError(f"Layer {layer_idx} value buffer is not allocated")
        active_key.ort_value, shadow_key.ort_value = (
            shadow_key.ort_value,
            active_key.ort_value,
        )
        active_value.ort_value, shadow_value.ort_value = (
            shadow_value.ort_value,
            active_value.ort_value,
        )

    def _publish_prefix(
        self,
        layer_idx: int,
        key: np.ndarray,
        value: np.ndarray,
        length: int,
    ) -> None:
        """
        Write a concatenated prefix into the shadow and make it active.

        ``length`` is the valid sequence, not a new allocation. The shadow
        stays ``[batch, heads, max_seq_len, head_dim]``.
        """
        published_key = np.ascontiguousarray(key, dtype=self.dtype)
        published_value = np.ascontiguousarray(value, dtype=self.dtype)
        if int(published_key.shape[_SEQ_AXIS]) != int(length):
            raise ValueError(
                f"Layer {layer_idx}: concatenated key length "
                f"{published_key.shape[_SEQ_AXIS]} != {length}"
            )
        if int(published_value.shape[_SEQ_AXIS]) != int(length):
            raise ValueError(
                f"Layer {layer_idx}: concatenated value length "
                f"{published_value.shape[_SEQ_AXIS]} != {length}"
            )
        if length < 0 or (
            self.max_seq_len is not None and int(length) > int(self.max_seq_len)
        ):
            raise ValueError(
                f"Layer {layer_idx}: sequence length {length} exceeds "
                f"max_seq_len {self.max_seq_len}"
            )
        shadow_key, shadow_value = self._shadow_pair(layer_idx)
        if length:
            _write_window(shadow_key, published_key, 0)
            _write_window(shadow_value, published_value, 0)
        self._swap_layer(layer_idx)

    def _concat_swap_dense(
        self,
        layer_idx: int,
        stored_key: Any,
        stored_value: Any,
        present_key: Any,
        present_value: Any,
        position: int,
        valid_seq_len: Optional[int],
    ) -> None:
        """
        Publish ``concat(old[:position], new)`` without writing the live buffer.

        A present tensor longer than ``position`` is already the concatenated
        cache, and its prefix of that length is what becomes active. A shorter
        tensor is the new tokens only. The shadow OrtValue, which is where
        ``bind_cache_to_session`` sends the present output, is adopted with
        ``valid_seq_len`` and no second copy.
        """
        shadow_key, shadow_value = self._shadow_pair(layer_idx)
        if _same_storage(shadow_key, present_key) and _same_storage(shadow_value, present_value):
            if valid_seq_len is None:
                raise ValueError(
                    f"Layer {layer_idx}: the present output is the shadow buffer "
                    "of max_seq_len. Pass valid_seq_len or call commit_present_swap."
                )
            self._swap_layer(layer_idx)
            new_len = int(valid_seq_len)
            self._commit_length(layer_idx, new_len)
            self._seen[layer_idx] = new_len
            self._seen_published = new_len
            return

        key_np = _as_numpy(present_key)
        value_np = _as_numpy(present_value)
        _, _, new_len = _placement(key_np, position)
        if new_len < 0 or (
            self.max_seq_len is not None and new_len > int(self.max_seq_len)
        ):
            raise ValueError(
                f"Layer {layer_idx}: sequence length {new_len} exceeds "
                f"max_seq_len {self.max_seq_len}"
            )
        seq = int(key_np.shape[_SEQ_AXIS])
        if seq == position:
            return
        if seq > position:
            published_key = np.ascontiguousarray(key_np[:, :, :new_len, :])
            published_value = np.ascontiguousarray(value_np[:, :, :new_len, :])
        elif position:
            published_key = _cat_seq([
                _read_prefix(stored_key, position),
                np.ascontiguousarray(key_np),
            ])
            published_value = _cat_seq([
                _read_prefix(stored_value, position),
                np.ascontiguousarray(value_np),
            ])
        else:
            published_key = np.ascontiguousarray(key_np)
            published_value = np.ascontiguousarray(value_np)
        self._publish_prefix(layer_idx, published_key, published_value, new_len)
        self._commit_length(layer_idx, new_len)
        self._seen[layer_idx] = int(new_len)
        self._seen_published = int(new_len)

    def commit_present_swap(self, layer_idx: int, valid_seq_len: int) -> int:
        """
        Install the shadow buffer as the next past cache.

        Call this after the NPU writes the concatenated present tensor into
        the output bound by ``bind_cache_to_session``. The active pointer
        changes. ``max_seq_len`` does not. ``valid_seq_len`` is the filled
        prefix, so the padded tail stays masked.
        """
        with self._lock:
            if not self._concat_swap_active():
                raise RuntimeError(
                    "commit_present_swap requires concat_swap on a static cache"
                )
            if layer_idx not in self._cache:
                raise KeyError(
                    f"Layer {layer_idx} not in cache (have {self.num_layers} layers)"
                )
            length = int(valid_seq_len)
            if length < 0 or (
                self.max_seq_len is not None and length > int(self.max_seq_len)
            ):
                raise ValueError(
                    f"Layer {layer_idx}: sequence length {length} exceeds "
                    f"max_seq_len {self.max_seq_len}"
                )
            self._invalidate_quant(layer_idx)
            self._swap_layer(layer_idx)
            self._commit_length(layer_idx, length)
            if length > int(self._seen[layer_idx]):
                self._seen[layer_idx] = length
                self._seen_published = int(self._seen[layer_idx])
            return length

    def run_bound_session(self, session: Any, layer_idx: int, valid_seq_len: int) -> Dict[str, str]:
        """Run one layer and install the present buffer as the next past.

        ``VitisAIExecutionProvider`` in the Ryzen AI 1.8 wheel has no NPU
        OrtValue device. The session reads and writes the host OrtValues
        from ``bind_cache_to_session``. ``concat_swap`` binds present to
        the shadow buffer, so this commits the swap only after the run
        returns.
        """
        binding = session.io_binding()
        bound = self.bind_cache_to_session(session, binding, layer_idx)
        session.run_with_iobinding(binding)
        if self._concat_swap_active():
            self.commit_present_swap(layer_idx, valid_seq_len)
        return bound

    def bind_cache_to_session(
        self,
        session: Any,
        io_binding: Any,
        layer_idx: int,
    ) -> Dict[str, str]:
        """
        Bind one layer's resident cache into an ONNX Runtime session.

        ``slot_write`` binds past and present to the same OrtValue.
        ``concat_swap`` binds past to the active OrtValue and present to
        the shadow OrtValue, so a concatenated present does not overlap
        the cache the graph is reading. Call ``commit_present_swap`` after
        the session runs. The static attention mask and a sequence-length
        scalar are bound when the session has those inputs. A present
        tensor whose concrete sequence length is not ``max_seq_len`` is
        rejected so a short output is not written over the preallocated
        buffer.

        Args:
            session: ONNX Runtime session, or any object with
                ``get_inputs`` and ``get_outputs``
            io_binding: Object with ``bind_ortvalue_input`` and
                ``bind_ortvalue_output``
            layer_idx: Transformer layer to bind

        Returns:
            Map of roles to the session tensor names that were bound
        """
        key_ort, value_ort = self.get_device_tensors(layer_idx)
        if not self._static or self.max_seq_len is None:
            raise RuntimeError(
                "bind_cache_to_session requires a static cache. "
                "Call allocate_device_cache(io_manager, max_seq_len) first."
            )
        inputs = list(session.get_inputs())
        outputs = list(session.get_outputs())
        input_names = [node.name for node in inputs]
        output_names = [node.name for node in outputs]
        past_key = _resolve_kv_name(input_names, layer_idx, "key", "past")
        past_value = _resolve_kv_name(input_names, layer_idx, "value", "past")
        present_key = _resolve_kv_name(output_names, layer_idx, "key", "present")
        present_value = _resolve_kv_name(output_names, layer_idx, "value", "present")
        if past_key is None or past_value is None:
            raise ValueError(
                f"Session has no past key/value inputs for layer {layer_idx}. "
                f"Inputs: {input_names}"
            )
        if present_key is None or present_value is None:
            raise ValueError(
                f"Session has no present key/value outputs for layer {layer_idx}. "
                f"Outputs: {output_names}"
            )
        cache_shape = _ort_shape(key_ort)
        nodes = {node.name: node for node in list(inputs) + list(outputs)}
        for name in (past_key, past_value, present_key, present_value):
            _require_static_shape(nodes[name], cache_shape, int(self.max_seq_len))

        present_key_ort = key_ort
        present_value_ort = value_ort
        if self._concat_swap_active():
            present_key_ort, present_value_ort = self.shadow_ort_values(layer_idx)
        io_binding.bind_ortvalue_input(past_key, key_ort)
        io_binding.bind_ortvalue_input(past_value, value_ort)
        io_binding.bind_ortvalue_output(present_key, present_key_ort)
        io_binding.bind_ortvalue_output(present_value, present_value_ort)
        bound = {
            "past_key": past_key,
            "past_value": past_value,
            "present_key": present_key,
            "present_value": present_value,
        }
        for mask_name in _mask_names(input_names):
            mask_node = nodes[mask_name]
            mask_shape = (self.batch_size, int(self.max_seq_len))
            _require_static_shape(mask_node, mask_shape, int(self.max_seq_len))
            io_binding.bind_ortvalue_input(mask_name, self._mask_ort_value())
            bound["attention_mask"] = mask_name
        for scalar_name in _scalar_names(input_names):
            _require_scalar_shape(nodes[scalar_name])
            io_binding.bind_ortvalue_input(scalar_name, self._scalar_ort_value())
            bound["sequence_length"] = scalar_name
        return bound

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
        if (
            self.num_layers > 0
            and self.policy_for(self._mask_layer) is KVCachePolicy.PAGED_BLOCKS
        ):
            mask = self._physical_mask(self._mask_layer)
            self._store_custom_mask(mask)
            return mask
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
        with self._lock:
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

            self._invalidate_quant(layer_idx)

            if not self._static:
                self._seen[layer_idx] = int(key_shape[_SEQ_AXIS])
                self._seen_published = self._seen[layer_idx]
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

            if self.policy_for(layer_idx) is not KVCachePolicy.DENSE_CONTIGUOUS:
                self._update_bounded(
                    layer_idx, stored_key, stored_value, present_key, present_value
                )
                return

            position = self._layer_len[layer_idx]
            if self._concat_swap_active():
                self._concat_swap_dense(
                    layer_idx,
                    stored_key,
                    stored_value,
                    present_key,
                    present_value,
                    position,
                    valid_seq_len,
                )
                return
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
            self._seen[layer_idx] = int(new_len)
            self._seen_published = int(new_len)

    def update_prefill(self, layer_idx: int, keys: Any, values: Any) -> int:
        """
        Write a prefill span for one layer.

        The first call fills an empty layer. A later call appends the next
        prefill chunk at the write cursor. A sliding-window layer records
        ``phase`` ``sliding`` and keeps ``execution_phase`` ``prefill``.
        A draft cache records ``speculative_draft``. Returns the resident
        length.
        """
        with self._lock:
            seq = self._phase_span(layer_idx, keys, values)
            if seq < 1:
                raise ValueError(
                    f"Layer {layer_idx}: prefill writes one or more tokens, got {seq}"
                )
            if self.layer_length(layer_idx) == 0:
                self.update_from_present(layer_idx, keys, values)
            else:
                self._append_new_tokens(layer_idx, keys, values)
            self._note_phase(layer_idx, "prefill")
            return self.layer_length(layer_idx)

    def update_decode(self, layer_idx: int, key: Any, value: Any) -> int:
        """
        Append one decode token at the write cursor.

        The sequence axis must be 1. A sliding-window layer records
        ``phase`` ``sliding``. A draft cache records ``speculative_draft``.
        Returns the resident length.
        """
        with self._lock:
            seq = self._phase_span(layer_idx, key, value)
            if seq != 1:
                raise ValueError(
                    f"Layer {layer_idx}: decode writes one token, got sequence length {seq}"
                )
            self._append_new_tokens(layer_idx, key, value)
            self._note_phase(layer_idx, "decode")
            return self.layer_length(layer_idx)

    def update_verify(
        self,
        layer_idx: int,
        draft_keys: Any,
        draft_values: Any,
        accepted_length: int,
    ) -> int:
        """
        Keep the accepted prefix of a draft span.

        ``accepted_length`` counts accepted draft tokens, from 0 through
        the draft sequence. Rejected tokens stay outside the cursor.
        ``phase`` becomes ``verify``. Returns the resident length.
        """
        with self._lock:
            seq = self._phase_span(layer_idx, draft_keys, draft_values)
            accepted = int(accepted_length)
            if accepted < 0 or accepted > seq:
                raise ValueError(
                    f"Layer {layer_idx}: accepted_length {accepted} is outside "
                    f"the draft span of {seq}"
                )
            if accepted:
                keys = _as_numpy(draft_keys)
                values = _as_numpy(draft_values)
                self._append_new_tokens(
                    layer_idx,
                    keys[:, :, :accepted, :],
                    values[:, :, :accepted, :],
                )
            self._note_phase(layer_idx, "verify")
            return self.layer_length(layer_idx)

    def _phase_span(self, layer_idx: int, key: Any, value: Any) -> int:
        if layer_idx not in self._cache:
            raise KeyError(
                f"Layer {layer_idx} not in cache (have {self.num_layers} layers)"
            )
        key_shape = _tensor_shape(key)
        value_shape = _tensor_shape(value)
        if key_shape != value_shape:
            raise ValueError(
                f"Layer {layer_idx}: key shape {key_shape} != value shape {value_shape}"
            )
        if len(key_shape) != 4:
            raise ValueError(
                f"Layer {layer_idx}: expected 4D tensor [batch, heads, seq, head_dim], "
                f"got shape {key_shape}"
            )
        if key_shape[1] != self.num_kv_heads or key_shape[3] != self.head_dim:
            raise ValueError(
                f"Layer {layer_idx}: expected heads {self.num_kv_heads} and "
                f"head_dim {self.head_dim}, got {key_shape}"
            )
        return int(key_shape[_SEQ_AXIS])

    def _note_phase(self, layer_idx: int, execution: str) -> None:
        self.execution_phase = execution
        if self.role == "draft":
            self.phase = "speculative_draft"
        elif execution == "verify":
            self.phase = "verify"
        elif self.policy_for(layer_idx) is KVCachePolicy.SLIDING_WINDOW:
            self.phase = "sliding"
        else:
            self.phase = execution

    def _append_new_tokens(self, layer_idx: int, key: Any, value: Any) -> None:
        """Append tokens at the layer cursor. The tensor is new tokens only."""
        self._invalidate_quant(layer_idx)
        key_np = np.ascontiguousarray(_as_numpy(key), dtype=self.dtype)
        value_np = np.ascontiguousarray(_as_numpy(value), dtype=self.dtype)
        added = int(key_np.shape[_SEQ_AXIS])
        if added < 1:
            return
        if not self._static:
            stored_key = self._cache[layer_idx]["key"]
            stored_value = self._cache[layer_idx]["value"]
            if int(_tensor_shape(stored_key)[_SEQ_AXIS]) == 0:
                self._cache[layer_idx]["key"] = key_np
                self._cache[layer_idx]["value"] = value_np
            else:
                self._cache[layer_idx]["key"] = np.concatenate(
                    [_as_numpy(stored_key), key_np], axis=_SEQ_AXIS
                )
                self._cache[layer_idx]["value"] = np.concatenate(
                    [_as_numpy(stored_value), value_np], axis=_SEQ_AXIS
                )
            self._seen[layer_idx] = int(self._cache[layer_idx]["key"].shape[_SEQ_AXIS])
            self._seen_published = int(self._seen[layer_idx])
            return
        stored_key = self._cache[layer_idx]["key"]
        stored_value = self._cache[layer_idx]["value"]
        if self.policy_for(layer_idx) is KVCachePolicy.DENSE_CONTIGUOUS:
            position = int(self._layer_len[layer_idx])
            new_len = position + added
            if self.max_seq_len is not None and new_len > int(self.max_seq_len):
                raise ValueError(
                    f"Layer {layer_idx}: sequence length {new_len} exceeds "
                    f"max_seq_len {self.max_seq_len}"
                )
            if self._concat_swap_active():
                if position:
                    published_key = _cat_seq([
                        _read_prefix(stored_key, position),
                        key_np,
                    ])
                    published_value = _cat_seq([
                        _read_prefix(stored_value, position),
                        value_np,
                    ])
                else:
                    published_key = key_np
                    published_value = value_np
                self._publish_prefix(layer_idx, published_key, published_value, new_len)
            else:
                _write_window(stored_key, key_np, position)
                _write_window(stored_value, value_np, position)
            self._commit_length(layer_idx, new_len)
            self._seen[layer_idx] = int(new_len)
            self._seen_published = int(new_len)
            return
        if self.policy_for(layer_idx) is KVCachePolicy.PAGED_BLOCKS:
            self._paged_append(layer_idx, stored_key, stored_value, key_np, value_np)
        else:
            self._compact_append(layer_idx, stored_key, stored_value, key_np, value_np)
        self._seen[layer_idx] += added
        self._seen_published = int(self._seen[layer_idx])

    def _append_position(self, layer_idx: int) -> int:
        if self.policy_for(layer_idx) is KVCachePolicy.PAGED_BLOCKS:
            return int(self._pages[layer_idx].write_pos)
        return int(self._layer_len[layer_idx])

    def _update_bounded(
        self,
        layer_idx: int,
        stored_key: Any,
        stored_value: Any,
        present_key: Any,
        present_value: Any,
    ) -> None:
        """Apply sliding, paged, or hybrid retention without resizing storage."""
        position = self._append_position(layer_idx)
        key_np = _as_numpy(present_key)
        value_np = _as_numpy(present_value)
        key_window, _, _ = _placement(key_np, position)
        value_window, _, _ = _placement(value_np, position)
        if key_window is None or value_window is None:
            return
        if self.policy_for(layer_idx) is KVCachePolicy.PAGED_BLOCKS:
            self._paged_append(layer_idx, stored_key, stored_value, key_window, value_window)
        else:
            self._compact_append(
                layer_idx, stored_key, stored_value, key_window, value_window
            )
        self._seen[layer_idx] += int(key_window.shape[_SEQ_AXIS])
        self._seen_published = int(self._seen[layer_idx])

    def _compact_append(
        self,
        layer_idx: int,
        stored_key: Any,
        stored_value: Any,
        added_key: np.ndarray,
        added_value: np.ndarray,
    ) -> None:
        prior = int(self._layer_len[layer_idx])
        if prior:
            key = np.concatenate(
                [_read_prefix(stored_key, prior), np.ascontiguousarray(added_key)],
                axis=_SEQ_AXIS,
            )
            value = np.concatenate(
                [_read_prefix(stored_value, prior), np.ascontiguousarray(added_value)],
                axis=_SEQ_AXIS,
            )
        else:
            key = np.ascontiguousarray(added_key)
            value = np.ascontiguousarray(added_value)
        key, value = self._retain(layer_idx, key, value, prior)
        if key.shape[_SEQ_AXIS] > int(self.max_seq_len):
            raise ValueError(
                f"Layer {layer_idx}: retained sequence {key.shape[_SEQ_AXIS]} "
                f"exceeds max_seq_len {self.max_seq_len}"
            )
        retained = int(key.shape[_SEQ_AXIS])
        if self._concat_swap_active():
            if retained:
                self._publish_prefix(layer_idx, key, value, retained)
            self._commit_length(layer_idx, retained)
            return
        if retained:
            _write_window(stored_key, key, 0)
            _write_window(stored_value, value, 0)
        self._commit_length(layer_idx, retained)

    def _retain(
        self,
        layer_idx: int,
        key: np.ndarray,
        value: np.ndarray,
        prior_len: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        policy = self.policy_for(layer_idx)
        if policy is KVCachePolicy.SLIDING_WINDOW:
            window = int(self.window_size)
            if key.shape[_SEQ_AXIS] > window:
                key = key[:, :, -window:, :]
                value = value[:, :, -window:, :]
            self._sink_held[layer_idx] = 0
            self._retrieved_len[layer_idx] = 0
            return np.ascontiguousarray(key), np.ascontiguousarray(value)
        if prior_len == 0:
            return self._retain_hybrid_prefill(layer_idx, key, value)
        return self._retain_hybrid_decode(layer_idx, key, value)

    def _retain_hybrid_prefill(
        self,
        layer_idx: int,
        key: np.ndarray,
        value: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        seq = int(key.shape[_SEQ_AXIS])
        sink_n = min(int(self.sink_size), seq)
        available = max(0, seq - sink_n)
        window_n = min(int(self.window_size), available)
        recent_start = seq - window_n
        retrieved_k, retrieved_v, retrieved_n = self._gather_pinned(
            layer_idx, key, value, sink_n, recent_start, seq
        )
        kept_key = _cat_seq(
            [key[:, :, :sink_n, :], retrieved_k, key[:, :, recent_start:, :]]
        )
        kept_value = _cat_seq(
            [value[:, :, :sink_n, :], retrieved_v, value[:, :, recent_start:, :]]
        )
        self._sink_held[layer_idx] = sink_n
        self._retrieved_len[layer_idx] = retrieved_n
        return kept_key, kept_value

    def _retain_hybrid_decode(
        self,
        layer_idx: int,
        key: np.ndarray,
        value: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        sink_n = int(self._sink_held[layer_idx])
        retrieved_n = int(self._retrieved_len[layer_idx])
        split = sink_n + retrieved_n
        recent_key = key[:, :, split:, :]
        recent_value = value[:, :, split:, :]
        window = int(self.window_size)
        if recent_key.shape[_SEQ_AXIS] > window:
            recent_key = recent_key[:, :, -window:, :]
            recent_value = recent_value[:, :, -window:, :]
        kept_key = _cat_seq(
            [key[:, :, :sink_n, :], key[:, :, sink_n:split, :], recent_key]
        )
        kept_value = _cat_seq(
            [value[:, :, :sink_n, :], value[:, :, sink_n:split, :], recent_value]
        )
        return kept_key, kept_value

    def _gather_pinned(
        self,
        layer_idx: int,
        key: np.ndarray,
        value: np.ndarray,
        sink_n: int,
        recent_start: int,
        seq: int,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        key_parts: List[np.ndarray] = []
        value_parts: List[np.ndarray] = []
        for block_id in sorted(self._pinned[layer_idx]):
            start = int(block_id) * int(self.block_size)
            end = min(seq, start + int(self.block_size))
            if start >= seq or end <= start:
                raise ValueError(
                    f"Layer {layer_idx}: pinned block {block_id} is outside "
                    f"the present sequence of {seq}"
                )
            keep_lo = max(start, sink_n)
            keep_hi = min(end, recent_start)
            if keep_hi > keep_lo:
                key_parts.append(key[:, :, keep_lo:keep_hi, :])
                value_parts.append(value[:, :, keep_lo:keep_hi, :])
        if not key_parts:
            empty = key[:, :, :0, :]
            return empty, value[:, :, :0, :], 0
        gathered_key = np.concatenate(key_parts, axis=_SEQ_AXIS)
        gathered_value = np.concatenate(value_parts, axis=_SEQ_AXIS)
        limit = int(self.retrieval_blocks) * int(self.block_size)
        if gathered_key.shape[_SEQ_AXIS] > limit:
            raise ValueError(
                f"Layer {layer_idx}: pinned tokens {gathered_key.shape[_SEQ_AXIS]} "
                f"exceed retrieval_blocks capacity {limit}"
            )
        return gathered_key, gathered_value, int(gathered_key.shape[_SEQ_AXIS])

    def _paged_append(
        self,
        layer_idx: int,
        stored_key: Any,
        stored_value: Any,
        added_key: np.ndarray,
        added_value: np.ndarray,
    ) -> None:
        if self._concat_swap_active():
            host_key = np.array(_as_numpy(stored_key), copy=True)
            host_value = np.array(_as_numpy(stored_value), copy=True)
            self._paged_fill(layer_idx, host_key, host_value, added_key, added_value)
            self._publish_prefix(
                layer_idx,
                host_key,
                host_value,
                int(self.max_seq_len),
            )
            self._commit_length(layer_idx, self._pages[layer_idx].resident)
            self._store_custom_mask(self._physical_mask(layer_idx))
            return
        self._paged_fill(layer_idx, stored_key, stored_value, added_key, added_value)
        self._commit_length(layer_idx, self._pages[layer_idx].resident)
        self._store_custom_mask(self._physical_mask(layer_idx))

    def _paged_fill(
        self,
        layer_idx: int,
        stored_key: Any,
        stored_value: Any,
        added_key: np.ndarray,
        added_value: np.ndarray,
    ) -> None:
        """Scatter new tokens into physical pages of ``stored_key``."""
        state = self._pages[layer_idx]
        remaining = int(added_key.shape[_SEQ_AXIS])
        source = 0
        position = int(state.write_pos)
        while remaining:
            logical = position // int(self.block_size)
            offset = position % int(self.block_size)
            physical = state.ensure(logical, self.max_seq_len)
            room = int(self.block_size) - offset
            take = min(room, remaining)
            dest = physical * int(self.block_size) + offset
            _write_window(stored_key, added_key[:, :, source:source + take, :], dest)
            _write_window(stored_value, added_value[:, :, source:source + take, :], dest)
            state.fill[logical] += take
            state.resident += take
            position += take
            source += take
            remaining -= take
        state.write_pos = position

    def _physical_mask(self, layer_idx: int) -> np.ndarray:
        width = int(self.max_seq_len)
        mask = np.zeros((self.batch_size, width), dtype=np.int64)
        state = self._pages[layer_idx]
        for logical, physical in enumerate(state.table):
            if physical < 0:
                continue
            start = int(physical) * int(self.block_size)
            filled = int(state.fill[logical])
            if filled:
                mask[:, start:start + filled] = 1
        return mask

    def _store_custom_mask(self, mask: np.ndarray) -> np.ndarray:
        if self._binder is None:
            return mask
        pool = self._binder.get_static_kv_pool("kv")
        pool.valid_seq_len = int(self._seq_len)
        buffer = self._binder.get_buffer(pool.mask_name)
        if buffer is not None and buffer.ort_value is not None:
            _refresh_ort_contents(buffer.ort_value, mask)
        return mask

    def _injected_past(self, layer_idx: int, stored: Any) -> Any:
        """
        Past tensor to feed a graph for one layer.

        A static CPU pool without a binder reserves ``max_seq_len`` rows
        but only the first ``layer_length`` rows are valid. Concat-style
        dynamic graphs append new tokens to whatever past they receive,
        so injecting the full pool makes ``present`` grow to
        ``max_seq_len + delta`` and overflow the pool on the first step.
        Slicing the valid prefix keeps ``present`` within capacity while
        the allocated pool address stays put for later writes.

        Device-resident caches (binder-attached AOT graphs) receive the
        pool unchanged: those graphs track the valid length through the
        binder's mask instead of the past shape.

        Args:
            layer_idx: Layer whose cache entry is being injected
            stored: Cache entry as held in ``self._cache``

        Returns:
            Contiguous prefix window, or ``stored`` unchanged when no
            windowing applies
        """
        if not (
            self._static
            and self._binder is None
            and isinstance(stored, np.ndarray)
        ):
            return stored
        length = self.layer_length(layer_idx)
        if stored.shape[_SEQ_AXIS] <= length:
            return stored
        slices: List[slice] = [slice(None)] * stored.ndim
        slices[_SEQ_AXIS] = slice(0, length)
        # ascontiguousarray: a partial slice along axis 2 is strided, and
        # IOBindingManager.bind_inputs creates OrtValues from the numpy
        # buffer, which requires contiguous memory.
        return np.ascontiguousarray(stored[tuple(slices)])

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

        A static pool without a binder injects the valid prefix window
        (see ``_injected_past``) so dynamic graphs observe a past whose
        sequence axis equals the committed length, not ``max_seq_len``.

        Args:
            registry: Dictionary to inject cache tensors into
            key_format: Format string for key tensor names (use {layer} placeholder)
            value_format: Format string for value tensor names (use {layer} placeholder)
        """
        with self._lock:
            for i in range(self.num_layers):
                key_name = key_format.format(layer=i)
                value_name = value_format.format(layer=i)
                registry[key_name] = self._injected_past(i, self._cache[i]["key"])
                registry[value_name] = self._injected_past(i, self._cache[i]["value"])

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
        with self._lock:
            updated = 0
            for i in range(self.num_layers):
                key_name = present_key_format.format(layer=i)
                value_name = present_value_format.format(layer=i)

                if key_name in registry and value_name in registry:
                    self.update_from_present(i, registry[key_name], registry[value_name])
                    updated += 1

            return updated

    def quantize_kv(self, layer_idx: Optional[int] = None) -> None:
        """
        Compress resident KV into ``cache_dtype``.

        ``key_scale`` and ``value_scale`` receive one float32 scale per
        layer for int8, fp8, int4, and fp4. fp16 and bf16 are stored by
        casting. The compute buffer and its OrtValue stay in place.
        """
        if self.kv_quantization is None:
            raise ValueError(
                "kv_quantization is not set. Choose fp16, bf16, int8, fp8, int4, or fp4."
            )
        if layer_idx is None:
            layers = range(self.num_layers)
        else:
            if layer_idx < 0 or layer_idx >= self.num_layers:
                raise KeyError(
                    f"Layer {layer_idx} not in cache (have {self.num_layers} layers)"
                )
            layers = (layer_idx,)
        for layer in layers:
            key, value = self._quant_source(layer)
            stored_key, key_scale = _quantize_array(key, self.kv_quantization)
            stored_value, value_scale = _quantize_array(value, self.kv_quantization)
            if stored_key.dtype != self.cache_dtype or stored_value.dtype != self.cache_dtype:
                raise RuntimeError(
                    f"Quantized KV dtype {stored_key.dtype} does not match "
                    f"cache_dtype {self.cache_dtype}"
                )
            self._compressed[layer] = {"key": stored_key, "value": stored_value}
            self._quant_valid[layer] = True
            if self.key_scale is not None:
                self.key_scale[layer] = np.float32(key_scale)
                self.value_scale[layer] = np.float32(value_scale)

    def dequantize_kv(self, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return one layer in the compute ``dtype``.

        A fresh quantized copy is expanded from ``cache_dtype``. After an
        update, the compressed copy is stale and this reads the compute
        buffer until ``quantize_kv`` runs again.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise KeyError(
                f"Layer {layer_idx} not in cache (have {self.num_layers} layers)"
            )
        if self.kv_quantization is None or not self._quant_valid[layer_idx]:
            key, value = self._quant_source(layer_idx)
            return (
                np.ascontiguousarray(key, dtype=self.dtype),
                np.ascontiguousarray(value, dtype=self.dtype),
            )
        stored = self._compressed[layer_idx]
        key_scale = None if self.key_scale is None else float(self.key_scale[layer_idx])
        value_scale = None if self.value_scale is None else float(self.value_scale[layer_idx])
        key = _dequantize_array(
            stored["key"], self.kv_quantization, key_scale, self.head_dim
        )
        value = _dequantize_array(
            stored["value"], self.kv_quantization, value_scale, self.head_dim
        )
        return (
            np.ascontiguousarray(key, dtype=self.dtype),
            np.ascontiguousarray(value, dtype=self.dtype),
        )

    def _invalidate_quant(self, layer_idx: int) -> None:
        self._quant_valid[layer_idx] = False
        if self.key_scale is not None:
            self.key_scale[layer_idx] = np.float32(np.nan)
            self.value_scale[layer_idx] = np.float32(np.nan)

    def _quant_source(self, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        key = _as_numpy(self._cache[layer_idx]["key"])
        value = _as_numpy(self._cache[layer_idx]["value"])
        if self.policy_for(layer_idx) is KVCachePolicy.PAGED_BLOCKS and self.max_seq_len:
            length = int(self.max_seq_len)
        elif self._static:
            length = int(self.layer_length(layer_idx))
        else:
            length = int(key.shape[_SEQ_AXIS])
        return key[:, :, :length, :], value[:, :, :length, :]

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

        shadow_bytes = 0
        for i in range(self.num_layers):
            key_bytes = _nbytes(self._cache[i]["key"])
            value_bytes = _nbytes(self._cache[i]["value"])
            layer_bytes = key_bytes + value_bytes
            for extra in ("key_shadow", "value_shadow"):
                if extra in self._cache[i]:
                    extra_bytes = _nbytes(self._cache[i][extra])
                    layer_bytes += extra_bytes
                    shadow_bytes += extra_bytes
            per_layer.append(layer_bytes)
            total_bytes += layer_bytes

        seq_len = self.get_sequence_length()
        compressed_bytes = 0
        quantized_layers = 0
        for layer, blob in self._compressed.items():
            if not self._quant_valid[layer]:
                continue
            quantized_layers += 1
            compressed_bytes += int(blob["key"].nbytes) + int(blob["value"].nbytes)

        return {
            "total_bytes": total_bytes,
            "total_mb": total_bytes / (1024 * 1024),
            "per_layer_bytes": per_layer,
            "sequence_length": seq_len,
            "num_layers": self.num_layers,
            "dtype": str(self.dtype),
            "cache_dtype": str(self.cache_dtype),
            "kv_quantization": self.kv_quantization,
            "compressed_bytes": compressed_bytes,
            "quantized_layers": quantized_layers,
            "backend": self.backend.value,
            "max_seq_len": self.max_seq_len,
            "current_seq_len": seq_len,
            "static": self._static,
            "policy": self.policy.value,
            "window_size": self.window_size,
            "block_size": self.block_size,
            "sink_size": self.sink_size,
            "seen_tokens": self.seen_tokens,
            "storage": "device_buffer" if self.device_resident else "numpy",
            "update_mode": self.update_mode.value,
            "shadow_bytes": shadow_bytes,
            "phase": self.phase,
            "execution_phase": self.execution_phase,
            "role": self.role,
        }

    def __repr__(self) -> str:
        seq_len = self.get_sequence_length()
        mem = self.get_memory_usage()
        return (
            f"KVCacheManager(layers={self.num_layers}, heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, seq_len={seq_len}, "
            f"backend={self.backend.value}, policy={self.policy.value}, "
            f"memory={mem['total_mb']:.2f}MB)"
        )


def _ort_shape(value: Any) -> Tuple[int, ...]:
    shape = value.shape() if callable(getattr(value, "shape", None)) else value.shape
    return tuple(int(dim) for dim in shape)


def _as_ort_contents(data: np.ndarray, host_shape: Tuple[int, ...]) -> np.ndarray:
    """Match a 0-d length to the 1-element vector some OrtValue builds store."""
    incoming = np.ascontiguousarray(data)
    if incoming.shape == host_shape:
        return incoming
    if incoming.shape == () and host_shape == (1,):
        return incoming.reshape(1)
    if incoming.shape == (1,) and host_shape == ():
        return incoming.reshape(())
    raise ValueError(
        f"Cannot refresh OrtValue shape {host_shape} with {incoming.shape}"
    )


def _same_scalar_storage(stored: Tuple[int, ...], incoming: Tuple[int, ...]) -> bool:
    if stored == incoming:
        return True
    return stored in {(), (1,)} and incoming in {(), (1,)}


def _refresh_ort_contents(ort_value: ort.OrtValue, data: np.ndarray) -> None:
    host = ort_value.numpy()
    incoming = _as_ort_contents(np.ascontiguousarray(data, dtype=host.dtype), tuple(host.shape))
    host[...] = incoming
    if int(host.ctypes.data) != int(ort_value.data_ptr()):
        ort_value.update_inplace(np.ascontiguousarray(host))


def _existing_kv_pool(io_manager: IOBindingManager) -> Any:
    try:
        return io_manager.get_static_kv_pool("kv")
    except KeyError:
        return None


def _require_pool_match(pool: Any, cache: "KVCacheManager", limit: int) -> None:
    problems = []
    if int(pool.max_seq_len) != int(limit):
        problems.append(f"max_seq_len {pool.max_seq_len} != {limit}")
    if int(pool.num_layers) != int(cache.num_layers):
        problems.append(f"num_layers {pool.num_layers} != {cache.num_layers}")
    if int(pool.num_kv_heads) != int(cache.num_kv_heads):
        problems.append(f"num_kv_heads {pool.num_kv_heads} != {cache.num_kv_heads}")
    if int(pool.head_dim) != int(cache.head_dim):
        problems.append(f"head_dim {pool.head_dim} != {cache.head_dim}")
    if int(pool.batch_size) != int(cache.batch_size):
        problems.append(f"batch_size {pool.batch_size} != {cache.batch_size}")
    if problems:
        raise ValueError(
            "IOBindingManager KV pool does not match this cache: " + "; ".join(problems)
        )


def _has_layer_index(name: str, layer_idx: int) -> bool:
    return re.search(rf"(^|[^\d]){int(layer_idx)}([^\d]|$)", name) is not None


def _resolve_kv_name(
    names: List[str],
    layer_idx: int,
    kind: str,
    role: str,
) -> Optional[str]:
    hits = [
        name
        for name in names
        if kind in name.lower() and _has_layer_index(name, layer_idx)
    ]
    if role == "past":
        preferred = [name for name in hits if "past" in name.lower()]
        if not preferred:
            preferred = [name for name in hits if "present" not in name.lower()]
    else:
        preferred = [name for name in hits if "present" in name.lower()]
        if not preferred:
            preferred = [name for name in hits if "past" not in name.lower()]
    if len(preferred) == 1:
        return preferred[0]
    defaults = {
        ("key", "past"): f"past_key_values.{layer_idx}.key",
        ("value", "past"): f"past_key_values.{layer_idx}.value",
        ("key", "present"): f"present.{layer_idx}.key",
        ("value", "present"): f"present.{layer_idx}.value",
    }
    chosen = defaults[(kind, role)]
    if chosen in preferred:
        return chosen
    if len(preferred) > 1:
        raise ValueError(
            f"Multiple {role} {kind} tensors for layer {layer_idx}: {preferred}"
        )
    return None


def _mask_names(names: List[str]) -> List[str]:
    found = []
    for name in names:
        lower = name.lower()
        if "attention_mask" in lower or lower in {"mask", "attn_mask"}:
            found.append(name)
    return found


_SCALAR_INPUTS = {"sequence_length", "seq_len", "current_seq_len", "kv_seq_len"}


def _scalar_names(names: List[str]) -> List[str]:
    return [name for name in names if name.lower() in _SCALAR_INPUTS]


def _node_shape(node: Any) -> Optional[Tuple[Any, ...]]:
    shape = getattr(node, "shape", None)
    if shape is None:
        return None
    return tuple(shape)


def _dim_is_dynamic(dim: Any) -> bool:
    return dim is None or isinstance(dim, str)


def _require_static_shape(node: Any, allocated: Tuple[int, ...], max_seq_len: int) -> None:
    declared = _node_shape(node)
    if declared is None:
        return
    if len(declared) != len(allocated):
        raise ValueError(
            f"Tensor '{node.name}' rank {len(declared)} does not match the "
            f"static KV cache rank {len(allocated)} at max_seq_len {max_seq_len}"
        )
    for want, have in zip(declared, allocated):
        if _dim_is_dynamic(want):
            continue
        if int(want) != int(have):
            raise ValueError(
                f"Tensor '{node.name}' shape {declared} does not match the "
                f"static KV cache {allocated} at max_seq_len {max_seq_len}"
            )


def _require_scalar_shape(node: Any) -> None:
    declared = _node_shape(node)
    if declared is None or len(declared) <= 1:
        return
    raise ValueError(
        f"Tensor '{node.name}' shape {declared} is not a sequence-length scalar"
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


_QUANT_ALIASES = {
    "none": None,
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "int8": "int8",
    "fp8": "fp8",
    "float8": "fp8",
    "e4m3": "fp8",
    "e4m3fn": "fp8",
    "int4": "int4",
    "fp4": "fp4",
    "float4": "fp4",
}

_FP4_MAGNITUDES = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32
)


def _coerce_kv_quantization(scheme: Optional[str]) -> Optional[str]:
    """Map aliases onto fp16, bf16, int8, fp8, int4, or fp4."""
    if scheme is None:
        return None
    text = str(scheme).strip().lower().replace("-", "_")
    if text in _QUANT_ALIASES:
        return _QUANT_ALIASES[text]
    allowed = "fp16, bf16, int8, fp8, int4, fp4"
    raise ValueError(
        f"Unknown kv_quantization '{scheme}'. Expected one of: {allowed}"
    )


def _quantization_uses_scale(scheme: Optional[str]) -> bool:
    return scheme in {"int8", "fp8", "int4", "fp4"}


def _bf16_storage_dtype() -> np.dtype:
    try:
        from .dtypes import get_bfloat16_dtype

        native = get_bfloat16_dtype()
    except Exception:
        native = None
    if native is None:
        return np.dtype(np.uint16)
    return np.dtype(native)


def _fp8_storage_dtype() -> np.dtype:
    from .dtypes import float8_storage_dtype

    return float8_storage_dtype("e4m3fn")


def _scheme_storage_dtype(scheme: str) -> np.dtype:
    if scheme == "fp16":
        return np.dtype(np.float16)
    if scheme == "bf16":
        return _bf16_storage_dtype()
    if scheme == "int8":
        return np.dtype(np.int8)
    if scheme == "fp8":
        return _fp8_storage_dtype()
    if scheme in {"int4", "fp4"}:
        return np.dtype(np.uint8)
    raise ValueError(f"Unknown kv_quantization '{scheme}'")


def _default_compute_dtype(dtype: Any) -> Any:
    """Use the default KV policy when the caller omitted a compute dtype."""
    if dtype is not None:
        return dtype
    from .dtypes import get_kv_cache_dtype

    return get_kv_cache_dtype("default")


def _resolve_cache_dtype(
    scheme: Optional[str], requested: Any, compute: Any
) -> np.dtype:
    """Storage dtype for the compressed copy. Compute dtype stays ``dtype``."""
    compute_dtype = np.dtype(compute)
    if scheme is None:
        if requested is None:
            return compute_dtype
        requested_dtype = np.dtype(requested)
        if requested_dtype != compute_dtype:
            raise ValueError(
                f"cache_dtype {requested_dtype} requires kv_quantization "
                f"when the compute dtype is {compute_dtype}"
            )
        return requested_dtype
    expected = _scheme_storage_dtype(scheme)
    if requested is None:
        return expected
    requested_dtype = np.dtype(requested)
    if requested_dtype != expected:
        raise ValueError(
            f"cache_dtype {requested_dtype} does not match {scheme} storage {expected}"
        )
    return expected


def _as_float32(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float32)


def _finite_float32(array: np.ndarray) -> np.ndarray:
    values = _as_float32(array)
    if values.size == 0:
        return values
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _symmetric_scale(values: np.ndarray, qmax: float) -> float:
    if values.size == 0:
        return 1.0
    peak = float(np.max(np.abs(values)))
    if peak == 0.0 or not np.isfinite(peak):
        return 1.0
    return peak / float(qmax)


def _quantize_bf16(values: np.ndarray) -> np.ndarray:
    from .dtypes import float32_to_bfloat16

    return float32_to_bfloat16(values)


def _dequantize_bf16(stored: np.ndarray) -> np.ndarray:
    from .dtypes import bfloat16_to_float32

    return bfloat16_to_float32(stored)


def _quantize_fp8(values: np.ndarray, scale: float) -> np.ndarray:
    from .dtypes import quantize_float8

    return quantize_float8(_finite_float32(values), scale, kind="e4m3fn")


def _dequantize_fp8(stored: np.ndarray, scale: float) -> np.ndarray:
    from .dtypes import dequantize_float8

    return dequantize_float8(stored, scale)


def _pack_nibbles(nibbles: np.ndarray) -> np.ndarray:
    """Pack two 4-bit codes into each uint8 along the last axis."""
    width = int(nibbles.shape[-1]) if nibbles.ndim else 0
    if width % 2:
        pad = [(0, 0)] * (nibbles.ndim - 1) + [(0, 1)]
        nibbles = np.pad(nibbles, pad, mode="constant")
    lo = nibbles[..., 0::2].astype(np.uint8)
    hi = nibbles[..., 1::2].astype(np.uint8)
    return np.ascontiguousarray(lo | (hi << np.uint8(4)))


def _unpack_nibbles(packed: np.ndarray, head_dim: int) -> np.ndarray:
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    lo = packed & np.uint8(0x0F)
    hi = (packed >> np.uint8(4)) & np.uint8(0x0F)
    restored = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.uint8)
    restored[..., 0::2] = lo
    restored[..., 1::2] = hi
    return restored[..., : int(head_dim)]


def _int_to_nibble(codes: np.ndarray) -> np.ndarray:
    return (codes.astype(np.int16) & np.int16(0xF)).astype(np.uint8)


def _nibble_to_int(nibbles: np.ndarray) -> np.ndarray:
    signed = nibbles.astype(np.int16)
    return np.where(signed >= 8, signed - 16, signed).astype(np.int8)


def _quantize_int4(values: np.ndarray, scale: float) -> np.ndarray:
    scaled = _finite_float32(values) / np.float32(scale)
    codes = np.clip(np.rint(scaled), -7, 7).astype(np.int8)
    return _pack_nibbles(_int_to_nibble(codes))


def _dequantize_int4(stored: np.ndarray, scale: float, head_dim: int) -> np.ndarray:
    codes = _nibble_to_int(_unpack_nibbles(stored, head_dim))
    return codes.astype(np.float32) * np.float32(scale)


def _quantize_fp4(values: np.ndarray, scale: float) -> np.ndarray:
    scaled = _finite_float32(values) / np.float32(scale)
    nearest = np.abs(np.abs(scaled)[..., None] - _FP4_MAGNITUDES).argmin(axis=-1)
    nibbles = nearest.astype(np.uint8)
    nibbles = np.where(scaled < 0, nibbles | np.uint8(0x8), nibbles).astype(np.uint8)
    return _pack_nibbles(nibbles)


def _dequantize_fp4(stored: np.ndarray, scale: float, head_dim: int) -> np.ndarray:
    nibbles = _unpack_nibbles(stored, head_dim)
    magnitudes = _FP4_MAGNITUDES[(nibbles & np.uint8(0x7)).astype(np.intp)]
    signed = np.where(nibbles & np.uint8(0x8), -magnitudes, magnitudes)
    return signed.astype(np.float32) * np.float32(scale)


def _quantize_array(array: np.ndarray, scheme: str) -> Tuple[np.ndarray, Optional[float]]:
    """Compress one tensor. The scale is 1 when the block is empty or all zeros."""
    if scheme == "fp16":
        return np.ascontiguousarray(array, dtype=np.float16), None
    if scheme == "bf16":
        return _quantize_bf16(array), None
    if scheme == "int8":
        values = _finite_float32(array)
        scale = _symmetric_scale(values, 127.0)
        codes = np.clip(np.rint(values / np.float32(scale)), -127, 127).astype(np.int8)
        return np.ascontiguousarray(codes), float(scale)
    if scheme == "fp8":
        values = _finite_float32(array)
        scale = _symmetric_scale(values, 448.0)
        return _quantize_fp8(values, scale), float(scale)
    if scheme == "int4":
        values = _finite_float32(array)
        scale = _symmetric_scale(values, 7.0)
        return _quantize_int4(values, scale), float(scale)
    if scheme == "fp4":
        values = _finite_float32(array)
        scale = _symmetric_scale(values, 6.0)
        return _quantize_fp4(values, scale), float(scale)
    raise ValueError(f"Unknown kv_quantization '{scheme}'")


def _dequantize_array(
    stored: np.ndarray,
    scheme: str,
    scale: Optional[float],
    head_dim: int,
) -> np.ndarray:
    """Expand one compressed tensor back to float32."""
    if scheme == "fp16":
        return np.asarray(stored, dtype=np.float32)
    if scheme == "bf16":
        return _dequantize_bf16(stored)
    factor = 1.0 if scale is None else float(scale)
    if scheme == "int8":
        return np.asarray(stored, dtype=np.float32) * np.float32(factor)
    if scheme == "fp8":
        return _dequantize_fp8(stored, factor)
    if scheme == "int4":
        return _dequantize_int4(stored, factor, head_dim)
    if scheme == "fp4":
        return _dequantize_fp4(stored, factor, head_dim)
    raise ValueError(f"Unknown kv_quantization '{scheme}'")


# =============================================================================
# MODULAR CACHE TYPES
# =============================================================================

_CACHE_TYPE_ALIASES = {
    "dense": "dense_kv",
    "dense_kv": "dense_kv",
    "kv": "dense_kv",
    "attention": "dense_kv",
    "recurrent": "recurrent_state",
    "recurrent_state": "recurrent_state",
    "deltanet": "recurrent_state",
    "gated_deltanet": "recurrent_state",
    "linear": "recurrent_state",
    "linear_attention": "recurrent_state",
    "sparse": "sparse_kv",
    "sparse_kv": "sparse_kv",
    "sparse_attention": "sparse_kv",
    "router": "router",
    "moe": "router",
    "moe_routing": "router",
    "expert": "router",
    "draft": "draft",
    "speculative": "draft",
    "verify": "verify",
    "verification": "verify",
}


def _coerce_cache_type(cache_type: Any) -> str:
    """Map an attention-style name onto a cache type string."""
    text = str(cache_type).strip().lower().replace("-", "_").replace(" ", "_")
    if text in _CACHE_TYPE_ALIASES:
        return _CACHE_TYPE_ALIASES[text]
    allowed = "dense_kv, recurrent_state, sparse_kv, router, draft, verify"
    raise ValueError(
        f"Unknown cache type '{cache_type}'. Expected one of: {allowed}"
    )


class CacheEntry:
    """
    One layer of cache, independent of the attention math that produced it.

    ``tensors`` holds the resident arrays or OrtValues. ``metadata`` holds
    lengths, indices, and anything else a consumer needs to interpret them.
    """

    def __init__(
        self,
        layer_idx: int,
        cache_type: str,
        tensors: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        layer = int(layer_idx)
        if layer < -1:
            raise ValueError(f"layer_idx must be >= -1, got {layer}")
        self.layer_idx = layer
        self.cache_type = _coerce_cache_type(cache_type)
        self.tensors: Dict[str, Any] = {} if tensors is None else dict(tensors)
        self.metadata: Dict[str, Any] = {} if metadata is None else dict(metadata)

    def __repr__(self) -> str:
        names = ", ".join(self.tensors)
        return (
            f"CacheEntry(layer={self.layer_idx}, type={self.cache_type}, "
            f"tensors=[{names}])"
        )


class DenseKVCache(KVCacheManager):
    """
    Standard key/value cache for dense attention.

    This is ``KVCacheManager`` with ``cache_type`` ``dense_kv``. Policies,
    static OrtValues, and quantization behave the same way.
    """

    cache_type = "dense_kv"


class RecurrentStateCache:
    """
    Fixed recurrent state for Gated DeltaNet and linear attention.

    The state is ``[batch, heads, head_dim, state_dim]`` and is updated in
    place. ``recurrent_step`` applies ``state = decay * state + update``.
    An optional convolution window of length ``conv_kernel - 1`` shifts
    when ``push_conv`` receives the next frame.
    """

    cache_type = "recurrent_state"

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: Any = None,
        batch_size: int = 1,
        state_dim: Optional[int] = None,
        conv_dim: Optional[int] = None,
        conv_kernel: Optional[int] = None,
        backend: Any = KVCacheBackend.CPU,
        device_id: int = 0,
    ):
        dtype = _default_compute_dtype(dtype)
        if num_layers < 1:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_heads < 1 or head_dim < 1:
            raise ValueError(
                f"num_heads and head_dim must be positive, got {num_heads}, {head_dim}"
            )
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.state_dim = int(head_dim if state_dim is None else state_dim)
        if self.state_dim < 1:
            raise ValueError(f"state_dim must be positive, got {self.state_dim}")
        self.dtype = dtype
        self.batch_size = int(batch_size)
        self.backend = coerce_kv_backend(backend)
        self.device_id = int(device_id)
        self.conv_dim = None if conv_dim is None else int(conv_dim)
        self.conv_kernel = None if conv_kernel is None else int(conv_kernel)
        if self.conv_kernel is not None and self.conv_kernel < 2:
            raise ValueError(
                f"conv_kernel must be at least 2 so the cache has a window, "
                f"got {self.conv_kernel}"
            )
        if (self.conv_dim is None) != (self.conv_kernel is None):
            raise ValueError("conv_dim and conv_kernel must be set together")
        state_shape = (
            self.batch_size,
            self.num_heads,
            self.head_dim,
            self.state_dim,
        )
        self._state = [
            np.zeros(state_shape, dtype=self.dtype) for _ in range(self.num_layers)
        ]
        self._conv: List[Optional[np.ndarray]] = [None] * self.num_layers
        if self.conv_dim is not None:
            conv_shape = (
                self.batch_size,
                self.conv_dim,
                self.conv_kernel - 1,
            )
            self._conv = [
                np.zeros(conv_shape, dtype=self.dtype) for _ in range(self.num_layers)
            ]

    def _check_layer(self, layer_idx: int) -> int:
        layer = int(layer_idx)
        if layer < 0 or layer >= self.num_layers:
            raise KeyError(
                f"Layer {layer} not in cache (have {self.num_layers} layers)"
            )
        return layer

    def reset(self) -> None:
        """Zero every recurrent state and convolution window."""
        for layer in range(self.num_layers):
            self._state[layer].fill(0)
            conv = self._conv[layer]
            if conv is not None:
                conv.fill(0)

    def get_state(self, layer_idx: int) -> np.ndarray:
        """Resident recurrent state for one layer. The array is not copied."""
        return self._state[self._check_layer(layer_idx)]

    def get_conv_state(self, layer_idx: int) -> np.ndarray:
        """Short-convolution window for one layer."""
        layer = self._check_layer(layer_idx)
        conv = self._conv[layer]
        if conv is None:
            raise ValueError(f"Layer {layer} has no convolution cache")
        return conv

    def update_state(
        self,
        layer_idx: int,
        state: np.ndarray,
        conv_state: Optional[np.ndarray] = None,
    ) -> None:
        """Replace one layer's state in the preallocated buffer."""
        layer = self._check_layer(layer_idx)
        incoming = np.asarray(state, dtype=self.dtype)
        slot = self._state[layer]
        if incoming.shape != slot.shape:
            raise ValueError(
                f"Layer {layer}: recurrent state shape {incoming.shape} "
                f"does not match {slot.shape}"
            )
        slot[...] = incoming
        if conv_state is None:
            return
        conv = self._conv[layer]
        if conv is None:
            raise ValueError(f"Layer {layer} has no convolution cache")
        incoming_conv = np.asarray(conv_state, dtype=self.dtype)
        if incoming_conv.shape != conv.shape:
            raise ValueError(
                f"Layer {layer}: conv state shape {incoming_conv.shape} "
                f"does not match {conv.shape}"
            )
        conv[...] = incoming_conv

    def recurrent_step(
        self,
        layer_idx: int,
        decay: Any,
        update: np.ndarray,
    ) -> np.ndarray:
        """
        Carry the state forward: ``state = decay * state + update``.

        ``decay`` and ``update`` broadcast onto the resident state. The
        same array is returned.
        """
        layer = self._check_layer(layer_idx)
        slot = self._state[layer]
        incoming = np.asarray(update, dtype=self.dtype)
        try:
            stepped = slot * np.asarray(decay, dtype=self.dtype) + incoming
        except ValueError as exc:
            raise ValueError(
                f"Layer {layer}: cannot apply decay and update to state {slot.shape}"
            ) from exc
        if stepped.shape != slot.shape:
            raise ValueError(
                f"Layer {layer}: recurrent update produced {stepped.shape}, "
                f"expected {slot.shape}"
            )
        slot[...] = stepped
        return slot

    def push_conv(self, layer_idx: int, features: np.ndarray) -> np.ndarray:
        """
        Shift the convolution window left and append one frame.

        ``features`` is ``[batch, conv_dim]`` or ``[conv_dim]``.
        """
        layer = self._check_layer(layer_idx)
        conv = self._conv[layer]
        if conv is None:
            raise ValueError(f"Layer {layer} has no convolution cache")
        frame = np.asarray(features, dtype=self.dtype)
        if frame.ndim == 1:
            frame = np.broadcast_to(frame, (self.batch_size, self.conv_dim)).copy()
        if frame.shape != (self.batch_size, self.conv_dim):
            raise ValueError(
                f"Layer {layer}: conv frame shape {frame.shape} does not match "
                f"({self.batch_size}, {self.conv_dim})"
            )
        if conv.shape[-1] == 1:
            conv[..., 0] = frame
        else:
            conv[..., :-1] = np.array(conv[..., 1:], copy=True)
            conv[..., -1] = frame
        return conv

    def entry(self, layer_idx: int) -> CacheEntry:
        layer = self._check_layer(layer_idx)
        tensors: Dict[str, Any] = {"state": self._state[layer]}
        if self._conv[layer] is not None:
            tensors["conv_state"] = self._conv[layer]
        return CacheEntry(
            layer,
            self.cache_type,
            tensors,
            {
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "state_dim": self.state_dim,
                "conv_dim": self.conv_dim,
                "conv_kernel": self.conv_kernel,
                "backend": self.backend.value,
                "batch_size": self.batch_size,
            },
        )

    def entries(self) -> List[CacheEntry]:
        return [self.entry(layer) for layer in range(self.num_layers)]

    def get_memory_usage(self) -> Dict[str, Any]:
        total = 0
        per_layer = []
        for layer in range(self.num_layers):
            nbytes = int(self._state[layer].nbytes)
            conv = self._conv[layer]
            if conv is not None:
                nbytes += int(conv.nbytes)
            per_layer.append(nbytes)
            total += nbytes
        return {
            "total_bytes": total,
            "total_mb": total / (1024 * 1024),
            "per_layer_bytes": per_layer,
            "num_layers": self.num_layers,
            "dtype": str(np.dtype(self.dtype)),
            "cache_type": self.cache_type,
            "backend": self.backend.value,
        }


class SparseKVCache:
    """
    Block and index cache for sparse attention.

    Keys and values live in ``[batch, heads, max_blocks, block_size, head_dim]``.
    ``block_indices`` records which logical block occupies each physical slot
    (``-1`` is empty). ``token_indices`` records the selected positions for
    the next sparse attention call.
    """

    cache_type = "sparse_kv"

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: Any = None,
        batch_size: int = 1,
        max_blocks: Optional[int] = None,
        block_size: int = 16,
        max_seq_len: Optional[int] = None,
        backend: Any = KVCacheBackend.CPU,
        device_id: int = 0,
    ):
        dtype = _default_compute_dtype(dtype)
        if num_layers < 1 or num_kv_heads < 1 or head_dim < 1:
            raise ValueError("num_layers, num_kv_heads, and head_dim must be positive")
        self.block_size = int(block_size)
        if self.block_size < 1:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if max_blocks is None:
            if max_seq_len is None:
                raise ValueError("sparse cache needs max_blocks or max_seq_len")
            limit = int(max_seq_len)
            if limit < 1:
                raise ValueError(f"max_seq_len must be positive, got {limit}")
            max_blocks = (limit + self.block_size - 1) // self.block_size
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.dtype = dtype
        self.batch_size = int(batch_size)
        self.max_blocks = int(max_blocks)
        if self.max_blocks < 1:
            raise ValueError(f"max_blocks must be positive, got {self.max_blocks}")
        self.max_seq_len = self.max_blocks * self.block_size
        self.backend = coerce_kv_backend(backend)
        self.device_id = int(device_id)
        shape = (
            self.batch_size,
            self.num_kv_heads,
            self.max_blocks,
            self.block_size,
            self.head_dim,
        )
        self._key = [np.zeros(shape, dtype=self.dtype) for _ in range(self.num_layers)]
        self._value = [np.zeros(shape, dtype=self.dtype) for _ in range(self.num_layers)]
        self._block_indices = [
            np.full((self.batch_size, self.max_blocks), -1, dtype=np.int32)
            for _ in range(self.num_layers)
        ]
        capacity = self.max_seq_len
        self._token_indices = [
            np.full((self.batch_size, capacity), -1, dtype=np.int32)
            for _ in range(self.num_layers)
        ]
        self._selected = [0] * self.num_layers

    def _check_layer(self, layer_idx: int) -> int:
        layer = int(layer_idx)
        if layer < 0 or layer >= self.num_layers:
            raise KeyError(
                f"Layer {layer} not in cache (have {self.num_layers} layers)"
            )
        return layer

    def reset(self) -> None:
        """Clear blocks, indices, and the selection list."""
        for layer in range(self.num_layers):
            self._key[layer].fill(0)
            self._value[layer].fill(0)
            self._block_indices[layer].fill(-1)
            self._token_indices[layer].fill(-1)
            self._selected[layer] = 0

    def block_indices(self, layer_idx: int) -> np.ndarray:
        return self._block_indices[self._check_layer(layer_idx)]

    def token_indices(self, layer_idx: int) -> np.ndarray:
        return self._token_indices[self._check_layer(layer_idx)]

    def write_blocks(
        self,
        layer_idx: int,
        key: np.ndarray,
        value: np.ndarray,
        block_ids: np.ndarray,
        logical_ids: Optional[np.ndarray] = None,
    ) -> None:
        """
        Scatter blocks into physical slots.

        ``key`` and ``value`` have shape
        ``[batch, heads, count, block_size, head_dim]``. ``block_ids[i] < 0``
        skips that block. ``logical_ids`` is what ``block_indices`` stores;
        it defaults to the physical slot.
        """
        layer = self._check_layer(layer_idx)
        keys = np.asarray(key, dtype=self.dtype)
        values = np.asarray(value, dtype=self.dtype)
        if keys.shape != values.shape:
            raise ValueError(
                f"Layer {layer}: sparse key shape {keys.shape} != value shape {values.shape}"
            )
        if keys.ndim != 5:
            raise ValueError(
                f"Layer {layer}: sparse blocks must be "
                f"[batch, heads, count, block_size, head_dim], got {keys.shape}"
            )
        ids = np.asarray(block_ids, dtype=np.int32).reshape(-1)
        if keys.shape[2] != ids.shape[0]:
            raise ValueError(
                f"Layer {layer}: {keys.shape[2]} blocks and {ids.shape[0]} block ids"
            )
        labels = ids if logical_ids is None else np.asarray(logical_ids, dtype=np.int32).reshape(-1)
        if labels.shape[0] != ids.shape[0]:
            raise ValueError(
                f"Layer {layer}: logical id count {labels.shape[0]} != {ids.shape[0]}"
            )
        slot_key = self._key[layer]
        slot_value = self._value[layer]
        table = self._block_indices[layer]
        for index, slot in enumerate(ids.tolist()):
            if slot < 0:
                continue
            if slot >= self.max_blocks:
                raise ValueError(
                    f"Layer {layer}: block id {slot} is outside max_blocks {self.max_blocks}"
                )
            slot_key[:, :, slot] = keys[:, :, index]
            slot_value[:, :, slot] = values[:, :, index]
            table[:, slot] = int(labels[index])

    def set_indices(self, layer_idx: int, indices: np.ndarray) -> None:
        """Record the selected token positions. The unused tail stays ``-1``."""
        layer = self._check_layer(layer_idx)
        selected = np.asarray(indices, dtype=np.int32)
        if selected.ndim == 1:
            selected = np.broadcast_to(
                selected, (self.batch_size, selected.shape[0])
            ).copy()
        if selected.ndim != 2 or selected.shape[0] != self.batch_size:
            raise ValueError(
                f"Layer {layer}: token indices shape {selected.shape} does not match "
                f"batch {self.batch_size}"
            )
        count = int(selected.shape[1])
        capacity = int(self._token_indices[layer].shape[1])
        if count > capacity:
            raise ValueError(
                f"Layer {layer}: {count} token indices exceed capacity {capacity}"
            )
        if np.any(selected < -1):
            raise ValueError(f"Layer {layer}: token indices must be >= -1")
        table = self._token_indices[layer]
        table[:, :count] = selected
        table[:, count:] = -1
        self._selected[layer] = count

    def update(
        self,
        layer_idx: int,
        key: np.ndarray,
        value: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        """
        Pack a selected token prefix into the block buffer and store its indices.

        ``key`` and ``value`` are ``[batch, heads, selected, head_dim]``.
        The tail of the buffer is cleared, so this replaces a previous
        ``write_blocks`` fill.
        """
        layer = self._check_layer(layer_idx)
        keys = np.asarray(key, dtype=self.dtype)
        values = np.asarray(value, dtype=self.dtype)
        if keys.shape != values.shape or keys.ndim != 4:
            raise ValueError(
                f"Layer {layer}: sparse update expects matching "
                f"[batch, heads, selected, head_dim] tensors, got {keys.shape}"
            )
        flat_key = self._key[layer].reshape(
            self.batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim
        )
        flat_value = self._value[layer].reshape(flat_key.shape)
        count = int(keys.shape[2])
        if count > self.max_seq_len:
            raise ValueError(
                f"Layer {layer}: {count} selected tokens exceed capacity {self.max_seq_len}"
            )
        incoming_indices = np.asarray(indices)
        if incoming_indices.ndim == 1:
            index_count = int(incoming_indices.shape[0])
        elif (
            incoming_indices.ndim == 2
            and incoming_indices.shape[0] == self.batch_size
        ):
            index_count = int(incoming_indices.shape[1])
        else:
            index_count = -1
        if index_count != count:
            raise ValueError(
                f"Layer {layer}: {count} selected tokens and {index_count} indices"
            )
        flat_key[:, :, :count] = keys
        flat_key[:, :, count:] = 0
        flat_value[:, :, :count] = values
        flat_value[:, :, count:] = 0
        self._block_indices[layer].fill(-1)
        occupied = (count + self.block_size - 1) // self.block_size if count else 0
        if occupied:
            self._block_indices[layer][:, :occupied] = np.arange(occupied, dtype=np.int32)
        self.set_indices(layer, indices)

    def entry(self, layer_idx: int) -> CacheEntry:
        layer = self._check_layer(layer_idx)
        return CacheEntry(
            layer,
            self.cache_type,
            {
                "key": self._key[layer],
                "value": self._value[layer],
                "block_indices": self._block_indices[layer],
                "token_indices": self._token_indices[layer],
            },
            {
                "max_blocks": self.max_blocks,
                "block_size": self.block_size,
                "selected": self._selected[layer],
                "backend": self.backend.value,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
            },
        )

    def entries(self) -> List[CacheEntry]:
        return [self.entry(layer) for layer in range(self.num_layers)]

    def get_memory_usage(self) -> Dict[str, Any]:
        total = 0
        per_layer = []
        for layer in range(self.num_layers):
            nbytes = (
                int(self._key[layer].nbytes)
                + int(self._value[layer].nbytes)
                + int(self._block_indices[layer].nbytes)
                + int(self._token_indices[layer].nbytes)
            )
            per_layer.append(nbytes)
            total += nbytes
        return {
            "total_bytes": total,
            "total_mb": total / (1024 * 1024),
            "per_layer_bytes": per_layer,
            "num_layers": self.num_layers,
            "dtype": str(np.dtype(self.dtype)),
            "cache_type": self.cache_type,
            "backend": self.backend.value,
            "max_blocks": self.max_blocks,
        }


class RouterCache:
    """
    Ring of MoE expert choices.

    ``expert_ids`` is ``[batch, history, top_k]`` with ``-1`` in unused
    slots. ``record`` advances the ring and drops the oldest choice once
    ``history`` is full.
    """

    cache_type = "router"

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        top_k: int = 1,
        history: int = 16,
        batch_size: int = 1,
    ):
        if num_layers < 1:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_experts < 1:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if top_k < 1 or top_k > num_experts:
            raise ValueError(
                f"top_k must be in 1..{num_experts}, got {top_k}"
            )
        if history < 1:
            raise ValueError(f"history must be positive, got {history}")
        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.history = int(history)
        self.batch_size = int(batch_size)
        self._expert_ids = [
            np.full((self.batch_size, self.history, self.top_k), -1, dtype=np.int32)
            for _ in range(self.num_layers)
        ]
        self._weights = [
            np.zeros((self.batch_size, self.history, self.top_k), dtype=np.float32)
            for _ in range(self.num_layers)
        ]
        self._cursor = [0] * self.num_layers
        self._count = [0] * self.num_layers

    def _check_layer(self, layer_idx: int) -> int:
        layer = int(layer_idx)
        if layer < 0 or layer >= self.num_layers:
            raise KeyError(
                f"Layer {layer} not in cache (have {self.num_layers} layers)"
            )
        return layer

    def reset(self) -> None:
        for layer in range(self.num_layers):
            self._expert_ids[layer].fill(-1)
            self._weights[layer].fill(0)
            self._cursor[layer] = 0
            self._count[layer] = 0

    def record(
        self,
        layer_idx: int,
        expert_ids: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        """Append one routing decision. The oldest decision falls out of the ring."""
        layer = self._check_layer(layer_idx)
        ids = np.asarray(expert_ids, dtype=np.int32)
        if ids.ndim == 1:
            ids = np.broadcast_to(ids, (self.batch_size, self.top_k)).copy()
        if ids.shape != (self.batch_size, self.top_k):
            raise ValueError(
                f"Layer {layer}: expert ids shape {ids.shape} does not match "
                f"({self.batch_size}, {self.top_k})"
            )
        if np.any(ids < 0) or np.any(ids >= self.num_experts):
            raise ValueError(
                f"Layer {layer}: expert ids must be in 0..{self.num_experts - 1}"
            )
        if weights is None:
            chosen = np.full(ids.shape, 1.0 / self.top_k, dtype=np.float32)
        else:
            chosen = np.asarray(weights, dtype=np.float32)
            if chosen.ndim == 1:
                chosen = np.broadcast_to(chosen, ids.shape).copy()
            if chosen.shape != ids.shape:
                raise ValueError(
                    f"Layer {layer}: router weights shape {chosen.shape} "
                    f"does not match {ids.shape}"
                )
        slot = self._cursor[layer] % self.history
        self._expert_ids[layer][:, slot, :] = ids
        self._weights[layer][:, slot, :] = chosen
        self._cursor[layer] += 1
        self._count[layer] = min(self.history, self._count[layer] + 1)

    def recent(self, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Expert ids and weights from oldest to newest."""
        layer = self._check_layer(layer_idx)
        count = self._count[layer]
        ids = self._expert_ids[layer]
        weights = self._weights[layer]
        if count == 0:
            empty_ids = ids[:, :0, :]
            return empty_ids, weights[:, :0, :]
        if self._cursor[layer] <= self.history:
            return ids[:, :count, :].copy(), weights[:, :count, :].copy()
        start = self._cursor[layer] % self.history
        ordered_ids = np.concatenate([ids[:, start:, :], ids[:, :start, :]], axis=1)
        ordered_weights = np.concatenate(
            [weights[:, start:, :], weights[:, :start, :]], axis=1
        )
        return ordered_ids, ordered_weights

    def entry(self, layer_idx: int) -> CacheEntry:
        layer = self._check_layer(layer_idx)
        return CacheEntry(
            layer,
            self.cache_type,
            {
                "expert_ids": self._expert_ids[layer],
                "weights": self._weights[layer],
            },
            {
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "history": self.history,
                "filled": self._count[layer],
                "cursor": self._cursor[layer],
            },
        )

    def entries(self) -> List[CacheEntry]:
        return [self.entry(layer) for layer in range(self.num_layers)]

    def get_memory_usage(self) -> Dict[str, Any]:
        total = 0
        per_layer = []
        for layer in range(self.num_layers):
            nbytes = int(self._expert_ids[layer].nbytes) + int(self._weights[layer].nbytes)
            per_layer.append(nbytes)
            total += nbytes
        return {
            "total_bytes": total,
            "total_mb": total / (1024 * 1024),
            "per_layer_bytes": per_layer,
            "num_layers": self.num_layers,
            "cache_type": self.cache_type,
            "num_experts": self.num_experts,
            "history": self.history,
        }


def _layer_spec(item: Any) -> Tuple[str, Dict[str, Any]]:
    if isinstance(item, str):
        return _coerce_cache_type(item), {}
    if isinstance(item, dict):
        if "type" not in item and "cache_type" not in item:
            raise ValueError("layer spec needs a type or cache_type")
        kind = _coerce_cache_type(item.get("type", item.get("cache_type")))
        options = {
            key: value
            for key, value in item.items()
            if key not in {"type", "cache_type"}
        }
        return kind, options
    raise TypeError(
        f"Layer spec must be a cache type name or a dict, got {type(item).__name__}"
    )


def _take_options(options: Dict[str, Any], allowed: Sequence[str], kind: str) -> Dict[str, Any]:
    unknown = sorted(key for key in options if key not in allowed)
    if unknown:
        raise ValueError(
            f"{kind} layer received unknown options: {', '.join(unknown)}"
        )
    return options


class HybridKVCacheManager:
    """
    One cache object per layer for a hybrid model.

    ``layer_types`` names the cache for each layer: dense attention,
    Gated DeltaNet, sparse attention, or MoE routing. ``draft`` and
    ``verify`` hold the speculative pair. They can be an existing cache,
    a layer-type list, or a dense layer count.
    """

    def __init__(
        self,
        layer_types: Sequence[Any],
        num_kv_heads: int,
        head_dim: int,
        dtype: Any = None,
        batch_size: int = 1,
        backend: Any = KVCacheBackend.CPU,
        max_seq_len: Optional[int] = None,
        device_id: int = 0,
        block_size: int = 16,
        state_dim: Optional[int] = None,
        conv_dim: Optional[int] = None,
        conv_kernel: Optional[int] = None,
        num_experts: Optional[int] = None,
        top_k: int = 1,
        history: int = 16,
        draft: Any = None,
        verify: Any = None,
    ):
        dtype = _default_compute_dtype(dtype)
        if not isinstance(layer_types, (list, tuple)) or len(layer_types) < 1:
            raise ValueError("layer_types must list at least one layer")
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.dtype = dtype
        self.batch_size = int(batch_size)
        self.backend = coerce_kv_backend(backend)
        self.max_seq_len = None if max_seq_len is None else int(max_seq_len)
        self.device_id = int(device_id)
        self.block_size = int(block_size)
        self._common = {
            "num_kv_heads": self.num_kv_heads,
            "num_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "backend": self.backend,
            "max_seq_len": self.max_seq_len,
            "device_id": self.device_id,
            "block_size": self.block_size,
            "state_dim": state_dim,
            "conv_dim": conv_dim,
            "conv_kernel": conv_kernel,
            "num_experts": num_experts,
            "top_k": int(top_k),
            "history": int(history),
        }
        self._layers = [
            self._build_layer(*_layer_spec(item)) for item in layer_types
        ]
        self.draft_cache = self._assign_role(self._build_role(draft, "draft"), "draft")
        self.verify_cache = self._assign_role(self._build_role(verify, "verify"), "verify")
        self.role = "target"
        self.phase = None
        self.execution_phase = None

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    def _build_layer(self, kind: str, options: Dict[str, Any]) -> Any:
        merged = dict(self._common)
        merged.update(options)
        if kind == "dense_kv":
            allowed = (
                "num_kv_heads", "head_dim", "dtype", "batch_size", "backend",
                "max_seq_len", "device_id", "policy", "window_size", "block_size",
                "sink_size", "retrieval_blocks", "cache_dtype", "kv_quantization",
            )
            chosen = _take_options(options, allowed, kind)
            body = {key: merged[key] for key in allowed if key in merged and merged[key] is not None}
            body.update(chosen)
            return DenseKVCache(num_layers=1, **body)
        if kind == "recurrent_state":
            allowed = (
                "num_heads", "head_dim", "dtype", "batch_size", "state_dim",
                "conv_dim", "conv_kernel", "backend", "device_id",
            )
            chosen = _take_options(options, allowed, kind)
            return RecurrentStateCache(
                num_layers=1,
                num_heads=int(chosen.get("num_heads", merged["num_heads"])),
                head_dim=int(chosen.get("head_dim", merged["head_dim"])),
                dtype=chosen.get("dtype", merged["dtype"]),
                batch_size=int(chosen.get("batch_size", merged["batch_size"])),
                state_dim=chosen.get("state_dim", merged["state_dim"]),
                conv_dim=chosen.get("conv_dim", merged["conv_dim"]),
                conv_kernel=chosen.get("conv_kernel", merged["conv_kernel"]),
                backend=chosen.get("backend", merged["backend"]),
                device_id=int(chosen.get("device_id", merged["device_id"])),
            )
        if kind == "sparse_kv":
            allowed = (
                "num_kv_heads", "head_dim", "dtype", "batch_size", "max_blocks",
                "block_size", "max_seq_len", "backend", "device_id",
            )
            chosen = _take_options(options, allowed, kind)
            return SparseKVCache(
                num_layers=1,
                num_kv_heads=int(chosen.get("num_kv_heads", merged["num_kv_heads"])),
                head_dim=int(chosen.get("head_dim", merged["head_dim"])),
                dtype=chosen.get("dtype", merged["dtype"]),
                batch_size=int(chosen.get("batch_size", merged["batch_size"])),
                max_blocks=chosen.get("max_blocks"),
                block_size=int(chosen.get("block_size", merged["block_size"])),
                max_seq_len=chosen.get("max_seq_len", merged["max_seq_len"]),
                backend=chosen.get("backend", merged["backend"]),
                device_id=int(chosen.get("device_id", merged["device_id"])),
            )
        if kind == "router":
            allowed = ("num_experts", "top_k", "history", "batch_size")
            chosen = _take_options(options, allowed, kind)
            experts = chosen.get("num_experts", merged["num_experts"])
            if experts is None:
                raise ValueError("router cache needs num_experts")
            return RouterCache(
                num_layers=1,
                num_experts=int(experts),
                top_k=int(chosen.get("top_k", merged["top_k"])),
                history=int(chosen.get("history", merged["history"])),
                batch_size=int(chosen.get("batch_size", merged["batch_size"])),
            )
        raise ValueError(
            f"Cache type '{kind}' cannot be a model layer. "
            "Use dense_kv, recurrent_state, sparse_kv, or router."
        )

    def _build_role(self, spec: Any, role: str) -> Any:
        if spec is None:
            return None
        if hasattr(spec, "entry") and hasattr(spec, "reset"):
            return spec
        if isinstance(spec, int):
            if spec < 1:
                raise ValueError(f"{role} layer count must be positive, got {spec}")
            return DenseKVCache(
                num_layers=spec,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                dtype=self.dtype,
                batch_size=self.batch_size,
                backend=self.backend,
                max_seq_len=self.max_seq_len,
                device_id=self.device_id,
                block_size=self.block_size,
            )
        if isinstance(spec, (list, tuple)):
            return HybridKVCacheManager(
                layer_types=spec,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                dtype=self.dtype,
                batch_size=self.batch_size,
                backend=self.backend,
                max_seq_len=self.max_seq_len,
                device_id=self.device_id,
                block_size=self.block_size,
                state_dim=self._common["state_dim"],
                conv_dim=self._common["conv_dim"],
                conv_kernel=self._common["conv_kernel"],
                num_experts=self._common["num_experts"],
                top_k=self._common["top_k"],
                history=self._common["history"],
            )
        raise TypeError(
            f"{role} cache must be a layer count, a layer-type list, or a cache object"
        )

    @staticmethod
    def _assign_role(cache: Any, role: str) -> Any:
        if cache is not None and hasattr(cache, "role"):
            cache.role = role
        return cache

    def _child(self, layer_idx: int) -> Any:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise KeyError(
                f"Layer {layer_idx} not in cache (have {self.num_layers} layers)"
            )
        return self._layers[layer_idx]

    def cache_for(self, layer_idx: int) -> Any:
        """The cache object that owns one model layer."""
        return self._child(layer_idx)

    def cache_type_for(self, layer_idx: int) -> str:
        return self._child(layer_idx).cache_type

    def update(self, layer_idx: int, **payload: Any) -> None:
        """Write one layer. The accepted fields depend on its cache type."""
        child = self._child(layer_idx)
        kind = child.cache_type
        if kind == "dense_kv":
            child.update_from_present(0, payload["key"], payload["value"])
            return
        if kind == "recurrent_state":
            if "state" in payload:
                child.update_state(0, payload["state"], payload.get("conv_state"))
                return
            if "update" in payload:
                child.recurrent_step(0, payload.get("decay", 1), payload["update"])
                if "conv_token" in payload:
                    child.push_conv(0, payload["conv_token"])
                return
            if "conv_token" in payload:
                child.push_conv(0, payload["conv_token"])
                return
            raise TypeError(
                f"Layer {layer_idx}: recurrent cache needs state, update, or conv_token"
            )
        if kind == "sparse_kv":
            if "block_ids" in payload:
                child.write_blocks(
                    0,
                    payload["key"],
                    payload["value"],
                    payload["block_ids"],
                    payload.get("logical_ids"),
                )
                return
            child.update(0, payload["key"], payload["value"], payload["indices"])
            return
        if kind == "router":
            child.record(0, payload["expert_ids"], payload.get("weights"))
            return
        raise TypeError(f"Layer {layer_idx}: unsupported cache type {kind}")

    def entry(self, layer_idx: int) -> CacheEntry:
        inner = self._child(layer_idx).entry(0)
        metadata = dict(inner.metadata)
        metadata["model_layer"] = int(layer_idx)
        return CacheEntry(layer_idx, inner.cache_type, inner.tensors, metadata)

    def entries(self) -> List[CacheEntry]:
        return [self.entry(layer) for layer in range(self.num_layers)]

    def _role_cache(self, role: str) -> Any:
        kind = _coerce_cache_type(role)
        if kind == "draft":
            child = self.draft_cache
        elif kind == "verify":
            child = self.verify_cache
        else:
            raise ValueError(f"Speculative role must be draft or verify, got '{role}'")
        if child is None:
            raise ValueError(f"No {kind} cache is attached")
        return child

    def role_entry(self, role: str) -> CacheEntry:
        """A single entry describing the draft or verify cache."""
        kind = _coerce_cache_type(role)
        child = self._role_cache(kind)
        return CacheEntry(
            -1,
            kind,
            {},
            {
                "role": kind,
                "num_layers": int(getattr(child, "num_layers", 1)),
                "child_cache_type": getattr(child, "cache_type", kind),
            },
        )

    def role_entries(self, role: str) -> List[CacheEntry]:
        """Per-layer entries of the draft or verify cache, tagged with that role."""
        kind = _coerce_cache_type(role)
        child = self._role_cache(kind)
        if isinstance(child, HybridKVCacheManager):
            found = child.entries()
        else:
            found = child.entries()
        for item in found:
            item.metadata["role"] = kind
        return found

    def reset(self) -> None:
        """Clear every layer and both speculative caches."""
        for child in self._layers:
            child.reset()
        if self.draft_cache is not None:
            self.draft_cache.reset()
        if self.verify_cache is not None:
            self.verify_cache.reset()

    def get_memory_usage(self) -> Dict[str, Any]:
        per_layer = []
        total = 0
        types = []
        for child in self._layers:
            usage = child.get_memory_usage()
            layer_bytes = int(usage["total_bytes"])
            per_layer.append(layer_bytes)
            total += layer_bytes
            types.append(child.cache_type)
        draft_bytes = 0
        verify_bytes = 0
        if self.draft_cache is not None:
            draft_bytes = int(self.draft_cache.get_memory_usage()["total_bytes"])
            total += draft_bytes
        if self.verify_cache is not None:
            verify_bytes = int(self.verify_cache.get_memory_usage()["total_bytes"])
            total += verify_bytes
        return {
            "total_bytes": total,
            "total_mb": total / (1024 * 1024),
            "per_layer_bytes": per_layer,
            "num_layers": self.num_layers,
            "cache_types": types,
            "draft_bytes": draft_bytes,
            "verify_bytes": verify_bytes,
            "dtype": str(np.dtype(self.dtype)),
            "backend": self.backend.value,
        }

    def __repr__(self) -> str:
        kinds = ",".join(child.cache_type for child in self._layers)
        return f"HybridKVCacheManager(layers={self.num_layers}, types=[{kinds}])"


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

__all__ = [
    "KVCacheManager",
    "KVCacheBackend",
    "KVCachePolicy",
    "coerce_kv_backend",
    "coerce_kv_policy",
    "CacheEntry",
    "DenseKVCache",
    "RecurrentStateCache",
    "SparseKVCache",
    "RouterCache",
    "HybridKVCacheManager",
]
