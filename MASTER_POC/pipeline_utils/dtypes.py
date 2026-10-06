#!/usr/bin/env python3
"""
dtypes.py - Comprehensive ONNX Data Type Handling Module

This module provides complete ONNX type mappings and safe type conversion
utilities for use across inference pipelines.

Usage:
    from pipeline_utils.dtypes import get_numpy_dtype, safe_cast, ONNX_DTYPE_MAP
"""

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np

# =============================================================================
# BFLOAT16 SUPPORT DETECTION
# =============================================================================

_BFLOAT16_DTYPE = None

def supports_bfloat16() -> bool:
    """Check if ml_dtypes bfloat16 is available."""
    global _BFLOAT16_DTYPE
    if _BFLOAT16_DTYPE is not None:
        return _BFLOAT16_DTYPE is not False
    try:
        import ml_dtypes
        _BFLOAT16_DTYPE = ml_dtypes.bfloat16
        return True
    except ImportError:
        _BFLOAT16_DTYPE = False
        return False


def get_bfloat16_dtype():
    """Get bfloat16 dtype if available, else None."""
    if supports_bfloat16():
        return _BFLOAT16_DTYPE
    return None


# =============================================================================
# ONNX TYPE MAPPINGS
# =============================================================================

# Complete ONNX TensorProto type string to numpy dtype mapping
# Covers all standard ONNX types including emerging quantized/low-precision types
ONNX_DTYPE_MAP = {
    # Standard Floating Point Types
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(bfloat16)": "bfloat16",  # Special handling required
    
    # Integer Types (all standard sizes)
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64,
    
    # Boolean
    "tensor(bool)": np.bool_,
    
    # Complex Types
    "tensor(complex64)": np.complex64,
    "tensor(complex128)": np.complex128,
    
    # String (special handling - stored as Python objects)
    "tensor(string)": object,
    
    # Quantized Types (INT4/UINT4 - packed in int8/uint8)
    "tensor(int4)": np.int8,      # Packed 2x per byte
    "tensor(uint4)": np.uint8,    # Packed 2x per byte
    
    # Float8 Types (emerging standard for inference)
    "tensor(float8e4m3fn)": np.float16,   # Fallback with precision warning
    "tensor(float8e5m2)": np.float16,     # Fallback with precision warning
    "tensor(float8e4m3fnuz)": np.float16, # Fallback with precision warning
    "tensor(float8e5m2fnuz)": np.float16, # Fallback with precision warning
}

# Types that require special handling or have precision implications
SPECIAL_TYPES = {
    "bfloat16": "Uses ml_dtypes if available, else falls back to float32",
    "int4": "Signed 4-bit values, two per int8 byte. INT4_BLOCKWISE records the group, scale, and zero point.",
    "uint4": "Unsigned 4-bit values, two per uint8 byte. UINT4_BLOCKWISE records the group, scale, and zero point.",
    "float8e4m3fn": "8-bit float (E4M3), falls back to float16",
    "float8e5m2": "8-bit float (E5M2), falls back to float16",
}

# Numeric ONNX TensorProto.DataType codes.
# 16 (BFLOAT16) stays the string sentinel used by ONNX_DTYPE_MAP. ml_dtypes
# is optional, so the reverse map has no bfloat16 key.
# 14 and 15 are COMPLEX64 and COMPLEX128. io_binding already converts those
# dtypes, so they stay on this shared map.
ONNX_ELEMENT_TYPE_TO_NUMPY = {
    1: np.float32,
    2: np.uint8,
    3: np.int8,
    4: np.uint16,
    5: np.int16,
    6: np.int32,
    7: np.int64,
    9: np.bool_,
    10: np.float16,
    11: np.float64,
    12: np.uint32,
    13: np.uint64,
    14: np.complex64,
    15: np.complex128,
    16: "bfloat16",
}

NUMPY_TO_ONNX_ELEMENT_TYPE = {
    np.float32: 1,
    np.uint8: 2,
    np.int8: 3,
    np.uint16: 4,
    np.int16: 5,
    np.int32: 6,
    np.int64: 7,
    np.bool_: 9,
    np.float16: 10,
    np.float64: 11,
    np.uint32: 12,
    np.uint64: 13,
    np.complex64: 14,
    np.complex128: 15,
}

# Short form type mappings (without "tensor()" wrapper)
SHORT_DTYPE_MAP = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "float": np.float32,
    "double": np.float64,
    "bfloat16": "bfloat16",
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "bool": np.bool_,
    "complex64": np.complex64,
    "complex128": np.complex128,
}

# GGUF format type mappings (for llama.cpp compatibility)
# GGUF uses its own type identifiers for quantized models
GGUF_DTYPE_MAP = {
    "F32": np.float32,
    "F16": np.float16,
    "BF16": "bfloat16",
    # Quantized types (stored as int8 but represent packed values)
    "Q4_0": np.int8,     # 4-bit quantization
    "Q4_1": np.int8,     # 4-bit with scaling
    "Q4_K": np.int8,     # 4-bit k-quant
    "Q4_K_S": np.int8,   # 4-bit k-quant small
    "Q4_K_M": np.int8,   # 4-bit k-quant medium
    "Q5_0": np.int8,     # 5-bit quantization
    "Q5_1": np.int8,     # 5-bit with scaling
    "Q5_K": np.int8,     # 5-bit k-quant
    "Q5_K_S": np.int8,   # 5-bit k-quant small
    "Q5_K_M": np.int8,   # 5-bit k-quant medium
    "Q8_0": np.int8,     # 8-bit quantization
    "Q8_1": np.int8,     # 8-bit with scaling
    "Q8_K": np.int8,     # 8-bit k-quant
    # Integer types
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
}

# AWQ (Activation-aware Weight Quantization) format type mappings
# Used by RyzenAI and similar NPU-optimized quantized models
AWQ_DTYPE_MAP = {
    # AWQ INT4 variants (most common for LLMs)
    "awq_int4": np.int8,       # 4-bit AWQ packed weights (2 values per byte)
    "awq_uint4": np.uint8,     # Unsigned 4-bit AWQ
    "awq_gemm_int4": np.int8,  # AWQ with GEMM optimizations
    "awq_gemv_int4": np.int8,  # AWQ with GEMV optimizations
    
    # AWQ INT8 variants
    "awq_int8": np.int8,       # 8-bit AWQ (higher precision)
    
    # AWQ with different activation types
    "awq_w4a16": np.int8,      # 4-bit weights, 16-bit activations
    "awq_w4a8": np.int8,       # 4-bit weights, 8-bit activations
    "awq_w8a16": np.int8,      # 8-bit weights, 16-bit activations
    
    # Activation dtypes used with AWQ
    "awq_fp16_act": np.float16,   # FP16 activations
    "awq_bf16_act": "bfloat16",   # BF16 activations
}


# =============================================================================
# QUANTIZED DTYPE METADATA
# =============================================================================

@dataclass(frozen=True)
class QuantizedDtype:
    """Packing metadata for a sub-byte weight dtype.

    ``ONNX_DTYPE_MAP`` still names the storage numpy dtype. This record
    names the logical width, the group, the scale, the zero point, and
    the packing order used by blocked INT4, MatMulNBits, AWQ, and GPTQ.
    """

    logical_dtype: str
    bits: int
    signed: bool
    storage_dtype: np.dtype
    group_size: Optional[int]
    has_scale: bool
    has_zero_point: bool
    scale_dtype: Optional[np.dtype]
    zero_point_dtype: Optional[np.dtype]
    layout: Optional[str]
    packing_axis: Optional[str] = None
    bit_order: str = "low_bits_first"
    interleave: int = 1

    def __post_init__(self):
        storage = np.dtype(self.storage_dtype)
        object.__setattr__(self, "storage_dtype", storage)
        if self.scale_dtype is not None:
            object.__setattr__(self, "scale_dtype", np.dtype(self.scale_dtype))
        if self.zero_point_dtype is not None:
            object.__setattr__(self, "zero_point_dtype", np.dtype(self.zero_point_dtype))
        width = storage.itemsize * 8
        if self.bits < 1 or self.bits > width:
            raise ValueError(f"bits must fit in {storage}, got {self.bits}")
        if width % self.bits != 0:
            raise ValueError(
                f"{self.bits}-bit values do not pack evenly into {storage}"
            )
        if self.group_size is not None:
            if int(self.group_size) < 1:
                raise ValueError(f"group_size must be positive, got {self.group_size}")
            object.__setattr__(self, "group_size", int(self.group_size))
        if self.has_scale and self.scale_dtype is None:
            raise ValueError("has_scale requires scale_dtype")
        if not self.has_scale and self.scale_dtype is not None:
            raise ValueError("scale_dtype is set but has_scale is false")
        if self.has_zero_point and self.zero_point_dtype is None:
            raise ValueError("has_zero_point requires zero_point_dtype")
        if not self.has_zero_point and self.zero_point_dtype is not None:
            raise ValueError("zero_point_dtype is set but has_zero_point is false")
        if self.bit_order not in ("low_bits_first", "high_bits_first"):
            raise ValueError(f"unsupported bit_order: {self.bit_order}")
        if int(self.interleave) < 1:
            raise ValueError(f"interleave must be positive, got {self.interleave}")
        object.__setattr__(self, "interleave", int(self.interleave))

    @property
    def values_per_storage_element(self) -> int:
        """Logical values stored in one storage scalar."""
        return (self.storage_dtype.itemsize * 8) // self.bits

    @property
    def block_size(self) -> Optional[int]:
        """MatMulNBits block size. These layouts use the group size."""
        return self.group_size

    def matmul_nbits_attributes(self) -> dict:
        """Attributes a MatMulNBits node needs besides K and N."""
        if self.group_size is None:
            raise ValueError("group_size is required for MatMulNBits block_size")
        return {"bits": int(self.bits), "block_size": int(self.group_size)}

    def packing_order(self) -> dict:
        """How logical values sit inside the storage container."""
        layout = (self.layout or "").lower()
        axis = self.packing_axis
        if axis is None:
            axis = {
                "blocked": "k_block",
                "matmul_nbits": "k_block",
                "awq": "output",
                "gptq": "input",
            }.get(layout)
        return {
            "layout": self.layout,
            "bit_order": self.bit_order,
            "axis": axis,
            "interleave": self.interleave,
            "values_per_storage_element": self.values_per_storage_element,
            "group_size": self.group_size,
            "block_size": self.group_size,
        }

    def scale_shape(self, weight_shape: tuple) -> tuple:
        """Scale tensor shape for a rank-2 weight of this layout."""
        if self.group_size is None:
            raise ValueError("group_size is required to place scales")
        if len(weight_shape) != 2:
            raise ValueError(f"weight shape must be rank 2, got {weight_shape}")
        layout = (self.layout or "").lower()
        if layout == "awq":
            rows, cols = weight_shape
            if rows % self.group_size != 0:
                raise ValueError(
                    f"AWQ K {rows} is not a multiple of group_size {self.group_size}"
                )
            return (rows // self.group_size, cols)
        rows, cols = weight_shape
        if cols % self.group_size != 0:
            raise ValueError(
                f"K {cols} is not a multiple of group_size {self.group_size}"
            )
        return (rows, cols // self.group_size)


INT4_BLOCKWISE = QuantizedDtype(
    logical_dtype="int4",
    bits=4,
    signed=True,
    storage_dtype=np.int8,
    group_size=32,
    has_scale=True,
    has_zero_point=True,
    scale_dtype=np.float16,
    zero_point_dtype=np.uint8,
    layout="blocked",
)

UINT4_BLOCKWISE = QuantizedDtype(
    logical_dtype="uint4",
    bits=4,
    signed=False,
    storage_dtype=np.uint8,
    group_size=32,
    has_scale=True,
    has_zero_point=True,
    scale_dtype=np.float16,
    zero_point_dtype=np.uint8,
    layout="blocked",
)

AWQ_INT4 = QuantizedDtype(
    logical_dtype="uint4",
    bits=4,
    signed=False,
    storage_dtype=np.int32,
    group_size=128,
    has_scale=True,
    has_zero_point=True,
    scale_dtype=np.float16,
    zero_point_dtype=np.int32,
    layout="awq",
    packing_axis="output",
)

GPTQ_INT4 = QuantizedDtype(
    logical_dtype="uint4",
    bits=4,
    signed=False,
    storage_dtype=np.int32,
    group_size=128,
    has_scale=True,
    has_zero_point=True,
    scale_dtype=np.float16,
    zero_point_dtype=np.int32,
    layout="gptq",
    packing_axis="input",
)


def quantized_dtype_from_matmul_nbits(
    bits: int = 4,
    block_size: int = 128,
    signed: bool = False,
    has_zero_point: bool = True,
    scale_dtype: Optional[np.dtype] = np.float16,
    zero_point_dtype: Optional[np.dtype] = np.uint8,
) -> QuantizedDtype:
    """Build MatMulNBits metadata from the node attributes.

    ORT packs ``block_size`` values along K. The low bits of each byte
    hold the first value. ``block_size`` is the group size. The default
    attributes are bits 4 and block size 128.
    """
    if bits not in (4, 8):
        raise ValueError(f"MatMulNBits bits must be 4 or 8, got {bits}")
    if int(block_size) < 1 or (int(block_size) * int(bits)) % 8 != 0:
        raise ValueError(
            f"block_size {block_size} * bits {bits} must fill whole bytes"
        )
    return QuantizedDtype(
        logical_dtype=f"{'int' if signed else 'uint'}{int(bits)}",
        bits=int(bits),
        signed=bool(signed),
        storage_dtype=np.int8 if signed else np.uint8,
        group_size=int(block_size),
        has_scale=True,
        has_zero_point=bool(has_zero_point),
        scale_dtype=scale_dtype,
        zero_point_dtype=zero_point_dtype if has_zero_point else None,
        layout="matmul_nbits",
        packing_axis="k_block",
    )


MATMUL_NBITS_INT4 = quantized_dtype_from_matmul_nbits()

QUANTIZED_DTYPE_MAP = {
    "int4": INT4_BLOCKWISE,
    "tensor(int4)": INT4_BLOCKWISE,
    "int4_blockwise": INT4_BLOCKWISE,
    "uint4": UINT4_BLOCKWISE,
    "tensor(uint4)": UINT4_BLOCKWISE,
    "uint4_blockwise": UINT4_BLOCKWISE,
    "matmul_nbits": MATMUL_NBITS_INT4,
    "matmulnbits": MATMUL_NBITS_INT4,
    "awq": AWQ_INT4,
    "awq_int4": AWQ_INT4,
    "awq_uint4": AWQ_INT4,
    "awq_gemm_int4": AWQ_INT4,
    "awq_gemv_int4": AWQ_INT4,
    "awq_w4a16": AWQ_INT4,
    "gptq": GPTQ_INT4,
    "gptq_int4": GPTQ_INT4,
}


def get_quantized_dtype(name: str) -> QuantizedDtype:
    """Return packing metadata for an INT4, UINT4, MatMulNBits, AWQ, or GPTQ name."""
    key = str(name).strip().lower()
    if key not in QUANTIZED_DTYPE_MAP:
        known = ", ".join(sorted(QUANTIZED_DTYPE_MAP))
        raise ValueError(f"Unrecognized quantized dtype: {name}. Known names: {known}")
    return QUANTIZED_DTYPE_MAP[key]


def _logical_to_codes(matrix: np.ndarray, spec: QuantizedDtype) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.int32)
    mask = (1 << spec.bits) - 1
    if spec.signed:
        limit = 1 << (spec.bits - 1)
        if np.any(values < -limit) or np.any(values > limit - 1):
            raise ValueError(
                f"signed {spec.bits}-bit values must be in [{-limit}, {limit - 1}]"
            )
        return (values & mask).astype(np.uint32)
    if np.any(values < 0) or np.any(values > mask):
        raise ValueError(f"unsigned {spec.bits}-bit values must be in [0, {mask}]")
    return values.astype(np.uint32)


def _as_unsigned_words(packed: np.ndarray, storage: np.dtype) -> np.ndarray:
    raw = np.asarray(packed)
    if storage == np.dtype(np.int8):
        return raw.view(np.uint8).astype(np.uint32)
    if storage == np.dtype(np.uint8):
        return raw.astype(np.uint32)
    if storage == np.dtype(np.int32):
        return raw.view(np.uint32)
    if storage == np.dtype(np.uint32):
        return raw.astype(np.uint32)
    raise ValueError(f"unsupported storage dtype {storage}")


def _store_words(words: np.ndarray, storage: np.dtype) -> np.ndarray:
    if storage.itemsize == 1:
        narrowed = words.astype(np.uint8)
        if storage == np.dtype(np.int8):
            return narrowed.view(np.int8)
        return narrowed
    if storage == np.dtype(np.int32):
        return words.astype(np.uint32).view(np.int32)
    if storage == np.dtype(np.uint32):
        return words.astype(np.uint32)
    raise ValueError(f"unsupported storage dtype {storage}")


def pack_quantized(values: np.ndarray, spec: QuantizedDtype) -> np.ndarray:
    """Pack the last axis into ``spec.storage_dtype``.

    Slot 0 occupies the low bits. A 4-bit byte layout stores two values
    per byte. AWQ and GPTQ store eight values per int32.
    """
    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError("values must include a logical axis")
    if spec.interleave != 1:
        raise ValueError(
            f"interleave {spec.interleave} is recorded; packing implements interleave 1"
        )
    if spec.bit_order != "low_bits_first":
        raise ValueError(f"bit_order {spec.bit_order} is not implemented")
    per = spec.values_per_storage_element
    logical = array.shape[-1]
    if logical % per != 0:
        raise ValueError(f"logical count {logical} is not a multiple of {per}")
    leading = array.shape[:-1]
    rows = int(np.prod(leading, dtype=np.int64)) if leading else 1
    codes = _logical_to_codes(np.reshape(array, (rows, logical)), spec)
    words = np.zeros((rows, logical // per), dtype=np.uint32)
    for slot in range(per):
        words |= codes[:, slot::per] << np.uint32(spec.bits * slot)
    stored = _store_words(words, spec.storage_dtype)
    return stored.reshape(leading + (logical // per,))


def unpack_quantized(packed: np.ndarray, spec: QuantizedDtype) -> np.ndarray:
    """Inverse of ``pack_quantized``. Signed layouts restore two's-complement nibbles."""
    array = np.asarray(packed)
    if array.ndim == 0:
        raise ValueError("packed values must include a storage axis")
    if spec.interleave != 1:
        raise ValueError(
            f"interleave {spec.interleave} is recorded; unpacking implements interleave 1"
        )
    if spec.bit_order != "low_bits_first":
        raise ValueError(f"bit_order {spec.bit_order} is not implemented")
    if array.dtype != spec.storage_dtype:
        raise ValueError(
            f"packed dtype {array.dtype} does not match storage {spec.storage_dtype}"
        )
    per = spec.values_per_storage_element
    leading = array.shape[:-1]
    stored = array.shape[-1]
    rows = int(np.prod(leading, dtype=np.int64)) if leading else 1
    words = _as_unsigned_words(np.reshape(array, (rows, stored)), spec.storage_dtype)
    logical = stored * per
    codes = np.zeros((rows, logical), dtype=np.int16)
    mask = (1 << spec.bits) - 1
    for slot in range(per):
        codes[:, slot::per] = (words >> np.uint32(spec.bits * slot)) & np.uint32(mask)
    if spec.signed:
        sign = 1 << (spec.bits - 1)
        codes = np.where(codes >= sign, codes - (1 << spec.bits), codes)
    return codes.reshape(leading + (logical,))


def pack_weights(weight: np.ndarray, spec: QuantizedDtype) -> np.ndarray:
    """Pack a rank-2 weight into the layout named by ``spec``.

    Blocked and MatMulNBits weights are ``[N, K]`` and become
    ``[N, K / block_size, blob]``. AWQ weights are ``[K, N]`` and become
    ``[K, N / values_per_word]``. GPTQ weights are ``[N, K]`` and become
    ``[N, K / values_per_word]``.
    """
    array = np.asarray(weight)
    if array.ndim != 2:
        raise ValueError(f"weight must be rank 2, got shape {array.shape}")
    layout = (spec.layout or "").lower()
    if layout in ("blocked", "matmul_nbits"):
        if spec.group_size is None:
            raise ValueError("group_size is required for blocked packing")
        _rows, cols = array.shape
        if cols % spec.group_size != 0:
            raise ValueError(
                f"K {cols} is not a multiple of block_size {spec.group_size}"
            )
        blocks = cols // spec.group_size
        grouped = array.reshape(array.shape[0], blocks, spec.group_size)
        return pack_quantized(grouped, spec)
    if layout == "awq":
        _rows, cols = array.shape
        if cols % spec.values_per_storage_element != 0:
            raise ValueError(
                f"AWQ N {cols} is not a multiple of {spec.values_per_storage_element}"
            )
        return pack_quantized(array, spec)
    if layout == "gptq":
        _rows, cols = array.shape
        if cols % spec.values_per_storage_element != 0:
            raise ValueError(
                f"GPTQ K {cols} is not a multiple of {spec.values_per_storage_element}"
            )
        return pack_quantized(array, spec)
    raise ValueError(f"unsupported quantized layout: {spec.layout}")


def unpack_weights(packed: np.ndarray, spec: QuantizedDtype) -> np.ndarray:
    """Inverse of ``pack_weights``."""
    array = np.asarray(packed)
    layout = (spec.layout or "").lower()
    if layout in ("blocked", "matmul_nbits"):
        if array.ndim != 3:
            raise ValueError(
                f"MatMulNBits storage must be [N, blocks, blob], got {array.shape}"
            )
        if spec.group_size is None:
            raise ValueError("group_size is required for blocked unpacking")
        logical = unpack_quantized(array, spec)
        if logical.shape[-1] != spec.group_size:
            raise ValueError(
                f"unpacked block {logical.shape[-1]} does not match group_size {spec.group_size}"
            )
        return logical.reshape(logical.shape[0], logical.shape[1] * logical.shape[2])
    if layout in ("awq", "gptq"):
        if array.ndim != 2:
            raise ValueError(f"{layout} storage must be rank 2, got {array.shape}")
        return unpack_quantized(array, spec)
    raise ValueError(f"unsupported quantized layout: {spec.layout}")


# =============================================================================
# TYPE CONVERSION UTILITIES
# =============================================================================

def get_numpy_dtype(onnx_type: str) -> np.dtype:
    """
    Get numpy dtype for an ONNX type string.
    
    Args:
        onnx_type: ONNX type string, e.g., "tensor(float16)" or "float16"
        
    Returns:
        numpy dtype suitable for the ONNX type
        
    Raises:
        ValueError: If the ONNX type is not recognized
    """
    # Try full form first
    if onnx_type in ONNX_DTYPE_MAP:
        result = ONNX_DTYPE_MAP[onnx_type]
        if result == "bfloat16":
            bf16 = get_bfloat16_dtype()
            if bf16 is not None:
                return bf16
            warnings.warn(
                "bfloat16 requested but ml_dtypes not available. "
                "Falling back to float32 to preserve precision.",
                UserWarning
            )
            return np.float32
        return result
    
    # Try extracting type from tensor() wrapper
    if onnx_type.startswith("tensor(") and onnx_type.endswith(")"):
        inner_type = onnx_type[7:-1]
        if inner_type in SHORT_DTYPE_MAP:
            result = SHORT_DTYPE_MAP[inner_type]
            if result == "bfloat16":
                bf16 = get_bfloat16_dtype()
                return bf16 if bf16 is not None else np.float32
            return result
    
    # Try short form
    if onnx_type in SHORT_DTYPE_MAP:
        result = SHORT_DTYPE_MAP[onnx_type]
        if result == "bfloat16":
            bf16 = get_bfloat16_dtype()
            return bf16 if bf16 is not None else np.float32
        return result
    
    # Fallback: try parsing common patterns
    onnx_lower = onnx_type.lower()
    if "float16" in onnx_lower:
        return np.float16
    elif "bfloat16" in onnx_lower:
        bf16 = get_bfloat16_dtype()
        return bf16 if bf16 is not None else np.float32
    elif "float8" in onnx_lower:
        warnings.warn(f"Float8 type '{onnx_type}' falling back to float16", UserWarning)
        return np.float16
    elif "float64" in onnx_lower or "double" in onnx_lower:
        return np.float64
    elif "float" in onnx_lower:
        return np.float32
    elif "int64" in onnx_lower:
        return np.int64
    elif "int32" in onnx_lower:
        return np.int32
    elif "int16" in onnx_lower:
        return np.int16
    elif "int8" in onnx_lower:
        return np.int8
    elif "int4" in onnx_lower:
        warnings.warn("INT4 type requires packed storage (2 values/byte)", UserWarning)
        return np.int8
    elif "uint" in onnx_lower:
        if "64" in onnx_lower:
            return np.uint64
        elif "32" in onnx_lower:
            return np.uint32
        elif "16" in onnx_lower:
            return np.uint16
        elif "4" in onnx_lower:
            return np.uint8
        return np.uint8
    elif "bool" in onnx_lower:
        return np.bool_
    
    raise ValueError(f"Unrecognized ONNX type: {onnx_type}")


def numpy_dtype_from_onnx_element_type(element_type: int):
    """
    Get the numpy dtype for an ONNX TensorProto.DataType code.

    Args:
        element_type: TensorProto.DataType integer, e.g. 1 for FLOAT

    Returns:
        numpy dtype for that code. BFLOAT16 (16) uses ml_dtypes when
        it is installed and float32 otherwise.

    Raises:
        ValueError: If the element type code is not recognized
    """
    if element_type not in ONNX_ELEMENT_TYPE_TO_NUMPY:
        raise ValueError(f"Unrecognized ONNX element type: {element_type}")
    result = ONNX_ELEMENT_TYPE_TO_NUMPY[element_type]
    if result == "bfloat16":
        bf16 = get_bfloat16_dtype()
        if bf16 is not None:
            return bf16
        warnings.warn(
            "bfloat16 requested but ml_dtypes not available. "
            "Falling back to float32 to preserve precision.",
            UserWarning
        )
        return np.float32
    return result


def safe_cast(
    array: np.ndarray, 
    target_dtype: np.dtype,
    warn_on_precision_loss: bool = True
) -> np.ndarray:
    """
    Safely cast a numpy array to target dtype with optional precision warnings.
    
    Args:
        array: Source numpy array
        target_dtype: Target numpy dtype
        warn_on_precision_loss: If True, warn when casting may lose precision
        
    Returns:
        Array cast to target dtype
    """
    if array.dtype == target_dtype:
        return array
    
    source_dtype = array.dtype
    
    # Check for potential precision loss
    if warn_on_precision_loss:
        precision_loss_cases = [
            # Downcast float precision
            (np.float64, np.float32),
            (np.float64, np.float16),
            (np.float32, np.float16),
            # Int to smaller int
            (np.int64, np.int32),
            (np.int64, np.int16),
            (np.int64, np.int8),
            (np.int32, np.int16),
            (np.int32, np.int8),
            (np.int16, np.int8),
            # Float to int
            (np.float64, np.int64),
            (np.float32, np.int32),
            (np.float16, np.int16),
        ]
        
        for src, tgt in precision_loss_cases:
            if np.issubdtype(source_dtype, src) and target_dtype == tgt:
                warnings.warn(
                    f"Casting from {source_dtype} to {target_dtype} may lose precision",
                    UserWarning
                )
                break
    
    return array.astype(target_dtype)


def get_dtype_info(dtype: np.dtype) -> dict:
    """
    Get information about a numpy dtype.
    
    Args:
        dtype: numpy dtype
        
    Returns:
        Dictionary with dtype info (name, itemsize, is_float, is_int, etc.)
    """
    info = {
        "name": str(dtype),
        "itemsize": dtype.itemsize,
        "is_float": np.issubdtype(dtype, np.floating),
        "is_int": np.issubdtype(dtype, np.integer),
        "is_unsigned": np.issubdtype(dtype, np.unsignedinteger),
        "is_complex": np.issubdtype(dtype, np.complexfloating),
        "is_bool": dtype == np.bool_,
    }
    
    if info["is_float"]:
        finfo = np.finfo(dtype)
        info["min"] = float(finfo.min)
        info["max"] = float(finfo.max)
        info["eps"] = float(finfo.eps)
    elif info["is_int"]:
        iinfo = np.iinfo(dtype)
        info["min"] = int(iinfo.min)
        info["max"] = int(iinfo.max)
    
    return info


# =============================================================================
# PROVIDER DTYPE POLICY
# =============================================================================

# Ordered logical dtypes for AMD hybrid execution. The first name is the
# preferred choice. Later names are the fallbacks for that tensor role.
# "int4" materializes as the packed int8 container described by INT4_BLOCKWISE.
PROVIDER_DTYPE_POLICY: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "CPUExecutionProvider": {
        "weight": ("int8", "fp32"),
        "activation": ("fp32", "int8"),
        "kv": ("fp16", "int8"),
    },
    "ROCmExecutionProvider": {
        "weight": ("fp16", "bf16", "int8"),
        "activation": ("fp16", "bf16"),
        "kv": ("fp16", "fp8", "int8"),
    },
    "VitisAIExecutionProvider": {
        "weight": ("int4", "int8"),
        "activation": ("int8", "fp16"),
        "kv": ("fp16", "int8"),
    },
    "DmlExecutionProvider": {
        "weight": ("fp16", "fp32"),
        "activation": ("fp16", "fp32"),
        "kv": ("fp16",),
    },
}

_PROVIDER_ALIASES = {
    "cpu": "CPUExecutionProvider",
    "cpuexecutionprovider": "CPUExecutionProvider",
    "rocm": "ROCmExecutionProvider",
    "rocmexecutionprovider": "ROCmExecutionProvider",
    "hip": "ROCmExecutionProvider",
    "igpu": "ROCmExecutionProvider",
    "vitisai": "VitisAIExecutionProvider",
    "vitisaiexecutionprovider": "VitisAIExecutionProvider",
    "npu": "VitisAIExecutionProvider",
    "ryzenai": "VitisAIExecutionProvider",
    "ryzenaiexecutionprovider": "VitisAIExecutionProvider",
    "dml": "DmlExecutionProvider",
    "directml": "DmlExecutionProvider",
    "dmlexecutionprovider": "DmlExecutionProvider",
}

_ROLE_ALIASES = {
    "weight": "weight",
    "weights": "weight",
    "activation": "activation",
    "activations": "activation",
    "kv": "kv",
    "kvcache": "kv",
    "kv_cache": "kv",
    "cache": "kv",
}


def _canonical_provider(provider: str) -> str:
    key = str(provider).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if key not in _PROVIDER_ALIASES:
        known = ", ".join(PROVIDER_DTYPE_POLICY)
        raise ValueError(f"Unrecognized provider for dtype policy: {provider}. Known providers: {known}")
    return _PROVIDER_ALIASES[key]


def _canonical_role(tensor_role: str) -> str:
    key = str(tensor_role).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if key not in _ROLE_ALIASES:
        raise ValueError(
            f"Unrecognized tensor role: {tensor_role}. Known roles: weight, activation, kv"
        )
    return _ROLE_ALIASES[key]


def _fp8_dtype():
    """OCP FP8 dtype from ml_dtypes, or None when that package is absent."""
    try:
        import ml_dtypes
    except ImportError:
        return None
    for name in ("float8_e4m3fn", "float8_e5m2"):
        candidate = getattr(ml_dtypes, name, None)
        if candidate is not None:
            return candidate
    return None


def _dtype_for_logical(logical: str):
    """Numpy dtype for a policy name. None when this process cannot construct it."""
    key = str(logical).strip().lower()
    if key == "int4":
        return np.dtype(np.int8)
    if key == "int8":
        return np.dtype(np.int8)
    if key == "fp16":
        return np.dtype(np.float16)
    if key == "fp32":
        return np.dtype(np.float32)
    if key in ("bf16", "bfloat16"):
        bf16 = get_bfloat16_dtype()
        return None if bf16 is None else np.dtype(bf16)
    if key in ("fp8", "float8"):
        fp8 = _fp8_dtype()
        return None if fp8 is None else np.dtype(fp8)
    raise ValueError(f"Unrecognized logical dtype: {logical}")


def get_dtype_preferences(provider: str, tensor_role: str = "activation") -> Tuple[str, ...]:
    """Ordered logical dtypes for a provider and tensor role."""
    canonical = _canonical_provider(provider)
    role = _canonical_role(tensor_role)
    return PROVIDER_DTYPE_POLICY[canonical][role]


def resolve_preferred_dtype(preferences: Sequence[str]) -> np.dtype:
    """Return the first listed dtype this process can construct.

    A missing bfloat16 or float8 dtype is skipped. The next listed dtype
    is used, and one warning names the dtype that was skipped.
    """
    if not preferences:
        raise ValueError("dtype preferences must not be empty")
    skipped = []
    for logical in preferences:
        dtype = _dtype_for_logical(logical)
        if dtype is None:
            skipped.append(str(logical))
            continue
        if skipped:
            warnings.warn(
                f"Preferred dtype {skipped[0]} is unavailable. Using {logical}.",
                UserWarning,
            )
        return dtype
    raise ValueError(
        "None of the preferred dtypes are available: " + ", ".join(str(name) for name in preferences)
    )


def get_recommended_dtype(
    provider: str,
    tensor_role: str = "activation",
) -> np.dtype:
    """Preferred numpy dtype for a provider and tensor role.

    VitisAI weights return the packed INT4 container (``np.int8``). The
    matching ``QuantizedDtype`` is ``INT4_BLOCKWISE``, available from
    ``get_recommended_quantized_dtype``. KV cache on that NPU is float16.
    RyzenAI uses the same NPU policy as VitisAI.
    """
    return resolve_preferred_dtype(get_dtype_preferences(provider, tensor_role))


def get_recommended_quantized_dtype(
    provider: str,
    tensor_role: str = "activation",
) -> Optional[QuantizedDtype]:
    """Packing record when the preferred dtype is sub-byte, else None."""
    for logical in get_dtype_preferences(provider, tensor_role):
        if _dtype_for_logical(logical) is None:
            continue
        if logical == "int4":
            return INT4_BLOCKWISE
        return None
    raise ValueError(
        f"No materialized dtype for {provider} role {tensor_role}"
    )


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

__all__ = [
    # Maps
    "ONNX_DTYPE_MAP",
    "ONNX_ELEMENT_TYPE_TO_NUMPY",
    "NUMPY_TO_ONNX_ELEMENT_TYPE",
    "SHORT_DTYPE_MAP",
    "SPECIAL_TYPES",
    "GGUF_DTYPE_MAP",
    "AWQ_DTYPE_MAP",
    "QUANTIZED_DTYPE_MAP",
    "PROVIDER_DTYPE_POLICY",
    # Quantized metadata
    "QuantizedDtype",
    "INT4_BLOCKWISE",
    "UINT4_BLOCKWISE",
    "MATMUL_NBITS_INT4",
    "AWQ_INT4",
    "GPTQ_INT4",
    # Functions
    "get_numpy_dtype",
    "numpy_dtype_from_onnx_element_type",
    "get_quantized_dtype",
    "quantized_dtype_from_matmul_nbits",
    "pack_quantized",
    "unpack_quantized",
    "pack_weights",
    "unpack_weights",
    "safe_cast",
    "get_dtype_info",
    "supports_bfloat16",
    "get_bfloat16_dtype",
    "get_dtype_preferences",
    "resolve_preferred_dtype",
    "get_recommended_dtype",
    "get_recommended_quantized_dtype",
]
