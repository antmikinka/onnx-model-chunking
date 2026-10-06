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


def _is_bfloat16_dtype(dtype) -> bool:
    """True for ml_dtypes.bfloat16 or the dtype get_bfloat16_dtype() returns."""
    try:
        resolved = np.dtype(dtype)
    except TypeError:
        return False
    native = get_bfloat16_dtype()
    if native is not None:
        try:
            if resolved == np.dtype(native):
                return True
        except TypeError:
            pass
    return "bfloat16" in resolved.name.lower()


def _round_float32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round float32 values to the nearest even bfloat16 and return uint16 bits.

    Bfloat16 keeps the top 16 bits of float32. The low bit of that half selects
    round-to-nearest-even. A NaN whose payload sits only in the discarded bits
    gets a quiet bit in the kept half so the result stays a NaN.
    """
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    nan = (bits & np.uint64(0x7FFFFFFF)) > np.uint64(0x7F800000)
    bits = np.where(nan, bits | np.uint64(0x00400000), bits)
    rounded = (
        bits + np.uint64(0x7FFF) + ((bits >> np.uint64(16)) & np.uint64(1))
    ) >> np.uint64(16)
    return np.ascontiguousarray(rounded.astype(np.uint16))


def bfloat16_to_uint16_view(arr: np.ndarray) -> np.ndarray:
    """View bfloat16 storage as uint16 bits.

    Accepts a native bfloat16 array or a uint16 container. The result shares
    memory with ``arr`` so a cache can be transported without a copy.
    """
    array = np.asarray(arr)
    if array.dtype == np.dtype(np.uint16):
        return array.view(np.uint16)
    if not _is_bfloat16_dtype(array.dtype):
        raise ValueError(
            "bfloat16_to_uint16_view expects bfloat16 or uint16 bit storage, "
            f"got {array.dtype}"
        )
    if array.dtype.itemsize != np.dtype(np.uint16).itemsize:
        raise ValueError(f"bfloat16 dtype {array.dtype} is not a 16-bit container")
    return array.view(np.uint16)


def uint16_view_to_bfloat16(arr: np.ndarray) -> np.ndarray:
    """View uint16 bfloat16 bits as the native dtype, or keep the uint16 container.

    NumPy has no built-in bfloat16. When ml_dtypes is absent the uint16 array
    is the bfloat16 value. When ml_dtypes is present the same bits are viewed
    as that dtype.
    """
    array = np.asarray(arr)
    if _is_bfloat16_dtype(array.dtype):
        return array
    if array.dtype != np.dtype(np.uint16):
        raise ValueError(
            f"uint16_view_to_bfloat16 expects uint16 bit storage, got {array.dtype}"
        )
    native = get_bfloat16_dtype()
    if native is None:
        return array
    native_dtype = np.dtype(native)
    if native_dtype.itemsize != np.dtype(np.uint16).itemsize:
        raise ValueError(f"bfloat16 dtype {native_dtype} is not a 16-bit container")
    return array.view(native_dtype)


def bfloat16_to_float32(arr: np.ndarray) -> np.ndarray:
    """Expand bfloat16 values to float32.

    ``arr`` is native bfloat16 or a uint16 bit container. Each code is placed
    in the high half of a float32 and the low half is zero.
    """
    bits = bfloat16_to_uint16_view(arr)
    wide = np.ascontiguousarray(bits).astype(np.uint32) << np.uint32(16)
    return wide.view(np.float32)


def float32_to_bfloat16(arr: np.ndarray) -> np.ndarray:
    """Round numeric values to bfloat16.

    The result is ml_dtypes.bfloat16 when that dtype is installed. Otherwise
    it is the uint16 bit container used to transport bfloat16 through NumPy
    and ORT session boundaries.
    """
    values = np.ascontiguousarray(arr, dtype=np.float32)
    return uint16_view_to_bfloat16(_round_float32_to_bf16_bits(values))


# =============================================================================
# FLOAT8 SUPPORT DETECTION
# =============================================================================

try:
    import ml_dtypes
    FP8_E4M3 = getattr(ml_dtypes, "float8_e4m3fn", None)
    FP8_E5M2 = getattr(ml_dtypes, "float8_e5m2", None)
    FP8_E4M3FNUZ = getattr(ml_dtypes, "float8_e4m3fnuz", None)
    FP8_E5M2FNUZ = getattr(ml_dtypes, "float8_e5m2fnuz", None)
except ImportError:
    FP8_E4M3 = None
    FP8_E5M2 = None
    FP8_E4M3FNUZ = None
    FP8_E5M2FNUZ = None

_FLOAT8_BY_KIND = {
    "e4m3": "e4m3fn",
    "e4m3fn": "e4m3fn",
    "float8e4m3fn": "e4m3fn",
    "e5m2": "e5m2",
    "float8e5m2": "e5m2",
    "e4m3fnuz": "e4m3fnuz",
    "float8e4m3fnuz": "e4m3fnuz",
    "e5m2fnuz": "e5m2fnuz",
    "float8e5m2fnuz": "e5m2fnuz",
}


def _float8_kind(kind: str) -> str:
    key = str(kind).strip().lower().replace("-", "").replace("_", "")
    if key not in _FLOAT8_BY_KIND:
        known = ", ".join(sorted(set(_FLOAT8_BY_KIND.values())))
        raise ValueError(f"Unrecognized float8 kind: {kind}. Known kinds: {known}")
    return _FLOAT8_BY_KIND[key]


def _float8_constant(kind: str):
    canonical = _float8_kind(kind)
    return {
        "e4m3fn": FP8_E4M3,
        "e5m2": FP8_E5M2,
        "e4m3fnuz": FP8_E4M3FNUZ,
        "e5m2fnuz": FP8_E5M2FNUZ,
    }[canonical]


def supports_float8() -> bool:
    """True when ml_dtypes provides both OCP E4M3FN and E5M2."""
    return FP8_E4M3 is not None and FP8_E5M2 is not None


def get_float8_dtype(kind: str = "e4m3fn"):
    """Return one FP8 dtype, or None when that variant is not installed."""
    return _float8_constant(kind)


def float8_storage_dtype(kind: str = "e4m3fn") -> np.dtype:
    """Dtype used to store compressed FP8 values.

    A native ml_dtypes value is stored as itself. E4M3 without that package
    is stored as uint8 codes. Other variants fall back to float16.
    """
    native = get_float8_dtype(kind)
    if native is not None:
        return np.dtype(native)
    if _float8_kind(kind) == "e4m3fn":
        return np.dtype(np.uint8)
    return np.dtype(np.float16)


def _build_fp8_e4m3_table() -> np.ndarray:
    """OCP E4M3FN codes. Only exponent 15 with mantissa 7 is NaN."""
    table = np.empty(256, dtype=np.float32)
    for code in range(256):
        sign = -1.0 if code & 0x80 else 1.0
        exp = (code >> 3) & 0xF
        mant = code & 0x7
        if exp == 15 and mant == 7:
            table[code] = np.nan
            continue
        if exp == 0:
            mag = (mant / 8.0) * (2.0 ** -6)
        else:
            mag = (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
        table[code] = np.float32(sign * mag)
    return table


_FP8_E4M3_TABLE = _build_fp8_e4m3_table()
_FP8_POS_ORDER = np.array(
    [
        code
        for code in np.argsort(_FP8_E4M3_TABLE[:128], kind="stable")
        if np.isfinite(_FP8_E4M3_TABLE[code])
    ],
    dtype=np.uint8,
)
_FP8_POS_LEVELS = np.ascontiguousarray(_FP8_E4M3_TABLE[_FP8_POS_ORDER])


def _quantize_e4m3_codes(values: np.ndarray) -> np.ndarray:
    scaled = np.nan_to_num(
        np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    abs_vals = np.minimum(np.abs(scaled), _FP8_POS_LEVELS[-1])
    index = np.searchsorted(_FP8_POS_LEVELS, abs_vals, side="left")
    index = np.clip(index, 1, len(_FP8_POS_LEVELS) - 1)
    left = index - 1
    right_dist = np.abs(_FP8_POS_LEVELS[index] - abs_vals)
    left_dist = np.abs(_FP8_POS_LEVELS[left] - abs_vals)
    pick = np.where(right_dist < left_dist, index, left)
    codes = _FP8_POS_ORDER[pick].astype(np.uint8)
    return np.where(scaled < 0, codes | np.uint8(0x80), codes).astype(np.uint8)


def quantize_float8(
    values: np.ndarray,
    scale: float = 1.0,
    kind: str = "e4m3fn",
) -> np.ndarray:
    """Compress values to FP8 for a cache that ORT will see only after dequantization.

    Native ml_dtypes keeps the real FP8 dtype. E4M3 without that package
    stores uint8 codes. The matching ``dequantize_float8`` call expands
    either form to float32 at the session boundary.
    """
    if float(scale) == 0.0 or not np.isfinite(scale):
        raise ValueError(f"float8 scale must be finite and non-zero, got {scale}")
    scaled = np.asarray(values, dtype=np.float32) / np.float32(scale)
    native = get_float8_dtype(kind)
    if native is not None:
        return np.ascontiguousarray(scaled.astype(native))
    if _float8_kind(kind) == "e4m3fn":
        return np.ascontiguousarray(_quantize_e4m3_codes(scaled))
    warnings.warn(
        f"Float8 kind '{kind}' falling back to float16",
        UserWarning,
    )
    return np.ascontiguousarray(scaled.astype(np.float16))


def dequantize_float8(stored: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Expand a compressed FP8 buffer to float32."""
    array = np.asarray(stored)
    factor = np.float32(scale)
    native_dtypes = [
        dtype
        for dtype in (FP8_E4M3, FP8_E5M2, FP8_E4M3FNUZ, FP8_E5M2FNUZ)
        if dtype is not None and array.dtype == np.dtype(dtype)
    ]
    if native_dtypes or array.dtype == np.dtype(np.float16):
        return np.asarray(array, dtype=np.float32) * factor
    if array.dtype == np.dtype(np.uint8) or array.dtype.itemsize == 1:
        codes = np.ascontiguousarray(array).view(np.uint8)
        return _FP8_E4M3_TABLE[codes] * factor
    raise ValueError(f"Cannot dequantize float8 storage dtype {array.dtype}")


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
    
    # Float8 Types. Resolved to ml_dtypes when installed, else float16.
    "tensor(float8e4m3fn)": "float8e4m3fn",
    "tensor(float8e5m2)": "float8e5m2",
    "tensor(float8e4m3fnuz)": "float8e4m3fnuz",
    "tensor(float8e5m2fnuz)": "float8e5m2fnuz",
}

# Types that require special handling or have precision implications
SPECIAL_TYPES = {
    "bfloat16": "Session dtype uses ml_dtypes if available, else float32. Transport stores round-to-nearest-even bits in uint16 when the native dtype is absent.",
    "int4": "Signed 4-bit values, two per int8 byte. INT4_BLOCKWISE records the group, scale, and zero point.",
    "uint4": "Unsigned 4-bit values, two per uint8 byte. UINT4_BLOCKWISE records the group, scale, and zero point.",
    "float8e4m3fn": "OCP E4M3FN. Uses ml_dtypes when available, else uint8 codes dequantized at the session boundary.",
    "float8e5m2": "OCP E5M2. Uses ml_dtypes when available, else falls back to float16.",
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
    17: "float8e4m3fn",
    18: "float8e4m3fnuz",
    19: "float8e5m2",
    20: "float8e5m2fnuz",
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
    "float8e4m3fn": "float8e4m3fn",
    "float8e5m2": "float8e5m2",
    "float8e4m3fnuz": "float8e4m3fnuz",
    "float8e5m2fnuz": "float8e5m2fnuz",
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

_FLOAT8_SENTINELS = {
    "float8e4m3fn": "e4m3fn",
    "float8e5m2": "e5m2",
    "float8e4m3fnuz": "e4m3fnuz",
    "float8e5m2fnuz": "e5m2fnuz",
}


def _resolve_float8_sentinel(name: str, onnx_type: str):
    native = get_float8_dtype(_FLOAT8_SENTINELS[name])
    if native is not None:
        return native
    warnings.warn(
        f"Float8 type '{onnx_type}' falling back to float16",
        UserWarning,
    )
    return np.float16


def _materialize_mapped_dtype(result, onnx_type: str):
    if result == "bfloat16":
        bf16 = get_bfloat16_dtype()
        if bf16 is not None:
            return bf16
        warnings.warn(
            "bfloat16 requested but ml_dtypes not available. "
            "Falling back to float32 to preserve precision.",
            UserWarning,
        )
        return np.float32
    if isinstance(result, str) and result in _FLOAT8_SENTINELS:
        return _resolve_float8_sentinel(result, onnx_type)
    return result


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
        return _materialize_mapped_dtype(ONNX_DTYPE_MAP[onnx_type], onnx_type)
    
    # Try extracting type from tensor() wrapper
    if onnx_type.startswith("tensor(") and onnx_type.endswith(")"):
        inner_type = onnx_type[7:-1]
        if inner_type in SHORT_DTYPE_MAP:
            return _materialize_mapped_dtype(SHORT_DTYPE_MAP[inner_type], onnx_type)
    
    # Try short form
    if onnx_type in SHORT_DTYPE_MAP:
        return _materialize_mapped_dtype(SHORT_DTYPE_MAP[onnx_type], onnx_type)
    
    # Fallback: try parsing common patterns
    onnx_lower = onnx_type.lower()
    if "float16" in onnx_lower:
        return np.float16
    elif "bfloat16" in onnx_lower:
        return _materialize_mapped_dtype("bfloat16", onnx_type)
    elif "float8" in onnx_lower:
        if "e4m3fnuz" in onnx_lower:
            sentinel = "float8e4m3fnuz"
        elif "e5m2fnuz" in onnx_lower:
            sentinel = "float8e5m2fnuz"
        elif "e5m2" in onnx_lower:
            sentinel = "float8e5m2"
        else:
            sentinel = "float8e4m3fn"
        return _resolve_float8_sentinel(sentinel, onnx_type)
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
    return _materialize_mapped_dtype(result, f"tensor({result})")


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
    """OCP E4M3FN from ml_dtypes, or None when that variant is absent."""
    native = get_float8_dtype("e4m3fn")
    return None if native is None else np.dtype(native)


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
    "bfloat16_to_float32",
    "float32_to_bfloat16",
    "bfloat16_to_uint16_view",
    "uint16_view_to_bfloat16",
    "FP8_E4M3",
    "FP8_E5M2",
    "FP8_E4M3FNUZ",
    "FP8_E5M2FNUZ",
    "supports_float8",
    "get_float8_dtype",
    "float8_storage_dtype",
    "quantize_float8",
    "dequantize_float8",
    "get_dtype_preferences",
    "resolve_preferred_dtype",
    "get_recommended_dtype",
    "get_recommended_quantized_dtype",
]
