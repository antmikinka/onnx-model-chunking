#!/usr/bin/env python3
"""
dtypes.py - Comprehensive ONNX Data Type Handling Module

This module provides complete ONNX type mappings and safe type conversion
utilities for use across inference pipelines.

Usage:
    from pipeline_utils.dtypes import get_numpy_dtype, safe_cast, ONNX_DTYPE_MAP
"""

import warnings
from typing import Optional, Union
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
    "int4": "Packed format - 2 values per byte, requires unpacking for computation",
    "uint4": "Packed format - 2 values per byte, requires unpacking for computation",
    "float8e4m3fn": "8-bit float (E4M3), falls back to float16",
    "float8e5m2": "8-bit float (E5M2), falls back to float16",
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
# CONVENIENCE EXPORTS
# =============================================================================

__all__ = [
    # Maps
    "ONNX_DTYPE_MAP",
    "SHORT_DTYPE_MAP",
    "SPECIAL_TYPES",
    "GGUF_DTYPE_MAP",
    "AWQ_DTYPE_MAP",
    # Functions
    "get_numpy_dtype",
    "safe_cast",
    "get_dtype_info",
    "supports_bfloat16",
    "get_bfloat16_dtype",
]
