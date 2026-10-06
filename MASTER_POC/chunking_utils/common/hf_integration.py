"""
HuggingFace Transformers integration for model chunking.

This module provides utilities for integrating with the HuggingFace ecosystem,
including AutoConfig for architecture detection and accelerate for memory-efficient loading.
"""

import os
from typing import Optional, Dict, Any

from .types import TensorDtype, MemoryMode
from .architecture import architecture_from_string, ArchitectureType

# Optional imports. ONNX Runtime is the execution target, so this module
# loads when PyTorch is absent.
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    from transformers import AutoConfig, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from accelerate import infer_auto_device_map
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False


# Dtype mapping between our enum and torch/ONNX types.
# INT4 has no native torch.dtype. AUTO preserves the checkpoint dtype.
if HAS_TORCH:
    TORCH_DTYPE_MAP: Dict[TensorDtype, Optional[Any]] = {
        TensorDtype.FLOAT32: torch.float32,
        TensorDtype.FLOAT16: torch.float16,
        TensorDtype.BFLOAT16: torch.bfloat16,
        TensorDtype.INT8: torch.int8,
        TensorDtype.INT4: None,  # Requires special quantization
        TensorDtype.AUTO: None,  # Preserve original
    }
else:
    TORCH_DTYPE_MAP = {item: None for item in TensorDtype}

# ONNX Runtime type strings. AUTO is omitted and falls back to float16.
ONNX_DTYPE_MAP: Dict[TensorDtype, str] = {
    TensorDtype.FLOAT32: "tensor(float)",
    TensorDtype.FLOAT16: "tensor(float16)",
    TensorDtype.BFLOAT16: "tensor(bfloat16)",
    TensorDtype.INT8: "tensor(int8)",
    TensorDtype.INT4: "tensor(int4)",
}


def get_torch_dtype(dtype: TensorDtype) -> Optional[Any]:
    """
    Convert TensorDtype enum to torch.dtype.
    
    Args:
        dtype: TensorDtype enum value.
    
    Returns:
        Corresponding torch.dtype, or None for AUTO and INT4.
    
    Raises:
        ImportError: If PyTorch is not installed.
    """
    if not HAS_TORCH:
        raise ImportError(
            "torch is required to resolve a PyTorch dtype. "
            "Install with: pip install torch"
        )
    return TORCH_DTYPE_MAP.get(dtype, None)


def get_onnx_dtype(dtype: TensorDtype) -> str:
    """
    Convert TensorDtype to an ONNX Runtime type string.

    Args:
        dtype: TensorDtype enum value.

    Returns:
        ONNX type such as tensor(float16). AUTO and unknown values use
        tensor(float16).
    """
    return ONNX_DTYPE_MAP.get(dtype, "tensor(float16)")


def detect_architecture_hf(model_path: str, trust_remote_code: bool = True) -> ArchitectureType:
    """
    Detect model architecture using HuggingFace AutoConfig.
    
    This is the recommended way to detect architecture as it handles
    all edge cases and custom model types.
    
    Args:
        model_path: Path to model or HuggingFace model ID.
        trust_remote_code: Allow custom code from model repos.
    
    Returns:
        Detected ArchitectureType.
    
    Raises:
        ImportError: If transformers not installed.
    """
    if not HAS_TRANSFORMERS:
        raise ImportError(
            "transformers is required for HuggingFace integration. "
            "Install with: pip install transformers"
        )
    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model_type = getattr(config, 'model_type', None)
    
    if model_type:
        return architecture_from_string(model_type)
    
    return ArchitectureType.UNKNOWN


def get_model_config(model_path: str, trust_remote_code: bool = True) -> Dict[str, Any]:
    """
    Get model configuration from HuggingFace config.
    
    Args:
        model_path: Path to model or HuggingFace model ID.
        trust_remote_code: Allow custom code from model repos.
    
    Returns:
        Dictionary with model configuration.
    """
    if not HAS_TRANSFORMERS:
        return {"error": "transformers not installed"}
    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    
    return {
        "model_type": getattr(config, 'model_type', 'unknown'),
        "num_hidden_layers": getattr(config, 'num_hidden_layers', None),
        "hidden_size": getattr(config, 'hidden_size', None),
        "vocab_size": getattr(config, 'vocab_size', None),
        "num_attention_heads": getattr(config, 'num_attention_heads', None),
        "torch_dtype": str(getattr(config, 'torch_dtype', 'unknown')),
    }


# Weight width in bits. INT4 is two values per byte.
_DTYPE_BITS = {
    TensorDtype.FLOAT32: 32,
    TensorDtype.FLOAT16: 16,
    TensorDtype.BFLOAT16: 16,
    TensorDtype.INT8: 8,
    TensorDtype.INT4: 4,
    TensorDtype.AUTO: 16,
}
_MEMORY_DTYPE_ALIASES = {
    "float32": TensorDtype.FLOAT32,
    "fp32": TensorDtype.FLOAT32,
    "float": TensorDtype.FLOAT32,
    "float16": TensorDtype.FLOAT16,
    "fp16": TensorDtype.FLOAT16,
    "half": TensorDtype.FLOAT16,
    "bfloat16": TensorDtype.BFLOAT16,
    "bf16": TensorDtype.BFLOAT16,
    "int8": TensorDtype.INT8,
    "uint8": TensorDtype.INT8,
    "int4": TensorDtype.INT4,
    "uint4": TensorDtype.INT4,
    "auto": TensorDtype.AUTO,
}
# Layout bytes match QuantizedDtype: AWQ and GPTQ use an int32 zero point,
# MatMulNBits and blocked INT4 use a uint8 zero point, and every group has
# a float16 scale except QDQ (float32) and bitsandbytes (float32 absmax).
_MEMORY_QUANT_ALIASES = {
    "awq": "awq",
    "awq_int4": "awq",
    "awq_w4a16": "awq",
    "gptq": "gptq",
    "gptq_int4": "gptq",
    "matmulnbits": "matmul_nbits",
    "matmul_nbits": "matmul_nbits",
    "blocked": "blocked",
    "int4_blockwise": "blocked",
    "int4": "blocked",
    "rtn": "rtn",
    "qdq": "qdq",
    "quark": "quark",
    "bitsandbytes": "bitsandbytes",
    "bnb": "bitsandbytes",
}
_QUANT_MEMORY_LAYOUT = {
    "awq": {"bits": 4, "group": 128, "scale": 2, "zero_point": 4, "pack": "int32"},
    "gptq": {"bits": 4, "group": 128, "scale": 2, "zero_point": 4, "pack": "int32"},
    "matmul_nbits": {"bits": 4, "group": 128, "scale": 2, "zero_point": 1, "pack": "block"},
    "blocked": {"bits": 4, "group": 32, "scale": 2, "zero_point": 1, "pack": "block"},
    "rtn": {"bits": None, "group": 32, "scale": 2, "zero_point": 1, "pack": "byte"},
    "qdq": {"bits": None, "group": None, "scale": 4, "zero_point": 1, "pack": "dense"},
    "quark": {"bits": 4, "group": 32, "scale": 2, "zero_point": 1, "pack": "block"},
    "bitsandbytes": {"bits": 4, "group": 64, "scale": 4, "zero_point": 0, "pack": "byte"},
}


def _coerce_memory_dtype(dtype: Any) -> TensorDtype:
    """Accept a TensorDtype or a short name such as fp16."""
    if isinstance(dtype, TensorDtype):
        return dtype
    if isinstance(dtype, bool) or not isinstance(dtype, str):
        raise ValueError(f"dtype must be a TensorDtype, got {dtype!r}")
    key = dtype.strip().lower().replace(" ", "").replace("-", "_")
    if key.startswith("torch."):
        key = key[6:]
    if key not in _MEMORY_DTYPE_ALIASES:
        known = ", ".join(item.value for item in TensorDtype)
        raise ValueError(f"Unknown dtype {dtype!r}. Expected one of: {known}")
    return _MEMORY_DTYPE_ALIASES[key]


def _coerce_memory_quantization(quantization: Optional[str]) -> Optional[str]:
    """Return a canonical scheme name. None leaves the dtype width unchanged."""
    if quantization is None:
        return None
    if isinstance(quantization, bool) or not isinstance(quantization, str):
        raise ValueError(f"quantization must be a string, got {quantization!r}")
    key = quantization.strip().lower().replace(" ", "").replace("-", "_")
    if not key:
        return None
    if key not in _MEMORY_QUANT_ALIASES:
        known = ", ".join(sorted(set(_MEMORY_QUANT_ALIASES.values())))
        raise ValueError(
            f"Unknown quantization {quantization!r}. Expected one of: {known}"
        )
    return _MEMORY_QUANT_ALIASES[key]


def _coerce_memory_group_size(group_size: Optional[int], default: Optional[int]) -> Optional[int]:
    """None uses the scheme default. -1 is per-channel and needs a channel count."""
    if group_size is None:
        return default
    if isinstance(group_size, bool) or not isinstance(group_size, int):
        raise ValueError(f"group_size must be an integer, got {group_size!r}")
    if group_size == -1:
        raise ValueError(
            "group_size -1 is per-channel and needs the channel count; "
            "pass a positive group_size"
        )
    if group_size < 1:
        raise ValueError(f"group_size must be positive or -1 for per-channel, got {group_size}")
    return group_size


def _storage_dtype(bits: int, requested: TensorDtype) -> TensorDtype:
    """Dtype that actually stores the weights after packing."""
    if bits == 4:
        return TensorDtype.INT4
    if bits == 8:
        return TensorDtype.INT8
    if bits == 16 and requested is TensorDtype.BFLOAT16:
        return TensorDtype.BFLOAT16
    if bits == 16:
        return TensorDtype.FLOAT16
    if bits == 32:
        return TensorDtype.FLOAT32
    return requested


def _weight_storage_bytes(num_params: int, bits: int, pack: str, group: Optional[int]) -> int:
    """Packed weight bytes, including a partial int32 word or a partial block."""
    if bits == 4 and pack == "int32":
        return ((num_params + 7) // 8) * 4
    if bits == 4 and pack == "block":
        if group is None:
            raise ValueError("group_size is required for blocked packing")
        if (group * bits) % 8 != 0:
            raise ValueError(
                f"group_size {group} * bits {bits} must fill whole bytes"
            )
        groups = (num_params + group - 1) // group if num_params else 0
        return groups * (group * bits // 8)
    if bits == 4:
        return (num_params + 1) // 2
    if bits % 8 != 0:
        raise ValueError(f"unsupported weight width {bits}")
    return num_params * (bits // 8)


def estimate_model_memory(
    num_params: int,
    dtype: TensorDtype = TensorDtype.FLOAT16,
    quantization: Optional[str] = None,
    group_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Estimate weight memory, including packed INT4 and quantization tensors.

    Args:
        num_params: Number of model parameters.
        dtype: Stored weight dtype. AWQ, GPTQ, and MatMulNBits still pack
            4-bit weights unless this dtype is INT8.
        quantization: Optional scheme such as awq, gptq, or matmul_nbits.
        group_size: Values sharing one scale. None uses the scheme default.
            -1 is per-channel and is rejected because the channel count is
            not in ``num_params``.

    Returns:
        Byte totals for the packed weights plus scale and zero-point tensors.
    """
    if isinstance(num_params, bool) or not isinstance(num_params, int):
        raise ValueError(f"num_params must be an integer, got {num_params!r}")
    if num_params < 0:
        raise ValueError(f"num_params must be non-negative, got {num_params}")
    requested = _coerce_memory_dtype(dtype)
    scheme = _coerce_memory_quantization(quantization)
    if scheme is None:
        bits = _DTYPE_BITS[requested]
        group = None
        scale_each = 0
        zero_each = 0
        pack = "byte" if bits == 4 else "dense"
    else:
        layout = _QUANT_MEMORY_LAYOUT[scheme]
        group = _coerce_memory_group_size(group_size, layout["group"])
        if layout["bits"] == 4 and requested is TensorDtype.INT8:
            bits = 8
            pack = "dense"
        elif layout["bits"] is None:
            bits = _DTYPE_BITS[requested]
            pack = "byte" if bits == 4 else "dense"
        else:
            bits = layout["bits"]
            pack = layout["pack"]
        scale_each = layout["scale"]
        zero_each = layout["zero_point"]
        if pack == "block" and group is not None and (group * bits) % 8 != 0:
            raise ValueError(
                f"group_size {group} * bits {bits} must fill whole bytes"
            )
    weight_bytes = _weight_storage_bytes(num_params, bits, pack, group)
    if scheme is None or num_params == 0:
        groups = 0
    elif group is None:
        groups = 1
    else:
        groups = (num_params + group - 1) // group
    scale_bytes = groups * scale_each
    zero_point_bytes = groups * zero_each
    overhead_bytes = scale_bytes + zero_point_bytes
    total_bytes = weight_bytes + overhead_bytes
    storage = _storage_dtype(bits, requested)
    return {
        "bytes": total_bytes,
        "megabytes": total_bytes / (1024 ** 2),
        "gigabytes": total_bytes / (1024 ** 3),
        "dtype": storage.value,
        "bytes_per_param": bits / 8,
        "quantization": scheme,
        "group_size": group,
        "packing": pack,
        "weight_bytes": weight_bytes,
        "scale_bytes": scale_bytes,
        "zero_point_bytes": zero_point_bytes,
        "overhead_bytes": overhead_bytes,
    }


def check_dependencies() -> Dict[str, bool]:
    """
    Check available optional dependencies.
    
    Returns:
        Dictionary indicating which libraries are available.
    """
    return {
        "transformers": HAS_TRANSFORMERS,
        "accelerate": HAS_ACCELERATE,
        "torch": HAS_TORCH,
    }
