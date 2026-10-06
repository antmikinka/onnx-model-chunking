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


def estimate_model_memory(
    num_params: int,
    dtype: TensorDtype = TensorDtype.FLOAT16
) -> Dict[str, float]:
    """
    Estimate memory requirements for a model.
    
    Args:
        num_params: Number of model parameters.
        dtype: Target data type.
    
    Returns:
        Dictionary with memory estimates in various units.
    """
    bytes_per_param = {
        TensorDtype.FLOAT32: 4,
        TensorDtype.FLOAT16: 2,
        TensorDtype.BFLOAT16: 2,
        TensorDtype.INT8: 1,
        TensorDtype.INT4: 0.5,
        TensorDtype.AUTO: 2,  # Assume float16
    }
    
    bpp = bytes_per_param.get(dtype, 2)
    total_bytes = num_params * bpp
    
    return {
        "bytes": total_bytes,
        "megabytes": total_bytes / (1024 ** 2),
        "gigabytes": total_bytes / (1024 ** 3),
        "dtype": dtype.value,
        "bytes_per_param": bpp,
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
