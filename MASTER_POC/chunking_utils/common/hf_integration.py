"""
HuggingFace Transformers integration for model chunking.

This module provides utilities for integrating with the HuggingFace ecosystem,
including AutoConfig for architecture detection and accelerate for memory-efficient loading.
"""

import os
import warnings
from typing import Optional, Dict, Any, Iterable, Tuple

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


# First positive value wins. max_position_embeddings is the Hugging Face name.
_CONTEXT_LENGTH_KEYS = (
    "max_seq_len",
    "max_position_embeddings",
    "max_sequence_length",
    "seq_length",
    "n_positions",
)
_NESTED_CONFIG_ATTRS = ("text_config", "llm_config", "decoder")


def _config_sources(config: Any) -> Tuple[Any, ...]:
    """Top config, then a nested text or decoder config when the top omits a field."""
    sources = [config]
    seen = {id(config)}
    for name in _NESTED_CONFIG_ATTRS:
        nested = _lookup(config, name)
        if nested is None or id(nested) in seen:
            continue
        if isinstance(nested, (str, int, float, bool, list, tuple)):
            continue
        sources.append(nested)
        seen.add(id(nested))
    return tuple(sources)


def _lookup(source: Any, name: str) -> Any:
    """Return an attribute or dict entry. Missing and null stay None."""
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def _as_config_int(value: Any, name: str) -> int:
    """Accept an integer dimension. Bools and fractional floats are rejected."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{name} must be an integer, got {value!r}")
        value = int(value)
    return int(value)


def _first_positive_int(config: Any, names: Iterable[str]) -> Optional[int]:
    """First positive integer among names. Zero falls through. Negatives raise."""
    for name in names:
        for source in _config_sources(config):
            value = _lookup(source, name)
            if value is None:
                continue
            parsed = _as_config_int(value, name)
            if parsed == 0:
                continue
            if parsed < 0:
                raise ValueError(f"{name} must be non-negative, got {parsed}")
            return parsed
    return None


def _first_bool(config: Any, name: str) -> Optional[bool]:
    """First real boolean. A missing flag stays unset."""
    for source in _config_sources(config):
        if isinstance(source, dict):
            if name not in source or source[name] is None:
                continue
            value = source[name]
        elif not hasattr(source, name):
            continue
        else:
            value = getattr(source, name)
            if value is None:
                continue
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean, got {value!r}")
        return value
    return None


def _first_number(config: Any, name: str, allow_zero: bool = False) -> Optional[float]:
    """First finite number. Zero is unset unless allow_zero is set."""
    for source in _config_sources(config):
        value = _lookup(source, name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number, got {value!r}")
        number = float(value)
        if number == 0 and not allow_zero:
            continue
        if number < 0:
            raise ValueError(f"{name} must be non-negative, got {value!r}")
        return number
    return None


def _first_mapping(config: Any, name: str) -> Optional[Dict[str, Any]]:
    """Copy a nested dict such as rope_scaling. A non-dict raises."""
    for source in _config_sources(config):
        if isinstance(source, dict):
            if name not in source or source[name] is None:
                continue
            value = source[name]
        elif not hasattr(source, name):
            continue
        else:
            value = getattr(source, name)
            if value is None:
                continue
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a dict, got {value!r}")
        return dict(value)
    return None


def _architecture_value(config: Any, name: str, default: Any = None) -> Any:
    """Top-level value, then a nested config, then the default."""
    if isinstance(config, dict):
        if name in config and config[name] is not None:
            return config[name]
    elif hasattr(config, name):
        value = getattr(config, name)
        if value is not None:
            return value
    for source in _config_sources(config)[1:]:
        value = _lookup(source, name)
        if value is not None:
            return value
    return default


def _fields_from_hf_config(config: Any) -> Dict[str, Any]:
    """
    Read architecture, context, KV, RoPE, and MoE fields from a config object.

    A missing top-level field is read from text_config, llm_config, or decoder.
    Zero context lengths fall through to the next alias.
    """
    context = _first_positive_int(config, _CONTEXT_LENGTH_KEYS)
    sliding = _first_positive_int(config, ("sliding_window", "sliding_window_size"))
    sliding_flag = _first_bool(config, "use_sliding_window")
    if sliding_flag is None:
        sliding_flag = sliding is not None
    attention_heads = _first_positive_int(config, ("num_attention_heads",))
    kv_heads = _first_positive_int(
        config, ("num_key_value_heads", "num_kv_heads")
    )
    if kv_heads is None:
        kv_heads = attention_heads
    hidden = _first_positive_int(config, ("hidden_size",))
    head_dim = _first_positive_int(config, ("head_dim", "head_size"))
    if head_dim is None and hidden is not None and attention_heads is not None:
        if hidden % attention_heads != 0:
            warnings.warn(
                f"hidden_size ({hidden}) not divisible by num_attention_heads "
                f"({attention_heads}). Config may be malformed.",
                UserWarning,
            )
        head_dim = hidden // attention_heads
    experts = _first_positive_int(config, ("num_experts", "num_local_experts")) or 0
    per_token = _first_positive_int(
        config, ("num_experts_per_token", "num_experts_per_tok")
    ) or 0
    if experts > 0 and per_token > experts:
        raise ValueError(
            "num_experts_per_token must not exceed num_experts, "
            f"got {per_token} and {experts}"
        )
    moe_flag = _first_bool(config, "is_moe")
    torch_dtype = _architecture_value(config, "torch_dtype", "unknown")
    return {
        "model_type": _architecture_value(config, "model_type", "unknown"),
        "num_hidden_layers": _architecture_value(config, "num_hidden_layers"),
        "hidden_size": _architecture_value(config, "hidden_size"),
        "vocab_size": _architecture_value(config, "vocab_size"),
        "num_attention_heads": _architecture_value(config, "num_attention_heads"),
        "torch_dtype": str(torch_dtype),
        "max_position_embeddings": _first_positive_int(config, ("max_position_embeddings",)),
        "max_sequence_length": _first_positive_int(config, ("max_sequence_length",)),
        "max_seq_len": 0 if context is None else context,
        "sliding_window": sliding,
        "use_sliding_window": sliding_flag,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "rope_theta": _first_number(config, "rope_theta"),
        "rope_scaling": _first_mapping(config, "rope_scaling"),
        "is_moe": experts > 0 if moe_flag is None else moe_flag,
        "num_experts": experts,
        "num_experts_per_token": per_token,
        "moe_intermediate_size": _first_positive_int(config, ("moe_intermediate_size",)),
        "router_aux_loss_coef": _first_number(config, "router_aux_loss_coef", allow_zero=True),
    }


def get_model_config(model_path: str, trust_remote_code: bool = True) -> Dict[str, Any]:
    """
    Get model configuration from HuggingFace config.

    The result includes the context length, sliding window, KV heads, head
    dim, RoPE theta, and MoE counts used to preallocate a cache and plan a
    static NPU graph.

    Args:
        model_path: Path to model or HuggingFace model ID.
        trust_remote_code: Allow custom code from model repos.

    Returns:
        Dictionary with model configuration.
    """
    if not HAS_TRANSFORMERS:
        return {"error": "transformers not installed"}

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    return _fields_from_hf_config(config)


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
