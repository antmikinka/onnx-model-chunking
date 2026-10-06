#!/usr/bin/env python3
"""
config_loader.py - Model Configuration Loading Utilities

This module provides utilities for loading and validating model configuration
from various sources (genai_config.json, config.json, model_info.json).

Usage:
    from pipeline_utils.config_loader import load_model_config, ModelConfig
    
    config = load_model_config("/path/to/model")
    print(config.num_layers, config.head_dim)
"""

import json
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import numpy as np


@dataclass
class ModelConfig:
    """
    Model configuration dataclass.
    
    Contains the essential parameters needed for inference:
    - Transformer architecture (layers, heads, dimensions)
    - Data types
    - Model metadata
    """
    num_layers: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int = 0
    vocab_size: int = 0
    num_attention_heads: int = 0
    intermediate_size: int = 0
    # 0 means the config named no static context length. A positive value is
    # the capacity used to preallocate the KV cache once.
    max_seq_len: int = 0
    # None means the checkpoint did not name a quantization scheme.
    # A positive group size is the INT4, AWQ, or MatMulNBits block.
    # -1 means per-channel quantization with one group along K.
    quantization_method: Optional[str] = None
    weight_dtype: Optional[str] = None
    activation_dtype: Optional[str] = None
    quantization_group_size: Optional[int] = None
    quantization_format: Optional[str] = None
    has_zero_point: Optional[bool] = None
    dtype: np.dtype = field(default_factory=lambda: np.dtype(np.float16))
    model_type: str = "unknown"
    
    # Source info
    source_file: str = ""
    
    def validate(self) -> None:
        """
        Validate configuration values.
        
        Raises:
            ValueError: If any values are invalid
        """
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {self.num_kv_heads}")
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {self.head_dim}")
        if self.max_seq_len < 0:
            raise ValueError(f"max_seq_len must be non-negative, got {self.max_seq_len}")
        if self.quantization_group_size is not None:
            group = self.quantization_group_size
            if isinstance(group, bool) or not isinstance(group, int) or group == 0 or group < -1:
                raise ValueError(
                    "quantization_group_size must be positive or -1 for "
                    f"per-channel, got {group!r}"
                )
        if self.has_zero_point is not None and not isinstance(self.has_zero_point, bool):
            raise ValueError(
                f"has_zero_point must be a boolean, got {self.has_zero_point!r}"
            )
        if self.head_dim % 8 != 0:
            warnings.warn(f"head_dim {self.head_dim} is not divisible by 8, may be non-standard")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "max_seq_len": self.max_seq_len,
            "quantization_method": self.quantization_method,
            "weight_dtype": self.weight_dtype,
            "activation_dtype": self.activation_dtype,
            "quantization_group_size": self.quantization_group_size,
            "quantization_format": self.quantization_format,
            "has_zero_point": self.has_zero_point,
            "dtype": str(self.dtype),
            "model_type": self.model_type,
            "source_file": self.source_file,
        }


def _load_json_safe(path: str) -> Optional[Dict[str, Any]]:
    """Safely load a JSON file, returning None on error."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, IOError):
        return None


def _parse_genai_config(config: Dict[str, Any], source: str) -> Optional[ModelConfig]:
    """Parse ONNX GenAI format config."""
    try:
        decoder = config.get("model", {}).get("decoder", {})
        if not decoder:
            return None
        
        num_layers = decoder.get("num_hidden_layers", 0)
        num_kv_heads = decoder.get("num_key_value_heads", 0)
        head_size = decoder.get("head_size", 0)
        hidden_size = decoder.get("hidden_size", 0)
        num_attention_heads = decoder.get("num_attention_heads", 0)
        model = config.get("model", {})
        
        if num_layers and num_kv_heads and head_size:
            return ModelConfig(
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_size,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                vocab_size=model.get("vocab_size", 0) if isinstance(model, dict) else 0,
                model_type=model.get("type", "unknown") if isinstance(model, dict) else "unknown",
                max_seq_len=_max_seq_len_from(decoder, model, config),
                **_quantization_from(decoder, model, config),
                dtype=get_dtype_from_config(config),
                source_file=source,
            )
    except (KeyError, TypeError):
        pass
    return None


def _parse_hf_config(config: Dict[str, Any], source: str) -> Optional[ModelConfig]:
    """Parse HuggingFace format config.json."""
    try:
        num_layers = config.get("num_hidden_layers", 0)
        num_kv_heads = config.get("num_key_value_heads", config.get("num_attention_heads", 0))
        num_attention_heads = config.get("num_attention_heads", 0)
        hidden_size = config.get("hidden_size", 0)
        
        # Calculate head_dim from hidden_size / num_attention_heads
        head_dim = 0
        if num_attention_heads > 0 and hidden_size > 0:
            if hidden_size % num_attention_heads != 0:
                warnings.warn(
                    f"hidden_size ({hidden_size}) not divisible by num_attention_heads ({num_attention_heads}). "
                    f"Config may be malformed.",
                    UserWarning
                )
            head_dim = hidden_size // num_attention_heads
        
        # Some configs have head_dim explicitly
        head_dim = config.get("head_dim", head_dim)
        
        if num_layers and num_kv_heads and head_dim:
            return ModelConfig(
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                hidden_size=hidden_size,
                vocab_size=config.get("vocab_size", 0),
                num_attention_heads=num_attention_heads,
                intermediate_size=config.get("intermediate_size", 0),
                model_type=config.get("model_type", "unknown"),
                max_seq_len=_max_seq_len_from(config),
                **_quantization_from(config),
                dtype=get_dtype_from_config(config),
                source_file=source,
            )
    except (KeyError, TypeError):
        pass
    return None


# Used only after JSON and the ONNX graph both omit a KV dimension.
# ``require_all_fields=True`` refuses these instead of building the wrong cache.
_DEFAULT_NUM_KV_HEADS = 8
_DEFAULT_HEAD_DIM = 128
_LAYER_NAME = re.compile(r"^layer_(\d+)(?:_of_\d+)?$")
_LAYER_FILE = re.compile(r"^layer_(\d+)(?:_of_\d+)?\.onnx$")
_EXTERNAL_LOCATION = b"\x0a\x08location\x12"


def _as_dim(value: Any) -> Optional[int]:
    """Return an integer dimension, or None when the field was absent."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"expected an integer dimension, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"expected an integer dimension, got {value!r}")
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        sign = text[1:] if text.startswith("-") else text
        if sign.isdigit():
            return int(text)
    raise ValueError(f"expected an integer dimension, got {value!r}")


# First positive value wins. ``max_position_embeddings`` is the Hugging Face name.
_MAX_SEQ_LEN_KEYS = (
    "max_seq_len",
    "max_position_embeddings",
    "max_sequence_length",
    "seq_length",
    "n_positions",
)


def _max_seq_len_from(*sources: Any) -> int:
    """Read a static context length. Missing and zero values stay unset."""
    for key in _MAX_SEQ_LEN_KEYS:
        for source in sources:
            if not isinstance(source, dict) or key not in source or source[key] is None:
                continue
            try:
                value = _as_dim(source[key])
            except ValueError as exc:
                raise ValueError(f"{key} must be an integer, got {source[key]!r}") from exc
            if value is None or value == 0:
                continue
            if value < 0:
                raise ValueError(f"{key} must be non-negative, got {value}")
            return value
    return 0


# Short names match the dtype policy used for NPU planning.
_QUANT_DTYPE_ALIASES = {
    "float16": "fp16",
    "fp16": "fp16",
    "half": "fp16",
    "float32": "fp32",
    "fp32": "fp32",
    "float": "fp32",
    "bfloat16": "bf16",
    "bf16": "bf16",
    "int4": "int4",
    "uint4": "uint4",
    "int8": "int8",
    "uint8": "uint8",
    "int4x2": "int4",
    "float8": "fp8",
    "fp8": "fp8",
    "float8_e4m3fn": "fp8",
    "nf4": "nf4",
    "fp4": "fp4",
}
_QUANT_METHOD_ALIASES = {
    "matmulnbits": "matmul_nbits",
    "matmul_nbits": "matmul_nbits",
    "bitsandbytes": "bitsandbytes",
    "bnb": "bitsandbytes",
}
_QUANT_FORMAT_ALIASES = {
    "matmulnbits": "matmul_nbits",
    "matmul_nbits": "matmul_nbits",
    "qdq": "qdq",
    "qoperator": "qoperator",
    "gemm": "gemm",
    "gemv": "gemv",
    "blocked": "blocked",
    "awq": "awq",
    "gptq": "gptq",
}
_QUANT_NESTED_KEYS = ("quantization_config", "quantization", "quant_config")


def _canon_quant_token(value: Any, aliases: Dict[str, str]) -> Optional[str]:
    """Return a lowercase planning name. Empty and structured values are absent."""
    if value is None or isinstance(value, (dict, list, tuple, bool)):
        return None
    text = str(value).strip().lower().replace(" ", "").replace("-", "_")
    if text.startswith("torch."):
        text = text[6:]
    if not text:
        return None
    return aliases.get(text, text)


def _quant_blocks(*sources: Any) -> list:
    """Nested quantization objects, in source order. The root dict is not one."""
    blocks = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in _QUANT_NESTED_KEYS:
            nested = source.get(key)
            if not isinstance(nested, dict):
                continue
            blocks.append(nested)
            global_config = nested.get("global_quant_config")
            if isinstance(global_config, dict):
                blocks.append(global_config)
    return blocks


def _first_present(dicts: list, keys: tuple) -> tuple:
    """Return ``(key, value)`` for the first non-empty field, else ``(None, None)``."""
    for item in dicts:
        if not isinstance(item, dict):
            continue
        for key in keys:
            if key not in item or item[key] is None:
                continue
            if isinstance(item[key], str) and not item[key].strip():
                continue
            return key, item[key]
    return None, None


def _child_dict(block: Dict[str, Any], names: tuple) -> Optional[Dict[str, Any]]:
    for name in names:
        child = block.get(name)
        if isinstance(child, dict):
            return child
    return None


def _group_size_from(dicts: list, keys: tuple) -> Optional[int]:
    """First positive group size, or -1 for per-channel. Zero is skipped."""
    for item in dicts:
        if not isinstance(item, dict):
            continue
        for key in keys:
            if key not in item or item[key] is None:
                continue
            if isinstance(item[key], str) and not item[key].strip():
                continue
            value = _group_size_value(key, item[key])
            if value is not None:
                return value
    return None


def _group_size_value(key: str, raw: Any) -> Optional[int]:
    """Parse one group size. Zero falls through. -1 is per-channel."""
    try:
        value = _as_dim(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc
    if value is None or value == 0:
        return None
    if value < -1:
        raise ValueError(
            f"{key} must be positive or -1 for per-channel, got {value}"
        )
    return value


def _weight_dtype_from_bits(blocks: list) -> Optional[str]:
    """Map a nested ``bits`` field to int4 or int8. Unsigned flags select uint."""
    key, raw = _first_present(blocks, ("bits",))
    if key is None:
        return None
    try:
        bits = _as_dim(raw)
    except ValueError as exc:
        raise ValueError(f"bits must be an integer, got {raw!r}") from exc
    if bits is None or bits <= 0:
        return None
    unsigned = False
    for block in blocks:
        for flag in ("signed", "is_signed"):
            if isinstance(block.get(flag), bool) and block[flag] is False:
                unsigned = True
    if bits in (4, 8):
        return f"{'uint' if unsigned else 'int'}{bits}"
    return None


def _quantization_from(*sources: Any) -> Dict[str, Any]:
    """Read NPU quantization metadata. A config with no scheme stays unset.

    Hugging Face ``quantization_config`` supplies AWQ, GPTQ, and Quark.
    GenAI and ``model_info.json`` supply the same names, plus a nested
    ``quantization`` object used by MatMulNBits exports. ``torch_dtype``
    becomes the activation dtype only after a quantization field is present.
    """
    roots = [source for source in sources if isinstance(source, dict)]
    blocks = _quant_blocks(*sources)
    searchable = roots + blocks

    method_key, method_raw = _first_present(
        searchable, ("quantization_method",)
    )
    if method_key is None:
        method_key, method_raw = _first_present(
            blocks, ("quant_method", "method")
        )
    method = _canon_quant_token(method_raw, _QUANT_METHOD_ALIASES)

    weight_key, weight_raw = _first_present(
        searchable, ("weight_dtype", "weights_dtype")
    )
    weight = _canon_quant_token(weight_raw, _QUANT_DTYPE_ALIASES)
    if weight is None:
        for block in blocks:
            child = _child_dict(block, ("weight", "weights"))
            if child is None:
                continue
            _, child_raw = _first_present((child,), ("dtype", "weight_dtype"))
            weight = _canon_quant_token(child_raw, _QUANT_DTYPE_ALIASES)
            if weight is not None:
                break
    if weight is None:
        weight = _weight_dtype_from_bits(blocks)

    group = _group_size_from(searchable, ("quantization_group_size",))
    if group is None:
        group = _group_size_from(blocks, ("group_size", "block_size"))
    if group is None:
        for block in blocks:
            child = _child_dict(block, ("weight", "weights"))
            if child is None:
                continue
            for key in ("group_size", "block_size", "quantization_group_size"):
                if key not in child or child[key] is None:
                    continue
                group = _group_size_value(key, child[key])
                if group is not None:
                    break
            if group is not None:
                break

    format_key, format_raw = _first_present(searchable, ("quantization_format",))
    if format_key is None:
        format_key, format_raw = _first_present(
            blocks, ("quant_format", "format", "layout")
        )
    quant_format = _canon_quant_token(format_raw, _QUANT_FORMAT_ALIASES)
    if quant_format is None:
        for block in blocks:
            version = block.get("version")
            if isinstance(version, str) and version.strip().lower() in {"gemm", "gemv"}:
                quant_format = version.strip().lower()
                break

    zero_point = None
    zp_key, zp_raw = _first_present(searchable, ("has_zero_point",))
    if zp_key is not None:
        if not isinstance(zp_raw, bool):
            raise ValueError(f"has_zero_point must be a boolean, got {zp_raw!r}")
        zero_point = zp_raw
    if zero_point is None:
        zp_key, zp_raw = _first_present(blocks, ("zero_point",))
        if zp_key is not None:
            zero_point = zp_raw if isinstance(zp_raw, bool) else bool(zp_raw)
    if zero_point is None:
        for block in blocks:
            child = _child_dict(block, ("weight", "weights"))
            if child is None or "zero_point" not in child or child["zero_point"] is None:
                continue
            raw_child = child["zero_point"]
            zero_point = raw_child if isinstance(raw_child, bool) else bool(raw_child)
            break
    if zero_point is None:
        for block in blocks:
            for key in ("sym", "symmetric", "is_symmetric"):
                if isinstance(block.get(key), bool):
                    zero_point = not block[key]
                    break
            if zero_point is not None:
                break

    activation_key, activation_raw = _first_present(
        searchable, ("activation_dtype", "act_dtype")
    )
    activation = _canon_quant_token(activation_raw, _QUANT_DTYPE_ALIASES)
    if activation is None:
        for block in blocks:
            child = _child_dict(block, ("activation", "activations"))
            if child is None:
                continue
            _, child_raw = _first_present((child,), ("dtype", "activation_dtype"))
            activation = _canon_quant_token(child_raw, _QUANT_DTYPE_ALIASES)
            if activation is not None:
                break
    if activation is None:
        _, compute_raw = _first_present(blocks, ("bnb_4bit_compute_dtype", "compute_dtype"))
        activation = _canon_quant_token(compute_raw, _QUANT_DTYPE_ALIASES)

    quantized = any(
        value is not None
        for value in (method, weight, group, quant_format, zero_point, activation)
    )
    if activation is None and quantized:
        _, torch_raw = _first_present(roots, ("torch_dtype",))
        activation = _canon_quant_token(torch_raw, _QUANT_DTYPE_ALIASES)

    return {
        "quantization_method": method,
        "weight_dtype": weight,
        "activation_dtype": activation,
        "quantization_group_size": group,
        "quantization_format": quant_format,
        "has_zero_point": zero_point,
    }


def _read_field(sources: tuple, keys: tuple) -> Any:
    """Return the first present non-null value for ``keys``."""
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            if key not in source or source[key] is None:
                continue
            return source[key]
    return None


def _concrete_axis(dim: Any) -> Optional[int]:
    """Return a positive graph dimension. Dynamic names and zeros stay unknown."""
    if dim is None or isinstance(dim, (bool, str)):
        return None
    try:
        number = int(dim)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number


def _is_past_key_input(name: str) -> bool:
    """True for a past-key input, excluding the paired past-value tensor."""
    token = str(name).lower()
    if ".value" in token:
        return False
    return "past" in token and "key" in token


def _kv_shape_from_session_inputs(inputs: Any) -> Optional[tuple]:
    """Read ``[batch, num_kv_heads, seq, head_dim]`` from past-key inputs.

    A symbolic axis stays unknown. A concrete axis is kept, so a graph can
    supply ``num_kv_heads`` while ``head_dim`` is still dynamic.
    """
    heads = None
    head_dim = None
    for item in inputs:
        name = getattr(item, "name", "")
        if not _is_past_key_input(name):
            continue
        shape = tuple(getattr(item, "shape", ()) or ())
        if len(shape) != 4:
            continue
        this_heads = _concrete_axis(shape[1])
        this_dim = _concrete_axis(shape[3])
        if this_heads is None and this_dim is None:
            continue
        if this_heads is not None:
            if heads is None:
                heads = this_heads
            elif heads != this_heads:
                warnings.warn(
                    f"past-key input {name} has num_kv_heads={this_heads}, "
                    f"which disagrees with num_kv_heads={heads}. "
                    f"Using the first past-key shape.",
                    UserWarning,
                )
        if this_dim is not None:
            if head_dim is None:
                head_dim = this_dim
            elif head_dim != this_dim:
                warnings.warn(
                    f"past-key input {name} has head_dim={this_dim}, "
                    f"which disagrees with head_dim={head_dim}. "
                    f"Using the first past-key shape.",
                    UserWarning,
                )
    if heads is None and head_dim is None:
        return None
    return heads, head_dim


def _onnx_candidates(model_dir: str, config: Dict[str, Any]) -> list:
    """ONNX files declared by model_info, then a monolithic model.onnx."""
    paths = []
    components = config.get("components")
    if isinstance(components, dict):
        for comp in components.values():
            if not isinstance(comp, dict):
                continue
            filename = comp.get("file")
            if not isinstance(filename, str) or not filename:
                continue
            path = filename if os.path.isabs(filename) else os.path.join(model_dir, filename)
            paths.append(path)
    paths.append(os.path.join(model_dir, "model.onnx"))
    unique = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _open_onnx_session(ort: Any, path: str):
    """Open a graph for inspection without compiling it."""
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        path,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def _infer_kv_shape_from_onnx(
    model_dir: str,
    config: Dict[str, Any],
) -> tuple:
    """
    Infer KV heads and head_dim from the first past-key ONNX input.

    Returns ``(None, None)`` when no concrete 4D past-key shape is available.
    A graph that cannot be opened is skipped so the caller can warn or raise.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        warnings.warn(
            "onnxruntime is unavailable, so num_kv_heads and head_dim cannot be "
            "read from the ONNX graph.",
            UserWarning,
        )
        return None, None

    for path in _onnx_candidates(model_dir, config):
        if not os.path.isfile(path):
            continue
        try:
            session = _open_onnx_session(ort, path)
            found = _kv_shape_from_session_inputs(session.get_inputs())
        except Exception as exc:
            warnings.warn(
                f"Could not read ONNX inputs from {path}: {exc}. "
                f"KV shape will not be inferred from this file.",
                UserWarning,
            )
            continue
        if found is not None:
            return found
    return None, None


def _require_or_default_kv_shape(
    num_kv_heads: Optional[int],
    head_dim: Optional[int],
    source: str,
    require_all_fields: bool,
) -> tuple:
    """Raise or warn when a KV dimension is still unknown."""
    missing = []
    if num_kv_heads is None:
        missing.append("num_kv_heads")
    if head_dim is None:
        missing.append("head_dim")
    if not missing:
        return num_kv_heads, head_dim
    if require_all_fields:
        raise ValueError(
            f"{source} is missing {', '.join(missing)}. "
            f"Refusing defaults num_kv_heads={_DEFAULT_NUM_KV_HEADS} and "
            f"head_dim={_DEFAULT_HEAD_DIM}. Add the missing fields to "
            f"model_info.json or provide an ONNX past-key input shaped "
            f"[batch, num_kv_heads, seq, head_dim]."
        )
    resolved_heads = _DEFAULT_NUM_KV_HEADS if num_kv_heads is None else num_kv_heads
    resolved_dim = _DEFAULT_HEAD_DIM if head_dim is None else head_dim
    warnings.warn(
        f"{source} is missing {', '.join(missing)}. "
        f"Using explicit defaults num_kv_heads={resolved_heads}, "
        f"head_dim={resolved_dim}. These may not match the model and can "
        f"corrupt generation. Pass require_all_fields=True to raise instead.",
        UserWarning,
    )
    return resolved_heads, resolved_dim


def _parse_model_info(
    config: Dict[str, Any],
    source: str,
    model_dir: str,
    require_all_fields: bool = False,
) -> Optional[ModelConfig]:
    """Parse model_info.json from a split model directory."""
    num_layers = config.get("num_layers", 0)
    if not num_layers:
        return None
    try:
        num_layers = int(num_layers)
    except (TypeError, ValueError):
        return None

    # Splitter output stores the architecture name ("llama"), not a shape dict.
    arch = config.get("architecture")
    arch_dict = arch if isinstance(arch, dict) else {}
    sources = (arch_dict, config)
    num_kv_heads = _as_dim(_read_field(
        sources, ("num_kv_heads", "num_key_value_heads")
    ))
    head_dim = _as_dim(_read_field(sources, ("head_dim", "head_size")))
    hidden_size = _as_dim(_read_field(sources, ("hidden_size",)))
    if head_dim is None:
        attention_heads = _as_dim(_read_field(sources, ("num_attention_heads",)))
        if (
            hidden_size is not None
            and attention_heads is not None
            and hidden_size > 0
            and attention_heads > 0
        ):
            if hidden_size % attention_heads != 0:
                warnings.warn(
                    f"hidden_size ({hidden_size}) is not divisible by "
                    f"num_attention_heads ({attention_heads}). "
                    f"head_dim will not be inferred from that ratio.",
                    UserWarning,
                )
            else:
                head_dim = hidden_size // attention_heads

    if num_kv_heads is None or head_dim is None:
        inferred_heads, inferred_dim = _infer_kv_shape_from_onnx(model_dir, config)
        if num_kv_heads is None:
            num_kv_heads = inferred_heads
        if head_dim is None:
            head_dim = inferred_dim
    num_kv_heads, head_dim = _require_or_default_kv_shape(
        num_kv_heads,
        head_dim,
        source,
        require_all_fields,
    )
    return ModelConfig(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size or 0,
        num_attention_heads=_as_dim(_read_field(sources, ("num_attention_heads",))) or 0,
        max_seq_len=_max_seq_len_from(*sources),
        **_quantization_from(*sources),
        dtype=get_dtype_from_config(config),
        model_type=str(config.get("strategy", config.get("model_type", "unknown"))),
        source_file=source,
    )


def load_model_config(
    model_dir: str,
    tokenizer_path: Optional[str] = None,
    require_all_fields: bool = False
) -> ModelConfig:
    """
    Load model configuration from available config files.
    
    Searches for configuration in order:
    1. genai_config.json (ONNX GenAI format)
    2. config.json (HuggingFace format)
    3. model_info.json (split model format)
    
    ``model_info.json`` often records only ``num_layers`` and an architecture
    name. Missing ``num_kv_heads`` or ``head_dim`` are read from a past-key
    ONNX input shaped ``[batch, num_kv_heads, seq, head_dim]``. When that
    shape is also unavailable, ``require_all_fields=True`` raises. Otherwise
    the loader warns and uses the explicit defaults 8 and 128.

    Quantization fields stay unset unless the file names a scheme such as
    AWQ, GPTQ, Quark, or MatMulNBits. ``quantization_group_size`` -1 means
    per-channel weights. ``dtype`` is read from ``torch_dtype``, ``dtype``,
    or ``model.decoder.dtype``. A file that names none of them stays float16.
    
    Args:
        model_dir: Directory containing model/config files
        tokenizer_path: Optional alternative path for config files
        require_all_fields: If True, raise if critical KV fields are missing
        
    Returns:
        ModelConfig with extracted parameters
        
    Raises:
        FileNotFoundError: If no config file found
        ValueError: If require_all_fields and critical fields are missing,
            or a present dimension is not a positive integer
    """
    search_dirs = [model_dir]
    if tokenizer_path:
        search_dirs.append(tokenizer_path)
    
    # Try each config file type
    for search_dir in search_dirs:
        # 1. Try genai_config.json
        genai_path = os.path.join(search_dir, "genai_config.json")
        if os.path.exists(genai_path):
            data = _load_json_safe(genai_path)
            if data:
                config = _parse_genai_config(data, genai_path)
                if config:
                    config.validate()
                    return config
        
        # 2. Try config.json
        hf_path = os.path.join(search_dir, "config.json")
        if os.path.exists(hf_path):
            data = _load_json_safe(hf_path)
            if data:
                config = _parse_hf_config(data, hf_path)
                if config:
                    config.validate()
                    return config
    
    # 3. Try model_info.json in model_dir
    info_path = os.path.join(model_dir, "model_info.json")
    if os.path.exists(info_path):
        data = _load_json_safe(info_path)
        if data:
            config = _parse_model_info(
                data,
                info_path,
                model_dir,
                require_all_fields=require_all_fields,
            )
            if config:
                config.validate()
                return config
    
    raise FileNotFoundError(
        f"No valid config found. Searched: genai_config.json, config.json, model_info.json "
        f"in directories: {search_dirs}"
    )


def _read_varint(data: bytes, pos: int) -> tuple:
    """Read one protobuf varint. Returns ``(None, pos)`` when it does not fit."""
    value = 0
    shift = 0
    while pos < len(data) and shift <= 63:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
    return None, pos


def _external_data_locations(path: str) -> list:
    """Return ONNX external-data ``location`` strings stored in ``path``."""
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError:
        return []
    refs = []
    start = 0
    while True:
        index = data.find(_EXTERNAL_LOCATION, start)
        if index < 0:
            break
        pos = index + len(_EXTERNAL_LOCATION)
        length, pos = _read_varint(data, pos)
        if length is None or length < 0 or pos + length > len(data):
            start = index + 1
            continue
        try:
            location = data[pos:pos + length].decode("utf-8")
        except UnicodeDecodeError:
            start = pos + length
            continue
        if location and location not in refs:
            refs.append(location)
        start = pos + length
    return refs


def _layer_index(name: str, comp: Dict[str, Any]) -> Optional[int]:
    """Return a layer component index, or None when ``comp`` is not a layer."""
    is_layer = comp.get("type") == "layer" or _LAYER_NAME.match(str(name)) is not None
    if not is_layer:
        return None
    layer_id = comp.get("layer_id")
    if layer_id is not None:
        try:
            return int(layer_id)
        except (TypeError, ValueError):
            pass
    match = _LAYER_NAME.match(str(name))
    if match:
        return int(match.group(1))
    numbers = re.findall(r"(\d+)", str(name))
    if numbers:
        return int(numbers[-1])
    return None


def _layer_components(model_dir: str, info: Dict[str, Any]) -> list:
    """Layer components from model_info plus any layer_*.onnx files on disk."""
    layers = []
    declared_paths = set()
    components = info.get("components") if isinstance(info, dict) else None
    if isinstance(components, dict):
        pending = []
        for name, comp in components.items():
            if not isinstance(comp, dict):
                continue
            index = _layer_index(str(name), comp)
            if index is None and comp.get("type") == "layer":
                pending.append((str(name), comp))
                continue
            if index is None:
                continue
            filename = comp.get("file") or f"{name}.onnx"
            path = filename if os.path.isabs(str(filename)) else os.path.join(model_dir, str(filename))
            layers.append({"name": str(name), "index": index, "path": path})
            declared_paths.add(os.path.abspath(path))
        next_index = max((item["index"] for item in layers), default=-1) + 1
        for name, comp in pending:
            filename = comp.get("file") or f"{name}.onnx"
            path = filename if os.path.isabs(str(filename)) else os.path.join(model_dir, str(filename))
            layers.append({"name": name, "index": next_index, "path": path})
            declared_paths.add(os.path.abspath(path))
            next_index += 1
    if os.path.isdir(model_dir):
        for filename in sorted(os.listdir(model_dir)):
            match = _LAYER_FILE.match(filename)
            if not match:
                continue
            path = os.path.join(model_dir, filename)
            if os.path.abspath(path) in declared_paths:
                continue
            layers.append({
                "name": f"layer_{int(match.group(1))}",
                "index": int(match.group(1)),
                "path": path,
            })
    layers.sort(key=lambda item: (item["index"], item["name"]))
    return layers


def _component_onnx_files(model_dir: str, info: Dict[str, Any], layers: list) -> list:
    """ONNX paths to inspect. A monolithic model.onnx is used when no layer file exists."""
    paths = []
    components = info.get("components") if isinstance(info, dict) else None
    if isinstance(components, dict):
        for comp in components.values():
            if not isinstance(comp, dict):
                continue
            filename = comp.get("file")
            if not isinstance(filename, str) or not filename:
                continue
            path = filename if os.path.isabs(filename) else os.path.join(model_dir, filename)
            paths.append(path)
    for layer in layers:
        paths.append(layer["path"])
    if not layers:
        paths.append(os.path.join(model_dir, "model.onnx"))
    unique = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _is_past_value_input(name: str) -> bool:
    token = str(name).lower()
    return "past" in token and ".value" in token


def _is_present_key_output(name: str) -> bool:
    token = str(name).lower()
    return "present" in token and "key" in token and ".value" not in token and not token.endswith("value")


def _is_present_value_output(name: str) -> bool:
    token = str(name).lower()
    if "present" not in token:
        return False
    return ".value" in token or token.endswith("value")


def _io_record(item: Any) -> Dict[str, Any]:
    shape = getattr(item, "shape", ()) or ()
    return {
        "name": str(getattr(item, "name", "")),
        "shape": [None if dim is None else dim for dim in shape],
        "dtype": str(getattr(item, "type", "")),
    }


def _past_layer_indexes(names: list) -> set:
    indexes = set()
    for name in names:
        if not _is_past_key_input(name):
            continue
        numbers = re.findall(r"(\d+)", name)
        indexes.add(int(numbers[-1]) if numbers else 0)
    return indexes


def _numpy_dtype_from_onnx_type(type_name: str) -> np.dtype:
    from .dtypes import get_numpy_dtype
    return np.dtype(get_numpy_dtype(type_name))


def validate_config_against_onnx(
    config: ModelConfig,
    model_dir: str,
) -> Dict[str, Any]:
    """
    Check ``config`` against the ONNX graphs in ``model_dir``.

    Split layer components are the source of the layer count. Each past-key
    input is ``[batch, num_kv_heads, seq, head_dim]``. ``config.num_kv_heads``
    must match axis 1 and ``config.head_dim`` must match axis 3. ``config.dtype``
    must match that tensor's ONNX type. Present key and value outputs must
    exist beside the past inputs, and every external-data location must exist
    on disk.

    Returns the inspected names, shapes, dtype, and external-data paths.
    A mismatch raises ``ValueError`` listing every difference.
    """
    if not isinstance(config, ModelConfig):
        raise TypeError(f"config must be a ModelConfig, got {type(config).__name__}")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"model directory not found: {model_dir}")

    info_path = os.path.join(model_dir, "model_info.json")
    info = _load_json_safe(info_path) if os.path.isfile(info_path) else {}
    if not isinstance(info, dict):
        info = {}
    layers = _layer_components(model_dir, info)
    files = _component_onnx_files(model_dir, info, layers)
    existing = [path for path in files if os.path.isfile(path)]
    if not existing:
        raise ValueError(f"No ONNX graph found in {model_dir}")

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ValueError(
            "onnxruntime is unavailable, so the ONNX graph cannot be validated"
        ) from exc

    problems = []
    inputs = []
    outputs = []
    past_names = []
    present_names = []
    external = []
    graph_heads = None
    graph_dim = None
    graph_dtype = None
    graph_type_name = None
    heads_from = None
    dim_from = None
    opened = 0

    for layer in layers:
        if not os.path.isfile(layer["path"]):
            problems.append(
                f"layer component {layer['name']} is missing {os.path.basename(layer['path'])}"
            )

    for path in existing:
        for location in _external_data_locations(path):
            resolved = location if os.path.isabs(location) else os.path.join(
                os.path.dirname(path), location
            )
            if os.path.isfile(resolved):
                if location not in external:
                    external.append(location)
            else:
                problems.append(
                    f"missing external data {location} referenced by {os.path.basename(path)}"
                )
        try:
            session = _open_onnx_session(ort, path)
        except Exception as exc:
            problems.append(
                f"could not read ONNX graph {os.path.basename(path)}: {exc}"
            )
            continue
        opened += 1
        graph_inputs = [_io_record(item) for item in session.get_inputs()]
        graph_outputs = [_io_record(item) for item in session.get_outputs()]
        inputs.extend(graph_inputs)
        outputs.extend(graph_outputs)
        input_names = [item["name"] for item in graph_inputs]
        output_names = [item["name"] for item in graph_outputs]
        has_past_key = any(_is_past_key_input(name) for name in input_names)
        has_past_value = any(_is_past_value_input(name) for name in input_names)
        has_present_key = any(_is_present_key_output(name) for name in output_names)
        has_present_value = any(_is_present_value_output(name) for name in output_names)
        if has_past_key and not has_present_key:
            problems.append(
                f"{os.path.basename(path)} has a past key input and no present key output"
            )
        if has_past_value and not has_present_value:
            problems.append(
                f"{os.path.basename(path)} has a past value input and no present value output"
            )
        for name in input_names:
            if _is_past_key_input(name) or _is_past_value_input(name):
                if name not in past_names:
                    past_names.append(name)
        for name in output_names:
            if _is_present_key_output(name) or _is_present_value_output(name):
                if name not in present_names:
                    present_names.append(name)
        for record in graph_inputs:
            name = record["name"]
            is_key = _is_past_key_input(name)
            is_value = _is_past_value_input(name)
            if not is_key and not is_value:
                continue
            if is_key:
                shape = record["shape"]
                if len(shape) != 4:
                    problems.append(
                        f"{name} rank {len(shape)} != 4; expected "
                        f"[batch, num_kv_heads, seq, head_dim]"
                    )
                else:
                    heads = _concrete_axis(shape[1])
                    head_dim = _concrete_axis(shape[3])
                    if heads is None:
                        problems.append(f"num_kv_heads is symbolic on {name}")
                    elif graph_heads is None:
                        graph_heads = heads
                        heads_from = name
                    elif heads != graph_heads:
                        problems.append(
                            f"num_kv_heads {heads} from {name} != {graph_heads} from {heads_from}"
                        )
                    if head_dim is None:
                        problems.append(f"head_dim is symbolic on {name}")
                    elif graph_dim is None:
                        graph_dim = head_dim
                        dim_from = name
                    elif head_dim != graph_dim:
                        problems.append(
                            f"head_dim {head_dim} from {name} != {graph_dim} from {dim_from}"
                        )
            type_name = record["dtype"]
            if not type_name:
                problems.append(f"{name} has no ONNX tensor dtype")
                continue
            try:
                numpy_dtype = _numpy_dtype_from_onnx_type(type_name)
            except ValueError:
                problems.append(f"unrecognized ONNX tensor dtype {type_name} on {name}")
                continue
            if graph_dtype is None:
                graph_dtype = numpy_dtype
                graph_type_name = type_name
            elif numpy_dtype != graph_dtype:
                problems.append(
                    f"dtype {type_name} on {name} != {graph_type_name}"
                )

    if layers:
        observed_layers = len(layers)
        layer_names = [item["name"] for item in layers]
    else:
        observed_layers = len(_past_layer_indexes(past_names))
        layer_names = [f"past_key_{index}" for index in sorted(_past_layer_indexes(past_names))]
    if int(config.num_layers) != observed_layers:
        problems.append(
            f"num_layers {config.num_layers} != {observed_layers} layer components"
            + (f" ({', '.join(layer_names)})" if layer_names else "")
        )
    if graph_heads is None:
        if opened and not any("symbolic" in item or "rank" in item for item in problems):
            problems.append(
                "ONNX graph has no past key input shaped "
                "[batch, num_kv_heads, seq, head_dim]"
            )
    elif int(config.num_kv_heads) != graph_heads:
        problems.append(
            f"num_kv_heads {config.num_kv_heads} != {graph_heads} from {heads_from}"
        )
    if graph_dim is not None and int(config.head_dim) != graph_dim:
        problems.append(
            f"head_dim {config.head_dim} != {graph_dim} from {dim_from}"
        )
    if graph_dtype is not None and np.dtype(config.dtype) != graph_dtype:
        problems.append(
            f"dtype {np.dtype(config.dtype)} != {graph_type_name}"
        )

    if problems:
        raise ValueError(
            f"ONNX graph does not match ModelConfig in {model_dir}: "
            + "; ".join(problems)
        )
    return {
        "num_layers": observed_layers,
        "layer_components": layer_names,
        "num_kv_heads": graph_heads,
        "head_dim": graph_dim,
        "dtype": graph_dtype,
        "onnx_dtype": graph_type_name,
        "inputs": inputs,
        "outputs": outputs,
        "past_key_values": past_names,
        "present_outputs": present_names,
        "external_data": external,
        "files": existing,
    }


# Names Hugging Face and GenAI use for the same numpy dtypes.
_CONFIG_DTYPE_ALIASES = {
    "fp16": "float16",
    "half": "float16",
    "torch.float16": "float16",
    "fp32": "float32",
    "torch.float32": "float32",
    "fp64": "float64",
    "torch.float64": "float64",
    "bf16": "bfloat16",
    "torch.bfloat16": "bfloat16",
}


def _dtype_sources(config: Any) -> list:
    """Root, then model, decoder, and an architecture dict."""
    if not isinstance(config, dict):
        return []
    sources = [config]
    model = config.get("model")
    if isinstance(model, dict):
        sources.append(model)
        nested = model.get("decoder")
        if isinstance(nested, dict):
            sources.append(nested)
    decoder = config.get("decoder")
    if isinstance(decoder, dict) and decoder not in sources:
        sources.append(decoder)
    architecture = config.get("architecture")
    if isinstance(architecture, dict):
        sources.append(architecture)
    return sources


def _config_dtype_name(value: Any) -> Optional[str]:
    """Return a dtype string ``get_numpy_dtype`` can read."""
    if value is None or isinstance(value, (dict, list, tuple, bool)):
        return None
    text = str(value).strip()
    if not text:
        return None
    key = text.lower().replace(" ", "")
    return _CONFIG_DTYPE_ALIASES.get(key, key)


def get_dtype_from_config(config: Dict[str, Any]) -> np.dtype:
    """
    Extract the compute dtype from a config dictionary.

    ``torch_dtype`` wins over ``dtype`` on the same object. A GenAI file
    stores the value at ``model.decoder.dtype``. An unknown name is skipped.
    When every field is missing, the result is float16.

    Args:
        config: Configuration dictionary

    Returns:
        numpy dtype
    """
    from .dtypes import get_numpy_dtype

    for source in _dtype_sources(config):
        for key in ("torch_dtype", "dtype"):
            if key not in source:
                continue
            name = _config_dtype_name(source[key])
            if name is None:
                continue
            try:
                return np.dtype(get_numpy_dtype(name))
            except ValueError:
                warnings.warn(
                    f"{key} {source[key]!r} is not a recognized dtype. "
                    f"Trying the next dtype field.",
                    UserWarning,
                )
    return np.dtype(np.float16)


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

__all__ = [
    "ModelConfig",
    "load_model_config",
    "validate_config_against_onnx",
    "get_dtype_from_config",
]
