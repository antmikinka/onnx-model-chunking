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
    dtype: np.dtype = field(default_factory=lambda: np.float16)
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
        
        if num_layers and num_kv_heads and head_size:
            return ModelConfig(
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_size,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                vocab_size=config.get("model", {}).get("vocab_size", 0),
                model_type=config.get("model", {}).get("type", "unknown"),
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
                source_file=source,
            )
    except (KeyError, TypeError):
        pass
    return None


# Used only after JSON and the ONNX graph both omit a KV dimension.
# ``require_all_fields=True`` refuses these instead of building the wrong cache.
_DEFAULT_NUM_KV_HEADS = 8
_DEFAULT_HEAD_DIM = 128


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
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            session = ort.InferenceSession(
                path,
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
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


def get_dtype_from_config(config: Dict[str, Any]) -> np.dtype:
    """
    Extract dtype from config dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        numpy dtype
    """
    from .dtypes import get_numpy_dtype
    
    # Check common dtype fields
    dtype_str = config.get("torch_dtype", "")
    if not dtype_str:
        dtype_str = config.get("dtype", "")
    if not dtype_str:
        dtype_str = config.get("model", {}).get("decoder", {}).get("dtype", "")
    
    if dtype_str:
        try:
            return get_numpy_dtype(dtype_str)
        except ValueError:
            pass
    
    return np.float16  # Default


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

__all__ = [
    "ModelConfig",
    "load_model_config",
    "get_dtype_from_config",
]
