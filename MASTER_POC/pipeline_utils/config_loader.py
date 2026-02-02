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


def _parse_model_info(config: Dict[str, Any], source: str) -> Optional[ModelConfig]:
    """Parse model_info.json from split model directory."""
    try:
        num_layers = config.get("num_layers", 0)
        
        # model_info may not have all details, but we can get num_layers
        if num_layers:
            # Try to get other info if available
            arch = config.get("architecture", {})
            
            # Check for missing critical fields and warn
            num_kv_heads = arch.get("num_kv_heads")
            head_dim = arch.get("head_dim")
            
            if num_kv_heads is None or head_dim is None:
                warnings.warn(
                    f"model_info.json missing architecture details. "
                    f"Using DEFAULTS: num_kv_heads={num_kv_heads or 8}, head_dim={head_dim or 128}. "
                    f"These may not match your model! Specify architecture in model_info.json.",
                    UserWarning
                )
            
            return ModelConfig(
                num_layers=num_layers,
                num_kv_heads=num_kv_heads or 8,  # Default with warning above
                head_dim=head_dim or 128,  # Default with warning above
                hidden_size=arch.get("hidden_size", 0),
                model_type=config.get("strategy", "unknown"),
                source_file=source,
            )
    except (KeyError, TypeError):
        pass
    return None


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
    
    Args:
        model_dir: Directory containing model/config files
        tokenizer_path: Optional alternative path for config files
        require_all_fields: If True, raise if critical fields are missing
        
    Returns:
        ModelConfig with extracted parameters
        
    Raises:
        FileNotFoundError: If no config file found
        ValueError: If require_all_fields and critical fields missing
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
            config = _parse_model_info(data, info_path)
            if config:
                if require_all_fields:
                    warnings.warn(
                        "model_info.json found but may have incomplete config. "
                        "Using defaults for num_kv_heads and head_dim."
                    )
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
