"""
SafeTensors model chunking utilities.

This module provides utilities for splitting and merging HuggingFace/SafeTensors
models into per-layer components.

Usage:
    from chunking_utils.safetensors import split_safetensors, merge_safetensors
    
    # Split a model
    result = split_safetensors(model="Qwen/Qwen2.5-0.5B", output="./output")
    
    # Merge back
    state_dict = merge_safetensors("./output/Qwen--Qwen2.5-0.5B")
"""

from .splitter import split_safetensors, detect_architecture_from_weights
from .merger import merge_safetensors
from .loader import load_state_dict, download_model, stream_tensors, get_safetensors_files
from .config import find_complete_config, ensure_complete_config
from .inference import infer

__all__ = [
    # Main operations
    "split_safetensors",
    "merge_safetensors",
    # Loader (memory-efficient)
    "load_state_dict",
    "download_model",
    "stream_tensors",
    "get_safetensors_files",
    # Config utilities
    "find_complete_config",
    "ensure_complete_config",
    # Architecture detection
    "detect_architecture_from_weights",
    # Inference
    "infer",
]
