"""
SafeTensors model splitting utilities.

This module provides the main splitting functionality for HuggingFace/SafeTensors
models, creating per-layer safetensors files for embeddings, layers, norm, and lm_head.
"""

import os
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer

from ..common.path_utils import is_local_path, sanitize_model_name
from ..common.architecture import (
    ArchitectureType,
    ArchitecturePrefixes,
    SAFETENSORS_PREFIXES,
    architecture_from_string,
    classify_qwen_family,
    get_prefixes,
)
from ..common.types import SplitConfig, SplitResult, ModelFormat

from .loader import load_state_dict
from .config import ensure_complete_config


def detect_architecture_from_weights(state_dict: Dict[str, torch.Tensor]) -> ArchitectureType:
    """
    Detect model architecture by examining weight key patterns.
    
    Args:
        state_dict: Dictionary of model weights.
    
    Returns:
        Detected ArchitectureType.
    
    Raises:
        ValueError: If architecture cannot be detected.
    """
    keys = list(state_dict.keys())
    sample_keys = keys[:50]  # Check first 50 keys
    
    # Check for common architecture patterns
    if any("model.layers." in k for k in sample_keys):
        variant = classify_qwen_family(keys)
        if variant is not None:
            return variant
        return ArchitectureType.LLAMA  # Generic LLaMA-style architecture
    elif any("transformer.h." in k for k in sample_keys):
        return ArchitectureType.GPT2
    
    raise ValueError(
        "Could not detect model architecture from weight keys. "
        "Please ensure your model has a valid config.json with 'model_type' field."
    )


def get_model_prefixes(config) -> Tuple[str, str, str, str]:
    """
    Return weight prefixes for different model architectures.
    
    Args:
        config: HuggingFace model config object.
    
    Returns:
        Tuple of (embed_prefix, layer_prefix, norm_prefix, lm_head_prefix).
    
    Raises:
        ValueError: If model type is not supported.
    """
    model_type = getattr(config, 'model_type', None)
    arch = architecture_from_string(model_type) if model_type else ArchitectureType.UNKNOWN
    
    prefixes = get_prefixes(arch, "safetensors")
    return (
        prefixes.embed_prefix,
        prefixes.layer_prefix,
        prefixes.norm_prefix,
        prefixes.lm_head_prefix,
    )


def split_safetensors(
    model: Optional[str] = None,
    output: Optional[str] = None,
    config: Optional[SplitConfig] = None,
    split_only: bool = False,
    **kwargs
) -> SplitResult:
    """
    Split a HuggingFace model into per-layer safetensors files.
    
    Creates separate files for:
    - embeddings.safetensors
    - layer_0_of_N.safetensors, layer_1_of_N.safetensors, ...
    - norm.safetensors
    - lm_head.safetensors
    
    Args:
        model: Path to model or HuggingFace model ID. Required if config not provided.
        output: Output directory path. Required if config not provided.
        config: Optional SplitConfig object (overrides model/output if provided).
        split_only: If True, skip inference validation.
        **kwargs: Additional arguments passed to SplitConfig.
    
    Returns:
        SplitResult with information about the split operation.
    
    Example:
        >>> result = split_safetensors(
        ...     model="Qwen/Qwen2.5-0.5B",
        ...     output="./output"
        ... )
        >>> print(f"Split into {result.num_layers} layers")
    """
    # Build config from arguments
    if config is None:
        if model is None or output is None:
            raise ValueError("Either 'config' or both 'model' and 'output' must be provided")
        config = SplitConfig(
            model=model,
            output=output,
            split_only=split_only,
            **kwargs
        )
    
    model_path = config.model
    
    # For local paths, ensure config is complete before loading
    if is_local_path(model_path) and os.path.exists(model_path):
        print(f"Detected local model path: {model_path}")
        ensure_complete_config(model_path)
    
    # Load config, tokenizer, and weights
    state_dict = None
    try:
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Could not load config normally: {e}")
        print("Attempting to load weights first for architecture detection...")
        state_dict = load_state_dict(model_path)
        detected_arch = detect_architecture_from_weights(state_dict)
        print(f"Detected architecture: {detected_arch.value}")
        # Try loading config again
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # Load tokenizer with fallback
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        print("Splitting will continue without tokenizer save.")
    
    # Load state dict if not already loaded
    if state_dict is None:
        state_dict = load_state_dict(model_path)

    # Create correlative output folder
    model_folder_name = sanitize_model_name(config.model)
    output_dir = os.path.join(config.output, model_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save config and tokenizer
    hf_config.save_pretrained(output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

    # Get prefixes for this architecture
    embed_prefix, layer_prefix, norm_prefix, lm_prefix = get_model_prefixes(hf_config)
    num_layers = hf_config.num_hidden_layers

    print(f"Splitting model into {num_layers} layer files + embeddings/norm/lm_head...")

    component_files: Dict[str, str] = {}

    # Save embeddings
    embed_keys = {k: v for k, v in state_dict.items() if k.startswith(embed_prefix)}
    if embed_keys:
        embed_path = os.path.join(output_dir, "embeddings.safetensors")
        save_file(embed_keys, embed_path)
        component_files["embedding"] = embed_path
        print(f"  Saved embeddings.safetensors ({len(embed_keys)} tensors)")

    # Save each layer individually
    for i in range(num_layers):
        layer_keys = {
            k: v for k, v in state_dict.items()
            if k.startswith(f"{layer_prefix}{i}.")
        }
        if layer_keys:
            filename = f"layer_{i}_of_{num_layers}.safetensors"
            filepath = os.path.join(output_dir, filename)
            save_file(layer_keys, filepath)
            component_files[f"layer_{i}"] = filepath
            print(f"  Saved {filename} ({len(layer_keys)} tensors)")

    # Save norm
    norm_keys = {k: v for k, v in state_dict.items() if k.startswith(norm_prefix)}
    if norm_keys:
        norm_path = os.path.join(output_dir, "norm.safetensors")
        save_file(norm_keys, norm_path)
        component_files["norm"] = norm_path
        print(f"  Saved norm.safetensors ({len(norm_keys)} tensors)")

    # Save LM head
    lm_keys = {k: v for k, v in state_dict.items() if k.startswith(lm_prefix)}
    if lm_keys:
        lm_path = os.path.join(output_dir, "lm_head.safetensors")
        save_file(lm_keys, lm_path)
        component_files["lm_head"] = lm_path
        print(f"  Saved lm_head.safetensors ({len(lm_keys)} tensors)")

    print(f"Model split complete. Output: {output_dir}")

    # Build model info
    model_info = {
        "original_model": str(model_path),
        "num_layers": num_layers,
        "num_components": len(component_files),
        "model_type": getattr(hf_config, 'model_type', 'unknown'),
        "hidden_size": getattr(hf_config, 'hidden_size', None),
        "vocab_size": getattr(hf_config, 'vocab_size', None),
    }

    return SplitResult(
        output_dir=output_dir,
        num_layers=num_layers,
        component_files=component_files,
        model_info=model_info,
        format=ModelFormat.SAFETENSORS,
    )
