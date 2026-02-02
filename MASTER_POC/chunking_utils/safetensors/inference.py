"""
SafeTensors model inference utilities.

This module provides utilities for running inference on split SafeTensors models,
including loading, generation, and KV cache saving.
"""

import os
from typing import List, Optional

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from ..common.types import SplitConfig
from .merger import merge_safetensors


def infer(
    output_dir: str,
    prompt: str = "Test prompt.",
    batch_size: int = 1,
    context: int = 1024,
    max_new_tokens: int = 50,
    save_cache: bool = True,
    config: Optional[SplitConfig] = None,
) -> List[str]:
    """
    Run inference on a split model.
    
    Loads the split model components, merges them, runs generation,
    and optionally saves the KV cache and logits.
    
    Args:
        output_dir: Directory containing the split model files.
        prompt: Text prompt for generation.
        batch_size: Number of prompts to process in parallel.
        context: Maximum context length for tokenization.
        max_new_tokens: Maximum number of new tokens to generate.
        save_cache: Whether to save KV cache and logits to files.
        config: Optional SplitConfig with inference parameters.
    
    Returns:
        List of generated text strings.
    
    Example:
        >>> texts = infer("./output/Qwen--Qwen2.5-0.5B", prompt="Hello, world!")
        >>> print(texts[0])
    """
    # Use config values if provided
    if config is not None:
        prompt = config.prompt
        batch_size = config.batch_size
        context = config.context
        max_new_tokens = config.max_new_tokens
    
    # Load config and merge weights
    hf_config = AutoConfig.from_pretrained(output_dir)
    full_state = merge_safetensors(output_dir)
    
    # Create model and load weights
    model = AutoModelForCausalLM.from_config(hf_config, torch_dtype=torch.float32)
    model.load_state_dict(full_state)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(output_dir)

    # Batched prompts
    prompts = [prompt] * batch_size
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        max_length=context,
        truncation=True,
        padding=True
    )

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False
        )

    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Save KV cache and logits if requested
    if save_cache:
        with torch.no_grad():
            out = model(**inputs, use_cache=True)

        # Save past key values
        past_kv = out.past_key_values
        kv_dict = {}
        for i, (k, v) in enumerate(past_kv):
            kv_dict[f"layer_{i}.key"] = k
            kv_dict[f"layer_{i}.value"] = v
        save_file(kv_dict, os.path.join(output_dir, "kv_cache.safetensors"))

        # Save logits
        logits_dict = {"logits": out.logits}
        save_file(logits_dict, os.path.join(output_dir, "logits.safetensors"))

    return texts


def validate_split_model(output_dir: str) -> bool:
    """
    Validate that a split model can be loaded and run.
    
    Args:
        output_dir: Directory containing the split model files.
    
    Returns:
        True if model validates successfully, False otherwise.
    """
    try:
        # Try to load config
        config = AutoConfig.from_pretrained(output_dir)
        
        # Try to merge weights
        state_dict = merge_safetensors(output_dir)
        
        # Check we have weights
        if not state_dict:
            print("Error: No weights found in split model")
            return False
        
        # Try to create model
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
        
        # Try to load weights
        model.load_state_dict(state_dict)
        
        print(f"Model validation successful: {len(state_dict)} tensors loaded")
        return True
        
    except Exception as e:
        print(f"Model validation failed: {e}")
        return False
