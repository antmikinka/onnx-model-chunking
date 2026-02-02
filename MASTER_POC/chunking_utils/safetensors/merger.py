"""
SafeTensors model merging utilities.

This module provides utilities for merging split safetensors files
back into a single state dict for inference.
"""

import os
from typing import Dict, Set

import torch
from safetensors import safe_open


# Files to exclude when merging (generated artifacts, not weights)
EXCLUDE_FILES: Set[str] = {
    "kv_cache.safetensors",
    "logits.safetensors",
}


def merge_safetensors(output_dir: str) -> Dict[str, torch.Tensor]:
    """
    Merge all layer/embedding/norm/lm_head safetensors into a single state dict.
    
    This is the inverse operation of split_safetensors, combining all the
    per-component files back into a unified state dict that can be loaded
    into a model.
    
    Args:
        output_dir: Directory containing the split safetensors files.
    
    Returns:
        Dictionary mapping weight names to tensors.
    
    Example:
        >>> state_dict = merge_safetensors("./output/Qwen--Qwen2.5-0.5B")
        >>> model.load_state_dict(state_dict)
    """
    state_dict: Dict[str, torch.Tensor] = {}

    for file in sorted(os.listdir(output_dir)):
        if file.endswith(".safetensors") and file not in EXCLUDE_FILES:
            path = os.path.join(output_dir, file)
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

    return state_dict


def merge_safetensors_to_file(
    output_dir: str,
    output_file: str,
    exclude_files: Set[str] = None,
) -> str:
    """
    Merge split safetensors files and save to a single file.
    
    Args:
        output_dir: Directory containing the split safetensors files.
        output_file: Path to save the merged safetensors file.
        exclude_files: Additional files to exclude from merging.
    
    Returns:
        Path to the saved merged file.
    """
    from safetensors.torch import save_file
    
    exclude = EXCLUDE_FILES.copy()
    if exclude_files:
        exclude.update(exclude_files)
    
    state_dict: Dict[str, torch.Tensor] = {}

    for file in sorted(os.listdir(output_dir)):
        if file.endswith(".safetensors") and file not in exclude:
            path = os.path.join(output_dir, file)
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

    save_file(state_dict, output_file)
    return output_file


def get_merge_info(output_dir: str) -> Dict[str, any]:
    """
    Get information about what would be merged from a directory.
    
    Args:
        output_dir: Directory containing the split safetensors files.
    
    Returns:
        Dictionary with merge statistics.
    """
    files_to_merge = []
    total_tensors = 0
    
    for file in sorted(os.listdir(output_dir)):
        if file.endswith(".safetensors") and file not in EXCLUDE_FILES:
            path = os.path.join(output_dir, file)
            with safe_open(path, framework="pt", device="cpu") as f:
                num_tensors = len(list(f.keys()))
                files_to_merge.append({
                    "file": file,
                    "num_tensors": num_tensors,
                })
                total_tensors += num_tensors
    
    return {
        "num_files": len(files_to_merge),
        "total_tensors": total_tensors,
        "files": files_to_merge,
    }
