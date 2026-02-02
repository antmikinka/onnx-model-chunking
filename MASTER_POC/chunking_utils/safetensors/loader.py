"""
SafeTensors/HuggingFace model loading utilities.

This module provides utilities for loading model weights from SafeTensors
or PyTorch format, with support for both local paths and HuggingFace Hub models.

Key Features:
- stream_tensors(): Memory-efficient loading (one tensor at a time)
- load_state_dict(): Traditional full load (backward compatible)
"""

import os
import json
from typing import Dict, Iterator, Tuple, List, Optional

import torch
from safetensors import safe_open

# Optional HuggingFace Hub import
try:
    from huggingface_hub import snapshot_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False


def download_model(model_id: str) -> str:
    """
    Ensure model is available locally, downloading from Hub if needed.
    
    Args:
        model_id: Either a local path or HuggingFace model ID.
    
    Returns:
        Local path to the model directory.
    
    Raises:
        ImportError: If huggingface_hub is not installed and model needs downloading.
        FileNotFoundError: If local path doesn't exist.
    """
    if os.path.exists(model_id):
        return model_id
    
    if not HAS_HF_HUB:
        raise ImportError(
            f"Path {model_id} not found locally and huggingface_hub is not installed. "
            "Install with: pip install huggingface_hub"
        )
    
    print(f"Path {model_id} not found. Downloading from Hub...")
    return snapshot_download(model_id)


def get_safetensors_files(model_path: str) -> List[str]:
    """
    Get list of safetensors files for a model.
    
    Args:
        model_path: Path to model directory.
    
    Returns:
        List of absolute paths to safetensors files.
    """
    model_path = download_model(model_path)
    
    # Check for sharded index
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)["weight_map"]
        files = sorted(set(index.values()))
        return [os.path.join(model_path, f) for f in files]
    
    # Check for single file
    single_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single_path):
        return [single_path]
    
    return []


def stream_tensors(
    model_path: str,
    device: str = "cpu"
) -> Iterator[Tuple[str, torch.Tensor]]:
    """
    Stream tensors one at a time from safetensors files.
    
    This is the MEMORY-EFFICIENT loading method. Only ONE tensor is in memory
    at a time, allowing processing of models larger than available RAM.
    
    Args:
        model_path: Path to model directory or HuggingFace model ID.
        device: Device to load tensors to ('cpu' recommended for splitting).
    
    Yields:
        Tuple of (tensor_name, tensor) for each weight in the model.
    
    Example:
        >>> for name, tensor in stream_tensors("./model"):
        ...     print(f"{name}: {tensor.shape}")
        ...     # Tensor is garbage collected after loop iteration!
    """
    files = get_safetensors_files(model_path)
    
    if not files:
        # Fallback to pytorch_model.bin
        model_path = download_model(model_path)
        pytorch_bin = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(pytorch_bin):
            state_dict = torch.load(pytorch_bin, map_location=device)
            for key, value in state_dict.items():
                yield key, value
            return
        raise FileNotFoundError(f"No model weights found in {model_path}")
    
    for file_path in files:
        with safe_open(file_path, framework="pt", device=device) as f:
            for key in f.keys():
                yield key, f.get_tensor(key)


def load_state_dict(model_path: str) -> Dict[str, torch.Tensor]:
    """
    Load state dict from safetensors or pytorch_model.bin.
    
    Supports:
    - Sharded safetensors (model.safetensors.index.json)
    - Single safetensors file (model.safetensors)
    - PyTorch checkpoint (pytorch_model.bin)
    
    Args:
        model_path: Path to model directory or HuggingFace model ID.
    
    Returns:
        Dictionary mapping weight names to tensors.
    
    Raises:
        FileNotFoundError: If no model weights found at path.
    """
    model_path = download_model(model_path)
    state_dict: Dict[str, torch.Tensor] = {}

    # Try sharded safetensors first
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        import json
        with open(index_path, "r") as f:
            index = json.load(f)["weight_map"]
        
        files = set(index.values())
        for file in files:
            file_path = os.path.join(model_path, file)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        return state_dict

    # Try single safetensors file
    safetensors_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(safetensors_path):
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        return state_dict

    # Fallback to pytorch_model.bin
    pytorch_bin = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(pytorch_bin):
        return torch.load(pytorch_bin, map_location="cpu")

    raise FileNotFoundError(
        f"Could not find model weights in {model_path} "
        "(checked safetensors and pytorch_model.bin)"
    )


def get_state_dict_info(state_dict: Dict[str, torch.Tensor]) -> Dict[str, any]:
    """
    Get summary information about a state dict.
    
    Args:
        state_dict: Dictionary of weight tensors.
    
    Returns:
        Dictionary with summary statistics.
    """
    total_params = 0
    total_bytes = 0
    dtypes = set()
    
    for name, tensor in state_dict.items():
        total_params += tensor.numel()
        total_bytes += tensor.numel() * tensor.element_size()
        dtypes.add(str(tensor.dtype))
    
    return {
        "num_tensors": len(state_dict),
        "total_params": total_params,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "dtypes": list(dtypes),
    }
