"""
Architecture definitions and prefix mappings for model chunking.

This module defines supported model architectures and their corresponding
weight/node name prefixes for both SafeTensors and ONNX formats.

To add support for a new architecture:
1. Add it to ArchitectureType enum
2. Add prefix mappings to SAFETENSORS_PREFIXES and ONNX_PREFIXES
3. Add to LLAMA_STYLE_ARCHITECTURES if it follows LLaMA conventions
"""

from enum import Enum
from typing import Dict, Tuple
from dataclasses import dataclass


class ArchitectureType(str, Enum):
    """
    Supported model architectures.
    
    The string value corresponds to the 'model_type' field in config.json.
    """
    LLAMA = "llama"
    MISTRAL = "mistral"
    PHI = "phi"
    GEMMA = "gemma"
    QWEN = "qwen"
    QWEN2 = "qwen2"
    QWEN3 = "qwen3"
    GPT2 = "gpt2"
    UNKNOWN = "unknown"


# Architectures that follow LLaMA-style naming conventions
LLAMA_STYLE_ARCHITECTURES = {
    ArchitectureType.LLAMA,
    ArchitectureType.MISTRAL,
    ArchitectureType.PHI,
    ArchitectureType.GEMMA,
    ArchitectureType.QWEN,
    ArchitectureType.QWEN2,
    ArchitectureType.QWEN3,
}


@dataclass(frozen=True)
class ArchitecturePrefixes:
    """
    Weight/node name prefixes for a model architecture.
    
    These prefixes are used to identify and categorize model components
    (embeddings, layers, norm, lm_head) during the splitting process.
    
    Attributes:
        embed_prefix: Prefix for embedding layer weights/nodes
        layer_prefix: Prefix for transformer layer weights/nodes (followed by layer index)
        norm_prefix: Prefix for final normalization layer weights/nodes
        lm_head_prefix: Prefix for language model head weights/nodes
    """
    embed_prefix: str
    layer_prefix: str
    norm_prefix: str
    lm_head_prefix: str


# ==============================================================================
# SafeTensors Prefix Mappings (PyTorch weight key format)
# ==============================================================================

SAFETENSORS_PREFIXES: Dict[ArchitectureType, ArchitecturePrefixes] = {
    # LLaMA and LLaMA-style architectures
    ArchitectureType.LLAMA: ArchitecturePrefixes(
        embed_prefix="model.embed_tokens.",
        layer_prefix="model.layers.",
        norm_prefix="model.norm.",
        lm_head_prefix="lm_head."
    ),
    ArchitectureType.MISTRAL: ArchitecturePrefixes(
        embed_prefix="model.embed_tokens.",
        layer_prefix="model.layers.",
        norm_prefix="model.norm.",
        lm_head_prefix="lm_head."
    ),
    ArchitectureType.PHI: ArchitecturePrefixes(
        embed_prefix="model.embed_tokens.",
        layer_prefix="model.layers.",
        norm_prefix="model.norm.",
        lm_head_prefix="lm_head."
    ),
    ArchitectureType.GEMMA: ArchitecturePrefixes(
        embed_prefix="model.embed_tokens.",
        layer_prefix="model.layers.",
        norm_prefix="model.norm.",
        lm_head_prefix="lm_head."
    ),
    ArchitectureType.QWEN: ArchitecturePrefixes(
        embed_prefix="model.embed_tokens.",
        layer_prefix="model.layers.",
        norm_prefix="model.norm.",
        lm_head_prefix="lm_head."
    ),
    ArchitectureType.QWEN2: ArchitecturePrefixes(
        embed_prefix="model.embed_tokens.",
        layer_prefix="model.layers.",
        norm_prefix="model.norm.",
        lm_head_prefix="lm_head."
    ),
    ArchitectureType.QWEN3: ArchitecturePrefixes(
        embed_prefix="model.embed_tokens.",
        layer_prefix="model.layers.",
        norm_prefix="model.norm.",
        lm_head_prefix="lm_head."
    ),
    # GPT-2 style
    ArchitectureType.GPT2: ArchitecturePrefixes(
        embed_prefix="transformer.wte.",
        layer_prefix="transformer.h.",
        norm_prefix="transformer.ln_f.",
        lm_head_prefix="lm_head."
    ),
    # Default/Unknown uses LLaMA style
    ArchitectureType.UNKNOWN: ArchitecturePrefixes(
        embed_prefix="model.embed_tokens.",
        layer_prefix="model.layers.",
        norm_prefix="model.norm.",
        lm_head_prefix="lm_head."
    ),
}


# ==============================================================================
# ONNX Prefix Mappings (ONNX node name format)
# ==============================================================================

ONNX_PREFIXES: Dict[ArchitectureType, ArchitecturePrefixes] = {
    # LLaMA and LLaMA-style architectures
    ArchitectureType.LLAMA: ArchitecturePrefixes(
        embed_prefix="/model/embed",
        layer_prefix="/model/layers.",
        norm_prefix="/model/norm",
        lm_head_prefix="/lm_head"
    ),
    ArchitectureType.MISTRAL: ArchitecturePrefixes(
        embed_prefix="/model/embed",
        layer_prefix="/model/layers.",
        norm_prefix="/model/norm",
        lm_head_prefix="/lm_head"
    ),
    ArchitectureType.PHI: ArchitecturePrefixes(
        embed_prefix="/model/embed",
        layer_prefix="/model/layers.",
        norm_prefix="/model/norm",
        lm_head_prefix="/lm_head"
    ),
    ArchitectureType.GEMMA: ArchitecturePrefixes(
        embed_prefix="/model/embed",
        layer_prefix="/model/layers.",
        norm_prefix="/model/norm",
        lm_head_prefix="/lm_head"
    ),
    ArchitectureType.QWEN: ArchitecturePrefixes(
        embed_prefix="/model/embed",
        layer_prefix="/model/layers.",
        norm_prefix="/model/norm",
        lm_head_prefix="/lm_head"
    ),
    ArchitectureType.QWEN2: ArchitecturePrefixes(
        embed_prefix="/model/embed",
        layer_prefix="/model/layers.",
        norm_prefix="/model/norm",
        lm_head_prefix="/lm_head"
    ),
    ArchitectureType.QWEN3: ArchitecturePrefixes(
        embed_prefix="/model/embed",
        layer_prefix="/model/layers.",
        norm_prefix="/model/norm",
        lm_head_prefix="/lm_head"
    ),
    # GPT-2 style
    ArchitectureType.GPT2: ArchitecturePrefixes(
        embed_prefix="/transformer/wte",
        layer_prefix="/transformer/h.",
        norm_prefix="/transformer/ln_f",
        lm_head_prefix="/lm_head"
    ),
    # Default/Unknown uses LLaMA style
    ArchitectureType.UNKNOWN: ArchitecturePrefixes(
        embed_prefix="/model/embed",
        layer_prefix="/model/layers.",
        norm_prefix="/model/norm",
        lm_head_prefix="/lm_head"
    ),
}


def get_prefixes(
    arch: ArchitectureType,
    format: str  # "safetensors" or "onnx"
) -> ArchitecturePrefixes:
    """
    Get the weight/node name prefixes for a given architecture and format.
    
    Args:
        arch: The model architecture type.
        format: Either "safetensors" or "onnx".
    
    Returns:
        ArchitecturePrefixes for the specified architecture and format.
    
    Raises:
        ValueError: If format is not "safetensors" or "onnx".
    
    Example:
        >>> prefixes = get_prefixes(ArchitectureType.LLAMA, "safetensors")
        >>> prefixes.layer_prefix
        'model.layers.'
    """
    format_lower = format.lower()
    
    if format_lower == "safetensors":
        prefix_map = SAFETENSORS_PREFIXES
    elif format_lower == "onnx":
        prefix_map = ONNX_PREFIXES
    else:
        raise ValueError(f"Unknown format: {format}. Must be 'safetensors' or 'onnx'.")
    
    # Return prefixes for architecture, falling back to UNKNOWN if not found
    return prefix_map.get(arch, prefix_map[ArchitectureType.UNKNOWN])


def get_prefixes_tuple(
    arch: ArchitectureType,
    format: str
) -> Tuple[str, str, str, str]:
    """
    Get prefixes as a tuple (for backward compatibility with existing code).
    
    Args:
        arch: The model architecture type.
        format: Either "safetensors" or "onnx".
    
    Returns:
        Tuple of (embed_prefix, layer_prefix, norm_prefix, lm_head_prefix).
    """
    p = get_prefixes(arch, format)
    return (p.embed_prefix, p.layer_prefix, p.norm_prefix, p.lm_head_prefix)


def architecture_from_string(model_type: str) -> ArchitectureType:
    """
    Convert a model_type string to ArchitectureType enum.
    
    Args:
        model_type: The model_type string from config.json.
    
    Returns:
        Corresponding ArchitectureType, or UNKNOWN if not recognized.
    """
    model_type_lower = model_type.lower() if model_type else ""
    
    for arch in ArchitectureType:
        if arch.value == model_type_lower:
            return arch
    
    # Check for partial matches
    if "llama" in model_type_lower:
        return ArchitectureType.LLAMA
    if "qwen" in model_type_lower:
        # Only check for explicit qwen3/qwen2 - NOT just "3" or "2" which causes false positives
        if "qwen3" in model_type_lower:
            return ArchitectureType.QWEN3
        if "qwen2" in model_type_lower:
            return ArchitectureType.QWEN2
        return ArchitectureType.QWEN
    
    return ArchitectureType.UNKNOWN
