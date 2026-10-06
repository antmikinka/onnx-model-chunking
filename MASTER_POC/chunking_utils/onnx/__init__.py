"""
ONNX model chunking utilities.

This module provides utilities for splitting and merging ONNX models
into per-component files (embeddings, layers, norm, lm_head).

Usage:
    from chunking_utils.onnx import split_onnx, merge_onnx
    
    # Split an ONNX model
    result = split_onnx(model="./model.onnx", output="./output")
    
    # Merge back
    merged_model = merge_onnx("./output/model")
"""

from .splitter import (
    split_onnx,
    extract_components,
    create_component_onnx,
    collect_boundary_tensors,
    resolve_tensor_type,
    extract_quantization_info,
    detect_architecture_from_onnx,
)
from .merger import merge_onnx
from .loader import load_onnx_model, resolve_model_path
from .types import ComponentInfo, OnnxSplitResult
from .inference import run_inference

__all__ = [
    # Main operations
    "split_onnx",
    "merge_onnx",
    # Loader
    "load_onnx_model",
    "resolve_model_path",
    # Component extraction
    "extract_components",
    "create_component_onnx",
    "collect_boundary_tensors",
    "resolve_tensor_type",
    "extract_quantization_info",
    # Architecture detection
    "detect_architecture_from_onnx",
    # Types
    "ComponentInfo",
    "OnnxSplitResult",
    # Inference
    "run_inference",
]
