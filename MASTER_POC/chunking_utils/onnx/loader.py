"""
ONNX model loading utilities.

This module provides utilities for loading ONNX models from disk,
with support for external data files and large models.
"""

import os
from typing import Optional

import onnx


def resolve_model_path(model_path: str) -> str:
    """
    Resolve the actual path to the ONNX model file.
    
    If a directory is provided, searches for common ONNX model filenames.
    
    Args:
        model_path: Path to ONNX model file or directory.
    
    Returns:
        Resolved path to the ONNX model file.
    
    Raises:
        FileNotFoundError: If no ONNX model found at path.
    
    Example:
        >>> resolve_model_path("./model_dir")
        './model_dir/model.onnx'
    """
    if os.path.isfile(model_path):
        return model_path
    elif os.path.isdir(model_path):
        # Look for common ONNX model filenames
        candidates = [
            "model.onnx",
            "model_optimized.onnx",
            "decoder_model.onnx",
            "encoder_model.onnx",
        ]
        for name in candidates:
            candidate_path = os.path.join(model_path, name)
            if os.path.exists(candidate_path):
                return candidate_path
        raise FileNotFoundError(f"No ONNX model found in directory: {model_path}")
    else:
        raise FileNotFoundError(f"Model path does not exist: {model_path}")


def load_onnx_model(
    model_path: str,
    load_external_data: bool = True,
    lazy_load_threshold_mb: float = 10000,
) -> onnx.ModelProto:
    """
    Load an ONNX model from disk.
    
    Supports:
    - Single ONNX files
    - ONNX files with external data (.onnx.data files)
    - Lazy loading for very large models (>10GB by default)
    
    Args:
        model_path: Path to ONNX model file or directory.
        load_external_data: Whether to load external data (weights).
        lazy_load_threshold_mb: If external data exceeds this size in MB,
            use lazy loading (graph structure only).
    
    Returns:
        Loaded ONNX ModelProto.
    
    Example:
        >>> model = load_onnx_model("./model.onnx")
        >>> print(f"Nodes: {len(model.graph.node)}")
    """
    model_file = resolve_model_path(model_path)
    print(f"Loading ONNX model from: {model_file}")
    
    # Check for external data
    data_file = model_file + ".data"
    has_external_data = os.path.exists(data_file)
    
    if has_external_data:
        size_mb = os.path.getsize(data_file) / (1024 * 1024)
        print(f"  External data file detected: {os.path.basename(data_file)} ({size_mb:.1f} MB)")
        
        # For very large models, use lazy loading to avoid OOM
        if size_mb > lazy_load_threshold_mb:
            print(f"  Using LAZY loading (graph structure only) for large model")
            model = onnx.load(model_file, load_external_data=False)
        else:
            model = onnx.load(model_file, load_external_data=load_external_data)
    else:
        model = onnx.load(model_file, load_external_data=load_external_data)
    
    print(f"  IR version: {model.ir_version}")
    print(f"  Producer: {model.producer_name}")
    print(f"  Nodes: {len(model.graph.node)}")
    print(f"  Initializers: {len(model.graph.initializer)}")
    
    return model


def get_model_info(model: onnx.ModelProto) -> dict:
    """
    Get summary information about an ONNX model.
    
    Args:
        model: ONNX ModelProto.
    
    Returns:
        Dictionary with model information.
    """
    graph = model.graph
    
    # Count node types
    node_types = {}
    for node in graph.node:
        op_type = node.op_type
        node_types[op_type] = node_types.get(op_type, 0) + 1
    
    # Get input/output info
    inputs = [
        {"name": inp.name, "shape": [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]}
        for inp in graph.input
        if inp.name not in {init.name for init in graph.initializer}
    ]
    outputs = [
        {"name": out.name}
        for out in graph.output
    ]
    
    return {
        "ir_version": model.ir_version,
        "producer": model.producer_name,
        "num_nodes": len(graph.node),
        "num_initializers": len(graph.initializer),
        "node_types": node_types,
        "inputs": inputs,
        "outputs": outputs,
    }


def validate_onnx_model(model_path: str) -> bool:
    """
    Validate that an ONNX model is well-formed.
    
    Args:
        model_path: Path to ONNX model file.
    
    Returns:
        True if model is valid, False otherwise.
    """
    try:
        model = load_onnx_model(model_path)
        onnx.checker.check_model(model)
        return True
    except Exception as e:
        print(f"Model validation failed: {e}")
        return False
