"""
ONNX model merging utilities.

This module provides utilities for merging split ONNX component files
back into a single unified ONNX model.
"""

import os
import json
from typing import Optional

import onnx
from onnx import helper

from .types import OnnxMergeResult


def merge_onnx(
    output_dir: str,
    save_path: Optional[str] = None,
) -> onnx.ModelProto:
    """
    Merge all component ONNX files back into a single model.
    
    Args:
        output_dir: Directory containing component ONNX files.
        save_path: Optional path to save the merged model.
    
    Returns:
        Merged ONNX ModelProto.
    
    Raises:
        FileNotFoundError: If model_info.json not found.
    
    Example:
        >>> merged_model = merge_onnx("./output/model")
        >>> onnx.save(merged_model, "./merged.onnx")
    """
    print(f"\nMerging components from: {output_dir}")
    
    # Load model info
    info_path = os.path.join(output_dir, "model_info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"model_info.json not found in {output_dir}")
    
    with open(info_path, 'r') as f:
        model_info = json.load(f)
    
    num_layers = model_info["num_layers"]
    
    # Collect all nodes and initializers in order
    all_nodes = []
    all_initializers = []
    all_inputs = []
    all_outputs = []
    lm_head_outputs = None  # Track lm_head outputs explicitly as final graph outputs
    opset_imports = None
    ir_version = None
    
    # Load in order: embedding, layers 0..N, norm, lm_head, other
    component_order = ["embedding"]
    for i in range(num_layers):
        component_order.append(f"layer_{i}")
    component_order.extend(["norm", "lm_head", "other"])
    
    seen_initializers = set()
    seen_inputs = set()
    
    for comp_name in component_order:
        if comp_name not in model_info["components"]:
            continue
        
        comp_info = model_info["components"][comp_name]
        filename = comp_info.get("file", "")
        if not filename:
            continue
        
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            print(f"  Warning: {filename} not found, skipping")
            continue
        
        # Load component
        comp_model = onnx.load(filepath, load_external_data=True)
        
        if opset_imports is None:
            opset_imports = list(comp_model.opset_import)
            ir_version = comp_model.ir_version
        
        # Add nodes
        all_nodes.extend(comp_model.graph.node)
        
        # Add initializers (avoiding duplicates)
        for init in comp_model.graph.initializer:
            if init.name not in seen_initializers:
                all_initializers.append(init)
                seen_initializers.add(init.name)
        
        # Track inputs and outputs
        for inp in comp_model.graph.input:
            if inp.name not in seen_inputs and inp.name not in seen_initializers:
                all_inputs.append(inp)
                seen_inputs.add(inp.name)
        
        # Track lm_head outputs explicitly (should be final graph outputs)
        if comp_name == "lm_head":
            lm_head_outputs = list(comp_model.graph.output)
        else:
            all_outputs = list(comp_model.graph.output)
        
        print(f"  Loaded {filename}: {len(comp_model.graph.node)} nodes")
    
    # Use lm_head outputs as final outputs if available (proper transformer output)
    final_outputs = lm_head_outputs if lm_head_outputs else all_outputs
    
    # Create merged graph
    merged_graph = helper.make_graph(
        nodes=all_nodes,
        name="merged_graph",
        inputs=all_inputs,
        outputs=final_outputs,
        initializer=all_initializers
    )
    
    # Create merged model
    merged_model = helper.make_model(
        merged_graph,
        producer_name="chunking_utils_merged",
        opset_imports=opset_imports or []
    )
    
    if ir_version:
        merged_model.ir_version = ir_version
    
    print(f"  Merged model: {len(all_nodes)} nodes, {len(all_initializers)} initializers")
    
    # Optionally save
    if save_path:
        onnx.save(merged_model, save_path)
        print(f"  Saved merged model to: {save_path}")
    
    return merged_model


def merge_onnx_with_result(
    output_dir: str,
    save_path: Optional[str] = None,
) -> OnnxMergeResult:
    """
    Merge components and return a result object.
    
    Args:
        output_dir: Directory containing component ONNX files.
        save_path: Optional path to save the merged model.
    
    Returns:
        OnnxMergeResult with merge information.
    """
    try:
        merged_model = merge_onnx(output_dir, save_path)
        return OnnxMergeResult(
            success=True,
            output_path=save_path,
            num_nodes=len(merged_model.graph.node),
            num_initializers=len(merged_model.graph.initializer),
        )
    except Exception as e:
        return OnnxMergeResult(
            success=False,
            error=str(e),
        )


def get_merge_info(output_dir: str) -> dict:
    """
    Get information about what would be merged from a directory.
    
    Args:
        output_dir: Directory containing component ONNX files.
    
    Returns:
        Dictionary with merge statistics.
    """
    info_path = os.path.join(output_dir, "model_info.json")
    if not os.path.exists(info_path):
        return {"error": "model_info.json not found"}
    
    with open(info_path, 'r') as f:
        model_info = json.load(f)
    
    components_to_merge = []
    total_nodes = 0
    
    for comp_name, comp_info in model_info.get("components", {}).items():
        filename = comp_info.get("file", "")
        if filename:
            filepath = os.path.join(output_dir, filename)
            exists = os.path.exists(filepath)
            num_nodes = comp_info.get("num_nodes", 0)
            components_to_merge.append({
                "name": comp_name,
                "file": filename,
                "exists": exists,
                "num_nodes": num_nodes,
            })
            if exists:
                total_nodes += num_nodes
    
    return {
        "num_components": len(components_to_merge),
        "total_nodes": total_nodes,
        "num_layers": model_info.get("num_layers", 0),
        "components": components_to_merge,
    }
