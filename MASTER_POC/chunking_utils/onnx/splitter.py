"""
ONNX model splitting utilities.

This module provides the main splitting functionality for ONNX models,
creating per-component ONNX files for embeddings, layers, norm, and lm_head.
"""

import os
import re
import json
from typing import Dict, List, Optional, Tuple, Any, Set

import onnx
from onnx import numpy_helper, helper, TensorProto
from onnx import shape_inference

from ..common.path_utils import sanitize_model_name
from ..common.architecture import (
    ArchitectureType,
    ONNX_PREFIXES,
    get_prefixes,
)
from ..common.types import SplitConfig, ModelFormat

from .types import ComponentInfo, OnnxSplitResult
from .loader import load_onnx_model


def detect_architecture_from_onnx(graph) -> ArchitectureType:
    """
    Detect model architecture from ONNX node name patterns.
    
    Args:
        graph: ONNX graph.
    
    Returns:
        Detected ArchitectureType.
    """
    sample_names = []
    for node in list(graph.node)[:100]:
        if node.name:
            sample_names.append(node.name)
    
    # Also check initializer names
    for init in list(graph.initializer)[:50]:
        if init.name:
            sample_names.append(init.name)
    
    # Check for architecture patterns
    if any("/model/layers." in n or "model.layers." in n for n in sample_names):
        # Check for Qwen3 specific patterns
        if any("q_norm" in n or "k_norm" in n for n in sample_names):
            return ArchitectureType.QWEN3
        return ArchitectureType.LLAMA
    elif any("/transformer/h." in n or "transformer.h." in n for n in sample_names):
        return ArchitectureType.GPT2
    
    return ArchitectureType.LLAMA  # Default


def get_onnx_prefixes(model_type: Optional[str] = None) -> Tuple[str, str, str, str]:
    """
    Return node name prefixes for different model architectures in ONNX.
    
    Args:
        model_type: Optional model type string.
    
    Returns:
        Tuple of (embed_prefix, layer_prefix, norm_prefix, lm_prefix).
    """
    if model_type:
        from ..common.architecture import architecture_from_string
        arch = architecture_from_string(model_type)
    else:
        arch = ArchitectureType.LLAMA  # Default
    
    prefixes = get_prefixes(arch, "onnx")
    return (
        prefixes.embed_prefix,
        prefixes.layer_prefix,
        prefixes.norm_prefix,
        prefixes.lm_head_prefix,
    )


def extract_components(model: onnx.ModelProto) -> Dict[str, ComponentInfo]:
    """
    Extract semantic components from the ONNX model.
    
    Creates one ComponentInfo for:
    - embedding
    - layer_0, layer_1, ..., layer_N (one per layer!)
    - norm
    - lm_head
    - other (nodes not matching any pattern)
    
    Args:
        model: ONNX model.
    
    Returns:
        Dictionary mapping component name to ComponentInfo.
    """
    graph = model.graph
    arch = detect_architecture_from_onnx(graph)
    prefixes = get_prefixes(arch, "onnx")
    embed_prefix = prefixes.embed_prefix
    layer_prefix = prefixes.layer_prefix
    norm_prefix = prefixes.norm_prefix
    lm_prefix = prefixes.lm_head_prefix
    
    print(f"\nDetected architecture: {arch.value}")
    print(f"  Embed prefix: {embed_prefix}")
    print(f"  Layer prefix: {layer_prefix}")
    print(f"  Norm prefix: {norm_prefix}")
    print(f"  LM head prefix: {lm_prefix}")
    
    components: Dict[str, ComponentInfo] = {
        "embedding": ComponentInfo(component_type="embedding"),
        "norm": ComponentInfo(component_type="norm"),
        "lm_head": ComponentInfo(component_type="lm_head"),
        "other": ComponentInfo(component_type="other"),
    }
    
    # Build initializer lookup
    initializer_names = {init.name for init in graph.initializer}
    
    # Classify each node
    for i, node in enumerate(graph.node):
        node_name = node.name or ""
        node_name_lower = node_name.lower()
        
        # Check for embedding
        if embed_prefix.lower() in node_name_lower or "embed_tokens" in node_name_lower:
            components["embedding"].node_indices.append(i)
            components["embedding"].node_names.append(node_name)
            for inp in node.input:
                if inp in initializer_names:
                    components["embedding"].initializer_names.add(inp)
            continue
        
        # Check for layer pattern
        layer_id = None
        
        # Try architecture-specific prefix
        if layer_prefix.lower() in node_name_lower:
            try:
                suffix = node_name_lower.split(layer_prefix.lower())[-1]
                match = re.match(r'^(\d+)', suffix)
                if match:
                    layer_id = int(match.group(1))
            except Exception:
                pass
        
        # Fallback to generic "layers" regex
        if layer_id is None:
            layer_match = re.search(r'layers[./](\d+)', node_name_lower)
            if layer_match:
                layer_id = int(layer_match.group(1))
            # Fallback for GPT2 style "h.N"
            elif "h." in node_name_lower:
                h_match = re.search(r'h[./](\d+)', node_name_lower)
                if h_match:
                    layer_id = int(h_match.group(1))
        
        # RyzenAI/AWQ optimized operator patterns:
        # MatMulNBits_k_6_0, MatMulNBits_q_6_1, gqo_8_1_0, ssmlp_3_0_0
        # Pattern: <op_name>_<params>_<layer_id>
        if layer_id is None:
            ryzenai_patterns = [
                r'^matmulnbits_[a-z]_\d+_(\d+)$',   # MatMulNBits_k_6_0
                r'^gqo_\d+_\d+_(\d+)$',             # gqo_8_1_0
                r'^ssmlp_\d+_\d+_(\d+)$',           # ssmlp_3_0_0
                # Note: Concat_N nodes have arbitrary numbering, not layer indices
            ]
            for pattern in ryzenai_patterns:
                match = re.match(pattern, node_name_lower)
                if match:
                    layer_id = int(match.group(1))
                    break
        
        # Also check initializer names for layer association
        if layer_id is None:
            for inp in node.input:
                if inp in initializer_names:
                    init_match = re.search(r'\.layers\.(\d+)\.', inp)
                    if init_match:
                        layer_id = int(init_match.group(1))
                        break

        if layer_id is not None:
            layer_key = f"layer_{layer_id}"
            
            if layer_key not in components:
                components[layer_key] = ComponentInfo(
                    component_type="layer",
                    layer_id=layer_id
                )
            
            components[layer_key].node_indices.append(i)
            components[layer_key].node_names.append(node_name)
            for inp in node.input:
                if inp in initializer_names:
                    components[layer_key].initializer_names.add(inp)
            continue
        
        # Check for final norm (but NOT layer norms inside layers)
        if (norm_prefix.lower() in node_name_lower and 
            "layers" not in node_name_lower and
            "layer_norm" not in node_name_lower):
            components["norm"].node_indices.append(i)
            components["norm"].node_names.append(node_name)
            for inp in node.input:
                if inp in initializer_names:
                    components["norm"].initializer_names.add(inp)
            continue
        
        # Check for lm_head
        if lm_prefix.lower() in node_name_lower:
            components["lm_head"].node_indices.append(i)
            components["lm_head"].node_names.append(node_name)
            for inp in node.input:
                if inp in initializer_names:
                    components["lm_head"].initializer_names.add(inp)
            continue
        
        # Everything else
        components["other"].node_indices.append(i)
        components["other"].node_names.append(node_name)
        for inp in node.input:
            if inp in initializer_names:
                components["other"].initializer_names.add(inp)
    
    # Remove empty components
    components = {k: v for k, v in components.items() if v.node_indices}
    
    # Print summary
    print(f"\nExtracted {len(components)} components:")
    layer_count = 0
    for name, info in sorted(components.items()):
        if info.component_type == "layer":
            layer_count += 1
        print(f"  {name}: {len(info.node_indices)} nodes, {len(info.initializer_names)} initializers")
    
    return components


def create_component_onnx(
    model: onnx.ModelProto,
    component: ComponentInfo,
    component_name: str
) -> onnx.ModelProto:
    """
    Create a standalone ONNX model for a single component.
    
    Args:
        model: Original ONNX model.
        component: ComponentInfo for this component.
        component_name: Name for the component graph.
    
    Returns:
        ONNX ModelProto for this component.
    """
    graph = model.graph
    all_nodes = list(graph.node)
    
    # Build lookup structures
    graph_input_names = {inp.name for inp in graph.input}
    initializer_names = {init.name for init in graph.initializer}
    initializer_dict = {init.name: init for init in graph.initializer}
    
    # Get nodes for this component
    component_nodes = [all_nodes[i] for i in component.node_indices]
    
    # Collect all outputs produced by this component
    component_outputs = set()
    for node in component_nodes:
        for output in node.output:
            component_outputs.add(output)
    
    # Collect required inputs (from outside this component)
    required_inputs = set()
    for node in component_nodes:
        for inp in node.input:
            if inp and inp not in component_outputs:
                required_inputs.add(inp)
    
    # Separate initializer inputs from boundary inputs
    boundary_inputs = []
    component_initializers = []
    
    for inp in required_inputs:
        if inp in initializer_names:
            component_initializers.append(initializer_dict[inp])
        else:
            boundary_inputs.append(inp)
    
    # Also add explicitly tracked initializers
    for init_name in component.initializer_names:
        if init_name in initializer_dict and init_name not in [i.name for i in component_initializers]:
            component_initializers.append(initializer_dict[init_name])
    
    # Determine outputs (consumed by nodes outside this component OR graph outputs)
    graph_output_names = {out.name for out in graph.output}
    component_node_set = set(component.node_indices)
    
    output_tensors = set()
    for node in component_nodes:
        for output in node.output:
            # Check if it's a graph output
            if output in graph_output_names:
                output_tensors.add(output)
                continue
            # Check if consumed by nodes outside this component
            for other_idx, other_node in enumerate(all_nodes):
                if other_idx not in component_node_set:
                    if output in other_node.input:
                        output_tensors.add(output)
                        break
    
    # If no outputs identified, use all outputs from the last node
    if not output_tensors and component_nodes:
        for output in component_nodes[-1].output:
            output_tensors.add(output)
    
    # Build lookups for value info
    init_dict = {i.name: i for i in graph.initializer}
    value_info_dict = {vi.name: vi for vi in graph.value_info}
    for gi in graph.input:
        value_info_dict[gi.name] = gi
    for go in graph.output:
        value_info_dict[go.name] = go
    
    # Create input ValueInfoProtos
    input_value_infos = []
    for inp_name in boundary_inputs:
        if not inp_name:
            continue
        vi = value_info_dict.get(inp_name)
        if vi and vi.name and vi.type.tensor_type.elem_type != 0:
            input_value_infos.append(vi)
            continue
            
        if inp_name in init_dict:
            init = init_dict[inp_name]
            shape = list(init.dims)
            elem_type = init.data_type
            new_vi = helper.make_tensor_value_info(inp_name, elem_type, shape)
            input_value_infos.append(new_vi)
            continue
            
        # Heuristic for unknown inputs
        rank = 1
        if "2D" in inp_name: rank = 2
        elif "3D" in inp_name: rank = 3
        elif "4D" in inp_name: rank = 4
        
        elem_type = TensorProto.FLOAT16
        if "position" in inp_name or "mask" in inp_name or "idx" in inp_name or "constants" in inp_name:
            elem_type = TensorProto.INT64
            
        shape = [None] * rank
        new_vi = helper.make_tensor_value_info(inp_name, elem_type, shape)
        input_value_infos.append(new_vi)
    
    # Create output ValueInfoProtos
    output_value_infos = []
    for out_name in output_tensors:
        if not out_name:
            continue
        vi = value_info_dict.get(out_name)
        if vi and vi.name and vi.type.tensor_type.elem_type != 0:
            output_value_infos.append(vi)
            continue
        
        # Fallback for unknown outputs
        elem_type = TensorProto.FLOAT16
        if any(word in out_name.lower() for word in ["mask", "idx", "position", "count"]):
            elem_type = TensorProto.INT32
        
        shape = [None]
        new_vi = helper.make_tensor_value_info(out_name, elem_type, shape)
        output_value_infos.append(new_vi)
    
    # Create the graph
    component_graph = helper.make_graph(
        nodes=component_nodes,
        name=f"{component_name}_graph",
        inputs=input_value_infos,
        outputs=output_value_infos,
        initializer=component_initializers
    )
    
    # Create the model
    component_model = helper.make_model(
        component_graph,
        producer_name=f"chunking_utils_onnx_{component_name}",
        opset_imports=list(model.opset_import)
    )
    
    # Copy IR version
    component_model.ir_version = model.ir_version
    
    return component_model


def split_onnx(
    model: Optional[str] = None,
    output: Optional[str] = None,
    config: Optional[SplitConfig] = None,
    split_only: bool = False,
    **kwargs
) -> OnnxSplitResult:
    """
    Split an ONNX model into per-component ONNX files.
    
    Creates:
    - embeddings.onnx
    - layer_0_of_N.onnx, layer_1_of_N.onnx, ... (one per layer!)
    - norm.onnx
    - lm_head.onnx
    - model_info.json
    
    Args:
        model: Path to ONNX model file or directory. Required if config not provided.
        output: Output directory path. Required if config not provided.
        config: Optional SplitConfig object (overrides model/output if provided).
        split_only: If True, skip inference validation.
        **kwargs: Additional arguments passed to SplitConfig.
    
    Returns:
        OnnxSplitResult with information about the split operation.
    
    Example:
        >>> result = split_onnx(
        ...     model="./model.onnx",
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
    
    # Load model
    onnx_model = load_onnx_model(model_path)
    
    # Run shape inference
    print("Running static shape inference on the master model...")
    try:
        onnx_model = shape_inference.infer_shapes(onnx_model)
        print("  Shape inference successful.")
    except Exception as e:
        print(f"  Warning: Shape inference failed: {e}. Subgraphs may have unknown shapes.")
    
    # Extract components
    components = extract_components(onnx_model)
    
    # Count layers
    layer_components = {k: v for k, v in components.items() if v.component_type == "layer"}
    num_layers = len(layer_components)
    
    # Create output directory
    model_folder_name = sanitize_model_name(config.model)
    output_dir = os.path.join(config.output, model_folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSplitting model into {len(components)} components ({num_layers} layers)...")
    print(f"Output directory: {output_dir}")
    
    component_files: Dict[str, str] = {}
    
    # Save each component
    for comp_name, comp_info in sorted(components.items()):
        # Determine filename
        if comp_info.component_type == "layer":
            filename = f"layer_{comp_info.layer_id}_of_{num_layers}.onnx"
        elif comp_info.component_type == "embedding":
            filename = "embeddings.onnx"
        elif comp_info.component_type == "norm":
            filename = "norm.onnx"
        elif comp_info.component_type == "lm_head":
            filename = "lm_head.onnx"
        else:
            filename = f"{comp_name}.onnx"
        
        filepath = os.path.join(output_dir, filename)
        
        # Create component ONNX
        component_model = create_component_onnx(onnx_model, comp_info, comp_name)
        
        # Capture I/O metadata
        comp_info.input_names = [i.name for i in component_model.graph.input]
        comp_info.output_names = [o.name for o in component_model.graph.output]
        
        # Save (with external data for large models)
        try:
            data_filename = filename + ".data"
            onnx.save(
                component_model,
                filepath,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=data_filename,
                size_threshold=1024 * 1024  # 1MB threshold
            )
        except Exception:
            # Fallback: save without external data
            onnx.save(component_model, filepath)
        
        component_files[comp_name] = filepath
        print(f"  Saved {filename} ({len(comp_info.node_indices)} nodes, {len(comp_info.initializer_names)} initializers)")
    
    # Save model info
    model_info = {
        "original_model": str(model_path),
        "num_layers": num_layers,
        "num_components": len(components),
        "architecture": detect_architecture_from_onnx(onnx_model.graph).value,
        "ir_version": onnx_model.ir_version,
        "producer": onnx_model.producer_name,
        "components": {
            name: {
                "type": info.component_type,
                "layer_id": info.layer_id,
                "num_nodes": len(info.node_indices),
                "num_initializers": len(info.initializer_names),
                "input_names": info.input_names,
                "output_names": info.output_names,
                "file": os.path.basename(component_files.get(name, ""))
            }
            for name, info in components.items()
        }
    }
    
    info_path = os.path.join(output_dir, "model_info.json")
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"  Saved model_info.json")
    
    print(f"\nModel split complete. Output: {output_dir}")
    
    return OnnxSplitResult(
        output_dir=output_dir,
        num_layers=num_layers,
        component_files=component_files,
        model_info=model_info,
        format=ModelFormat.ONNX,
        ir_version=onnx_model.ir_version,
        producer=onnx_model.producer_name,
        components=components,
    )
