#!/usr/bin/env python3
"""
grok_onnx_split.py - True ONNX Semantic Model Splitter

A TRUE ONNX-based equivalent of grok_hf_split.py that splits ONNX models
into per-component files:
- embeddings.onnx
- layer_0_of_N.onnx, layer_1_of_N.onnx, ... (one per layer!)
- norm.onnx
- lm_head.onnx

Supports:
- Local ONNX model paths
- External data files (.onnx.data)
- Model merging for inference
- ONNX Runtime inference

Usage:
    python grok_onnx_split.py --model <path> --output <dir> [--split-only]
"""

import os
import re
import json
import argparse
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import numpy as np

import onnx
from onnx import numpy_helper, helper, TensorProto
from onnx import shape_inference

# Optional ONNX Runtime for inference
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class ComponentInfo:
    """Information about a semantic model component"""
    component_type: str  # 'embedding', 'layer', 'norm', 'lm_head', 'other'
    layer_id: Optional[int] = None  # For layer components
    node_indices: List[int] = field(default_factory=list)
    node_names: List[str] = field(default_factory=list)
    initializer_names: Set[str] = field(default_factory=set)
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)


@dataclass
class SplitResult:
    """Result of the semantic split operation"""
    output_dir: str
    num_layers: int
    component_files: Dict[str, str]  # component_name -> file_path
    model_info: Dict[str, Any]


# ==============================================================================
# Utility Functions
# ==============================================================================

def is_local_path(model_id: str) -> bool:
    """Check if the model_id is a local filesystem path."""
    return os.path.isdir(model_id) or os.path.isabs(model_id) or os.path.exists(model_id)


def sanitize_model_name(model_id: str) -> str:
    """Convert a model ID or path to a filesystem-safe folder name."""
    if is_local_path(model_id):
        normalized = os.path.normpath(model_id)
        safe_name = os.path.basename(normalized)
    else:
        safe_name = model_id.replace("/", "--")
    
    # Remove illegal Windows characters
    safe_name = re.sub(r'[<>:"|?*\\]', '', safe_name)
    return safe_name.strip()


def get_onnx_model_prefixes(model_type: Optional[str] = None) -> Tuple[str, str, str, str]:
    """
    Return node name prefixes for different model architectures in ONNX.
    
    Returns:
        Tuple of (embed_prefix, layer_prefix, norm_prefix, lm_prefix)
    """
    if model_type in ["llama", "mistral", "phi", "gemma", "qwen2", "qwen3", "qwen"]:
        embed_prefix = "/model/embed"
        layer_prefix = "/model/layers."
        norm_prefix = "/model/norm"
        lm_prefix = "/lm_head"
    elif model_type == "gpt2":
        embed_prefix = "/transformer/wte"
        layer_prefix = "/transformer/h."
        norm_prefix = "/transformer/ln_f"
        lm_prefix = "/lm_head"
    else:
        # Default to LLaMA-style (most common for modern models)
        embed_prefix = "/model/embed"
        layer_prefix = "/model/layers."
        norm_prefix = "/model/norm"
        lm_prefix = "/lm_head"
    
    return embed_prefix, layer_prefix, norm_prefix, lm_prefix


def detect_architecture_from_onnx(graph) -> str:
    """
    Detect model architecture from ONNX node name patterns.
    
    Args:
        graph: ONNX graph
        
    Returns:
        Detected model_type string
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
    # Increase sample size or look for specific RyzenAI ops
    all_names = sample_names
    
    if any("/model/layers." in n or "model.layers." in n or "MatMulNBits" in n for n in all_names):
        # Check for Qwen3 specific patterns (q_norm or the custom MatMulNBits suffix numbering)
        if any("q_norm" in n or "k_norm" in n or "gqo_" in n or "ssmlp_" in n for n in all_names):
            return "qwen3"
        return "llama"
    elif any("/transformer/h." in n or "transformer.h." in n for n in all_names):
        return "gpt2"
    
    return "llama"  # Default


# ==============================================================================
# Model Loading
# ==============================================================================

def resolve_model_path(model_path: str) -> str:
    """Resolve the actual path to the ONNX model file."""
    if os.path.isfile(model_path):
        return model_path
    elif os.path.isdir(model_path):
        # Look for common ONNX model filenames
        candidates = ["model.onnx", "model_optimized.onnx", "decoder_model.onnx"]
        for name in candidates:
            candidate_path = os.path.join(model_path, name)
            if os.path.exists(candidate_path):
                return candidate_path
        raise FileNotFoundError(f"No ONNX model found in directory: {model_path}")
    else:
        raise FileNotFoundError(f"Model path does not exist: {model_path}")


def load_onnx_model(model_path: str) -> onnx.ModelProto:
    """
    Load an ONNX model from disk.
    
    Args:
        model_path: Path to ONNX model file or directory
        
    Returns:
        Loaded ONNX ModelProto
    """
    model_file = resolve_model_path(model_path)
    print(f"Loading ONNX model from: {model_file}")
    
    # Check for external data
    data_file = model_file + ".data"
    has_external_data = os.path.exists(data_file)
    if has_external_data:
        size_mb = os.path.getsize(data_file) / (1024 * 1024)
        print(f"  External data file detected: {os.path.basename(data_file)} ({size_mb:.1f} MB)")
        # For very large models (>10GB), use lazy loading to avoid OOM
        if size_mb > 10000:
            print(f"  Using LAZY loading (graph structure only) for large model")
            model = onnx.load(model_file, load_external_data=False)
        else:
            model = onnx.load(model_file, load_external_data=True)
    else:
        model = onnx.load(model_file, load_external_data=True)
    
    print(f"  IR version: {model.ir_version}")
    print(f"  Producer: {model.producer_name}")
    print(f"  Nodes: {len(model.graph.node)}")
    print(f"  Initializers: {len(model.graph.initializer)}")
    
    return model


# ==============================================================================
# Component Extraction
# ==============================================================================

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
        model: ONNX model
        
    Returns:
        Dictionary mapping component name to ComponentInfo
    """
    graph = model.graph
    model_type = detect_architecture_from_onnx(graph)
    embed_prefix, layer_prefix, norm_prefix, lm_prefix = get_onnx_model_prefixes(model_type)
    
    print(f"\nDetected architecture: {model_type}")
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
    print(f"DEBUG: Starting classification of {len(graph.node)} nodes...")
    for i, node in enumerate(graph.node):
        node_name = node.name or ""
        node_name_lower = node_name.lower()
        
        if i < 10:
             print(f"  DEBUG: Node {i}: '{node_name}' (Type: {node.op_type})")
        
        # Check for embedding
        if embed_prefix.lower() in node_name_lower or "embed_tokens" in node_name_lower:
            if i < 50: print(f"    DEBUG: Node {i} matched EMBEDDING")
            components["embedding"].node_indices.append(i)
            components["embedding"].node_names.append(node_name)
            for inp in node.input:
                if inp in initializer_names:
                    components["embedding"].initializer_names.add(inp)
            continue
        
        # Check for layer pattern using architecture-specific prefix
        # This handles 'model.layers.0' (Llama) and 'transformer.h.0' (GPT2)
        # We try the specific prefix first, then a generic fallback
        layer_id = None
        
        # 1. Try architecture-specific prefix
        if layer_prefix.lower() in node_name_lower:
            # Extract number after prefix
            # e.g. prefix="/model/layers.", node="/model/layers.5/..."
            try:
                # Find the part after the prefix
                suffix = node_name_lower.split(layer_prefix.lower())[-1]
                # The next characters should be the ID
                match = re.match(r'^(\d+)', suffix)
                if match:
                    layer_id = int(match.group(1))
            except Exception:
                pass
        
        # 2. Fallback to generic "layers" regex if specific prefix didn't match
        if layer_id is None:
            layer_match = re.search(r'layers[./](\d+)', node_name_lower)
            if layer_match:
                layer_id = int(layer_match.group(1))
            # 3. Fallback for GPT2 style "h.N" if not caught above
            elif "h." in node_name_lower:
                 h_match = re.search(r'h[./](\d+)', node_name_lower)
                 if h_match:
                     layer_id = int(h_match.group(1))
        
        # 4. RyzenAI/AWQ optimized operator patterns
        # These custom ops have layer index as suffix: MatMulNBits_k_6_0, gqo_8_1_0, ssmlp_3_0_0
        if layer_id is None:
            ryzenai_patterns = [
                r'^matmulnbits_[a-z]_\d+_(\d+)$',   # MatMulNBits_k_6_0
                r'^gqo_\d+_\d+_(\d+)$',             # gqo_8_1_0
                r'^ssmlp_\d+_\d+_(\d+)$',           # ssmlp_3_0_0
            ]
            for pattern in ryzenai_patterns:
                match = re.match(pattern, node_name_lower)
                if match:
                    layer_id = int(match.group(1))
                    break
        
        # 5. Fallback: Check initializer names for layer association
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
    
    # Phase 2: Connectivity-based assignment for 'other' nodes
    # If a node is 'other' but its inputs primarily come from a specific layer,
    # assign it to that layer to avoid dependency cycles.
    print("DEBUG: Starting connectivity-based assignment pass...")
    
    # Map output names to the component they belong to
    output_to_comp = {}
    for comp_name, info in components.items():
        for node_idx in info.node_indices:
            node = graph.node[node_idx]
            for out in node.output:
                output_to_comp[out] = comp_name
                
    # Also include graph inputs (e.g. input_ids -> embedding)
    for inp in graph.input:
        if "/model/embed" in inp.name or "input_ids" in inp.name:
            output_to_comp[inp.name] = "embedding"

    # Iterate through 'other' nodes and try to assign them
    changed = True
    iterations = 0
    while changed and iterations < 3:  # Limit iterations
        changed = False
        iterations += 1
        other_node_indices = list(components["other"].node_indices)
        
        for idx in other_node_indices:
            node = graph.node[idx]
            
            # Skip assignment if the node is already in a core component
            current_comp = output_to_comp.get(node.output[0]) if node.output else "other"
            if current_comp in ["embedding", "norm", "lm_head"]:
                continue

            # See where inputs come from
            input_comps = set()
            for inp in node.input:
                if inp in output_to_comp:
                    input_comps.add(output_to_comp[inp])
            
            # If all inputs (that we know) come from one component, assign to it
            if len(input_comps) == 1:
                target_comp = list(input_comps)[0]
                if target_comp != "other" and target_comp != current_comp:
                    # Reassign
                    if current_comp == "other":
                        components["other"].node_indices.remove(idx)
                        if node.name in components["other"].node_names:
                             components["other"].node_names.remove(node.name)
                    
                    components[target_comp].node_indices.append(idx)
                    components[target_comp].node_names.append(node.name)
                    for inp_name in node.input:
                        if inp_name in initializer_names:
                            components[target_comp].initializer_names.add(inp_name)
                    
                    # Update output mapping
                    for out in node.output:
                        output_to_comp[out] = target_comp
                    
                    changed = True
                    if idx < 50: print(f"    DEBUG: Node {idx} ('{node.name}') re-assigned from '{current_comp}' to '{target_comp}'")

    # Remove empty components
    components = {k: v for k, v in components.items() if v.node_indices}
    
    # Print summary
    print(f"\nExtracted {len(components)} components (after connectivity pass):")
    layer_count = 0
    for name, info in sorted(components.items()):
        if info.component_type == "layer":
            layer_count += 1
        print(f"  {name}: {len(info.node_indices)} nodes, {len(info.initializer_names)} initializers")
    
    return components


def compute_boundary_tensors(
    component: ComponentInfo,
    all_nodes: List,
    graph_inputs: Set[str],
    all_outputs: Dict[str, int]  # output_name -> node_index that produces it
) -> Tuple[List[str], List[str]]:
    """
    Compute input and output boundary tensors for a component.
    
    Args:
        component: ComponentInfo for this component
        all_nodes: All nodes in the graph
        graph_inputs: Set of graph input names
        all_outputs: Mapping of output tensor names to producing node index
        
    Returns:
        Tuple of (input_tensors, output_tensors)
    """
    component_node_set = set(component.node_indices)
    component_outputs = set()
    

def create_component_onnx(
    model: onnx.ModelProto,
    component: ComponentInfo,
    component_name: str
) -> onnx.ModelProto:
    """
    Create a standalone ONNX model for a single component.
    
    Args:
        model: Original ONNX model
        component: ComponentInfo for this component
        component_name: Name for the component graph
        
    Returns:
        ONNX ModelProto for this component
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
    
    # Create input ValueInfoProtos
    # Build a lookup for initializers to get their shapes
    init_dict = {i.name: i for i in graph.initializer}
    
    # Build a lookup of all value_info from the original graph
    value_info_dict = {vi.name: vi for vi in graph.value_info}
    for gi in graph.input:
        value_info_dict[gi.name] = gi
    for go in graph.output:
        value_info_dict[go.name] = go
    
    input_value_infos = []
    for inp_name in boundary_inputs:
        if not inp_name:
            continue
        # 1. Try to find existing ValueInfo from original graph (even without shape info)
        vi = value_info_dict.get(inp_name)
        if vi and vi.name and vi.type.tensor_type.elem_type != 0:
            # If shape is missing, we still use the type info
            input_value_infos.append(vi)
            continue
            
        # 2. Try to get shape from initializer
        if inp_name in init_dict:
            init = init_dict[inp_name]
            shape = list(init.dims)
            elem_type = init.data_type
            new_vi = helper.make_tensor_value_info(inp_name, elem_type, shape)
            input_value_infos.append(new_vi)
            continue
            
        # 3. Last resort: Heuristic based on name or default to Rank 1
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
    
    # Create output ValueInfoProtos (similar logic)
    output_value_infos = []
    # Build a lookup of nodes by their outputs to infer types
    node_by_output = {}
    for node in component_nodes:
        for out in node.output:
            node_by_output[out] = node
            
    for out_name in output_tensors:
        if not out_name:
            continue
        vi = value_info_dict.get(out_name)
        if vi and vi.name and vi.type.tensor_type.elem_type != 0:
            output_value_infos.append(vi)
            continue
            rank = 1
            if "2D" in out_name: rank = 2
            elif "3D" in out_name: rank = 3
            elif "4D" in out_name: rank = 4
            
            # Infer type from producing node if missing
            elem_type = TensorProto.FLOAT16
            if out_name in node_by_output:
                node = node_by_output[out_name]
                if node.op_type == "Cast":
                    for attr in node.attribute:
                        if attr.name == "to":
                            elem_type = attr.i
                            break
                elif any(word in out_name.lower() for word in ["mask", "idx", "position", "count"]):
                    elem_type = TensorProto.INT32  # Prefer INT32 for masks in optimized models
            elif any(word in out_name.lower() for word in ["mask", "idx", "position", "count"]):
                elem_type = TensorProto.INT32
            
            shape = [None] * rank
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
        producer_name=model.producer_name,      # CRITICAL: Preserve producer (e.g., onnxruntime-genai)
        producer_version=model.producer_version, # CRITICAL: Preserve version
        opset_imports=list(model.opset_import),
        functions=model.functions,              # CRITICAL: Preserve function definitions
        doc_string=model.doc_string
    )
    
    # Copy IR version
    component_model.ir_version = model.ir_version
    
    # Preserve Metadata Props (e.g., for custom EP config)
    if hasattr(model, "metadata_props"):
        component_model.metadata_props.extend(model.metadata_props)
    
    return component_model


# ==============================================================================
# Model Splitting (Per-Component Output)
# ==============================================================================

def split_onnx_model(params) -> SplitResult:
    """
    Split an ONNX model into per-component ONNX files.
    
    Creates:
    - embeddings.onnx
    - layer_0_of_N.onnx, layer_1_of_N.onnx, ... (one per layer!)
    - norm.onnx
    - lm_head.onnx
    
    Args:
        params: Namespace with model, output, split_only attributes
        
    Returns:
        SplitResult with output information
    """
    model_path = params.model
    
    # Load model
    model = load_onnx_model(model_path)
    
    # Static Shape Propagation: Run shape inference to enrich the graph metadata.
    # This ensures that intermediate tensors have rank and shape information,
    # which is critical for operators like 'Reshape' in the split subgraphs.
    # print("Running static shape inference on the master model...")
    # try:
    #     model = shape_inference.infer_shapes(model)
    #     print("  Shape inference successful.")
    # except Exception as e:
    #     print(f"  Warning: Shape inference failed: {e}. Subgraphs may have unknown shapes.")
    
    # Extract components
    components = extract_components(model)
    
    # Count layers
    layer_components = {k: v for k, v in components.items() if v.component_type == "layer"}
    num_layers = len(layer_components)
    
    # Create output directory
    model_folder_name = sanitize_model_name(params.model)
    output_dir = os.path.join(params.output, model_folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSplitting model into {len(components)} components ({num_layers} layers)...")
    print(f"Output directory: {output_dir}")
    
    component_files = {}
    
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
        component_model = create_component_onnx(model, comp_info, comp_name)
        
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
        "architecture": detect_architecture_from_onnx(model.graph),
        "ir_version": model.ir_version,
        "producer": model.producer_name,
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
    
    return SplitResult(
        output_dir=output_dir,
        num_layers=num_layers,
        component_files=component_files,
        model_info=model_info
    )


# ==============================================================================
# Model Merging (for Inference)
# ==============================================================================

def merge_onnx_components(output_dir: str) -> onnx.ModelProto:
    """
    Merge all component ONNX files back into a single model.
    
    Args:
        output_dir: Directory containing component ONNX files
        
    Returns:
        Merged ONNX ModelProto
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
        
        all_outputs = list(comp_model.graph.output)  # Last component's outputs
        
        print(f"  Loaded {filename}: {len(comp_model.graph.node)} nodes")
    
    # Create merged graph
    merged_graph = helper.make_graph(
        nodes=all_nodes,
        name="merged_graph",
        inputs=all_inputs,
        outputs=all_outputs,
        initializer=all_initializers
    )
    
    # Create merged model
    merged_model = helper.make_model(
        merged_graph,
        producer_name="grok_onnx_split_merged",
        opset_imports=opset_imports or []
    )
    
    if ir_version:
        merged_model.ir_version = ir_version
    
    print(f"  Merged model: {len(all_nodes)} nodes, {len(all_initializers)} initializers")
    
    return merged_model


# ==============================================================================
# Inference
# ==============================================================================

def run_inference(output_dir: str, params) -> List[str]:
    """
    Run inference on the split/merged model.
    
    Args:
        output_dir: Directory containing split model
        params: Parameters including prompt
        
    Returns:
        List of generated texts
    """
    if not HAS_ORT:
        print("Warning: onnxruntime not available, skipping inference")
        return []
    
    print("\nRunning inference...")
    
    # Merge components
    merged_model = merge_onnx_components(output_dir)
    
    # Save merged model temporarily
    merged_path = os.path.join(output_dir, "merged_model.onnx")
    onnx.save(merged_model, merged_path)
    
    # Create inference session
    session = ort.InferenceSession(merged_path)
    
    # Get input info
    input_info = session.get_inputs()
    print(f"  Model inputs: {[i.name for i in input_info]}")
    
    # Note: Full inference would require tokenizer integration
    # For now, just validate the model can load
    print("  Model loaded successfully for inference")
    print("  (Full inference requires tokenizer integration)")
    
    return ["Model validated for inference"]


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Split ONNX model into per-component files (ONNX equivalent of grok_hf_split.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split a model into semantic components
  python grok_onnx_split.py --model "C:\\path\\to\\model.onnx" --output "C:\\output"
  
  # Split only, skip inference validation
  python grok_onnx_split.py --model "C:\\path\\to\\model" --output "./split_output" --split-only
  
Output files created:
  - embeddings.onnx
  - layer_0_of_N.onnx, layer_1_of_N.onnx, ...
  - norm.onnx
  - lm_head.onnx
  - model_info.json
"""
    )
    
    parser.add_argument(
        "--model", required=True,
        help="Path to ONNX model file or directory"
    )
    parser.add_argument(
        "--output", required=True,
        help="Base directory for split output"
    )
    parser.add_argument(
        "--split-only", action="store_true",
        help="Only split the model, skip inference validation"
    )
    parser.add_argument(
        "--prompt", default="Test prompt.",
        help="Prompt for inference test"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def main():
    params = parse_args()
    
    print("=" * 60)
    print("ONNX Model Splitter - Semantic Component Extraction")
    print("(True ONNX equivalent of grok_hf_split.py)")
    print("=" * 60)
    print(f"Model: {params.model}")
    print(f"Output: {params.output}")
    print(f"Split Only: {params.split_only}")
    print("=" * 60)
    
    try:
        # Split model
        result = split_onnx_model(params)
        
        if params.split_only:
            print("\n--split-only specified, skipping inference.")
            print(f"Split complete! Files saved to: {result.output_dir}")
        else:
            print("\nValidating split model...")
            run_inference(result.output_dir, params)
        
        # Print summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Total components: {len(result.component_files)}")
        print(f"Transformer layers: {result.num_layers}")
        print(f"Output directory: {result.output_dir}")
        print("\nComponent files:")
        for name, path in sorted(result.component_files.items()):
            print(f"  - {os.path.basename(path)}")
        print("  - model_info.json")
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        if params.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
