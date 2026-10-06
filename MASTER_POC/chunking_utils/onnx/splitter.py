"""
ONNX model splitting utilities.

This module provides the main splitting functionality for ONNX models,
creating per-component ONNX files for embeddings, layers, norm, and lm_head.
"""

import os
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
from ..common.architecture_detector import detect_architecture
from ..common.graph_topology import analyze_graph_topology
from ..common.types import BoundaryTensor, ModelFormat, SplitConfig

from .types import ComponentInfo, OnnxSplitResult
from .loader import load_onnx_model


def detect_architecture_from_onnx(graph) -> ArchitectureType:
    """
    Detect model architecture from an ONNX graph.

    The descriptor lives in ``architecture_detector``. This wrapper returns
    the architecture type used to choose ONNX prefixes.

    Args:
        graph: ONNX graph.

    Returns:
        Detected ArchitectureType.
    """
    return detect_architecture(graph).arch_type


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


def extract_components(
    model: onnx.ModelProto,
    config_json=None,
    genai_config=None,
    model_info=None,
) -> Dict[str, ComponentInfo]:
    """
    Extract semantic components from the ONNX model.

    Classification uses node-name patterns, graph topology, initializer
    ownership, and optional config metadata. A fused or renamed node joins
    the layer that encloses it. Creates one ComponentInfo for:

    - embedding
    - layer_0, layer_1, ..., layer_N (one per layer)
    - norm
    - lm_head
    - other (nodes that stay outside every closure)

    Args:
        model: ONNX model.
        config_json: Optional Hugging Face config object or JSON string.
        genai_config: Optional GenAI config object or JSON string.
        model_info: Optional split model_info object or JSON string.

    Returns:
        Dictionary mapping component name to ComponentInfo.
    """
    if model is None or not hasattr(model, "graph"):
        raise ValueError("model must contain a graph")
    graph = model.graph
    arch = detect_architecture(
        graph,
        config_json=config_json,
        genai_config=genai_config,
        model_info=model_info,
    ).arch_type
    prefixes = get_prefixes(arch, "onnx")

    print(f"\nDetected architecture: {arch.value}")
    print(f"  Embed prefix: {prefixes.embed_prefix}")
    print(f"  Layer prefix: {prefixes.layer_prefix}")
    print(f"  Norm prefix: {prefixes.norm_prefix}")
    print(f"  LM head prefix: {prefixes.lm_head_prefix}")

    topology = analyze_graph_topology(
        graph,
        config_json=config_json,
        genai_config=genai_config,
        model_info=model_info,
    )
    nodes = list(graph.node)
    owned: Dict[int, Set[str]] = {}
    for init_name, indices in topology.initializer_ownership.items():
        for index in indices:
            owned.setdefault(index, set()).add(init_name)

    components: Dict[str, ComponentInfo] = {}
    for group in topology.groups:
        info = ComponentInfo(
            component_type=group.component_type,
            layer_id=group.layer_id,
        )
        for index in group.node_indices:
            node = nodes[index]
            info.node_indices.append(index)
            info.node_names.append(getattr(node, "name", None) or "")
            info.initializer_names.update(owned.get(index, ()))
        components[group.name] = info

    print(f"\nExtracted {len(components)} components:")
    for name, info in sorted(components.items()):
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


_ONNX_ELEM_DTYPE = {
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    9: "bool",
    10: "float16",
    11: "float64",
    12: "uint32",
    13: "uint64",
    16: "bfloat16",
}


def _boundary_graph(model):
    """Return the ONNX graph object. A model is unwrapped. A non-graph raises."""
    if model is None:
        raise ValueError("graph must be an ONNX graph")
    graph = model
    if not hasattr(graph, "node") and hasattr(graph, "graph"):
        graph = getattr(graph, "graph")
    nodes = getattr(graph, "node", None)
    if nodes is None or isinstance(nodes, (str, bytes)):
        raise ValueError("graph must be an ONNX graph")
    try:
        return graph, list(nodes)
    except TypeError as exc:
        raise ValueError("graph must be an ONNX graph") from exc


def _component_owners(components, node_count: int) -> Dict[int, str]:
    """Map each node index to the component that owns it."""
    if not isinstance(components, dict):
        raise ValueError(f"components must be a dict, got {components!r}")
    owners: Dict[int, str] = {}
    for raw_name, info in components.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f"component name must be a non-empty string, got {raw_name!r}")
        name = raw_name.strip()
        indices = getattr(info, "node_indices", None)
        if indices is None and isinstance(info, dict):
            indices = info.get("node_indices")
        if indices is None or isinstance(indices, (str, bytes)):
            raise ValueError(f"component {name} must list node_indices")
        try:
            index_list = list(indices)
        except TypeError as exc:
            raise ValueError(f"component {name} must list node_indices") from exc
        for index in index_list:
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError(
                    f"component {name} node index must be an integer, got {index!r}"
                )
            if index < 0 or index >= node_count:
                raise ValueError(
                    f"component {name} node index {index} is outside the graph"
                )
            previous = owners.get(index)
            if previous is not None:
                raise ValueError(f"node {index} belongs to {previous} and {name}")
            owners[index] = name
    for index in range(node_count):
        if index not in owners:
            raise ValueError(f"node {index} is not in a component")
    return owners


def _value_info_map(graph) -> Dict[str, Any]:
    """Tensor name to the value info that carries dtype and shape."""
    found: Dict[str, Any] = {}
    for collection in (
        getattr(graph, "value_info", ()) or (),
        getattr(graph, "input", ()) or (),
        getattr(graph, "output", ()) or (),
    ):
        for item in collection:
            name = getattr(item, "name", None)
            if isinstance(name, str) and name and name not in found:
                found[name] = item
    return found


def _dims_from_value(value) -> Optional[List[Optional[int]]]:
    """Read a shape. A symbolic axis is None. A missing shape stays unset."""
    type_proto = getattr(value, "type", None)
    tensor_type = getattr(type_proto, "tensor_type", None) if type_proto is not None else None
    if tensor_type is None:
        return None
    shape_proto = getattr(tensor_type, "shape", None)
    if shape_proto is None:
        return None
    dims = getattr(shape_proto, "dim", None)
    if dims is None:
        return None
    parsed: List[Optional[int]] = []
    for dim in dims:
        param = getattr(dim, "dim_param", "") or ""
        if isinstance(param, str) and param:
            parsed.append(None)
            continue
        number = getattr(dim, "dim_value", 0)
        if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
            parsed.append(None)
        else:
            parsed.append(number)
    return parsed


def _dtype_from_value(value, name: str) -> Optional[str]:
    """ONNX element type as a dtype name. An unset type stays unset."""
    type_proto = getattr(value, "type", None)
    tensor_type = getattr(type_proto, "tensor_type", None) if type_proto is not None else None
    if tensor_type is None:
        return None
    elem = getattr(tensor_type, "elem_type", 0)
    if isinstance(elem, bool) or not isinstance(elem, int) or elem == 0:
        return None
    dtype = _ONNX_ELEM_DTYPE.get(elem)
    if dtype is None:
        raise ValueError(f"boundary {name} has unsupported ONNX element type {elem}")
    return dtype


def _heuristic_dtype(name: str) -> str:
    """Dtype used when the graph does not record an element type."""
    lowered = name.lower()
    if any(word in lowered for word in ("position", "mask", "idx", "constants")):
        return "int64"
    return "float16"


def _heuristic_shape(name: str) -> List[Optional[int]]:
    """Rank guessed from a 2D, 3D, or 4D marker in the tensor name."""
    rank = 1
    if "4D" in name:
        rank = 4
    elif "3D" in name:
        rank = 3
    elif "2D" in name:
        rank = 2
    return [None] * rank


def _unique_component_names(names: List[str]) -> List[str]:
    seen = set()
    unique: List[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        unique.append(name)
    return unique


def collect_boundary_tensors(model, components) -> List[BoundaryTensor]:
    """
    Build the boundary contract for a split ONNX graph.

    A boundary is a tensor that leaves a component: a graph input, a graph
    output, a stateful KV tensor, or a value one component writes and another
    reads. Initializers stay inside the component that owns them. Dtype and
    shape come from value info. A missing type uses the same name heuristic
    as ``create_component_onnx``. KV tensors prefer the NPU and are zero-copy.

    Args:
        model: ONNX model or graph.
        components: Component name to an object with ``node_indices``.

    Returns:
        Boundary tensors in graph-input order, then producer order.

    Raises:
        ValueError: If the graph, a node index, or an element type is invalid.
    """
    graph, nodes = _boundary_graph(model)
    owners = _component_owners(components, len(nodes))
    initializer_names = set()
    for init in getattr(graph, "initializer", ()) or ():
        name = getattr(init, "name", None)
        if isinstance(name, str) and name:
            initializer_names.add(name)

    producer_node: Dict[str, int] = {}
    consumer_nodes: Dict[str, List[int]] = {}
    for index, node in enumerate(nodes):
        for output in getattr(node, "output", ()) or ():
            if isinstance(output, str) and output and output not in producer_node:
                producer_node[output] = index
        for inp in getattr(node, "input", ()) or ():
            if isinstance(inp, str) and inp:
                consumer_nodes.setdefault(inp, []).append(index)

    graph_inputs = []
    for item in getattr(graph, "input", ()) or ():
        name = getattr(item, "name", None)
        if isinstance(name, str) and name and name not in initializer_names:
            graph_inputs.append(name)
    graph_outputs = []
    for item in getattr(graph, "output", ()) or ():
        name = getattr(item, "name", None)
        if isinstance(name, str) and name:
            graph_outputs.append(name)
    input_names = set(graph_inputs)
    output_names = set(graph_outputs)
    info = _value_info_map(graph)

    ordered: List[str] = []
    seen = set()
    for name in graph_inputs:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    produced = sorted(
        producer_node.items(),
        key=lambda item: (item[1], item[0]),
    )
    for name, _index in produced:
        if name in initializer_names or name in seen:
            continue
        ordered.append(name)
        seen.add(name)
    for name in graph_outputs:
        if name not in seen and name not in initializer_names:
            ordered.append(name)
            seen.add(name)

    boundaries: List[BoundaryTensor] = []
    for name in ordered:
        writer = producer_node.get(name)
        producer = owners[writer] if writer is not None else "external"
        if writer is None and name not in input_names and name not in output_names:
            continue
        consumer_components = _unique_component_names(
            [owners[index] for index in consumer_nodes.get(name, [])]
        )
        other = [item for item in consumer_components if item != producer]
        stateful = (
            name.lower() == "past_key_values"
            or name.lower().startswith("past_key_values.")
            or name.lower() == "present"
            or name.lower().startswith("present.")
        )
        activation = name in {"hidden_states", "attention_mask", "position_ids"}
        crosses = bool(other) or name in output_names or (name in input_names and consumer_components)
        if not stateful and not activation and not crosses:
            continue
        value = info.get(name)
        dtype = _dtype_from_value(value, name) if value is not None else None
        shape = _dims_from_value(value) if value is not None else None
        if dtype is None:
            dtype = _heuristic_dtype(name)
        if shape is None:
            shape = _heuristic_shape(name)
        boundaries.append(BoundaryTensor(
            name=name,
            dtype=dtype,
            shape=shape,
            producer_component=producer,
            consumer_components=consumer_components,
            is_stateful=stateful,
        ))
    return boundaries


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
    
    # Extract components and the tensors that cross them
    components = extract_components(onnx_model)
    boundaries = collect_boundary_tensors(onnx_model, components)
    
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
        },
        "boundary_tensors": [tensor.to_dict() for tensor in boundaries],
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
        boundary_tensors=boundaries,
    )
