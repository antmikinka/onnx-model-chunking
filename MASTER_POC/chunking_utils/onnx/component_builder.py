"""Build per-component ONNX graphs from a detected architecture."""

from collections import defaultdict
from typing import Dict, Set

import onnx

from ..common.architecture import ArchitectureType, get_prefixes
from ..common.architecture_detector import detect_architecture
from ..common.graph_topology import analyze_graph_topology
from .graph_analysis import ResolvedTensorType, resolve_tensor_type
from .types import ComponentInfo


def _onnx_helper():
    """Return the helper object tests patch on the splitter module."""
    from . import splitter
    return splitter.helper


def _architecture_type(value):
    """An ArchitectureType, or the arch_type on a descriptor. Anything else is unset."""
    if isinstance(value, ArchitectureType):
        return value
    arch_type = getattr(value, "arch_type", None)
    if isinstance(arch_type, ArchitectureType):
        return arch_type
    return None


def _separate_architecture(config_json, arch):
    """A descriptor in the config slot is the architecture. A dict or string stays config."""
    positional = _architecture_type(config_json)
    if positional is not None:
        chosen = positional
        config_json = None
    else:
        chosen = None
    if arch is None:
        return config_json, chosen
    explicit = _architecture_type(arch)
    if explicit is None:
        raise ValueError(f"arch must be an ArchitectureType, got {arch!r}")
    if chosen is not None and chosen != explicit:
        raise ValueError(
            f"architecture {chosen.value} and {explicit.value} disagree"
        )
    return config_json, explicit


def extract_components(
    model: onnx.ModelProto,
    config_json=None,
    genai_config=None,
    model_info=None,
    arch=None,
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
            An ArchitectureType or a descriptor with arch_type in this
            position selects the architecture instead of a config.
        genai_config: Optional GenAI config object or JSON string.
        model_info: Optional split model_info object or JSON string.
        arch: Optional ArchitectureType or descriptor. When omitted, the
            architecture is detected from the graph and configs.

    Returns:
        Dictionary mapping component name to ComponentInfo.
    """
    if model is None or not hasattr(model, "graph"):
        raise ValueError("model must contain a graph")
    graph = model.graph
    config_json, chosen = _separate_architecture(config_json, arch)
    if chosen is None:
        chosen = detect_architecture(
            graph,
            config_json=config_json,
            genai_config=genai_config,
            model_info=model_info,
        ).arch_type
    prefixes = get_prefixes(chosen, "onnx")

    print(f"\nDetected architecture: {chosen.value}")
    print(f"  Embed prefix: {prefixes.embed_prefix}")
    print(f"  Layer prefix: {prefixes.layer_prefix}")
    print(f"  Norm prefix: {prefixes.norm_prefix}")
    print(f"  LM head prefix: {prefixes.lm_head_prefix}")

    topology = analyze_graph_topology(
        graph,
        config_json=config_json,
        genai_config=genai_config,
        model_info=model_info,
        arch=chosen,
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

    Output names are chosen from one consumer map. A tensor is an output when
    it is a graph output or a node outside this component reads it. Dtype and
    shape come from value info, then an initializer, then a name heuristic.
    The heuristic tier warns.

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
    
    # Determine outputs (consumed by nodes outside this component OR graph outputs).
    # The consumer map is built once so each output is a membership test, not
    # another walk over the whole graph.
    graph_output_names = {out.name for out in graph.output}
    component_node_set = set(component.node_indices)
    consumers: Dict[str, List[int]] = defaultdict(list)
    for idx, node in enumerate(graph.node):
        for inp in node.input:
            consumers[inp].append(idx)

    # Preserve component node order. A set would follow hash order.
    output_tensors = []
    seen_outputs = set()
    for node in component_nodes:
        for output in node.output:
            if output in seen_outputs:
                continue
            if output in graph_output_names or any(
                consumer_idx not in component_node_set
                for consumer_idx in consumers[output]
            ):
                output_tensors.append(output)
                seen_outputs.add(output)
    
    # If no outputs identified, use all outputs from the last node
    if not output_tensors and component_nodes:
        for output in component_nodes[-1].output:
            if output not in seen_outputs:
                output_tensors.append(output)
                seen_outputs.add(output)
    
    # Build lookups for value info
    init_dict = {i.name: i for i in graph.initializer}
    value_info_dict = {vi.name: vi for vi in graph.value_info}
    for gi in graph.input:
        value_info_dict[gi.name] = gi
    for go in graph.output:
        value_info_dict[go.name] = go
    
    # Create input and output ValueInfoProtos. A complete value info is kept
    # so symbolic dimension names survive. A guessed dtype or shape warns.
    input_value_infos = []
    for inp_name in boundary_inputs:
        if not inp_name:
            continue
        vi = value_info_dict.get(inp_name)
        resolved = resolve_tensor_type(inp_name, vi, init_dict.get(inp_name), role="input")
        input_value_infos.append(_component_value_info(inp_name, vi, resolved))

    output_value_infos = []
    for out_name in output_tensors:
        if not out_name:
            continue
        vi = value_info_dict.get(out_name)
        resolved = resolve_tensor_type(out_name, vi, init_dict.get(out_name), role="output")
        output_value_infos.append(_component_value_info(out_name, vi, resolved))
    
    # Create the graph
    component_graph = _onnx_helper().make_graph(
        nodes=component_nodes,
        name=f"{component_name}_graph",
        inputs=input_value_infos,
        outputs=output_value_infos,
        initializer=component_initializers
    )
    
    # Create the model
    component_model = _onnx_helper().make_model(
        component_graph,
        producer_name=f"chunking_utils_onnx_{component_name}",
        opset_imports=list(model.opset_import)
    )
    
    # Copy IR version
    component_model.ir_version = model.ir_version
    
    return component_model

def _component_value_info(name: str, value, resolved: ResolvedTensorType):
    """Reuse a complete value info. A filled-in type becomes a new proto."""
    if resolved.dtype_source == "value_info" and resolved.shape_source == "value_info" and value is not None:
        return value
    return _onnx_helper().make_tensor_value_info(name, resolved.elem_type, list(resolved.shape))
