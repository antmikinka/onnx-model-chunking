"""Boundary tensors that cross ONNX components."""

from typing import Any, Dict, List

from ..common.types import BoundaryTensor
from .graph_analysis import _boundary_graph, _value_info_map, resolve_tensor_type

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
    shape use value info, then an initializer, then a name heuristic. The
    heuristic tier warns. KV tensors prefer the NPU and are zero-copy.

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
        resolved = resolve_tensor_type(name, value, role="boundary")
        boundaries.append(BoundaryTensor(
            name=name,
            dtype=resolved.dtype,
            shape=list(resolved.shape),
            producer_component=producer,
            consumer_components=consumer_components,
            is_stateful=stateful,
        ))
    return boundaries
