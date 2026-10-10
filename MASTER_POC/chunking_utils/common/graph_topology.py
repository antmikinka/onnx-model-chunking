"""
Graph-topology component grouping for ONNX splitters.

Name patterns still seed embeddings, layers, the final norm, and the lm head.
Fused kernels, quantized contrib ops, MoE expert subgraphs, and sparse-attention
branches often lose those names. Grouping then uses four structures:

- producer map: tensor name to the node index that writes it
- consumer map: tensor name to the node indices that read it
- initializer ownership: weight name to the node indices that read it
- topological order: every node index, dependencies first

An unlabeled node joins a seed when the nearest labeled producers and
consumers agree. A side branch joins its producer when it does not write a
graph output. ``num_hidden_layers`` or ``layer_types`` splits a fully unnamed
backbone between the embedding and the final norm into contiguous layers.
"""

import json
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Tuple
from dataclasses import dataclass

from .architecture import ArchitectureType, get_prefixes
from .architecture_detector import detect_architecture


_NESTED_CONFIG_KEYS = {
    "model",
    "decoder",
    "text_config",
    "config",
    "quantization_config",
    "quantization",
    "quant_config",
    "architecture_extras",
}

_LAYER_COUNT_KEYS = ("num_hidden_layers", "n_layer", "num_layers")


@dataclass(frozen=True)
class NodeGroup:
    """
    One dependency closure after topology classification.

    ``node_indices`` follows topological order. A layer name is ``layer_N``.
    """

    name: str
    component_type: str
    layer_id: Optional[int]
    node_indices: Tuple[int, ...]


@dataclass
class GraphTopology:
    """
    Maps and groups for one ONNX graph.

    Attributes:
        producer_map: Output tensor name to the node index that writes it.
        consumer_map: Tensor name to the node indices that read it.
        initializer_ownership: Initializer name to the node indices that read it.
        topological_order: Node indices with producers before consumers.
        groups: Classified dependency closures.
    """

    producer_map: Dict[str, int]
    consumer_map: Dict[str, List[int]]
    initializer_ownership: Dict[str, List[int]]
    topological_order: List[int]
    groups: List[NodeGroup]


@dataclass(frozen=True)
class _Label:
    component_type: str
    layer_id: Optional[int] = None

    @property
    def name(self) -> str:
        if self.component_type == "layer":
            return f"layer_{self.layer_id}"
        return self.component_type


def _prefix_architecture(arch) -> ArchitectureType:
    """ArchitectureType, or the arch_type on a descriptor."""
    if isinstance(arch, ArchitectureType):
        return arch
    arch_type = getattr(arch, "arch_type", None)
    if isinstance(arch_type, ArchitectureType):
        return arch_type
    raise ValueError(f"arch must be an ArchitectureType, got {arch!r}")


def analyze_graph_topology(
    graph,
    config_json=None,
    genai_config=None,
    model_info=None,
    arch=None,
) -> GraphTopology:
    """
    Build topology maps and classify nodes into components.

    ``config_json``, ``genai_config``, and ``model_info`` are objects or JSON
    strings. A later source overrides an earlier layer count. A fused node
    whose name has no layer id joins the surrounding layer. A RyzenAI name
    such as ``gqo_8_1_0`` is still a layer seed. Broken JSON is rejected.

    Args:
        graph: ONNX graph or model.
        config_json: Hugging Face ``config.json`` object.
        genai_config: GenAI config object.
        model_info: Split ``model_info.json`` object.
        arch: Optional ArchitectureType or descriptor. When omitted, the
            architecture is detected from the graph and configs.

    Returns:
        GraphTopology with maps, topological order, and component groups.
    """
    graph, nodes = _as_graph(graph)
    configs = (
        _as_object(config_json, "config_json"),
        _as_object(genai_config, "genai_config"),
        _as_object(model_info, "model_info"),
    )
    if arch is None:
        detected = detect_architecture(
            graph,
            config_json=config_json,
            genai_config=genai_config,
            model_info=model_info,
        ).arch_type
    else:
        detected = _prefix_architecture(arch)
    prefixes = get_prefixes(detected, "onnx")
    embed_prefix = prefixes.embed_prefix.lower()
    layer_prefix = prefixes.layer_prefix.lower()
    norm_prefix = prefixes.norm_prefix.lower()
    lm_prefix = prefixes.lm_head_prefix.lower()

    producer_map: Dict[str, int] = {}
    consumer_map: Dict[str, List[int]] = {}
    node_inputs: List[List[str]] = []
    node_outputs: List[List[str]] = []
    for index, node in enumerate(nodes):
        outputs = _name_list(getattr(node, "output", None))
        inputs = _name_list(getattr(node, "input", None))
        node_outputs.append(outputs)
        node_inputs.append(inputs)
        for tensor in outputs:
            producer_map.setdefault(tensor, index)
            consumer_map.setdefault(tensor, [])
        for tensor in inputs:
            consumer_map.setdefault(tensor, []).append(index)

    for tensor, users in consumer_map.items():
        consumer_map[tensor] = _unique(users)

    initializer_names = []
    seen_inits = set()
    for init in getattr(graph, "initializer", ()) or ():
        name = getattr(init, "name", None)
        if isinstance(name, str) and name and name not in seen_inits:
            seen_inits.add(name)
            initializer_names.append(name)
    ownership = {
        name: list(consumer_map.get(name, []))
        for name in initializer_names
    }
    init_name_set = set(initializer_names)

    successors, predecessors = _edges(len(nodes), producer_map, consumer_map)
    topological_order = _topological_order(successors, predecessors)
    graph_writers = _graph_writers(graph, node_outputs)

    labels: Dict[int, _Label] = {}
    for index, node in enumerate(nodes):
        label = _label_from_name(
            getattr(node, "name", None) or "",
            embed_prefix,
            layer_prefix,
            norm_prefix,
            lm_prefix,
        )
        if label is not None:
            labels[index] = label
    for index, inputs in enumerate(node_inputs):
        if index in labels:
            continue
        label = _label_from_initializers(
            inputs,
            init_name_set,
            embed_prefix,
            layer_prefix,
            norm_prefix,
            lm_prefix,
        )
        if label is not None:
            labels[index] = label

    _assign_configured_layers(
        labels,
        topological_order,
        successors,
        predecessors,
        _layer_count(configs),
    )
    _assign_dependency_closures(
        labels,
        successors,
        predecessors,
        graph_writers,
    )

    grouped: Dict[str, List[int]] = {}
    stored: Dict[str, _Label] = {}
    for index in topological_order:
        label = labels.get(index, _Label("other"))
        grouped.setdefault(label.name, []).append(index)
        stored[label.name] = label
    groups = [
        NodeGroup(
            name=name,
            component_type=stored[name].component_type,
            layer_id=stored[name].layer_id,
            node_indices=tuple(grouped[name]),
        )
        for name in sorted(stored, key=lambda item: _group_key(stored[item]))
    ]
    return GraphTopology(
        producer_map=producer_map,
        consumer_map=consumer_map,
        initializer_ownership=ownership,
        topological_order=topological_order,
        groups=groups,
    )


def _as_graph(graph: object) -> Tuple[object, List[Any]]:
    if graph is None:
        raise ValueError("graph must be an ONNX graph")
    if not hasattr(graph, "node") and hasattr(graph, "graph"):
        graph = getattr(graph, "graph")
    nodes = getattr(graph, "node", None)
    if nodes is None or isinstance(nodes, (str, bytes)):
        raise ValueError("graph must be an ONNX graph")
    try:
        return graph, list(nodes)
    except TypeError as exc:
        raise ValueError("graph must be an ONNX graph") from exc


def _as_object(value: object, label: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{label} must be an object")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} is not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{label} is not an object")
        return parsed
    raise ValueError(f"{label} must be an object")


def _iter_dicts(payload: Dict[str, Any], depth: int = 0) -> Iterable[Dict[str, Any]]:
    yield payload
    if depth >= 4:
        return
    for key, value in payload.items():
        if key in _NESTED_CONFIG_KEYS and isinstance(value, dict):
            yield from _iter_dicts(value, depth + 1)


def _layer_count(payloads: Iterable[Dict[str, Any]]) -> int:
    found = 0
    for payload in payloads:
        chosen = 0
        for blob in _iter_dicts(payload):
            for key in _LAYER_COUNT_KEYS:
                value = blob.get(key)
                if type(value) is int and value > 0:
                    chosen = value
                    break
            if chosen:
                break
        if chosen:
            found = chosen
    if found:
        return found
    types: List[str] = []
    for payload in payloads:
        chosen_list = None
        for blob in _iter_dicts(payload):
            value = blob.get("layer_types")
            if isinstance(value, list) and value:
                chosen_list = [
                    item.strip()
                    for item in value
                    if isinstance(item, str) and item.strip()
                ]
                if chosen_list:
                    break
        if chosen_list:
            types = chosen_list
    return len(types)


def _name_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    names = []
    for item in value:
        if isinstance(item, str) and item:
            names.append(item)
    return names


def _unique(values: List[int]) -> List[int]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _edges(
    count: int,
    producer_map: Dict[str, int],
    consumer_map: Dict[str, List[int]],
) -> Tuple[List[List[int]], List[List[int]]]:
    successor_sets = [set() for _ in range(count)]
    for tensor, producer in producer_map.items():
        for consumer in consumer_map.get(tensor, []):
            if consumer != producer:
                successor_sets[producer].add(consumer)
    successors = [sorted(items) for items in successor_sets]
    predecessors: List[List[int]] = [[] for _ in range(count)]
    for index, targets in enumerate(successors):
        for target in targets:
            predecessors[target].append(index)
    return successors, predecessors


def _topological_order(
    successors: List[List[int]],
    predecessors: List[List[int]],
) -> List[int]:
    count = len(successors)
    indegree = [len(items) for items in predecessors]
    ready = deque(index for index in range(count) if indegree[index] == 0)
    order: List[int] = []
    while ready:
        index = ready.popleft()
        order.append(index)
        for target in successors[index]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if len(order) != count:
        seen = set(order)
        for index in range(count):
            if index not in seen:
                order.append(index)
    return order


def _graph_writers(graph: object, node_outputs: List[List[str]]) -> List[bool]:
    names = set()
    for item in getattr(graph, "output", ()) or ():
        if isinstance(item, str) and item:
            names.add(item)
            continue
        name = getattr(item, "name", None)
        if isinstance(name, str) and name:
            names.add(name)
    return [any(tensor in names for tensor in outputs) for outputs in node_outputs]


def _layer_id_from_text(lowered: str, layer_prefix: str) -> Optional[int]:
    if layer_prefix and layer_prefix in lowered:
        suffix = lowered.split(layer_prefix)[-1]
        match = _leading_int(suffix)
        if match is not None:
            return match
    for token in ("layers.", "layers/"):
        start = 0
        while True:
            found = lowered.find(token, start)
            if found < 0:
                break
            value = _leading_int(lowered[found + len(token):])
            if value is not None:
                return value
            start = found + len(token)
    if "h." in lowered:
        start = 0
        while True:
            found = lowered.find("h.", start)
            if found < 0:
                break
            value = _leading_int(lowered[found + 2:])
            if value is not None:
                return value
            start = found + 2
    tail = lowered.rsplit("/", 1)[-1]
    return _ryzen_layer_id(tail)


def _leading_int(text: str) -> Optional[int]:
    end = 0
    while end < len(text) and text[end].isdigit():
        end += 1
    if end == 0:
        return None
    return int(text[:end])


def _ryzen_layer_id(tail: str) -> Optional[int]:
    """
    Layer id from a compiled RyzenAI op name.

    ``matmulnbits_k_6_0``, ``gqo_8_1_0``, and ``ssmlp_3_0_0`` keep the last
    number as the layer id. Other tails are not compiled layer names.
    """
    parts = tail.split("_")
    if len(parts) == 4 and parts[0] == "matmulnbits" and len(parts[1]) == 1 and parts[1].isalpha():
        if parts[2].isdigit() and parts[3].isdigit():
            return int(parts[3])
    if len(parts) == 4 and parts[0] in ("gqo", "ssmlp"):
        if parts[1].isdigit() and parts[2].isdigit() and parts[3].isdigit():
            return int(parts[3])
    return None


def _label_from_name(
    name: str,
    embed_prefix: str,
    layer_prefix: str,
    norm_prefix: str,
    lm_prefix: str,
) -> Optional[_Label]:
    lowered = name.lower()
    if not lowered:
        return None
    if (embed_prefix and embed_prefix in lowered) or "embed_tokens" in lowered:
        return _Label("embedding")
    layer_id = _layer_id_from_text(lowered, layer_prefix)
    if layer_id is not None:
        return _Label("layer", layer_id)
    if (
        norm_prefix
        and norm_prefix in lowered
        and "layers" not in lowered
        and "layer_norm" not in lowered
    ):
        return _Label("norm")
    if "model.norm" in lowered or "/model/norm" in lowered or ".norm." in lowered:
        if "layers" not in lowered and "layer_norm" not in lowered:
            return _Label("norm")
    if (lm_prefix and lm_prefix in lowered) or "lm_head" in lowered:
        return _Label("lm_head")
    if "mtp_head" in lowered or "multi_token" in lowered or "/mtp" in lowered or ".mtp" in lowered:
        return _Label("other")
    if "ngram" in lowered:
        return _Label("other")
    return None


def _label_from_initializers(
    inputs: List[str],
    initializer_names: set,
    embed_prefix: str,
    layer_prefix: str,
    norm_prefix: str,
    lm_prefix: str,
) -> Optional[_Label]:
    signals = []
    for name in inputs:
        if name not in initializer_names:
            continue
        label = _label_from_name(
            name,
            embed_prefix,
            layer_prefix,
            norm_prefix,
            lm_prefix,
        )
        if label is not None:
            signals.append(label)
    if not signals:
        return None
    first = signals[0]
    if any(item != first for item in signals):
        return None
    return first


def _assign_configured_layers(
    labels: Dict[int, _Label],
    topological_order: List[int],
    successors: List[List[int]],
    predecessors: List[List[int]],
    layer_count: int,
) -> None:
    if layer_count <= 0:
        return
    embed_nodes = [
        index for index, label in labels.items() if label.component_type == "embedding"
    ]
    right_nodes = [
        index
        for index, label in labels.items()
        if label.component_type in ("norm", "lm_head")
    ]
    if not embed_nodes or not right_nodes:
        return
    position = {index: place for place, index in enumerate(topological_order)}
    left = max(position[index] for index in embed_nodes)
    right = min(position[index] for index in right_nodes)
    if left >= right:
        return
    window = set(topological_order[left + 1:right])
    if any(
        labels.get(index) is not None and labels[index].component_type == "layer"
        for index in window
    ):
        return
    reached_from_left = _reachable(embed_nodes, successors)
    reaches_right = _reachable(right_nodes, predecessors)
    span = [
        index
        for index in topological_order
        if index in window
        and index not in labels
        and reached_from_left[index]
        and reaches_right[index]
    ]
    if not span or len(span) % layer_count != 0:
        return
    size = len(span) // layer_count
    if size <= 0:
        return
    for layer_id in range(layer_count):
        start = layer_id * size
        for index in span[start:start + size]:
            labels[index] = _Label("layer", layer_id)


def _reachable(seeds: List[int], neighbors: List[List[int]]) -> List[bool]:
    seen = [False] * len(neighbors)
    stack = []
    for index in seeds:
        if not seen[index]:
            seen[index] = True
            stack.append(index)
    while stack:
        index = stack.pop()
        for target in neighbors[index]:
            if not seen[target]:
                seen[target] = True
                stack.append(target)
    return seen


def _assign_dependency_closures(
    labels: Dict[int, _Label],
    successors: List[List[int]],
    predecessors: List[List[int]],
    graph_writers: List[bool],
) -> None:
    count = len(successors)
    for _ in range(count + 1):
        backward = _nearest(labels, predecessors, count)
        forward = _nearest(labels, successors, count)
        assigned = False
        for index in range(count):
            if index in labels:
                continue
            choice = _closure_label(
                backward[index],
                forward[index],
                graph_writers[index],
            )
            if choice is None:
                continue
            labels[index] = choice
            assigned = True
        if not assigned:
            return


def _nearest(
    labels: Dict[int, _Label],
    neighbors: List[List[int]],
    count: int,
) -> List[set]:
    found = [set() for _ in range(count)]
    pending = []
    for index in range(count):
        label = labels.get(index)
        if label is not None:
            found[index].add(label)
        else:
            pending.append(index)
    for _ in range(len(pending) + 1):
        changed = False
        for index in pending:
            if index in labels:
                continue
            acc = set()
            for neighbor in neighbors[index]:
                if neighbor in labels:
                    acc.add(labels[neighbor])
                else:
                    acc |= found[neighbor]
            if acc != found[index]:
                found[index] = acc
                changed = True
        if not changed:
            break
    return found


def _closure_label(backward: set, forward: set, writes_graph_output: bool) -> Optional[_Label]:
    if len(backward) == 1 and backward == forward:
        return next(iter(backward))
    if writes_graph_output:
        return None
    if len(backward) == 1 and not forward:
        return next(iter(backward))
    if len(forward) == 1 and not backward:
        return next(iter(forward))
    return None


def _group_key(label: _Label) -> Tuple[int, int, str]:
    order = {"embedding": 0, "layer": 1, "norm": 2, "lm_head": 3, "other": 4}
    layer_id = label.layer_id if label.layer_id is not None else -1
    return (order.get(label.component_type, 5), layer_id, label.name)
