"""
ONNX model splitting utilities.

This module provides the main splitting functionality for ONNX models,
creating per-component ONNX files for embeddings, layers, norm, and lm_head.
"""

import os
import re
import json
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Set

import onnx
from onnx import numpy_helper, helper
from onnx import shape_inference

from ..common.path_utils import sanitize_model_name
from ..common.architecture import (
    ArchitectureType,
    ONNX_PREFIXES,
    get_prefixes,
    quantized_op_scheme,
)
from ..common.architecture_detector import detect_architecture
from ..common.graph_topology import analyze_graph_topology
from ..common.types import BoundaryTensor, ComponentType, ModelFormat, SplitConfig

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


def _heuristic_dtype(name: str) -> str:
    """Dtype used when the graph does not record an element type.

    Index-like names use int64 on both inputs and outputs so a guessed mask
    does not change width at a component boundary.
    """
    lowered = name.lower()
    if any(word in lowered for word in ("position", "mask", "idx", "constants", "count")):
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


@dataclass(frozen=True)
class ResolvedTensorType:
    """Dtype and shape chosen for one boundary tensor.

    ``dtype_source`` and ``shape_source`` are ``value_info``, ``initializer``,
    or ``heuristic``. A heuristic field is a guess from the tensor name.
    """

    name: str
    dtype: str
    elem_type: int
    shape: List[Optional[int]]
    dtype_source: str
    shape_source: str


def _elem_type_code(value) -> Optional[int]:
    """Nonzero ONNX element type, or None when the value info leaves it unset."""
    type_proto = getattr(value, "type", None)
    tensor_type = getattr(type_proto, "tensor_type", None) if type_proto is not None else None
    if tensor_type is None:
        return None
    elem = getattr(tensor_type, "elem_type", 0)
    if isinstance(elem, bool) or not isinstance(elem, int) or elem == 0:
        return None
    return elem


def _dtype_name_for_elem(elem: int, name: str) -> str:
    dtype = _ONNX_ELEM_DTYPE.get(elem)
    if dtype is None:
        raise ValueError(f"boundary {name} has unsupported ONNX element type {elem}")
    return dtype


def _elem_for_dtype(dtype: str) -> int:
    for code, name in _ONNX_ELEM_DTYPE.items():
        if name == dtype:
            return code
    raise ValueError(f"boundary dtype {dtype} has no ONNX element type")


def _initializer_dtype_and_shape(initializer, name: str):
    """Tier 2: initializer data type and dimensions. Unset type stays unset."""
    if initializer is None:
        return None, None
    elem = getattr(initializer, "data_type", 0)
    if isinstance(elem, bool) or not isinstance(elem, int):
        raise ValueError(
            f"boundary {name} initializer data_type must be an integer, got {elem!r}"
        )
    dtype = None if elem == 0 else _dtype_name_for_elem(elem, name)
    dims = getattr(initializer, "dims", None)
    if dims is None:
        return dtype, None
    shape: List[Optional[int]] = []
    for dim in dims:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise ValueError(
                f"boundary {name} initializer dim must be an integer, got {dim!r}"
            )
        if dim < 0:
            raise ValueError(f"boundary {name} initializer dim {dim} is negative")
        shape.append(dim)
    return dtype, shape


def _warn_heuristic_type(role: str, name: str, dtype: str, shape, guessed_dtype: bool, guessed_shape: bool) -> None:
    parts = []
    if guessed_dtype:
        parts.append(f"dtype {dtype}")
    if guessed_shape:
        parts.append(f"shape {list(shape)}")
    warnings.warn(
        f"{role} tensor {name} uses a name heuristic for {' and '.join(parts)} "
        f"because the ONNX graph does not record it",
        UserWarning,
        stacklevel=3,
    )


def resolve_tensor_type(name, value=None, initializer=None, *, role: str = "boundary") -> ResolvedTensorType:
    """
    Resolve one tensor dtype and shape in three tiers.

    Tier 1 is value info, a graph input, or a graph output. Tier 2 is an
    initializer. Tier 3 is the name heuristic. Dtype and shape are chosen
    separately, so a declared dtype stays in place when only the shape is
    missing. The heuristic tier warns. An unknown element type, a blank
    name, and a bad initializer dimension raise.

    Args:
        name: Tensor name.
        value: Optional value info or graph input or output.
        initializer: Optional initializer with ``data_type`` and ``dims``.
        role: ``input``, ``output``, or ``boundary``. Named in the warning.

    Returns:
        Resolved dtype name, ONNX element type, shape, and the tier of each.
    """
    if role not in {"input", "output", "boundary"}:
        raise ValueError(f"tensor role must be input, output, or boundary, got {role!r}")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{role} tensor name must be a non-empty string, got {name!r}")

    declared_dtype = None
    declared_elem = None
    declared_shape = None
    if value is not None:
        declared_elem = _elem_type_code(value)
        if declared_elem is not None:
            declared_dtype = _dtype_name_for_elem(declared_elem, name)
        declared_shape = _dims_from_value(value)

    init_dtype, init_shape = _initializer_dtype_and_shape(initializer, name)
    dtype = declared_dtype if declared_dtype is not None else init_dtype
    shape = declared_shape if declared_shape is not None else init_shape
    dtype_source = "value_info" if declared_dtype is not None else ("initializer" if init_dtype is not None else None)
    shape_source = "value_info" if declared_shape is not None else ("initializer" if init_shape is not None else None)
    guessed_dtype = dtype is None
    guessed_shape = shape is None
    if guessed_dtype:
        dtype = _heuristic_dtype(name)
        dtype_source = "heuristic"
        declared_elem = _elem_for_dtype(dtype)
    elif declared_elem is None:
        declared_elem = _elem_for_dtype(dtype)
    if guessed_shape:
        shape = _heuristic_shape(name)
        shape_source = "heuristic"
    if guessed_dtype or guessed_shape:
        _warn_heuristic_type(role, name, dtype, shape, guessed_dtype, guessed_shape)
    return ResolvedTensorType(
        name=name,
        dtype=dtype,
        elem_type=declared_elem,
        shape=list(shape),
        dtype_source=dtype_source,
        shape_source=shape_source,
    )


def _component_value_info(name: str, value, resolved: ResolvedTensorType):
    """Reuse a complete value info. A filled-in type becomes a new proto."""
    if resolved.dtype_source == "value_info" and resolved.shape_source == "value_info" and value is not None:
        return value
    return helper.make_tensor_value_info(name, resolved.elem_type, list(resolved.shape))


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


_QUANT_EXTRA_ELEM = {
    21: "uint4",
    22: "int4",
}
_QUANT_BITS = {2, 3, 4, 8}
_QUANT_INFO_KEYS = (
    "method",
    "weight_dtype",
    "activation_dtype",
    "group_size",
    "has_zero_point",
    "packing_format",
    "bits",
    "scale_dtype",
    "zero_point_dtype",
)


def _empty_quantization_info() -> Dict[str, Any]:
    """A dense graph has no quantization contract. The key is still recorded."""
    return {
        "method": None,
        "weight_dtype": None,
        "activation_dtype": None,
        "group_size": None,
        "has_zero_point": False,
        "packing_format": None,
        "bits": None,
        "scale_dtype": None,
        "zero_point_dtype": None,
    }


def _quant_node_label(node) -> str:
    name = getattr(node, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    op_type = getattr(node, "op_type", None)
    if isinstance(op_type, str) and op_type.strip():
        return op_type.strip()
    return "node"


def _quant_kind(node) -> Optional[str]:
    """MatMulNBits, AWQ, GPTQ, or QLinearMatMul. A dense op stays unset."""
    op_type = getattr(node, "op_type", "") or ""
    name = getattr(node, "name", "") or ""
    domain = getattr(node, "domain", "") or ""
    if not isinstance(op_type, str):
        op_type = ""
    if not isinstance(name, str):
        name = ""
    if not isinstance(domain, str):
        domain = ""
    text = " ".join(part.lower() for part in (op_type, name, domain))
    if "gptq" in text:
        return "GPTQ"
    if "awq" in text:
        return "AWQ"
    if not op_type.strip():
        return None
    scheme = quantized_op_scheme(op_type)
    if scheme == "int4_weight":
        return "MatMulNBits"
    if scheme == "awq_int4":
        return "AWQ"
    if scheme == "int8_quantized":
        return "QLinearMatMul"
    return None


def _attribute_map(node, label: str) -> Dict[str, Any]:
    """Read ONNX attributes. Integers, floats, and strings are accepted."""
    raw = getattr(node, "attribute", None)
    if raw is None:
        return {}
    if isinstance(raw, dict):
        copied: Dict[str, Any] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(
                    f"quantization attribute for {label} must name a field, got {key!r}"
                )
            copied[key.strip()] = value
        return copied
    if isinstance(raw, (str, bytes)):
        raise ValueError(
            f"quantization attributes for {label} must be a list, got {raw!r}"
        )
    try:
        items = list(raw)
    except TypeError as exc:
        raise ValueError(
            f"quantization attributes for {label} must be a list, got {raw!r}"
        ) from exc
    found: Dict[str, Any] = {}
    for attr in items:
        name = getattr(attr, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"quantization attribute for {label} must name a field, got {name!r}"
            )
        name = name.strip()
        type_code = getattr(attr, "type", 0)
        if isinstance(type_code, bool) or not isinstance(type_code, int):
            raise ValueError(
                f"quantization attribute {name} type must be an integer, got {type_code!r}"
            )
        if type_code in (0, 2):
            value = getattr(attr, "i", None)
            if isinstance(value, bool) or (value is not None and not isinstance(value, int)):
                raise ValueError(
                    f"quantization attribute {name} must be an integer, got {value!r}"
                )
            if isinstance(value, int):
                found[name] = value
            elif type_code == 2:
                raise ValueError(
                    f"quantization attribute {name} must be an integer, got {value!r}"
                )
            continue
        if type_code == 1:
            value = getattr(attr, "f", None)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"quantization attribute {name} must be a float, got {value!r}"
                )
            found[name] = float(value)
            continue
        if type_code == 3:
            raw_s = getattr(attr, "s", b"")
            if isinstance(raw_s, bytes):
                found[name] = raw_s.decode("utf-8", "replace")
            elif isinstance(raw_s, str):
                found[name] = raw_s
            else:
                raise ValueError(
                    f"quantization attribute {name} must be a string, got {raw_s!r}"
                )
            continue
        raise ValueError(
            f"quantization attribute {name} has unsupported type {type_code}"
        )
    return found


def _optional_int_attr(attrs: Dict[str, Any], label: str, *names: str) -> Optional[int]:
    """Return one integer attribute. Two names that disagree raise."""
    folded = {key.lower(): key for key in attrs}
    chosen = None
    for name in names:
        key = name if name in attrs else folded.get(name.lower())
        if key is None:
            continue
        value = attrs[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"quantization attribute {key} for {label} must be an integer, got {value!r}"
            )
        if chosen is not None and value != chosen:
            raise ValueError(
                f"quantization {names[0]} for {label} disagrees: {chosen} and {value}"
            )
        chosen = value
    return chosen


def _quant_elem_name(elem: int, name: str) -> str:
    extra = _QUANT_EXTRA_ELEM.get(elem)
    if extra is not None:
        return extra
    return _dtype_name_for_elem(elem, name)


def _recorded_quant_dtype(tensor_name: str, info: Dict[str, Any], initializers: Dict[str, Any]) -> Optional[str]:
    """Value info wins. An initializer fills a missing element type."""
    if not isinstance(tensor_name, str) or not tensor_name.strip():
        return None
    value = info.get(tensor_name)
    if value is not None:
        elem = _elem_type_code(value)
        if elem is not None:
            return _quant_elem_name(elem, tensor_name)
    initializer = initializers.get(tensor_name)
    if initializer is None:
        return None
    elem = getattr(initializer, "data_type", 0)
    if isinstance(elem, bool) or not isinstance(elem, int):
        raise ValueError(
            f"quantization {tensor_name} initializer data_type must be an integer, got {elem!r}"
        )
    if elem == 0:
        return None
    return _quant_elem_name(elem, tensor_name)


def _recorded_quant_shape(tensor_name: str, info: Dict[str, Any], initializers: Dict[str, Any]):
    """Concrete shape from value info, then an initializer. Symbolic dims stay unset."""
    if not isinstance(tensor_name, str) or not tensor_name.strip():
        return None
    value = info.get(tensor_name)
    if value is not None:
        shape = _dims_from_value(value)
        if shape is not None and shape and all(isinstance(dim, int) for dim in shape):
            return shape
    initializer = initializers.get(tensor_name)
    if initializer is None:
        return None
    _dtype, shape = _initializer_dtype_and_shape(initializer, tensor_name)
    if not shape or any(dim is None for dim in shape):
        return None
    return shape


def _node_input_names(node) -> List[str]:
    inputs = getattr(node, "input", ()) or ()
    if isinstance(inputs, (str, bytes)):
        raise ValueError(
            f"quantized node {_quant_node_label(node)} inputs must be a list, got {inputs!r}"
        )
    try:
        items = list(inputs)
    except TypeError as exc:
        raise ValueError(
            f"quantized node {_quant_node_label(node)} inputs must be a list, got {inputs!r}"
        ) from exc
    names: List[str] = []
    for item in items:
        if item is None:
            names.append("")
        elif isinstance(item, str):
            names.append(item.strip())
        else:
            raise ValueError(
                f"quantized node {_quant_node_label(node)} input must be a string, got {item!r}"
            )
    return names


def _input_at(names: List[str], index: int) -> str:
    if index < 0 or index >= len(names):
        return ""
    return names[index]


def _require_bits(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in _QUANT_BITS:
        raise ValueError(
            f"quantization bits for {label} must be 2, 3, 4, or 8, got {value!r}"
        )
    return value


def _require_group_size(value: int, bits: Optional[int], label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"quantization group_size for {label} must be a positive integer, got {value!r}"
        )
    if bits is not None and bits < 8 and (value * bits) % 8 != 0:
        raise ValueError(
            f"quantization group_size {value} and bits {bits} for {label} do not fill whole bytes"
        )
    return value


def _infer_group_size(bits: Optional[int], weight_shape, scale_shape, k_attr: Optional[int], label: str) -> Optional[int]:
    """MatMulNBits stores weights as [N, blocks, block_size * bits / 8]."""
    inferred = None
    if (
        bits is not None
        and weight_shape is not None
        and len(weight_shape) == 3
        and isinstance(weight_shape[2], int)
        and weight_shape[2] > 0
        and (weight_shape[2] * 8) % bits == 0
    ):
        inferred = weight_shape[2] * 8 // bits
    from_k = None
    blocks = None
    if scale_shape is not None and len(scale_shape) >= 2 and isinstance(scale_shape[-1], int):
        blocks = scale_shape[-1]
    elif weight_shape is not None and len(weight_shape) >= 2 and isinstance(weight_shape[1], int):
        blocks = weight_shape[1]
    if k_attr is not None and blocks:
        if k_attr <= 0 or k_attr % blocks != 0:
            raise ValueError(
                f"quantization group_size for {label} cannot be derived from K {k_attr}"
            )
        from_k = k_attr // blocks
    if inferred is not None and from_k is not None and inferred != from_k:
        raise ValueError(
            f"quantization group_size for {label} disagrees: {inferred} and {from_k}"
        )
    return inferred if inferred is not None else from_k


def _weight_dtype_for_bits(bits: int, storage: Optional[str]) -> str:
    if bits < 8:
        return f"int{bits}"
    if storage == "uint8":
        return "uint8"
    if storage in {"int8", "uint4", "int4"}:
        return storage
    return "int8"


def _packing_for_storage(kind: str, bits: Optional[int], storage: Optional[str]) -> Optional[str]:
    if kind == "QLinearMatMul":
        return "qdq"
    if bits is not None and bits < 8:
        if storage == "uint8" or storage is None:
            return "packed_uint8" if storage == "uint8" else None
        if storage == "int32":
            return "packed_int32"
        return storage
    if storage in {"int8", "uint8"}:
        return "dense"
    return None


def _warn_quant_default(label: str, field: str, value: Any) -> None:
    warnings.warn(
        f"quantized node {label} uses a default {field} {value} "
        f"because the ONNX graph does not record it",
        UserWarning,
        stacklevel=3,
    )


def _node_quantization(node, info: Dict[str, Any], initializers: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One quantized node. A missing recorded field uses the op default and warns."""
    kind = _quant_kind(node)
    if kind is None:
        return None
    label = _quant_node_label(node)
    attrs = _attribute_map(node, label)
    inputs = _node_input_names(node)
    qlinear = kind == "QLinearMatMul"
    if qlinear:
        activation_name = _input_at(inputs, 0)
        weight_name = _input_at(inputs, 3)
        scale_name = _input_at(inputs, 4)
        zero_name = _input_at(inputs, 5) or _input_at(inputs, 2)
        has_zero_point = bool(zero_name)
    else:
        activation_name = _input_at(inputs, 0)
        weight_name = _input_at(inputs, 1)
        scale_name = _input_at(inputs, 2)
        zero_name = _input_at(inputs, 3)
        has_zero_point = bool(zero_name)
        if _input_at(inputs, 4):
            kind = "GPTQ"

    bits = _optional_int_attr(attrs, label, "bits")
    group_size = _optional_int_attr(attrs, label, "block_size", "group_size")
    k_attr = _optional_int_attr(attrs, label, "K", "k")
    bits_recorded = bits is not None
    if bits_recorded:
        bits = _require_bits(bits, label)
    else:
        bits = 8 if qlinear else 4
    storage = _recorded_quant_dtype(weight_name, info, initializers)
    activation = _recorded_quant_dtype(activation_name, info, initializers)
    scale_dtype = _recorded_quant_dtype(scale_name, info, initializers)
    zero_dtype = _recorded_quant_dtype(zero_name, info, initializers) if zero_name else None
    if group_size is None:
        group_size = _infer_group_size(
            bits if bits_recorded else None,
            _recorded_quant_shape(weight_name, info, initializers),
            _recorded_quant_shape(scale_name, info, initializers),
            k_attr,
            label,
        )
    if group_size is not None:
        group_size = _require_group_size(group_size, bits, label)

    if not bits_recorded:
        _warn_quant_default(label, "bits", bits)
    if group_size is None and not qlinear:
        group_size = _require_group_size(128, bits, label)
        _warn_quant_default(label, "group_size", group_size)
    if activation is None:
        activation = "uint8" if qlinear else "float16"
        _warn_quant_default(label, "activation_dtype", activation)
    if scale_dtype is None:
        scale_dtype = "float32" if qlinear else activation
        _warn_quant_default(label, "scale_dtype", scale_dtype)
    if has_zero_point and zero_dtype is None:
        zero_dtype = "int8" if qlinear else "uint8"
        _warn_quant_default(label, "zero_point_dtype", zero_dtype)
    packing = _packing_for_storage(kind, bits, storage)
    if packing is None:
        packing = "packed_uint8"
        _warn_quant_default(label, "packing_format", packing)
    weight_dtype = _weight_dtype_for_bits(bits, storage)
    return {
        "method": kind,
        "weight_dtype": weight_dtype,
        "activation_dtype": activation,
        "group_size": group_size,
        "has_zero_point": has_zero_point,
        "packing_format": packing,
        "bits": bits,
        "scale_dtype": scale_dtype,
        "zero_point_dtype": zero_dtype,
    }


def _merge_quantization(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One contract for every split component. A disagreement raises."""
    if not records:
        return _empty_quantization_info()
    merged = dict(records[0])
    for other in records[1:]:
        for key in _QUANT_INFO_KEYS:
            current = merged.get(key)
            value = other.get(key)
            if current is None:
                merged[key] = value
            elif value is not None and value != current:
                raise ValueError(
                    f"quantization {key} disagrees across components: {current!r} and {value!r}"
                )
    for key in _QUANT_INFO_KEYS:
        merged.setdefault(key, None)
    return {key: merged.get(key) for key in _QUANT_INFO_KEYS}


def _initializer_map(graph) -> Dict[str, Any]:
    found: Dict[str, Any] = {}
    for initializer in getattr(graph, "initializer", ()) or ():
        name = getattr(initializer, "name", None)
        if isinstance(name, str) and name and name not in found:
            found[name] = initializer
    return found


def extract_quantization_info(model) -> Dict[str, Any]:
    """
    Read the quantization contract from an ONNX graph.

    MatMulNBits contributes bits, group size (``block_size``), scale dtype,
    zero-point dtype, activation dtype, weight dtype, and packing layout.
    A connected ``g_idx`` or a GPTQ name selects GPTQ. An AWQ op or name
    selects AWQ. QLinearMatMul is recorded as int8 QDQ. A field the graph
    does not record uses the op default and warns. Quantized nodes that
    disagree raise, so split components keep one contract. A dense graph
    returns empty fields.

    Args:
        model: ONNX model or graph.

    Returns:
        JSON record with method, dtypes, group size, zero point, and packing.

    Raises:
        ValueError: If the graph, an attribute, or two quantized nodes disagree.
    """
    graph, nodes = _boundary_graph(model)
    info = _value_info_map(graph)
    initializers = _initializer_map(graph)
    records: List[Dict[str, Any]] = []
    for node in nodes:
        record = _node_quantization(node, info, initializers)
        if record is not None:
            records.append(record)
    return _merge_quantization(records)


_KV_ROLES = ("past_key", "past_value", "present_key", "present_value")
_KV_INFO_KEYS = (
    "num_layers",
    "num_kv_heads",
    "head_dim",
    "layout",
    "dtype",
    "past_key_pattern",
    "past_value_pattern",
    "present_key_pattern",
    "present_value_pattern",
    "max_seq_len",
    "stateful",
    "stateful_contract",
)
_KV_AXIS_LETTERS = {
    "b": "B",
    "batch": "B",
    "batch_size": "B",
    "n": "N",
    "heads": "N",
    "head": "N",
    "kv_heads": "N",
    "kv_head": "N",
    "num_heads": "N",
    "num_kv_heads": "N",
    "num_key_value_heads": "N",
    "num_attention_heads": "N",
    "n_heads": "N",
    "n_head": "N",
    "n_kv_heads": "N",
    "s": "S",
    "seq": "S",
    "sequence": "S",
    "seq_len": "S",
    "seq_length": "S",
    "sequence_len": "S",
    "sequence_length": "S",
    "past_sequence": "S",
    "past_sequence_length": "S",
    "past_seq_len": "S",
    "past_seq_length": "S",
    "total_sequence_length": "S",
    "total_seq_len": "S",
    "max_seq_len": "S",
    "max_sequence_length": "S",
    "d": "D",
    "dim": "D",
    "head_dim": "D",
    "head_size": "D",
}
_KV_KIND_TOKEN = re.compile(r"(^|[^a-z])(key|value)([^a-z]|$)")


def _empty_kv_cache_info() -> Dict[str, Any]:
    """A graph with no past or present KV tensors still records the key."""
    return _copy_kv_cache_info({
        "num_layers": 0,
        "num_kv_heads": None,
        "head_dim": None,
        "layout": None,
        "dtype": None,
        "past_key_pattern": None,
        "past_value_pattern": None,
        "present_key_pattern": None,
        "present_value_pattern": None,
        "max_seq_len": None,
        "stateful": False,
        "stateful_contract": None,
    })


def _copy_kv_cache_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """Return a fresh record. The stateful contract list is not shared."""
    copied = {key: info.get(key) for key in _KV_INFO_KEYS}
    contract = copied.get("stateful_contract")
    if isinstance(contract, dict):
        roles = contract.get("roles") or []
        if isinstance(roles, (str, bytes)) or not isinstance(roles, list):
            raise ValueError(
                f"kv cache stateful_contract roles must be a list, got {roles!r}"
            )
        zero_copy = contract.get("zero_copy")
        if not isinstance(zero_copy, bool):
            raise ValueError(
                f"kv cache stateful_contract zero_copy must be a bool, got {zero_copy!r}"
            )
        copied["stateful_contract"] = {
            "roles": list(roles),
            "zero_copy": zero_copy,
        }
    return copied


def _warn_kv_default(label: str, field: str, value: Any) -> None:
    warnings.warn(
        f"kv cache tensor {label} uses a default {field} {value} "
        f"because the ONNX graph does not record it",
        UserWarning,
        stacklevel=3,
    )


def _warn_kv_missing(label: str, field: str) -> None:
    warnings.warn(
        f"kv cache tensor {label} does not record {field}",
        UserWarning,
        stacklevel=3,
    )


def _kv_role(name: str) -> Optional[str]:
    """Past or present key or value. The family name past_key_values is not a kind."""
    lowered = name.lower()
    if "past" not in lowered and "present" not in lowered:
        return None
    if "past" in lowered and "present" in lowered:
        raise ValueError(f"kv cache tensor {name} is both past and present")
    stripped = lowered.replace("past_key_values", " ").replace("present_key_values", " ")
    found = {match.group(2) for match in _KV_KIND_TOKEN.finditer(stripped)}
    if "key" in found and "value" in found:
        raise ValueError(f"kv cache tensor {name} is both key and value")
    if "key" not in found and "value" not in found:
        return None
    side = "present" if "present" in lowered else "past"
    kind = "key" if "key" in found else "value"
    return f"{side}_{kind}"


def _append_kv_name(found: List[str], seen: Set[str], name: Any) -> None:
    if not isinstance(name, str):
        return
    text = name.strip()
    if not text or text in seen:
        return
    seen.add(text)
    found.append(text)


def _kv_tensor_names(graph, nodes) -> List[str]:
    """Graph inputs, outputs, value info, then node edges."""
    found: List[str] = []
    seen: Set[str] = set()
    for collection in (
        getattr(graph, "input", ()) or (),
        getattr(graph, "output", ()) or (),
        getattr(graph, "value_info", ()) or (),
    ):
        for item in collection:
            _append_kv_name(found, seen, getattr(item, "name", None))
    for node in nodes:
        for attr in ("input", "output"):
            raw = getattr(node, attr, ()) or ()
            if isinstance(raw, (str, bytes)):
                raise ValueError(f"kv cache node {attr} must be a list, got {raw!r}")
            try:
                edges = list(raw)
            except TypeError as exc:
                raise ValueError(f"kv cache node {attr} must be a list, got {raw!r}") from exc
            for name in edges:
                _append_kv_name(found, seen, name)
    return found


def _pattern_for_role(role: str, names: List[str]) -> Tuple[str, Dict[str, int]]:
    """One ``{layer}`` format string and the layer index of each tensor name."""
    spans = [list(re.finditer(r"\d+", name)) for name in names]
    if len(names) == 1 and not spans[0]:
        _warn_kv_default(names[0], "layer index", 0)
        return names[0], {names[0]: 0}
    if len(names) == 1:
        span = spans[0][0]
        pattern = names[0][:span.start()] + "{layer}" + names[0][span.end():]
        return pattern, {names[0]: int(span.group())}

    candidates = []
    width = min((len(found) for found in spans), default=0)
    for index in range(width):
        pattern = None
        layers: Dict[str, int] = {}
        matched = True
        for name, found in zip(names, spans):
            span = found[index]
            candidate = name[:span.start()] + "{layer}" + name[span.end():]
            if pattern is None:
                pattern = candidate
            elif candidate != pattern:
                matched = False
                break
            layers[name] = int(span.group())
        if matched and pattern is not None:
            candidates.append((pattern, layers))
    if len(candidates) == 1:
        chosen = candidates[0]
    else:
        varying = [item for item in candidates if len(set(item[1].values())) > 1]
        if len(varying) == 1:
            chosen = varying[0]
        else:
            raise ValueError(f"kv cache {role} layer index is ambiguous: {names}")
    pattern, layers = chosen
    seen: Dict[int, str] = {}
    for name, layer in layers.items():
        previous = seen.get(layer)
        if previous is not None:
            raise ValueError(
                f"kv cache {role} layer {layer} is named {previous!r} and {name!r}"
            )
        seen[layer] = name
    return pattern, layers


def _axis_letter(param: Any, name: str) -> Optional[str]:
    if param is None:
        return None
    if not isinstance(param, str):
        raise ValueError(
            f"kv cache tensor {name} axis name must be a string, got {param!r}"
        )
    token = param.strip().lower().replace("-", "_").replace(" ", "_")
    if not token:
        return None
    return _KV_AXIS_LETTERS.get(token)


def _axis_number(number: Any, name: str) -> Optional[int]:
    if number is None or number == "":
        return None
    if isinstance(number, bool) or not isinstance(number, int):
        raise ValueError(
            f"kv cache tensor {name} dimension must be an integer, got {number!r}"
        )
    if number < 0:
        raise ValueError(f"kv cache tensor {name} dimension {number} is negative")
    if number == 0:
        return None
    return number


def _axes_from_dims(dims, name: str) -> List[Tuple[Optional[str], Optional[int]]]:
    if isinstance(dims, (str, bytes)):
        raise ValueError(f"kv cache tensor {name} shape must be a list, got {dims!r}")
    try:
        items = list(dims)
    except TypeError as exc:
        raise ValueError(f"kv cache tensor {name} shape must be a list, got {dims!r}") from exc
    parsed: List[Tuple[Optional[str], Optional[int]]] = []
    for dim in items:
        if isinstance(dim, bool) or isinstance(dim, int):
            parsed.append((None, _axis_number(dim, name)))
            continue
        param = getattr(dim, "dim_param", "")
        number = getattr(dim, "dim_value", 0)
        if param in ("", None):
            param = None
        parsed.append((_axis_letter(param, name), _axis_number(number, name)))
    return parsed


def _axes_from_value(name: str, value) -> Optional[List[Tuple[Optional[str], Optional[int]]]]:
    if value is None:
        return None
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
    return _axes_from_dims(dims, name)


def _axes_from_initializer(name: str, initializer) -> Optional[List[Tuple[Optional[str], Optional[int]]]]:
    if initializer is None:
        return None
    dims = getattr(initializer, "dims", None)
    if dims is None:
        return None
    return _axes_from_dims(dims, name)


def _kv_dtype_name(name: str, value, initializer) -> Optional[str]:
    """Value info wins. An initializer is the fallback. A missing type stays unset."""
    if value is not None:
        elem = _elem_type_code(value)
        if elem is not None:
            return _dtype_name_for_elem(elem, name)
    if initializer is None:
        return None
    elem = getattr(initializer, "data_type", 0)
    if isinstance(elem, bool) or not isinstance(elem, int):
        raise ValueError(
            f"kv cache tensor {name} initializer data_type must be an integer, got {elem!r}"
        )
    if elem == 0:
        return None
    return _dtype_name_for_elem(elem, name)


def _layout_from_axes(
    axes: List[Tuple[Optional[str], Optional[int]]],
    name: str,
) -> Optional[str]:
    """Named axes spell a layout. A subset that matches BNSD confirms BNSD."""
    letters = [letter for letter, _number in axes]
    if all(letter is None for letter in letters):
        return None
    if len(letters) != 4:
        raise ValueError(
            f"kv cache tensor {name} rank {len(letters)} is not the BNSD rank 4 contract"
        )
    if all(letter is not None for letter in letters):
        layout = "".join(letters)
        if sorted(layout) != sorted("BNSD"):
            raise ValueError(
                f"kv cache tensor {name} layout {layout} is not a 4D cache layout"
            )
        return layout
    for index, letter in enumerate(letters):
        if letter is not None and letter != "BNSD"[index]:
            raise ValueError(
                f"kv cache tensor {name} names axis {index} as {letter}, "
                f"which does not match layout BNSD"
            )
    return "BNSD"


def _axis_at(
    axes: List[Tuple[Optional[str], Optional[int]]],
    layout: str,
    letter: str,
) -> Optional[int]:
    index = layout.find(letter)
    if index < 0 or index >= len(axes):
        return None
    return axes[index][1]


def _fill_kv_field(merged: Dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    current = merged.get(key)
    if current is None:
        merged[key] = value
    elif value != current:
        raise ValueError(
            f"kv cache {key} disagrees across components: {current!r} and {value!r}"
        )


def _kv_layer_count(role_layers: Dict[str, Dict[str, int]]) -> int:
    sets = []
    for layers in role_layers.values():
        found = set(layers.values())
        if found:
            sets.append(found)
    if not sets:
        return 0
    first = sets[0]
    for other in sets[1:]:
        if other != first:
            raise ValueError(
                f"kv cache layers disagree across components: "
                f"{sorted(first)} and {sorted(other)}"
            )
    ordered = sorted(first)
    if ordered != list(range(ordered[-1] + 1)):
        raise ValueError(f"kv cache layers are not contiguous from 0: {ordered}")
    return len(ordered)


def extract_kv_cache_info(model) -> Dict[str, Any]:
    """
    Read the KV cache contract from an ONNX graph.

    Past and present key and value names become the format strings
    ``kv_cache.py`` uses, with the layer index written as ``{layer}``.
    Layout BNSD is batch, heads, sequence, head dim, the shape
    ``KVCacheManager`` allocates. ``num_kv_heads``, ``head_dim``, and
    ``max_seq_len`` come from those axes. A rank-4 tensor with no axis
    names uses BNSD and warns. A missing element type uses float16 and
    warns. A symbolic sequence axis leaves ``max_seq_len`` unset and warns.
    The name heuristic is not used, so a mask token in the name stays
    float16. Layers must be contiguous from 0 and the same on every role.
    Disagreeing tensors raise before a split writes component files. A
    graph with no KV tensors returns empty fields.

    ``stateful_contract`` lists the roles that are present and sets
    ``zero_copy``, so each past and present tensor stays one OrtValue.

    Args:
        model: ONNX model or graph.

    Returns:
        JSON record with layer count, heads, head dim, layout, dtype,
        name patterns, max sequence length, and the stateful contract.

    Raises:
        ValueError: If the graph, a dimension, or two KV tensors disagree.
    """
    graph, nodes = _boundary_graph(model)
    info = _value_info_map(graph)
    initializers = _initializer_map(graph)
    grouped: Dict[str, List[str]] = {role: [] for role in _KV_ROLES}
    for name in _kv_tensor_names(graph, nodes):
        role = _kv_role(name)
        if role is not None:
            grouped[role].append(name)
    if not any(grouped.values()):
        return _empty_kv_cache_info()

    patterns: Dict[str, str] = {}
    role_layers: Dict[str, Dict[str, int]] = {}
    for role in _KV_ROLES:
        names = grouped[role]
        if not names:
            continue
        pattern, layers = _pattern_for_role(role, names)
        patterns[role] = pattern
        role_layers[role] = layers

    tensors = []
    for role in _KV_ROLES:
        for name in grouped[role]:
            value = info.get(name)
            initializer = initializers.get(name)
            tensors.append({
                "name": name,
                "dtype": _kv_dtype_name(name, value, initializer),
                "axes": _axes_from_value(name, value) if value is not None else _axes_from_initializer(name, initializer),
            })
            if value is not None and tensors[-1]["axes"] is None:
                tensors[-1]["axes"] = _axes_from_initializer(name, initializer)

    layouts = []
    ranks = []
    for item in tensors:
        if item["axes"] is None:
            continue
        ranks.append(len(item["axes"]))
        layout = _layout_from_axes(item["axes"], item["name"])
        if layout is not None:
            layouts.append((item["name"], layout))
    chosen = None
    layout_defaulted = False
    if layouts:
        chosen = layouts[0][1]
        for name, layout in layouts[1:]:
            if layout != chosen:
                raise ValueError(
                    f"kv cache layout disagrees across components: {chosen!r} and {layout!r}"
                )
    elif any(rank == 4 for rank in ranks):
        chosen = "BNSD"
        layout_defaulted = True

    merged: Dict[str, Any] = {
        "num_kv_heads": None,
        "head_dim": None,
        "dtype": None,
        "max_seq_len": None,
    }
    for item in tensors:
        _fill_kv_field(merged, "dtype", item["dtype"])
        axes = item["axes"]
        if axes is None or chosen is None:
            continue
        if len(axes) != len(chosen):
            raise ValueError(
                f"kv cache tensor {item['name']} rank {len(axes)} "
                f"does not match layout {chosen}"
            )
        _fill_kv_field(merged, "num_kv_heads", _axis_at(axes, chosen, "N"))
        _fill_kv_field(merged, "head_dim", _axis_at(axes, chosen, "D"))
        _fill_kv_field(merged, "max_seq_len", _axis_at(axes, chosen, "S"))

    num_layers = _kv_layer_count(role_layers)
    label = tensors[0]["name"]
    if layout_defaulted:
        _warn_kv_default(label, "layout", chosen)
    elif chosen is None:
        if ranks:
            warnings.warn(
                f"kv cache tensor {label} rank {ranks[0]} "
                f"is not the BNSD rank 4 contract",
                UserWarning,
                stacklevel=3,
            )
        else:
            _warn_kv_missing(label, "layout")
    if chosen is not None:
        if merged["num_kv_heads"] is None:
            _warn_kv_missing(label, "num_kv_heads")
        if merged["head_dim"] is None:
            _warn_kv_missing(label, "head_dim")
        if merged["max_seq_len"] is None:
            _warn_kv_missing(label, "max_seq_len")
    if merged["dtype"] is None:
        merged["dtype"] = "float16"
        _warn_kv_default(label, "dtype", "float16")

    roles = [role for role in _KV_ROLES if role in patterns]
    record = {
        "num_layers": num_layers,
        "num_kv_heads": merged["num_kv_heads"],
        "head_dim": merged["head_dim"],
        "layout": chosen,
        "dtype": merged["dtype"],
        "past_key_pattern": patterns.get("past_key"),
        "past_value_pattern": patterns.get("past_value"),
        "present_key_pattern": patterns.get("present_key"),
        "present_value_pattern": patterns.get("present_value"),
        "max_seq_len": merged["max_seq_len"],
        "stateful": True,
        "stateful_contract": {
            "roles": roles,
            "zero_copy": True,
        },
    }
    return _copy_kv_cache_info(record)


_EXEC_KEYS = (
    "npu_safe",
    "recommended_mode",
    "prefill_device",
    "decode_device",
    "resident_components",
    "streamed_components",
    "graph_form",
    "layer_schedule",
    "moe_routing",
)
_EXEC_MODES = {"hybrid_phase_split", "monolithic", "sequential", "persistent"}
_EXEC_DEVICES = {"npu", "igpu", "gpu", "cpu"}
_EXEC_RESIDENT = (
    "embedding",
    "norm",
    "lm_head",
    "moe_router",
    "mtp_head",
    "ngram_embedding",
)
_EXEC_STREAMED = (
    "layer",
    "moe_expert",
    "gated_deltanet",
    "sparse_attention",
    "other",
)
_NPU_QUANT_METHODS = {"MatMulNBits", "AWQ", "GPTQ", "QLinearMatMul"}
_NPU_LEGAL_ELEMS = {2, 3, 10, 16, 21, 22}
_NPU_UNSAFE_ELEMS = {1, 11}


def _copy_execution_policy(info: Dict[str, Any]) -> Dict[str, Any]:
    """Return a fresh record. Resident and streamed lists are not shared."""
    copied = {key: info.get(key) for key in _EXEC_KEYS}
    if not isinstance(copied.get("npu_safe"), bool):
        raise ValueError(
            f"execution policy npu_safe must be a bool, got {copied.get('npu_safe')!r}"
        )
    mode = copied.get("recommended_mode")
    if mode not in _EXEC_MODES:
        raise ValueError(
            f"execution policy recommended_mode {mode!r} is not a known mode"
        )
    for key in ("prefill_device", "decode_device"):
        device = copied.get(key)
        if device not in _EXEC_DEVICES:
            raise ValueError(f"execution policy {key} {device!r} is not a known device")
    for key in ("resident_components", "streamed_components"):
        values = copied.get(key)
        if isinstance(values, (str, bytes)) or not isinstance(values, list):
            raise ValueError(f"execution policy {key} must be a list, got {values!r}")
        copied[key] = list(values)
    form = copied.get("graph_form")
    if form not in {"monolithic", "split"}:
        raise ValueError(f"execution policy graph_form {form!r} is not monolithic or split")
    schedule = copied.get("layer_schedule")
    if schedule not in {"sequential", "persistent"}:
        raise ValueError(
            f"execution policy layer_schedule {schedule!r} is not sequential or persistent"
        )
    routing = copied.get("moe_routing")
    if routing is not None and routing != "moe_routing":
        raise ValueError(f"execution policy moe_routing {routing!r} is not moe_routing")
    return copied


def _warn_execution_default() -> None:
    warnings.warn(
        "execution policy uses a default npu_safe False "
        "because the ONNX graph does not record an NPU-legal dtype",
        UserWarning,
        stacklevel=3,
    )


def _warn_execution_mixed(npu_safe: bool) -> None:
    warnings.warn(
        "execution policy saw float32 and an NPU-legal dtype; "
        f"npu_safe follows the majority and is {npu_safe}",
        UserWarning,
        stacklevel=3,
    )


def _execution_component_types(components) -> List[str]:
    """Unique component types in resident-then-streamed order."""
    if not isinstance(components, dict):
        raise ValueError(f"components must be a dict, got {components!r}")
    if not components:
        raise ValueError("components must name at least one component")
    found = set()
    for raw_name, info in components.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f"component name must be a non-empty string, got {raw_name!r}")
        raw_type = getattr(info, "component_type", None)
        if raw_type is None and isinstance(info, dict):
            raw_type = info.get("component_type")
        if isinstance(raw_type, bool) or not isinstance(raw_type, str) or not raw_type.strip():
            raise ValueError(
                f"component {raw_name.strip()} type must be a non-empty string, got {raw_type!r}"
            )
        text = raw_type.strip().lower().replace("-", "_").replace(" ", "_")
        try:
            kind = ComponentType(text).value
        except ValueError as exc:
            known = ", ".join(item.value for item in ComponentType)
            raise ValueError(
                f"Unknown component type {raw_type!r}. Expected one of: {known}"
            ) from exc
        found.add(kind)
    ordered = [kind for kind in _EXEC_RESIDENT + _EXEC_STREAMED if kind in found]
    return ordered


def _npu_safety(graph, quantization_info) -> Tuple[bool, str]:
    """Return npu_safe and whether the dtype was recorded, mixed, or defaulted."""
    if quantization_info is None:
        quantization_info = {}
    if not isinstance(quantization_info, dict):
        raise ValueError(
            f"quantization_info must be a dict, got {quantization_info!r}"
        )
    method = quantization_info.get("method")
    if isinstance(method, bool):
        raise ValueError(f"quantization method must be a string, got {method!r}")
    if isinstance(method, str) and method.strip() in _NPU_QUANT_METHODS:
        return True, "recorded"
    legal = 0
    unsafe = 0
    for initializer in getattr(graph, "initializer", ()) or ():
        elem = getattr(initializer, "data_type", 0)
        if isinstance(elem, bool) or not isinstance(elem, int):
            name = getattr(initializer, "name", None) or "weight"
            raise ValueError(
                f"execution policy weight {name} data_type must be an integer, got {elem!r}"
            )
        if elem in _NPU_LEGAL_ELEMS:
            legal += 1
        elif elem in _NPU_UNSAFE_ELEMS:
            unsafe += 1
    if legal == 0 and unsafe == 0:
        return False, "default"
    if legal and unsafe:
        return legal >= unsafe, "mixed"
    if legal:
        return True, "recorded"
    return False, "recorded"


def extract_execution_policy(model, components, quantization_info=None) -> Dict[str, Any]:
    """
    Read the AMD execution contract for one split.

    ``recommended_mode`` ``hybrid_phase_split`` prefetches on the NPU and
    decodes on the iGPU. ``embedding``, ``norm``, and ``lm_head`` stay
    resident. ``layer_*`` streams. A MatMulNBits, AWQ, GPTQ, or
    QLinearMatMul graph, or a float16 weight graph, is ``npu_safe``. A
    graph that records float32 stays on the iGPU and may run sequential
    layers. A graph that records no dtype uses ``npu_safe`` false and
    warns. One component is monolithic. A router stays resident, experts
    stream, and ``moe_routing`` records that policy. Sequential execution
    on the NPU is not recommended.

    Args:
        model: ONNX model or graph.
        components: Component name to ComponentInfo or a dict with ``component_type``.
        quantization_info: Optional record from ``extract_quantization_info``.

    Returns:
        JSON record with the device policy, residency, graph form, layer
        schedule, and MoE routing.

    Raises:
        ValueError: If the graph, a component type, or a weight dtype is invalid.
    """
    graph, _nodes = _boundary_graph(model)
    kinds = _execution_component_types(components)
    present = set(kinds)
    npu_safe, source = _npu_safety(graph, quantization_info)
    if source == "default":
        _warn_execution_default()
    elif source == "mixed":
        _warn_execution_mixed(npu_safe)
    split = len(components) > 1
    if not split:
        mode = "monolithic"
        device = "npu" if npu_safe else "igpu"
        prefill = device
        decode = device
        schedule = "persistent"
        resident = list(kinds)
        streamed: List[str] = []
    elif source == "recorded" and not npu_safe:
        mode = "sequential"
        prefill = "igpu"
        decode = "igpu"
        schedule = "sequential"
        resident = [kind for kind in _EXEC_RESIDENT if kind in present]
        streamed = [
            "other" if kind == "other" else f"{kind}_*"
            for kind in _EXEC_STREAMED
            if kind in present
        ]
    else:
        mode = "hybrid_phase_split"
        prefill = "npu"
        decode = "igpu"
        schedule = "persistent"
        resident = [kind for kind in _EXEC_RESIDENT if kind in present]
        streamed = [
            "other" if kind == "other" else f"{kind}_*"
            for kind in _EXEC_STREAMED
            if kind in present
        ]
    routing = "moe_routing" if "moe_router" in present or "moe_expert" in present else None
    return _copy_execution_policy({
        "npu_safe": npu_safe,
        "recommended_mode": mode,
        "prefill_device": prefill,
        "decode_device": decode,
        "resident_components": resident,
        "streamed_components": streamed,
        "graph_form": "split" if split else "monolithic",
        "layer_schedule": schedule,
        "moe_routing": routing,
    })


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
    - model_info.json, including quantization_info, kv_cache_info, and execution_policy
    
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

    # One quantization contract and one KV cache contract for every component.
    # A disagreement raises before any component file is written.
    quantization_info = extract_quantization_info(onnx_model)
    kv_cache_info = extract_kv_cache_info(onnx_model)

    # Extract components and the tensors that cross them
    components = extract_components(onnx_model)
    boundaries = collect_boundary_tensors(onnx_model, components)
    # One execution contract for the split. A bad component raises before
    # any component file is written.
    execution_policy = extract_execution_policy(
        onnx_model,
        components,
        quantization_info=quantization_info,
    )
    
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
        "quantization_info": quantization_info,
        "kv_cache_info": kv_cache_info,
        "execution_policy": execution_policy,
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
