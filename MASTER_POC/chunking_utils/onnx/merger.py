"""
ONNX model merging utilities.

This module provides utilities for merging split ONNX component files
back into a single unified ONNX model.
"""

import gc
import json
import math
import os
from typing import Any, Dict, List, Optional

import onnx
from onnx import helper

from chunking_utils.common.types import (
    ExecutionPlan,
    _canonical_dtype_name,
    _coerce_execution_plan,
)

from .graph_analysis import _boundary_graph, _value_info_map
from .quantization import (
    _attribute_map,
    _node_input_names,
    _optional_int_attr,
    _packing_for_storage,
    _quant_kind,
    _quant_node_label,
    _recorded_quant_dtype,
)
from .types import OnnxMergeResult
from .validator import _as_signal, _real_number, _warn_psnr_failed, compute_psnr


def _require_merge_dir(output_dir: str) -> None:
    """Reject a blank directory and a file used as a directory."""
    if isinstance(output_dir, bool) or not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError(
            f"output directory must be a non-empty string, got {output_dir!r}"
        )
    if os.path.isfile(output_dir):
        raise ValueError(f"output directory must be a directory, got a file {output_dir}")


def _require_save_path(save_path: Optional[str]) -> Optional[str]:
    """None and an empty string mean do not save. Anything else is a file path."""
    if save_path is None or save_path == "":
        return None
    if isinstance(save_path, bool) or not isinstance(save_path, str) or not save_path.strip():
        raise ValueError(f"save_path must be a non-empty string, got {save_path!r}")
    if os.path.isdir(save_path):
        raise ValueError(f"save_path must be a file path, got a directory {save_path}")
    return save_path


def _load_model_info(output_dir: str) -> dict:
    """Read model_info.json. Broken JSON and a non-object document raise."""
    info_path = os.path.join(output_dir, "model_info.json")
    if os.path.isdir(info_path):
        raise ValueError(f"model_info.json must be a file, got a directory {info_path}")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"model_info.json not found in {output_dir}")
    with open(info_path, "r", encoding="utf-8") as handle:
        try:
            model_info = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError("model_info.json is not valid JSON") from exc
    if not isinstance(model_info, dict):
        raise ValueError(f"model_info must be a dict, got {model_info!r}")
    return model_info


def _component_map(model_info: dict) -> dict:
    """The component table is a dict of component name to file metadata."""
    components = model_info.get("components")
    if not isinstance(components, dict):
        raise ValueError(f"model_info components must be a dict, got {components!r}")
    return components


def _name_list(value, label: str) -> List[str]:
    """Copy a list of tensor or component names. A string or a blank entry raises."""
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise ValueError(f"{label} must be a list, got {value!r}")
    names: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{label} entries must be non-empty strings, got {item!r}")
        text = item.strip()
        if text not in names:
            names.append(text)
    return names


def _output_owners(components: dict, skip_name: str) -> Dict[str, str]:
    """Map each recorded output tensor to the component that writes it."""
    owners: Dict[str, str] = {}
    for name, comp in components.items():
        if name == skip_name:
            continue
        if not isinstance(comp, dict):
            raise ValueError(f"component {name} must be a dict, got {comp!r}")
        if "output_names" not in comp:
            continue
        for tensor in _name_list(comp.get("output_names"), f"{name} output_names"):
            previous = owners.get(tensor)
            if previous is not None and previous != name:
                raise ValueError(
                    f"tensor {tensor!r} is produced by {previous!r} and {name!r}"
                )
            owners[tensor] = name
    return owners


def _boundary_predecessors(model_info: dict, component_name: str):
    """Components that write a boundary tensor this component reads. None if unrecorded."""
    boundaries = model_info.get("boundary_tensors")
    if boundaries is None:
        return None
    if isinstance(boundaries, (str, bytes)) or not isinstance(boundaries, list):
        raise ValueError(f"boundary_tensors must be a list, got {boundaries!r}")
    found = False
    predecessors: List[str] = []
    for item in boundaries:
        if not isinstance(item, dict):
            raise ValueError(f"boundary tensor must be a dict, got {item!r}")
        consumers = item.get("consumer_components", item.get("consumers"))
        if consumers is None:
            continue
        consumer_names = _name_list(consumers, "consumer_components")
        if component_name not in consumer_names:
            continue
        found = True
        producer = item.get("producer_component", item.get("producer"))
        if not isinstance(producer, str) or not producer.strip():
            raise ValueError(
                f"producer_component must be a non-empty string, got {producer!r}"
            )
        producer = producer.strip()
        if producer in ("external", component_name) or producer in predecessors:
            continue
        predecessors.append(producer)
    if not found:
        return None
    return predecessors


def _required_by(components: dict, component_name: str, info: dict, model_info: dict) -> List[str]:
    """Components that read a tensor this component writes."""
    readers: List[str] = []
    outputs: List[str] = []
    if "output_names" in info:
        outputs = _name_list(info.get("output_names"), "output_names")
    if outputs:
        produced = set(outputs)
        for name, comp in components.items():
            if name == component_name:
                continue
            if not isinstance(comp, dict) or "input_names" not in comp:
                continue
            for tensor in _name_list(comp.get("input_names"), f"{name} input_names"):
                if tensor in produced and name not in readers:
                    readers.append(name)
    boundaries = model_info.get("boundary_tensors")
    if boundaries is None:
        return readers
    if isinstance(boundaries, (str, bytes)) or not isinstance(boundaries, list):
        raise ValueError(f"boundary_tensors must be a list, got {boundaries!r}")
    for item in boundaries:
        if not isinstance(item, dict):
            raise ValueError(f"boundary tensor must be a dict, got {item!r}")
        producer = item.get("producer_component", item.get("producer"))
        if not isinstance(producer, str) or producer.strip() != component_name:
            continue
        consumers = item.get("consumer_components", item.get("consumers", []))
        for name in _name_list(consumers, "consumer_components"):
            if name not in ("external", component_name) and name not in readers:
                readers.append(name)
    return readers


def analyze_component_dependencies(component_name: str, model_info: dict):
    """
    Find what must run before one component, and what reads it.

    None means model_info does not record a dependency, so the linear chain
    keeps that component at the end. A dict carries depends_on and required_by.
    Graph inputs that no component writes are external and do not create a
    predecessor. An n-gram lookup, a mask preprocess, or a position encode
    that only reads those inputs therefore has an empty depends_on.
    """
    if not isinstance(component_name, str) or not component_name.strip():
        raise ValueError(
            f"component name must be a non-empty string, got {component_name!r}"
        )
    component_name = component_name.strip()
    if not isinstance(model_info, dict):
        raise ValueError(f"model_info must be a dict, got {model_info!r}")
    components = model_info.get("components")
    if components is None:
        return None
    if not isinstance(components, dict):
        raise ValueError(f"model_info components must be a dict, got {components!r}")
    info = components.get(component_name)
    if info is None:
        return None
    if not isinstance(info, dict):
        raise ValueError(f"component {component_name} must be a dict, got {info!r}")

    known = False
    depends_on: List[str] = []
    if "depends_on" in info:
        known = True
        depends_on = _name_list(info.get("depends_on"), "depends_on")
        if component_name in depends_on:
            raise ValueError(f"component {component_name!r} depends on itself")

    if "input_names" in info and "depends_on" not in info:
        known = True
        owners = _output_owners(components, component_name)
        for tensor in _name_list(info.get("input_names"), "input_names"):
            owner = owners.get(tensor)
            if owner and owner not in depends_on:
                depends_on.append(owner)
    elif "depends_on" not in info:
        predecessors = _boundary_predecessors(model_info, component_name)
        if predecessors is not None:
            known = True
            depends_on = predecessors

    required_by = _required_by(components, component_name, info, model_info)
    if required_by:
        known = True
    if not known:
        return None
    return {"depends_on": depends_on, "required_by": required_by}


def find_correct_insert_position(other_dependencies, component_order) -> int:
    """
    Index where the dependent component belongs in the linear chain.

    None keeps the historical end position. An empty depends_on inserts at
    the front. Named predecessors insert immediately after the latest one.
    A consumer that would then be merged first is a cycle.
    """
    if isinstance(component_order, (str, bytes)) or not isinstance(component_order, list):
        raise ValueError(f"component order must be a list, got {component_order!r}")
    if other_dependencies is None:
        return len(component_order)
    if isinstance(other_dependencies, list):
        depends_on = _name_list(other_dependencies, "dependencies")
        required_by: List[str] = []
    elif isinstance(other_dependencies, dict):
        if "depends_on" not in other_dependencies:
            raise ValueError(
                f"dependencies must name depends_on, got {other_dependencies!r}"
            )
        depends_on = _name_list(other_dependencies.get("depends_on"), "depends_on")
        required_by = _name_list(
            other_dependencies.get("required_by", []), "required_by"
        )
    else:
        raise ValueError(
            f"dependencies must be a list or a dict, got {other_dependencies!r}"
        )

    index: Dict[str, int] = {}
    for position, name in enumerate(component_order):
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"component order entries must be non-empty strings, got {name!r}")
        if name in index:
            raise ValueError(f"component order repeats {name!r}")
        index[name] = position

    latest = -1
    for name in depends_on:
        if name not in index:
            raise ValueError(
                f"component 'other' depends on {name!r}, which is not in the merge order"
            )
        latest = max(latest, index[name])
    earliest = len(component_order)
    for name in required_by:
        if name not in index:
            raise ValueError(
                f"component 'other' is required by {name!r}, which is not in the merge order"
            )
        earliest = min(earliest, index[name])
    position = latest + 1
    if position > earliest:
        raise ValueError(
            "component 'other' depends on a component that also reads 'other'"
        )
    return position


def _linear_component_order(model_info: dict) -> List[str]:
    """Dense transformer order, with other inserted from its input dependencies."""
    if "num_layers" not in model_info:
        raise ValueError("model_info num_layers is required")
    num_layers = model_info["num_layers"]
    if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 0:
        raise ValueError(
            f"num_layers must be a non-negative integer, got {num_layers!r}"
        )
    order = ["embedding"]
    for index in range(num_layers):
        order.append(f"layer_{index}")
    order.extend(["norm", "lm_head"])
    # Instead of blindly appending "other" at the end,
    # analyze its input dependencies and insert at the correct position.
    other_dependencies = analyze_component_dependencies("other", model_info)
    insert_position = find_correct_insert_position(other_dependencies, order)
    order.insert(insert_position, "other")
    return order


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


def _component_graph(component):
    """Return the graph on a loaded component model."""
    graph = getattr(component, "graph", None)
    if graph is None or isinstance(graph, (str, bytes, int, bool)):
        raise ValueError(f"component must be an ONNX model, got {component!r}")
    return graph


def _io_values(component, role: str):
    """Names and value infos for graph inputs or graph outputs."""
    graph = _component_graph(component)
    values = getattr(graph, role, None)
    if values is None or isinstance(values, (str, bytes)):
        raise ValueError(f"component graph {role} must be a list, got {values!r}")
    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError(f"component graph {role} must be a list, got {values!r}") from exc
    names: List[str] = []
    by_name: Dict[str, Any] = {}
    for item in items:
        name = getattr(item, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"component graph {role} name must be a non-empty string, got {name!r}"
            )
        name = name.strip()
        if name in by_name:
            raise ValueError(f"component graph {role} repeats {name!r}")
        names.append(name)
        by_name[name] = item
    return names, by_name


def _value_dtype(value) -> Optional[str]:
    """Canonical dtype from a value info. An unset element type stays unset."""
    type_proto = getattr(value, "type", None)
    tensor_type = getattr(type_proto, "tensor_type", None) if type_proto is not None else None
    if tensor_type is None:
        return None
    elem = getattr(tensor_type, "elem_type", 0)
    if isinstance(elem, bool) or not isinstance(elem, int) or elem == 0:
        return None
    dtype = _ONNX_ELEM_DTYPE.get(elem)
    if dtype is None:
        name = getattr(value, "name", "")
        raise ValueError(f"tensor {name} has unsupported ONNX element type {elem}")
    return _canonical_dtype_name(dtype)


def _value_shape(value) -> Optional[List[Optional[int]]]:
    """Shape from a value info. A symbolic axis is None. A missing shape stays unset."""
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
    name = getattr(value, "name", "")
    for dim in dims:
        param = getattr(dim, "dim_param", "") or ""
        if isinstance(param, str) and param:
            parsed.append(None)
            continue
        number = getattr(dim, "dim_value", 0)
        if isinstance(number, bool) or not isinstance(number, int):
            raise ValueError(
                f"tensor {name} dim must be an integer, got {number!r}"
            )
        if number < 0:
            raise ValueError(f"tensor {name} dim {number} is negative")
        parsed.append(None if number == 0 else number)
    return parsed


def _declared_shape(shape, label: str) -> List[Optional[int]]:
    """Copy a boundary shape. Null is a symbolic axis."""
    if isinstance(shape, (str, bytes)) or not isinstance(shape, list):
        raise ValueError(f"{label} must be a list, got {shape!r}")
    parsed: List[Optional[int]] = []
    for dim in shape:
        if dim is None:
            parsed.append(None)
            continue
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise ValueError(f"{label} dim must be an integer or null, got {dim!r}")
        if dim < 0:
            raise ValueError(f"{label} dim {dim} is negative")
        parsed.append(dim)
    return parsed


def _shapes_compatible(left: List[Optional[int]], right: List[Optional[int]]) -> bool:
    """Same rank. A symbolic axis matches a concrete axis. Concrete axes must be equal."""
    if len(left) != len(right):
        return False
    for declared, observed in zip(left, right):
        if declared is None or observed is None:
            continue
        if declared != observed:
            return False
    return True


def _explicit_graph_inputs(model_info: dict) -> set:
    """Names recorded as graph-level inputs. The key may be absent."""
    if "graph_inputs" not in model_info:
        return set()
    raw = model_info["graph_inputs"]
    if isinstance(raw, (str, bytes)) or not isinstance(raw, list):
        raise ValueError(f"graph_inputs must be a list, got {raw!r}")
    names = set()
    for item in raw:
        if isinstance(item, str):
            if not item.strip():
                raise ValueError(
                    f"graph input must be a non-empty string, got {item!r}"
                )
            names.add(item.strip())
            continue
        if isinstance(item, dict):
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"graph input name must be a non-empty string, got {name!r}"
                )
            names.add(name.strip())
            continue
        raise ValueError(f"graph input must be a string or a dict, got {item!r}")
    return names


def _boundary_records(model_info: dict) -> List[dict]:
    """Named boundary tensors. A record without a name is not a reconnect contract."""
    boundaries = model_info.get("boundary_tensors")
    if boundaries is None:
        return []
    if isinstance(boundaries, (str, bytes)) or not isinstance(boundaries, list):
        raise ValueError(f"boundary_tensors must be a list, got {boundaries!r}")
    records = []
    seen = set()
    for item in boundaries:
        if not isinstance(item, dict):
            raise ValueError(f"boundary tensor must be a dict, got {item!r}")
        if "name" not in item:
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"boundary tensor name must be a non-empty string, got {name!r}"
            )
        name = name.strip()
        if name in seen:
            raise ValueError(f"boundary_tensors contains a duplicate name {name!r}")
        seen.add(name)
        producer = item.get("producer_component", item.get("producer"))
        if not isinstance(producer, str) or not producer.strip():
            raise ValueError(
                f"producer_component must be a non-empty string, got {producer!r}"
            )
        producer = producer.strip()
        consumers = item.get("consumer_components", item.get("consumers", []))
        if consumers is None:
            consumers = []
        dtype = None
        if "dtype" in item and item.get("dtype") is not None:
            dtype = _canonical_dtype_name(item.get("dtype"))
        shape = None
        if "shape" in item and item.get("shape") is not None:
            shape = _declared_shape(item.get("shape"), f"boundary {name} shape")
        records.append({
            "name": name,
            "producer": producer,
            "consumers": _name_list(consumers, "consumer_components"),
            "dtype": dtype,
            "shape": shape,
        })
    return records


def _match_component_models(components: list, model_info: dict) -> Dict[str, Any]:
    """Map a component name to the loaded model when the graph identifies it."""
    info = model_info.get("components")
    if info is None:
        info = {}
    if not isinstance(info, dict):
        raise ValueError(f"model_info components must be a dict, got {info!r}")
    matched: Dict[str, Any] = {}
    for model in components:
        graph = _component_graph(model)
        label = None
        graph_name = getattr(graph, "name", None)
        if isinstance(graph_name, str) and graph_name.strip():
            text = graph_name.strip()
            if text.endswith("_graph"):
                candidate = text[: -len("_graph")]
            else:
                candidate = text
            if candidate in info:
                label = candidate
        if label is None:
            output_names, _ = _io_values(model, "output")
            for name, comp in info.items():
                if name in matched or not isinstance(comp, dict):
                    continue
                if "output_names" not in comp:
                    continue
                recorded = _name_list(comp.get("output_names"), f"{name} output_names")
                if recorded and set(recorded) == set(output_names):
                    label = name
                    break
        if label is None:
            continue
        if label in matched:
            raise ValueError(f"component {label!r} matches more than one loaded model")
        matched[label] = model
    return matched


def validate_boundary_reconnection(components, model_info) -> Dict[str, Any]:
    """
    Verify all boundary tensors reconnect correctly.

    A downstream input must be an output of an earlier component or a
    graph-level input. A named boundary must use that exact name on the
    producer output and on each consumer input. Declared dtypes must agree.
    Shapes must share a rank, and a symbolic axis matches a concrete axis.

    Returns:
        valid and the list of reconnect errors.

    Raises:
        ValueError: If components, model_info, a tensor name, or a dtype is unusable.
    """
    if isinstance(components, (str, bytes)) or not isinstance(components, list):
        raise ValueError(f"components must be a list, got {components!r}")
    if not isinstance(model_info, dict):
        raise ValueError(f"model_info must be a dict, got {model_info!r}")

    parsed = []
    for component in components:
        input_names, input_values = _io_values(component, "input")
        output_names, output_values = _io_values(component, "output")
        parsed.append({
            "model": component,
            "inputs": input_names,
            "outputs": output_names,
            "input_values": input_values,
            "output_values": output_values,
        })

    errors: List[str] = []
    graph_inputs = _explicit_graph_inputs(model_info)
    boundaries = _boundary_records(model_info)
    for record in boundaries:
        if record["producer"] == "external":
            graph_inputs.add(record["name"])
    for index, item in enumerate(parsed):
        for name in item["inputs"]:
            produced_elsewhere = any(
                name in other["outputs"]
                for other_index, other in enumerate(parsed)
                if other_index != index
            )
            if not produced_elsewhere:
                graph_inputs.add(name)

    # Check that downstream inputs are satisfied by an earlier output
    # or by graph-level inputs. Outputs already merged stay available,
    # so a component inserted between a producer and its consumer still
    # leaves that tensor connected.
    available = set(graph_inputs)
    for index, item in enumerate(parsed):
        upstream_outputs = set(parsed[index - 1]["outputs"]) if index else set()
        for inp in item["inputs"]:
            if inp not in upstream_outputs and inp not in available:
                errors.append(f"Unresolved input: {inp}")
        available.update(item["outputs"])

    producers_of: Dict[str, Any] = {}
    for index, item in enumerate(parsed):
        for name, value in item["output_values"].items():
            previous = producers_of.get(name)
            if previous is not None:
                raise ValueError(
                    f"tensor {name!r} is produced by more than one component"
                )
            producers_of[name] = (index, value)

    matched = _match_component_models(components, model_info)
    parsed_by_model = {id(item["model"]): item for item in parsed}

    for record in boundaries:
        name = record["name"]
        producer = record["producer"]
        consumers = [
            consumer for consumer in record["consumers"]
            if consumer not in ("external", producer)
        ]
        if producer == "external":
            if not any(name in item["inputs"] for item in parsed):
                errors.append(f"boundary {name} has no matching input")
        else:
            producer_item = None
            if producer in matched:
                producer_item = parsed_by_model.get(id(matched[producer]))
            if producer_item is not None:
                if name not in producer_item["outputs"]:
                    errors.append(f"boundary {name} is not an output of {producer}")
            elif name not in producers_of:
                errors.append(f"boundary {name} has no matching output")
            if consumers:
                for consumer in consumers:
                    if consumer in matched:
                        consumer_item = parsed_by_model.get(id(matched[consumer]))
                        present = consumer_item is not None and name in consumer_item["inputs"]
                    else:
                        present = any(name in item["inputs"] for item in parsed)
                    if not present:
                        errors.append(f"boundary {name} is not an input of {consumer}")
            elif "external" not in record["consumers"]:
                producer_index = None
                if producer_item is not None:
                    producer_index = next(
                        item_index
                        for item_index, item in enumerate(parsed)
                        if item is producer_item
                    )
                elif name in producers_of:
                    producer_index = producers_of[name][0]
                if producer_index is None:
                    if not any(name in item["inputs"] for item in parsed):
                        errors.append(f"boundary {name} has no matching input")
                elif producer_index < len(parsed) - 1:
                    later = parsed[producer_index + 1:]
                    if not any(name in item["inputs"] for item in later):
                        errors.append(f"boundary {name} has no matching input")

        declared_dtype = record["dtype"]
        declared_shape = record["shape"]
        compared = []
        if name in producers_of and producer != "external":
            compared.append(producers_of[name][1])
        for item in parsed:
            if name in item["input_values"]:
                compared.append(item["input_values"][name])
        for value in compared:
            dtype = _value_dtype(value)
            shape = _value_shape(value)
            if declared_dtype and dtype and dtype != declared_dtype:
                errors.append(
                    f"boundary {name} dtype {dtype} does not match {declared_dtype}"
                )
            if (
                declared_shape is not None
                and shape is not None
                and not _shapes_compatible(shape, declared_shape)
            ):
                errors.append(
                    f"boundary {name} shape {list(shape)} does not match {list(declared_shape)}"
                )

    for name, (producer_index, producer_value) in producers_of.items():
        producer_dtype = _value_dtype(producer_value)
        producer_shape = _value_shape(producer_value)
        for item in parsed[producer_index + 1:]:
            if name not in item["input_values"]:
                continue
            consumer_value = item["input_values"][name]
            consumer_dtype = _value_dtype(consumer_value)
            consumer_shape = _value_shape(consumer_value)
            if producer_dtype and consumer_dtype and producer_dtype != consumer_dtype:
                errors.append(
                    f"boundary {name} dtype {producer_dtype} does not match {consumer_dtype}"
                )
            if (
                producer_shape is not None
                and consumer_shape is not None
                and not _shapes_compatible(producer_shape, consumer_shape)
            ):
                errors.append(
                    f"boundary {name} shape {list(producer_shape)} does not match {list(consumer_shape)}"
                )

    unique: List[str] = []
    for error in errors:
        if error not in unique:
            unique.append(error)
    return {"valid": len(unique) == 0, "errors": unique}


_QUANT_METHODS = ("MatMulNBits", "AWQ", "GPTQ", "QLinearMatMul")
_QUANT_COUNT_KEYS = _QUANT_METHODS + (
    "DequantizeLinear",
    "QuantizeLinear",
    "DynamicQuantizeLinear",
)


def _quant_initializer_map(graph) -> Dict[str, Any]:
    """Initializer name to tensor. A blank or repeated name raises."""
    raw = getattr(graph, "initializer", None)
    if raw is None:
        return {}
    if isinstance(raw, (str, bytes)):
        raise ValueError(f"graph initializers must be a list, got {raw!r}")
    try:
        items = list(raw)
    except TypeError as exc:
        raise ValueError(f"graph initializers must be a list, got {raw!r}") from exc
    found: Dict[str, Any] = {}
    for initializer in items:
        name = getattr(initializer, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"initializer name must be a non-empty string, got {name!r}"
            )
        name = name.strip()
        if name in found:
            raise ValueError(f"initializer {name!r} is duplicated")
        found[name] = initializer
    return found


def _remember_quant_tensor(name: str, names: List[str], present: List[str], initializers: Dict[str, Any]) -> None:
    """Record a scale or zero-point input and whether its initializer survived."""
    if not name:
        return
    names.append(name)
    if name in initializers:
        present.append(name)


def count_quant_ops(model) -> Dict[str, Any]:
    """
    Count quantization ops and the tensors that keep them quantized.

    MatMulNBits, AWQ, GPTQ, and QLinearMatMul contribute a scale, an optional
    zero point, a packing format, bits, and a group size. DequantizeLinear is
    counted so a float MatMul that replaced a packed weight is visible. Packing
    comes from the weight initializer. A missing dtype stays unset.
    """
    graph, nodes = _boundary_graph(model)
    info = _value_info_map(graph)
    initializers = _quant_initializer_map(graph)
    ops = {key: 0 for key in _QUANT_COUNT_KEYS}
    scales: List[str] = []
    zeros: List[str] = []
    scale_initializers: List[str] = []
    zero_initializers: List[str] = []
    packings: List[str] = []
    bit_values: List[int] = []
    unset_bits = 0
    group_values: List[int] = []
    unset_groups = 0
    for node in nodes:
        op_type = getattr(node, "op_type", "")
        if op_type is None:
            op_type = ""
        if not isinstance(op_type, str):
            raise ValueError(f"node op_type must be a string, got {op_type!r}")
        op_type = op_type.strip()
        if op_type in ("DequantizeLinear", "QuantizeLinear", "DynamicQuantizeLinear"):
            ops[op_type] += 1
            if op_type == "DynamicQuantizeLinear":
                continue
            inputs = _node_input_names(node)
            _remember_quant_tensor(_input_at_quant(inputs, 1), scales, scale_initializers, initializers)
            _remember_quant_tensor(_input_at_quant(inputs, 2), zeros, zero_initializers, initializers)
            continue
        kind = _quant_kind(node)
        if kind is None:
            continue
        label = _quant_node_label(node)
        attrs = _attribute_map(node, label)
        inputs = _node_input_names(node)
        if kind == "QLinearMatMul":
            weight = _input_at_quant(inputs, 3)
            scale = _input_at_quant(inputs, 4)
            zero = _input_at_quant(inputs, 5) or _input_at_quant(inputs, 2)
        else:
            weight = _input_at_quant(inputs, 1)
            scale = _input_at_quant(inputs, 2)
            zero = _input_at_quant(inputs, 3)
            if _input_at_quant(inputs, 4) and kind != "AWQ":
                kind = "GPTQ"
        if kind not in ops:
            raise ValueError(f"quantization op {kind} is unknown")
        ops[kind] += 1
        _remember_quant_tensor(scale, scales, scale_initializers, initializers)
        _remember_quant_tensor(zero, zeros, zero_initializers, initializers)
        bits = _optional_int_attr(attrs, label, "bits")
        group_size = _optional_int_attr(attrs, label, "block_size", "group_size")
        storage = _recorded_quant_dtype(weight, info, initializers) if weight else None
        packing = _packing_for_storage(kind, bits, storage)
        packings.append(packing if packing is not None else "unset")
        if bits is None:
            unset_bits += 1
        else:
            bit_values.append(bits)
        if group_size is None:
            unset_groups += 1
        else:
            group_values.append(group_size)
    return {
        "ops": ops,
        "scale_tensors": sorted(scales),
        "zero_point_tensors": sorted(zeros),
        "scale_initializers": sorted(scale_initializers),
        "zero_point_initializers": sorted(zero_initializers),
        "packing_formats": sorted(packings),
        "bits": sorted(bit_values),
        "unset_bits": unset_bits,
        "group_sizes": sorted(group_values),
        "unset_groups": unset_groups,
    }


def _input_at_quant(names: List[str], index: int) -> str:
    if index < 0 or index >= len(names):
        return ""
    return names[index]


def verify_quantization_coherence(original_model, merged_model) -> Dict[str, Any]:
    """
    Verify quantization ops survived split/merge.

    Scale tensors, zero-point tensors, and packing formats are part of the
    count. A DequantizeLinear that was not in the original model means the
    packed weights were expanded. ``coherent`` is true when both counts match.

    Returns:
        The original count, the merged count, and whether they match.

    Raises:
        ValueError: If either graph, a quant attribute, or an initializer is unusable.
    """
    original_quant_ops = count_quant_ops(original_model)
    merged_quant_ops = count_quant_ops(merged_model)
    return {
        "original": original_quant_ops,
        "merged": merged_quant_ops,
        "coherent": original_quant_ops == merged_quant_ops,
    }


def _check_saved_quantization(merged_model, quantization_info) -> None:
    """Raise when the merged graph drops the quantization contract saved at split."""
    if quantization_info is None:
        return
    if not isinstance(quantization_info, dict):
        raise ValueError(
            f"quantization_info must be a dict, got {quantization_info!r}"
        )
    counted = count_quant_ops(merged_model)
    method = quantization_info.get("method")
    if method is None:
        for name in _QUANT_METHODS:
            if counted["ops"][name]:
                raise ValueError(
                    f"quantization coherence failed: merged graph has {name}"
                )
        if counted["ops"]["DequantizeLinear"]:
            raise ValueError(
                "quantization coherence failed: merged graph has DequantizeLinear"
            )
        return
    if not isinstance(method, str) or method not in _QUANT_METHODS:
        raise ValueError(
            f"quantization_info method must be a quantization op, got {method!r}"
        )
    if counted["ops"][method] == 0:
        raise ValueError(
            f"quantization coherence failed: quantization_info records {method} "
            "but the merged graph has none"
        )
    if method != "QLinearMatMul" and counted["ops"]["DequantizeLinear"]:
        raise ValueError(
            "quantization coherence failed: merged graph has DequantizeLinear"
        )
    if not counted["scale_tensors"] or not counted["scale_initializers"]:
        raise ValueError("quantization coherence failed: scale tensors are missing")
    has_zero_point = quantization_info.get("has_zero_point")
    if has_zero_point is not None and not isinstance(has_zero_point, bool):
        raise ValueError(
            f"quantization_info has_zero_point must be a bool, got {has_zero_point!r}"
        )
    if has_zero_point is True and not counted["zero_point_initializers"]:
        raise ValueError("quantization coherence failed: zero-point tensors are missing")
    if has_zero_point is False and counted["zero_point_tensors"]:
        raise ValueError("quantization coherence failed: unexpected zero-point tensors")
    packing = quantization_info.get("packing_format")
    if isinstance(packing, str) and packing.strip():
        packing = packing.strip()
        if any(item != packing for item in counted["packing_formats"]):
            raise ValueError(
                "quantization coherence failed: packing "
                f"{counted['packing_formats']} does not match {packing}"
            )
    bits = quantization_info.get("bits")
    if bits is not None:
        if isinstance(bits, bool) or not isinstance(bits, int):
            raise ValueError(f"quantization_info bits must be an integer, got {bits!r}")
        if counted["unset_bits"] or any(item != bits for item in counted["bits"]):
            raise ValueError(
                f"quantization coherence failed: bits {counted['bits']} do not match {bits}"
            )
    group_size = quantization_info.get("group_size")
    if group_size is not None:
        if isinstance(group_size, bool) or not isinstance(group_size, int):
            raise ValueError(
                f"quantization_info group_size must be an integer, got {group_size!r}"
            )
        if counted["unset_groups"] or any(item != group_size for item in counted["group_sizes"]):
            raise ValueError(
                "quantization coherence failed: group size "
                f"{counted['group_sizes']} does not match {group_size}"
            )


def _plan_from_value(execution_plan) -> ExecutionPlan:
    """Copy an ExecutionPlan, or build one from a dict. Other types raise."""
    if isinstance(execution_plan, ExecutionPlan):
        return _coerce_execution_plan(execution_plan)
    if isinstance(execution_plan, dict):
        return ExecutionPlan.from_dict(execution_plan)
    raise ValueError(
        f"execution_plan must be an ExecutionPlan or a dict, got {execution_plan!r}"
    )


def _plan_component_order(execution_plan, components: dict) -> List[str]:
    """
    Merge in step-list order.

    The plan names every component in model_info. A dependency that appears
    later in that list is rejected, because that component would be consumed
    before it is merged.
    """
    plan = _plan_from_value(execution_plan)
    names = [step.component_name for step in plan.steps]
    position = {name: index for index, name in enumerate(names)}
    for step in plan.steps:
        for dependency in step.depends_on:
            if position[dependency] > position[step.component_name]:
                raise ValueError(
                    f"Step {step.component_name!r} depends on {dependency!r}, "
                    "which is merged later"
                )
    planned = set(names)
    for name in names:
        if name not in components:
            raise ValueError(f"execution plan names unknown component {name!r}")
    for name in components:
        if name not in planned:
            raise ValueError(f"execution plan omits component {name!r}")
    return names


def merge_onnx(
    output_dir: str,
    save_path: Optional[str] = None,
    execution_plan: Optional[Dict] = None,
) -> onnx.ModelProto:
    """
    Merge all component ONNX files back into a single model.

    Args:
        output_dir: Directory containing component ONNX files.
        save_path: Optional path to save the merged model.
        execution_plan: Optional plan whose step list is the merge order.
            When omitted, components merge as embedding, layers, norm,
            and lm_head. other is inserted where its inputs are produced.
            Loaded components are checked with validate_boundary_reconnection
            before the graph is built. A renamed boundary, a dtype mismatch,
            or a shape mismatch raises. When model_info records
            quantization_info, the merged graph must still contain that
            op, its scale and zero-point tensors, and its packing format.

    Returns:
        Merged ONNX ModelProto.

    Raises:
        FileNotFoundError: If model_info.json not found.
        ValueError: If the directory, the plan, or a required component file
            is unusable.

    Example:
        >>> merged_model = merge_onnx("./output/model")
        >>> onnx.save(merged_model, "./merged.onnx")
    """
    print(f"\nMerging components from: {output_dir}")
    _require_merge_dir(output_dir)
    destination = _require_save_path(save_path)
    model_info = _load_model_info(output_dir)
    components = _component_map(model_info)
    planned = execution_plan is not None
    if planned:
        component_order = _plan_component_order(execution_plan, components)
    else:
        component_order = _linear_component_order(model_info)

    # Collect all nodes and initializers in order
    all_nodes = []
    all_initializers = []
    all_inputs = []
    all_outputs = []
    lm_head_outputs = None  # Track lm_head outputs explicitly as final graph outputs
    opset_imports = None
    ir_version = None

    seen_initializers = set()
    seen_inputs = set()
    loaded_models = []

    for comp_name in component_order:
        if comp_name not in components:
            continue

        comp_info = components[comp_name]
        if not isinstance(comp_info, dict):
            raise ValueError(
                f"component {comp_name} must be a dict, got {comp_info!r}"
            )
        filename = comp_info.get("file", "")
        if not isinstance(filename, str) or not filename.strip():
            if planned:
                raise ValueError(f"component {comp_name!r} has no file")
            continue

        filepath = os.path.join(output_dir, filename)
        if os.path.isdir(filepath):
            raise ValueError(f"component file must be a file, got a directory {filepath}")
        if not os.path.isfile(filepath):
            if planned:
                raise ValueError(f"component file not found: {filename}")
            print(f"  Warning: {filename} not found, skipping")
            continue

        # Load component
        comp_model = onnx.load(filepath, load_external_data=True)
        loaded_models.append(comp_model)

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

        # A plan uses the last step as the graph output, including an MTP head.
        # The linear chain still prefers lm_head when that component loaded.
        if planned or comp_name != "lm_head":
            all_outputs = list(comp_model.graph.output)
        if not planned and comp_name == "lm_head":
            lm_head_outputs = list(comp_model.graph.output)

        print(f"  Loaded {filename}: {len(comp_model.graph.node)} nodes")

    reconnection = validate_boundary_reconnection(loaded_models, model_info)
    if not reconnection["valid"]:
        detail = "; ".join(reconnection["errors"])
        raise ValueError(f"boundary reconnection failed: {detail}")

    # Use lm_head outputs as final outputs if available (proper transformer output)
    final_outputs = all_outputs if planned else (lm_head_outputs if lm_head_outputs else all_outputs)

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

    _check_saved_quantization(merged_model, model_info.get("quantization_info"))

    print(f"  Merged model: {len(all_nodes)} nodes, {len(all_initializers)} initializers")

    # Optionally save
    if destination:
        onnx.save(merged_model, destination)
        print(f"  Saved merged model to: {destination}")

    return merged_model


class _HeldGraph:
    """Input and output names kept after the component model is released."""

    def __init__(self, inputs, outputs):
        self.input = inputs
        self.output = outputs
        self.node = []
        self.initializer = []
        self.value_info = []


class _HeldModel:
    """Boundary view that does not retain the component weight payload."""

    def __init__(self, inputs, outputs):
        self.graph = _HeldGraph(inputs, outputs)


def _require_memory_budget(memory_budget_mb) -> int:
    """A positive whole number of megabytes. A bool or a zero raises."""
    if isinstance(memory_budget_mb, bool) or not isinstance(memory_budget_mb, int):
        raise ValueError(
            f"memory_budget_mb must be a positive int, got {memory_budget_mb!r}"
        )
    if memory_budget_mb <= 0:
        raise ValueError(
            f"memory_budget_mb must be a positive int, got {memory_budget_mb!r}"
        )
    return memory_budget_mb


def _require_streaming_save_path(save_path: str) -> str:
    """Streaming merge always writes a file. None and a blank path raise."""
    destination = _require_save_path(save_path)
    if destination is None:
        raise ValueError(
            f"save_path must be a non-empty string, got {save_path!r}"
        )
    return destination


def _graph_sequence(value, role: str) -> list:
    """Copy a graph field into a list. A string or a missing field raises."""
    if value is None or isinstance(value, (str, bytes)):
        raise ValueError(f"component graph {role} must be a list, got {value!r}")
    try:
        return list(value)
    except TypeError as exc:
        raise ValueError(
            f"component graph {role} must be a list, got {value!r}"
        ) from exc


def _detach_message(value):
    """Own a protobuf message so the source model can be released."""
    copy_from = getattr(value, "CopyFrom", None)
    if not callable(copy_from):
        return value
    try:
        clone = value.__class__()
    except Exception:
        return value
    clone.CopyFrom(value)
    return clone


def _tensor_bytes(init) -> int:
    """Bytes of one initializer. raw_data wins, then a declared memory_bytes."""
    raw = getattr(init, "raw_data", None)
    if isinstance(raw, (bytes, bytearray)) and len(raw) > 0:
        return len(raw)
    for field, width in (
        ("float_data", 4),
        ("int32_data", 4),
        ("int64_data", 8),
        ("double_data", 8),
        ("uint64_data", 8),
    ):
        data = getattr(init, field, None)
        if data is None or isinstance(data, (str, bytes)):
            continue
        try:
            count = len(data)
        except TypeError:
            continue
        if count > 0:
            return count * width
    explicit = getattr(init, "memory_bytes", None)
    if explicit is None:
        return 0
    if isinstance(explicit, bool) or not isinstance(explicit, int) or explicit < 0:
        raise ValueError(
            f"initializer memory_bytes must be a non-negative int, got {explicit!r}"
        )
    return explicit


def merge_onnx_streaming(
    output_dir: str,
    save_path: str,
    memory_budget_mb: int = 4096,
) -> onnx.ModelProto:
    """
    Merge component files one at a time and write the result.

    Each component is loaded with its external data, its nodes and
    initializers are appended to the merged graph, and that component
    model is released before the next file is opened. Retained initializer
    bytes must stay within ``memory_budget_mb`` (4096 MB by default).
    The linear order is embedding, layers, norm, and lm_head, with other
    inserted from its inputs. Boundary reconnection and a saved
    quantization contract are checked before the file is written.

    Args:
        output_dir: Directory containing component ONNX files.
        save_path: File path for the merged model.
        memory_budget_mb: Maximum retained initializer bytes, in megabytes.

    Returns:
        Merged ONNX ModelProto.

    Raises:
        FileNotFoundError: If model_info.json is missing.
        ValueError: If the directory, the save path, the budget, or a
            component would pass the budget.
    """
    budget = _require_memory_budget(memory_budget_mb)
    destination = _require_streaming_save_path(save_path)
    print(f"\nStreaming merge from: {output_dir}")
    _require_merge_dir(output_dir)
    model_info = _load_model_info(output_dir)
    components = _component_map(model_info)
    component_order = _linear_component_order(model_info)
    budget_bytes = budget * 1024 * 1024

    all_nodes = []
    all_initializers = []
    all_inputs = []
    all_outputs = []
    lm_head_outputs = None
    opset_imports = None
    ir_version = None
    seen_initializers = set()
    seen_inputs = set()
    loaded_views = []
    retained = 0

    for comp_name in component_order:
        if comp_name not in components:
            continue

        comp_info = components[comp_name]
        if not isinstance(comp_info, dict):
            raise ValueError(
                f"component {comp_name} must be a dict, got {comp_info!r}"
            )
        filename = comp_info.get("file", "")
        if not isinstance(filename, str) or not filename.strip():
            continue

        filepath = os.path.join(output_dir, filename)
        if os.path.isdir(filepath):
            raise ValueError(f"component file must be a file, got a directory {filepath}")
        if not os.path.isfile(filepath):
            print(f"  Warning: {filename} not found, skipping")
            continue

        comp_model = onnx.load(filepath, load_external_data=True)
        try:
            graph = _component_graph(comp_model)
            nodes = [
                _detach_message(item)
                for item in _graph_sequence(getattr(graph, "node", None), "node")
            ]
            initializers = [
                _detach_message(item)
                for item in _graph_sequence(
                    getattr(graph, "initializer", None), "initializer"
                )
            ]
            inputs = [
                _detach_message(item)
                for item in _graph_sequence(getattr(graph, "input", None), "input")
            ]
            outputs = [
                _detach_message(item)
                for item in _graph_sequence(getattr(graph, "output", None), "output")
            ]
            if opset_imports is None:
                opset_value = getattr(comp_model, "opset_import", [])
                if opset_value is None:
                    opset_value = []
                opset_imports = [
                    _detach_message(item)
                    for item in _graph_sequence(opset_value, "opset_import")
                ]
                ir_version = getattr(comp_model, "ir_version", None)

            kept = []
            added = 0
            for init in initializers:
                name = getattr(init, "name", None)
                if not isinstance(name, str):
                    raise ValueError(
                        f"initializer name must be a string, got {name!r}"
                    )
                if name in seen_initializers:
                    continue
                added += _tensor_bytes(init)
                kept.append((name, init))
            if retained + added > budget_bytes:
                raise ValueError(
                    f"component {comp_name} exceeds memory budget of {budget} MB"
                )
            retained += added
            for name, init in kept:
                seen_initializers.add(name)
                all_initializers.append(init)

            all_nodes.extend(nodes)
            for inp in inputs:
                name = getattr(inp, "name", None)
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(
                        f"component graph input name must be a non-empty string, got {name!r}"
                    )
                if name not in seen_inputs and name not in seen_initializers:
                    all_inputs.append(inp)
                    seen_inputs.add(name)

            if comp_name != "lm_head":
                all_outputs = list(outputs)
            if comp_name == "lm_head":
                lm_head_outputs = list(outputs)
            loaded_views.append(_HeldModel(inputs, outputs))
            print(f"  Loaded {filename}: {len(nodes)} nodes")
        finally:
            comp_model = None
            gc.collect()

    reconnection = validate_boundary_reconnection(loaded_views, model_info)
    if not reconnection["valid"]:
        detail = "; ".join(reconnection["errors"])
        raise ValueError(f"boundary reconnection failed: {detail}")

    final_outputs = lm_head_outputs if lm_head_outputs else all_outputs
    merged_graph = helper.make_graph(
        nodes=all_nodes,
        name="merged_graph",
        inputs=all_inputs,
        outputs=final_outputs,
        initializer=all_initializers,
    )
    merged_model = helper.make_model(
        merged_graph,
        producer_name="chunking_utils_merged",
        opset_imports=opset_imports or [],
    )
    if ir_version:
        merged_model.ir_version = ir_version

    _check_saved_quantization(merged_model, model_info.get("quantization_info"))
    print(f"  Merged model: {len(all_nodes)} nodes, {len(all_initializers)} initializers")
    onnx.save(merged_model, destination)
    print(f"  Saved merged model to: {destination}")
    return merged_model


def merge_onnx_with_result(
    output_dir: str,
    save_path: Optional[str] = None,
    execution_plan: Optional[Dict] = None,
) -> OnnxMergeResult:
    """
    Merge components and return a result object.

    Args:
        output_dir: Directory containing component ONNX files.
        save_path: Optional path to save the merged model.
        execution_plan: Optional plan whose step list is the merge order.

    Returns:
        OnnxMergeResult with merge information.
    """
    try:
        merged_model = merge_onnx(output_dir, save_path, execution_plan)
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


def _require_sample_inputs(sample_inputs) -> Dict[str, Any]:
    """Named numeric feeds. A string, a blank name, or an empty map raises."""
    if isinstance(sample_inputs, (str, bytes)) or not isinstance(sample_inputs, dict):
        raise ValueError(f"sample_inputs must be a dict, got {sample_inputs!r}")
    if not sample_inputs:
        raise ValueError("sample_inputs must contain at least one input")
    feeds: Dict[str, Any] = {}
    for name, value in sample_inputs.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"sample input name must be a non-empty string, got {name!r}")
        text = name.strip()
        if text in feeds:
            raise ValueError(f"sample input {text!r} is duplicated")
        _as_signal(value, text)
        feeds[text] = value
    return feeds


def _original_model_file(output_dir: str, model_info: dict) -> str:
    """Resolve model_info original_model to a file. A directory raises."""
    raw = model_info.get("original_model")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(
            f"model_info original_model must be a non-empty string, got {raw!r}"
        )
    raw = raw.strip()
    if os.path.isabs(raw):
        candidates = [raw]
    else:
        parent = os.path.dirname(os.path.abspath(output_dir))
        candidates = [os.path.join(output_dir, raw), os.path.join(parent, raw), raw]
    for path in candidates:
        if os.path.isdir(path):
            raise ValueError(f"original model must be a file, got a directory {path}")
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"original model not found: {raw}")


def _ort_session(model):
    """CPU session for a model path or an in-memory ONNX model."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ValueError("onnxruntime is required to verify merge PSNR") from exc
    chosen = None
    available = getattr(ort, "get_available_providers", lambda: [])()
    if "CPUExecutionProvider" in available:
        chosen = ["CPUExecutionProvider"]
    if isinstance(model, str):
        if chosen:
            return ort.InferenceSession(model, providers=chosen)
        return ort.InferenceSession(model)
    serialize = getattr(model, "SerializeToString", None)
    if not callable(serialize):
        raise ValueError("merged model must be an ONNX model")
    payload = serialize()
    if not isinstance(payload, (bytes, bytearray)):
        raise ValueError("merged model must serialize to bytes")
    if chosen:
        return ort.InferenceSession(payload, providers=chosen)
    return ort.InferenceSession(payload)


def _run_onnx_outputs(model, feeds: Dict[str, Any]) -> Dict[str, Any]:
    """Run one model and return output name to array."""
    session = _ort_session(model)
    names: List[str] = []
    for item in session.get_outputs():
        name = getattr(item, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"model output name must be a non-empty string, got {name!r}")
        text = name.strip()
        if text in names:
            raise ValueError(f"model output {text!r} is duplicated")
        names.append(text)
    if not names:
        raise ValueError("model outputs must contain at least one value")
    values = session.run(names, feeds)
    if not isinstance(values, (list, tuple)) or len(values) != len(names):
        raise ValueError("onnxruntime returned a different number of outputs")
    return {name: value for name, value in zip(names, values)}


def _worst_psnr(reference: Dict[str, Any], candidate: Dict[str, Any]) -> float:
    """Lowest PSNR across named outputs. Disagreeing names raise."""
    if set(reference) != set(candidate):
        raise ValueError(
            f"output names disagree: {sorted(reference)} and {sorted(candidate)}"
        )
    if not reference:
        raise ValueError("model outputs must contain at least one value")
    worst = None
    for name in sorted(reference):
        score = compute_psnr(reference[name], candidate[name])
        if worst is None or score < worst:
            worst = score
    if worst is None:
        raise ValueError("model outputs must contain at least one value")
    return worst


def merge_and_verify(
    output_dir: str,
    sample_inputs: Dict[str, Any],
    save_path: Optional[str] = None,
    min_psnr: float = 35.0,
) -> OnnxMergeResult:
    """
    Merge components and verify numerical correctness.

    The original model recorded in model_info and the merged graph both run
    on ``sample_inputs``. ``compute_psnr`` compares each output. The lowest
    score is kept. A result of at least ``min_psnr`` (35 dB) succeeds. A
    result above 40 dB is production accuracy. An exact match is infinite.

    Args:
        output_dir: Directory containing component ONNX files.
        sample_inputs: Named numeric arrays fed to both models.
        save_path: Optional path to save the merged model.
        min_psnr: Minimum PSNR in decibels. The default is 35.

    Returns:
        OnnxMergeResult with the PSNR. ``success`` is true when the score
        meets ``min_psnr``.

    Raises:
        ValueError: If the feeds, the threshold, the original model path,
            or the output names are unusable.
        FileNotFoundError: If model_info.json or the original model is missing.
    """
    feeds = _require_sample_inputs(sample_inputs)
    minimum = _real_number(min_psnr, "min_psnr")
    merged_model = merge_onnx(output_dir, save_path)
    model_info = _load_model_info(output_dir)
    original_path = _original_model_file(output_dir, model_info)
    original_outputs = _run_onnx_outputs(original_path, feeds)
    merged_outputs = _run_onnx_outputs(merged_model, feeds)
    psnr = _worst_psnr(original_outputs, merged_outputs)
    passed = psnr >= minimum
    error = None
    if not passed:
        _warn_psnr_failed(psnr, minimum)
        shown = "infinite" if math.isinf(psnr) else f"{psnr:.6g}"
        error = f"PSNR {shown} dB is below {minimum:g} dB"
    saved = save_path if isinstance(save_path, str) and save_path.strip() else None
    return OnnxMergeResult(
        success=passed,
        output_path=saved,
        num_nodes=len(merged_model.graph.node),
        num_initializers=len(merged_model.graph.initializer),
        error=error,
        psnr=psnr,
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
