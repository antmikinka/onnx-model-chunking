"""
ONNX model merging utilities.

This module provides utilities for merging split ONNX component files
back into a single unified ONNX model.
"""

import json
import os
from typing import Dict, List, Optional

import onnx
from onnx import helper

from chunking_utils.common.types import ExecutionPlan, _coerce_execution_plan

from .types import OnnxMergeResult


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

    print(f"  Merged model: {len(all_nodes)} nodes, {len(all_initializers)} initializers")

    # Optionally save
    if destination:
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
