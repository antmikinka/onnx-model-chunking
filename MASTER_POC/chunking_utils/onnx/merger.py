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


def _linear_component_order(model_info: dict) -> List[str]:
    """Dense transformer order: embedding, layers, norm, lm_head, other."""
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
    order.extend(["norm", "lm_head", "other"])
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
            lm_head, then other.

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
