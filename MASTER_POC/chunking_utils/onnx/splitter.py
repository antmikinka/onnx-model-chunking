"""
ONNX model splitting utilities.

``split_onnx`` is the orchestrator. Graph analysis, component graphs,
boundary contracts, quantization, KV cache, and the NPU execution policy
live in the modules next to this file. This module re-exports those
public functions so callers and tests keep one import path.
"""

import json
import os
from typing import Any, Dict, Optional

import onnx
from onnx import numpy_helper, helper
from onnx import shape_inference

from ..common.path_utils import sanitize_model_name
from ..common.architecture_detector import detect_architecture
from ..common.types import BoundaryTensor, ModelFormat, SplitConfig
from .boundary_contract import collect_boundary_tensors
from .component_builder import create_component_onnx, extract_components
from .graph_analysis import (
    ResolvedTensorType,
    detect_architecture_from_onnx,
    get_onnx_prefixes,
    resolve_tensor_type,
)
from .kv_contract import extract_kv_cache_info
from .attention_mask import rewrite_attention_mask_ops
from .loader import infer_shapes_with_lazy_fallback, load_onnx_model
from .metadata_writer import write_execution_contract
from .npu_policy import extract_execution_policy
from .quantization import build_quantization_info, extract_quantization_info, extract_quantization_tensors
from .types import OnnxSplitResult
from .validator import validate_boundary_contracts, validate_component_io
from .weight_store import (
    SIZE_THRESHOLD,
    STREAMING_CHUNK_SIZE,
    WeightStrategy,
    coerce_weight_strategy,
    store_component_weights,
)


def build_boundary_contract(model, components):
    """
    Build the boundary contract for components of one ONNX model.

    The collector is the function on this module, so a test patch of
    ``collect_boundary_tensors`` applies here.
    """
    return collect_boundary_tensors(model, components)


def save_components(
    model,
    components,
    boundaries,
    output_dir,
    *,
    strategy,
    size_threshold,
    chunk_size,
):
    """
    Create component graphs, validate them, store weights, and save files.

    IO and boundary checks run before any component file is written.
    External weights use the selected strategy before those files are written.
    """
    layer_components = {
        key: value for key, value in components.items() if value.component_type == "layer"
    }
    num_layers = len(layer_components)
    print(f"\nSplitting model into {len(components)} components ({num_layers} layers)...")
    print(f"Output directory: {output_dir}")

    component_files: Dict[str, str] = {}
    prepared = []
    for comp_name, comp_info in sorted(components.items()):
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
        component_model = create_component_onnx(model, comp_info, comp_name)
        comp_info.input_names = [item.name for item in component_model.graph.input]
        comp_info.output_names = [item.name for item in component_model.graph.output]
        prepared.append((comp_name, comp_info, filename, filepath, component_model))

    retained = _retain_live_boundaries(components, boundaries)
    if isinstance(boundaries, list):
        boundaries.clear()
        boundaries.extend(retained)
        live_boundaries = boundaries
    else:
        live_boundaries = retained
    validate_component_io(components, live_boundaries)
    validate_boundary_contracts(live_boundaries, components)

    if prepared:
        weight_storage = store_component_weights(
            [
                {"name": comp_name, "file": filename, "model": component_model}
                for comp_name, _comp_info, filename, _filepath, component_model in prepared
            ],
            output_dir,
            strategy=strategy,
            size_threshold=size_threshold,
            chunk_size=chunk_size,
        )
    else:
        if strategy is WeightStrategy.STREAMING and chunk_size is None:
            empty_chunk = STREAMING_CHUNK_SIZE
        elif strategy is WeightStrategy.STREAMING:
            empty_chunk = chunk_size
        else:
            empty_chunk = None
        weight_storage = {
            "strategy": strategy.value,
            "size_threshold": size_threshold,
            "chunk_size": empty_chunk,
            "deduplicated": strategy is not WeightStrategy.COPY,
            "shared_file": None,
            "manifest": None,
            "files": [],
            "components": [],
        }

    for comp_name, comp_info, filename, filepath, component_model in prepared:
        onnx.save(component_model, filepath)
        component_files[comp_name] = filepath
        print(
            f"  Saved {filename} ({len(comp_info.node_indices)} nodes, "
            f"{len(comp_info.initializer_names)} initializers)"
        )
    return component_files, weight_storage, num_layers


def _retain_live_boundaries(components, boundaries):
    """Drop crossings that were copied into the consumer graph.

    Shared Constant nodes are inlined, so they are no longer inputs.
    A boundary with no remaining reader is not a handoff.
    """
    kept = []
    for tensor in boundaries:
        consumers = []
        for name in list(getattr(tensor, "consumer_components", ()) or ()):
            info = components.get(name)
            inputs = getattr(info, "input_names", None) if info is not None else None
            if isinstance(inputs, (str, bytes)) or not isinstance(inputs, (list, tuple)):
                continue
            if tensor.name in inputs:
                consumers.append(name)
        if not consumers:
            continue
        if consumers == list(tensor.consumer_components):
            kept.append(tensor)
            continue
        kept.append(BoundaryTensor(
            name=tensor.name,
            dtype=tensor.dtype,
            shape=list(tensor.shape),
            producer_component=tensor.producer_component,
            consumer_components=consumers,
            is_stateful=tensor.is_stateful,
            cache_role=tensor.cache_role,
            device_preference=tensor.device_preference,
        ))
    return kept


def split_onnx(
    model: Optional[str] = None,
    output: Optional[str] = None,
    config: Optional[SplitConfig] = None,
    split_only: bool = False,
    weight_strategy: Any = None,
    weight_size_threshold: Optional[int] = None,
    weight_chunk_size: Optional[int] = None,
    **kwargs
) -> OnnxSplitResult:
    """
    Split an ONNX model into per-component ONNX files.

    Creates:
    - embeddings.onnx
    - layer_0_of_N.onnx, layer_1_of_N.onnx, ... (one per layer!)
    - norm.onnx
    - lm_head.onnx
    - model_info.json, including quantization_info, kv_cache_info,
      architecture_descriptor, execution_policy, and weight_storage

    Boundary contracts and component IO are validated before any component file is written.
    External weights use the selected weight strategy before those files are written.

    Args:
        model: Path to ONNX model file or directory. Required if config not provided.
        output: Output directory path. Required if config not provided.
        config: Optional SplitConfig object (overrides model/output if provided).
        split_only: If True, skip inference validation.
        weight_strategy: copy, hardlink, manifest, or streaming. Defaults to copy.
        weight_size_threshold: Minimum initializer size stored outside the graph.
        weight_chunk_size: Streaming chunk size. Other strategies ignore it.
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
    if config is None:
        if model is None or output is None:
            raise ValueError("Either 'config' or both 'model' and 'output' must be provided")
        config = SplitConfig(
            model=model,
            output=output,
            split_only=split_only,
            **kwargs
        )

    # A bad strategy raises before the model is loaded or any file is written.
    if weight_strategy is None:
        strategy = WeightStrategy.COPY
    else:
        strategy = coerce_weight_strategy(weight_strategy)
    if weight_size_threshold is None:
        weight_size_threshold = SIZE_THRESHOLD

    model_path = config.model
    onnx_model = load_onnx_model(model_path)
    # Attention Where/MaskedFill lowers to a scalar -1e4 add. Shape inference
    # then sees the Cast/Sub/Mul/Add nodes. A graph with no such fill is unchanged.
    replaced_masks = rewrite_attention_mask_ops(onnx_model)
    if replaced_masks:
        print(
            f"Rewrote {replaced_masks} attention mask Where/MaskedFill ops "
            "into an additive -1e4 bias"
        )

    print("Running static shape inference on the master model...")
    # Resolve infer_shapes at call time so a test patch still applies, and so a
    # missing attribute is the same caught failure as an inference error.
    onnx_model = infer_shapes_with_lazy_fallback(
        onnx_model,
        infer_shapes=lambda loaded: shape_inference.infer_shapes(loaded),
    )

    config_json = _load_sibling_config(model_path, "config.json")
    genai_config = _load_sibling_config(model_path, "genai_config.json")
    arch = detect_architecture(
        onnx_model.graph,
        config_json=config_json,
        genai_config=genai_config,
    )
    components = extract_components(
        onnx_model,
        config_json=config_json,
        genai_config=genai_config,
        arch=arch,
    )
    boundary_contract = build_boundary_contract(onnx_model, components)
    quant_info = extract_quantization_info(onnx_model)
    quant_tensors = extract_quantization_tensors(onnx_model)
    recorded_quant = dict(quant_info)
    recorded_quant.update(quant_tensors)
    kv_info = extract_kv_cache_info(onnx_model, components)
    execution_policy = extract_execution_policy(
        onnx_model,
        components,
        quantization_info=quant_info,
    )

    model_folder_name = sanitize_model_name(config.model)
    output_dir = os.path.join(config.output, model_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    component_files, weight_storage, num_layers = save_components(
        onnx_model,
        components,
        boundary_contract,
        output_dir,
        strategy=strategy,
        size_threshold=weight_size_threshold,
        chunk_size=weight_chunk_size,
    )

    model_info = {
        "original_model": str(model_path),
        "num_layers": num_layers,
        "num_components": len(components),
        "architecture": arch.arch_type.value,
        "architecture_descriptor": arch.to_dict(),
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
                "file": os.path.basename(component_files.get(name, "")),
            }
            for name, info in components.items()
        },
        "boundary_tensors": [tensor.to_dict() for tensor in boundary_contract],
        "quantization_info": recorded_quant,
        "kv_cache_info": kv_info,
        "execution_policy": execution_policy,
        "weight_storage": weight_storage,
    }
    write_execution_contract(output_dir, model_info)
    print("  Saved model_info.json")
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
        boundary_tensors=boundary_contract,
        quantization=build_quantization_info(onnx_model),
    )


def _load_sibling_config(model_path: str, filename: str) -> Optional[Dict[str, Any]]:
    """Read one JSON object beside the model. A missing file stays unset."""
    if isinstance(model_path, bool) or not isinstance(model_path, str) or not model_path.strip():
        raise ValueError(f"model path must be a non-empty string, got {model_path!r}")
    directory = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
    path = os.path.join(directory, filename)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            parsed = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{filename} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{filename} must be a JSON object")
    return parsed


__all__ = [
    "split_onnx",
    "extract_components",
    "create_component_onnx",
    "collect_boundary_tensors",
    "build_boundary_contract",
    "save_components",
    "write_execution_contract",
    "resolve_tensor_type",
    "ResolvedTensorType",
    "extract_quantization_info",
    "extract_kv_cache_info",
    "extract_execution_policy",
    "detect_architecture_from_onnx",
    "get_onnx_prefixes",
]
