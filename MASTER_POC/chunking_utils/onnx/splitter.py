"""
ONNX model splitting utilities.

``split_onnx`` is the orchestrator. Graph analysis, component graphs,
boundary contracts, quantization, KV cache, and the NPU execution policy
live in the modules next to this file. This module re-exports those
public functions so callers and tests keep one import path.
"""

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import onnx
from onnx import TensorProto, numpy_helper, helper
from onnx import shape_inference

from ..common.path_utils import sanitize_model_name
from ..common.architecture_detector import detect_architecture
from ..common.types import BoundaryTensor, DeviceType, ModelFormat, SplitConfig
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
from .static_shapes import _bake_matmul_nbits, apply_static_contract
from .metadata_writer import write_execution_contract
from .npu_policy import extract_execution_policy
from .quantization import build_quantization_info, extract_quantization_info, extract_quantization_tensors
from .quark_optimize import prepare_model_for_split
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


def _component_filename(comp_name: str, comp_info, num_layers: int) -> str:
    if comp_info.component_type == "layer":
        return f"layer_{comp_info.layer_id}_of_{num_layers}.onnx"
    if comp_info.component_type == "embedding":
        return "embeddings.onnx"
    if comp_info.component_type == "norm":
        return "norm.onnx"
    if comp_info.component_type == "lm_head":
        return "lm_head.onnx"
    return f"{comp_name}.onnx"


def _resolve_only_components(components, only_components, num_layers: int) -> Optional[List[str]]:
    """Return the component keys to write. None keeps every component."""
    if only_components is None:
        return None
    if isinstance(only_components, str) or not isinstance(only_components, (list, tuple)):
        raise ValueError(
            f"only_components must be a sequence of component names, got {only_components!r}"
        )
    by_name = {}
    for name, info in components.items():
        filename = _component_filename(name, info, num_layers)
        by_name[name] = name
        by_name[filename] = name
        by_name[os.path.splitext(filename)[0]] = name
    resolved = []
    for item in only_components:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"component name must be a non-empty string, got {item!r}")
        key = by_name.get(item.strip())
        if key is None:
            known = ", ".join(sorted(components))
            raise ValueError(f"unknown component {item!r}; known: {known}")
        if key not in resolved:
            resolved.append(key)
    if not resolved:
        raise ValueError("only_components did not select a component")
    return resolved


def _merge_weight_storage(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise ValueError("weight storage requires at least one component record")
    merged = dict(records[0])
    files: List[Any] = []
    stored: List[Any] = []
    for record in records:
        files.extend(record.get("files") or [])
        stored.extend(record.get("components") or [])
    merged["files"] = files
    merged["components"] = stored
    merged["shared_file"] = None
    return merged


def _record_component_io(comp_info, component_model) -> None:
    comp_info.input_names = [item.name for item in component_model.graph.input]
    comp_info.output_names = [item.name for item in component_model.graph.output]


def _rewrite_saved_component(
    component_model,
    comp_info,
    *,
    split_lm_head: bool,
    npu_bf16_fallback: bool,
    external_data_dir: Optional[str],
    lm_head_chunk_rows: Optional[int],
    apply_component_contract: bool,
    static_batch_size: int,
    static_max_seq_len: int,
    static_query_len: int,
    decompose_gqa: bool = False,
    sink_size: int = 0,
):
    """Rewrite one owned component graph. The master graph is not an argument."""
    if apply_component_contract:
        report = apply_static_contract(
            component_model,
            batch_size=int(static_batch_size),
            max_seq_len=int(static_max_seq_len),
            query_len=int(static_query_len),
            external_data_dir=external_data_dir,
            npu_bf16_fallback=False,
            defer_matmul_bake=True,
            decompose_gqa=bool(decompose_gqa),
            sink_size=int(sink_size),
        )
        print(
            f"  Component static contract: norms {len(report['decomposed_norms'])}, "
            f"split_matmuls {len(report['split_matmuls'])}, "
            f"bake_deferred {report['matmul_bake_deferred']}"
        )
    if split_lm_head and comp_info.component_type == "lm_head":
        from .lm_head_uniform import rewrite_uniform_lm_head

        # The uniform rewrite bakes this LM head when the flag is set, then
        # tiles it. A second dense bake would skip that pad.
        lm_report = rewrite_uniform_lm_head(
            component_model,
            chunk_rows=lm_head_chunk_rows,
            npu_bf16_fallback=bool(npu_bf16_fallback),
            external_data_dir=external_data_dir,
        )
        print(
            f"  LM head uniform split: {lm_report['chunks']} x {lm_report['chunk_rows']} "
            f"mode={lm_report['mode']} padded={lm_report['padded_vocab']} "
            f"vocab={lm_report['vocab']}"
        )
        return component_model
    if npu_bf16_fallback:
        baked = _bake_matmul_nbits(component_model, external_data_dir)
        print(f"  Baked {len(baked)} MatMulNBits in this component")
    return component_model


def save_components(
    model,
    components,
    boundaries,
    output_dir,
    *,
    strategy,
    size_threshold,
    chunk_size,
    split_lm_head: bool = False,
    npu_bf16_fallback: bool = False,
    external_data_dir: Optional[str] = None,
    lm_head_chunk_rows: Optional[int] = None,
    only_components: Optional[Sequence[str]] = None,
    apply_component_contract: bool = False,
    static_batch_size: int = 1,
    static_max_seq_len: int = 16,
    static_query_len: int = 1,
    decompose_gqa: bool = False,
    sink_size: int = 0,
):
    """
    Create component graphs, validate them, store weights, and save files.

    IO and boundary checks run before any component file is written.
    External weights use the selected strategy before those files are written.

    When ``npu_bf16_fallback`` is set, each saved component is contracted and
    baked after that validation. The master model's packed weights stay packed.
    One baked component is stored and released before the next one is built.
    """
    layer_components = {
        key: value for key, value in components.items() if value.component_type == "layer"
    }
    num_layers = len(layer_components)
    selected = _resolve_only_components(components, only_components, num_layers)
    print(f"\nSplitting model into {len(components)} components ({num_layers} layers)...")
    if selected is not None:
        print(f"Saving {len(selected)} of {len(components)} components: {', '.join(selected)}")
    print(f"Output directory: {output_dir}")

    component_files: Dict[str, str] = {}
    prepared = []
    for comp_name, comp_info in sorted(components.items()):
        filename = _component_filename(comp_name, comp_info, num_layers)
        filepath = os.path.join(output_dir, filename)
        component_model = create_component_onnx(model, comp_info, comp_name)
        keep = selected is None or comp_name in selected
        # The fallback bake happens after validation, one component at a time.
        # The packed LM-head rewrite still runs here when the flag is off.
        if keep and split_lm_head and comp_info.component_type == "lm_head" and not npu_bf16_fallback:
            _rewrite_saved_component(
                component_model,
                comp_info,
                split_lm_head=True,
                npu_bf16_fallback=False,
                external_data_dir=external_data_dir,
                lm_head_chunk_rows=lm_head_chunk_rows,
                apply_component_contract=False,
                static_batch_size=static_batch_size,
                static_max_seq_len=static_max_seq_len,
                static_query_len=static_query_len,
                decompose_gqa=bool(decompose_gqa),
                sink_size=int(sink_size),
            )
        _record_component_io(comp_info, component_model)
        if keep and not npu_bf16_fallback:
            prepared.append((comp_name, comp_info, filename, filepath, component_model))
        else:
            del component_model

    retained = _retain_live_boundaries(components, boundaries)
    if isinstance(boundaries, list):
        boundaries.clear()
        boundaries.extend(retained)
        live_boundaries = boundaries
    else:
        live_boundaries = retained
    validate_component_io(components, live_boundaries)
    validate_boundary_contracts(live_boundaries, components)

    if npu_bf16_fallback:
        # Rebuild each saved component from the packed master, bake that copy,
        # write it, and drop it before the next component is expanded.
        names = selected if selected is not None else [
            name for name, _info in sorted(components.items())
        ]
        records = []
        for comp_name in names:
            comp_info = components[comp_name]
            filename = _component_filename(comp_name, comp_info, num_layers)
            filepath = os.path.join(output_dir, filename)
            component_model = create_component_onnx(model, comp_info, comp_name)
            _rewrite_saved_component(
                component_model,
                comp_info,
                split_lm_head=bool(split_lm_head),
                npu_bf16_fallback=bool(npu_bf16_fallback),
                external_data_dir=external_data_dir,
                lm_head_chunk_rows=lm_head_chunk_rows,
                apply_component_contract=bool(apply_component_contract),
                static_batch_size=static_batch_size,
                static_max_seq_len=static_max_seq_len,
                static_query_len=static_query_len,
                decompose_gqa=bool(decompose_gqa),
                sink_size=int(sink_size),
            )
            _record_component_io(comp_info, component_model)
            if external_data_dir:
                inlined = _materialize_external_initializers(
                    component_model,
                    external_data_dir,
                )
                if inlined:
                    print(
                        f"  Inlined {inlined} master external initializers into {filename}"
                    )
            record = store_component_weights(
                [{"name": comp_name, "file": filename, "model": component_model}],
                output_dir,
                strategy=strategy,
                size_threshold=size_threshold,
                chunk_size=chunk_size,
            )
            records.append(record)
            onnx.save(component_model, filepath)
            component_files[comp_name] = filepath
            print(
                f"  Saved {filename} ({len(component_model.graph.node)} nodes, "
                f"{len(component_model.graph.initializer)} initializers)"
            )
            del component_model
        weight_storage = _merge_weight_storage(records)
        return component_files, weight_storage, num_layers

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
        node_count = len(getattr(component_model.graph, "node", comp_info.node_indices))
        init_count = len(getattr(component_model.graph, "initializer", comp_info.initializer_names))
        print(
            f"  Saved {filename} ({node_count} nodes, "
            f"{init_count} initializers)"
        )
    return component_files, weight_storage, num_layers


def _materialize_external_initializers(model, external_data_dir: str) -> int:
    """Read leftover external initializer bytes into this component proto.

    The weight store only externalizes tensors that already have raw bytes.
    Norms and RoPE caches stay pointed at the master's ``model.onnx.data``.
    That relative name does not exist beside the saved component. This reads
    those slices and clears the external location. The weight store then
    writes anything at least as large as its threshold into the component
    data file. The master file is only read.
    """
    if not external_data_dir:
        return 0
    count = 0
    for initializer in model.graph.initializer:
        if initializer.data_location != TensorProto.EXTERNAL:
            continue
        if initializer.raw_data:
            continue
        meta = {entry.key: entry.value for entry in initializer.external_data}
        location = meta.get("location", "")
        parts = location.replace("\\", "/").split("/")
        if (
            not location
            or os.path.isabs(location)
            or any(part in ("", ".", "..") for part in parts)
        ):
            raise ValueError(
                f"initializer {initializer.name} has an unsafe external location {location!r}"
            )
        path = os.path.join(external_data_dir, *parts)
        offset = int(meta.get("offset") or "0")
        with open(path, "rb") as handle:
            handle.seek(offset)
            if meta.get("length"):
                length = int(meta["length"])
                raw = handle.read(length)
            else:
                raw = handle.read()
                length = len(raw)
        if len(raw) != length:
            raise ValueError(
                f"initializer {initializer.name} external read was {len(raw)} bytes, "
                f"expected {length}"
            )
        initializer.raw_data = raw
        initializer.data_location = TensorProto.DEFAULT
        initializer.ClearField("external_data")
        count += 1
    return count


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
    static_shapes: Optional[bool] = None,
    static_max_seq_len: Optional[int] = None,
    static_query_len: Optional[int] = None,
    npu_bf16_fallback: bool = False,
    split_lm_head: bool = False,
    lm_head_chunk_rows: Optional[int] = None,
    only_components: Optional[Sequence[str]] = None,
    load_external_data: bool = True,
    quark: bool = False,
    quark_config: Optional[Any] = None,
    decompose_gqa: bool = False,
    sink_size: int = 0,
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
        npu_bf16_fallback: Bake each saved component's 4-bit MatMulNBits into
            bfloat16 MatMuls. Default False. An NPU target does not turn this
            on. The loaded master graph is not baked. Ryzen AI 1.8.0 aborts
            in superkernel_GemmBf16Int4
            (https://github.com/amd/RyzenAI-SW/issues/407).
        quark: When True, run AMD Quark before the model is loaded. Default
            False. A missing Quark install raises before any component file
            is written. Rewrite stays with the caller.
        quark_config: Optional QuarkOptimizeConfig. Ignored when quark is False.
        split_lm_head: Pad the LM-head vocabulary into equal tiles and slice
            the pad off. Default False. Only the lm_head component is
            rewritten. Embeddings keep the short tail. When this flag and
            npu_bf16_fallback are both set, that component is baked as it is
            tiled. The master graph stays packed.
        lm_head_chunk_rows: Tile width for split_lm_head. The default is the
            first SRAM tile from the vocabulary planner.
        only_components: Component keys or filenames to write. The default
            writes every component. Names that are not in the model raise.
        load_external_data: When False, the master graph is loaded with its
            weight bytes left on disk. Baking a component reads those bytes
            itself. A hardlinked ``model.onnx.data`` cannot be loaded by
            ``onnx.load`` while the bytes stay shared.
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

    # Rewrite is already done by the caller. Quark runs only when asked,
    # and it runs before this function loads or bakes the graph.
    config.model = prepare_model_for_split(
        config.model,
        config.output,
        quark=quark,
        config=quark_config,
    )
    model_path = config.model
    if load_external_data:
        onnx_model = load_onnx_model(model_path)
    else:
        onnx_model = load_onnx_model(model_path, load_external_data=False)
    # Attention Where/MaskedFill lowers to a scalar -1e4 add. Shape inference
    # then sees the Cast/Sub/Mul/Add nodes. A graph with no such fill is unchanged.
    replaced_masks = rewrite_attention_mask_ops(onnx_model)
    if replaced_masks:
        print(
            f"Rewrote {replaced_masks} attention mask Where/MaskedFill ops "
            "into an additive -1e4 bias"
        )

    use_static = static_shapes
    if use_static is None:
        use_static = config.target_device == DeviceType.NPU
    max_seq = int(static_max_seq_len or config.kv_cache.max_seq_len or config.context)
    query = int(static_query_len or max_seq)
    if use_static and not npu_bf16_fallback:
        # Flag false keeps today's master contract, including the bfloat16
        # activation cast while MatMulNBits remain. The fallback flag must
        # not take this branch: that cast is not the baked graph.
        apply_static_contract(
            onnx_model,
            batch_size=int(config.batch_size),
            max_seq_len=max_seq,
            query_len=query,
            npu_bf16_fallback=False,
            decompose_gqa=bool(decompose_gqa),
            sink_size=int(sink_size),
        )
        print(
            f"Froze static shapes at batch {config.batch_size}, "
            f"query {query}, max_seq {max_seq}, "
            f"bf16_fallback False"
        )
    elif npu_bf16_fallback:
        packed = sum(
            node.op_type == "MatMulNBits" and node.domain == "com.microsoft"
            for node in onnx_model.graph.node
        )
        print(
            "npu_bf16_fallback stays off the master graph. "
            f"Master MatMulNBits remaining: {packed}. "
            "Each saved component is contracted and baked on its own."
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

    model_dir = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
    component_files, weight_storage, num_layers = save_components(
        onnx_model,
        components,
        boundary_contract,
        output_dir,
        strategy=strategy,
        size_threshold=weight_size_threshold,
        chunk_size=weight_chunk_size,
        split_lm_head=bool(split_lm_head),
        npu_bf16_fallback=bool(npu_bf16_fallback),
        external_data_dir=model_dir,
        lm_head_chunk_rows=lm_head_chunk_rows,
        only_components=only_components,
        apply_component_contract=bool(use_static and npu_bf16_fallback),
        static_batch_size=int(config.batch_size),
        static_max_seq_len=max_seq,
        static_query_len=query,
        decompose_gqa=bool(decompose_gqa),
        sink_size=int(sink_size),
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
