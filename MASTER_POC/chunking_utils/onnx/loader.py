"""
ONNX model loading utilities.

This module provides utilities for loading ONNX models from disk,
with support for external data files and large models.
"""

import os
from typing import Any, List, Optional

import onnx


def resolve_model_path(model_path: str) -> str:
    """
    Resolve the actual path to the ONNX model file.
    
    If a directory is provided, searches for common ONNX model filenames.
    
    Args:
        model_path: Path to ONNX model file or directory.
    
    Returns:
        Resolved path to the ONNX model file.
    
    Raises:
        FileNotFoundError: If no ONNX model found at path.
    
    Example:
        >>> resolve_model_path("./model_dir")
        './model_dir/model.onnx'
    """
    if os.path.isfile(model_path):
        return model_path
    elif os.path.isdir(model_path):
        # Look for common ONNX model filenames
        candidates = [
            "model.onnx",
            "model_optimized.onnx",
            "decoder_model.onnx",
            "encoder_model.onnx",
        ]
        for name in candidates:
            candidate_path = os.path.join(model_path, name)
            if os.path.exists(candidate_path):
                return candidate_path
        raise FileNotFoundError(f"No ONNX model found in directory: {model_path}")
    else:
        raise FileNotFoundError(f"Model path does not exist: {model_path}")


def _positive_index(text: str) -> Optional[int]:
    """Return a shard index. Zero and a leading zero are not GenAI suffixes."""
    if not text.isdigit() or (len(text) > 1 and text.startswith("0")):
        return None
    value = int(text)
    if value <= 0:
        return None
    return value


def find_all_external_data_files(model_file: str) -> List[str]:
    """
    Find every external-data file for one ONNX graph.

    GenAI splits a model past the protobuf limit into ``model.onnx.data``,
    ``model.onnx.data_1``, and ``model.onnx_1.data``. Only files that exist
    are returned, unsuffixed first and then by numeric suffix.
    """
    if isinstance(model_file, bool) or not isinstance(model_file, str) or not model_file.strip():
        raise ValueError(f"model_file must be a non-empty string, got {model_file!r}")
    model_file = os.path.normpath(model_file.strip())
    if os.path.isdir(model_file):
        raise ValueError(f"model_file must be a file, got {model_file}")
    directory = os.path.dirname(model_file) or "."
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"external data directory does not exist: {directory}")
    name = os.path.basename(model_file)
    primary = None
    data_shards = []
    onnx_shards = []
    try:
        entries = os.listdir(directory)
    except OSError as exc:
        raise FileNotFoundError(
            f"external data directory is not readable: {directory}"
        ) from exc
    data_prefix = name + ".data_"
    onnx_prefix = name + "_"
    for entry in entries:
        path = os.path.normpath(os.path.join(directory, entry))
        if not os.path.isfile(path):
            continue
        if entry == name + ".data":
            primary = path
            continue
        if entry.startswith(data_prefix):
            index = _positive_index(entry[len(data_prefix):])
            if index is not None:
                data_shards.append((index, path))
            continue
        if entry.startswith(onnx_prefix) and entry.endswith(".data"):
            middle = entry[len(onnx_prefix):-len(".data")]
            index = _positive_index(middle)
            if index is not None:
                onnx_shards.append((index, path))
    data_shards.sort(key=lambda item: (item[0], item[1]))
    onnx_shards.sort(key=lambda item: (item[0], item[1]))
    found: List[str] = []
    if primary is not None:
        found.append(primary)
    found.extend(path for _, path in data_shards)
    found.extend(path for _, path in onnx_shards)
    return found


def _graph_for_data_file(path: str) -> Optional[str]:
    """Return the graph path a GenAI data filename belongs to."""
    directory = os.path.dirname(path)
    name = os.path.basename(path)
    marker = ".data_"
    position = name.rfind(marker)
    if position > 0 and _positive_index(name[position + len(marker):]) is not None:
        return os.path.join(directory, name[:position])
    if not name.endswith(".data"):
        return None
    stem = name[:-len(".data")]
    split = stem.rfind("_")
    if split > 0 and _positive_index(stem[split + 1:]) is not None:
        return os.path.join(directory, stem[:split])
    return os.path.join(directory, stem)


def _fallback_external_files(fallback: str, model_file: str) -> tuple:
    """
    Return fallback data files and whether missing locations use one master.

    Shards named for this graph are kept distinct. Any other single data
    file is the shared master from the hardlink fallback.
    """
    raw = os.path.normpath(fallback)
    model_name = os.path.basename(model_file)
    if os.path.isfile(raw):
        if not raw.endswith(".data"):
            raise ValueError(
                f"external_data_dir must be a directory or a .data file, got {raw}"
            )
        graph = _graph_for_data_file(raw)
        if graph is not None and os.path.basename(graph) == model_name:
            named = find_all_external_data_files(graph)
            if named:
                return named, False
        return [raw], True
    if not os.path.isdir(raw):
        raise FileNotFoundError(f"external data path does not exist: {raw}")
    named = find_all_external_data_files(os.path.join(raw, model_name))
    if named:
        return named, False
    return [_resolve_master_data(raw)], True


def _coerce_optional_path(value: Optional[str], label: str) -> Optional[str]:
    """Return a stripped path. None stays unset. A blank or bool raises."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string, got {value!r}")
    return value.strip()


def _resolve_master_data(path: str) -> str:
    """
    Return the master external-data file for a fallback path.

    A ``.data`` file is used as itself. A directory prefers
    ``model.onnx.data``, then the only other ``.data`` file it holds.
    """
    raw = os.path.normpath(path)
    if os.path.isfile(raw):
        if not raw.endswith(".data"):
            raise ValueError(
                f"external_data_dir must be a directory or a .data file, got {raw}"
            )
        return raw
    if not os.path.isdir(raw):
        raise FileNotFoundError(f"external data path does not exist: {raw}")
    master = os.path.join(raw, "model.onnx.data")
    if os.path.isfile(master):
        return master
    found = []
    try:
        names = os.listdir(raw)
    except OSError as exc:
        raise FileNotFoundError(f"external data path is not readable: {raw}") from exc
    for name in names:
        if not name.endswith(".data"):
            continue
        candidate = os.path.join(raw, name)
        if os.path.isfile(candidate):
            found.append(candidate)
    found.sort()
    if len(found) == 1:
        return found[0]
    if not found:
        raise FileNotFoundError(f"no external data file in {raw}")
    raise ValueError(f"external_data_dir {raw} contains multiple data files")


def _entry_pair(entry: Any) -> tuple:
    if isinstance(entry, dict):
        return entry.get("key"), entry.get("value")
    return getattr(entry, "key", None), getattr(entry, "value", None)


def _external_entries(tensor: Any) -> list:
    external = getattr(tensor, "external_data", None)
    if not external:
        return []
    return list(external)


def _location_of(tensor: Any) -> Optional[str]:
    for entry in _external_entries(tensor):
        key, value = _entry_pair(entry)
        if key == "location" and isinstance(value, str) and value.strip():
            return value
    return None


def _set_location(tensor: Any, location: str) -> None:
    external = getattr(tensor, "external_data", None)
    if not external:
        raise ValueError("initializer has no external data location")
    for entry in external:
        if isinstance(entry, dict):
            if entry.get("key") == "location":
                entry["value"] = location
                return
        elif getattr(entry, "key", None) == "location":
            try:
                entry.value = location
            except Exception as exc:
                raise ValueError("external data location cannot be updated") from exc
            return
    raise ValueError("external data location is missing")


def _external_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{label} must be a non-negative integer, got {value!r}")
    if isinstance(value, str):
        text = value.strip()
        if not text.isdigit():
            raise ValueError(f"{label} must be a non-negative integer, got {value!r}")
        return int(text)
    if value < 0:
        raise ValueError(f"{label} must be a non-negative integer, got {value!r}")
    return value


def _external_span(tensor: Any) -> tuple:
    offset = 0
    length = None
    for entry in _external_entries(tensor):
        key, value = _entry_pair(entry)
        if key == "offset":
            offset = _external_int(value, "offset")
        elif key == "length":
            length = _external_int(value, "length")
    return offset, length


def _resolve_external_file(model_file: str, location: str) -> Optional[str]:
    if os.path.isabs(location):
        return location if os.path.isfile(location) else None
    beside = os.path.normpath(os.path.join(os.path.dirname(model_file), location))
    if os.path.isfile(beside):
        return beside
    return None


def _location_for_master(model_file: str, data_file: str) -> str:
    model_dir = os.path.dirname(os.path.normpath(model_file))
    data_file = os.path.normpath(data_file)
    if os.path.dirname(data_file) == model_dir:
        return os.path.basename(data_file)
    return os.path.abspath(data_file)


def _iter_external_tensors(model: onnx.ModelProto):
    graph = getattr(model, "graph", None)
    if graph is None:
        return
    yield from _graph_external_tensors(graph)


def _graph_external_tensors(graph: Any):
    for name in ("initializer", "sparse_initializer"):
        for tensor in getattr(graph, name, []) or []:
            if _location_of(tensor):
                yield tensor
    for node in getattr(graph, "node", []) or []:
        for attr in getattr(node, "attribute", []) or []:
            tensor = getattr(attr, "t", None)
            if tensor is not None and _location_of(tensor):
                yield tensor
            for item in getattr(attr, "tensors", []) or []:
                if _location_of(item):
                    yield item
            subgraph = getattr(attr, "g", None)
            if subgraph is not None:
                yield from _graph_external_tensors(subgraph)


def _read_span(path: str, offset: int, length: Optional[int]) -> bytes:
    try:
        with open(path, "rb") as handle:
            handle.seek(offset)
            payload = handle.read() if length is None else handle.read(length)
    except OSError as exc:
        raise FileNotFoundError(f"external data file is not readable: {path}") from exc
    if length is not None and len(payload) != length:
        raise ValueError(
            f"external data file {path} does not contain {length} bytes at offset {offset}"
        )
    return payload


def _store_raw(tensor: Any, payload: bytes) -> None:
    try:
        tensor.raw_data = payload
    except Exception as exc:
        raise ValueError("initializer does not accept raw_data") from exc


def _load_known_external_data(
    model: onnx.ModelProto,
    model_file: str,
    data_files: List[str],
) -> None:
    """Read each external tensor from the shard named by its location."""
    known = {}
    for path in data_files:
        known[os.path.basename(path)] = path
        known[os.path.normpath(path)] = path
        known[os.path.abspath(path)] = path
    for tensor in _iter_external_tensors(model):
        location = _location_of(tensor)
        if location is None:
            continue
        resolved = _resolve_external_file(model_file, location)
        if resolved is None:
            resolved = known.get(os.path.basename(location))
            if resolved is None:
                resolved = known.get(os.path.normpath(location))
        if resolved is None:
            raise FileNotFoundError(
                f"external data location {location!r} is not available for {model_file}"
            )
        offset, length = _external_span(tensor)
        _store_raw(tensor, _read_span(resolved, offset, length))


# Ops whose output shape is the values of a small tensor, not the graph structure.
_VALUE_SHAPE_INPUT = {
    "ConstantOfShape": 0,
    "Reshape": 1,
    "Expand": 1,
    "Tile": 1,
}


def _mark_lazy_loaded(model: Any, lazy: bool) -> None:
    """Record whether this load skipped weight bytes. A bool is stored as itself."""
    try:
        setattr(model, "_chunking_lazy_loaded", True if lazy else False)
    except Exception:
        return


def _is_lazy_loaded(model: Any) -> bool:
    """True only when this load took the size-threshold lazy path."""
    return getattr(model, "_chunking_lazy_loaded", False) is True


def _named_items(items: Any) -> dict:
    found = {}
    for item in items or []:
        name = getattr(item, "name", None)
        if isinstance(name, str) and name and name not in found:
            found[name] = item
    return found


def _has_external_location(tensor: Any) -> bool:
    for entry in getattr(tensor, "external_data", None) or []:
        if isinstance(entry, dict):
            key = entry.get("key")
            value = entry.get("value")
        else:
            key = getattr(entry, "key", None)
            value = getattr(entry, "value", None)
        if key == "location" and isinstance(value, str) and value.strip():
            return True
    return False


def _initializer_has_values(tensor: Any) -> bool:
    raw = getattr(tensor, "raw_data", b"")
    if isinstance(raw, (bytes, bytearray)) and len(raw) > 0:
        return True
    for field in (
        "float_data",
        "int32_data",
        "int64_data",
        "double_data",
        "uint64_data",
        "string_data",
    ):
        payload = getattr(tensor, field, None)
        if payload:
            return True
    return False


def _output_shape_unresolved(info: Any) -> bool:
    """A missing shape is unresolved. A scalar or one concrete dim is resolved."""
    if info is None:
        return True
    type_proto = getattr(info, "type", None)
    tensor_type = getattr(type_proto, "tensor_type", None) if type_proto is not None else None
    if tensor_type is None:
        return True
    shape = getattr(tensor_type, "shape", None)
    if shape is None:
        return True
    dims = getattr(shape, "dim", None)
    if dims is None:
        return True
    dims = list(dims)
    if not dims:
        return False
    for dim in dims:
        value = getattr(dim, "dim_value", 0)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if value > 0:
            return False
    return True


def _lazy_value_shapes_unresolved(model: Any) -> bool:
    """True when a value-dependent op still has no concrete output dim."""
    graph = getattr(model, "graph", None)
    if graph is None:
        return False
    initializers = _named_items(getattr(graph, "initializer", None))
    outputs = {}
    outputs.update(_named_items(getattr(graph, "value_info", None)))
    outputs.update(_named_items(getattr(graph, "output", None)))
    for node in getattr(graph, "node", None) or []:
        index = _VALUE_SHAPE_INPUT.get(getattr(node, "op_type", None))
        if index is None:
            continue
        inputs = list(getattr(node, "input", None) or [])
        if index >= len(inputs):
            continue
        tensor = initializers.get(inputs[index])
        if tensor is None:
            continue
        if not _has_external_location(tensor) or _initializer_has_values(tensor):
            continue
        node_outputs = list(getattr(node, "output", None) or [])
        if not node_outputs:
            return True
        for name in node_outputs:
            if _output_shape_unresolved(outputs.get(name)):
                return True
    return False


def infer_shapes_with_lazy_fallback(model: Any, infer_shapes=None) -> Any:
    """
    Run shape inference and keep the graph when a lazy load has no weights.

    Reshape and ConstantOfShape read the values of a shape tensor. Lazy loading
    leaves those bytes on disk, so inference can raise or leave the output
    dynamic. A failure returns the original model. A dynamic value-dependent
    output returns the inferred model and warns that static NPU buffers may
    not be allocated.
    """
    if model is None or isinstance(model, (bool, str, int, float, bytes, bytearray)):
        raise ValueError(f"model must be an ONNX model, got {model!r}")
    if infer_shapes is None:
        shape_inference = getattr(onnx, "shape_inference", None)
        infer_shapes = getattr(shape_inference, "infer_shapes", None)
        if infer_shapes is None or not callable(infer_shapes):
            raise ValueError("shape inference is not available")
    elif isinstance(infer_shapes, bool) or not callable(infer_shapes):
        raise ValueError(f"infer_shapes must be callable, got {infer_shapes!r}")

    lazy = _is_lazy_loaded(model)
    try:
        inferred = infer_shapes(model)
    except Exception as exc:
        if lazy:
            print(
                "  Warning: Shape inference failed on a lazy-loaded model: "
                f"{exc}. Weights were not loaded, so Reshape and ConstantOfShape "
                "shapes stay unresolved. Falling back to the graph without inferred "
                "shapes. Static NPU buffers may not be allocated."
            )
        else:
            print(
                f"  Warning: Shape inference failed: {exc}. Subgraphs may have unknown shapes."
            )
        return model
    if inferred is None:
        if lazy:
            print(
                "  Warning: Shape inference failed on a lazy-loaded model: "
                "no model returned. Weights were not loaded, so Reshape and "
                "ConstantOfShape shapes stay unresolved. Falling back to the graph "
                "without inferred shapes. Static NPU buffers may not be allocated."
            )
        else:
            print(
                "  Warning: Shape inference failed: no model returned. "
                "Subgraphs may have unknown shapes."
            )
        return model
    if lazy and inferred is not model:
        _mark_lazy_loaded(inferred, True)
    if lazy and _lazy_value_shapes_unresolved(inferred):
        print(
            "  Warning: Shape inference left dynamic shapes on a lazy-loaded model. "
            "Reshape and ConstantOfShape need the weight values that lazy loading "
            "left on disk. Static NPU buffers may not be allocated."
        )
        return inferred
    print("  Shape inference successful.")
    return inferred


def _map_external_data(model: onnx.ModelProto, model_file: str, data_file: str) -> None:
    """Point missing external locations at the master data file and read them."""
    stored = _location_for_master(model_file, data_file)
    for tensor in _iter_external_tensors(model):
        location = _location_of(tensor)
        if location is None:
            continue
        if _resolve_external_file(model_file, location) is None:
            _set_location(tensor, stored)
    for tensor in _iter_external_tensors(model):
        location = _location_of(tensor)
        if location is None:
            continue
        resolved = _resolve_external_file(model_file, location)
        if resolved is None:
            raise FileNotFoundError(
                f"external data location {location!r} is not available for {model_file}"
            )
        offset, length = _external_span(tensor)
        _store_raw(tensor, _read_span(resolved, offset, length))


def load_onnx_model(
    model_path: str,
    external_data_dir: Optional[str] = None,
    load_external_data: bool = True,
    lazy_load_threshold_mb: float = 10000,
) -> onnx.ModelProto:
    """
    Load an ONNX model from disk.
    
    Supports:
    - Single ONNX files
    - ONNX files with external data (.onnx.data files)
    - GenAI shards ``model.onnx.data``, ``model.onnx.data_1``, and ``model.onnx_1.data``
    - A shared master data file when the component sibling is absent
    - Lazy loading for very large models (>10GB by default). A lazy model is
      marked so shape inference can warn when Reshape or ConstantOfShape
      still need those weight values.
    
    Args:
        model_path: Path to ONNX model file or directory.
        external_data_dir: Fallback to the master .data file. A directory
            or a path to that file. Used when ``<model>.onnx.data`` is absent.
        load_external_data: Whether to load external data (weights).
        lazy_load_threshold_mb: If external data exceeds this size in MB,
            use lazy loading (graph structure only).
    
    Returns:
        Loaded ONNX ModelProto.
    
    Example:
        >>> model = load_onnx_model("./model.onnx")
        >>> print(f"Nodes: {len(model.graph.node)}")
    """
    fallback = _coerce_optional_path(external_data_dir, "external_data_dir")
    model_file = resolve_model_path(model_path)
    print(f"Loading ONNX model from: {model_file}")
    
    # Numbered GenAI shards count with the unsuffixed file. A hardlinked
    # master file is the fallback when this graph has no shards of its own.
    data_files = find_all_external_data_files(model_file)
    retarget = False
    lazy = False
    if not data_files and fallback is not None:
        data_files, retarget = _fallback_external_files(fallback, model_file)
    
    if data_files:
        total_mb = sum(os.path.getsize(path) for path in data_files) / (1024 * 1024)
        for path in data_files:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  External data file detected: {os.path.basename(path)} ({size_mb:.1f} MB)")
        sibling = os.path.normpath(model_file + ".data")
        single_sibling = (
            len(data_files) == 1 and os.path.normpath(data_files[0]) == sibling
        )
        
        # For very large models, use lazy loading to avoid OOM
        if total_mb > lazy_load_threshold_mb:
            print(f"  Using LAZY loading (graph structure only) for large model")
            lazy = True
            model = onnx.load(model_file, load_external_data=False)
        elif not load_external_data:
            model = onnx.load(model_file, load_external_data=False)
        elif retarget:
            model = onnx.load(model_file, load_external_data=False)
            _map_external_data(model, model_file, data_files[0])
        elif single_sibling:
            model = onnx.load(model_file, load_external_data=load_external_data)
        else:
            model = onnx.load(model_file, load_external_data=False)
            _load_known_external_data(model, model_file, data_files)
    else:
        model = onnx.load(model_file, load_external_data=load_external_data)
    
    print(f"  IR version: {model.ir_version}")
    print(f"  Producer: {model.producer_name}")
    print(f"  Nodes: {len(model.graph.node)}")
    print(f"  Initializers: {len(model.graph.initializer)}")
    _mark_lazy_loaded(model, lazy)
    
    return model


def get_model_info(model: onnx.ModelProto) -> dict:
    """
    Get summary information about an ONNX model.
    
    Args:
        model: ONNX ModelProto.
    
    Returns:
        Dictionary with model information.
    """
    graph = model.graph
    
    # Count node types
    node_types = {}
    for node in graph.node:
        op_type = node.op_type
        node_types[op_type] = node_types.get(op_type, 0) + 1
    
    # Get input/output info
    inputs = [
        {"name": inp.name, "shape": [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]}
        for inp in graph.input
        if inp.name not in {init.name for init in graph.initializer}
    ]
    outputs = [
        {"name": out.name}
        for out in graph.output
    ]
    
    return {
        "ir_version": model.ir_version,
        "producer": model.producer_name,
        "num_nodes": len(graph.node),
        "num_initializers": len(graph.initializer),
        "node_types": node_types,
        "inputs": inputs,
        "outputs": outputs,
    }


# Short names used by the AMD stack. RyzenAI ships as the VitisAI provider.
_PROVIDER_ALIASES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "dml": "DmlExecutionProvider",
    "directml": "DmlExecutionProvider",
    "vitisai": "VitisAIExecutionProvider",
    "ryzenai": "VitisAIExecutionProvider",
    "azure": "AzureExecutionProvider",
}


def _normalize_provider(name: Any) -> str:
    """Return the ONNX Runtime provider id. A bool or a blank name raises."""
    if isinstance(name, bool) or not isinstance(name, str) or not name.strip():
        raise ValueError(f"provider must be a non-empty string, got {name!r}")
    text = name.strip()
    alias = _PROVIDER_ALIASES.get(text.lower())
    if alias is not None:
        return alias
    if text.endswith("ExecutionProvider"):
        return text
    raise ValueError(f"provider must be an ONNX Runtime provider, got {name!r}")


def _coerce_providers(providers: Any) -> Optional[List[str]]:
    """None keeps the installed providers. A string or a bool raises."""
    if providers is None:
        return None
    if isinstance(providers, (bool, str)) or not isinstance(providers, (list, tuple)):
        raise ValueError(f"providers must be a list of provider names, got {providers!r}")
    if len(providers) == 0:
        raise ValueError("providers must name at least one provider")
    return [_normalize_provider(name) for name in providers]


def _validate_lazy_structure(model: Any) -> Optional[str]:
    """Check the graph without reading weight bytes. A problem is a message."""
    graph = getattr(model, "graph", None)
    if graph is None:
        return "model has no graph"
    for kind in ("input", "output"):
        for item in getattr(graph, kind, None) or []:
            name = getattr(item, "name", None)
            if isinstance(name, bool) or not isinstance(name, str) or not name.strip():
                return f"graph {kind} has a blank name"
    for node in getattr(graph, "node", None) or []:
        op_type = getattr(node, "op_type", None)
        if isinstance(op_type, bool) or not isinstance(op_type, str) or not op_type.strip():
            return "a node has a blank op_type"
    tensors = list(getattr(graph, "initializer", None) or [])
    tensors.extend(getattr(graph, "sparse_initializer", None) or [])
    for tensor in tensors:
        external = getattr(tensor, "external_data", None)
        if external and not _has_external_location(tensor):
            tensor_name = getattr(tensor, "name", "")
            return f"initializer {tensor_name} external data has no location"
    return None


def _installed_runtime() -> Optional[tuple]:
    """Return installed providers and (domain, op, provider) kernels."""
    try:
        import onnxruntime as ort
        from onnxruntime.capi import onnxruntime_pybind11_state as state
    except ImportError:
        return None
    available = getattr(ort, "get_available_providers", None)
    kernels_of = getattr(state, "get_all_opkernel_def", None)
    if not callable(available) or not callable(kernels_of):
        return None
    try:
        providers = [provider for provider in available() if isinstance(provider, str) and provider]
        kernels = []
        for kernel in kernels_of():
            op_name = getattr(kernel, "op_name", None)
            provider = getattr(kernel, "provider", None)
            if not isinstance(op_name, str) or not op_name or not isinstance(provider, str) or not provider:
                continue
            domain = getattr(kernel, "domain", "") or ""
            if not isinstance(domain, str):
                domain = ""
            kernels.append((domain, op_name, provider))
    except Exception:
        return None
    return providers, kernels


def _kernel_supports(domain: str, op_type: str, providers: List[str], kernels: List[tuple]) -> bool:
    """True when one selected provider has a kernel for this op."""
    accepted = {domain}
    if domain == "":
        accepted.add("com.microsoft")
    selected = set(providers)
    for kernel_domain, kernel_op, kernel_provider in kernels:
        if kernel_op != op_type or kernel_provider not in selected:
            continue
        if (kernel_domain or "") in accepted:
            return True
    return False


def _unsupported_ops(model: Any, providers: List[str], kernels: List[tuple]) -> List[str]:
    """Name each op that the selected providers cannot run."""
    graph = getattr(model, "graph", None)
    if graph is None:
        return ["model graph"]
    missing = []
    seen = set()
    for node in getattr(graph, "node", None) or []:
        op_type = getattr(node, "op_type", None)
        if not isinstance(op_type, str) or not op_type.strip():
            continue
        op_type = op_type.strip()
        domain = getattr(node, "domain", "") or ""
        if not isinstance(domain, str):
            domain = ""
        domain = domain.strip()
        key = (domain, op_type)
        if key in seen:
            continue
        seen.add(key)
        if _kernel_supports(domain, op_type, providers, kernels):
            continue
        if domain:
            missing.append(f"{op_type} ({domain})")
        else:
            missing.append(op_type)
    return missing


def validate_onnx_model(model_path: str, providers: Optional[List[str]] = None) -> bool:
    """
    Validate an ONNX model and the kernels of the installed ONNX Runtime.

    A lazy-loaded model has no weight bytes, so ``onnx.checker`` is skipped
    and only the graph structure is checked. Every node must have a kernel
    on a selected installed provider. MatMulNBits is the com.microsoft op
    that a VitisAI wheel often does not register.
    """
    if isinstance(model_path, bool) or not isinstance(model_path, str) or not model_path.strip():
        raise ValueError(f"model_path must be a non-empty string, got {model_path!r}")
    selected = _coerce_providers(providers)
    try:
        model = load_onnx_model(model_path)
    except Exception as exc:
        print(f"Model validation failed: {exc}")
        return False
    try:
        if _is_lazy_loaded(model):
            print(
                "  Warning: lazy-loaded model skipped tensor-data checks. "
                "onnx.checker was not run because external weights were not loaded."
            )
            problem = _validate_lazy_structure(model)
            if problem is not None:
                print(f"Model validation failed: {problem}")
                return False
        else:
            checker = getattr(onnx, "checker", None)
            check_model = getattr(checker, "check_model", None) if checker is not None else None
            if check_model is None or not callable(check_model):
                print("Model validation failed: onnx.checker is not available.")
                return False
            check_model(model)
    except Exception as exc:
        print(f"Model validation failed: {exc}")
        return False
    runtime = _installed_runtime()
    if runtime is None:
        print(
            "Model validation failed: onnxruntime is not installed, "
            "so provider kernels were not checked."
        )
        return False
    available, kernels = runtime
    if selected is None:
        selected = list(available)
    if not selected:
        print("Model validation failed: onnxruntime reported no installed provider.")
        return False
    installed = set(available)
    for provider in selected:
        if provider not in installed:
            print(
                f"Model validation failed: {provider} is not an installed ONNX Runtime provider."
            )
            return False
    missing = _unsupported_ops(model, selected, kernels)
    if missing:
        joined = ", ".join(missing)
        provider_list = ", ".join(selected)
        print(
            f"Model validation failed: {joined} is not supported by {provider_list}."
        )
        return False
    print("Model validation successful.")
    return True
