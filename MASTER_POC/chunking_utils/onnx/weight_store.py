"""
External weight storage for an ONNX split.

COPY writes one data file per component, which is the historical saver.
HARDLINK, MANIFEST, and STREAMING keep one copy of each initializer payload
and record the slice each component loads. A shared manifest is written for
those three strategies.
"""

import hashlib
import json
import os
import warnings
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


SIZE_THRESHOLD = 1024 * 1024
STREAMING_CHUNK_SIZE = 8 * 1024 * 1024

_SHARED_FILE = "weights/shared.bin"
_MANIFEST_FILE = "weights/manifest.json"
_STREAM_FILE = "weights/stream.json"
# onnx.TensorProto.DataLocation.EXTERNAL. A location of 0 makes ORT ignore the file.
_EXTERNAL_DATA_LOCATION = 1


class WeightStrategy(str, Enum):
    """
    How split initializers are stored on disk.

    COPY writes a data file beside each component. HARDLINK points those
    files at one shared store. MANIFEST records every slice in one manifest.
    STREAMING packs the shared store into chunk files.
    """

    COPY = "copy"
    HARDLINK = "hardlink"
    MANIFEST = "manifest"
    STREAMING = "streaming"


_STRATEGY_ALIASES = {
    "copy": WeightStrategy.COPY,
    "hardlink": WeightStrategy.HARDLINK,
    "hard_link": WeightStrategy.HARDLINK,
    "manifest": WeightStrategy.MANIFEST,
    "streaming": WeightStrategy.STREAMING,
}


def coerce_weight_strategy(value: Any) -> WeightStrategy:
    """Accept a WeightStrategy or its name. A blank or unknown value raises."""
    if isinstance(value, WeightStrategy):
        return value
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise ValueError(f"weight strategy must be a WeightStrategy, got {value!r}")
    text = value.strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    found = _STRATEGY_ALIASES.get(text)
    if found is None:
        known = ", ".join(item.value for item in WeightStrategy)
        raise ValueError(
            f"Unknown weight strategy {value!r}. Expected one of: {known}"
        )
    return found


def plan_weight_storage(
    components: Any,
    output_dir: str,
    strategy: Any = WeightStrategy.COPY,
    size_threshold: int = SIZE_THRESHOLD,
    chunk_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Plan external weight files without writing them or clearing payloads.

    The returned dict is a fresh copy. A duplicated COPY payload warns.
    """
    _directory_string(output_dir)
    if os.path.isfile(output_dir):
        raise ValueError(f"output directory must be a directory, got {output_dir!r}")
    record, notes, _bindings, resolved_chunk = _build(
        components,
        strategy,
        size_threshold,
        chunk_size,
    )
    _emit(notes, resolved_chunk)
    return _copy_record(record)


def store_component_weights(
    components: Any,
    output_dir: str,
    strategy: Any = WeightStrategy.COPY,
    size_threshold: int = SIZE_THRESHOLD,
    chunk_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Write external weight files and point each large initializer at its slice.

    HARDLINK, MANIFEST, and STREAMING store each payload once. COPY keeps a
    data file per component and warns when the same payload is stored again.
    The returned dict is a fresh copy.
    """
    directory = _ensure_directory(output_dir)
    record, notes, bindings, resolved_chunk = _build(
        components,
        strategy,
        size_threshold,
        chunk_size,
    )
    failures = _write_files(directory, record)
    notes["hardlink_failures"].extend(failures)
    _write_manifest(directory, record)
    _emit(notes, resolved_chunk)
    _apply_bindings(bindings)
    return _copy_record(record)


def _emit(notes: Dict[str, Any], chunk_size: Optional[int]) -> None:
    count = notes["duplicate_payloads"]
    if count == 1:
        warnings.warn(
            "weight strategy copy duplicates 1 initializer payload across components",
            UserWarning,
            stacklevel=3,
        )
    elif count > 1:
        warnings.warn(
            "weight strategy copy duplicates "
            f"{count} initializer payloads across components",
            UserWarning,
            stacklevel=3,
        )
    for name, length in notes["oversized"]:
        warnings.warn(
            f"weight streaming stored {name} in its own chunk because its "
            f"{length} bytes exceed chunk_size {chunk_size}",
            UserWarning,
            stacklevel=3,
        )
    for path in notes["hardlink_failures"]:
        warnings.warn(
            f"weight hardlink failed for {path}; copied the shared store instead",
            UserWarning,
            stacklevel=3,
        )


def _build(
    components: Any,
    strategy: Any,
    size_threshold: Any,
    chunk_size: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Tuple[Any, Dict[str, Any]]], Optional[int]]:
    selected = coerce_weight_strategy(strategy)
    threshold = _non_negative_int(size_threshold, "size_threshold")
    resolved_chunk = _resolve_chunk_size(selected, chunk_size)
    grouped, blobs, order = _collect(components, threshold)
    names: Dict[str, str] = {}
    for component in grouped:
        for tensor in component["tensors"]:
            names.setdefault(tensor["digest"], tensor["name"])
    notes: Dict[str, Any] = {
        "duplicate_payloads": _cross_component_duplicates(grouped) if selected is WeightStrategy.COPY else 0,
        "oversized": [],
        "hardlink_failures": [],
        "_names": names,
    }
    placement, shared_bytes = _shared_placement(order, blobs)
    if selected is WeightStrategy.STREAMING:
        chunk_placement, chunk_files = _pack_chunks(
            order,
            blobs,
            int(resolved_chunk),
            notes,
        )
    else:
        chunk_placement = {}
        chunk_files = []
    record, bindings = _assemble(
        selected,
        threshold,
        resolved_chunk,
        grouped,
        blobs,
        shared_bytes,
        placement,
        chunk_placement,
        chunk_files,
    )
    return record, notes, bindings, resolved_chunk


def _assemble(
    strategy: WeightStrategy,
    threshold: int,
    chunk_size: Optional[int],
    grouped: List[Dict[str, Any]],
    blobs: Dict[str, bytes],
    shared_bytes: bytes,
    placement: Dict[str, int],
    chunk_placement: Dict[str, Tuple[str, int]],
    chunk_files: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Tuple[Any, Dict[str, Any]]]]:
    files: List[Dict[str, Any]] = []
    component_records: List[Dict[str, Any]] = []
    bindings: List[Tuple[Any, Dict[str, Any]]] = []
    shared = strategy in (
        WeightStrategy.HARDLINK,
        WeightStrategy.MANIFEST,
        WeightStrategy.STREAMING,
    )
    shared_file = _SHARED_FILE if shared and shared_bytes and strategy is not WeightStrategy.STREAMING else None
    if strategy is WeightStrategy.COPY:
        for component in grouped:
            data, slices, bound = _copy_component(component, blobs)
            bindings.extend(bound)
            data_file = None
            if data:
                data_file = component["file"] + ".data"
                files.append(_file_record(data_file, data, None))
            component_records.append({
                "name": component["name"],
                "file": component["file"],
                "data_file": data_file,
                "slices": slices,
            })
        manifest = None
    elif strategy is WeightStrategy.STREAMING:
        for item in chunk_files:
            files.append(_file_record(item["path"], item["data"], None))
        for component in grouped:
            slices, bound = _placed_slices(component, chunk_placement, blobs)
            bindings.extend(bound)
            locations = []
            for sliver in slices:
                if sliver["location"] not in locations:
                    locations.append(sliver["location"])
            component_records.append({
                "name": component["name"],
                "file": component["file"],
                "data_file": locations[0] if len(locations) == 1 else None,
                "slices": slices,
            })
        manifest = _STREAM_FILE
    else:
        if shared_bytes:
            files.append(_file_record(_SHARED_FILE, shared_bytes, None))
        for component in grouped:
            slices, bound = _placed_slices(
                component,
                {digest: (_slice_location(strategy, component), offset) for digest, offset in placement.items()},
                blobs,
            )
            bindings.extend(bound)
            data_file = None
            if slices and strategy is WeightStrategy.HARDLINK:
                data_file = component["file"] + ".data"
                files.append(_file_record(data_file, shared_bytes, _SHARED_FILE))
                for sliver in slices:
                    sliver["location"] = data_file
            elif slices:
                data_file = _SHARED_FILE
            component_records.append({
                "name": component["name"],
                "file": component["file"],
                "data_file": data_file,
                "slices": slices,
            })
        manifest = _MANIFEST_FILE
    record = {
        "strategy": strategy.value,
        "size_threshold": threshold,
        "chunk_size": chunk_size if strategy is WeightStrategy.STREAMING else None,
        "deduplicated": strategy is not WeightStrategy.COPY,
        "shared_file": shared_file,
        "manifest": manifest,
        "files": files,
        "components": component_records,
    }
    return record, bindings


def _slice_location(strategy: WeightStrategy, component: Dict[str, Any]) -> str:
    if strategy is WeightStrategy.MANIFEST:
        return _SHARED_FILE
    return component["file"] + ".data"


def _copy_component(
    component: Dict[str, Any],
    blobs: Dict[str, bytes],
) -> Tuple[bytes, List[Dict[str, Any]], List[Tuple[Any, Dict[str, Any]]]]:
    local: Dict[str, int] = {}
    pieces: List[bytes] = []
    offset = 0
    slices: List[Dict[str, Any]] = []
    bindings: List[Tuple[Any, Dict[str, Any]]] = []
    location = component["file"] + ".data"
    for tensor in component["tensors"]:
        digest = tensor["digest"]
        if digest not in local:
            local[digest] = offset
            pieces.append(blobs[digest])
            offset += len(blobs[digest])
        sliver = {
            "name": tensor["name"],
            "location": location,
            "offset": local[digest],
            "length": len(blobs[digest]),
            "digest": digest,
        }
        slices.append(sliver)
        bindings.append((tensor["tensor"], sliver))
    return b"".join(pieces), slices, bindings


def _placed_slices(
    component: Dict[str, Any],
    placement: Dict[str, Tuple[str, int]],
    blobs: Dict[str, bytes],
) -> Tuple[List[Dict[str, Any]], List[Tuple[Any, Dict[str, Any]]]]:
    slices: List[Dict[str, Any]] = []
    bindings: List[Tuple[Any, Dict[str, Any]]] = []
    for tensor in component["tensors"]:
        location, offset = placement[tensor["digest"]]
        sliver = {
            "name": tensor["name"],
            "location": location,
            "offset": offset,
            "length": len(blobs[tensor["digest"]]),
            "digest": tensor["digest"],
        }
        slices.append(sliver)
        bindings.append((tensor["tensor"], sliver))
    return slices, bindings


def _file_record(path: str, data: bytes, hardlink_of: Optional[str]) -> Dict[str, Any]:
    return {
        "path": path,
        "bytes": len(data),
        "digest": hashlib.sha256(data).hexdigest(),
        "hardlink_of": hardlink_of,
        "_data": data,
    }


def _shared_placement(
    order: List[str],
    blobs: Dict[str, bytes],
) -> Tuple[Dict[str, int], bytes]:
    placement: Dict[str, int] = {}
    pieces: List[bytes] = []
    offset = 0
    for digest in order:
        placement[digest] = offset
        pieces.append(blobs[digest])
        offset += len(blobs[digest])
    return placement, b"".join(pieces)


def _pack_chunks(
    order: List[str],
    blobs: Dict[str, bytes],
    chunk_size: int,
    notes: Dict[str, Any],
) -> Tuple[Dict[str, Tuple[str, int]], List[Dict[str, Any]]]:
    chunks: List[Dict[str, Any]] = []
    names = notes.get("_names", {})
    for digest in order:
        payload = blobs[digest]
        length = len(payload)
        if length > chunk_size:
            notes["oversized"].append((names.get(digest, digest), length))
            chunks.append({
                "digests": [digest],
                "pieces": [payload],
                "size": length,
                "oversized": True,
            })
            continue
        if not chunks or chunks[-1]["size"] + length > chunk_size or chunks[-1].get("oversized"):
            chunks.append({"digests": [], "pieces": [], "size": 0, "oversized": False})
        chunks[-1]["digests"].append(digest)
        chunks[-1]["pieces"].append(payload)
        chunks[-1]["size"] += length
    placement: Dict[str, Tuple[str, int]] = {}
    files: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        path = f"weights/chunks/chunk_{index:06d}.bin"
        offset = 0
        for digest, piece in zip(chunk["digests"], chunk["pieces"]):
            placement[digest] = (path, offset)
            offset += len(piece)
        files.append({"path": path, "data": b"".join(chunk["pieces"])})
    return placement, files


def _collect(
    components: Any,
    threshold: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, bytes], List[str]]:
    grouped: List[Dict[str, Any]] = []
    blobs: Dict[str, bytes] = {}
    order: List[str] = []
    seen_components = set()
    for component in _component_items(components):
        name, filename, tensors = _component_tensors(component)
        if name in seen_components:
            raise ValueError(f"duplicate component {name}")
        seen_components.add(name)
        kept = []
        seen_names = set()
        for tensor in tensors:
            tensor_name = _tensor_name(tensor)
            if tensor_name in seen_names:
                raise ValueError(f"duplicate initializer {tensor_name} in {name}")
            seen_names.add(tensor_name)
            payload = _payload(tensor)
            if payload is None or len(payload) < threshold:
                continue
            digest = hashlib.sha256(payload).hexdigest()
            if digest not in blobs:
                blobs[digest] = payload
                order.append(digest)
            kept.append({
                "name": tensor_name,
                "digest": digest,
                "tensor": tensor,
            })
        grouped.append({"name": name, "file": filename, "tensors": kept})
    return grouped, blobs, order


def _cross_component_duplicates(grouped: List[Dict[str, Any]]) -> int:
    owners: Dict[str, int] = {}
    for component in grouped:
        seen = set()
        for tensor in component["tensors"]:
            digest = tensor["digest"]
            if digest in seen:
                continue
            seen.add(digest)
            owners[digest] = owners.get(digest, 0) + 1
    return sum(1 for count in owners.values() if count > 1)


def _component_items(components: Any) -> List[Any]:
    if isinstance(components, dict):
        if not components:
            raise ValueError("components must contain at least one component")
        items = []
        for key, body in components.items():
            if isinstance(key, bool) or not isinstance(key, str) or not key.strip():
                raise ValueError(f"component name must be a non-empty string, got {key!r}")
            name = key.strip()
            _base_name(name, "component name")
            if isinstance(body, dict):
                item = dict(body)
                inner = item.get("name", name)
                if isinstance(inner, bool) or not isinstance(inner, str) or inner.strip() != name:
                    raise ValueError(f"component name must match {name!r}, got {inner!r}")
                item["name"] = name
                items.append(item)
            else:
                items.append({"name": name, "model": body})
        return items
    if isinstance(components, (str, bytes)) or not isinstance(components, list):
        raise ValueError(f"components must be a dict or a list, got {components!r}")
    if not components:
        raise ValueError("components must contain at least one component")
    return list(components)


def _component_tensors(component: Any) -> Tuple[str, str, List[Any]]:
    if not isinstance(component, dict):
        raise ValueError(f"component must be a dict, got {component!r}")
    name = component.get("name")
    if isinstance(name, bool) or not isinstance(name, str) or not name.strip():
        raise ValueError(f"component name must be a non-empty string, got {name!r}")
    name = name.strip()
    _base_name(name, "component name")
    filename = component.get("file", f"{name}.onnx")
    if filename is None:
        filename = f"{name}.onnx"
    if isinstance(filename, bool) or not isinstance(filename, str) or not filename.strip():
        raise ValueError(f"component file must be a non-empty string, got {filename!r}")
    filename = _base_name(filename.strip(), "component file")
    if "model" in component and component["model"] is not None:
        tensors = _model_initializers(component["model"], name)
    elif "initializers" in component:
        tensors = component["initializers"]
        if isinstance(tensors, (str, bytes)) or not isinstance(tensors, (list, tuple)):
            raise ValueError(f"initializers must be a list, got {tensors!r}")
        tensors = list(tensors)
    else:
        raise ValueError(f"component {name} must include a model or initializers")
    return name, filename, tensors


def _model_initializers(model: Any, name: str) -> List[Any]:
    graph = getattr(model, "graph", None)
    if graph is None:
        raise ValueError(f"component {name} model must have a graph")
    initializers = getattr(graph, "initializer", None)
    if initializers is None:
        return []
    if isinstance(initializers, (str, bytes)):
        raise ValueError(f"initializers must be a list, got {initializers!r}")
    try:
        return list(initializers)
    except TypeError as exc:
        raise ValueError(f"initializers must be a list, got {initializers!r}") from exc


def _tensor_name(tensor: Any) -> str:
    if isinstance(tensor, dict):
        name = tensor.get("name")
    else:
        name = getattr(tensor, "name", None)
    if isinstance(name, bool) or not isinstance(name, str) or not name.strip():
        raise ValueError(f"initializer name must be a non-empty string, got {name!r}")
    return name.strip()


def _payload(tensor: Any) -> Optional[bytes]:
    if isinstance(tensor, dict):
        if "data" in tensor:
            raw = tensor.get("data")
        else:
            raw = tensor.get("raw_data")
    else:
        raw = getattr(tensor, "raw_data", None)
    if raw is None:
        return None
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    if isinstance(raw, bytearray):
        raw = bytes(raw)
    if isinstance(raw, bool) or not isinstance(raw, bytes):
        raise ValueError(f"initializer payload must be bytes, got {raw!r}")
    if len(raw) == 0:
        return None
    return raw


def _base_name(value: str, label: str) -> str:
    if (
        os.path.isabs(value)
        or value in (".", "..")
        or "/" in value
        or "\\" in value
        or ".." in value.split(os.sep)
    ):
        raise ValueError(f"{label} must be a base name, got {value!r}")
    return value


def _resolve_chunk_size(strategy: WeightStrategy, chunk_size: Any) -> Optional[int]:
    if chunk_size is None:
        if strategy is WeightStrategy.STREAMING:
            return STREAMING_CHUNK_SIZE
        return None
    size = _positive_int(chunk_size, "chunk_size")
    if strategy is not WeightStrategy.STREAMING:
        return None
    return size


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return value


def _positive_int(value: Any, name: str) -> int:
    number = _non_negative_int(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return number


def _directory_string(output_dir: Any) -> str:
    if isinstance(output_dir, bool) or not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError(f"output directory must be a non-empty string, got {output_dir!r}")
    return output_dir.strip()


def _ensure_directory(output_dir: Any) -> str:
    directory = _directory_string(output_dir)
    if os.path.isfile(directory):
        raise ValueError(f"output directory must be a directory, got {output_dir!r}")
    os.makedirs(directory, exist_ok=True)
    if not os.path.isdir(directory):
        raise ValueError(f"output directory must be a directory, got {output_dir!r}")
    return directory


def _write_files(directory: str, record: Dict[str, Any]) -> List[str]:
    failures: List[str] = []
    shared_path = None
    if record["shared_file"]:
        shared_path = _absolute(directory, record["shared_file"])
    for item in record["files"]:
        data = item.pop("_data")
        path = _absolute(directory, item["path"])
        link_of = item["hardlink_of"]
        if link_of is None:
            _write_bytes(path, data)
            continue
        source = shared_path if shared_path is not None else _absolute(directory, link_of)
        if _try_link(source, path):
            continue
        _write_bytes(path, data)
        item["hardlink_of"] = None
        failures.append(item["path"])
    return failures


def _write_manifest(directory: str, record: Dict[str, Any]) -> None:
    if record["manifest"] is None:
        return
    tensors = []
    for component in record["components"]:
        for sliver in component["slices"]:
            tensors.append({
                "component": component["name"],
                "name": sliver["name"],
                "location": sliver["location"],
                "offset": sliver["offset"],
                "length": sliver["length"],
                "digest": sliver["digest"],
            })
    if record["strategy"] == WeightStrategy.STREAMING.value:
        chunks = []
        for item in record["files"]:
            held = [
                {"component": tensor["component"], "name": tensor["name"]}
                for tensor in tensors
                if tensor["location"] == item["path"]
            ]
            chunks.append({
                "file": item["path"],
                "bytes": item["bytes"],
                "tensors": held,
            })
        body: Dict[str, Any] = {
            "strategy": record["strategy"],
            "chunk_size": record["chunk_size"],
            "deduplicated": record["deduplicated"],
            "chunks": chunks,
            "tensors": tensors,
        }
    elif record["strategy"] == WeightStrategy.HARDLINK.value:
        body = {
            "strategy": record["strategy"],
            "file": record["shared_file"],
            "deduplicated": record["deduplicated"],
            "links": [
                {"file": item["path"], "hardlink_of": item["hardlink_of"]}
                for item in record["files"]
                if item["hardlink_of"] is not None or item["path"] != record["shared_file"]
            ],
            "tensors": tensors,
        }
    else:
        body = {
            "strategy": record["strategy"],
            "file": record["shared_file"],
            "deduplicated": record["deduplicated"],
            "tensors": tensors,
        }
    _write_bytes(
        _absolute(directory, record["manifest"]),
        (json.dumps(body, indent=2) + "\n").encode("utf-8"),
    )


def _apply_bindings(bindings: List[Tuple[Any, Dict[str, Any]]]) -> None:
    for tensor, sliver in bindings:
        _assign_external(tensor, sliver["location"], sliver["offset"], sliver["length"])


def _assign_external(tensor: Any, location: str, offset: int, length: int) -> None:
    pairs = (
        ("location", location),
        ("offset", str(offset)),
        ("length", str(length)),
    )
    _mark_external_location(tensor)
    if isinstance(tensor, dict):
        tensor["external_data"] = [{"key": key, "value": value} for key, value in pairs]
        if "data" in tensor:
            tensor["data"] = b""
        tensor["raw_data"] = b""
        return
    external = getattr(tensor, "external_data", None)
    if external is not None and hasattr(external, "add"):
        clearer = getattr(external, "clear", None)
        if callable(clearer):
            clearer()
        else:
            del external[:]
        for key, value in pairs:
            entry = external.add()
            entry.key = key
            entry.value = value
    else:
        try:
            tensor.external_data = [{"key": key, "value": value} for key, value in pairs]
        except Exception as exc:
            name = getattr(tensor, "name", "")
            raise ValueError(f"initializer {name} does not accept external_data") from exc
    clear_field = getattr(tensor, "ClearField", None)
    if callable(clear_field):
        try:
            clear_field("raw_data")
            return
        except Exception:
            pass
    if hasattr(tensor, "raw_data"):
        tensor.raw_data = b""


def _mark_external_location(tensor: Any) -> None:
    """Tell ORT the payload lives in the external file, not in raw_data."""
    if isinstance(tensor, dict):
        tensor["data_location"] = _EXTERNAL_DATA_LOCATION
        return
    try:
        tensor.data_location = _EXTERNAL_DATA_LOCATION
    except (AttributeError, TypeError):
        return


def _try_link(source: str, destination: str) -> bool:
    if os.path.isdir(destination):
        raise ValueError(f"weight file must be a file, got {destination}")
    if os.path.lexists(destination):
        os.remove(destination)
    try:
        os.link(source, destination)
    except OSError:
        return False
    return True


def _write_bytes(path: str, data: bytes) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.isdir(path):
        raise ValueError(f"weight file must be a file, got {path}")
    temporary = path + ".tmp"
    with open(temporary, "wb") as handle:
        handle.write(data)
    os.replace(temporary, path)


def _absolute(directory: str, relative: str) -> str:
    return os.path.join(directory, *relative.split("/"))


def _copy_record(record: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = _without_private(record)
    return json.loads(json.dumps(cleaned))


def _without_private(value: Any) -> Any:
    if isinstance(value, list):
        return [_without_private(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _without_private(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    return value
