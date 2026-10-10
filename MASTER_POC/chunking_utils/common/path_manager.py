"""
Path manager for model discovery, split directories, and NPU caches.

``path_utils`` still answers whether a name is a local path and how to
sanitize it. This module finds the ONNX graph, its external weight file,
a split directory, a model cache folder, the NPU artifact cache,
and the provider compilation cache under ``.npu_cache``.
"""

import json
import os
from typing import List, Optional

from .path_utils import (
    find_external_data_files as _candidate_external_data_files,
    get_npu_cache_dir,
    is_local_path,
    sanitize_model_name,
    validate_split_model,
)

# Re-exported so callers can use one path module.
__all__ = [
    "is_local_path",
    "sanitize_model_name",
    "find_onnx_model",
    "find_external_data",
    "find_external_data_files",
    "validate_split_model",
    "validate_split_model_dir",
    "get_cache_dir",
    "get_npu_cache_dir",
    "resolve_npu_artifact_cache",
]

# Same order as chunking_utils.onnx.loader.resolve_model_path, plus the
# merged graph written by the ONNX inference helper.
_PREFERRED_ONNX = (
    "model.onnx",
    "model_optimized.onnx",
    "decoder_model.onnx",
    "encoder_model.onnx",
    "merged_model.onnx",
)

# Protobuf key used by the ONNX external-data location field.
_EXTERNAL_LOCATION_KEY = b"\x0a\x08location\x12"


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _read_varint(data: bytes, pos: int) -> tuple:
    """Read one protobuf varint. Returns (None, pos) when it does not fit."""
    value = 0
    shift = 0
    while pos < len(data) and shift <= 63:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
    return None, pos


def _external_locations(path: str) -> List[str]:
    """Return ONNX external-data location strings stored in path."""
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError:
        return []
    refs: List[str] = []
    start = 0
    while True:
        index = data.find(_EXTERNAL_LOCATION_KEY, start)
        if index < 0:
            break
        pos = index + len(_EXTERNAL_LOCATION_KEY)
        length, pos = _read_varint(data, pos)
        if length is None or length < 0 or pos + length > len(data):
            start = index + 1
            continue
        try:
            location = data[pos:pos + length].decode("utf-8")
        except UnicodeDecodeError:
            start = pos + length
            continue
        if location and location not in refs:
            refs.append(location)
        start = pos + length
    return refs


def _onnx_files(directory: str) -> List[str]:
    names = []
    try:
        entries = os.listdir(directory)
    except OSError:
        return names
    for name in entries:
        if not name.endswith(".onnx"):
            continue
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            names.append(name)
    names.sort()
    return names


def find_onnx_model(directory: str) -> Optional[str]:
    """
    Return the ONNX graph for a file or a model directory.

    A file that already ends in ``.onnx`` is returned as itself.
    A directory prefers ``model.onnx``, then the other well-known graph
    names, then the first remaining ``.onnx`` file in sorted order.
    A directory with no graph returns None.

    Args:
        directory: ONNX file or directory to search.

    Returns:
        Path to one ONNX file, or None when no graph is present.
    """
    raw = os.path.normpath(_require_text(directory, "directory"))
    if os.path.isfile(raw):
        return raw if raw.endswith(".onnx") else None
    if not os.path.isdir(raw):
        return None
    for name in _PREFERRED_ONNX:
        candidate = os.path.join(raw, name)
        if os.path.isfile(candidate):
            return candidate
    found = _onnx_files(raw)
    if not found:
        return None
    return os.path.join(raw, found[0])


def _resolve_location(model_file: str, location: str) -> Optional[str]:
    if os.path.isabs(location):
        return location if os.path.isfile(location) else None
    beside = os.path.normpath(os.path.join(os.path.dirname(model_file), location))
    if os.path.isfile(beside):
        return beside
    parent = os.path.normpath(os.path.join(os.path.dirname(model_file), os.pardir, location))
    if os.path.isfile(parent):
        return parent
    return None


def find_external_data(model_path: str) -> Optional[str]:
    """
    Return the external weight file for an ONNX model.

    An embedded ``location`` beside the graph wins. Otherwise the
    conventional ``<model>.onnx.data`` sibling is used. A path that is
    already an ``.onnx.data`` file is returned as itself. A model with
    no weight file returns None.

    Args:
        model_path: ONNX file, external-data file, or model directory.

    Returns:
        Path to the external data file, or None.
    """
    raw = os.path.normpath(_require_text(model_path, "model_path"))
    if os.path.isfile(raw) and (raw.endswith(".onnx.data") or raw.endswith(".data")):
        return raw
    if os.path.isfile(raw) and raw.endswith(".onnx"):
        model = raw
    else:
        model = find_onnx_model(raw)
    if model is None or not os.path.isfile(model):
        return None
    for location in _external_locations(model):
        resolved = _resolve_location(model, location)
        if resolved is not None:
            return resolved
    sibling = model + ".data"
    if os.path.isfile(sibling):
        return os.path.normpath(sibling)
    return None


def find_external_data_files(model_path: str) -> List[str]:
    """
    Find all .data files associated with an ONNX model.

    A directory is resolved to its ONNX graph first. The graph is then
    checked for ``<model>.onnx.data``, ``<base>.data``, and
    ``<base>_data``.

    Args:
        model_path: ONNX file or model directory.

    Returns:
        Existing external data files, in candidate order.
    """
    raw = os.path.normpath(_require_text(model_path, "model_path"))
    target = raw
    if os.path.isdir(raw):
        model = find_onnx_model(raw)
        if model is None:
            return []
        target = model
    return _candidate_external_data_files(target)


def validate_split_model_dir(directory: str) -> bool:
    """
    Return whether directory is a split model with a readable manifest.

    ``model_info.json`` must be a JSON object. A ``components`` map must
    name an existing ``.onnx`` file for every entry. A manifest without
    that map is valid when the directory itself holds at least one
    ``.onnx`` file. A monolithic graph with no manifest is not a split.

    Args:
        directory: Directory to validate.

    Returns:
        True when the directory is a usable split model.
    """
    raw = os.path.normpath(_require_text(directory, "directory"))
    if not os.path.isdir(raw):
        return False
    info_path = os.path.join(raw, "model_info.json")
    if not os.path.isfile(info_path):
        return False
    try:
        with open(info_path, "r", encoding="utf-8") as handle:
            info = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeError):
        return False
    if not isinstance(info, dict):
        return False
    components = info.get("components")
    if components is not None:
        if not isinstance(components, dict) or not components:
            return False
        for name, comp in components.items():
            if not isinstance(name, str) or not name.strip() or not isinstance(comp, dict):
                return False
            filename = comp.get("file")
            if not isinstance(filename, str) or not filename.strip():
                return False
            path = filename if os.path.isabs(filename) else os.path.join(raw, filename)
            if not os.path.isfile(path) or not str(path).endswith(".onnx"):
                return False
        return True
    return bool(_onnx_files(raw))


def get_cache_dir(base_dir: str, model_id: str) -> str:
    """
    Return the cache directory for one model and create it.

    The folder name is ``sanitize_model_name(model_id)`` under ``base_dir``.
    A Hub id keeps the double-dash form. A local path keeps its basename.

    Args:
        base_dir: Root that holds per-model cache folders.
        model_id: Hugging Face id or local model path.

    Returns:
        Path to the model cache directory.
    """
    root = os.path.normpath(_require_text(base_dir, "base_dir"))
    model_id = _require_text(model_id, "model_id")
    folder = sanitize_model_name(model_id)
    if not folder:
        raise ValueError("model_id must contain a usable name")
    path = os.path.join(root, folder)
    os.makedirs(path, exist_ok=True)
    return path


def resolve_npu_artifact_cache(model_dir: str) -> str:
    """
    Return ``<model_dir>/.ort_cache`` and create the NPU artifact folders.

    An ONNX file uses its parent directory. The cache root matches the
    session manager default. ``xclbin`` and ``vitisai`` are created under
    that root.

    Args:
        model_dir: Model directory or ONNX file.

    Returns:
        Path to the NPU artifact cache root.
    """
    raw = os.path.normpath(_require_text(model_dir, "model_dir"))
    root = os.path.dirname(raw) if os.path.isfile(raw) else raw
    cache = os.path.join(root, ".ort_cache")
    os.makedirs(os.path.join(cache, "xclbin"), exist_ok=True)
    os.makedirs(os.path.join(cache, "vitisai"), exist_ok=True)
    return cache
