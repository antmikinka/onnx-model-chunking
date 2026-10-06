"""
Path utilities for model chunking operations.

This module provides utilities for handling filesystem paths,
including detection of local vs remote paths and sanitization
of model names for safe filesystem usage.

Model discovery, split directories, and the NPU artifact cache live
in path_manager.py. This module names the external weight files that
sit beside a graph.
"""

import os
import re
from typing import List


def is_local_path(model_id: str) -> bool:
    """
    Check if the model_id is a local filesystem path.
    
    Args:
        model_id: Either a HuggingFace model ID (e.g., "meta-llama/Llama-2-7b")
                  or a local filesystem path.
    
    Returns:
        True if model_id appears to be a local path, False otherwise.
    
    Examples:
        >>> is_local_path("meta-llama/Llama-2-7b")
        False
        >>> is_local_path("C:\\Users\\model")
        True
        >>> is_local_path("/home/user/model")
        True
    """
    return os.path.isdir(model_id) or os.path.isabs(model_id) or os.path.exists(model_id)


def sanitize_model_name(model_id: str) -> str:
    """
    Convert a model ID or path to a filesystem-safe folder name.
    
    For local paths, uses the folder basename to keep output names clean.
    For Hub IDs, replaces slash with double dash.
    
    Args:
        model_id: Either a HuggingFace model ID or local filesystem path.
    
    Returns:
        A sanitized string safe for use as a directory name.
    
    Examples:
        >>> sanitize_model_name("meta-llama/Llama-2-7b")
        'meta-llama--Llama-2-7b'
        >>> sanitize_model_name("C:\\Users\\model\\my-model")
        'my-model'
    """
    # For local paths, use just the folder basename
    if is_local_path(model_id):
        # Normalize path and get the base folder name
        normalized = os.path.normpath(model_id)
        safe_name = os.path.basename(normalized)
    else:
        # For Hub IDs like "meta-llama/Llama-2-7b"
        safe_name = model_id.replace("/", "--")
    
    # Remove illegal Windows characters: < > : " | ? * \
    safe_name = re.sub(r'[<>:"|?*\\]', '', safe_name)
    
    # Strip leading/trailing whitespace
    return safe_name.strip()


def find_external_data_files(model_path: str) -> List[str]:
    """
    Find all .data files associated with an ONNX model.

    The conventional sibling is ``<model>.onnx.data``. Exporters also
    write ``<base>.data`` and ``<base>_data``, where ``<base>`` is the
    model path with its last extension removed. Only files that exist
    are returned, in that order.

    Args:
        model_path: ONNX graph path used to build the candidate names.

    Returns:
        Existing external data files. Empty when none of them exist.
    """
    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError("model_path must be a non-empty string")
    model_path = os.path.normpath(model_path.strip())
    base = os.path.splitext(model_path)[0]
    candidates = [
        model_path + ".data",
        base + ".data",
        base + "_data",
    ]
    found: List[str] = []
    seen = set()
    for candidate in candidates:
        if candidate in seen or not os.path.isfile(candidate):
            continue
        seen.add(candidate)
        found.append(candidate)
    return found
