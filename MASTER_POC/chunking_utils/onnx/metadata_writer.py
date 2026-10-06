"""Write the execution contract that a split stores next to its components."""

import json
import os
from typing import Any, Dict


def write_execution_contract(output_dir, model_info: Dict[str, Any]) -> str:
    """
    Write model_info.json for one split.

    The file is the execution contract: architecture, boundary tensors,
    quantization, KV cache, execution policy, and weight storage. A blank
    directory, a file used as a directory, and a model_info value that is
    not a dict raise. A value JSON cannot encode raises ValueError.

    Args:
        output_dir: Directory that already holds the component files.
        model_info: JSON-ready mapping written as model_info.json.

    Returns:
        Path to the written model_info.json file.
    """
    if isinstance(output_dir, bool) or not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError(f"output directory must be a non-empty string, got {output_dir!r}")
    if os.path.isfile(output_dir):
        raise ValueError(f"output directory must be a directory, got a file {output_dir}")
    if not isinstance(model_info, dict):
        raise ValueError(f"model_info must be a dict, got {model_info!r}")
    info_path = os.path.join(output_dir, "model_info.json")
    if os.path.isdir(info_path):
        raise ValueError(f"model_info.json must be a file, got a directory {info_path}")
    try:
        payload = json.dumps(model_info, indent=2)
    except TypeError as exc:
        raise ValueError("model_info must be JSON-serializable") from exc
    with open(info_path, "w", encoding="utf-8") as handle:
        handle.write(payload)
    return info_path
