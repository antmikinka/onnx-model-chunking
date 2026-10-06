"""
HuggingFace and split-ONNX configuration for model chunking.

AutoConfig still detects a Hugging Face architecture. A split directory is
read from model_info.json, the ONNX graphs, boundary tensors, and external
data files.
"""

import json
import os
import re
import warnings
from typing import Optional, Dict, Any, Iterable, List, Tuple

from .types import TensorDtype, MemoryMode
from .architecture import architecture_from_string, ArchitectureType

# Optional imports. ONNX Runtime is the execution target, so this module
# loads when PyTorch is absent.
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    from transformers import AutoConfig, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from accelerate import infer_auto_device_map
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False


# Dtype mapping between our enum and torch/ONNX types.
# INT4 has no native torch.dtype. AUTO preserves the checkpoint dtype.
if HAS_TORCH:
    TORCH_DTYPE_MAP: Dict[TensorDtype, Optional[Any]] = {
        TensorDtype.FLOAT32: torch.float32,
        TensorDtype.FLOAT16: torch.float16,
        TensorDtype.BFLOAT16: torch.bfloat16,
        TensorDtype.INT8: torch.int8,
        TensorDtype.INT4: None,  # Requires special quantization
        TensorDtype.AUTO: None,  # Preserve original
    }
else:
    TORCH_DTYPE_MAP = {item: None for item in TensorDtype}

# ONNX Runtime type strings. AUTO is omitted and falls back to float16.
ONNX_DTYPE_MAP: Dict[TensorDtype, str] = {
    TensorDtype.FLOAT32: "tensor(float)",
    TensorDtype.FLOAT16: "tensor(float16)",
    TensorDtype.BFLOAT16: "tensor(bfloat16)",
    TensorDtype.INT8: "tensor(int8)",
    TensorDtype.INT4: "tensor(int4)",
}


def get_torch_dtype(dtype: TensorDtype) -> Optional[Any]:
    """
    Convert TensorDtype enum to torch.dtype.
    
    Args:
        dtype: TensorDtype enum value.
    
    Returns:
        Corresponding torch.dtype, or None for AUTO and INT4.
    
    Raises:
        ImportError: If PyTorch is not installed.
    """
    if not HAS_TORCH:
        raise ImportError(
            "torch is required to resolve a PyTorch dtype. "
            "Install with: pip install torch"
        )
    return TORCH_DTYPE_MAP.get(dtype, None)


def get_onnx_dtype(dtype: TensorDtype) -> str:
    """
    Convert TensorDtype to an ONNX Runtime type string.

    Args:
        dtype: TensorDtype enum value.

    Returns:
        ONNX type such as tensor(float16). AUTO and unknown values use
        tensor(float16).
    """
    return ONNX_DTYPE_MAP.get(dtype, "tensor(float16)")


def detect_architecture_hf(model_path: str, trust_remote_code: bool = True) -> ArchitectureType:
    """
    Detect model architecture using HuggingFace AutoConfig.
    
    This is the recommended way to detect architecture as it handles
    all edge cases and custom model types.
    
    Args:
        model_path: Path to model or HuggingFace model ID.
        trust_remote_code: Allow custom code from model repos.
    
    Returns:
        Detected ArchitectureType.
    
    Raises:
        ImportError: If transformers not installed.
    """
    if not HAS_TRANSFORMERS:
        raise ImportError(
            "transformers is required for HuggingFace integration. "
            "Install with: pip install transformers"
        )
    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model_type = getattr(config, 'model_type', None)
    
    if model_type:
        return architecture_from_string(model_type)
    
    return ArchitectureType.UNKNOWN


# First positive value wins. max_position_embeddings is the Hugging Face name.
_CONTEXT_LENGTH_KEYS = (
    "max_seq_len",
    "max_position_embeddings",
    "max_sequence_length",
    "seq_length",
    "n_positions",
)
_NESTED_CONFIG_ATTRS = ("text_config", "llm_config", "decoder")


def _config_sources(config: Any) -> Tuple[Any, ...]:
    """Top config, then a nested text or decoder config when the top omits a field."""
    sources = [config]
    seen = {id(config)}
    for name in _NESTED_CONFIG_ATTRS:
        nested = _lookup(config, name)
        if nested is None or id(nested) in seen:
            continue
        if isinstance(nested, (str, int, float, bool, list, tuple)):
            continue
        sources.append(nested)
        seen.add(id(nested))
    return tuple(sources)


def _lookup(source: Any, name: str) -> Any:
    """Return an attribute or dict entry. Missing and null stay None."""
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def _as_config_int(value: Any, name: str) -> int:
    """Accept an integer dimension. Bools and fractional floats are rejected."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{name} must be an integer, got {value!r}")
        value = int(value)
    return int(value)


def _first_positive_int(config: Any, names: Iterable[str]) -> Optional[int]:
    """First positive integer among names. Zero falls through. Negatives raise."""
    for name in names:
        for source in _config_sources(config):
            value = _lookup(source, name)
            if value is None:
                continue
            parsed = _as_config_int(value, name)
            if parsed == 0:
                continue
            if parsed < 0:
                raise ValueError(f"{name} must be non-negative, got {parsed}")
            return parsed
    return None


def _first_bool(config: Any, name: str) -> Optional[bool]:
    """First real boolean. A missing flag stays unset."""
    for source in _config_sources(config):
        if isinstance(source, dict):
            if name not in source or source[name] is None:
                continue
            value = source[name]
        elif not hasattr(source, name):
            continue
        else:
            value = getattr(source, name)
            if value is None:
                continue
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean, got {value!r}")
        return value
    return None


def _first_number(config: Any, name: str, allow_zero: bool = False) -> Optional[float]:
    """First finite number. Zero is unset unless allow_zero is set."""
    for source in _config_sources(config):
        value = _lookup(source, name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number, got {value!r}")
        number = float(value)
        if number == 0 and not allow_zero:
            continue
        if number < 0:
            raise ValueError(f"{name} must be non-negative, got {value!r}")
        return number
    return None


def _first_mapping(config: Any, name: str) -> Optional[Dict[str, Any]]:
    """Copy a nested dict such as rope_scaling. A non-dict raises."""
    for source in _config_sources(config):
        if isinstance(source, dict):
            if name not in source or source[name] is None:
                continue
            value = source[name]
        elif not hasattr(source, name):
            continue
        else:
            value = getattr(source, name)
            if value is None:
                continue
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a dict, got {value!r}")
        return dict(value)
    return None


def _architecture_value(config: Any, name: str, default: Any = None) -> Any:
    """Top-level value, then a nested config, then the default."""
    if isinstance(config, dict):
        if name in config and config[name] is not None:
            return config[name]
    elif hasattr(config, name):
        value = getattr(config, name)
        if value is not None:
            return value
    for source in _config_sources(config)[1:]:
        value = _lookup(source, name)
        if value is not None:
            return value
    return default


def _fields_from_hf_config(config: Any) -> Dict[str, Any]:
    """
    Read architecture, context, KV, RoPE, and MoE fields from a config object.

    A missing top-level field is read from text_config, llm_config, or decoder.
    Zero context lengths fall through to the next alias.
    """
    context = _first_positive_int(config, _CONTEXT_LENGTH_KEYS)
    sliding = _first_positive_int(config, ("sliding_window", "sliding_window_size"))
    sliding_flag = _first_bool(config, "use_sliding_window")
    if sliding_flag is None:
        sliding_flag = sliding is not None
    attention_heads = _first_positive_int(config, ("num_attention_heads",))
    kv_heads = _first_positive_int(
        config, ("num_key_value_heads", "num_kv_heads")
    )
    if kv_heads is None:
        kv_heads = attention_heads
    hidden = _first_positive_int(config, ("hidden_size",))
    head_dim = _first_positive_int(config, ("head_dim", "head_size"))
    if head_dim is None and hidden is not None and attention_heads is not None:
        if hidden % attention_heads != 0:
            warnings.warn(
                f"hidden_size ({hidden}) not divisible by num_attention_heads "
                f"({attention_heads}). Config may be malformed.",
                UserWarning,
            )
        head_dim = hidden // attention_heads
    experts = _first_positive_int(config, ("num_experts", "num_local_experts")) or 0
    per_token = _first_positive_int(
        config, ("num_experts_per_token", "num_experts_per_tok")
    ) or 0
    if experts > 0 and per_token > experts:
        raise ValueError(
            "num_experts_per_token must not exceed num_experts, "
            f"got {per_token} and {experts}"
        )
    moe_flag = _first_bool(config, "is_moe")
    torch_dtype = _architecture_value(config, "torch_dtype", "unknown")
    return {
        "model_type": _architecture_value(config, "model_type", "unknown"),
        "num_hidden_layers": _architecture_value(config, "num_hidden_layers"),
        "hidden_size": _architecture_value(config, "hidden_size"),
        "vocab_size": _architecture_value(config, "vocab_size"),
        "num_attention_heads": _architecture_value(config, "num_attention_heads"),
        "torch_dtype": str(torch_dtype),
        "max_position_embeddings": _first_positive_int(config, ("max_position_embeddings",)),
        "max_sequence_length": _first_positive_int(config, ("max_sequence_length",)),
        "max_seq_len": 0 if context is None else context,
        "sliding_window": sliding,
        "use_sliding_window": sliding_flag,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "rope_theta": _first_number(config, "rope_theta"),
        "rope_scaling": _first_mapping(config, "rope_scaling"),
        "is_moe": experts > 0 if moe_flag is None else moe_flag,
        "num_experts": experts,
        "num_experts_per_token": per_token,
        "moe_intermediate_size": _first_positive_int(config, ("moe_intermediate_size",)),
        "router_aux_loss_coef": _first_number(config, "router_aux_loss_coef", allow_zero=True),
    }


def get_model_config(model_path: str, trust_remote_code: bool = True) -> Dict[str, Any]:
    """
    Get model configuration from a split directory or a HuggingFace config.

    A directory that contains model_info.json or an ONNX graph is read with
    get_split_model_config. Any other path uses AutoConfig. The result
    includes the context length, sliding window, KV heads, head dim, RoPE
    theta, and MoE counts used to preallocate a cache and plan a static NPU
    graph.

    Args:
        model_path: Split directory, local model path, or HuggingFace model ID.
        trust_remote_code: Allow custom code from model repos.

    Returns:
        Dictionary with model configuration.
    """
    if _is_split_model_dir(model_path):
        return get_split_model_config(model_path)
    if not HAS_TRANSFORMERS:
        return {"error": "transformers not installed"}

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    return _fields_from_hf_config(config)


# layer_0.onnx and layer_0_of_32.onnx are split layer graphs.
_LAYER_ONNX_FILE = re.compile(r"^layer_(\d+)(?:_of_\d+)?\.onnx$")
_LAYER_COMPONENT_NAME = re.compile(r"^layer_(\d+)$")
# Protobuf field 1 string "location" followed by field 2, the path.
_EXTERNAL_LOCATION_KEY = b"\x0a\x08location\x12"
_STRUCTURAL_INFO_KEYS = {
    "components",
    "boundary_tensors",
    "component_files",
    "execution_plan",
    "model_info",
    "architecture",
    "output_dir",
    "format",
    "original_model",
    "producer",
    "ir_version",
    "num_components",
}
_ONNX_TYPE_ALIASES = {
    "tensor(float)": "float32",
    "float": "float32",
    "float32": "float32",
    "fp32": "float32",
    "tensor(float16)": "float16",
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "tensor(bfloat16)": "bfloat16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "tensor(int8)": "int8",
    "int8": "int8",
    "tensor(int4)": "int4",
    "int4": "int4",
    "tensor(int64)": "int64",
    "int64": "int64",
    "tensor(bool)": "bool",
    "bool": "bool",
}
_ACTIVATION_BOUNDARY_NAMES = {"hidden_states", "attention_mask", "position_ids"}


def _is_split_model_dir(model_path: str) -> bool:
    """True when the path is a directory of ONNX components or model_info.json."""
    if not isinstance(model_path, str) or not model_path or not os.path.isdir(model_path):
        return False
    if os.path.isfile(os.path.join(model_path, "model_info.json")):
        return True
    try:
        names = os.listdir(model_path)
    except OSError:
        return False
    return any(name.endswith(".onnx") for name in names)


def _read_varint(data: bytes, pos: int) -> Tuple[Optional[int], int]:
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


def _load_json_object(path: str, label: str) -> Dict[str, Any]:
    """Load a JSON object. A missing file is the caller's choice."""
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    except OSError as exc:
        raise ValueError(f"could not read {label}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a dict, got {payload!r}")
    return payload


def _copy_json_value(value: Any) -> Any:
    """Copy dicts and lists so a caller cannot change the loaded document."""
    if isinstance(value, dict):
        return {key: _copy_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_json_value(item) for item in value]
    return value


def _nested_model_info(info: Dict[str, Any]) -> Dict[str, Any]:
    nested = info.get("model_info")
    if isinstance(nested, dict):
        return nested
    return {}


def _positive_config_int(value: Any, name: str) -> Optional[int]:
    """A positive integer. Zero falls through. A negative value raises."""
    if value is None:
        return None
    parsed = _as_config_int(value, name)
    if parsed == 0:
        return None
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative, got {parsed}")
    return parsed


def _architecture_view(model_dir: str, info: Dict[str, Any]) -> Dict[str, Any]:
    """Merge config.json under model_info. An architecture dict wins."""
    view: Dict[str, Any] = {}
    config_path = os.path.join(model_dir, "config.json")
    if os.path.isfile(config_path):
        view.update({
            key: value
            for key, value in _load_json_object(config_path, "config.json").items()
            if key not in _STRUCTURAL_INFO_KEYS and value is not None
        })
    nested = _nested_model_info(info)
    sources = [nested, info]
    for source in sources:
        for key, value in source.items():
            if key in _STRUCTURAL_INFO_KEYS or value is None:
                continue
            view[key] = value
    for source in sources:
        arch = source.get("architecture")
        if isinstance(arch, dict):
            for key, value in arch.items():
                if value is not None:
                    view[key] = value
    model_type = None
    for source in sources:
        arch = source.get("architecture")
        if isinstance(arch, str) and arch.strip():
            model_type = arch.strip()
    if model_type:
        view["model_type"] = model_type
    elif not view.get("model_type") and isinstance(view.get("strategy"), str):
        view["model_type"] = view["strategy"]
    layer_count = None
    for source in sources:
        parsed = _positive_config_int(source.get("num_layers"), "num_layers")
        if parsed is not None:
            layer_count = parsed
    if layer_count is not None:
        view["num_hidden_layers"] = layer_count
    if view.get("torch_dtype") is None and view.get("dtype") is not None:
        view["torch_dtype"] = view["dtype"]
    return view


def _name_list(value: Any, label: str) -> Optional[List[str]]:
    """Copy a list of names. None means the graph will supply the names."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise ValueError(f"{label} must be a list of names, got {value!r}")
    names: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{label} entry must be a non-empty string, got {item!r}")
        names.append(item.strip())
    return names


def _component_type_name(name: str, comp: Dict[str, Any]) -> str:
    raw = comp.get("type", comp.get("component_type"))
    if isinstance(raw, str) and raw.strip():
        return raw.strip().lower()
    if _LAYER_COMPONENT_NAME.fullmatch(name):
        return "layer"
    if name in ("embedding", "embeddings"):
        return "embedding"
    if name in ("norm", "lm_head", "model"):
        return name
    return "unknown"


def _component_layer_id(name: str, comp: Dict[str, Any]) -> Optional[int]:
    if comp.get("layer_id") is not None:
        layer_id = _as_config_int(comp["layer_id"], "layer_id")
        if layer_id < 0:
            raise ValueError(f"layer_id must be non-negative, got {layer_id}")
        return layer_id
    match = _LAYER_COMPONENT_NAME.fullmatch(name)
    if match:
        return int(match.group(1))
    return None


def _component_file(model_dir: str, name: str, comp: Dict[str, Any], files_map: Dict[str, Any]) -> str:
    filename = None
    for key in ("file", "file_path"):
        value = comp.get(key)
        if isinstance(value, str) and value.strip():
            filename = value.strip()
            break
    if filename is None:
        mapped = files_map.get(name)
        if isinstance(mapped, str) and mapped.strip():
            filename = mapped.strip()
    if filename is None:
        filename = f"{name}.onnx"
    if os.path.isabs(filename):
        return filename
    return os.path.join(model_dir, filename)


def _files_map(info: Dict[str, Any]) -> Dict[str, Any]:
    mapped: Dict[str, Any] = {}
    for source in (_nested_model_info(info), info):
        raw = source.get("component_files")
        if isinstance(raw, dict):
            mapped.update(raw)
        elif raw is not None:
            raise ValueError(f"component_files must be a dict, got {raw!r}")
    return mapped


def _component_map(info: Dict[str, Any]) -> Dict[str, Any]:
    nested = _nested_model_info(info)
    for source in (info, nested):
        raw = source.get("components") if isinstance(source, dict) else None
        if isinstance(raw, dict):
            return raw
        if raw is not None:
            raise ValueError(f"components must be a dict, got {raw!r}")
    return {}


def _blank_component(name: str, path: str, comp: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    body = comp if isinstance(comp, dict) else {}
    declared_external = body.get("external_data")
    if declared_external is None:
        declared_external = body.get("external_data_files")
    return {
        "name": name,
        "type": _component_type_name(name, body),
        "layer_id": _component_layer_id(name, body),
        "file": os.path.basename(path),
        "_path": path,
        "_declared_inputs": _name_list(body.get("input_names"), f"input_names of {name}"),
        "_declared_outputs": _name_list(body.get("output_names"), f"output_names of {name}"),
        "_declared_external": _name_list(declared_external, f"external_data of {name}") or [],
        "input_names": [],
        "output_names": [],
        "inputs": [],
        "outputs": [],
        "external_data": [],
    }


def _collect_components(model_dir: str, info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Declared components, then layer graphs that the manifest omitted."""
    files_map = _files_map(info)
    records: List[Dict[str, Any]] = []
    seen = set()
    for name, comp in _component_map(info).items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"component name must be a non-empty string, got {name!r}")
        if not isinstance(comp, dict):
            raise ValueError(f"component {name} must be a dict, got {comp!r}")
        path = _component_file(model_dir, name.strip(), comp, files_map)
        records.append(_blank_component(name.strip(), path, comp))
        seen.add(os.path.abspath(path))
    if os.path.isdir(model_dir):
        discovered = []
        for filename in os.listdir(model_dir):
            match = _LAYER_ONNX_FILE.match(filename)
            if not match:
                continue
            path = os.path.join(model_dir, filename)
            if os.path.abspath(path) in seen:
                continue
            name = f"layer_{int(match.group(1))}"
            discovered.append((int(match.group(1)), _blank_component(
                name,
                path,
                {"type": "layer", "layer_id": int(match.group(1)), "file": filename},
            )))
        for _index, record in sorted(discovered, key=lambda item: item[0]):
            records.append(record)
            seen.add(os.path.abspath(record["_path"]))
    if not records:
        monolithic = os.path.join(model_dir, "model.onnx")
        if os.path.isfile(monolithic):
            records.append(_blank_component("model", monolithic, {"type": "model", "file": "model.onnx"}))
    if not records:
        raise ValueError(f"No ONNX graph found in {model_dir}")
    return records


def _resolve_external(model_dir: str, onnx_path: str, location: str) -> str:
    if os.path.isabs(location):
        return location
    beside = os.path.join(os.path.dirname(onnx_path), location)
    if os.path.isfile(beside):
        return beside
    return os.path.join(model_dir, location)


def _attach_external_data(model_dir: str, records: List[Dict[str, Any]]) -> List[str]:
    """Require every declared or embedded external-data file to exist."""
    found: List[str] = []
    for record in records:
        path = record["_path"]
        if not os.path.isfile(path):
            raise ValueError(f"missing component file {record['file']}")
        locations = list(_external_locations(path))
        for location in record["_declared_external"]:
            if location not in locations:
                locations.append(location)
        for location in locations:
            resolved = _resolve_external(model_dir, path, location)
            if not os.path.isfile(resolved):
                raise ValueError(
                    f"missing external data {location} referenced by {record['file']}"
                )
            if location not in found:
                found.append(location)
        record["external_data"] = locations
    return found


def _open_onnx_session(path: str):
    """Open a graph for inspection without compiling it."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ValueError(
            "onnxruntime is unavailable, so the ONNX graph cannot be read"
        ) from exc
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        return ort.InferenceSession(
            path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise ValueError(
            f"could not read ONNX graph {os.path.basename(path)}: {exc}"
        ) from exc


def _tensor_records(items: Iterable[Any]) -> List[Dict[str, Any]]:
    records = []
    for item in items:
        shape = []
        for dim in getattr(item, "shape", ()) or ():
            if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
                shape.append(None)
            else:
                shape.append(int(dim))
        records.append({
            "name": str(getattr(item, "name", "")),
            "shape": shape,
            "dtype": str(getattr(item, "type", "")),
        })
    return records


def _read_component_graphs(records: List[Dict[str, Any]]) -> None:
    for record in records:
        session = _open_onnx_session(record["_path"])
        inputs = _tensor_records(session.get_inputs())
        outputs = _tensor_records(session.get_outputs())
        input_names = [item["name"] for item in inputs]
        output_names = [item["name"] for item in outputs]
        declared_inputs = record["_declared_inputs"]
        declared_outputs = record["_declared_outputs"]
        if declared_inputs is not None and declared_inputs != input_names:
            raise ValueError(
                f"input_names for {record['name']} do not match the ONNX graph"
            )
        if declared_outputs is not None and declared_outputs != output_names:
            raise ValueError(
                f"output_names for {record['name']} do not match the ONNX graph"
            )
        record["inputs"] = inputs
        record["outputs"] = outputs
        record["input_names"] = input_names
        record["output_names"] = output_names


def _is_past_key_name(name: str) -> bool:
    token = str(name).lower()
    if ".value" in token:
        return False
    return "past" in token and "key" in token


def _past_layer_index(name: str) -> int:
    numbers = re.findall(r"(\d+)", name)
    if not numbers:
        return 0
    return int(numbers[-1])


def _canonical_type_name(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return _ONNX_TYPE_ALIASES.get(value.strip().lower())


def _shapes_agree(contract: List[Optional[int]], graph: List[Optional[int]]) -> bool:
    if len(contract) != len(graph):
        return False
    for declared, observed in zip(contract, graph):
        if declared is None or observed is None:
            continue
        if declared != observed:
            return False
    return True


def _graph_kv_and_dtype(records: List[Dict[str, Any]]) -> Tuple[Optional[int], Optional[int], Optional[str], Optional[str]]:
    """First concrete past-key heads, head dim, and dtype. Later values must agree."""
    heads = None
    head_dim = None
    dtype_name = None
    dtype_raw = None
    for record in records:
        for item in record["inputs"]:
            if not _is_past_key_name(item["name"]):
                continue
            shape = item["shape"]
            if len(shape) == 4:
                if shape[1] is not None:
                    if heads is None:
                        heads = shape[1]
                    elif heads != shape[1]:
                        raise ValueError(
                            f"num_key_value_heads {shape[1]} != {heads}"
                        )
                if shape[3] is not None:
                    if head_dim is None:
                        head_dim = shape[3]
                    elif head_dim != shape[3]:
                        raise ValueError(f"head_dim {shape[3]} != {head_dim}")
            canonical = _canonical_type_name(item["dtype"])
            if canonical is None:
                continue
            if dtype_name is None:
                dtype_name = canonical
                dtype_raw = item["dtype"]
            elif dtype_name != canonical:
                raise ValueError(f"dtype {item['dtype']} != {dtype_raw}")
    return heads, head_dim, dtype_name, dtype_raw


def _observed_layer_count(records: List[Dict[str, Any]]) -> int:
    layers = [
        record for record in records
        if record["type"] == "layer" or record["layer_id"] is not None
    ]
    if layers:
        return len(layers)
    indexes = set()
    for record in records:
        for item in record["inputs"]:
            if _is_past_key_name(item["name"]):
                indexes.add(_past_layer_index(item["name"]))
    return len(indexes)


def _apply_graph_planning_fields(
    fields: Dict[str, Any],
    records: List[Dict[str, Any]],
    dtype_was_set: bool,
) -> None:
    heads, head_dim, dtype_name, dtype_raw = _graph_kv_and_dtype(records)
    if fields["num_key_value_heads"] is None:
        fields["num_key_value_heads"] = heads
    elif heads is not None and fields["num_key_value_heads"] != heads:
        raise ValueError(
            f"num_key_value_heads {fields['num_key_value_heads']} != {heads}"
        )
    if fields["head_dim"] is None:
        fields["head_dim"] = head_dim
    elif head_dim is not None and fields["head_dim"] != head_dim:
        raise ValueError(f"head_dim {fields['head_dim']} != {head_dim}")
    observed = _observed_layer_count(records)
    declared = fields["num_hidden_layers"]
    if declared is None:
        fields["num_hidden_layers"] = observed or None
    elif observed and declared != observed:
        raise ValueError(f"num_layers {declared} != {observed}")
    canonical = _canonical_type_name(str(fields.get("torch_dtype")))
    if not dtype_was_set:
        if dtype_name is not None:
            fields["torch_dtype"] = dtype_name
    elif dtype_name is not None and canonical != dtype_name:
        raise ValueError(f"torch_dtype {fields['torch_dtype']} != {dtype_raw}")


def _name_is_stateful(name: str) -> bool:
    """Past and present KV names carry state. presentation does not."""
    lowered = name.lower()
    return (
        lowered == "past_key_values"
        or lowered.startswith("past_key_values.")
        or lowered == "present"
        or lowered.startswith("present.")
    )


def _boundary_shape(name: str, value: Any) -> List[Optional[int]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise ValueError(f"shape of {name!r} must be a list, got {value!r}")
    dims: List[Optional[int]] = []
    for dim in value:
        if dim is None:
            dims.append(None)
            continue
        number = _as_config_int(dim, f"shape of {name}")
        if number < 0:
            raise ValueError(f"shape of {name!r} must be non-negative, got {number}")
        dims.append(number)
    return dims


def _declared_boundary_items(info: Dict[str, Any]) -> Tuple[List[Any], bool]:
    for source in (info, _nested_model_info(info)):
        if "boundary_tensors" not in source or source.get("boundary_tensors") is None:
            continue
        raw = source["boundary_tensors"]
        if isinstance(raw, (str, bytes)) or not isinstance(raw, list):
            raise ValueError(f"boundary_tensors must be a list, got {raw!r}")
        return raw, True
    return [], False


def _boundary_from_document(item: Any, component_names: set) -> Dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"boundary tensor must be a dict, got {item!r}")
    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"boundary tensor name must be a non-empty string, got {name!r}")
    name = name.strip()
    dtype = item.get("dtype")
    if not isinstance(dtype, str) or not dtype.strip():
        raise ValueError(f"boundary tensor dtype must be a non-empty string, got {dtype!r}")
    producer = item.get("producer_component")
    if not isinstance(producer, str) or not producer.strip():
        raise ValueError(
            f"producer_component must be a non-empty string, got {producer!r}"
        )
    producer = producer.strip()
    if component_names and producer not in component_names:
        raise ValueError(
            f"boundary {name} producer {producer} is not a component"
        )
    consumers = _name_list(item.get("consumer_components"), "consumer_components")
    if consumers is None:
        raise ValueError("consumer_components must be a list of names, got None")
    for consumer in consumers:
        if component_names and consumer not in component_names:
            raise ValueError(
                f"boundary {name} consumer {consumer} is not a component"
            )
    if "is_stateful" in item and item.get("is_stateful") is not None:
        if not isinstance(item["is_stateful"], bool):
            raise ValueError(
                f"is_stateful must be a bool, got {item['is_stateful']!r}"
            )
        stateful = item["is_stateful"]
    else:
        stateful = False
    if _name_is_stateful(name):
        stateful = True
    device = item.get("device_preference")
    if device is not None:
        if isinstance(device, bool) or not isinstance(device, str):
            raise ValueError(f"device_preference must be a string, got {device!r}")
        device = device.strip() or None
    return {
        "name": name,
        "dtype": dtype.strip(),
        "shape": _boundary_shape(name, item.get("shape")),
        "producer_component": producer,
        "consumer_components": consumers,
        "is_stateful": stateful,
        "device_preference": device,
    }


def _io_index(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for record in records:
        for item in list(record["inputs"]) + list(record["outputs"]):
            indexed.setdefault(item["name"], item)
    return indexed


def _check_boundaries_against_graph(
    boundaries: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
) -> None:
    indexed = _io_index(records)
    for boundary in boundaries:
        observed = indexed.get(boundary["name"])
        if observed is None:
            continue
        declared = _canonical_type_name(boundary["dtype"])
        graph_type = _canonical_type_name(observed["dtype"])
        if declared and graph_type and declared != graph_type:
            raise ValueError(
                f"boundary {boundary['name']} dtype {boundary['dtype']} != {observed['dtype']}"
            )
        if not _shapes_agree(boundary["shape"], observed["shape"]):
            raise ValueError(
                f"boundary {boundary['name']} shape {boundary['shape']} != {observed['shape']}"
            )


def _synthesized_boundaries(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """KV and cross-component tensors when model_info has no boundary list."""
    producers: Dict[str, str] = {}
    consumers: Dict[str, List[str]] = {}
    indexed: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for record in records:
        for item in record["outputs"]:
            producers.setdefault(item["name"], record["name"])
            if item["name"] not in indexed:
                indexed[item["name"]] = item
                order.append(item["name"])
        for item in record["inputs"]:
            consumers.setdefault(item["name"], []).append(record["name"])
            if item["name"] not in indexed:
                indexed[item["name"]] = item
                order.append(item["name"])
    boundaries = []
    for name in order:
        item = indexed[name]
        other_consumers = [
            consumer for consumer in consumers.get(name, [])
            if consumer != producers.get(name)
        ]
        if (
            not _name_is_stateful(name)
            and name not in _ACTIVATION_BOUNDARY_NAMES
            and not other_consumers
        ):
            continue
        boundaries.append({
            "name": name,
            "dtype": item["dtype"],
            "shape": list(item["shape"]),
            "producer_component": producers.get(name, "external"),
            "consumer_components": list(consumers.get(name, [])),
            "is_stateful": _name_is_stateful(name),
            "device_preference": None,
        })
    return boundaries


def _boundary_contracts(
    info: Dict[str, Any],
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    component_names = {record["name"] for record in records}
    raw, declared = _declared_boundary_items(info)
    if not declared:
        return _synthesized_boundaries(records)
    boundaries = []
    seen = set()
    for item in raw:
        boundary = _boundary_from_document(item, component_names)
        if boundary["name"] in seen:
            raise ValueError(
                f"boundary_tensors contains a duplicate name {boundary['name']!r}"
            )
        seen.add(boundary["name"])
        boundaries.append(boundary)
    _check_boundaries_against_graph(boundaries, records)
    return boundaries


def _public_component(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": record["name"],
        "type": record["type"],
        "layer_id": record["layer_id"],
        "file": record["file"],
        "input_names": list(record["input_names"]),
        "output_names": list(record["output_names"]),
        "inputs": _copy_json_value(record["inputs"]),
        "outputs": _copy_json_value(record["outputs"]),
        "external_data": list(record["external_data"]),
    }


def _execution_plan_copy(info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for source in (info, _nested_model_info(info)):
        if "execution_plan" not in source or source.get("execution_plan") is None:
            continue
        raw = source["execution_plan"]
        if not isinstance(raw, dict):
            raise ValueError(f"execution_plan must be a dict, got {raw!r}")
        return _copy_json_value(raw)
    return None


def get_split_model_config(model_dir: str) -> Dict[str, Any]:
    """
    Read a split ONNX directory for KV preallocation and static graph planning.

    model_info.json supplies the architecture, component list, and boundary
    contract. config.json fills a field the manifest omits. Each component
    graph supplies input and output names, shapes, and dtypes. Past-key axis
    1 and axis 3 are the KV heads and head dim. External data locations named
    by the graph or the component must exist beside that graph.

    Args:
        model_dir: Directory containing model_info.json and component ONNX files.

    Returns:
        Architecture fields plus components, boundary tensors, and external data.

    Raises:
        FileNotFoundError: If model_dir is not a directory.
        ValueError: If a graph, boundary, or external data file disagrees.
    """
    if not isinstance(model_dir, str) or not os.path.isdir(model_dir):
        raise FileNotFoundError(f"model directory not found: {model_dir}")
    info_path = os.path.join(model_dir, "model_info.json")
    info = _load_json_object(info_path, "model_info.json") if os.path.isfile(info_path) else {}
    view = _architecture_view(model_dir, info)
    dtype_was_set = view.get("torch_dtype") is not None or view.get("dtype") is not None
    fields = _fields_from_hf_config(view)
    records = _collect_components(model_dir, info)
    external = _attach_external_data(model_dir, records)
    _read_component_graphs(records)
    _apply_graph_planning_fields(fields, records, dtype_was_set)
    nested = _nested_model_info(info)
    producer = info.get("producer", nested.get("producer"))
    ir_version = info.get("ir_version", nested.get("ir_version"))
    fields.update({
        "format": "onnx",
        "producer": producer if isinstance(producer, str) else None,
        "ir_version": ir_version if isinstance(ir_version, int) and not isinstance(ir_version, bool) else None,
        "components": [_public_component(record) for record in records],
        "boundary_tensors": _boundary_contracts(info, records),
        "external_data": external,
        "files": [record["file"] for record in records],
        "execution_plan": _execution_plan_copy(info),
    })
    return fields


# Weight width in bits. INT4 is two values per byte.
_DTYPE_BITS = {
    TensorDtype.FLOAT32: 32,
    TensorDtype.FLOAT16: 16,
    TensorDtype.BFLOAT16: 16,
    TensorDtype.INT8: 8,
    TensorDtype.INT4: 4,
    TensorDtype.AUTO: 16,
}
_MEMORY_DTYPE_ALIASES = {
    "float32": TensorDtype.FLOAT32,
    "fp32": TensorDtype.FLOAT32,
    "float": TensorDtype.FLOAT32,
    "float16": TensorDtype.FLOAT16,
    "fp16": TensorDtype.FLOAT16,
    "half": TensorDtype.FLOAT16,
    "bfloat16": TensorDtype.BFLOAT16,
    "bf16": TensorDtype.BFLOAT16,
    "int8": TensorDtype.INT8,
    "uint8": TensorDtype.INT8,
    "int4": TensorDtype.INT4,
    "uint4": TensorDtype.INT4,
    "auto": TensorDtype.AUTO,
}
# Layout bytes match QuantizedDtype: AWQ and GPTQ use an int32 zero point,
# MatMulNBits and blocked INT4 use a uint8 zero point, and every group has
# a float16 scale except QDQ (float32) and bitsandbytes (float32 absmax).
_MEMORY_QUANT_ALIASES = {
    "awq": "awq",
    "awq_int4": "awq",
    "awq_w4a16": "awq",
    "gptq": "gptq",
    "gptq_int4": "gptq",
    "matmulnbits": "matmul_nbits",
    "matmul_nbits": "matmul_nbits",
    "blocked": "blocked",
    "int4_blockwise": "blocked",
    "int4": "blocked",
    "rtn": "rtn",
    "qdq": "qdq",
    "quark": "quark",
    "bitsandbytes": "bitsandbytes",
    "bnb": "bitsandbytes",
}
_QUANT_MEMORY_LAYOUT = {
    "awq": {"bits": 4, "group": 128, "scale": 2, "zero_point": 4, "pack": "int32"},
    "gptq": {"bits": 4, "group": 128, "scale": 2, "zero_point": 4, "pack": "int32"},
    "matmul_nbits": {"bits": 4, "group": 128, "scale": 2, "zero_point": 1, "pack": "block"},
    "blocked": {"bits": 4, "group": 32, "scale": 2, "zero_point": 1, "pack": "block"},
    "rtn": {"bits": None, "group": 32, "scale": 2, "zero_point": 1, "pack": "byte"},
    "qdq": {"bits": None, "group": None, "scale": 4, "zero_point": 1, "pack": "dense"},
    "quark": {"bits": 4, "group": 32, "scale": 2, "zero_point": 1, "pack": "block"},
    "bitsandbytes": {"bits": 4, "group": 64, "scale": 4, "zero_point": 0, "pack": "byte"},
}


def _coerce_memory_dtype(dtype: Any) -> TensorDtype:
    """Accept a TensorDtype or a short name such as fp16."""
    if isinstance(dtype, TensorDtype):
        return dtype
    if isinstance(dtype, bool) or not isinstance(dtype, str):
        raise ValueError(f"dtype must be a TensorDtype, got {dtype!r}")
    key = dtype.strip().lower().replace(" ", "").replace("-", "_")
    if key.startswith("torch."):
        key = key[6:]
    if key not in _MEMORY_DTYPE_ALIASES:
        known = ", ".join(item.value for item in TensorDtype)
        raise ValueError(f"Unknown dtype {dtype!r}. Expected one of: {known}")
    return _MEMORY_DTYPE_ALIASES[key]


def _coerce_memory_quantization(quantization: Optional[str]) -> Optional[str]:
    """Return a canonical scheme name. None leaves the dtype width unchanged."""
    if quantization is None:
        return None
    if isinstance(quantization, bool) or not isinstance(quantization, str):
        raise ValueError(f"quantization must be a string, got {quantization!r}")
    key = quantization.strip().lower().replace(" ", "").replace("-", "_")
    if not key:
        return None
    if key not in _MEMORY_QUANT_ALIASES:
        known = ", ".join(sorted(set(_MEMORY_QUANT_ALIASES.values())))
        raise ValueError(
            f"Unknown quantization {quantization!r}. Expected one of: {known}"
        )
    return _MEMORY_QUANT_ALIASES[key]


def _coerce_memory_group_size(group_size: Optional[int], default: Optional[int]) -> Optional[int]:
    """None uses the scheme default. -1 is per-channel and needs a channel count."""
    if group_size is None:
        return default
    if isinstance(group_size, bool) or not isinstance(group_size, int):
        raise ValueError(f"group_size must be an integer, got {group_size!r}")
    if group_size == -1:
        raise ValueError(
            "group_size -1 is per-channel and needs the channel count; "
            "pass a positive group_size"
        )
    if group_size < 1:
        raise ValueError(f"group_size must be positive or -1 for per-channel, got {group_size}")
    return group_size


def _storage_dtype(bits: int, requested: TensorDtype) -> TensorDtype:
    """Dtype that actually stores the weights after packing."""
    if bits == 4:
        return TensorDtype.INT4
    if bits == 8:
        return TensorDtype.INT8
    if bits == 16 and requested is TensorDtype.BFLOAT16:
        return TensorDtype.BFLOAT16
    if bits == 16:
        return TensorDtype.FLOAT16
    if bits == 32:
        return TensorDtype.FLOAT32
    return requested


def _weight_storage_bytes(num_params: int, bits: int, pack: str, group: Optional[int]) -> int:
    """Packed weight bytes, including a partial int32 word or a partial block."""
    if bits == 4 and pack == "int32":
        return ((num_params + 7) // 8) * 4
    if bits == 4 and pack == "block":
        if group is None:
            raise ValueError("group_size is required for blocked packing")
        if (group * bits) % 8 != 0:
            raise ValueError(
                f"group_size {group} * bits {bits} must fill whole bytes"
            )
        groups = (num_params + group - 1) // group if num_params else 0
        return groups * (group * bits // 8)
    if bits == 4:
        return (num_params + 1) // 2
    if bits % 8 != 0:
        raise ValueError(f"unsupported weight width {bits}")
    return num_params * (bits // 8)


def estimate_model_memory(
    num_params: int,
    dtype: TensorDtype = TensorDtype.FLOAT16,
    quantization: Optional[str] = None,
    group_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Estimate weight memory, including packed INT4 and quantization tensors.

    Args:
        num_params: Number of model parameters.
        dtype: Stored weight dtype. AWQ, GPTQ, and MatMulNBits still pack
            4-bit weights unless this dtype is INT8.
        quantization: Optional scheme such as awq, gptq, or matmul_nbits.
        group_size: Values sharing one scale. None uses the scheme default.
            -1 is per-channel and is rejected because the channel count is
            not in ``num_params``.

    Returns:
        Byte totals for the packed weights plus scale and zero-point tensors.
    """
    if isinstance(num_params, bool) or not isinstance(num_params, int):
        raise ValueError(f"num_params must be an integer, got {num_params!r}")
    if num_params < 0:
        raise ValueError(f"num_params must be non-negative, got {num_params}")
    requested = _coerce_memory_dtype(dtype)
    scheme = _coerce_memory_quantization(quantization)
    if scheme is None:
        bits = _DTYPE_BITS[requested]
        group = None
        scale_each = 0
        zero_each = 0
        pack = "byte" if bits == 4 else "dense"
    else:
        layout = _QUANT_MEMORY_LAYOUT[scheme]
        group = _coerce_memory_group_size(group_size, layout["group"])
        if layout["bits"] == 4 and requested is TensorDtype.INT8:
            bits = 8
            pack = "dense"
        elif layout["bits"] is None:
            bits = _DTYPE_BITS[requested]
            pack = "byte" if bits == 4 else "dense"
        else:
            bits = layout["bits"]
            pack = layout["pack"]
        scale_each = layout["scale"]
        zero_each = layout["zero_point"]
        if pack == "block" and group is not None and (group * bits) % 8 != 0:
            raise ValueError(
                f"group_size {group} * bits {bits} must fill whole bytes"
            )
    weight_bytes = _weight_storage_bytes(num_params, bits, pack, group)
    if scheme is None or num_params == 0:
        groups = 0
    elif group is None:
        groups = 1
    else:
        groups = (num_params + group - 1) // group
    scale_bytes = groups * scale_each
    zero_point_bytes = groups * zero_each
    overhead_bytes = scale_bytes + zero_point_bytes
    total_bytes = weight_bytes + overhead_bytes
    storage = _storage_dtype(bits, requested)
    return {
        "bytes": total_bytes,
        "megabytes": total_bytes / (1024 ** 2),
        "gigabytes": total_bytes / (1024 ** 3),
        "dtype": storage.value,
        "bytes_per_param": bits / 8,
        "quantization": scheme,
        "group_size": group,
        "packing": pack,
        "weight_bytes": weight_bytes,
        "scale_bytes": scale_bytes,
        "zero_point_bytes": zero_point_bytes,
        "overhead_bytes": overhead_bytes,
    }


# Backend NPU order is VitisAI, then RyzenAI. The iGPU follows DirectML,
# MIGraphX, then ROCm. CUDA is a discrete GPU. CPU is last.
_NPU_PROVIDERS = ("VitisAIExecutionProvider", "RyzenAIExecutionProvider")
_IGPU_PROVIDERS = (
    "DmlExecutionProvider",
    "MIGraphXExecutionProvider",
    "ROCmExecutionProvider",
)
_GPU_PROVIDERS = ("CUDAExecutionProvider",)
_CPU_PROVIDER = "CPUExecutionProvider"
_PROVIDER_ALIASES = {
    "vitisai": "VitisAIExecutionProvider",
    "vitisaiexecutionprovider": "VitisAIExecutionProvider",
    "ryzenai": "RyzenAIExecutionProvider",
    "ryzenaiexecutionprovider": "RyzenAIExecutionProvider",
    "rocm": "ROCmExecutionProvider",
    "hip": "ROCmExecutionProvider",
    "rocmeexecutionprovider": "ROCmExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "cudaexecutionprovider": "CUDAExecutionProvider",
    "dml": "DmlExecutionProvider",
    "directml": "DmlExecutionProvider",
    "dmlexecutionprovider": "DmlExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "migraphxexecutionprovider": "MIGraphXExecutionProvider",
    "cpu": "CPUExecutionProvider",
    "cpuexecutionprovider": "CPUExecutionProvider",
}
_NPU_LEGAL_DTYPES = {TensorDtype.FLOAT16, TensorDtype.INT8, TensorDtype.INT4}
_KV_ELEMENT_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int4": 1,
}


def _canonical_provider(name: Any) -> str:
    """Map an alias onto an ONNX Runtime provider name."""
    if isinstance(name, bool) or not isinstance(name, str) or not name.strip():
        raise ValueError(f"available_providers entry must be a provider name, got {name!r}")
    key = name.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    if key not in _PROVIDER_ALIASES:
        raise ValueError(f"Unknown provider {name!r}")
    return _PROVIDER_ALIASES[key]


def _available_provider_set(available_providers: List[str]) -> List[str]:
    """Copy recognized providers. The caller list is left unchanged."""
    if isinstance(available_providers, (str, bytes)) or not isinstance(available_providers, list):
        raise ValueError(
            f"available_providers must be a list of provider names, got {available_providers!r}"
        )
    if not available_providers:
        raise ValueError("available_providers must name at least one provider")
    found: List[str] = []
    for name in available_providers:
        canonical = _canonical_provider(name)
        if canonical not in found:
            found.append(canonical)
    return found


def _config_field(config: Dict[str, Any], names: Iterable[str]) -> Any:
    """First present non-null field."""
    for name in names:
        if name in config and config[name] is not None:
            return config[name]
    return None


def _optional_positive(config: Dict[str, Any], names: Iterable[str], label: str) -> Optional[int]:
    value = _config_field(config, names)
    if value is None:
        return None
    return _positive_config_int(value, label)


def _estimate_num_params(config: Dict[str, Any]) -> int:
    """
    Use num_params when it is present.

    Otherwise count a Llama-style model: one embedding, SwiGLU layers, and an
    untied lm_head. tie_word_embeddings drops the second vocabulary matrix.
    A missing intermediate size uses four times hidden_size. MoE weights count
    every expert.
    """
    supplied = _config_field(config, ("num_params",))
    if supplied is not None:
        count = _as_config_int(supplied, "num_params")
        if count < 0:
            raise ValueError(f"num_params must be non-negative, got {count}")
        return count
    hidden = _optional_positive(config, ("hidden_size",), "hidden_size")
    layers = _optional_positive(config, ("num_hidden_layers", "num_layers"), "num_hidden_layers")
    vocab = _optional_positive(config, ("vocab_size",), "vocab_size")
    if hidden is None or layers is None or vocab is None:
        raise ValueError(
            "num_params is required when hidden_size, num_hidden_layers, and vocab_size are absent"
        )
    heads = _optional_positive(config, ("num_attention_heads",), "num_attention_heads") or 1
    head_dim = _optional_positive(config, ("head_dim", "head_size"), "head_dim")
    if head_dim is None:
        if hidden % heads != 0:
            warnings.warn(
                f"hidden_size ({hidden}) not divisible by num_attention_heads ({heads}). "
                "The parameter estimate uses floor division.",
                UserWarning,
            )
        head_dim = hidden // heads
    kv_heads = _optional_positive(
        config, ("num_key_value_heads", "num_kv_heads"), "num_key_value_heads"
    ) or heads
    intermediate = _optional_positive(
        config, ("intermediate_size",), "intermediate_size"
    )
    experts = _optional_positive(config, ("num_experts", "num_local_experts"), "num_experts") or 0
    moe_intermediate = _optional_positive(
        config, ("moe_intermediate_size",), "moe_intermediate_size"
    )
    if intermediate is None:
        intermediate = 4 * hidden
    query = heads * head_dim
    key_value = kv_heads * head_dim
    attention = hidden * query + (2 * hidden * key_value) + (query * hidden)
    if experts > 0:
        expert_width = moe_intermediate or intermediate
        mlp = experts * 3 * hidden * expert_width
    else:
        mlp = 3 * hidden * intermediate
    embedding = vocab * hidden
    total = embedding + layers * (attention + mlp)
    if config.get("tie_word_embeddings") is not True:
        total += embedding
    return total


def _kv_cache_bytes(config: Dict[str, Any]) -> int:
    """Bytes for key and value at the static context, or 0 when the shape is incomplete."""
    layers = _optional_positive(config, ("num_hidden_layers", "num_layers"), "num_hidden_layers")
    kv_heads = _optional_positive(
        config, ("num_key_value_heads", "num_kv_heads"), "num_key_value_heads"
    )
    head_dim = _optional_positive(config, ("head_dim", "head_size"), "head_dim")
    sliding = _optional_positive(config, ("sliding_window", "sliding_window_size"), "sliding_window")
    context = _optional_positive(
        config, ("max_seq_len", "max_position_embeddings", "max_sequence_length"), "max_seq_len"
    )
    if config.get("use_sliding_window") is False:
        seq = context
    elif sliding is not None:
        seq = sliding
    else:
        seq = context
    if layers is None or kv_heads is None or head_dim is None or seq is None:
        return 0
    cache_dtype = _config_field(config, ("kv_cache_dtype",))
    if cache_dtype is None:
        token = "float16"
    else:
        token = _coerce_memory_dtype(cache_dtype).value
    elements = layers * kv_heads * seq * head_dim
    if token == "int4":
        per_tensor = (elements + 1) // 2
    else:
        per_tensor = elements * _KV_ELEMENT_BYTES[token]
    return 2 * per_tensor


def _plan_is_npu_legal(dtype: TensorDtype, scheme: Optional[str]) -> bool:
    """True when the stored weights are float16, int8, or int4."""
    if scheme in {"awq", "gptq", "matmul_nbits", "blocked", "quark", "bitsandbytes"}:
        return True
    return dtype in _NPU_LEGAL_DTYPES


def _quantization_ladder(
    dtype: TensorDtype,
    scheme: Optional[str],
) -> List[Tuple[TensorDtype, Optional[str]]]:
    """Current weights first, then float16, int8, and MatMulNBits int4."""
    ladder: List[Tuple[TensorDtype, Optional[str]]] = [(dtype, scheme)]
    for candidate in (
        (TensorDtype.FLOAT16, None),
        (TensorDtype.INT8, None),
        (TensorDtype.INT4, "matmul_nbits"),
    ):
        if candidate not in ladder:
            ladder.append(candidate)
    return ladder


def _providers_for_device(device: str, available: List[str]) -> List[str]:
    if device == "npu":
        order = _NPU_PROVIDERS + (_CPU_PROVIDER,)
    elif device == "igpu":
        order = _IGPU_PROVIDERS + (_CPU_PROVIDER,)
    elif device == "gpu":
        order = _GPU_PROVIDERS + ("ROCmExecutionProvider", _CPU_PROVIDER)
    else:
        order = (_CPU_PROVIDER,)
    return [name for name in order if name in available]


def _fallback_device(available: List[str]) -> Tuple[str, str]:
    """iGPU, then CUDA, then CPU."""
    for name in _IGPU_PROVIDERS:
        if name in available:
            return "igpu", name
    for name in _GPU_PROVIDERS:
        if name in available:
            return "gpu", name
    if _CPU_PROVIDER in available:
        return "cpu", _CPU_PROVIDER
    raise ValueError("available_providers must include a CPU, GPU, or NPU provider")


def recommend_execution_strategy(
    model_config: Dict[str, Any],
    available_providers: List[str],
    memory_budget_mb: int,
) -> Dict[str, Any]:
    """
    Recommend an AMD ONNX Runtime placement for one model.

    The NPU preference is VitisAI, then RyzenAI. Prefill uses that provider
    when the weights are float16, int8, or int4 and the weight plus KV bytes
    fit in ``memory_budget_mb``. Decode uses the iGPU when DirectML, MIGraphX,
    or ROCm is available, so the execution is hybrid. The NPU graph stays
    monolithic. A sequential layer split is recommended only off the NPU.

    A float32 or bfloat16 model is offered float16, then int8, then MatMulNBits
    int4. The first scheme that fits is the one that is needed. When int4
    still exceeds the budget, the execution is split and ``fits_npu_memory``
    stays false.

    Args:
        model_config: Architecture and dtype fields from get_model_config or
            get_split_model_config. ``num_params`` overrides the estimate.
        available_providers: ONNX Runtime providers this process can run.
        memory_budget_mb: NPU memory budget in mebibytes.

    Returns:
        Provider, devices, execution, memory fit, and the quantization needed.
    """
    if not isinstance(model_config, dict):
        raise ValueError(f"model_config must be a dict, got {model_config!r}")
    if isinstance(memory_budget_mb, bool) or not isinstance(memory_budget_mb, int):
        raise ValueError(f"memory_budget_mb must be an integer, got {memory_budget_mb!r}")
    if memory_budget_mb < 0:
        raise ValueError(f"memory_budget_mb must be non-negative, got {memory_budget_mb}")
    available = _available_provider_set(available_providers)
    num_params = _estimate_num_params(model_config)
    raw_dtype = _config_field(model_config, ("torch_dtype", "dtype", "weight_dtype"))
    dtype = TensorDtype.FLOAT16 if raw_dtype is None else _coerce_memory_dtype(raw_dtype)
    scheme = _coerce_memory_quantization(
        _config_field(model_config, ("quantization", "quantization_method"))
    )
    group_size = _config_field(model_config, ("group_size", "quantization_group_size"))
    if group_size is not None:
        group_size = _as_config_int(group_size, "group_size")
    kv_bytes = _kv_cache_bytes(model_config)
    budget_bytes = memory_budget_mb * 1024 * 1024
    has_npu = any(name in available for name in _NPU_PROVIDERS)
    chosen_memory = None
    chosen_dtype = dtype
    chosen_scheme = scheme
    fits = False
    saw_candidate = False
    for candidate_dtype, candidate_scheme in _quantization_ladder(dtype, scheme):
        if has_npu and not _plan_is_npu_legal(candidate_dtype, candidate_scheme):
            continue
        memory = estimate_model_memory(
            num_params,
            candidate_dtype,
            candidate_scheme,
            group_size,
        )
        saw_candidate = True
        chosen_memory = memory
        chosen_dtype = candidate_dtype
        chosen_scheme = candidate_scheme
        total_bytes = memory["bytes"] + kv_bytes
        if total_bytes <= budget_bytes:
            fits = True
            break
    if not saw_candidate:
        chosen_memory = estimate_model_memory(num_params, dtype, scheme, group_size)
        total_bytes = chosen_memory["bytes"] + kv_bytes
        fits = total_bytes <= budget_bytes
    else:
        total_bytes = chosen_memory["bytes"] + kv_bytes
    npu_legal = _plan_is_npu_legal(chosen_dtype, chosen_scheme)
    if has_npu and npu_legal:
        device = "npu"
        provider = next(name for name in _NPU_PROVIDERS if name in available)
        others = [name for name in available if name not in _NPU_PROVIDERS]
        if others:
            decode_device, decode_provider = _fallback_device(others)
        else:
            decode_device = "npu"
            decode_provider = provider
        if not fits:
            execution = "split"
        elif decode_device != "npu":
            execution = "hybrid"
        else:
            execution = "monolithic"
        sequential_split_safe = False
    else:
        device, provider = _fallback_device(available)
        decode_device = device
        decode_provider = provider
        execution = "monolithic" if fits else "split"
        sequential_split_safe = True
    session_providers = _providers_for_device(device, available)
    if provider not in session_providers:
        session_providers.insert(0, provider)
    quantization_needed = chosen_dtype is not dtype or chosen_scheme != scheme
    supports_static_kv = kv_bytes > 0 or _optional_positive(
        model_config,
        ("max_seq_len", "max_position_embeddings", "sliding_window", "sliding_window_size"),
        "max_seq_len",
    ) is not None
    if execution == "hybrid" and decode_device == "igpu":
        reason = "Prefill stays on the NPU and decode stays on the iGPU. The NPU graph stays monolithic."
    elif execution == "hybrid":
        reason = "Prefill stays on the NPU. The NPU graph stays monolithic."
    elif execution == "split" and device == "npu":
        reason = "The weights and KV cache exceed the NPU budget, so the run is split. The NPU graph stays monolithic."
    elif execution == "split":
        reason = "The weights and KV cache exceed the memory budget, so the run is split."
    elif device == "npu":
        reason = "The model fits one monolithic NPU graph."
    else:
        reason = "This provider keeps one monolithic graph."
    if quantization_needed and chosen_scheme == "matmul_nbits":
        reason = reason + " MatMulNBits int4 is the quantization that fits."
    elif quantization_needed and chosen_scheme is None and chosen_dtype is TensorDtype.FLOAT16:
        reason = reason + " Float16 weights are required for the NPU."
    elif quantization_needed and chosen_dtype is TensorDtype.INT8:
        reason = reason + " Int8 weights are required to fit the budget."
    return {
        "provider": provider,
        "providers": session_providers,
        "decode_provider": decode_provider,
        "device": device,
        "prefill_device": device,
        "decode_device": decode_device,
        "execution": execution,
        "npu_mode": "monolithic",
        "fits_npu_memory": bool(has_npu and npu_legal and fits),
        "fits_memory": fits,
        "memory_budget_mb": memory_budget_mb,
        "estimated_memory_mb": total_bytes / (1024 ** 2),
        "weight_bytes": chosen_memory["bytes"],
        "kv_bytes": kv_bytes,
        "num_params": num_params,
        "dtype": chosen_memory["dtype"],
        "quantization": chosen_scheme,
        "quantization_needed": quantization_needed,
        "npu_safe": npu_legal,
        "sequential_split_safe": sequential_split_safe,
        "supports_static_kv_cache": supports_static_kv,
        "reason": reason,
    }


def check_dependencies() -> Dict[str, bool]:
    """
    Check available optional dependencies.
    
    Returns:
        Dictionary indicating which libraries are available.
    """
    return {
        "transformers": HAS_TRANSFORMERS,
        "accelerate": HAS_ACCELERATE,
        "torch": HAS_TORCH,
    }
