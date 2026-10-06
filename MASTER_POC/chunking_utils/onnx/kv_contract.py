"""KV cache contract extracted from an ONNX graph."""

import re
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

from .graph_analysis import (
    _boundary_graph,
    _dims_from_value,
    _dtype_name_for_elem,
    _elem_type_code,
    _initializer_dtype_and_shape,
    _initializer_map,
    _value_info_map,
)

_KV_ROLES = ("past_key", "past_value", "present_key", "present_value")
_KV_INFO_KEYS = (
    "num_layers",
    "num_kv_heads",
    "head_dim",
    "layout",
    "dtype",
    "past_key_pattern",
    "past_value_pattern",
    "present_key_pattern",
    "present_value_pattern",
    "max_seq_len",
    "stateful",
    "stateful_contract",
)
_KV_AXIS_LETTERS = {
    "b": "B",
    "batch": "B",
    "batch_size": "B",
    "n": "N",
    "heads": "N",
    "head": "N",
    "kv_heads": "N",
    "kv_head": "N",
    "num_heads": "N",
    "num_kv_heads": "N",
    "num_key_value_heads": "N",
    "num_attention_heads": "N",
    "n_heads": "N",
    "n_head": "N",
    "n_kv_heads": "N",
    "s": "S",
    "seq": "S",
    "sequence": "S",
    "seq_len": "S",
    "seq_length": "S",
    "sequence_len": "S",
    "sequence_length": "S",
    "past_sequence": "S",
    "past_sequence_length": "S",
    "past_seq_len": "S",
    "past_seq_length": "S",
    "total_sequence_length": "S",
    "total_seq_len": "S",
    "max_seq_len": "S",
    "max_sequence_length": "S",
    "d": "D",
    "dim": "D",
    "head_dim": "D",
    "head_size": "D",
}
_KV_KIND_TOKEN = re.compile(r"(^|[^a-z])(key|value)([^a-z]|$)")


def _empty_kv_cache_info() -> Dict[str, Any]:
    """A graph with no past or present KV tensors still records the key."""
    return _copy_kv_cache_info({
        "num_layers": 0,
        "num_kv_heads": None,
        "head_dim": None,
        "layout": None,
        "dtype": None,
        "past_key_pattern": None,
        "past_value_pattern": None,
        "present_key_pattern": None,
        "present_value_pattern": None,
        "max_seq_len": None,
        "stateful": False,
        "stateful_contract": None,
    })


def _copy_kv_cache_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """Return a fresh record. The stateful contract list is not shared."""
    copied = {key: info.get(key) for key in _KV_INFO_KEYS}
    contract = copied.get("stateful_contract")
    if isinstance(contract, dict):
        roles = contract.get("roles") or []
        if isinstance(roles, (str, bytes)) or not isinstance(roles, list):
            raise ValueError(
                f"kv cache stateful_contract roles must be a list, got {roles!r}"
            )
        zero_copy = contract.get("zero_copy")
        if not isinstance(zero_copy, bool):
            raise ValueError(
                f"kv cache stateful_contract zero_copy must be a bool, got {zero_copy!r}"
            )
        copied["stateful_contract"] = {
            "roles": list(roles),
            "zero_copy": zero_copy,
        }
    return copied


def _warn_kv_default(label: str, field: str, value: Any) -> None:
    warnings.warn(
        f"kv cache tensor {label} uses a default {field} {value} "
        f"because the ONNX graph does not record it",
        UserWarning,
        stacklevel=3,
    )


def _warn_kv_missing(label: str, field: str) -> None:
    warnings.warn(
        f"kv cache tensor {label} does not record {field}",
        UserWarning,
        stacklevel=3,
    )


def _kv_role(name: str) -> Optional[str]:
    """Past or present key or value. The family name past_key_values is not a kind."""
    lowered = name.lower()
    if "past" not in lowered and "present" not in lowered:
        return None
    if "past" in lowered and "present" in lowered:
        raise ValueError(f"kv cache tensor {name} is both past and present")
    stripped = lowered.replace("past_key_values", " ").replace("present_key_values", " ")
    found = {match.group(2) for match in _KV_KIND_TOKEN.finditer(stripped)}
    if "key" in found and "value" in found:
        raise ValueError(f"kv cache tensor {name} is both key and value")
    if "key" not in found and "value" not in found:
        return None
    side = "present" if "present" in lowered else "past"
    kind = "key" if "key" in found else "value"
    return f"{side}_{kind}"


def _append_kv_name(found: List[str], seen: Set[str], name: Any) -> None:
    if not isinstance(name, str):
        return
    text = name.strip()
    if not text or text in seen:
        return
    seen.add(text)
    found.append(text)


def _kv_tensor_names(graph, nodes) -> List[str]:
    """Graph inputs, outputs, value info, then node edges."""
    found: List[str] = []
    seen: Set[str] = set()
    for collection in (
        getattr(graph, "input", ()) or (),
        getattr(graph, "output", ()) or (),
        getattr(graph, "value_info", ()) or (),
    ):
        for item in collection:
            _append_kv_name(found, seen, getattr(item, "name", None))
    for node in nodes:
        for attr in ("input", "output"):
            raw = getattr(node, attr, ()) or ()
            if isinstance(raw, (str, bytes)):
                raise ValueError(f"kv cache node {attr} must be a list, got {raw!r}")
            try:
                edges = list(raw)
            except TypeError as exc:
                raise ValueError(f"kv cache node {attr} must be a list, got {raw!r}") from exc
            for name in edges:
                _append_kv_name(found, seen, name)
    return found


def _pattern_for_role(role: str, names: List[str]) -> Tuple[str, Dict[str, int]]:
    """One ``{layer}`` format string and the layer index of each tensor name."""
    spans = [list(re.finditer(r"\d+", name)) for name in names]
    if len(names) == 1 and not spans[0]:
        _warn_kv_default(names[0], "layer index", 0)
        return names[0], {names[0]: 0}
    if len(names) == 1:
        span = spans[0][0]
        pattern = names[0][:span.start()] + "{layer}" + names[0][span.end():]
        return pattern, {names[0]: int(span.group())}

    candidates = []
    width = min((len(found) for found in spans), default=0)
    for index in range(width):
        pattern = None
        layers: Dict[str, int] = {}
        matched = True
        for name, found in zip(names, spans):
            span = found[index]
            candidate = name[:span.start()] + "{layer}" + name[span.end():]
            if pattern is None:
                pattern = candidate
            elif candidate != pattern:
                matched = False
                break
            layers[name] = int(span.group())
        if matched and pattern is not None:
            candidates.append((pattern, layers))
    if len(candidates) == 1:
        chosen = candidates[0]
    else:
        varying = [item for item in candidates if len(set(item[1].values())) > 1]
        if len(varying) == 1:
            chosen = varying[0]
        else:
            raise ValueError(f"kv cache {role} layer index is ambiguous: {names}")
    pattern, layers = chosen
    seen: Dict[int, str] = {}
    for name, layer in layers.items():
        previous = seen.get(layer)
        if previous is not None:
            raise ValueError(
                f"kv cache {role} layer {layer} is named {previous!r} and {name!r}"
            )
        seen[layer] = name
    return pattern, layers


def _axis_letter(param: Any, name: str) -> Optional[str]:
    if param is None:
        return None
    if not isinstance(param, str):
        raise ValueError(
            f"kv cache tensor {name} axis name must be a string, got {param!r}"
        )
    token = param.strip().lower().replace("-", "_").replace(" ", "_")
    if not token:
        return None
    return _KV_AXIS_LETTERS.get(token)


def _axis_number(number: Any, name: str) -> Optional[int]:
    if number is None or number == "":
        return None
    if isinstance(number, bool) or not isinstance(number, int):
        raise ValueError(
            f"kv cache tensor {name} dimension must be an integer, got {number!r}"
        )
    if number < 0:
        raise ValueError(f"kv cache tensor {name} dimension {number} is negative")
    if number == 0:
        return None
    return number


def _axes_from_dims(dims, name: str) -> List[Tuple[Optional[str], Optional[int]]]:
    if isinstance(dims, (str, bytes)):
        raise ValueError(f"kv cache tensor {name} shape must be a list, got {dims!r}")
    try:
        items = list(dims)
    except TypeError as exc:
        raise ValueError(f"kv cache tensor {name} shape must be a list, got {dims!r}") from exc
    parsed: List[Tuple[Optional[str], Optional[int]]] = []
    for dim in items:
        if isinstance(dim, bool) or isinstance(dim, int):
            parsed.append((None, _axis_number(dim, name)))
            continue
        param = getattr(dim, "dim_param", "")
        number = getattr(dim, "dim_value", 0)
        if param in ("", None):
            param = None
        parsed.append((_axis_letter(param, name), _axis_number(number, name)))
    return parsed


def _axes_from_value(name: str, value) -> Optional[List[Tuple[Optional[str], Optional[int]]]]:
    if value is None:
        return None
    type_proto = getattr(value, "type", None)
    tensor_type = getattr(type_proto, "tensor_type", None) if type_proto is not None else None
    if tensor_type is None:
        return None
    shape_proto = getattr(tensor_type, "shape", None)
    if shape_proto is None:
        return None
    dims = getattr(shape_proto, "dim", None)
    if dims is None:
        return None
    return _axes_from_dims(dims, name)


def _axes_from_initializer(name: str, initializer) -> Optional[List[Tuple[Optional[str], Optional[int]]]]:
    if initializer is None:
        return None
    dims = getattr(initializer, "dims", None)
    if dims is None:
        return None
    return _axes_from_dims(dims, name)


def _kv_dtype_name(name: str, value, initializer) -> Optional[str]:
    """Value info wins. An initializer is the fallback. A missing type stays unset."""
    if value is not None:
        elem = _elem_type_code(value)
        if elem is not None:
            return _dtype_name_for_elem(elem, name)
    if initializer is None:
        return None
    elem = getattr(initializer, "data_type", 0)
    if isinstance(elem, bool) or not isinstance(elem, int):
        raise ValueError(
            f"kv cache tensor {name} initializer data_type must be an integer, got {elem!r}"
        )
    if elem == 0:
        return None
    return _dtype_name_for_elem(elem, name)


def _layout_from_axes(
    axes: List[Tuple[Optional[str], Optional[int]]],
    name: str,
) -> Optional[str]:
    """Named axes spell a layout. A subset that matches BNSD confirms BNSD."""
    letters = [letter for letter, _number in axes]
    if all(letter is None for letter in letters):
        return None
    if len(letters) != 4:
        raise ValueError(
            f"kv cache tensor {name} rank {len(letters)} is not the BNSD rank 4 contract"
        )
    if all(letter is not None for letter in letters):
        layout = "".join(letters)
        if sorted(layout) != sorted("BNSD"):
            raise ValueError(
                f"kv cache tensor {name} layout {layout} is not a 4D cache layout"
            )
        return layout
    for index, letter in enumerate(letters):
        if letter is not None and letter != "BNSD"[index]:
            raise ValueError(
                f"kv cache tensor {name} names axis {index} as {letter}, "
                f"which does not match layout BNSD"
            )
    return "BNSD"


def _axis_at(
    axes: List[Tuple[Optional[str], Optional[int]]],
    layout: str,
    letter: str,
) -> Optional[int]:
    index = layout.find(letter)
    if index < 0 or index >= len(axes):
        return None
    return axes[index][1]


def _fill_kv_field(merged: Dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    current = merged.get(key)
    if current is None:
        merged[key] = value
    elif value != current:
        raise ValueError(
            f"kv cache {key} disagrees across components: {current!r} and {value!r}"
        )


def _kv_layer_count(role_layers: Dict[str, Dict[str, int]]) -> int:
    sets = []
    for layers in role_layers.values():
        found = set(layers.values())
        if found:
            sets.append(found)
    if not sets:
        return 0
    first = sets[0]
    for other in sets[1:]:
        if other != first:
            raise ValueError(
                f"kv cache layers disagree across components: "
                f"{sorted(first)} and {sorted(other)}"
            )
    ordered = sorted(first)
    if ordered != list(range(ordered[-1] + 1)):
        raise ValueError(f"kv cache layers are not contiguous from 0: {ordered}")
    return len(ordered)


def extract_kv_cache_info(model, components=None) -> Dict[str, Any]:
    """
    Read the KV cache contract from an ONNX graph.

    Past and present key and value names become the format strings
    ``kv_cache.py`` uses, with the layer index written as ``{layer}``.
    Layout BNSD is batch, heads, sequence, head dim, the shape
    ``KVCacheManager`` allocates. ``num_kv_heads``, ``head_dim``, and
    ``max_seq_len`` come from those axes. A rank-4 tensor with no axis
    names uses BNSD and warns. A missing element type uses float16 and
    warns. A symbolic sequence axis leaves ``max_seq_len`` unset and warns.
    The name heuristic is not used, so a mask token in the name stays
    float16. Layers must be contiguous from 0 and the same on every role.
    Disagreeing tensors raise before a split writes component files. A
    graph with no KV tensors returns empty fields.

    ``stateful_contract`` lists the roles that are present and sets
    ``zero_copy``, so each past and present tensor stays one OrtValue.

    Args:
        model: ONNX model or graph.
        components: Optional component map. A non-dict raises. The KV
            record still comes from the graph, so a layer count on the
            components does not replace a graph with no KV tensors.

    Returns:
        JSON record with layer count, heads, head dim, layout, dtype,
        name patterns, max sequence length, and the stateful contract.

    Raises:
        ValueError: If the graph, a dimension, or two KV tensors disagree.
    """
    if components is not None and not isinstance(components, dict):
        raise ValueError(f"components must be a dict, got {components!r}")
    graph, nodes = _boundary_graph(model)
    info = _value_info_map(graph)
    initializers = _initializer_map(graph)
    grouped: Dict[str, List[str]] = {role: [] for role in _KV_ROLES}
    for name in _kv_tensor_names(graph, nodes):
        role = _kv_role(name)
        if role is not None:
            grouped[role].append(name)
    if not any(grouped.values()):
        return _empty_kv_cache_info()

    patterns: Dict[str, str] = {}
    role_layers: Dict[str, Dict[str, int]] = {}
    for role in _KV_ROLES:
        names = grouped[role]
        if not names:
            continue
        pattern, layers = _pattern_for_role(role, names)
        patterns[role] = pattern
        role_layers[role] = layers

    tensors = []
    for role in _KV_ROLES:
        for name in grouped[role]:
            value = info.get(name)
            initializer = initializers.get(name)
            tensors.append({
                "name": name,
                "dtype": _kv_dtype_name(name, value, initializer),
                "axes": _axes_from_value(name, value) if value is not None else _axes_from_initializer(name, initializer),
            })
            if value is not None and tensors[-1]["axes"] is None:
                tensors[-1]["axes"] = _axes_from_initializer(name, initializer)

    layouts = []
    ranks = []
    for item in tensors:
        if item["axes"] is None:
            continue
        ranks.append(len(item["axes"]))
        layout = _layout_from_axes(item["axes"], item["name"])
        if layout is not None:
            layouts.append((item["name"], layout))
    chosen = None
    layout_defaulted = False
    if layouts:
        chosen = layouts[0][1]
        for name, layout in layouts[1:]:
            if layout != chosen:
                raise ValueError(
                    f"kv cache layout disagrees across components: {chosen!r} and {layout!r}"
                )
    elif any(rank == 4 for rank in ranks):
        chosen = "BNSD"
        layout_defaulted = True

    merged: Dict[str, Any] = {
        "num_kv_heads": None,
        "head_dim": None,
        "dtype": None,
        "max_seq_len": None,
    }
    for item in tensors:
        _fill_kv_field(merged, "dtype", item["dtype"])
        axes = item["axes"]
        if axes is None or chosen is None:
            continue
        if len(axes) != len(chosen):
            raise ValueError(
                f"kv cache tensor {item['name']} rank {len(axes)} "
                f"does not match layout {chosen}"
            )
        _fill_kv_field(merged, "num_kv_heads", _axis_at(axes, chosen, "N"))
        _fill_kv_field(merged, "head_dim", _axis_at(axes, chosen, "D"))
        _fill_kv_field(merged, "max_seq_len", _axis_at(axes, chosen, "S"))

    num_layers = _kv_layer_count(role_layers)
    label = tensors[0]["name"]
    if layout_defaulted:
        _warn_kv_default(label, "layout", chosen)
    elif chosen is None:
        if ranks:
            warnings.warn(
                f"kv cache tensor {label} rank {ranks[0]} "
                f"is not the BNSD rank 4 contract",
                UserWarning,
                stacklevel=3,
            )
        else:
            _warn_kv_missing(label, "layout")
    if chosen is not None:
        if merged["num_kv_heads"] is None:
            _warn_kv_missing(label, "num_kv_heads")
        if merged["head_dim"] is None:
            _warn_kv_missing(label, "head_dim")
        if merged["max_seq_len"] is None:
            _warn_kv_missing(label, "max_seq_len")
    if merged["dtype"] is None:
        merged["dtype"] = "float16"
        _warn_kv_default(label, "dtype", "float16")

    roles = [role for role in _KV_ROLES if role in patterns]
    record = {
        "num_layers": num_layers,
        "num_kv_heads": merged["num_kv_heads"],
        "head_dim": merged["head_dim"],
        "layout": chosen,
        "dtype": merged["dtype"],
        "past_key_pattern": patterns.get("past_key"),
        "past_value_pattern": patterns.get("past_value"),
        "present_key_pattern": patterns.get("present_key"),
        "present_value_pattern": patterns.get("present_value"),
        "max_seq_len": merged["max_seq_len"],
        "stateful": True,
        "stateful_contract": {
            "roles": roles,
            "zero_copy": True,
        },
    }
    return _copy_kv_cache_info(record)
