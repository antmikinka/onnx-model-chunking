"""Quantization contract extracted from an ONNX graph."""

import warnings
from typing import Any, Dict, List, Optional, Tuple

from ..common.architecture import quantized_op_scheme
from .graph_analysis import (
    _boundary_graph,
    _dims_from_value,
    _dtype_name_for_elem,
    _elem_type_code,
    _initializer_dtype_and_shape,
    _initializer_map,
    _value_info_map,
)

_QUANT_EXTRA_ELEM = {
    21: "uint4",
    22: "int4",
}
_QUANT_BITS = {2, 3, 4, 8}
_QUANT_INFO_KEYS = (
    "method",
    "weight_dtype",
    "activation_dtype",
    "group_size",
    "has_zero_point",
    "packing_format",
    "bits",
    "scale_dtype",
    "zero_point_dtype",
)


def _empty_quantization_info() -> Dict[str, Any]:
    """A dense graph has no quantization contract. The key is still recorded."""
    return {
        "method": None,
        "weight_dtype": None,
        "activation_dtype": None,
        "group_size": None,
        "has_zero_point": False,
        "packing_format": None,
        "bits": None,
        "scale_dtype": None,
        "zero_point_dtype": None,
    }


def _quant_node_label(node) -> str:
    name = getattr(node, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    op_type = getattr(node, "op_type", None)
    if isinstance(op_type, str) and op_type.strip():
        return op_type.strip()
    return "node"


def _quant_kind(node) -> Optional[str]:
    """MatMulNBits, AWQ, GPTQ, or QLinearMatMul. A dense op stays unset."""
    op_type = getattr(node, "op_type", "") or ""
    name = getattr(node, "name", "") or ""
    domain = getattr(node, "domain", "") or ""
    if not isinstance(op_type, str):
        op_type = ""
    if not isinstance(name, str):
        name = ""
    if not isinstance(domain, str):
        domain = ""
    text = " ".join(part.lower() for part in (op_type, name, domain))
    if "gptq" in text:
        return "GPTQ"
    if "awq" in text:
        return "AWQ"
    if not op_type.strip():
        return None
    scheme = quantized_op_scheme(op_type)
    if scheme == "int4_weight":
        return "MatMulNBits"
    if scheme == "awq_int4":
        return "AWQ"
    if scheme == "int8_quantized":
        return "QLinearMatMul"
    return None


def _attribute_map(node, label: str) -> Dict[str, Any]:
    """Read ONNX attributes. Integers, floats, and strings are accepted."""
    raw = getattr(node, "attribute", None)
    if raw is None:
        return {}
    if isinstance(raw, dict):
        copied: Dict[str, Any] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(
                    f"quantization attribute for {label} must name a field, got {key!r}"
                )
            copied[key.strip()] = value
        return copied
    if isinstance(raw, (str, bytes)):
        raise ValueError(
            f"quantization attributes for {label} must be a list, got {raw!r}"
        )
    try:
        items = list(raw)
    except TypeError as exc:
        raise ValueError(
            f"quantization attributes for {label} must be a list, got {raw!r}"
        ) from exc
    found: Dict[str, Any] = {}
    for attr in items:
        name = getattr(attr, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"quantization attribute for {label} must name a field, got {name!r}"
            )
        name = name.strip()
        type_code = getattr(attr, "type", 0)
        if isinstance(type_code, bool) or not isinstance(type_code, int):
            raise ValueError(
                f"quantization attribute {name} type must be an integer, got {type_code!r}"
            )
        if type_code in (0, 2):
            value = getattr(attr, "i", None)
            if isinstance(value, bool) or (value is not None and not isinstance(value, int)):
                raise ValueError(
                    f"quantization attribute {name} must be an integer, got {value!r}"
                )
            if isinstance(value, int):
                found[name] = value
            elif type_code == 2:
                raise ValueError(
                    f"quantization attribute {name} must be an integer, got {value!r}"
                )
            continue
        if type_code == 1:
            value = getattr(attr, "f", None)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"quantization attribute {name} must be a float, got {value!r}"
                )
            found[name] = float(value)
            continue
        if type_code == 3:
            raw_s = getattr(attr, "s", b"")
            if isinstance(raw_s, bytes):
                found[name] = raw_s.decode("utf-8", "replace")
            elif isinstance(raw_s, str):
                found[name] = raw_s
            else:
                raise ValueError(
                    f"quantization attribute {name} must be a string, got {raw_s!r}"
                )
            continue
        raise ValueError(
            f"quantization attribute {name} has unsupported type {type_code}"
        )
    return found


def _optional_int_attr(attrs: Dict[str, Any], label: str, *names: str) -> Optional[int]:
    """Return one integer attribute. Two names that disagree raise."""
    folded = {key.lower(): key for key in attrs}
    chosen = None
    for name in names:
        key = name if name in attrs else folded.get(name.lower())
        if key is None:
            continue
        value = attrs[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"quantization attribute {key} for {label} must be an integer, got {value!r}"
            )
        if chosen is not None and value != chosen:
            raise ValueError(
                f"quantization {names[0]} for {label} disagrees: {chosen} and {value}"
            )
        chosen = value
    return chosen


def _quant_elem_name(elem: int, name: str) -> str:
    extra = _QUANT_EXTRA_ELEM.get(elem)
    if extra is not None:
        return extra
    return _dtype_name_for_elem(elem, name)


def _recorded_quant_dtype(tensor_name: str, info: Dict[str, Any], initializers: Dict[str, Any]) -> Optional[str]:
    """Value info wins. An initializer fills a missing element type."""
    if not isinstance(tensor_name, str) or not tensor_name.strip():
        return None
    value = info.get(tensor_name)
    if value is not None:
        elem = _elem_type_code(value)
        if elem is not None:
            return _quant_elem_name(elem, tensor_name)
    initializer = initializers.get(tensor_name)
    if initializer is None:
        return None
    elem = getattr(initializer, "data_type", 0)
    if isinstance(elem, bool) or not isinstance(elem, int):
        raise ValueError(
            f"quantization {tensor_name} initializer data_type must be an integer, got {elem!r}"
        )
    if elem == 0:
        return None
    return _quant_elem_name(elem, tensor_name)


def _recorded_quant_shape(tensor_name: str, info: Dict[str, Any], initializers: Dict[str, Any]):
    """Concrete shape from value info, then an initializer. Symbolic dims stay unset."""
    if not isinstance(tensor_name, str) or not tensor_name.strip():
        return None
    value = info.get(tensor_name)
    if value is not None:
        shape = _dims_from_value(value)
        if shape is not None and shape and all(isinstance(dim, int) for dim in shape):
            return shape
    initializer = initializers.get(tensor_name)
    if initializer is None:
        return None
    _dtype, shape = _initializer_dtype_and_shape(initializer, tensor_name)
    if not shape or any(dim is None for dim in shape):
        return None
    return shape


def _node_input_names(node) -> List[str]:
    inputs = getattr(node, "input", ()) or ()
    if isinstance(inputs, (str, bytes)):
        raise ValueError(
            f"quantized node {_quant_node_label(node)} inputs must be a list, got {inputs!r}"
        )
    try:
        items = list(inputs)
    except TypeError as exc:
        raise ValueError(
            f"quantized node {_quant_node_label(node)} inputs must be a list, got {inputs!r}"
        ) from exc
    names: List[str] = []
    for item in items:
        if item is None:
            names.append("")
        elif isinstance(item, str):
            names.append(item.strip())
        else:
            raise ValueError(
                f"quantized node {_quant_node_label(node)} input must be a string, got {item!r}"
            )
    return names


def _input_at(names: List[str], index: int) -> str:
    if index < 0 or index >= len(names):
        return ""
    return names[index]


def _require_bits(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in _QUANT_BITS:
        raise ValueError(
            f"quantization bits for {label} must be 2, 3, 4, or 8, got {value!r}"
        )
    return value


def _require_group_size(value: int, bits: Optional[int], label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"quantization group_size for {label} must be a positive integer, got {value!r}"
        )
    if bits is not None and bits < 8 and (value * bits) % 8 != 0:
        raise ValueError(
            f"quantization group_size {value} and bits {bits} for {label} do not fill whole bytes"
        )
    return value


def _infer_group_size(bits: Optional[int], weight_shape, scale_shape, k_attr: Optional[int], label: str) -> Optional[int]:
    """MatMulNBits stores weights as [N, blocks, block_size * bits / 8]."""
    inferred = None
    if (
        bits is not None
        and weight_shape is not None
        and len(weight_shape) == 3
        and isinstance(weight_shape[2], int)
        and weight_shape[2] > 0
        and (weight_shape[2] * 8) % bits == 0
    ):
        inferred = weight_shape[2] * 8 // bits
    from_k = None
    blocks = None
    if scale_shape is not None and len(scale_shape) >= 2 and isinstance(scale_shape[-1], int):
        blocks = scale_shape[-1]
    elif weight_shape is not None and len(weight_shape) >= 2 and isinstance(weight_shape[1], int):
        blocks = weight_shape[1]
    if k_attr is not None and blocks:
        if k_attr <= 0 or k_attr % blocks != 0:
            raise ValueError(
                f"quantization group_size for {label} cannot be derived from K {k_attr}"
            )
        from_k = k_attr // blocks
    if inferred is not None and from_k is not None and inferred != from_k:
        raise ValueError(
            f"quantization group_size for {label} disagrees: {inferred} and {from_k}"
        )
    return inferred if inferred is not None else from_k


def _weight_dtype_for_bits(bits: int, storage: Optional[str]) -> str:
    if bits < 8:
        return f"int{bits}"
    if storage == "uint8":
        return "uint8"
    if storage in {"int8", "uint4", "int4"}:
        return storage
    return "int8"


def _packing_for_storage(kind: str, bits: Optional[int], storage: Optional[str]) -> Optional[str]:
    if kind == "QLinearMatMul":
        return "qdq"
    if bits is not None and bits < 8:
        if storage == "uint8" or storage is None:
            return "packed_uint8" if storage == "uint8" else None
        if storage == "int32":
            return "packed_int32"
        return storage
    if storage in {"int8", "uint8"}:
        return "dense"
    return None


def _warn_quant_default(label: str, field: str, value: Any) -> None:
    warnings.warn(
        f"quantized node {label} uses a default {field} {value} "
        f"because the ONNX graph does not record it",
        UserWarning,
        stacklevel=3,
    )


def _node_quantization(node, info: Dict[str, Any], initializers: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One quantized node. A missing recorded field uses the op default and warns."""
    kind = _quant_kind(node)
    if kind is None:
        return None
    label = _quant_node_label(node)
    attrs = _attribute_map(node, label)
    inputs = _node_input_names(node)
    qlinear = kind == "QLinearMatMul"
    if qlinear:
        activation_name = _input_at(inputs, 0)
        weight_name = _input_at(inputs, 3)
        scale_name = _input_at(inputs, 4)
        zero_name = _input_at(inputs, 5) or _input_at(inputs, 2)
        has_zero_point = bool(zero_name)
    else:
        activation_name = _input_at(inputs, 0)
        weight_name = _input_at(inputs, 1)
        scale_name = _input_at(inputs, 2)
        zero_name = _input_at(inputs, 3)
        has_zero_point = bool(zero_name)
        if _input_at(inputs, 4):
            kind = "GPTQ"

    bits = _optional_int_attr(attrs, label, "bits")
    group_size = _optional_int_attr(attrs, label, "block_size", "group_size")
    k_attr = _optional_int_attr(attrs, label, "K", "k")
    bits_recorded = bits is not None
    if bits_recorded:
        bits = _require_bits(bits, label)
    else:
        bits = 8 if qlinear else 4
    storage = _recorded_quant_dtype(weight_name, info, initializers)
    activation = _recorded_quant_dtype(activation_name, info, initializers)
    scale_dtype = _recorded_quant_dtype(scale_name, info, initializers)
    zero_dtype = _recorded_quant_dtype(zero_name, info, initializers) if zero_name else None
    if group_size is None:
        group_size = _infer_group_size(
            bits if bits_recorded else None,
            _recorded_quant_shape(weight_name, info, initializers),
            _recorded_quant_shape(scale_name, info, initializers),
            k_attr,
            label,
        )
    if group_size is not None:
        group_size = _require_group_size(group_size, bits, label)

    if not bits_recorded:
        _warn_quant_default(label, "bits", bits)
    if group_size is None and not qlinear:
        group_size = _require_group_size(128, bits, label)
        _warn_quant_default(label, "group_size", group_size)
    if activation is None:
        activation = "uint8" if qlinear else "float16"
        _warn_quant_default(label, "activation_dtype", activation)
    if scale_dtype is None:
        scale_dtype = "float32" if qlinear else activation
        _warn_quant_default(label, "scale_dtype", scale_dtype)
    if has_zero_point and zero_dtype is None:
        zero_dtype = "int8" if qlinear else "uint8"
        _warn_quant_default(label, "zero_point_dtype", zero_dtype)
    packing = _packing_for_storage(kind, bits, storage)
    if packing is None:
        packing = "packed_uint8"
        _warn_quant_default(label, "packing_format", packing)
    weight_dtype = _weight_dtype_for_bits(bits, storage)
    return {
        "method": kind,
        "weight_dtype": weight_dtype,
        "activation_dtype": activation,
        "group_size": group_size,
        "has_zero_point": has_zero_point,
        "packing_format": packing,
        "bits": bits,
        "scale_dtype": scale_dtype,
        "zero_point_dtype": zero_dtype,
    }


def _merge_quantization(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One contract for every split component. A disagreement raises."""
    if not records:
        return _empty_quantization_info()
    merged = dict(records[0])
    for other in records[1:]:
        for key in _QUANT_INFO_KEYS:
            current = merged.get(key)
            value = other.get(key)
            if current is None:
                merged[key] = value
            elif value is not None and value != current:
                raise ValueError(
                    f"quantization {key} disagrees across components: {current!r} and {value!r}"
                )
    for key in _QUANT_INFO_KEYS:
        merged.setdefault(key, None)
    return {key: merged.get(key) for key in _QUANT_INFO_KEYS}


def extract_quantization_info(model) -> Dict[str, Any]:
    """
    Read the quantization contract from an ONNX graph.

    MatMulNBits contributes bits, group size (``block_size``), scale dtype,
    zero-point dtype, activation dtype, weight dtype, and packing layout.
    A connected ``g_idx`` or a GPTQ name selects GPTQ. An AWQ op or name
    selects AWQ. QLinearMatMul is recorded as int8 QDQ. A field the graph
    does not record uses the op default and warns. Quantized nodes that
    disagree raise, so split components keep one contract. A dense graph
    returns empty fields.

    Args:
        model: ONNX model or graph.

    Returns:
        JSON record with method, dtypes, group size, zero point, and packing.

    Raises:
        ValueError: If the graph, an attribute, or two quantized nodes disagree.
    """
    graph, nodes = _boundary_graph(model)
    info = _value_info_map(graph)
    initializers = _initializer_map(graph)
    records: List[Dict[str, Any]] = []
    for node in nodes:
        record = _node_quantization(node, info, initializers)
        if record is not None:
            records.append(record)
    return _merge_quantization(records)


def _initializer_numbers(initializer) -> Optional[List[float]]:
    """Host numbers stored on an initializer. A missing payload stays unset."""
    if initializer is None:
        return None
    explicit = getattr(initializer, "values", None)
    candidates = [("values", explicit)]
    for name in ("float_data", "double_data", "int32_data", "int64_data", "uint64_data"):
        candidates.append((name, getattr(initializer, name, None)))
    for _name, raw in candidates:
        if raw is None or isinstance(raw, (str, bytes, bytearray)):
            continue
        if isinstance(raw, bool):
            continue
        if isinstance(raw, (int, float)):
            return [float(raw)]
        try:
            values = list(raw)
        except TypeError:
            continue
        if not values:
            continue
        numbers: List[float] = []
        for item in values:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                return None
            numbers.append(float(item))
        return numbers
    return None


def _quant_parameter_names(node) -> Tuple[str, str]:
    """Scale and zero-point input names for one quantized node."""
    kind = _quant_kind(node)
    if kind is None:
        return "", ""
    inputs = _node_input_names(node)
    if kind == "QLinearMatMul":
        scale_name = _input_at(inputs, 4)
        zero_name = _input_at(inputs, 5) or _input_at(inputs, 2)
    else:
        scale_name = _input_at(inputs, 2)
        zero_name = _input_at(inputs, 3)
    return scale_name, zero_name


def extract_quantization_tensors(model) -> Dict[str, Any]:
    """
    Names and host values of MatMulNBits, AWQ, GPTQ, and QLinear scales
    and zero-points.

    The contract dict from ``extract_quantization_info`` stays unchanged.
    A tensor with no stored numbers contributes its name and no values.
    """
    _graph, nodes = _boundary_graph(model)
    initializers = _initializer_map(_graph)
    scale_names: List[str] = []
    zero_names: List[str] = []
    scales: List[float] = []
    zero_points: List[int] = []
    for node in nodes:
        scale_name, zero_name = _quant_parameter_names(node)
        if scale_name and scale_name not in scale_names:
            scale_names.append(scale_name)
            numbers = _initializer_numbers(initializers.get(scale_name))
            if numbers is not None:
                scales.extend(numbers)
        if zero_name and zero_name not in zero_names:
            zero_names.append(zero_name)
            numbers = _initializer_numbers(initializers.get(zero_name))
            if numbers is not None:
                zero_points.extend(int(item) for item in numbers)
    return {
        "scale_tensors": scale_names,
        "zero_point_tensors": zero_names,
        "scales": scales,
        "zero_points": zero_points,
    }


def build_quantization_info(model):
    """
    QuantizationInfo for a MatMulNBits, AWQ, or GPTQ graph.

    Group size, bits, and zero-point tensors come from the graph. A dense
    graph and QLinearMatMul stay unset because QuantizationInfo does not
    record that method. Returns None when there is no supported contract.
    """
    from .types import QuantizationInfo

    info = extract_quantization_info(model)
    method = info.get("method")
    if method not in {"MatMulNBits", "AWQ", "GPTQ"}:
        return None
    tensors = extract_quantization_tensors(model)
    weight = info.get("weight_dtype") or "float16"
    bits = info.get("bits")
    if bits is None:
        bits = {"int4": 4, "int8": 8, "float16": 16, "float32": 32}.get(weight, 16)
    return QuantizationInfo(
        method=method,
        weight_bits=int(bits),
        weight_dtype=weight,
        activation_dtype=info.get("activation_dtype") or "float16",
        group_size=info.get("group_size"),
        has_zero_point=bool(info.get("has_zero_point")),
        scale_tensors=tuple(tensors["scale_tensors"]),
        zero_point_tensors=tuple(tensors["zero_point_tensors"]),
        scales=tuple(float(item) for item in tensors["scales"]),
        zero_points=tuple(int(item) for item in tensors["zero_points"]),
    )
