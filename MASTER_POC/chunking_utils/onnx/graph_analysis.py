"""Shared ONNX graph analysis for the splitter.

Architecture detection, prefix selection, and tensor dtype and shape
resolution live here so the component, boundary, quantization, KV, and
execution modules share one implementation.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..common.architecture import ArchitectureType, get_prefixes
from ..common.architecture_detector import detect_architecture

def detect_architecture_from_onnx(graph) -> ArchitectureType:
    """
    Detect model architecture from an ONNX graph.

    The descriptor lives in ``architecture_detector``. This wrapper returns
    the architecture type used to choose ONNX prefixes.

    Args:
        graph: ONNX graph.

    Returns:
        Detected ArchitectureType.
    """
    return detect_architecture(graph).arch_type


def get_onnx_prefixes(model_type: Optional[str] = None) -> Tuple[str, str, str, str]:
    """
    Return node name prefixes for different model architectures in ONNX.
    
    Args:
        model_type: Optional model type string.
    
    Returns:
        Tuple of (embed_prefix, layer_prefix, norm_prefix, lm_prefix).
    """
    if model_type:
        from ..common.architecture import architecture_from_string
        arch = architecture_from_string(model_type)
    else:
        arch = ArchitectureType.LLAMA  # Default
    
    prefixes = get_prefixes(arch, "onnx")
    return (
        prefixes.embed_prefix,
        prefixes.layer_prefix,
        prefixes.norm_prefix,
        prefixes.lm_head_prefix,
    )

_ONNX_ELEM_DTYPE = {
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    9: "bool",
    10: "float16",
    11: "float64",
    12: "uint32",
    13: "uint64",
    16: "bfloat16",
}


def _boundary_graph(model):
    """Return the ONNX graph object. A model is unwrapped. A non-graph raises."""
    if model is None:
        raise ValueError("graph must be an ONNX graph")
    graph = model
    if not hasattr(graph, "node") and hasattr(graph, "graph"):
        graph = getattr(graph, "graph")
    nodes = getattr(graph, "node", None)
    if nodes is None or isinstance(nodes, (str, bytes)):
        raise ValueError("graph must be an ONNX graph")
    try:
        return graph, list(nodes)
    except TypeError as exc:
        raise ValueError("graph must be an ONNX graph") from exc


def _value_info_map(graph) -> Dict[str, Any]:
    """Tensor name to the value info that carries dtype and shape."""
    found: Dict[str, Any] = {}
    for collection in (
        getattr(graph, "value_info", ()) or (),
        getattr(graph, "input", ()) or (),
        getattr(graph, "output", ()) or (),
    ):
        for item in collection:
            name = getattr(item, "name", None)
            if isinstance(name, str) and name and name not in found:
                found[name] = item
    return found


def _dims_from_value(value) -> Optional[List[Optional[int]]]:
    """Read a shape. A symbolic axis is None. A missing shape stays unset."""
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
    parsed: List[Optional[int]] = []
    for dim in dims:
        param = getattr(dim, "dim_param", "") or ""
        if isinstance(param, str) and param:
            parsed.append(None)
            continue
        number = getattr(dim, "dim_value", 0)
        if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
            parsed.append(None)
        else:
            parsed.append(number)
    return parsed


def _heuristic_dtype(name: str) -> str:
    """Dtype used when the graph does not record an element type.

    Index-like names use int64 on both inputs and outputs so a guessed mask
    does not change width at a component boundary.
    """
    lowered = name.lower()
    if any(word in lowered for word in ("position", "mask", "idx", "constants", "count")):
        return "int64"
    return "float16"


def _heuristic_shape(name: str) -> List[Optional[int]]:
    """Rank guessed from a 2D, 3D, or 4D marker in the tensor name."""
    rank = 1
    if "4D" in name:
        rank = 4
    elif "3D" in name:
        rank = 3
    elif "2D" in name:
        rank = 2
    return [None] * rank


@dataclass(frozen=True)
class ResolvedTensorType:
    """Dtype and shape chosen for one boundary tensor.

    ``dtype_source`` and ``shape_source`` are ``value_info``, ``initializer``,
    or ``heuristic``. A heuristic field is a guess from the tensor name.
    """

    name: str
    dtype: str
    elem_type: int
    shape: List[Optional[int]]
    dtype_source: str
    shape_source: str


def _elem_type_code(value) -> Optional[int]:
    """Nonzero ONNX element type, or None when the value info leaves it unset."""
    type_proto = getattr(value, "type", None)
    tensor_type = getattr(type_proto, "tensor_type", None) if type_proto is not None else None
    if tensor_type is None:
        return None
    elem = getattr(tensor_type, "elem_type", 0)
    if isinstance(elem, bool) or not isinstance(elem, int) or elem == 0:
        return None
    return elem


def _dtype_name_for_elem(elem: int, name: str) -> str:
    dtype = _ONNX_ELEM_DTYPE.get(elem)
    if dtype is None:
        raise ValueError(f"boundary {name} has unsupported ONNX element type {elem}")
    return dtype


def _elem_for_dtype(dtype: str) -> int:
    for code, name in _ONNX_ELEM_DTYPE.items():
        if name == dtype:
            return code
    raise ValueError(f"boundary dtype {dtype} has no ONNX element type")


def _initializer_dtype_and_shape(initializer, name: str):
    """Tier 2: initializer data type and dimensions. Unset type stays unset."""
    if initializer is None:
        return None, None
    elem = getattr(initializer, "data_type", 0)
    if isinstance(elem, bool) or not isinstance(elem, int):
        raise ValueError(
            f"boundary {name} initializer data_type must be an integer, got {elem!r}"
        )
    dtype = None if elem == 0 else _dtype_name_for_elem(elem, name)
    dims = getattr(initializer, "dims", None)
    if dims is None:
        return dtype, None
    shape: List[Optional[int]] = []
    for dim in dims:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise ValueError(
                f"boundary {name} initializer dim must be an integer, got {dim!r}"
            )
        if dim < 0:
            raise ValueError(f"boundary {name} initializer dim {dim} is negative")
        shape.append(dim)
    return dtype, shape


def _warn_heuristic_type(role: str, name: str, dtype: str, shape, guessed_dtype: bool, guessed_shape: bool) -> None:
    parts = []
    if guessed_dtype:
        parts.append(f"dtype {dtype}")
    if guessed_shape:
        parts.append(f"shape {list(shape)}")
    warnings.warn(
        f"{role} tensor {name} uses a name heuristic for {' and '.join(parts)} "
        f"because the ONNX graph does not record it",
        UserWarning,
        stacklevel=3,
    )


def resolve_tensor_type(name, value=None, initializer=None, *, role: str = "boundary") -> ResolvedTensorType:
    """
    Resolve one tensor dtype and shape in three tiers.

    Tier 1 is value info, a graph input, or a graph output. Tier 2 is an
    initializer. Tier 3 is the name heuristic. Dtype and shape are chosen
    separately, so a declared dtype stays in place when only the shape is
    missing. The heuristic tier warns. An unknown element type, a blank
    name, and a bad initializer dimension raise.

    Args:
        name: Tensor name.
        value: Optional value info or graph input or output.
        initializer: Optional initializer with ``data_type`` and ``dims``.
        role: ``input``, ``output``, or ``boundary``. Named in the warning.

    Returns:
        Resolved dtype name, ONNX element type, shape, and the tier of each.
    """
    if role not in {"input", "output", "boundary"}:
        raise ValueError(f"tensor role must be input, output, or boundary, got {role!r}")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{role} tensor name must be a non-empty string, got {name!r}")

    declared_dtype = None
    declared_elem = None
    declared_shape = None
    if value is not None:
        declared_elem = _elem_type_code(value)
        if declared_elem is not None:
            declared_dtype = _dtype_name_for_elem(declared_elem, name)
        declared_shape = _dims_from_value(value)

    init_dtype, init_shape = _initializer_dtype_and_shape(initializer, name)
    dtype = declared_dtype if declared_dtype is not None else init_dtype
    shape = declared_shape if declared_shape is not None else init_shape
    dtype_source = "value_info" if declared_dtype is not None else ("initializer" if init_dtype is not None else None)
    shape_source = "value_info" if declared_shape is not None else ("initializer" if init_shape is not None else None)
    guessed_dtype = dtype is None
    guessed_shape = shape is None
    if guessed_dtype:
        dtype = _heuristic_dtype(name)
        dtype_source = "heuristic"
        declared_elem = _elem_for_dtype(dtype)
    elif declared_elem is None:
        declared_elem = _elem_for_dtype(dtype)
    if guessed_shape:
        shape = _heuristic_shape(name)
        shape_source = "heuristic"
    if guessed_dtype or guessed_shape:
        _warn_heuristic_type(role, name, dtype, shape, guessed_dtype, guessed_shape)
    return ResolvedTensorType(
        name=name,
        dtype=dtype,
        elem_type=declared_elem,
        shape=list(shape),
        dtype_source=dtype_source,
        shape_source=shape_source,
    )

def _initializer_map(graph) -> Dict[str, Any]:
    found: Dict[str, Any] = {}
    for initializer in getattr(graph, "initializer", ()) or ():
        name = getattr(initializer, "name", None)
        if isinstance(name, str) and name and name not in found:
            found[name] = initializer
    return found
