"""AMD NPU and iGPU execution policy for a split ONNX graph."""

import warnings
from typing import Any, Dict, List, Tuple

from ..common.types import ComponentType
from .graph_analysis import _boundary_graph

_EXEC_KEYS = (
    "npu_safe",
    "recommended_mode",
    "prefill_device",
    "decode_device",
    "resident_components",
    "streamed_components",
    "graph_form",
    "layer_schedule",
    "moe_routing",
)
_EXEC_MODES = {"hybrid_phase_split", "monolithic", "sequential", "persistent"}
_EXEC_DEVICES = {"npu", "igpu", "gpu", "cpu"}
_EXEC_RESIDENT = (
    "embedding",
    "norm",
    "lm_head",
    "moe_router",
    "mtp_head",
    "ngram_embedding",
)
_EXEC_STREAMED = (
    "layer",
    "moe_expert",
    "gated_deltanet",
    "sparse_attention",
    "other",
)
_NPU_QUANT_METHODS = {"MatMulNBits", "AWQ", "GPTQ", "QLinearMatMul"}
_NPU_LEGAL_ELEMS = {2, 3, 10, 16, 21, 22}
_NPU_UNSAFE_ELEMS = {1, 11}


def _copy_execution_policy(info: Dict[str, Any]) -> Dict[str, Any]:
    """Return a fresh record. Resident and streamed lists are not shared."""
    copied = {key: info.get(key) for key in _EXEC_KEYS}
    if not isinstance(copied.get("npu_safe"), bool):
        raise ValueError(
            f"execution policy npu_safe must be a bool, got {copied.get('npu_safe')!r}"
        )
    mode = copied.get("recommended_mode")
    if mode not in _EXEC_MODES:
        raise ValueError(
            f"execution policy recommended_mode {mode!r} is not a known mode"
        )
    for key in ("prefill_device", "decode_device"):
        device = copied.get(key)
        if device not in _EXEC_DEVICES:
            raise ValueError(f"execution policy {key} {device!r} is not a known device")
    for key in ("resident_components", "streamed_components"):
        values = copied.get(key)
        if isinstance(values, (str, bytes)) or not isinstance(values, list):
            raise ValueError(f"execution policy {key} must be a list, got {values!r}")
        copied[key] = list(values)
    form = copied.get("graph_form")
    if form not in {"monolithic", "split"}:
        raise ValueError(f"execution policy graph_form {form!r} is not monolithic or split")
    schedule = copied.get("layer_schedule")
    if schedule not in {"sequential", "persistent"}:
        raise ValueError(
            f"execution policy layer_schedule {schedule!r} is not sequential or persistent"
        )
    routing = copied.get("moe_routing")
    if routing is not None and routing != "moe_routing":
        raise ValueError(f"execution policy moe_routing {routing!r} is not moe_routing")
    return copied


def _warn_execution_default() -> None:
    warnings.warn(
        "execution policy uses a default npu_safe False "
        "because the ONNX graph does not record an NPU-legal dtype",
        UserWarning,
        stacklevel=3,
    )


def _warn_execution_mixed(npu_safe: bool) -> None:
    warnings.warn(
        "execution policy saw float32 and an NPU-legal dtype; "
        f"npu_safe follows the majority and is {npu_safe}",
        UserWarning,
        stacklevel=3,
    )


def _execution_component_types(components) -> List[str]:
    """Unique component types in resident-then-streamed order."""
    if not isinstance(components, dict):
        raise ValueError(f"components must be a dict, got {components!r}")
    if not components:
        raise ValueError("components must name at least one component")
    found = set()
    for raw_name, info in components.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f"component name must be a non-empty string, got {raw_name!r}")
        raw_type = getattr(info, "component_type", None)
        if raw_type is None and isinstance(info, dict):
            raw_type = info.get("component_type")
        if isinstance(raw_type, bool) or not isinstance(raw_type, str) or not raw_type.strip():
            raise ValueError(
                f"component {raw_name.strip()} type must be a non-empty string, got {raw_type!r}"
            )
        text = raw_type.strip().lower().replace("-", "_").replace(" ", "_")
        try:
            kind = ComponentType(text).value
        except ValueError as exc:
            known = ", ".join(item.value for item in ComponentType)
            raise ValueError(
                f"Unknown component type {raw_type!r}. Expected one of: {known}"
            ) from exc
        found.add(kind)
    ordered = [kind for kind in _EXEC_RESIDENT + _EXEC_STREAMED if kind in found]
    return ordered


def _npu_safety(graph, quantization_info) -> Tuple[bool, str]:
    """Return npu_safe and whether the dtype was recorded, mixed, or defaulted."""
    if quantization_info is None:
        quantization_info = {}
    if not isinstance(quantization_info, dict):
        raise ValueError(
            f"quantization_info must be a dict, got {quantization_info!r}"
        )
    method = quantization_info.get("method")
    if isinstance(method, bool):
        raise ValueError(f"quantization method must be a string, got {method!r}")
    if isinstance(method, str) and method.strip() in _NPU_QUANT_METHODS:
        return True, "recorded"
    legal = 0
    unsafe = 0
    for initializer in getattr(graph, "initializer", ()) or ():
        elem = getattr(initializer, "data_type", 0)
        if isinstance(elem, bool) or not isinstance(elem, int):
            name = getattr(initializer, "name", None) or "weight"
            raise ValueError(
                f"execution policy weight {name} data_type must be an integer, got {elem!r}"
            )
        if elem in _NPU_LEGAL_ELEMS:
            legal += 1
        elif elem in _NPU_UNSAFE_ELEMS:
            unsafe += 1
    if legal == 0 and unsafe == 0:
        return False, "default"
    if legal and unsafe:
        return legal >= unsafe, "mixed"
    if legal:
        return True, "recorded"
    return False, "recorded"


def extract_execution_policy(model, components, quantization_info=None) -> Dict[str, Any]:
    """
    Read the AMD execution contract for one split.

    ``recommended_mode`` ``hybrid_phase_split`` prefetches on the NPU and
    decodes on the iGPU. ``embedding``, ``norm``, and ``lm_head`` stay
    resident. ``layer_*`` streams. A MatMulNBits, AWQ, GPTQ, or
    QLinearMatMul graph, or a float16 weight graph, is ``npu_safe``. A
    graph that records float32 stays on the iGPU and may run sequential
    layers. A graph that records no dtype uses ``npu_safe`` false and
    warns. One component is monolithic. A router stays resident, experts
    stream, and ``moe_routing`` records that policy. Sequential execution
    on the NPU is not recommended.

    Args:
        model: ONNX model or graph.
        components: Component name to ComponentInfo or a dict with ``component_type``.
        quantization_info: Optional record from ``extract_quantization_info``.

    Returns:
        JSON record with the device policy, residency, graph form, layer
        schedule, and MoE routing.

    Raises:
        ValueError: If the graph, a component type, or a weight dtype is invalid.
    """
    graph, _nodes = _boundary_graph(model)
    kinds = _execution_component_types(components)
    present = set(kinds)
    npu_safe, source = _npu_safety(graph, quantization_info)
    if source == "default":
        _warn_execution_default()
    elif source == "mixed":
        _warn_execution_mixed(npu_safe)
    split = len(components) > 1
    if not split:
        mode = "monolithic"
        device = "npu" if npu_safe else "igpu"
        prefill = device
        decode = device
        schedule = "persistent"
        resident = list(kinds)
        streamed: List[str] = []
    elif source == "recorded" and not npu_safe:
        mode = "sequential"
        prefill = "igpu"
        decode = "igpu"
        schedule = "sequential"
        resident = [kind for kind in _EXEC_RESIDENT if kind in present]
        streamed = [
            "other" if kind == "other" else f"{kind}_*"
            for kind in _EXEC_STREAMED
            if kind in present
        ]
    else:
        mode = "hybrid_phase_split"
        prefill = "npu"
        decode = "igpu"
        schedule = "persistent"
        resident = [kind for kind in _EXEC_RESIDENT if kind in present]
        streamed = [
            "other" if kind == "other" else f"{kind}_*"
            for kind in _EXEC_STREAMED
            if kind in present
        ]
    routing = "moe_routing" if "moe_router" in present or "moe_expert" in present else None
    return _copy_execution_policy({
        "npu_safe": npu_safe,
        "recommended_mode": mode,
        "prefill_device": prefill,
        "decode_device": decode,
        "resident_components": resident,
        "streamed_components": streamed,
        "graph_form": "split" if split else "monolithic",
        "layer_schedule": schedule,
        "moe_routing": routing,
    })
