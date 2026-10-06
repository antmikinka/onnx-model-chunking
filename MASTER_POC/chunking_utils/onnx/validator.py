"""
ONNX split validation and PSNR verification.

``compute_psnr`` compares one reference tensor with the split output.
``verify_split_correctness`` applies the documented thresholds: above 35 dB
is good, and above 40 dB is production. ``validate_boundary_contracts`` and
``validate_component_io`` check the records ``split_onnx`` stores in
``model_info.json`` before component files are written.
"""

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple


PSNR_GOOD_DB = 35.0
PSNR_PRODUCTION_DB = 40.0


def _warn_psnr_failed(psnr: float, minimum: float) -> None:
    shown = "infinite" if math.isinf(psnr) else f"{psnr:.6g}"
    warnings.warn(
        f"PSNR verification failed: {shown} dB is not above {minimum:g} dB",
        UserWarning,
        stacklevel=3,
    )


def _real_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _positive_range(value: Any) -> float:
    number = _real_number(value, "data_range")
    if number <= 0:
        raise ValueError(f"data_range must be positive, got {value!r}")
    return number


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ValueError("numpy is required to compute PSNR") from exc
    return np


def _as_signal(value: Any, label: str):
    """Return a finite float64 array. A bool, a string, or a NaN raises."""
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a numeric array, got {value!r}")
    np = _numpy()
    try:
        array = np.asarray(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be a numeric array, got {value!r}") from exc
    if array.dtype.kind == "b":
        raise ValueError(f"{label} must be a numeric array, got a bool array")
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{label} must be a numeric array, got dtype {array.dtype}")
    if array.size == 0:
        raise ValueError(f"{label} must contain at least one value")
    floats = array.astype(np.float64, copy=False)
    if not np.isfinite(floats).all():
        raise ValueError(f"{label} must be finite")
    return floats


def _is_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _is_output_sequence(value: Any) -> bool:
    """A list of arrays is several outputs. A nested list is one signal."""
    if isinstance(value, (str, bytes)) or isinstance(value, dict):
        return False
    if not isinstance(value, (list, tuple)) or not value:
        return False
    first = value[0]
    if isinstance(first, (str, bytes, bool, int, float, list, tuple)):
        return False
    return True


def _output_pairs(reference: Any, candidate: Any) -> List[Tuple[str, Any, Any]]:
    if _is_mapping(reference) or _is_mapping(candidate):
        if not _is_mapping(reference) or not _is_mapping(candidate):
            raise ValueError("reference and candidate must both be mappings or both be arrays")
        if set(reference) != set(candidate):
            raise ValueError(
                f"output names disagree: {sorted(reference)} and {sorted(candidate)}"
            )
        pairs = []
        for name in reference:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"output name must be a non-empty string, got {name!r}")
            pairs.append((name.strip(), reference[name], candidate[name]))
        return pairs
    if _is_output_sequence(reference) or _is_output_sequence(candidate):
        if not _is_output_sequence(reference) or not _is_output_sequence(candidate):
            raise ValueError("reference and candidate must both be sequences of outputs")
        if len(reference) != len(candidate):
            raise ValueError(
                f"output count disagrees: {len(reference)} and {len(candidate)}"
            )
        return [
            (f"output_{index}", reference[index], candidate[index])
            for index in range(len(reference))
        ]
    return [("output", reference, candidate)]


def _signal_error(reference, candidate, data_range=None) -> Tuple[float, float, float]:
    """Return PSNR, mean squared error, and the peak used for that PSNR."""
    left = _as_signal(reference, "reference")
    right = _as_signal(candidate, "candidate")
    if left.shape != right.shape:
        raise ValueError(f"shape {tuple(left.shape)} does not match {tuple(right.shape)}")
    np = _numpy()
    mse = float(np.mean((right - left) ** 2))
    observed = float(np.max(np.abs(left)))
    peak = observed if data_range is None else _positive_range(data_range)
    if mse == 0.0:
        return math.inf, mse, peak
    if peak == 0.0:
        return 0.0, mse, peak
    return 20.0 * math.log10(peak / math.sqrt(mse)), mse, peak


def compute_psnr(reference, candidate, data_range=None) -> float:
    """
    Peak signal-to-noise ratio in decibels.

    Identical arrays return infinity. ``data_range`` is the peak. When it is
    omitted, the peak is the maximum absolute reference value. A zero peak
    with a nonzero error returns 0 dB.

    Raises:
        ValueError: If an array is empty, non-numeric, non-finite, or a
            different shape, or if ``data_range`` is not positive.
    """
    psnr, _mse, _peak = _signal_error(reference, candidate, data_range=data_range)
    return psnr


def _quality(psnr: float, minimum: float, production: float) -> str:
    if psnr > production:
        return "production"
    if psnr > minimum:
        return "good"
    return "failed"


def _psnr_record(psnr: float, mse: float, peak: float, minimum: float, production: float) -> Dict[str, Any]:
    exact = math.isinf(psnr)
    return {
        "psnr_db": None if exact else psnr,
        "exact": exact,
        "mse": mse,
        "data_range": peak,
        "min_psnr_db": minimum,
        "production_psnr_db": production,
        "quality": _quality(psnr, minimum, production),
        "passed": psnr > minimum,
        "production_ready": psnr > production,
    }


def verify_split_correctness(
    reference,
    candidate,
    min_psnr: float = PSNR_GOOD_DB,
    production_psnr: float = PSNR_PRODUCTION_DB,
    data_range=None,
) -> Dict[str, Any]:
    """
    Compare split outputs with the original outputs.

    A result above ``min_psnr`` (35 dB) is good. A result above
    ``production_psnr`` (40 dB) is production. An exact match is production
    and records ``exact`` true with ``psnr_db`` null. The worst output
    decides a mapping or a sequence of tensors. A result that is not above
    the minimum warns and returns ``passed`` false.

    Raises:
        ValueError: If the arrays, the output names, or a threshold is invalid.
    """
    minimum = _real_number(min_psnr, "min_psnr")
    production = _real_number(production_psnr, "production_psnr")
    if production < minimum:
        raise ValueError(
            f"production_psnr {production} is below min_psnr {minimum}"
        )
    pairs = _output_pairs(reference, candidate)
    if not pairs:
        raise ValueError("reference must contain at least one output")
    records = {}
    worst_name = None
    worst_psnr = None
    for name, left, right in pairs:
        psnr, mse, peak = _signal_error(left, right, data_range=data_range)
        records[name] = _psnr_record(psnr, mse, peak, minimum, production)
        if worst_psnr is None or psnr < worst_psnr:
            worst_psnr = psnr
            worst_name = name
    if worst_name is None or worst_psnr is None:
        raise ValueError("reference must contain at least one output")
    if not (worst_psnr > minimum):
        _warn_psnr_failed(worst_psnr, minimum)
    summary = dict(records[worst_name])
    summary["outputs"] = {name: dict(record) for name, record in records.items()}
    return summary


def _stateful_name(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered == "past_key_values"
        or lowered.startswith("past_key_values.")
        or lowered == "present"
        or lowered.startswith("present.")
    )


def _expected_cache_role(name: str) -> Optional[str]:
    lowered = name.lower()
    if lowered == "past_key_values" or lowered.startswith("past_key_values."):
        if ".key" in lowered:
            return "past_key"
        if ".value" in lowered:
            return "past_value"
        return "past"
    if lowered == "present" or lowered.startswith("present."):
        if ".key" in lowered:
            return "present_key"
        if ".value" in lowered:
            return "present_value"
        return "present"
    return None


def _boundary_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        mapped = value.to_dict()
        if isinstance(mapped, dict):
            return mapped
    raise ValueError(f"boundary tensor must be a mapping, got {value!r}")


def _field(mapping: Dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _shape_list(shape: Any, name: str) -> List[Optional[int]]:
    if isinstance(shape, (str, bytes)) or not isinstance(shape, list):
        raise ValueError(f"{name} must be a list of dimensions, got {shape!r}")
    dims: List[Optional[int]] = []
    for dim in shape:
        if dim is None:
            dims.append(None)
            continue
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            raise ValueError(
                f"{name} dimension must be a non-negative integer or None, got {dim!r}"
            )
        dims.append(dim)
    return dims


def _component_map(components: Any) -> Dict[str, Any]:
    if not isinstance(components, dict):
        raise ValueError(f"components must be a dict, got {components!r}")
    if not components:
        raise ValueError("components must name at least one component")
    cleaned = {}
    for name, info in components.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"component name must be a non-empty string, got {name!r}")
        cleaned[name.strip()] = info
    return cleaned


def _optional_name_list(info: Any, key: str, component: str) -> Optional[List[str]]:
    if isinstance(info, dict):
        if key not in info:
            return None
        value = info.get(key)
    else:
        if not hasattr(info, key):
            return None
        value = getattr(info, key)
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise ValueError(
            f"component {component} {key} must be a list, got {value!r}"
        )
    names = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"component {component} {key} must be non-empty strings, got {item!r}"
            )
        names.append(item.strip())
    if len(names) != len(set(names)):
        raise ValueError(f"component {component} {key} contains a duplicate name")
    return names


def _io_recorded(components: Dict[str, Any]) -> bool:
    for name, info in components.items():
        inputs = _optional_name_list(info, "input_names", name)
        outputs = _optional_name_list(info, "output_names", name)
        if inputs or outputs:
            return True
    return False


def _party_name(value: Any, label: str, tensor: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"boundary {tensor} {label} must be a non-empty string, got {value!r}")
    return value.strip()


def _consumer_list(value: Any, tensor: str) -> List[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise ValueError(f"boundary {tensor} consumers must be a list, got {value!r}")
    names = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"boundary {tensor} consumer must be a non-empty string, got {item!r}"
            )
        names.append(item.strip())
    if len(names) != len(set(names)):
        raise ValueError(f"boundary {tensor} consumers contain a duplicate name")
    return names


def validate_boundary_contracts(boundaries, components=None) -> List[Dict[str, Any]]:
    """
    Check boundary tensors against the split components.

    A past or present tensor is stateful, zero-copy, and NPU-resident.
    A tensor written by one component and read by another is zero-copy.
    When component IO names are recorded, the producer outputs the tensor
    and each downstream consumer inputs it. ``external`` is a graph input.

    Returns:
        A fresh list of boundary records.

    Raises:
        ValueError: If a name, dtype, shape, producer, or stateful contract is invalid.
    """
    if isinstance(boundaries, (str, bytes)) or not isinstance(boundaries, list):
        raise ValueError(f"boundaries must be a list, got {boundaries!r}")
    component_map = None if components is None else _component_map(components)
    recorded = False if component_map is None else _io_recorded(component_map)
    seen = set()
    records = []
    for item in boundaries:
        mapping = _boundary_mapping(item)
        raw_name = _field(mapping, "name")
        name = _party_name(raw_name, "name", "tensor")
        if name in seen:
            raise ValueError(f"duplicate boundary {name!r}")
        seen.add(name)
        dtype = _field(mapping, "dtype")
        if not isinstance(dtype, str) or not dtype.strip():
            raise ValueError(f"boundary {name} dtype must be a non-empty string, got {dtype!r}")
        dtype = dtype.strip()
        shape = _shape_list(_field(mapping, "shape"), f"boundary {name} shape")
        producer = _party_name(
            _field(mapping, "producer", "producer_component"),
            "producer",
            name,
        )
        consumers = _consumer_list(
            _field(mapping, "consumers", "consumer_components"),
            name,
        )
        if component_map is not None:
            if producer != "external" and producer not in component_map:
                raise ValueError(
                    f"boundary {name!r} producer {producer!r} is not a component"
                )
            for consumer in consumers:
                if consumer != "external" and consumer not in component_map:
                    raise ValueError(
                        f"boundary {name!r} consumer {consumer!r} is not a component"
                    )
        downstream = [item_name for item_name in consumers if item_name != producer]
        stateful = _field(mapping, "is_stateful")
        named_state = _stateful_name(name)
        if named_state and stateful is not True:
            raise ValueError(f"boundary {name} is stateful and is_stateful must be true")
        if not named_state and stateful is True:
            raise ValueError(f"boundary {name} is not a past or present tensor")
        if not isinstance(stateful, bool):
            raise ValueError(f"boundary {name} is_stateful must be a bool, got {stateful!r}")
        zero_copy = _field(mapping, "zero_copy")
        handoff = producer != "external" and bool(downstream)
        if stateful or handoff:
            if zero_copy is not True:
                raise ValueError(f"boundary {name} zero_copy must be true")
        elif zero_copy is not None and not isinstance(zero_copy, bool):
            raise ValueError(f"boundary {name} zero_copy must be a bool, got {zero_copy!r}")
        device = _field(mapping, "device_preference")
        if stateful:
            if device != "npu":
                raise ValueError(f"boundary {name} device_preference must be npu, got {device!r}")
        elif device is not None and (not isinstance(device, str) or not device.strip()):
            raise ValueError(
                f"boundary {name} device_preference must be a string, got {device!r}"
            )
        role = _field(mapping, "cache_role")
        expected = _expected_cache_role(name)
        if stateful:
            if not isinstance(role, str) or role.strip() != expected:
                raise ValueError(
                    f"boundary {name} cache_role must be {expected}, got {role!r}"
                )
            role = expected
        elif role not in (None, ""):
            raise ValueError(f"boundary {name} cache_role must be unset, got {role!r}")
        else:
            role = None
        if recorded and component_map is not None:
            if producer != "external":
                outputs = _optional_name_list(
                    component_map[producer], "output_names", producer
                ) or []
                if name not in outputs:
                    raise ValueError(
                        f"boundary {name!r} is not an output of {producer!r}"
                    )
            for consumer in downstream:
                if consumer == "external":
                    continue
                inputs = _optional_name_list(
                    component_map[consumer], "input_names", consumer
                ) or []
                if name not in inputs:
                    raise ValueError(
                        f"boundary {name!r} is not an input of {consumer!r}"
                    )
        records.append({
            "name": name,
            "dtype": dtype,
            "shape": list(shape),
            "producer": producer,
            "consumers": list(consumers),
            "is_stateful": stateful,
            "zero_copy": zero_copy,
            "device_preference": device,
            "cache_role": role,
        })
    return records


def validate_component_io(components, boundaries=None) -> Dict[str, Any]:
    """
    Check component input and output names against the boundary list.

    A name that one component outputs and another inputs must be a boundary.
    When boundaries are omitted, only the name lists are checked.

    Returns:
        A fresh record of component IO and the crossings.

    Raises:
        ValueError: If a name list, a crossing, or a component map is invalid.
    """
    component_map = _component_map(components)
    io = {}
    producers = {}
    consumers = {}
    for name, info in component_map.items():
        inputs = _optional_name_list(info, "input_names", name)
        outputs = _optional_name_list(info, "output_names", name)
        if inputs is None or outputs is None:
            raise ValueError(f"component {name} must record input_names and output_names")
        io[name] = {"input_names": list(inputs), "output_names": list(outputs)}
        for output in outputs:
            if output in producers:
                raise ValueError(
                    f"output {output!r} is produced by {producers[output]!r} and {name!r}"
                )
            producers[output] = name
        for inp in inputs:
            consumers.setdefault(inp, []).append(name)
    crossings = []
    for tensor, producer in producers.items():
        readers = [item for item in consumers.get(tensor, []) if item != producer]
        if readers:
            crossings.append({
                "name": tensor,
                "producer": producer,
                "consumers": list(readers),
            })
    if boundaries is not None:
        contracts = validate_boundary_contracts(boundaries, component_map)
        by_name = {item["name"]: item for item in contracts}
        for crossing in crossings:
            contract = by_name.get(crossing["name"])
            if contract is None:
                raise ValueError(
                    f"crossing {crossing['name']!r} from {crossing['producer']!r} "
                    "is not a boundary"
                )
            if contract["producer"] != crossing["producer"]:
                raise ValueError(
                    f"boundary {crossing['name']!r} producer {contract['producer']!r} "
                    f"does not match {crossing['producer']!r}"
                )
            for reader in crossing["consumers"]:
                if reader not in contract["consumers"]:
                    raise ValueError(
                        f"boundary {crossing['name']!r} does not list consumer {reader!r}"
                    )
    return {
        "components": {name: {"input_names": list(item["input_names"]), "output_names": list(item["output_names"])} for name, item in io.items()},
        "crossings": crossings,
    }
