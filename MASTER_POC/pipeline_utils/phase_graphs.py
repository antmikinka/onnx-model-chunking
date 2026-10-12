"""
Two static graphs for prompt prefill and single-token decode.

A single graph with an If between those phases is the compiler failure
this split exists to avoid. Prefill is one static prompt bucket. Decode
is a different graph: one new token, then a constant slide of the KV
window. Both graphs read one external weight file.

The prefill graph right-aligns the prompt inside ``max_seq_len``. Leading
slots are zeros. Decode drops index 0 and appends the new token, which
keeps the window length fixed. A left-aligned cache that is not full
cannot use that constant slice.
"""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.external_data_helper import set_external_data


_CONTROL_FLOW = {"If", "Loop", "Scan"}
_QUERY_INPUTS = ("hidden", "input_ids", "tokens", "hidden_states")


def _walk_nodes(graph):
    for node in graph.node:
        yield node
        for attribute in node.attribute:
            if attribute.HasField("g"):
                yield from _walk_nodes(attribute.g)
            for subgraph in attribute.graphs:
                yield from _walk_nodes(subgraph)


def assert_no_control_flow(model) -> None:
    """Refuse If, Loop, and Scan. The host picks the phase graph."""
    graph = model.graph if hasattr(model, "graph") else model
    found = [node.op_type for node in _walk_nodes(graph) if node.op_type in _CONTROL_FLOW]
    if found:
        names = ", ".join(found)
        raise ValueError(
            f"A phase graph contains {names}. Prefill and decode are separate "
            "models; the host routes between them."
        )


def _check_lengths(hidden: int, prefill_query_len: int, max_seq_len: int, batch: int, heads: int) -> None:
    if hidden < 1 or batch < 1 or heads < 1:
        raise ValueError("batch, heads, and hidden must be positive")
    if prefill_query_len < 1:
        raise ValueError(f"prefill_query_len must be positive, got {prefill_query_len}")
    if max_seq_len < prefill_query_len:
        raise ValueError(
            f"max_seq_len {max_seq_len} is shorter than the prefill bucket {prefill_query_len}"
        )


def _model(graph) -> onnx.ModelProto:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    return model


def _external_weight(array: np.ndarray, location: str, path: str) -> onnx.TensorProto:
    blob = np.ascontiguousarray(array, dtype=np.float32)
    raw = blob.tobytes()
    with open(path, "wb") as handle:
        handle.write(raw)
    tensor = numpy_helper.from_array(blob, name="weight")
    set_external_data(tensor, location=location, offset=0, length=len(raw))
    tensor.ClearField("raw_data")
    tensor.ClearField("float_data")
    return tensor


def _matmul(weight_name: str = "weight") -> onnx.NodeProto:
    return helper.make_node("MatMul", ["hidden", weight_name], ["projected"], name="project")


def build_phase_models(
    directory: str,
    *,
    hidden: int,
    prefill_query_len: int,
    max_seq_len: int,
    weight: Optional[np.ndarray] = None,
    batch: int = 1,
    heads: int = 1,
    prefill_device: str = "npu",
    decode_device: str = "igpu",
) -> Dict[str, Any]:
    """
    Write ``prefill.onnx``, ``decode.onnx``, and one ``weights.bin``.

    Prefill reads ``hidden`` of shape ``[batch, heads, prefill_query_len, hidden]``
    and writes ``present`` of shape ``[batch, heads, max_seq_len, hidden]``,
    with the projection right-aligned. Decode reads one token plus ``past``
    of length ``max_seq_len`` and writes ``present`` of the same length.

    Returns the ``model_info`` record SessionManager loads.
    """
    _check_lengths(hidden, prefill_query_len, max_seq_len, batch, heads)
    if weight is None:
        weight = np.eye(hidden, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    if weight.shape != (hidden, hidden):
        raise ValueError(f"weight shape {weight.shape} != ({hidden}, {hidden})")
    os.makedirs(directory, exist_ok=True)
    weight_file = os.path.join(directory, "weights.bin")
    shared = _external_weight(weight, "weights.bin", weight_file)

    prefill_shape = [batch, heads, prefill_query_len, hidden]
    window_shape = [batch, heads, max_seq_len, hidden]
    token_shape = [batch, heads, 1, hidden]
    prefill = _prefill_model(shared, prefill_shape, window_shape, max_seq_len - prefill_query_len)
    decode_weight = onnx.TensorProto()
    decode_weight.CopyFrom(shared)
    decode = _decode_model(decode_weight, token_shape, window_shape, max_seq_len)
    assert_no_control_flow(prefill)
    assert_no_control_flow(decode)
    onnx.save(prefill, os.path.join(directory, "prefill.onnx"))
    onnx.save(decode, os.path.join(directory, "decode.onnx"))

    info = {
        "components": {
            "prefill": {
                "file": "prefill.onnx",
                "phase": "prefill",
                "device_preference": prefill_device,
                "input_names": ["hidden"],
                "output_names": ["present"],
                "input_shapes": {"hidden": prefill_shape},
                "output_shapes": {"present": window_shape},
            },
            "decode": {
                "file": "decode.onnx",
                "phase": "decode",
                "device_preference": decode_device,
                "input_names": ["hidden", "past"],
                "output_names": ["present"],
                "input_shapes": {"hidden": token_shape, "past": window_shape},
                "output_shapes": {"present": window_shape},
            },
        },
        "phase_split": {
            "prefill_component": "prefill",
            "decode_component": "decode",
            "prefill_query_len": prefill_query_len,
            "decode_query_len": 1,
            "max_seq_len": max_seq_len,
            "query_axis": 2,
            "prefill_device": prefill_device,
            "decode_device": decode_device,
            "shared_weights": "weights.bin",
        },
    }
    return info


def _prefill_model(weight, prefill_shape, window_shape, pad_len: int) -> onnx.ModelProto:
    hidden_in = helper.make_tensor_value_info("hidden", TensorProto.FLOAT, prefill_shape)
    present = helper.make_tensor_value_info("present", TensorProto.FLOAT, window_shape)
    pads = numpy_helper.from_array(
        np.array([0, 0, pad_len, 0, 0, 0, 0, 0], dtype=np.int64),
        name="prefill_pads",
    )
    fill = numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="prefill_fill")
    nodes = [
        _matmul(),
        helper.make_node(
            "Pad",
            ["projected", "prefill_pads", "prefill_fill"],
            ["present"],
            name="right_align",
            mode="constant",
        ),
    ]
    graph = helper.make_graph(nodes, "prefill", [hidden_in], [present], [weight, pads, fill])
    return _model(graph)


def _decode_model(weight, token_shape, window_shape, max_seq_len: int) -> onnx.ModelProto:
    hidden_in = helper.make_tensor_value_info("hidden", TensorProto.FLOAT, token_shape)
    past = helper.make_tensor_value_info("past", TensorProto.FLOAT, window_shape)
    present = helper.make_tensor_value_info("present", TensorProto.FLOAT, window_shape)
    starts = numpy_helper.from_array(np.array([1], dtype=np.int64), name="slice_start")
    ends = numpy_helper.from_array(np.array([max_seq_len], dtype=np.int64), name="slice_end")
    axes = numpy_helper.from_array(np.array([2], dtype=np.int64), name="slice_axis")
    nodes = [
        _matmul(),
        helper.make_node(
            "Slice",
            ["past", "slice_start", "slice_end", "slice_axis"],
            ["kept"],
            name="drop_oldest",
        ),
        helper.make_node("Concat", ["kept", "projected"], ["present"], name="append_token", axis=2),
    ]
    graph = helper.make_graph(
        nodes,
        "decode",
        [hidden_in, past],
        [present],
        [weight, starts, ends, axes],
    )
    return _model(graph)


def external_weight_key(model) -> Optional[Tuple[str, int, int]]:
    """Return location, offset, and length for the shared ``weight`` initializer."""
    for initializer in model.graph.initializer:
        if initializer.name != "weight":
            continue
        if initializer.data_location != TensorProto.EXTERNAL:
            return None
        fields = {entry.key: entry.value for entry in initializer.external_data}
        return fields.get("location"), int(fields.get("offset", "0")), int(fields.get("length", "0"))
    return None
