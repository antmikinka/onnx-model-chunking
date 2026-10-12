"""Static NPU shapes: frozen axes, a full KV window, and an additive tail."""

import ast
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from chunking_utils.onnx.static_shapes import (
    StaticShapeError,
    additive_length_bias,
    allow_shape_check_bypass,
    apply_static_contract,
    causal_additive_bias,
    freeze_graph_io,
    runtime_slices,
    static_attention_bias,
)


def _check(name, fn):
    try:
        fn()
    except Exception as exc:
        print(f"  {name}: FAIL {exc}")
        return False
    print(f"  {name}: PASS")
    return True


def _softmax(row):
    values = np.asarray(row, dtype=np.float64)
    values = values - np.max(values)
    weights = np.exp(values)
    return weights / np.sum(weights)


def _model(nodes, inputs, outputs, initializers):
    graph = helper.make_graph(nodes, "static_shapes", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    return model


def _dim(value, name):
    text = "" if value else name
    number = 0 if name else value
    info = helper.make_tensor_value_info("unused", TensorProto.FLOAT, [1])
    dim = info.type.tensor_type.shape.dim[0]
    if name:
        dim.ClearField("dim_value")
        dim.dim_param = name
    else:
        dim.ClearField("dim_param")
        dim.dim_value = int(value)
    return dim


def _value(name, dims, dtype=TensorProto.FLOAT):
    info = helper.make_tensor_value_info(name, dtype, [1] * len(dims))
    del info.type.tensor_type.shape.dim[:]
    info.type.tensor_type.shape.dim.extend(dims)
    return info


def test_freeze_past_and_present_match():
    past = _value(
        "past_key_values.0.key",
        [_dim(0, "batch_size"), _dim(4, ""), _dim(0, "past_sequence_length"), _dim(8, "")],
    )
    present = _value(
        "present.0.key",
        [_dim(0, "batch_size"), _dim(4, ""), _dim(0, "total_sequence_length"), _dim(8, "")],
    )
    ids = _value("input_ids", [_dim(0, "batch_size"), _dim(0, "sequence_length")], TensorProto.INT64)
    mask = _value(
        "attention_mask",
        [_dim(0, "batch_size"), _dim(0, "total_sequence_length")],
        TensorProto.INT64,
    )
    model = _model([], [past, ids, mask], [present], [])
    frozen = freeze_graph_io(model, batch_size=1, query_len=1, max_seq_len=16)
    assert frozen == 8
    past_dims = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
    present_dims = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]
    assert past_dims == [1, 4, 16, 8]
    assert present_dims == [1, 4, 16, 8]
    assert [dim.dim_value for dim in model.graph.input[1].type.tensor_type.shape.dim] == [1, 1]
    assert all(dim.dim_param == "" for dim in model.graph.input[0].type.tensor_type.shape.dim)


def test_unknown_dim_raises():
    logits = _value("logits", [_dim(0, "batch_size"), _dim(0, "foo"), _dim(4, "")])
    model = _model([], [logits], [], [])
    try:
        freeze_graph_io(model, batch_size=1, query_len=4, max_seq_len=4)
    except StaticShapeError:
        return
    raise AssertionError("an unknown symbolic axis was frozen")


def test_prefix_slice_becomes_the_full_window():
    past = _value(
        "past_key_values.0.key",
        [_dim(0, "batch_size"), _dim(4, ""), _dim(0, "past_sequence_length"), _dim(8, "")],
    )
    ends = helper.make_tensor_value_info("output_mask", TensorProto.INT64, [1])
    sliced = helper.make_tensor_value_info("sliced", TensorProto.FLOAT, [1, 4, 16, 8])
    node = helper.make_node(
        "Slice",
        ["past_key_values.0.key", "starts", "output_mask", "axes"],
        ["sliced"],
        name="kv_slice",
    )
    model = _model(
        [node],
        [past, ends],
        [sliced],
        [
            numpy_helper.from_array(np.asarray([0], dtype=np.int64), "starts"),
            numpy_helper.from_array(np.asarray([2], dtype=np.int64), "axes"),
        ],
    )
    assert runtime_slices(model) == ["kv_slice"]
    report = apply_static_contract(model, batch_size=1, max_seq_len=16, query_len=1)
    assert report["rewritten_slices"] == ["kv_slice"]
    assert runtime_slices(model) == []
    assert "output_mask" not in [item.name for item in model.graph.input]
    data = np.arange(1 * 4 * 16 * 8, dtype=np.float32).reshape(1, 4, 16, 8)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    assert [item.name for item in session.get_inputs()] == ["past_key_values.0.key"]
    (got,) = session.run(None, {"past_key_values.0.key": data})
    np.testing.assert_array_equal(got, data)


def test_nonzero_start_is_not_a_prefix():
    past = _value(
        "past_key_values.0.key",
        [_dim(1, ""), _dim(4, ""), _dim(16, ""), _dim(8, "")],
    )
    ends = helper.make_tensor_value_info("output_mask", TensorProto.INT64, [1])
    sliced = helper.make_tensor_value_info("sliced", TensorProto.FLOAT, [1, 4, 8, 8])
    node = helper.make_node(
        "Slice",
        ["past_key_values.0.key", "starts", "output_mask", "axes"],
        ["sliced"],
        name="window_slice",
    )
    model = _model(
        [node],
        [past, ends],
        [sliced],
        [
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), "starts"),
            numpy_helper.from_array(np.asarray([2], dtype=np.int64), "axes"),
        ],
    )
    try:
        apply_static_contract(model, batch_size=1, max_seq_len=16, query_len=16)
    except StaticShapeError as exc:
        assert "does not start at 0" in str(exc)
        return
    raise AssertionError("a slice that starts at 1 was widened")


def test_non_kv_runtime_slice_raises():
    hidden = helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, 4])
    ends = helper.make_tensor_value_info("output_mask", TensorProto.INT64, [1])
    out = helper.make_tensor_value_info("sliced", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Slice", ["hidden", "starts", "output_mask", "axes"], ["sliced"], name="hidden_slice")
    model = _model(
        [node],
        [hidden, ends],
        [out],
        [
            numpy_helper.from_array(np.asarray([0], dtype=np.int64), "starts"),
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), "axes"),
        ],
    )
    try:
        apply_static_contract(model, batch_size=1, max_seq_len=4, query_len=4)
    except StaticShapeError as exc:
        assert "hidden_slice" in str(exc)
        return
    raise AssertionError("a runtime slice of a non-KV tensor was rewritten")


def test_additive_tail_matches_the_short_softmax():
    scores = np.asarray([0.5, 1.5, -0.25, 3.0, 2.0], dtype=np.float32)
    bias = additive_length_bias(3, 5, dtype=np.float32)[0]
    assert bias[:3].tolist() == [0, 0, 0]
    assert np.all(bias[3:] == np.float32(-10000))
    full = _softmax(scores + bias)
    short = _softmax(scores[:3])
    np.testing.assert_allclose(full[:3], short, atol=1e-6)
    assert float(np.max(full[3:])) < 1e-8

    square = static_attention_bias(4, 4, 4)[0, 0]
    assert square[0, 0] == 0
    assert np.all(square[0, 1:] == np.float32(-10000))
    row = _softmax(scores[:4] + square[1])
    np.testing.assert_allclose(row[:2], _softmax(scores[:2]), atol=1e-6)

    decode = static_attention_bias(3, 1, 6)[0, 0, 0]
    assert decode[:3].tolist() == [0, 0, 0]
    assert np.all(decode[3:] < -1000)
    assert np.all(causal_additive_bias(1, 6) == 0)


def test_shape_check_bypass_skips_aot_providers():
    assert allow_shape_check_bypass(["CPUExecutionProvider"], True) is True
    assert allow_shape_check_bypass(["CPUExecutionProvider"], False) is False
    assert allow_shape_check_bypass(["CPUExecutionProvider"], True, static_kv_cache=True) is False
    assert allow_shape_check_bypass(
        ["VitisAIExecutionProvider", "CPUExecutionProvider"],
        True,
    ) is False
    assert allow_shape_check_bypass(["RyzenAIExecutionProvider"], True) is False
    assert allow_shape_check_bypass(["OpenVINOExecutionProvider"], True) is False
    assert allow_shape_check_bypass(["CoreMLExecutionProvider"], True) is False


def test_attention_bias_does_not_resize_the_pool():
    from pipeline_utils.io_binding import IOBindingManager, MemoryMode

    manager = IOBindingManager(
        device_type="cpu",
        memory_mode=MemoryMode.NPU_STATIC,
        max_seq_len=32,
        auto_reallocate=True,
    )
    manager.allocate_static_kv_cache(
        "kv",
        num_layers=1,
        num_kv_heads=4,
        head_dim=8,
        dtype=np.float32,
        batch_size=1,
        max_seq_len=32,
    )
    key = manager.get_buffer("kv.layer0.key")
    pointer = int(key.ort_value.data_ptr())
    bias = manager.attention_bias(5)
    assert bias.shape == (1, 32)
    assert bias.dtype == np.float16
    assert np.all(bias[0, :5] == np.float16(0))
    assert np.all(bias[0, 5:] == np.float16(-10000))
    assert key.shape == (1, 4, 32, 8)
    assert int(key.ort_value.data_ptr()) == pointer
    manager.attention_bias(6)
    assert int(key.ort_value.data_ptr()) == pointer


def test_builder_static_mask_has_no_shape_or_slice():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "builder.py")
    source = open(path, encoding="utf-8").read()
    tree = ast.parse(source)
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model")
    methods = {node.name: node for node in model.body if isinstance(node, ast.FunctionDef)}
    for name in ("make_static_attention_bias", "make_static_gqa_lengths", "_freeze_declared_shape", "_wants_static_shapes"):
        assert name in methods, name

    def calls(fn):
        return [
            node.func.attr
            for node in ast.walk(fn)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]

    bias_calls = calls(methods["make_static_attention_bias"])
    for op in ("make_cast", "make_sub", "make_mul", "make_unsqueeze", "make_expand", "make_tile"):
        assert op in bias_calls, op
    for banned in ("make_shape", "make_slice", "make_where"):
        assert banned not in bias_calls
    gqa_calls = calls(methods["make_static_gqa_lengths"])
    assert "make_shape" not in gqa_calls
    assert "make_slice" not in gqa_calls
    assert "make_reduce_sum" in gqa_calls
    assert "if self.static_shapes:" in ast.get_source_segment(source, methods["make_attention_mask_reformatting"])
    assert "apply_static_contract" in open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "chunking_utils", "onnx", "splitter.py"),
        encoding="utf-8",
    ).read()


def main():
    checks = [
        ("freeze_window", test_freeze_past_and_present_match),
        ("unknown_dim", test_unknown_dim_raises),
        ("prefix_slice", test_prefix_slice_becomes_the_full_window),
        ("nonzero_start", test_nonzero_start_is_not_a_prefix),
        ("non_kv_slice", test_non_kv_runtime_slice_raises),
        ("additive_tail", test_additive_tail_matches_the_short_softmax),
        ("shape_check_bypass", test_shape_check_bypass_skips_aot_providers),
        ("attention_bias", test_attention_bias_does_not_resize_the_pool),
        ("builder_static_mask", test_builder_static_mask_has_no_shape_or_slice),
    ]
    return all(_check(name, fn) for name, fn in checks)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
