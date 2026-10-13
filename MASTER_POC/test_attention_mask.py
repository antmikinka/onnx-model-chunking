"""Additive attention-mask rewrite.

Blocked attention positions become a scalar -10000 add. Shape repair and the
lm_head token ban stay Where ops.
"""

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from chunking_utils.onnx.attention_mask import rewrite_attention_mask_ops
from chunking_utils.onnx.splitter import split_onnx


def _model(graph):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    # This onnxruntime build rejects IR version 14.
    model.ir_version = 10
    return model


def _run(model, feeds):
    onnx.checker.check_model(model)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(None, feeds)


def _scalar_init(model, name):
    for initializer in model.graph.initializer:
        if initializer.name == name:
            return initializer
    raise AssertionError(name)


def test_keep_mask_becomes_scalar_bias():
    keep = helper.make_tensor_value_info("keep", TensorProto.BOOL, [1, 2])
    bias = helper.make_tensor_value_info("bias", TensorProto.FLOAT16, [1, 2])
    zero = numpy_helper.from_array(np.array(0, dtype=np.float16), name="zero")
    sentinel = numpy_helper.from_array(
        np.array(np.finfo(np.float16).min, dtype=np.float16),
        name="sentinel",
    )
    node = helper.make_node(
        "Where",
        ["keep", "zero", "sentinel"],
        ["bias"],
        name="/model/attn/Where",
    )
    graph = helper.make_graph([node], "keep", [keep], [bias], [zero, sentinel])
    model = _model(graph)
    assert rewrite_attention_mask_ops(model) == 1
    assert [node.op_type for node in model.graph.node] == ["Cast", "Sub", "Mul"]
    assert {initializer.name for initializer in model.graph.initializer} == {
        f"spatial_npu_attention_bias_{TensorProto.FLOAT16}",
        f"spatial_npu_attention_one_{TensorProto.FLOAT16}",
    }
    stored = _scalar_init(model, f"spatial_npu_attention_bias_{TensorProto.FLOAT16}")
    assert stored.data_type == TensorProto.FLOAT16
    assert float(numpy_helper.to_array(stored)) == np.float16(-10000)
    assert not any(initializer.dims and initializer.dims[0] > 1 for initializer in model.graph.initializer)
    output = _run(model, {"keep": np.array([[True, False]])})[0]
    assert output.dtype == np.float16
    assert output.tolist() == [[0.0, -10000.0]]
    assert rewrite_attention_mask_ops(model) == 0


def test_scores_where_softmax_zeroes_blocked():
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT16, [1, 2])
    blocked = helper.make_tensor_value_info("blocked", TensorProto.BOOL, [1, 2])
    probs = helper.make_tensor_value_info("probs", TensorProto.FLOAT16, [1, 2])
    fill = numpy_helper.from_array(np.array(-np.inf, dtype=np.float16), name="neg_inf")
    where = helper.make_node(
        "Where",
        ["blocked", "neg_inf", "scores"],
        ["masked"],
        name="/layers.0/self_attn/Where",
    )
    softmax = helper.make_node("Softmax", ["masked"], ["probs"], name="softmax")
    graph = helper.make_graph([where, softmax], "scores", [scores, blocked], [probs], [fill])
    model = _model(graph)
    assert rewrite_attention_mask_ops(model) == 1
    assert "Where" not in {node.op_type for node in model.graph.node}
    assert "Add" in {node.op_type for node in model.graph.node}
    output = _run(
        model,
        {
            "scores": np.array([[1, 2]], dtype=np.float16),
            "blocked": np.array([[False, True]]),
        },
    )[0]
    assert float(output[0, 1]) == 0.0
    assert float(output[0, 0]) > 0.99


def test_expand_and_constant_of_shape_fill():
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 2])
    blocked = helper.make_tensor_value_info("blocked", TensorProto.BOOL, [1, 2])
    masked = helper.make_tensor_value_info("masked", TensorProto.FLOAT, [1, 2])
    shape = numpy_helper.from_array(np.array([1, 2], dtype=np.int64), name="shape")
    fill = helper.make_node(
        "ConstantOfShape",
        ["shape"],
        ["filled"],
        name="fill",
        value=numpy_helper.from_array(np.array([-65504.0], dtype=np.float32)),
    )
    expand = helper.make_node("Expand", ["filled", "shape"], ["expanded"], name="expand_fill")
    where = helper.make_node(
        "Where",
        ["blocked", "expanded", "scores"],
        ["masked"],
        name="/model/attention/Where",
    )
    graph = helper.make_graph(
        [fill, expand, where],
        "expanded",
        [scores, blocked],
        [masked],
        [shape],
    )
    model = _model(graph)
    assert rewrite_attention_mask_ops(model) == 1
    assert "Where" not in {node.op_type for node in model.graph.node}
    assert "ConstantOfShape" not in {node.op_type for node in model.graph.node}
    stored = _scalar_init(model, f"spatial_npu_attention_bias_{TensorProto.FLOAT}")
    assert stored.data_type == TensorProto.FLOAT
    assert float(numpy_helper.to_array(stored)) == -10000.0
    output = _run(
        model,
        {
            "scores": np.array([[4.0, 5.0]], dtype=np.float32),
            "blocked": np.array([[True, False]]),
        },
    )[0]
    assert output[0, 0] < -1000
    assert output[0, 1] == 5.0


def test_masked_fill_op_uses_score_dtype():
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT16, [1, 2])
    mask = helper.make_tensor_value_info("mask", TensorProto.BOOL, [1, 2])
    filled = helper.make_tensor_value_info("filled", TensorProto.FLOAT16, [1, 2])
    node = helper.make_node(
        "MaskedFill",
        ["scores", "mask"],
        ["filled"],
        name="/model/causal/MaskedFill",
        value=float("-inf"),
    )
    graph = helper.make_graph([node], "masked_fill", [scores, mask], [filled])
    model = _model(graph)
    assert rewrite_attention_mask_ops(model) == 1
    stored = _scalar_init(model, f"spatial_npu_attention_bias_{TensorProto.FLOAT16}")
    assert stored.data_type == TensorProto.FLOAT16
    output = _run(
        model,
        {
            "scores": np.array([[3, 8]], dtype=np.float16),
            "mask": np.array([[False, True]]),
        },
    )[0]
    assert float(output[0, 0]) == 3.0
    assert float(output[0, 1]) < -1000.0


def test_exclusions_stay_where_and_unmatched_graph_is_unchanged():
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [2])
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [2])
    shape_out = helper.make_tensor_value_info("shape_out", TensorProto.INT64, [4])
    banned = helper.make_tensor_value_info("banned", TensorProto.FLOAT, [2])
    plain = helper.make_tensor_value_info("plain", TensorProto.FLOAT, [2])
    sink = numpy_helper.from_array(np.array(-65504.0, dtype=np.float32), name="sink")
    ones = numpy_helper.from_array(np.array([1, 1, 1, 1], dtype=np.int64), name="ones")
    dims = numpy_helper.from_array(np.array([2, 1, 2, 1], dtype=np.int64), name="dims")
    lm_head = helper.make_node(
        "Where",
        ["cond", "sink", "scores"],
        ["banned"],
        name="/lm_head/Where",
    )
    softmax = helper.make_node("Softmax", ["banned"], ["plain"], name="logit_softmax")
    repair = helper.make_node(
        "Where",
        ["cond", "ones", "dims"],
        ["shape_out"],
        name="/model/attn_mask_reformat/Where_1",
    )
    # cond is rank-1 and ones is rank-1 of a different length; this node is
    # never executed. It only has to survive as Where.
    graph = helper.make_graph(
        [lm_head, softmax, repair],
        "excluded",
        [scores, cond],
        [banned, plain, shape_out],
        [sink, ones, dims],
    )
    model = _model(graph)
    before = model.SerializeToString()
    assert rewrite_attention_mask_ops(model) == 0
    assert model.SerializeToString() == before
    assert [node.op_type for node in model.graph.node] == ["Where", "Softmax", "Where"]


def test_small_negative_and_gelu_select_stay():
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [2])
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [2])
    half_out = helper.make_tensor_value_info("half_out", TensorProto.FLOAT, [2])
    gelu_out = helper.make_tensor_value_info("gelu_out", TensorProto.FLOAT, [2])
    half = numpy_helper.from_array(np.array(-0.5, dtype=np.float32), name="half")
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [2])
    clip = helper.make_tensor_value_info("clip", TensorProto.FLOAT, [2])
    mild = helper.make_node(
        "Where",
        ["cond", "half", "scores"],
        ["half_out"],
        name="/model/attention/Where",
    )
    gelu = helper.make_node(
        "Where",
        ["cond", "data", "clip"],
        ["gelu_out"],
        name="/model/attn/gelu/Where",
    )
    graph = helper.make_graph(
        [mild, gelu],
        "kept",
        [scores, cond, data, clip],
        [half_out, gelu_out],
        [half],
    )
    model = _model(graph)
    before = model.SerializeToString()
    assert rewrite_attention_mask_ops(model) == 0
    assert model.SerializeToString() == before


def test_combined_mask_stays_finite_in_float16():
    bias = np.float16(-10000)
    assert bias + bias == np.float16(-20000)
    assert np.isfinite(bias + bias)
    sink = np.float16(np.finfo(np.float16).min)
    with np.errstate(over="ignore"):
        assert not np.isfinite(sink + sink)


def test_unrelated_initializer_and_shared_fill_stay():
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [2])
    blocked = helper.make_tensor_value_info("blocked", TensorProto.BOOL, [2])
    masked = helper.make_tensor_value_info("masked", TensorProto.FLOAT, [2])
    shared = helper.make_tensor_value_info("shared", TensorProto.FLOAT, [])
    fill = numpy_helper.from_array(np.array(-np.inf, dtype=np.float32), name="neg_inf")
    unused = numpy_helper.from_array(np.array([1.0, 2.0, 3.0], dtype=np.float32), name="unused_weight")
    where = helper.make_node(
        "Where",
        ["blocked", "neg_inf", "scores"],
        ["masked"],
        name="/model/attention/Where",
    )
    identity = helper.make_node("Identity", ["neg_inf"], ["shared"], name="share_fill")
    graph = helper.make_graph(
        [where, identity],
        "shared_fill",
        [scores, blocked],
        [masked, shared],
        [fill, unused],
    )
    model = _model(graph)
    assert rewrite_attention_mask_ops(model) == 1
    names = {initializer.name for initializer in model.graph.initializer}
    assert "neg_inf" in names
    assert "unused_weight" in names
    assert "Where" not in {node.op_type for node in model.graph.node}
    assert "Identity" in {node.op_type for node in model.graph.node}


def test_split_onnx_rewrites_before_shape_inference():
    import chunking_utils.onnx.splitter as splitter

    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 2])
    blocked = helper.make_tensor_value_info("blocked", TensorProto.BOOL, [1, 2])
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2])
    fill = numpy_helper.from_array(np.array(-np.inf, dtype=np.float32), name="neg_inf")
    where = helper.make_node(
        "Where",
        ["blocked", "neg_inf", "scores"],
        ["masked"],
        name="/model/layers.0/self_attn/Where",
    )
    softmax = helper.make_node("Softmax", ["masked"], ["logits"])
    graph = helper.make_graph([where, softmax], "tiny", [scores, blocked], [logits], [fill])
    model = _model(graph)

    def fake_load(path):
        return model

    def fake_infer(loaded, infer_shapes):
        ops = [node.op_type for node in loaded.graph.node]
        raise RuntimeError("ops:" + ",".join(ops))

    original_load = splitter.load_onnx_model
    original_infer = splitter.infer_shapes_with_lazy_fallback
    splitter.load_onnx_model = fake_load
    splitter.infer_shapes_with_lazy_fallback = fake_infer
    try:
        try:
            split_onnx(model="tiny.onnx", output="/tmp/attn-mask-out", split_only=True)
        except RuntimeError as exc:
            text = str(exc)
            assert text.startswith("ops:")
            assert "Where" not in text.split(":", 1)[1].split(",")
            assert "Add" in text
        else:
            raise AssertionError("shape inference hook did not run")
    finally:
        splitter.load_onnx_model = original_load
        splitter.infer_shapes_with_lazy_fallback = original_infer


def test_masked_projected_graph_matches_reference():
    from chunking_utils.onnx.golden_attention_subgraph import (
        default_projection,
        default_token,
    )
    from chunking_utils.onnx.masked_projected_attention import (
        MASKED_PROJECTED_OPS,
        attention_bias_f16,
        build_masked_projected_attention_model,
        masked_projected_facts,
        reference_masked_projected_attention,
    )

    seq, head = 8, 64
    model = build_masked_projected_attention_model(seq_len=seq, head_dim=head)
    facts = masked_projected_facts(model)
    assert facts["ops"] == list(MASKED_PROJECTED_OPS)
    assert facts["ir_version"] == 10
    assert facts["bias_dtype"] == TensorProto.FLOAT16
    assert "Where" not in facts["ops"] and "Transpose" not in facts["ops"]
    shape = (1, 1, seq, head)
    count = int(np.prod(shape))
    past_k = (np.arange(count, dtype=np.float32) / np.float32(count)).reshape(shape)
    past_v = (np.arange(count, dtype=np.float32) / np.float32(count / 2)).reshape(shape)
    hidden = np.ascontiguousarray(default_token(head))
    weights = [default_projection(seed, head, head) for seed in (1, 2, 3, 4)]
    bias = attention_bias_f16(seq)
    expect_k, expect_v, expect_h = reference_masked_projected_attention(
        past_k, past_v, hidden, *weights, bias
    )
    key, value, projected = _run(
        model,
        {
            "past_key_values.0.key": past_k,
            "past_key_values.0.value": past_v,
            "hidden_states": hidden,
        },
    )
    assert np.max(np.abs(key - expect_k)) < 1e-5
    assert np.max(np.abs(value - expect_v)) < 1e-5
    assert np.max(np.abs(projected - expect_h)) < 1e-5


def test_masked_projected_multihead_matches_reference():
    from chunking_utils.onnx.golden_attention_subgraph import default_projection
    from chunking_utils.onnx.masked_projected_attention import (
        MASKED_PROJECTED_OPS,
        attention_bias_f16,
        build_masked_projected_attention_model,
        masked_projected_facts,
        reference_masked_projected_attention,
    )

    seq, heads, head = 8, 8, 128
    model = build_masked_projected_attention_model(seq_len=seq, head_dim=head, num_heads=heads)
    facts = masked_projected_facts(model)
    assert facts["ops"] == list(MASKED_PROJECTED_OPS)
    assert "Transpose" not in facts["ops"]
    shape = (1, heads, seq, head)
    count = int(np.prod(shape))
    past_k = (np.arange(count, dtype=np.float32) / np.float32(count)).reshape(shape)
    past_v = (np.arange(count, dtype=np.float32) / np.float32(count / 2)).reshape(shape)
    token_count = heads * head
    hidden = (
        np.arange(token_count, dtype=np.float32) / np.float32(token_count) - np.float32(0.5)
    ).reshape(1, heads, 1, head)
    weights = [default_projection(seed, head, head) for seed in (1, 2, 3, 4)]
    bias = attention_bias_f16(seq)
    expect_k, expect_v, expect_h = reference_masked_projected_attention(
        past_k, past_v, hidden, *weights, bias
    )
    key, value, projected = _run(
        model,
        {
            "past_key_values.0.key": past_k,
            "past_key_values.0.value": past_v,
            "hidden_states": np.ascontiguousarray(hidden),
        },
    )
    assert np.max(np.abs(key - expect_k)) < 1e-5
    assert np.max(np.abs(value - expect_v)) < 1e-5
    assert np.max(np.abs(projected - expect_h)) < 1e-5


def main():
    tests = [
        test_keep_mask_becomes_scalar_bias,
        test_scores_where_softmax_zeroes_blocked,
        test_expand_and_constant_of_shape_fill,
        test_masked_fill_op_uses_score_dtype,
        test_exclusions_stay_where_and_unmatched_graph_is_unchanged,
        test_small_negative_and_gelu_select_stay,
        test_combined_mask_stays_finite_in_float16,
        test_unrelated_initializer_and_shared_fill_stay,
        test_split_onnx_rewrites_before_shape_inference,
        test_masked_projected_graph_matches_reference,
        test_masked_projected_multihead_matches_reference,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: ok")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
