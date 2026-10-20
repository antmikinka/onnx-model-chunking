"""Static NPU shapes: frozen axes, a full KV window, and an additive tail."""

import ast
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

import inspect

from chunking_utils.onnx.splitter import split_onnx
from chunking_utils.onnx.static_shapes import (
    StaticShapeError,
    _choose_chunk_k,
    _f32_to_bf16_raw,
    _initializer_array,
    _split_large_matmul_nbits,
    additive_length_bias,
    allow_shape_check_bypass,
    apply_static_contract,
    causal_additive_bias,
    freeze_graph_io,
    rewrite_fused_norms,
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


def _dims(value):
    return [dim.dim_value for dim in value.type.tensor_type.shape.dim]


def test_qwen_layer_freezes_to_the_static_window():
    path = os.environ.get(
        "QWEN_ATTENTION_LAYER",
        r"C:\Users\antmi\qwen3-work\qwen3-8b-split\model.onnx\layer_0_of_36.onnx",
    )
    if not os.path.exists(path):
        raise AssertionError(f"Qwen attention layer was not found at {path}")
    model = onnx.load(path, load_external_data=False)
    directory = os.path.dirname(path)
    stored_before = {item.name: item for item in model.graph.initializer}
    down = next(node for node in model.graph.node if node.name == "/model/layers.0/mlp/down_proj/MatMul_Q4")
    orig_w = _initializer_array(stored_before[down.input[1]], directory)
    orig_s = _initializer_array(stored_before[down.input[2]], directory)
    assert orig_w.shape == (4096, 384, 16)
    assert orig_s.shape == (4096, 384)
    matmul_nbits = sum(node.op_type == "MatMulNBits" for node in model.graph.node)
    report = apply_static_contract(
        model,
        batch_size=1,
        query_len=1,
        max_seq_len=16,
        external_data_dir=directory,
    )
    assert report["activation"] == "bfloat16"
    assert matmul_nbits == 5
    assert report["split_matmuls"] == ["/model/layers.0/mlp/down_proj/MatMul_Q4"]
    assert sum(node.op_type == "MatMulNBits" for node in model.graph.node) == 7
    assert sum(node.op_type == "GroupQueryAttention" for node in model.graph.node) == 1
    assert report["rewritten_attention"] == ["/model/layers.0/attn/GroupQueryAttention"]
    assert report["rewritten_slices"] == []
    by_name = {value.name: value for value in list(model.graph.input) + list(model.graph.output)}
    assert _dims(by_name["past_key_values.0.key"]) == [1, 8, 16, 128]
    assert _dims(by_name["past_key_values.0.value"]) == [1, 8, 16, 128]
    assert _dims(by_name["present.0.key"]) == [1, 8, 16, 128]
    assert _dims(by_name["present.0.value"]) == [1, 8, 16, 128]
    assert _dims(by_name["/model/embed_tokens/Gather/output_0"]) == [1, 1, 4096]
    assert _dims(by_name["attention_bias"]) == [1, 1, 1, 17]
    assert "output_mask" not in by_name
    removed = {
        "/model/attn_mask_reformat/attn_mask_subgraph/Sub/Cast/output_0",
        "/model/attn_mask_reformat/attn_mask_subgraph/Gather/Cast/output_0",
    }
    assert removed.isdisjoint(by_name)
    gqa = next(node for node in model.graph.node if node.op_type == "GroupQueryAttention")
    constants = {item.name for item in model.graph.initializer}
    assert gqa.input[5] in constants
    assert gqa.input[6] in constants
    seqlens = numpy_helper.to_array(
        next(item for item in model.graph.initializer if item.name == gqa.input[5])
    )
    total = numpy_helper.to_array(
        next(item for item in model.graph.initializer if item.name == gqa.input[6])
    )
    assert seqlens.tolist() == [16]
    assert int(total) == 17
    assert runtime_slices(model) == []
    slices = [node for node in model.graph.node if node.op_type == "Slice"]
    assert [node.output[0] for node in slices if node.output[0].startswith("present.")] == [
        "present.0.key",
        "present.0.value",
    ]
    down_slices = [node for node in slices if "down_proj" in node.name]
    assert len(down_slices) == 3
    assert sum(node.op_type == "Concat" for node in model.graph.node) == 0
    bf16_casts = [
        node for node in model.graph.node
        if node.op_type == "Cast" and node.attribute[0].i == TensorProto.BFLOAT16
    ]
    fp32_casts = [
        node for node in model.graph.node
        if node.op_type == "Cast" and node.attribute[0].i == TensorProto.FLOAT
    ]
    assert sorted(node.name for node in bf16_casts if node.name.startswith("Cast_")) == [
        "Cast_/model/embed_tokens/Gather/output_0_bf16",
        "Cast_attention_bias_bf16",
        "Cast_cos_cache_bf16",
        "Cast_model.layers.0.attn.o_proj.MatMul.weight_scales_bf16",
        "Cast_model.layers.0.attn.qkv_proj.MatMul.weight_scales_bf16",
        "Cast_model.layers.0.mlp.down_proj.MatMul.weight_scales/k0_bf16",
        "Cast_model.layers.0.mlp.down_proj.MatMul.weight_scales/k4096_bf16",
        "Cast_model.layers.0.mlp.down_proj.MatMul.weight_scales/k8192_bf16",
        "Cast_model.layers.0.mlp.gate_proj.MatMul.weight_scales_bf16",
        "Cast_model.layers.0.mlp.up_proj.MatMul.weight_scales_bf16",
        "Cast_past_key_values.0.key_bf16",
        "Cast_past_key_values.0.value_bf16",
        "Cast_sin_cache_bf16",
    ]
    assert sum(node.name.endswith("/Cast_inv") for node in bf16_casts) == 4
    assert [node.name.endswith("/Cast_x") for node in fp32_casts] == [True, True, True, True]
    assert sum(node.op_type == "ReduceMean" for node in model.graph.node) == 0
    assert sum(node.op_type == "ReduceSum" for node in model.graph.node) == 4
    counts = sorted(
        float(numpy_helper.to_array(item))
        for item in model.graph.initializer
        if item.name.endswith("/rms_count")
    )
    assert counts == [128.0, 128.0, 4096.0, 4096.0]
    hidden = "/model/embed_tokens/Gather/output_0"
    assert by_name[hidden].type.tensor_type.elem_type == TensorProto.FLOAT
    assert by_name["past_key_values.0.key"].type.tensor_type.elem_type == TensorProto.FLOAT
    assert by_name["past_key_values.0.value"].type.tensor_type.elem_type == TensorProto.FLOAT
    for value in model.graph.output:
        assert value.type.tensor_type.elem_type == TensorProto.BFLOAT16
    gqa_inputs = list(gqa.input)
    assert gqa_inputs[3] == "past_key_values.0.key/bf16"
    assert gqa_inputs[4] == "past_key_values.0.value/bf16"
    assert gqa_inputs[7] == "cos_cache/bf16"
    assert gqa_inputs[8] == "sin_cache/bf16"
    assert gqa_inputs[9] == "position_ids"
    assert gqa_inputs[10] == "attention_bias/bf16"
    assert _dims(by_name["position_ids"]) == [1, 1]
    assert by_name["position_ids"].type.tensor_type.elem_type == TensorProto.INT64
    norm_weights = [
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.attn.q_norm.layernorm.weight",
        "model.layers.0.attn.k_norm.layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
    ]
    assert sorted(report["bf16_norm_weights"]) == sorted(norm_weights)
    stored = {item.name: item for item in model.graph.initializer}
    for name in norm_weights:
        item = stored[name]
        assert item.data_type == TensorProto.BFLOAT16
        assert len(item.raw_data) == int(item.dims[0]) * 2
        assert not item.external_data
    packed = stored["model.layers.0.attn.qkv_proj.MatMul.weight_Q4"]
    assert packed.data_type == TensorProto.UINT8
    assert packed.data_location == 1
    assert "model.layers.0.mlp.down_proj.MatMul.weight_Q4" not in stored
    assert "model.layers.0.mlp.down_proj.MatMul.weight_scales" not in stored
    down_nodes = [
        node for node in model.graph.node
        if node.op_type == "MatMulNBits" and "down_proj" in node.name
    ]
    assert [node.name for node in down_nodes] == [
        "/model/layers.0/mlp/down_proj/MatMul_Q4/k0",
        "/model/layers.0/mlp/down_proj/MatMul_Q4/k4096",
        "/model/layers.0/mlp/down_proj/MatMul_Q4/k8192",
    ]
    chunk_w = []
    chunk_s = []
    for node in down_nodes:
        attrs = {attr.name: int(attr.i) for attr in node.attribute}
        assert attrs["K"] == 4096
        assert attrs["N"] == 4096
        assert attrs["bits"] == 4
        assert attrs["block_size"] == 32
        assert attrs["accuracy_level"] == 4
        weight = stored[node.input[1]]
        assert list(weight.dims) == [4096, 128, 16]
        assert weight.data_location != 1
        chunk_w.append(numpy_helper.to_array(weight))
        scale_name = node.input[2][: -len("/bf16")]
        scale = stored[scale_name]
        assert list(scale.dims) == [4096, 128]
        assert scale.data_type == TensorProto.FLOAT
        chunk_s.append(numpy_helper.to_array(scale))
    assert np.array_equal(np.concatenate(chunk_w, axis=1), orig_w)
    assert np.array_equal(np.concatenate(chunk_s, axis=1), orig_s)
    down_out = "/model/layers.0/mlp/down_proj/MatMul/output_0"
    producer = next(node for node in model.graph.node if down_out in node.output)
    assert producer.op_type == "Add"
    for node in model.graph.node:
        if node.op_type != "MatMulNBits":
            continue
        weight = stored[node.input[1]]
        assert weight.data_type == TensorProto.UINT8
        assert node.input[2].endswith("/bf16")
        if "down_proj" not in node.name:
            assert weight.data_location == 1
            assert int(next(attr.i for attr in node.attribute if attr.name == "K")) == 4096
    starts = numpy_helper.to_array(
        next(item for item in model.graph.initializer if item.name.endswith("present_slice_starts"))
    )
    ends = numpy_helper.to_array(
        next(item for item in model.graph.initializer if item.name.endswith("present_slice_ends"))
    )
    assert starts.tolist() == [1]
    assert ends.tolist() == [17]
    assert not any(
        node.op_type in {"SkipSimplifiedLayerNormalization", "SimplifiedLayerNormalization"}
        for node in model.graph.node
    )
    assert "/model/layers.0/post_attention_layernorm/SkipLayerNorm" in report["decomposed_norms"]
    bias = static_attention_bias(4, 1, 16)[0, 0, 0]
    assert bias[:4].tolist() == [0, 0, 0, 0]
    assert np.all(bias[4:] == np.float32(-10000))


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


def _session(model):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def test_gqa_slice_keeps_the_newest_window():
    past = 16
    total = past + 1
    heads = 1
    head = 16
    hidden_q = 32
    hidden_kv = 16

    def symbolic():
        q = _value("q", [_dim(0, "batch_size"), _dim(0, "sequence_length"), _dim(hidden_q, "")])
        k = _value("k", [_dim(0, "batch_size"), _dim(0, "sequence_length"), _dim(hidden_kv, "")])
        v = _value("v", [_dim(0, "batch_size"), _dim(0, "sequence_length"), _dim(hidden_kv, "")])
        past_k = _value(
            "past_key_values.0.key",
            [_dim(0, "batch_size"), _dim(heads, ""), _dim(0, "past_sequence_length"), _dim(head, "")],
        )
        past_v = _value(
            "past_key_values.0.value",
            [_dim(0, "batch_size"), _dim(heads, ""), _dim(0, "past_sequence_length"), _dim(head, "")],
        )
        seqlens = _value("seqlens", [_dim(0, "batch_size")], TensorProto.INT32)
        total_in = _value("total", [], TensorProto.INT32)
        present_k = _value(
            "present.0.key",
            [_dim(0, "batch_size"), _dim(heads, ""), _dim(0, "total_sequence_length"), _dim(head, "")],
        )
        present_v = _value(
            "present.0.value",
            [_dim(0, "batch_size"), _dim(heads, ""), _dim(0, "total_sequence_length"), _dim(head, "")],
        )
        context = _value("context", [_dim(0, "batch_size"), _dim(0, "sequence_length"), _dim(hidden_q, "")])
        node = helper.make_node(
            "GroupQueryAttention",
            ["q", "k", "v", "past_key_values.0.key", "past_key_values.0.value", "seqlens", "total", "", "", "", ""],
            ["context", "present.0.key", "present.0.value"],
            name="gqa",
            domain="com.microsoft",
            num_heads=2,
            kv_num_heads=heads,
            scale=float(head) ** -0.5,
            do_rotary=0,
        )
        graph = helper.make_graph(
            [node],
            "gqa_window",
            [q, k, v, past_k, past_v, seqlens, total_in],
            [context, present_k, present_v],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
        )
        model.ir_version = 10
        return model

    reference = symbolic()
    freeze_graph_io(reference, batch_size=1, query_len=1, max_seq_len=past)
    for value in reference.graph.output:
        if value.name.startswith("present."):
            value.type.tensor_type.shape.dim[2].dim_value = total
    rewritten = symbolic()
    report = apply_static_contract(rewritten, batch_size=1, query_len=1, max_seq_len=past)
    assert report["rewritten_attention"] == ["gqa"]
    rng = np.random.default_rng(0)
    feeds = {
        "q": rng.normal(size=(1, 1, hidden_q)).astype(np.float32),
        "k": rng.normal(size=(1, 1, hidden_kv)).astype(np.float32),
        "v": rng.normal(size=(1, 1, hidden_kv)).astype(np.float32),
        "past_key_values.0.key": rng.normal(size=(1, heads, past, head)).astype(np.float32),
        "past_key_values.0.value": rng.normal(size=(1, heads, past, head)).astype(np.float32),
        "seqlens": np.asarray([past], dtype=np.int32),
        "total": np.asarray(total, dtype=np.int32),
    }
    ref_out = _session(reference).run(None, feeds)
    got_feeds = dict(feeds)
    del got_feeds["seqlens"]
    del got_feeds["total"]
    got_feeds["attention_bias"] = np.zeros((1, 1, 1, total), dtype=np.float32)
    got = _session(rewritten).run(None, got_feeds)
    np.testing.assert_allclose(got[0], ref_out[0], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(got[1], ref_out[1][:, :, 1:, :], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(got[2], ref_out[2][:, :, 1:, :], atol=1e-6, rtol=1e-6)
    gqa = next(node for node in rewritten.graph.node if node.op_type == "GroupQueryAttention")
    assert gqa.input[9] == ""
    assert "position_ids" not in {value.name for value in rewritten.graph.input}


def _rotate_half(vec, cos, sin):
    half = vec.shape[-1] // 2
    first, second = vec[..., :half], vec[..., half:]
    out = np.empty_like(vec)
    out[..., :half] = first * cos - second * sin
    out[..., half:] = second * cos + first * sin
    return out


def test_gqa_position_ids_selects_the_rope_row():
    past = 16
    total = past + 1
    heads = 1
    head = 16
    hidden_q = 32
    hidden_kv = 16

    def symbolic():
        q = _value("q", [_dim(0, "batch_size"), _dim(0, "sequence_length"), _dim(hidden_q, "")])
        k = _value("k", [_dim(0, "batch_size"), _dim(0, "sequence_length"), _dim(hidden_kv, "")])
        v = _value("v", [_dim(0, "batch_size"), _dim(0, "sequence_length"), _dim(hidden_kv, "")])
        past_k = _value(
            "past_key_values.0.key",
            [_dim(0, "batch_size"), _dim(heads, ""), _dim(0, "past_sequence_length"), _dim(head, "")],
        )
        past_v = _value(
            "past_key_values.0.value",
            [_dim(0, "batch_size"), _dim(heads, ""), _dim(0, "past_sequence_length"), _dim(head, "")],
        )
        seqlens = _value("seqlens", [_dim(0, "batch_size")], TensorProto.INT32)
        total_in = _value("total", [], TensorProto.INT32)
        cos = _value("cos_cache", [_dim(32, ""), _dim(head // 2, "")])
        sin = _value("sin_cache", [_dim(32, ""), _dim(head // 2, "")])
        present_k = _value(
            "present.0.key",
            [_dim(0, "batch_size"), _dim(heads, ""), _dim(0, "total_sequence_length"), _dim(head, "")],
        )
        present_v = _value(
            "present.0.value",
            [_dim(0, "batch_size"), _dim(heads, ""), _dim(0, "total_sequence_length"), _dim(head, "")],
        )
        context = _value("context", [_dim(0, "batch_size"), _dim(0, "sequence_length"), _dim(hidden_q, "")])
        node = helper.make_node(
            "GroupQueryAttention",
            [
                "q",
                "k",
                "v",
                "past_key_values.0.key",
                "past_key_values.0.value",
                "seqlens",
                "total",
                "cos_cache",
                "sin_cache",
                "",
                "",
            ],
            ["context", "present.0.key", "present.0.value"],
            name="gqa",
            domain="com.microsoft",
            num_heads=2,
            kv_num_heads=heads,
            scale=float(head) ** -0.5,
            do_rotary=1,
            rotary_interleaved=0,
        )
        graph = helper.make_graph(
            [node],
            "gqa_rope",
            [q, k, v, past_k, past_v, seqlens, total_in, cos, sin],
            [context, present_k, present_v],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
        )
        model.ir_version = 10
        return model

    rng = np.random.default_rng(0)
    q = rng.normal(size=(1, 1, hidden_q)).astype(np.float32)
    k = rng.normal(size=(1, 1, hidden_kv)).astype(np.float32)
    v = rng.normal(size=(1, 1, hidden_kv)).astype(np.float32)
    past_k = np.zeros((1, heads, past, head), dtype=np.float32)
    past_v = np.zeros((1, heads, past, head), dtype=np.float32)
    cos = np.zeros((32, head // 2), dtype=np.float32)
    sin = np.zeros((32, head // 2), dtype=np.float32)
    for index in range(32):
        cos[index] = 0.25 + 0.01 * index
        sin[index] = -0.15 + 0.02 * index
    k_heads = k.reshape(1, 1, 1, head)
    frozen = symbolic()
    freeze_graph_io(frozen, batch_size=1, query_len=1, max_seq_len=past)
    for value in frozen.graph.output:
        if value.name.startswith("present."):
            value.type.tensor_type.shape.dim[2].dim_value = total
    frozen_feeds = {
        "q": q,
        "k": k,
        "v": v,
        "past_key_values.0.key": past_k,
        "past_key_values.0.value": past_v,
        "seqlens": np.asarray([past], dtype=np.int32),
        "total": np.asarray(total, dtype=np.int32),
        "cos_cache": cos,
        "sin_cache": sin,
    }
    frozen_key = _session(frozen).run(None, frozen_feeds)[1][0, 0, -1]
    row16 = _rotate_half(k_heads, cos[16], sin[16])[0, 0, 0]
    row0 = _rotate_half(k_heads, cos[0], sin[0])[0, 0, 0]
    assert np.max(np.abs(frozen_key - row16)) < 1e-6
    assert np.max(np.abs(frozen_key - row0)) > 0.5

    rewritten = symbolic()
    report = apply_static_contract(rewritten, batch_size=1, query_len=1, max_seq_len=past)
    assert report["rewritten_attention"] == ["gqa"]
    gqa = next(node for node in rewritten.graph.node if node.op_type == "GroupQueryAttention")
    assert gqa.input[9] == "position_ids"
    assert _int_attr_value(gqa, "do_rotary") == 1
    assert [node.op_type for node in rewritten.graph.node] == ["GroupQueryAttention", "Slice", "Slice"]
    position = next(value for value in rewritten.graph.input if value.name == "position_ids")
    assert position.type.tensor_type.elem_type == TensorProto.INT64
    assert _dims(position) == [1, 1]
    seqlens = numpy_helper.to_array(next(item for item in rewritten.graph.initializer if item.name == gqa.input[5]))
    assert seqlens.tolist() == [past]
    feeds = {
        "q": q,
        "k": k,
        "v": v,
        "past_key_values.0.key": past_k,
        "past_key_values.0.value": past_v,
        "cos_cache": cos,
        "sin_cache": sin,
        "attention_bias": np.zeros((1, 1, 1, total), dtype=np.float32),
    }
    session = _session(rewritten)
    for pos in (0, 1, 8, 16, 20):
        got_feeds = dict(feeds)
        got_feeds["position_ids"] = np.asarray([[pos]], dtype=np.int64)
        got = session.run(None, got_feeds)[1][0, 0, -1]
        expect = _rotate_half(k_heads, cos[pos], sin[pos])[0, 0, 0]
        assert np.max(np.abs(got - expect)) < 1e-6, pos


def _int_attr_value(node, name):
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    raise AssertionError(name)


def test_skip_norm_matches_the_fused_op():
    weight = np.linspace(0.5, 1.5, 32, dtype=np.float32)
    node = helper.make_node(
        "SkipSimplifiedLayerNormalization",
        ["x", "skip", "weight"],
        ["y", "", "", "residual"],
        name="skip_norm",
        domain="com.microsoft",
        epsilon=1e-6,
    )
    graph = helper.make_graph(
        [node],
        "skip_norm",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 32]),
            helper.make_tensor_value_info("skip", TensorProto.FLOAT, [1, 1, 32]),
        ],
        [
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 32]),
            helper.make_tensor_value_info("residual", TensorProto.FLOAT, [1, 1, 32]),
        ],
        [numpy_helper.from_array(weight, "weight")],
    )
    fused = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
    )
    fused.ir_version = 10
    primitive = onnx.ModelProto()
    primitive.CopyFrom(fused)
    assert rewrite_fused_norms(primitive) == ["skip_norm"]
    assert not any(node.op_type == "SkipSimplifiedLayerNormalization" for node in primitive.graph.node)
    source_cast = next(node for node in primitive.graph.node if node.name == "skip_norm/Cast_x")
    inv_cast = next(node for node in primitive.graph.node if node.name == "skip_norm/Cast_inv")
    square = next(node for node in primitive.graph.node if node.name == "skip_norm/Mul_square")
    assert source_cast.attribute[0].i == TensorProto.FLOAT
    assert source_cast.output[0] == square.input[0] == square.input[1]
    assert inv_cast.attribute[0].i == TensorProto.FLOAT
    add_eps = next(node for node in primitive.graph.node if node.name == "skip_norm/Add_epsilon")
    assert add_eps.input[1] in {item.name for item in primitive.graph.initializer}
    assert not any(node.op_type == "ReduceMean" for node in primitive.graph.node)
    assert any(node.op_type == "ReduceSum" for node in primitive.graph.node)
    count = numpy_helper.to_array(
        next(item for item in primitive.graph.initializer if item.name == "skip_norm/rms_count")
    )
    assert float(count) == 32.0
    rng = np.random.default_rng(1)
    feeds = {
        "x": rng.normal(size=(1, 1, 32)).astype(np.float32),
        "skip": rng.normal(size=(1, 1, 32)).astype(np.float32),
    }
    ref = _session(fused).run(None, feeds)
    got = _session(primitive).run(None, feeds)
    np.testing.assert_allclose(got[0], ref[0], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(got[1], ref[1], atol=1e-6, rtol=1e-6)


def _nbits_model(k, n, block, weight, scales, bias=None, g_idx=False, shape=None):
    if shape is None:
        shape = [2, 3, int(k)]
    out_shape = list(shape[:-1]) + [int(n)]
    inputs = ["activation", "weight", "scales"]
    initializers = [
        numpy_helper.from_array(np.ascontiguousarray(weight), "weight"),
        numpy_helper.from_array(np.ascontiguousarray(scales), "scales"),
    ]
    if bias is not None or g_idx:
        inputs.append("")
    if g_idx or bias is not None:
        inputs.append("g_idx" if g_idx else "")
    if bias is not None:
        inputs.append("bias")
        initializers.append(numpy_helper.from_array(np.ascontiguousarray(bias), "bias"))
    node = helper.make_node(
        "MatMulNBits",
        inputs,
        ["output"],
        name="down",
        domain="com.microsoft",
        K=int(k),
        N=int(n),
        bits=4,
        block_size=int(block),
        accuracy_level=4,
    )
    graph = helper.make_graph(
        [node],
        "down",
        [helper.make_tensor_value_info("activation", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, out_shape)],
        initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    return model


def _gap(got, ref):
    delta = float(np.max(np.abs(got.astype(np.float64) - ref.astype(np.float64))))
    scale = max(float(np.max(np.abs(ref))), 1.0)
    return delta, scale


def test_down_proj_k_split_matches_unsplit():
    assert _choose_chunk_k(12288, 32) == 4096
    assert _choose_chunk_k(14336, 128) == 3584
    assert _choose_chunk_k(4096, 32) is None
    assert _choose_chunk_k(12288, 32, 3584) == 3072
    rng = np.random.default_rng(2)
    weight = rng.integers(0, 256, size=(4, 4, 16), dtype=np.uint8)
    scales = (rng.normal(size=(4, 4)) * 0.02).astype(np.float32)
    bias = rng.normal(size=(4,)).astype(np.float32)
    blocked = _nbits_model(128, 4, 32, weight, scales, g_idx=True)
    assert _split_large_matmul_nbits(blocked, ".", chunk_k=64) == []
    assert sum(node.op_type == "MatMulNBits" for node in blocked.graph.node) == 1
    model = _nbits_model(128, 4, 32, weight, scales, bias=bias)
    held = onnx.ModelProto()
    held.CopyFrom(model)
    assert _split_large_matmul_nbits(held, None, chunk_k=64) == []
    feed = rng.normal(size=(2, 3, 128)).astype(np.float32)
    ref = _session(model).run(None, {"activation": feed})[0]
    assert _split_large_matmul_nbits(model, ".", chunk_k=64) == ["down"]
    pieces = [node for node in model.graph.node if node.op_type == "MatMulNBits"]
    assert len(pieces) == 2
    assert sum(node.input[5] == "bias" for node in pieces) == 1
    got = _session(model).run(None, {"activation": feed})[0]
    delta, scale = _gap(got, ref)
    assert np.isfinite(got).all()
    assert delta <= 1e-4 * scale, f"synthetic gap {delta} scale {scale}"
    path = os.environ.get(
        "QWEN_ATTENTION_LAYER",
        r"C:\Users\antmi\qwen3-work\qwen3-8b-split\model.onnx\layer_0_of_36.onnx",
    )
    source = onnx.load(path, load_external_data=False)
    directory = os.path.dirname(path)
    original = next(node for node in source.graph.node if node.name.endswith("down_proj/MatMul_Q4"))
    stored = {item.name: item for item in source.graph.initializer}
    real_w = _initializer_array(stored[original.input[1]], directory)
    real_s = _initializer_array(stored[original.input[2]], directory)
    real = _nbits_model(12288, 4096, 32, real_w, real_s, shape=[1, 1, 12288])
    real_feed = rng.normal(size=(1, 1, 12288)).astype(np.float32)
    real_ref = _session(real).run(None, {"activation": real_feed})[0]
    assert _split_large_matmul_nbits(real, directory) == ["down"]
    real_got = _session(real).run(None, {"activation": real_feed})[0]
    real_delta, real_scale = _gap(real_got, real_ref)
    assert np.isfinite(real_got).all()
    assert real_delta <= 1e-4 * real_scale, f"down_proj gap {real_delta} scale {real_scale}"
    print(f"  down_proj_gap {real_delta:.3e} ref_abs {real_scale:.3e}")


def _dequant_pair(weight, scales, zp_packed, k, n, block):
    """Float32 ``[K, N]`` and the same product with zero point 0."""
    n_blocks = int(k) // int(block)
    flat = np.ascontiguousarray(weight, dtype=np.uint8).reshape(int(n), -1)
    nib = np.empty((int(n), flat.shape[1] * 2), dtype=np.float32)
    nib[:, 0::2] = flat & np.uint8(0x0F)
    nib[:, 1::2] = (flat >> np.uint8(4)) & np.uint8(0x0F)
    nibbles = np.ascontiguousarray(nib[:, : int(k)].T)
    if zp_packed is None:
        zp = np.full((n_blocks, int(n)), 8.0, dtype=np.float32)
    else:
        zflat = np.ascontiguousarray(zp_packed, dtype=np.uint8).reshape(int(n), -1)
        packed = np.empty((int(n), zflat.shape[1] * 2), dtype=np.float32)
        packed[:, 0::2] = zflat & np.uint8(0x0F)
        packed[:, 1::2] = (zflat >> np.uint8(4)) & np.uint8(0x0F)
        zp = np.ascontiguousarray(packed[:, :n_blocks].T)
    scale_full = np.repeat(np.ascontiguousarray(scales, dtype=np.float32).T, int(block), axis=0)
    zp_full = np.repeat(zp, int(block), axis=0)
    return (nibbles - zp_full) * scale_full, nibbles * scale_full


def _bf16_matrix(raw, shape):
    bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    return (bits << np.uint32(16)).view(np.float32).reshape(shape).copy()


def _nbits_zp_model(k, n, block, weight, scales, zp):
    shape = [2, 3, int(k)]
    node = helper.make_node(
        "MatMulNBits",
        ["activation", "weight", "scales", "zp"],
        ["output"],
        name="down",
        domain="com.microsoft",
        K=int(k),
        N=int(n),
        bits=4,
        block_size=int(block),
    )
    graph = helper.make_graph(
        [node],
        "down",
        [helper.make_tensor_value_info("activation", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3, int(n)])],
        [
            numpy_helper.from_array(np.ascontiguousarray(weight), "weight"),
            numpy_helper.from_array(np.ascontiguousarray(scales), "scales"),
            numpy_helper.from_array(np.ascontiguousarray(zp), "zp"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    return model


def _assert_baked_matches_ort(model, feed, exact, zp0, bias=None):
    ref = _session(model).run(None, {"activation": feed})[0]
    got_exact = np.matmul(feed, exact)
    got_zp0 = np.matmul(feed, zp0)
    if bias is not None:
        got_exact = got_exact + bias
        got_zp0 = got_zp0 + bias
    np.testing.assert_allclose(got_exact, ref, atol=1e-5, rtol=1e-5)
    baked = onnx.ModelProto()
    baked.CopyFrom(model)
    report = apply_static_contract(
        baked,
        batch_size=1,
        query_len=1,
        max_seq_len=16,
        npu_bf16_fallback=True,
    )
    assert report["bf16_fallback"] is True
    assert report["activation"] == "float32"
    assert report["bf16_fallback_matmuls"] == ["down"]
    assert not any(node.op_type in {"MatMulNBits", "DequantizeLinear"} for node in baked.graph.node)
    assert baked.graph.output[0].type.tensor_type.elem_type == int(TensorProto.FLOAT)
    item = next(tensor for tensor in baked.graph.initializer if tensor.name == "down/bf16_weight")
    assert list(item.dims) == [exact.shape[0], exact.shape[1]]
    assert int(item.data_type) == int(TensorProto.BFLOAT16)
    assert bytes(item.raw_data) == _f32_to_bf16_raw(exact)
    decoded = _bf16_matrix(bytes(item.raw_data), tuple(exact.shape))
    got = np.matmul(feed, decoded)
    if bias is not None:
        got = got + bias
    gap, scale = _gap(got, ref)
    wrong, _ = _gap(got_zp0, ref)
    assert gap < wrong
    assert gap <= 5e-2 * scale, f"baked gap {gap} scale {scale} zp0 {wrong}"
    print(f"  baked_gap {gap:.3e} zp0_gap {wrong:.3e} ref_abs {scale:.3e}")


def test_bf16_fallback_matches_matmul_nbits_and_stays_off():
    assert inspect.signature(split_onnx).parameters["npu_bf16_fallback"].default is False
    splitter_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "chunking_utils",
        "onnx",
        "splitter.py",
    )
    source = open(splitter_path, encoding="utf-8").read()
    assert "npu_bf16_fallback=bool(npu_bf16_fallback)" in source
    assert "npu_bf16_fallback=True" not in source
    assert "npu_bf16_fallback = True" not in source
    assert "defer_matmul_bake=True" in source
    assert "npu_bf16_fallback=False" in source
    assert inspect.signature(split_onnx).parameters["only_components"].default is None
    assert inspect.signature(split_onnx).parameters["load_external_data"].default is True
    rng = np.random.default_rng(7)
    k, n, block = 32, 4, 32
    weight = rng.integers(0, 256, size=(n, 1, 16), dtype=np.uint8)
    scales = (rng.normal(size=(n, 1)) * 0.05).astype(np.float32)
    bias = rng.normal(size=(n,)).astype(np.float32)
    # accuracy_level 4 selects the int8 MatMulNBits kernel. The bake stores
    # the float weight (nibble - zp) * scale, which is what the kernel does
    # when that attribute is omitted.
    omitted = _nbits_model(k, n, block, weight, scales, bias=bias)
    for node in omitted.graph.node:
        kept = [attr for attr in node.attribute if attr.name != "accuracy_level"]
        del node.attribute[:]
        node.attribute.extend(kept)
    parked = onnx.ModelProto()
    parked.CopyFrom(omitted)
    off = apply_static_contract(parked, batch_size=1, query_len=1, max_seq_len=16)
    assert off["bf16_fallback"] is False
    assert off["matmul_bake_deferred"] is False
    assert off["bf16_fallback_matmuls"] == []
    assert off["activation"] == "bfloat16"
    assert sum(node.op_type == "MatMulNBits" for node in parked.graph.node) == 1
    assert parked.graph.output[0].type.tensor_type.elem_type == int(TensorProto.BFLOAT16)
    deferred = onnx.ModelProto()
    deferred.CopyFrom(omitted)
    deferred_report = apply_static_contract(
        deferred,
        batch_size=1,
        query_len=1,
        max_seq_len=16,
        defer_matmul_bake=True,
    )
    assert deferred_report["matmul_bake_deferred"] is True
    assert deferred_report["bf16_fallback"] is False
    assert deferred_report["activation"] == "float32"
    assert deferred_report["bf16_fallback_matmuls"] == []
    assert sum(node.op_type == "MatMulNBits" for node in deferred.graph.node) == 1
    assert deferred.graph.output[0].type.tensor_type.elem_type == int(TensorProto.FLOAT)
    try:
        apply_static_contract(
            deferred,
            batch_size=1,
            query_len=1,
            max_seq_len=16,
            npu_bf16_fallback=True,
            defer_matmul_bake=True,
        )
    except StaticShapeError as exc:
        assert "defer_matmul_bake" in str(exc)
    else:
        raise AssertionError("defer_matmul_bake combined with baking")
    exact, zp0 = _dequant_pair(weight, scales, None, k, n, block)
    feed = rng.normal(size=(2, 3, k)).astype(np.float32)
    _assert_baked_matches_ort(omitted, feed, exact, zp0, bias=bias)
    packed_k, packed_n = 64, 2
    packed_weight = rng.integers(0, 256, size=(packed_n, 2, 16), dtype=np.uint8)
    packed_scales = (rng.normal(size=(packed_n, 2)) * 0.05).astype(np.float32)
    zp = np.asarray([[3 | (9 << 4)], [1 | (12 << 4)]], dtype=np.uint8)
    packed = _nbits_zp_model(packed_k, packed_n, 32, packed_weight, packed_scales, zp)
    packed_exact, packed_zp0 = _dequant_pair(packed_weight, packed_scales, zp, packed_k, packed_n, 32)
    packed_feed = rng.normal(size=(2, 3, packed_k)).astype(np.float32)
    _assert_baked_matches_ort(packed, packed_feed, packed_exact, packed_zp0)
    blocked = _nbits_model(
        128,
        4,
        32,
        rng.integers(0, 256, size=(4, 4, 16), dtype=np.uint8),
        np.zeros((4, 4), dtype=np.float32),
        g_idx=True,
    )
    try:
        apply_static_contract(
            blocked,
            batch_size=1,
            query_len=1,
            max_seq_len=16,
            npu_bf16_fallback=True,
        )
    except StaticShapeError as exc:
        assert "g_idx" in str(exc)
    else:
        raise AssertionError("g_idx MatMulNBits was baked")


def test_component_bake_does_not_expand_the_master():
    import tempfile

    from chunking_utils.onnx.splitter import save_components
    from chunking_utils.onnx.types import ComponentInfo
    from chunking_utils.onnx.weight_store import WeightStrategy

    rng = np.random.default_rng(3)
    k, n, block = 32, 4, 32
    nodes = []
    initializers = []
    inputs = []
    outputs = []
    for index, name in enumerate(("down0", "down1")):
        weight = rng.integers(0, 256, size=(n, 1, 16), dtype=np.uint8)
        scales = (rng.normal(size=(n, 1)) * 0.05).astype(np.float32)
        activation = f"activation_{index}"
        result = f"output_{index}"
        nodes.append(helper.make_node(
            "MatMulNBits",
            [activation, f"weight_{index}", f"scales_{index}"],
            [result],
            name=name,
            domain="com.microsoft",
            K=int(k),
            N=int(n),
            bits=4,
            block_size=int(block),
        ))
        inputs.append(helper.make_tensor_value_info(activation, TensorProto.FLOAT, [1, 1, int(k)]))
        outputs.append(helper.make_tensor_value_info(result, TensorProto.FLOAT, [1, 1, int(n)]))
        initializers.append(numpy_helper.from_array(np.ascontiguousarray(weight), f"weight_{index}"))
        initializers.append(numpy_helper.from_array(np.ascontiguousarray(scales), f"scales_{index}"))
    model = helper.make_model(
        helper.make_graph(nodes, "two", inputs, outputs, initializers),
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    layer = ComponentInfo("layer", 0)
    layer.node_indices = [0]
    tail = ComponentInfo("other", None)
    tail.node_indices = [1]
    before = [node.op_type for node in model.graph.node]
    with tempfile.TemporaryDirectory() as tmp:
        files, _storage, layers = save_components(
            model,
            {"layer_0": layer, "tail": tail},
            [],
            tmp,
            strategy=WeightStrategy.COPY,
            size_threshold=1024 * 1024,
            chunk_size=None,
            npu_bf16_fallback=True,
            only_components=["layer_0"],
            apply_component_contract=True,
            static_batch_size=1,
            static_max_seq_len=16,
            static_query_len=1,
        )
        assert layers == 1
        assert set(files) == {"layer_0"}
        saved = onnx.load(files["layer_0"], load_external_data=True)
        assert any(node.op_type == "MatMul" for node in saved.graph.node)
        assert not any(node.op_type == "MatMulNBits" for node in saved.graph.node)
        assert not os.path.exists(os.path.join(tmp, "tail.onnx"))
    assert [node.op_type for node in model.graph.node] == before == ["MatMulNBits", "MatMulNBits"]


def test_component_bake_copies_master_external_slices():
    import tempfile

    from chunking_utils.onnx.splitter import _materialize_external_initializers, save_components
    from chunking_utils.onnx.types import ComponentInfo
    from chunking_utils.onnx.weight_store import WeightStrategy

    rng = np.random.default_rng(4)
    k, n, block = 32, 4, 32
    weight = rng.integers(0, 256, size=(n, 1, 16), dtype=np.uint8)
    scales = (rng.normal(size=(n, 1)) * 0.05).astype(np.float32)
    bias = rng.normal(size=(64,)).astype(np.float32)
    nodes = [
        helper.make_node(
            "MatMulNBits",
            ["activation", "weight_0", "scales_0"],
            ["hidden"],
            name="down0",
            domain="com.microsoft",
            K=int(k),
            N=int(n),
            bits=4,
            block_size=int(block),
        ),
        helper.make_node("Add", ["hidden", "bias"], ["output_0"], name="bias_add"),
    ]
    bias_init = numpy_helper.from_array(np.ascontiguousarray(bias), "bias")
    bias_init.ClearField("raw_data")
    bias_init.data_location = TensorProto.EXTERNAL
    for key, value in (
        ("location", "model.onnx.data"),
        ("offset", "0"),
        ("length", str(int(bias.nbytes))),
    ):
        entry = bias_init.external_data.add()
        entry.key = key
        entry.value = value
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "external",
            [helper.make_tensor_value_info("activation", TensorProto.FLOAT, [1, 1, int(k)])],
            [helper.make_tensor_value_info("output_0", TensorProto.FLOAT, [1, 1, int(n)])],
            [
                numpy_helper.from_array(np.ascontiguousarray(weight), "weight_0"),
                numpy_helper.from_array(np.ascontiguousarray(scales), "scales_0"),
                bias_init,
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    layer = ComponentInfo("layer", 0)
    layer.node_indices = [0, 1]
    with tempfile.TemporaryDirectory() as tmp:
        master = os.path.join(tmp, "master")
        out = os.path.join(tmp, "out")
        os.makedirs(master)
        with open(os.path.join(master, "model.onnx.data"), "wb") as handle:
            handle.write(bias.tobytes())
        unsafe = onnx.ModelProto()
        unsafe.CopyFrom(model)
        unsafe.graph.initializer[2].external_data[0].value = "../model.onnx.data"
        try:
            _materialize_external_initializers(unsafe, master)
        except ValueError as exc:
            assert "unsafe" in str(exc)
        else:
            raise AssertionError("parent external location was accepted")
        files, _storage, layers = save_components(
            model,
            {"layer_0": layer},
            [],
            out,
            strategy=WeightStrategy.COPY,
            size_threshold=16,
            chunk_size=None,
            npu_bf16_fallback=True,
            external_data_dir=master,
            only_components=["layer_0"],
            apply_component_contract=False,
            static_batch_size=1,
            static_max_seq_len=16,
            static_query_len=1,
        )
        assert layers == 1
        saved_path = files["layer_0"]
        shutil = __import__("shutil")
        shutil.rmtree(master)
        saved = onnx.load(saved_path, load_external_data=True)
        locations = []
        found = None
        for initializer in saved.graph.initializer:
            if initializer.data_location == TensorProto.EXTERNAL:
                locations.extend(entry.value for entry in initializer.external_data if entry.key == "location")
            if initializer.name == "bias":
                found = numpy_helper.to_array(initializer)
        assert "model.onnx.data" not in locations
        assert found is not None
        assert np.allclose(found, bias)
    master_bias = model.graph.initializer[2]
    assert master_bias.data_location == TensorProto.EXTERNAL
    assert master_bias.external_data[0].value == "model.onnx.data"


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
        ("qwen_layer", test_qwen_layer_freezes_to_the_static_window),
        ("down_split", test_down_proj_k_split_matches_unsplit),
        ("bf16_fallback", test_bf16_fallback_matches_matmul_nbits_and_stays_off),
        ("component_bake", test_component_bake_does_not_expand_the_master),
        ("external_slice", test_component_bake_copies_master_external_slices),
        ("gqa_slice", test_gqa_slice_keeps_the_newest_window),
        ("gqa_position", test_gqa_position_ids_selects_the_rope_row),
        ("skip_norm", test_skip_norm_matches_the_fused_op),
        ("builder_static_mask", test_builder_static_mask_has_no_shape_or_slice),
    ]
    return all(_check(name, fn) for name, fn in checks)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
