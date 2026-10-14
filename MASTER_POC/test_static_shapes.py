"""Static NPU shapes: frozen axes, a full KV window, and an additive tail."""

import ast
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from chunking_utils.onnx.static_shapes import (
    StaticShapeError,
    _choose_chunk_k,
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
    assert gqa_inputs[10] == "attention_bias/bf16"
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
        ("gqa_slice", test_gqa_slice_keeps_the_newest_window),
        ("skip_norm", test_skip_norm_matches_the_fused_op),
        ("builder_static_mask", test_builder_static_mask_has_no_shape_or_slice),
    ]
    return all(_check(name, fn) for name, fn in checks)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
