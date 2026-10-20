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
    decompose_gqa_to_npu_primitives,
    freeze_graph_io,
    rewrite_fused_norms,
    right_aligned_attention_bias,
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


def _make_external_tensor(name, data_type, dims, location, offset, length):
    tensor = TensorProto()
    tensor.name = name
    tensor.data_type = data_type
    tensor.dims.extend(dims)
    tensor.data_location = TensorProto.EXTERNAL
    for key, val in (
        ("location", str(location)),
        ("offset", str(offset)),
        ("length", str(length)),
    ):
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = val
    return tensor


def _build_synthetic_qwen_layer(directory):
    rng = np.random.default_rng(7)
    down_w = rng.integers(0, 256, size=(4096, 384, 16), dtype=np.uint8)
    down_s = rng.normal(size=(4096, 384)).astype(np.float32)
    data_file = "layer_0_of_36.onnx.data"
    data_path = os.path.join(directory, data_file)
    with open(data_path, "wb") as handle:
        handle.write(down_w.tobytes())
        s_offset = handle.tell()
        handle.write(down_s.tobytes())

    initializers = [
        numpy_helper.from_array(np.ones((4096,), dtype=np.float32), "model.layers.0.input_layernorm.weight"),
        numpy_helper.from_array(np.ones((128,), dtype=np.float32), "model.layers.0.attn.q_norm.layernorm.weight"),
        numpy_helper.from_array(np.ones((128,), dtype=np.float32), "model.layers.0.attn.k_norm.layernorm.weight"),
        numpy_helper.from_array(np.ones((4096,), dtype=np.float32), "model.layers.0.post_attention_layernorm.weight"),
        numpy_helper.from_array(np.ones((64, 64), dtype=np.float32), "cos_cache"),
        numpy_helper.from_array(np.zeros((64, 64), dtype=np.float32), "sin_cache"),
    ]
    proj_specs = [
        ("attn/qkv_proj", "model.layers.0.attn.qkv_proj.MatMul", 4096, 6144, "norm0_out", "qkv_out"),
        ("attn/o_proj", "model.layers.0.attn.o_proj.MatMul", 4096, 4096, "gqa_out", "o_out"),
        ("mlp/gate_proj", "model.layers.0.mlp.gate_proj.MatMul", 4096, 12288, "post_norm_out", "gate_out"),
        ("mlp/up_proj", "model.layers.0.mlp.up_proj.MatMul", 4096, 12288, "post_norm_out", "up_out"),
    ]
    nodes = [
        helper.make_node(
            "SimplifiedLayerNormalization",
            ["/model/embed_tokens/Gather/output_0", "model.layers.0.input_layernorm.weight"],
            ["norm0_out"],
            name="/model/layers.0/input_layernorm/LayerNorm",
            axis=-1,
            epsilon=1e-6,
        ),
    ]
    for suffix, prefix, k_dim, n_dim, inp_name, out_name in proj_specs:
        blocks = k_dim // 32
        w_name = f"{prefix}.weight_Q4"
        s_name = f"{prefix}.weight_scales"
        initializers.append(
            _make_external_tensor(w_name, TensorProto.UINT8, [n_dim, blocks, 16], data_file, 0, n_dim * blocks * 16)
        )
        initializers.append(
            _make_external_tensor(s_name, TensorProto.FLOAT, [n_dim, blocks], data_file, 0, n_dim * blocks * 4)
        )
        nodes.append(
            helper.make_node(
                "MatMulNBits",
                [inp_name, w_name, s_name],
                [out_name],
                name=f"/model/layers.0/{suffix}/MatMul_Q4",
                domain="com.microsoft",
                K=k_dim,
                N=n_dim,
                bits=4,
                block_size=32,
                accuracy_level=4,
            )
        )
    initializers.append(
        _make_external_tensor(
            "model.layers.0.mlp.down_proj.MatMul.weight_Q4",
            TensorProto.UINT8,
            [4096, 384, 16],
            data_file,
            0,
            int(down_w.nbytes),
        )
    )
    initializers.append(
        _make_external_tensor(
            "model.layers.0.mlp.down_proj.MatMul.weight_scales",
            TensorProto.FLOAT,
            [4096, 384],
            data_file,
            s_offset,
            int(down_s.nbytes),
        )
    )
    nodes.extend([
        helper.make_node(
            "SimplifiedLayerNormalization",
            ["qkv_out", "model.layers.0.attn.q_norm.layernorm.weight"],
            ["q_norm_out"],
            name="/model/layers.0/attn/q_norm/LayerNorm",
            axis=-1,
            epsilon=1e-6,
        ),
        helper.make_node(
            "SimplifiedLayerNormalization",
            ["qkv_out", "model.layers.0.attn.k_norm.layernorm.weight"],
            ["k_norm_out"],
            name="/model/layers.0/attn/k_norm/LayerNorm",
            axis=-1,
            epsilon=1e-6,
        ),
        helper.make_node(
            "GroupQueryAttention",
            [
                "q_norm_out",
                "k_norm_out",
                "qkv_out",
                "past_key_values.0.key",
                "past_key_values.0.value",
                "/model/attn_mask_reformat/attn_mask_subgraph/Sub/Cast/output_0",
                "/model/attn_mask_reformat/attn_mask_subgraph/Gather/Cast/output_0",
                "cos_cache",
                "sin_cache",
                "",
                "attention_bias",
            ],
            ["gqa_out", "present.0.key", "present.0.value"],
            name="/model/layers.0/attn/GroupQueryAttention",
            domain="com.microsoft",
            num_heads=32,
            kv_num_heads=8,
            do_rotary=1,
            past_present_share_buffer=0,
        ),
        helper.make_node(
            "SkipSimplifiedLayerNormalization",
            ["/model/embed_tokens/Gather/output_0", "o_out", "model.layers.0.post_attention_layernorm.weight"],
            ["post_norm_out", "", "", ""],
            name="/model/layers.0/post_attention_layernorm/SkipLayerNorm",
            domain="com.microsoft",
            epsilon=1e-6,
        ),
        helper.make_node("Mul", ["gate_out", "up_out"], ["mlp_hidden"], name="/model/layers.0/mlp/Mul"),
        helper.make_node(
            "MatMulNBits",
            [
                "mlp_hidden",
                "model.layers.0.mlp.down_proj.MatMul.weight_Q4",
                "model.layers.0.mlp.down_proj.MatMul.weight_scales",
            ],
            ["/model/layers.0/mlp/down_proj/MatMul/output_0"],
            name="/model/layers.0/mlp/down_proj/MatMul_Q4",
            domain="com.microsoft",
            K=12288,
            N=4096,
            bits=4,
            block_size=32,
            accuracy_level=4,
        ),
    ])
    inputs = [
        _value("/model/embed_tokens/Gather/output_0", [_dim(0, "batch_size"), _dim(0, "sequence_length"), _dim(4096, "")]),
        _value("past_key_values.0.key", [_dim(0, "batch_size"), _dim(8, ""), _dim(0, "past_sequence_length"), _dim(128, "")]),
        _value("past_key_values.0.value", [_dim(0, "batch_size"), _dim(8, ""), _dim(0, "past_sequence_length"), _dim(128, "")]),
        _value("/model/attn_mask_reformat/attn_mask_subgraph/Sub/Cast/output_0", [_dim(0, "batch_size")], TensorProto.INT32),
        _value("/model/attn_mask_reformat/attn_mask_subgraph/Gather/Cast/output_0", [_dim(1, "")], TensorProto.INT32),
        _value("attention_bias", [_dim(0, "batch_size"), _dim(1, ""), _dim(0, "sequence_length"), _dim(0, "total_sequence_length")]),
    ]
    outputs = [
        _value("/model/layers.0/mlp/down_proj/MatMul/output_0", [_dim(0, "batch_size"), _dim(0, "sequence_length"), _dim(4096, "")]),
        _value("present.0.key", [_dim(0, "batch_size"), _dim(8, ""), _dim(0, "total_sequence_length"), _dim(128, "")]),
        _value("present.0.value", [_dim(0, "batch_size"), _dim(8, ""), _dim(0, "total_sequence_length"), _dim(128, "")]),
    ]
    model = helper.make_model(
        helper.make_graph(nodes, "qwen_layer_0", inputs, outputs, initializers),
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    model_path = os.path.join(directory, "layer_0_of_36.onnx")
    onnx.save_model(model, model_path, save_as_external_data=False)
    return model_path


def test_qwen_layer_freezes_to_the_static_window():
    import tempfile

    path = os.environ.get(
        "QWEN_ATTENTION_LAYER",
        r"C:\Users\antmi\qwen3-work\qwen3-8b-split\model.onnx\layer_0_of_36.onnx",
    )
    tmp_ctx = None
    if not os.path.exists(path):
        tmp_ctx = tempfile.TemporaryDirectory()
        path = _build_synthetic_qwen_layer(tmp_ctx.name)
    try:
        _run_qwen_layer_freeze_assertions(path)
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def _run_qwen_layer_freeze_assertions(path):
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
    import tempfile

    path = os.environ.get(
        "QWEN_ATTENTION_LAYER",
        r"C:\Users\antmi\qwen3-work\qwen3-8b-split\model.onnx\layer_0_of_36.onnx",
    )
    tmp_ctx = None
    if not os.path.exists(path):
        tmp_ctx = tempfile.TemporaryDirectory()
        path = _build_synthetic_qwen_layer(tmp_ctx.name)
    try:
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
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


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


def test_decompose_gqa_matches_fused_and_avoids_hostile_ops():
    rng = np.random.default_rng(42)
    batch_size = 1
    query_len = 1
    max_seq_len = 8
    num_heads = 4
    kv_num_heads = 2
    head_dim = 16
    q_hidden = num_heads * head_dim
    kv_hidden = kv_num_heads * head_dim
    cos_data = rng.normal(size=(64, head_dim // 2)).astype(np.float32)
    sin_data = rng.normal(size=(64, head_dim // 2)).astype(np.float32)

    for share_buffer in (0, 1):
        past_seq = max_seq_len + query_len if share_buffer else max_seq_len
        q_in = _value("q", [_dim(0, "batch_size"), _dim(0, "sequence_length"), _dim(q_hidden, "")])
        k_in = _value("k", [_dim(0, "batch_size"), _dim(0, "sequence_length"), _dim(kv_hidden, "")])
        v_in = _value("v", [_dim(0, "batch_size"), _dim(0, "sequence_length"), _dim(kv_hidden, "")])
        pk_in = _value(
            "past_key_values.0.key",
            [_dim(0, "batch_size"), _dim(kv_num_heads, ""), _dim(0, "past_sequence_length"), _dim(head_dim, "")],
        )
        pv_in = _value(
            "past_key_values.0.value",
            [_dim(0, "batch_size"), _dim(kv_num_heads, ""), _dim(0, "past_sequence_length"), _dim(head_dim, "")],
        )
        seqlens_in = _value("seqlens_k", [_dim(0, "batch_size")], TensorProto.INT32)
        total_in = _value("total_seq_len", [_dim(1, "")], TensorProto.INT32)
        bias_in = _value(
            "attention_bias",
            [_dim(0, "batch_size"), _dim(1, ""), _dim(0, "sequence_length"), _dim(0, "total_sequence_length")],
        )
        out_val = _value("attn_out", [_dim(0, "batch_size"), _dim(0, "sequence_length"), _dim(q_hidden, "")])
        pres_k = _value(
            "present.0.key",
            [_dim(0, "batch_size"), _dim(kv_num_heads, ""), _dim(0, "total_sequence_length"), _dim(head_dim, "")],
        )
        pres_v = _value(
            "present.0.value",
            [_dim(0, "batch_size"), _dim(kv_num_heads, ""), _dim(0, "total_sequence_length"), _dim(head_dim, "")],
        )
        gqa_kwargs = dict(
            name="/model/layers.0/attn/GroupQueryAttention",
            domain="com.microsoft",
            num_heads=num_heads,
            kv_num_heads=kv_num_heads,
            do_rotary=1,
        )
        if share_buffer:
            gqa_kwargs["past_present_share_buffer"] = 1
        gqa_node = helper.make_node(
            "GroupQueryAttention",
            [
                "q",
                "k",
                "v",
                "past_key_values.0.key",
                "past_key_values.0.value",
                "seqlens_k",
                "total_seq_len",
                "cos_cache",
                "sin_cache",
                "",
                "attention_bias",
            ],
            ["attn_out", "present.0.key", "present.0.value"],
            **gqa_kwargs,
        )
        inits = [
            numpy_helper.from_array(cos_data, "cos_cache"),
            numpy_helper.from_array(sin_data, "sin_cache"),
        ]
        base_model = helper.make_model(
            helper.make_graph(
                [gqa_node],
                "gqa_test",
                [q_in, k_in, v_in, pk_in, pv_in, seqlens_in, total_in, bias_in],
                [out_val, pres_k, pres_v],
                inits,
            ),
            opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
        )
        base_model.ir_version = 10

        decomp_model = onnx.ModelProto()
        decomp_model.CopyFrom(base_model)
        report = apply_static_contract(
            decomp_model,
            batch_size=batch_size,
            query_len=query_len,
            max_seq_len=max_seq_len,
            decompose_gqa=True,
        )
        assert report["rewritten_attention"] == ["/model/layers.0/attn/GroupQueryAttention"]
        assert report["decomposed_gqa"] is True
        ops = {node.op_type for node in decomp_model.graph.node}
        assert "GroupQueryAttention" not in ops
        for hostile in ("Transpose", "Where", "Scatter", "ScatterND", "ScatterElements"):
            assert hostile not in ops, f"hostile op {hostile} found in decomposed GQA"

        decomp_sess = _session(decomp_model)
        q_arr = rng.normal(size=(1, 1, q_hidden)).astype(np.float32)
        k_arr = rng.normal(size=(1, 1, kv_hidden)).astype(np.float32)
        v_arr = rng.normal(size=(1, 1, kv_hidden)).astype(np.float32)
        pk_arr = rng.normal(size=(1, kv_num_heads, max_seq_len, head_dim)).astype(np.float32)
        pv_arr = rng.normal(size=(1, kv_num_heads, max_seq_len, head_dim)).astype(np.float32)

        if not share_buffer:
            fused_model = onnx.ModelProto()
            fused_model.CopyFrom(base_model)
            apply_static_contract(
                fused_model,
                batch_size=batch_size,
                query_len=query_len,
                max_seq_len=max_seq_len,
                decompose_gqa=False,
            )
            fused_sess = _session(fused_model)
            for step_pos, valid_past in ((0, 0), (5, 5), (42, max_seq_len)):
                bias_arr = right_aligned_attention_bias(valid_past, max_seq_len, query_len=1)
                feeds = {
                    "q": q_arr,
                    "k": k_arr,
                    "v": v_arr,
                    "past_key_values.0.key": pk_arr,
                    "past_key_values.0.value": pv_arr,
                    "position_ids": np.asarray([[step_pos]], dtype=np.int64),
                    "attention_bias": bias_arr,
                }
                f_out, f_pk, f_pv = fused_sess.run(None, feeds)
                d_out, d_pk, d_pv = decomp_sess.run(None, feeds)
                assert np.max(np.abs(f_out - d_out)) < 1e-5
                assert np.max(np.abs(f_pk - d_pk)) < 1e-5
                assert np.max(np.abs(f_pv - d_pv)) < 1e-5
        else:
            # In-place shared buffer mode (past_seq == max_seq_len, total_len == max_seq_len):
            # Compare against fused GroupQueryAttention on the kept (max_seq_len - 1) past window
            # plus the new token (total_len == max_seq_len).
            fused_shared_model = onnx.ModelProto()
            fused_shared_model.CopyFrom(base_model)
            for node in fused_shared_model.graph.node:
                if node.op_type == "GroupQueryAttention":
                    kept_attrs = [a for a in node.attribute if a.name != "past_present_share_buffer"]
                    del node.attribute[:]
                    node.attribute.extend(kept_attrs)
            apply_static_contract(
                fused_shared_model,
                batch_size=batch_size,
                query_len=query_len,
                max_seq_len=max_seq_len - 1,
                decompose_gqa=False,
            )
            fused_shared_sess = _session(fused_shared_model)
            pk_kept = np.ascontiguousarray(pk_arr[:, :, 1:, :])
            pv_kept = np.ascontiguousarray(pv_arr[:, :, 1:, :])
            for step_pos, valid_past in ((0, 0), (5, 5), (42, max_seq_len - 1)):
                bias_arr = right_aligned_attention_bias(
                    valid_past, max_seq_len - 1, query_len=1, total_len=max_seq_len
                )
                d_out, d_pk, d_pv = decomp_sess.run(
                    None,
                    {
                        "q": q_arr,
                        "k": k_arr,
                        "v": v_arr,
                        "past_key_values.0.key": pk_arr,
                        "past_key_values.0.value": pv_arr,
                        "position_ids": np.asarray([[step_pos]], dtype=np.int64),
                        "attention_bias": bias_arr,
                    },
                )
                f_out, f_pk, f_pv = fused_shared_sess.run(
                    None,
                    {
                        "q": q_arr,
                        "k": k_arr,
                        "v": v_arr,
                        "past_key_values.0.key": pk_kept,
                        "past_key_values.0.value": pv_kept,
                        "position_ids": np.asarray([[step_pos]], dtype=np.int64),
                        "attention_bias": bias_arr,
                    },
                )
                assert np.max(np.abs(f_out - d_out)) < 1e-5
                assert np.max(np.abs(f_pk - d_pk[:, :, 1:, :])) < 1e-5
                assert np.max(np.abs(f_pv - d_pv[:, :, 1:, :])) < 1e-5
                np.testing.assert_allclose(d_pk[:, :, :-1, :], pk_kept, atol=1e-6)
                np.testing.assert_allclose(d_pv[:, :, :-1, :], pv_kept, atol=1e-6)

    # Also verify sink_size > 0 preserves the leading sink slots in present KV
    sink_model = onnx.ModelProto()
    sink_model.CopyFrom(base_model)
    sink_report = apply_static_contract(
        sink_model,
        batch_size=1,
        query_len=1,
        max_seq_len=max_seq_len,
        decompose_gqa=True,
        sink_size=2,
    )
    assert sink_report["rewritten_attention"] == ["/model/layers.0/attn/GroupQueryAttention"]
    assert sink_report["decomposed_gqa"] is True
    assert sink_report["sink_size"] == 2
    sink_sess = _session(sink_model)
    pk_sink = np.arange(1 * kv_num_heads * max_seq_len * head_dim, dtype=np.float32).reshape(
        1, kv_num_heads, max_seq_len, head_dim
    )
    pv_sink = pk_sink * 0.5
    _, out_pk, out_pv = sink_sess.run(
        None,
        {
            "q": q_arr,
            "k": k_arr,
            "v": v_arr,
            "past_key_values.0.key": pk_sink,
            "past_key_values.0.value": pv_sink,
            "position_ids": np.asarray([[20]], dtype=np.int64),
            "attention_bias": right_aligned_attention_bias(
                max_seq_len, max_seq_len, query_len=1, total_len=max_seq_len, sink_size=2
            ),
        },
    )
    np.testing.assert_allclose(out_pk[:, :, :2, :], pk_sink[:, :, :2, :])
    np.testing.assert_allclose(out_pk[:, :, 2:-1, :], pk_sink[:, :, 3:, :])
    np.testing.assert_allclose(out_pv[:, :, :2, :], pv_sink[:, :, :2, :])
    np.testing.assert_allclose(out_pv[:, :, 2:-1, :], pv_sink[:, :, 3:, :])


def test_right_aligned_attention_bias_and_sink_window():
    from chunking_utils.onnx.kv_concat import static_kv_concat_recipe

    bias0 = right_aligned_attention_bias(0, 8, query_len=1)
    assert bias0.shape == (1, 1, 1, 9)
    assert np.all(bias0[0, 0, 0, :8] == np.float32(-10000.0))
    assert bias0[0, 0, 0, 8] == np.float32(0.0)

    bias3 = right_aligned_attention_bias(3, 8, query_len=1)
    assert np.all(bias3[0, 0, 0, :5] == np.float32(-10000.0))
    assert np.all(bias3[0, 0, 0, 5:] == np.float32(0.0))

    bias_sink = right_aligned_attention_bias(4, 8, query_len=1, sink_size=2)
    assert np.all(bias_sink[0, 0, 0, :2] == np.float32(0.0))
    assert np.all(bias_sink[0, 0, 0, 2:6] == np.float32(-10000.0))
    assert np.all(bias_sink[0, 0, 0, 6:] == np.float32(0.0))

    recipe = static_kv_concat_recipe(max_seq_len=8, query_len=1, layout="sink_window", sink_size=2)
    assert recipe["layout"] == "sink_window"
    assert recipe["output_seq"] == 8
    assert recipe["present_shape_is_static"] is True
    assert recipe["op_types"] == ["Slice", "Slice", "Concat"]
    assert recipe["nodes"][0]["starts"] == [0]
    assert recipe["nodes"][0]["ends"] == [2]
    assert recipe["nodes"][1]["starts"] == [3]
    assert recipe["nodes"][1]["ends"] == [8]


def test_kv_cache_rolling_swap_position_ids_and_bucket_resize():
    from pipeline_utils.kv_cache import KVCacheManager, KVUpdateMode

    mgr = KVCacheManager(
        num_layers=2,
        batch_size=1,
        num_kv_heads=2,
        max_seq_len=4,
        head_dim=8,
        dtype=np.float32,
        update_mode=KVUpdateMode.CONCAT_SWAP,
        sink_size=1,
        bucket_sizes=(4, 8, 16),
    )
    for step in range(10):
        pos = mgr.position_ids(query_len=1, layer_idx=0)
        assert int(pos[0, 0]) == step
        bias = mgr.attention_bias(query_len=1, align="right", layer_idx=0)
        assert bias.shape == (1, 1, 1, 5)
        for layer_idx in range(2):
            pk, pv = mgr.get_past_tensors(layer_idx)
            sk = mgr._cache[layer_idx]["key_shadow"]
            sv = mgr._cache[layer_idx]["value_shadow"]
            sk[:, :, :-1, :] = pk[:, :, 1:, :]
            sk[:, :, -1, :] = float(step + 1)
            sv[:, :, :-1, :] = pv[:, :, 1:, :]
            sv[:, :, -1, :] = float((step + 1) * 10)
            mgr.commit_present_swap(layer_idx, step + 1, total_seen=step + 1, rolling=True)

    assert mgr.seen_tokens == 10
    assert mgr.current_seq_len == 4
    # First token (1.0) was promoted to sink slot 0 when window filled at step 4
    k0, _ = mgr.get_past_tensors(0)
    assert float(k0[0, 0, 0, 0]) == 1.0
    assert float(k0[0, 0, -1, 0]) == 10.0

    # Dynamically resize static window from 4 -> 8 via bucket capacity
    new_cap = mgr.ensure_bucket_capacity(6, align="right")
    assert new_cap == 8
    assert mgr.max_seq_len == 8
    resized_k0, _ = mgr.get_past_tensors(0)
    assert resized_k0.shape == (1, 2, 8, 8)
    assert float(resized_k0[0, 0, -1, 0]) == 10.0


def test_split_inference_engine_static_window_decode():
    import json
    import tempfile
    from chunking_utils.onnx.inference import InferenceConfig, SplitModelInferenceEngine

    with tempfile.TemporaryDirectory() as tmp:
        vocab_size = 16
        hidden_dim = 8
        num_heads = 2
        kv_heads = 2
        head_dim = 4
        max_seq = 4
        rng = np.random.default_rng(99)

        # 1. Embeddings component
        emb_weight = rng.normal(size=(vocab_size, hidden_dim)).astype(np.float32)
        emb_graph = helper.make_graph(
            [helper.make_node("Gather", ["emb_w", "input_ids"], ["hidden_0"], axis=0)],
            "embeddings",
            [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 1])],
            [helper.make_tensor_value_info("hidden_0", TensorProto.FLOAT, [1, 1, hidden_dim])],
            [numpy_helper.from_array(emb_weight, "emb_w")],
        )
        emb_model = helper.make_model(emb_graph, opset_imports=[helper.make_opsetid("", 18)])
        emb_model.ir_version = 10
        onnx.save_model(emb_model, os.path.join(tmp, "embeddings.onnx"))

        # 2. Layer 0 component (decomposed GQA static window)
        cos_data = np.ones((32, head_dim // 2), dtype=np.float32)
        sin_data = np.zeros((32, head_dim // 2), dtype=np.float32)
        gqa_node = helper.make_node(
            "GroupQueryAttention",
            [
                "hidden_0",
                "hidden_0",
                "hidden_0",
                "past_key_values.0.key",
                "past_key_values.0.value",
                "seqlens_k",
                "total_seq_len",
                "cos_cache",
                "sin_cache",
                "",
                "attention_bias",
            ],
            ["hidden_1", "present.0.key", "present.0.value"],
            name="/model/layers.0/attn/GroupQueryAttention",
            domain="com.microsoft",
            num_heads=num_heads,
            kv_num_heads=kv_heads,
            do_rotary=1,
            past_present_share_buffer=0,
        )
        layer_graph = helper.make_graph(
            [gqa_node],
            "layer_0",
            [
                _value("hidden_0", [_dim(0, "batch_size"), _dim(0, "sequence_length"), _dim(hidden_dim, "")]),
                _value("past_key_values.0.key", [_dim(0, "batch_size"), _dim(kv_heads, ""), _dim(0, "past_sequence_length"), _dim(head_dim, "")]),
                _value("past_key_values.0.value", [_dim(0, "batch_size"), _dim(kv_heads, ""), _dim(0, "past_sequence_length"), _dim(head_dim, "")]),
                _value("seqlens_k", [_dim(0, "batch_size")], TensorProto.INT32),
                _value("total_seq_len", [_dim(1, "")], TensorProto.INT32),
                _value("attention_bias", [_dim(0, "batch_size"), _dim(1, ""), _dim(0, "sequence_length"), _dim(0, "total_sequence_length")]),
            ],
            [
                _value("hidden_1", [_dim(0, "batch_size"), _dim(0, "sequence_length"), _dim(hidden_dim, "")]),
                _value("present.0.key", [_dim(0, "batch_size"), _dim(kv_heads, ""), _dim(0, "total_sequence_length"), _dim(head_dim, "")]),
                _value("present.0.value", [_dim(0, "batch_size"), _dim(kv_heads, ""), _dim(0, "total_sequence_length"), _dim(head_dim, "")]),
            ],
            [
                numpy_helper.from_array(cos_data, "cos_cache"),
                numpy_helper.from_array(sin_data, "sin_cache"),
            ],
        )
        layer_model = helper.make_model(
            layer_graph,
            opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
        )
        layer_model.ir_version = 10
        apply_static_contract(
            layer_model,
            batch_size=1,
            query_len=1,
            max_seq_len=max_seq,
            decompose_gqa=True,
        )
        onnx.save_model(layer_model, os.path.join(tmp, "layer_0_of_1.onnx"))

        # 3. LM Head component
        lm_w = rng.normal(size=(hidden_dim, vocab_size)).astype(np.float32)
        lm_graph = helper.make_graph(
            [helper.make_node("MatMul", ["hidden_1", "lm_w"], ["logits"])],
            "lm_head",
            [helper.make_tensor_value_info("hidden_1", TensorProto.FLOAT, [1, 1, hidden_dim])],
            [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1, vocab_size])],
            [numpy_helper.from_array(lm_w, "lm_w")],
        )
        lm_model = helper.make_model(lm_graph, opset_imports=[helper.make_opsetid("", 18)])
        lm_model.ir_version = 10
        onnx.save_model(lm_model, os.path.join(tmp, "lm_head.onnx"))

        meta = {
            "num_layers": 1,
            "components": {
                "embedding": {
                    "file": "embeddings.onnx",
                    "input_names": ["input_ids"],
                    "output_names": ["hidden_0"],
                },
                "layer_0": {
                    "file": "layer_0_of_1.onnx",
                    "input_names": [item.name for item in layer_model.graph.input],
                    "output_names": [item.name for item in layer_model.graph.output],
                },
                "lm_head": {
                    "file": "lm_head.onnx",
                    "input_names": ["hidden_1"],
                    "output_names": ["logits"],
                },
            },
            "kv_cache_info": {
                "num_layers": 1,
                "num_kv_heads": kv_heads,
                "head_dim": head_dim,
                "dtype": "float32",
                "max_seq_len": max_seq,
            },
        }
        with open(os.path.join(tmp, "model_info.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle)
        with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "num_hidden_layers": 1,
                    "num_attention_heads": num_heads,
                    "num_key_value_heads": kv_heads,
                    "head_dim": head_dim,
                    "hidden_size": hidden_dim,
                    "torch_dtype": "float32",
                },
                handle,
            )

        class _DummyTokenizer:
            eos_token_id = None

            def __call__(self, text, return_tensors="np"):
                return {"input_ids": np.asarray([[1, 2, 3]], dtype=np.int64)}

            def decode(self, ids):
                return " ".join(str(int(x)) for x in ids)

        cfg = InferenceConfig(max_seq_len=max_seq, kv_cache_dtype="float32", temperature=0.0)
        engine = SplitModelInferenceEngine(tmp, config=cfg, verbose=False)
        engine.tokenizer = _DummyTokenizer()
        assert engine._static_window_layers is True
        text = engine.generate("hello world", max_new_tokens=6)
        result = engine.get_result()
        assert len(result.token_ids) == 6
        assert isinstance(text, str) and len(text.split()) == 9
        assert engine.kv_cache.seen_tokens == 8
        assert engine.kv_cache.current_seq_len == max_seq


def test_matmul_sub4096_ksplit_and_ozaki_dual_bf16_compensation():
    rng = np.random.default_rng(2026)
    k_dim = 4096
    n_dim = 64
    block_size = 32
    blocks = k_dim // block_size  # 128 blocks

    packed_w = rng.integers(0, 256, size=(n_dim, blocks, 16), dtype=np.uint8)
    # Use high-precision float32 scales that do not round cleanly to 7-bit bf16 mantissa
    scales = (rng.uniform(0.01, 0.08, size=(n_dim, blocks)).astype(np.float32) * np.float32(1.00390625))
    # Packed 4-bit zero points: shape (N, blocks // 2) = (64, 64)
    packed_zp = rng.integers(0, 256, size=(n_dim, blocks // 2), dtype=np.uint8)
    bias = rng.normal(0.0, 0.1, size=(n_dim,)).astype(np.float32)

    def _make_nbits_model(accuracy_level=None):
        kwargs = {
            "domain": "com.microsoft",
            "K": k_dim,
            "N": n_dim,
            "bits": 4,
            "block_size": block_size,
        }
        if accuracy_level is not None:
            kwargs["accuracy_level"] = int(accuracy_level)
        node = helper.make_node(
            "MatMulNBits",
            ["x", "w", "scales", "zp", "", "bias"],
            ["y"],
            name="/layer/proj/MatMul_Q4",
            **kwargs,
        )
        graph = helper.make_graph(
            [node],
            "nbits_graph",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, k_dim])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, n_dim])],
            [
                numpy_helper.from_array(packed_w, "w"),
                numpy_helper.from_array(scales, "scales"),
                numpy_helper.from_array(packed_zp, "zp"),
                numpy_helper.from_array(bias, "bias"),
            ],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
        )
        model.ir_version = 10
        return model

    # Use float32 activations that are exact in bfloat16 so we isolate weight + accumulation error
    x_raw = rng.normal(0.0, 1.0, size=(1, 1, k_dim)).astype(np.float32)
    x_bf16_exact = ((x_raw.view(np.uint32) & np.uint32(0xFFFF0000))).view(np.float32)

    ref_model = _make_nbits_model()
    ref_out = _session(ref_model).run(None, {"x": x_bf16_exact})[0]

    # 1. Verify sub-4096 K-split (chunk_k=1024) with packed 4-bit zero points matches unsplit in ORT
    ksplit_only_model = _make_nbits_model()
    ksplit_report = apply_static_contract(
        ksplit_only_model,
        batch_size=1,
        query_len=1,
        max_seq_len=16,
        external_data_dir=".",
        npu_bf16_fallback=False,
        defer_matmul_bake=True,
        matmul_chunk_k=1024,
    )
    assert ksplit_report["split_matmuls"] == ["/layer/proj/MatMul_Q4"]
    ksplit_nodes = [n for n in ksplit_only_model.graph.node if n.op_type == "MatMulNBits"]
    assert len(ksplit_nodes) == 4
    ksplit_out = _session(ksplit_only_model).run(None, {"x": x_bf16_exact})[0]
    np.testing.assert_allclose(ksplit_out, ref_out, atol=1e-4, rtol=1e-4)

    # 2. Uncompensated single-bf16 bake at K=4096
    single_bf16_model = _make_nbits_model()
    apply_static_contract(
        single_bf16_model,
        batch_size=1,
        query_len=1,
        max_seq_len=16,
        external_data_dir=".",
        npu_bf16_fallback=True,
        npu_bf16_compensate=False,
    )
    single_inits = {t.name: t for t in single_bf16_model.graph.initializer}
    w_single = _bf16_matrix(bytes(single_inits["/layer/proj/MatMul_Q4/bf16_weight"].raw_data), (k_dim, n_dim))
    single_out = np.matmul(x_bf16_exact, w_single) + bias

    # 3. Sub-4096 K-split (chunk_k=1024) + Ozaki dual-bf16 residual compensation
    comp_model = _make_nbits_model()
    report = apply_static_contract(
        comp_model,
        batch_size=1,
        query_len=1,
        max_seq_len=16,
        external_data_dir=".",
        npu_bf16_fallback=True,
        npu_bf16_compensate=True,
        matmul_chunk_k=1024,
    )
    assert report["split_matmuls"] == ["/layer/proj/MatMul_Q4"]
    # 4 K-chunks * 2 MatMuls per chunk (W_hi + W_lo) = 8 MatMul nodes
    matmul_nodes = [n for n in comp_model.graph.node if n.op_type == "MatMul"]
    assert len(matmul_nodes) == 8
    allowed_ops = {"Cast", "MatMul", "Add", "Slice"}
    assert {n.op_type for n in comp_model.graph.node} <= allowed_ops

    comp_inits = {t.name: t for t in comp_model.graph.initializer}
    partial_sums = []
    for k_idx, k_start in enumerate((0, 1024, 2048, 3072)):
        prefix = f"/layer/proj/MatMul_Q4/k{k_start}"
        w_hi = _bf16_matrix(bytes(comp_inits[f"{prefix}/bf16_weight"].raw_data), (1024, n_dim))
        w_lo = _bf16_matrix(bytes(comp_inits[f"{prefix}/bf16_weight_residual"].raw_data), (1024, n_dim))
        x_slice = x_bf16_exact[:, :, k_start : k_start + 1024]
        chunk_out = np.matmul(x_slice, w_hi).astype(np.float32) + np.matmul(x_slice, w_lo).astype(np.float32)
        if k_idx == 3:
            chunk_out = chunk_out + bias
        partial_sums.append(chunk_out)
    comp_out = (partial_sums[0] + partial_sums[1]) + (partial_sums[2] + partial_sums[3])

    err_single = float(np.max(np.abs(single_out - ref_out)))
    err_comp = float(np.max(np.abs(comp_out - ref_out)))
    assert err_single > 1e-3, f"expected noticeable single-bf16 error, got {err_single}"
    assert err_comp < err_single / 50.0, (
        f"dual-bf16 compensation should reduce error by >50x: single={err_single}, comp={err_comp}"
    )


def _quantize_bfp16ebs8(array: np.ndarray, axis: int) -> np.ndarray:
    """Simulate XDNA2 AIE2P ``bfp16ebs8`` (8-element shared exponent + 8-bit signed mantissa)."""
    arr = np.ascontiguousarray(array, dtype=np.float32)
    moved = np.moveaxis(arr, axis, -1)
    orig_shape = moved.shape
    width = orig_shape[-1]
    assert width % 8 == 0
    blocks = moved.reshape(-1, width // 8, 8)
    max_abs = np.max(np.abs(blocks), axis=-1, keepdims=True)
    # Shared power-of-two scale per 8-element block (7 fractional mantissa bits in [-127, 127])
    with np.errstate(divide="ignore"):
        exp = np.floor(np.log2(np.maximum(max_abs, 1e-38))) - 6.0
    step = np.where(max_abs > 0.0, np.exp2(exp), 1.0)
    mant = np.clip(np.round(blocks / step), -127.0, 127.0)
    dequant = (mant * step).reshape(orig_shape)
    return np.moveaxis(dequant, -1, axis).astype(np.float32)


def test_rmsnorm_equilibration_reduces_bfp16ebs8_swamping_and_qdq_homogenize():
    from chunking_utils.onnx.static_shapes import homogenize_qdq_weights_and_split_k
    from benchmark_npu_decode import _parse_layer_spec

    # 1. Verify _parse_layer_spec helper for zero-recompile hybrid bisect
    assert _parse_layer_spec("", 36) == set()
    assert _parse_layer_spec("none", 36) == set()
    assert _parse_layer_spec("all", 36) == set(range(36))
    assert _parse_layer_spec("0,2,5-7", 36) == {0, 2, 5, 6, 7}

    # 2. Verify RMSNorm-to-MatMul 8-element BFP16 block equilibration (0 extra nodes, 0 extra bytes)
    rng = np.random.default_rng(1234)
    k_dim, n_dim, block_size = 64, 16, 32
    n_blocks = k_dim // block_size
    packed_w = rng.integers(0, 256, size=(n_dim, n_blocks, 16), dtype=np.uint8)
    scales = rng.uniform(0.01, 0.05, size=(n_dim, n_blocks)).astype(np.float32)
    # Create an RMSNorm weight with 10x outlier spikes inside each 8-element block
    gamma = np.ones((k_dim,), dtype=np.float32)
    gamma[0::8] = 12.0

    def _make_norm_nbits_model():
        inits = [
            numpy_helper.from_array(gamma.copy(), name="norm_w"),
            numpy_helper.from_array(packed_w.copy(), name="w_q4"),
            numpy_helper.from_array(scales.copy(), name="w_scales"),
        ]
        nodes = [
            helper.make_node(
                "SimplifiedLayerNormalization",
                ["x", "norm_w"],
                ["norm_out"],
                name="/layer/input_layernorm/LayerNorm",
                axis=-1,
                epsilon=1e-6,
            ),
            helper.make_node(
                "MatMulNBits",
                ["norm_out", "w_q4", "w_scales"],
                ["y"],
                name="/layer/qkv/MatMul_Q4",
                domain="com.microsoft",
                K=k_dim,
                N=n_dim,
                bits=4,
                block_size=block_size,
            ),
        ]
        g = helper.make_graph(
            nodes,
            "norm_nbits_test",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, k_dim])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, n_dim])],
            inits,
        )
        m = helper.make_model(
            g,
            opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
        )
        m.ir_version = 10
        return m

    x_val = rng.normal(0.0, 1.0, size=(1, 1, k_dim)).astype(np.float32)
    ref_m = _make_norm_nbits_model()
    ref_y = _session(ref_m).run(None, {"x": x_val})[0]

    unequil_m = _make_norm_nbits_model()
    apply_static_contract(
        unequil_m,
        batch_size=1,
        query_len=1,
        max_seq_len=16,
        npu_bf16_fallback=True,
        npu_bf16_equilibrate=False,
    )
    equil_m = _make_norm_nbits_model()
    eq_report = apply_static_contract(
        equil_m,
        batch_size=1,
        query_len=1,
        max_seq_len=16,
        npu_bf16_fallback=True,
        npu_bf16_equilibrate=True,
    )
    assert eq_report["bf16_equilibrate"] is True
    # Zero extra nodes and zero extra weight bytes compared to standard bf16 bake
    assert len(equil_m.graph.node) == len(unequil_m.graph.node)

    # Simulate GemmBfp16 (bfp16ebs8 along K) on both models:
    # AIE2P GemmBfp16 converts the input activation vector along K (axis -1)
    # and EACH OUTPUT COLUMN j of W along K (axis 0) into 8-element shared-exponent
    # BFP16 blocks. Because W's shared exponent is per (8-row x 1-col) block while
    # the activation's shared exponent is across the 8 elements of x, absorbing
    # sqrt(gamma) balances both 8-element BFP16 block conversions.
    gamma[:] = 1.0
    gamma[0::8] = 64.0
    ref_m = _make_norm_nbits_model()
    ref_y = _session(ref_m).run(None, {"x": x_val})[0]
    unequil_m = _make_norm_nbits_model()
    apply_static_contract(
        unequil_m,
        batch_size=1,
        query_len=1,
        max_seq_len=16,
        npu_bf16_fallback=True,
        npu_bf16_equilibrate=False,
    )
    equil_m = _make_norm_nbits_model()
    eq_report = apply_static_contract(
        equil_m,
        batch_size=1,
        query_len=1,
        max_seq_len=16,
        npu_bf16_fallback=True,
        npu_bf16_equilibrate=True,
    )
    assert eq_report["bf16_equilibrate"] is True
    assert len(equil_m.graph.node) == len(unequil_m.graph.node)

    def _eval_simulated_bfp16(baked_model):
        inits = {t.name: t for t in baked_model.graph.initializer}
        g_vec = numpy_helper.to_array(inits["norm_w"]).astype(np.float32)
        w_bf16 = _bf16_matrix(bytes(inits["/layer/qkv/MatMul_Q4/bf16_weight"].raw_data), (k_dim, n_dim))
        rms = np.sqrt(np.mean(np.square(x_val), axis=-1, keepdims=True) + 1e-6)
        act = (x_val / rms) * g_vec
        act_bfp = _quantize_bfp16ebs8(act, axis=-1)
        w_bfp = _quantize_bfp16ebs8(w_bf16, axis=0)
        return np.matmul(act_bfp, w_bfp)

    err_unequil = float(np.sqrt(np.mean(np.square(_eval_simulated_bfp16(unequil_m) - ref_y))))
    err_equil = float(np.sqrt(np.mean(np.square(_eval_simulated_bfp16(equil_m) - ref_y))))
    assert err_equil < err_unequil, (
        f"equilibration should reduce bfp16ebs8 RMSE: unequil={err_unequil}, equil={err_equil}"
    )

    # 3. Verify homogenize_qdq_weights_and_split_k converts int8 -> uint8 (+128) and splits K=96 -> 3x32
    k_qdq, n_qdq = 96, 12
    w_i8 = rng.integers(-120, 120, size=(k_qdq, n_qdq), dtype=np.int8)
    w_scale = rng.uniform(0.005, 0.02, size=(n_qdq,)).astype(np.float32)
    w_zp_i8 = np.zeros((n_qdq,), dtype=np.int8)
    act_scale = np.asarray(0.05, dtype=np.float32)
    act_zp_u8 = np.asarray(128, dtype=np.uint8)

    qdq_inits = [
        numpy_helper.from_array(act_scale, name="act_s"),
        numpy_helper.from_array(act_zp_u8, name="act_zp"),
        numpy_helper.from_array(w_i8, name="down_w_i8"),
        numpy_helper.from_array(w_scale, name="down_w_s"),
        numpy_helper.from_array(w_zp_i8, name="down_w_zp"),
    ]
    qdq_nodes = [
        helper.make_node("QuantizeLinear", ["x_in", "act_s", "act_zp"], ["x_q"], name="act_Q"),
        helper.make_node("DequantizeLinear", ["x_q", "act_s", "act_zp"], ["x_dq"], name="act_DQ"),
        helper.make_node(
            "DequantizeLinear",
            ["down_w_i8", "down_w_s", "down_w_zp"],
            ["w_dq"],
            name="weight_DQ",
            axis=1,
        ),
        helper.make_node("MatMul", ["x_dq", "w_dq"], ["y_out"], name="down_proj_MatMul"),
    ]
    qdq_model = helper.make_model(
        helper.make_graph(
            qdq_nodes,
            "qdq_test",
            [helper.make_tensor_value_info("x_in", TensorProto.FLOAT, [1, 1, k_qdq])],
            [helper.make_tensor_value_info("y_out", TensorProto.FLOAT, [1, 1, n_qdq])],
            qdq_inits,
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    qdq_model.ir_version = 10

    x_qdq_feed = rng.uniform(-2.0, 2.0, size=(1, 1, k_qdq)).astype(np.float32)
    qdq_ref = _session(qdq_model).run(None, {"x_in": x_qdq_feed})[0]

    qdq_rewritten = onnx.ModelProto()
    qdq_rewritten.CopyFrom(qdq_model)
    qdq_rep = homogenize_qdq_weights_and_split_k(qdq_rewritten, chunk_k=32)
    assert qdq_rep["converted_u8_weights"] == ["down_w_i8"]
    assert qdq_rep["split_qdq_matmuls"] == ["down_proj_MatMul"]
    # No signed INT8 initializers remain; all 3 split weight chunks are UINT8
    assert not any(int(t.data_type) == int(TensorProto.INT8) for t in qdq_rewritten.graph.initializer)
    assert sum(n.op_type == "MatMul" for n in qdq_rewritten.graph.node) == 3

    qdq_got = _session(qdq_rewritten).run(None, {"x_in": x_qdq_feed})[0]
    np.testing.assert_allclose(qdq_got, qdq_ref, atol=1e-5, rtol=1e-5)


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
        ("decompose_gqa", test_decompose_gqa_matches_fused_and_avoids_hostile_ops),
        ("right_aligned_bias_and_sink", test_right_aligned_attention_bias_and_sink_window),
        ("kv_cache_rolling_and_buckets", test_kv_cache_rolling_swap_position_ids_and_bucket_resize),
        ("split_engine_static_window", test_split_inference_engine_static_window_decode),
        ("matmul_ksplit_and_ozaki_bf16", test_matmul_sub4096_ksplit_and_ozaki_dual_bf16_compensation),
        ("equilibrate_and_qdq_homogenize", test_rmsnorm_equilibration_reduces_bfp16ebs8_swamping_and_qdq_homogenize),
    ]
    return all(_check(name, fn) for name, fn in checks)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
