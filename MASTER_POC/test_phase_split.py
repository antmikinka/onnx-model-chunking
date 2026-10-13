"""Prefill and decode are two static graphs. The host routes between them."""

import os
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from pipeline_utils.phase_graphs import (
    assert_no_control_flow,
    build_phase_models,
    external_weight_key,
)
from pipeline_utils.session_manager import (
    SessionConfig,
    SessionManager,
    select_execution_phase,
    select_prefill_device,
)


def _session(path):
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def test_select_phase_ignores_cache_values():
    assert select_execution_phase(8, 0) == "prefill"
    assert select_execution_phase(8, 8) == "prefill"
    assert select_execution_phase(1, 0) == "prefill"
    assert select_execution_phase(1, 5) == "decode"
    assert select_execution_phase(1, 5, explicit="prefill") == "prefill"
    try:
        select_execution_phase(4, 5, explicit="decode")
        raise AssertionError("a multi-token decode was accepted")
    except ValueError as exc:
        assert "Decode" in str(exc)
    try:
        select_execution_phase(0, 0)
        raise AssertionError("an empty query was accepted")
    except ValueError as exc:
        assert "query_len" in str(exc)


def test_phase_graphs_share_weights_and_run():
    hidden = 4
    prompt = 3
    window = 6
    weight = (np.arange(hidden * hidden, dtype=np.float32).reshape(hidden, hidden) + 0.25)
    directory = tempfile.mkdtemp(prefix="phase-split-")
    info = build_phase_models(
        directory,
        hidden=hidden,
        prefill_query_len=prompt,
        max_seq_len=window,
        weight=weight,
        prefill_device="npu",
        decode_device="igpu",
    )
    prefill_path = os.path.join(directory, "prefill.onnx")
    decode_path = os.path.join(directory, "decode.onnx")
    prefill_model = onnx.load(prefill_path, load_external_data=False)
    decode_model = onnx.load(decode_path, load_external_data=False)
    assert_no_control_flow(prefill_model)
    assert_no_control_flow(decode_model)
    assert external_weight_key(prefill_model) == external_weight_key(decode_model)
    location, offset, length = external_weight_key(prefill_model)
    assert location == "weights.bin"
    assert offset == 0
    assert length == weight.nbytes
    blob = open(os.path.join(directory, "weights.bin"), "rb").read()
    assert blob == weight.tobytes()
    assert weight.tobytes() not in open(prefill_path, "rb").read()
    assert weight.tobytes() not in open(decode_path, "rb").read()

    tokens = np.arange(prompt * hidden, dtype=np.float32).reshape(1, 1, prompt, hidden)
    prefill_out = _session(prefill_path).run(None, {"hidden": tokens})[0]
    projected = tokens @ weight
    expected = np.concatenate(
        [np.zeros((1, 1, window - prompt, hidden), dtype=np.float32), projected],
        axis=2,
    )
    assert prefill_out.shape == (1, 1, window, hidden)
    assert np.allclose(prefill_out, expected)

    step = np.arange(hidden, dtype=np.float32).reshape(1, 1, 1, hidden) + 3
    decode_out = _session(decode_path).run(None, {"hidden": step, "past": prefill_out})[0]
    new = step @ weight
    slid = np.concatenate([prefill_out[:, :, 1:, :], new], axis=2)
    assert decode_out.shape == (1, 1, window, hidden)
    assert np.allclose(decode_out, slid)
    assert info["phase_split"]["decode_query_len"] == 1
    assert info["phase_split"]["shared_weights"] == "weights.bin"


def test_control_flow_is_rejected():
    branch = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    then_out = helper.make_tensor_value_info("then_y", TensorProto.FLOAT, [1])
    else_out = helper.make_tensor_value_info("else_y", TensorProto.FLOAT, [1])
    then_graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["then_y"])],
        "then",
        [],
        [then_out],
    )
    else_graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["else_y"])],
        "else",
        [],
        [else_out],
    )
    node = helper.make_node("If", ["cond"], ["y"], then_branch=then_graph, else_branch=else_graph)
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    graph = helper.make_graph([node], "branched", [cond, branch], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    try:
        assert_no_control_flow(model)
        raise AssertionError("an If phase graph was accepted")
    except ValueError as exc:
        assert "If" in str(exc)


def test_session_manager_routes_phases():
    directory = tempfile.mkdtemp(prefix="phase-route-")
    hidden = 4
    weight = np.eye(hidden, dtype=np.float32)
    info = build_phase_models(
        directory,
        hidden=hidden,
        prefill_query_len=4,
        max_seq_len=8,
        weight=weight,
    )
    manager = SessionManager(
        directory,
        SessionConfig(providers=["CPUExecutionProvider"]),
        model_info=info,
        available_providers=["CPUExecutionProvider"],
    )
    prefill = manager.route(4, 0)
    assert prefill.phase == "prefill"
    assert prefill.components == ("prefill",)
    assert prefill.device == "npu"
    assert prefill.static_query_len == 4
    assert prefill.session_loaded
    chunk = manager.route(2, 4)
    assert chunk.phase == "prefill"
    assert chunk.past_len == 4
    decode = manager.route(1, 4)
    assert decode.phase == "decode"
    assert decode.components == ("decode",)
    assert decode.device == "igpu"
    assert decode.static_query_len == 1
    assert decode.session_loaded
    session = manager.get_session("decode")
    token = np.ones((1, 1, 1, hidden), dtype=np.float32)
    past = np.zeros((1, 1, 8, hidden), dtype=np.float32)
    past[:, :, -1, :] = 2
    present = session.run(None, {"hidden": token, "past": past})[0]
    assert present.shape[-2] == 8
    assert np.allclose(present[:, :, -1, :], token)
    assert np.allclose(present[:, :, -2, :], past[:, :, -1, :])
    try:
        manager.route(9, 0)
        raise AssertionError("a prompt past the prefill bucket was accepted")
    except ValueError as exc:
        assert "bucket" in str(exc)

    forced = SessionManager(
        directory,
        SessionConfig(
            providers=["RyzenAIExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"],
            execution_policy="phase_split",
        ),
        model_info=info,
        available_providers=[
            "RyzenAIExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    npu_decode = dict(info)
    npu_decode["components"] = {
        name: dict(component) for name, component in info["components"].items()
    }
    npu_decode["components"]["decode"]["device_preference"] = "npu"
    npu_decode["phase_split"] = dict(info["phase_split"], decode_device="npu")
    resident = SessionManager(
        directory,
        SessionConfig(providers=["CPUExecutionProvider"]),
        model_info=npu_decode,
        available_providers=["CPUExecutionProvider"],
    )
    stayed = resident.route(1, 2)
    assert stayed.phase == "decode"
    assert stayed.device == "npu"
    assert forced.route(1, 2).provider == "DmlExecutionProvider"


def test_layer_series_is_not_a_phase_graph():
    info = {
        "components": {
            "layer_0": {
                "file": "layer_0.onnx",
                "phase": "prefill",
                "input_names": [],
                "output_names": [],
            },
            "layer_1": {
                "file": "layer_1.onnx",
                "phase": "prefill",
                "input_names": [],
                "output_names": [],
            },
        },
        "phase_split": {
            "prefill_component": ["layer_0", "layer_1"],
            "decode_component": "decode",
        },
    }
    info["components"]["decode"] = {
        "file": "decode.onnx",
        "phase": "decode",
        "input_names": [],
        "output_names": [],
    }
    manager = SessionManager(
        ".",
        SessionConfig(providers=["CPUExecutionProvider"]),
        model_info=info,
    )
    try:
        manager.route(4, 0)
        raise AssertionError("a layer series was routed as prefill")
    except RuntimeError as exc:
        assert "layer split" in str(exc)


def test_prefill_attention_matches_reference():
    from onnx import TensorProto

    from chunking_utils.onnx.golden_attention_subgraph import default_projection
    from chunking_utils.onnx.prefill_attention import (
        PREFILL_OPS,
        build_prefill_attention_model,
        causal_bias_f16,
        default_prompt,
        prefill_attention_facts,
        reference_prefill_attention,
    )

    prompt, head = 8, 64
    model = build_prefill_attention_model(prompt_len=prompt, head_dim=head)
    facts = prefill_attention_facts(model)
    assert facts["ops"] == list(PREFILL_OPS)
    assert facts["ir_version"] == 10
    assert facts["bias_dtype"] == TensorProto.FLOAT16
    assert "Transpose" not in facts["ops"] and "Where" not in facts["ops"]
    hidden = default_prompt(prompt, head)
    weights = [default_projection(seed, head, head) for seed in (1, 2, 3, 4)]
    bias = causal_bias_f16(prompt)
    expect_k, expect_v, expect_h = reference_prefill_attention(hidden, *weights, bias)
    plain = np.zeros_like(bias)
    plain_h = reference_prefill_attention(hidden, *weights, plain)[2]
    assert np.max(np.abs(expect_h - plain_h)) > 1e-3
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
    )
    key, value, projected = session.run(None, {"hidden_states": np.ascontiguousarray(hidden)})
    assert np.max(np.abs(key - expect_k)) < 1e-5
    assert np.max(np.abs(value - expect_v)) < 1e-5
    assert np.max(np.abs(projected - expect_h)) < 1e-5


def test_prefill_device_uses_measured_layout_only():
    assert select_prefill_device(255, 8, 128) == "cpu"
    assert select_prefill_device(256, 8, 128) == "npu"
    assert select_prefill_device(512, 8, 128) == "npu"
    assert select_prefill_device(512, 1, 64) == "unmeasured"
    assert select_execution_phase(1, 5) == "decode"


def test_multihead_handoff_matches_reference():
    from chunking_utils.onnx.phase_handoff import (
        reference_prefill_decode_handoff,
        run_prefill_decode_handoff,
    )

    got = run_prefill_decode_handoff(
        prompt_len=8,
        head_dim=128,
        steps=5,
        num_heads=8,
        providers=["CPUExecutionProvider"],
    )
    expect = reference_prefill_decode_handoff(
        prompt_len=8, head_dim=128, steps=5, num_heads=8
    )
    assert all(got["ptr_moved"])
    assert np.max(np.abs(got["prefill_hidden"] - expect["prefill_hidden"])) < 1e-4
    for left, right in zip(got["decode_hidden"], expect["decode_hidden"]):
        assert np.max(np.abs(left - right)) < 1e-4
    assert np.max(np.abs(got["final_key"] - expect["final_key"])) < 1e-4
    assert np.max(np.abs(got["final_value"] - expect["final_value"])) < 1e-4


def test_multihead_prefill_matches_reference():
    from onnx import TensorProto

    from chunking_utils.onnx.golden_attention_subgraph import default_projection
    from chunking_utils.onnx.prefill_attention import (
        MEASURED_NPU_FASTER_PREFILL_PROMPT,
        PREFILL_MULTIHEAD_OPS,
        build_prefill_attention_model,
        causal_bias_f16,
        default_prompt,
        prefill_attention_facts,
        reference_prefill_attention,
    )

    prompt, heads, head = 8, 8, 128
    model = build_prefill_attention_model(prompt_len=prompt, head_dim=head, num_heads=heads)
    facts = prefill_attention_facts(model)
    assert MEASURED_NPU_FASTER_PREFILL_PROMPT == 256
    assert facts["ops"] == list(PREFILL_MULTIHEAD_OPS)
    assert facts["ir_version"] == 10
    assert facts["bias_dtype"] == TensorProto.FLOAT16
    hidden = default_prompt(prompt, head, heads)
    weights = [default_projection(seed, head, head) for seed in (1, 2, 3, 4)]
    bias = causal_bias_f16(prompt)
    expect_k, expect_v, expect_h = reference_prefill_attention(hidden, *weights, bias)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
    )
    key, value, projected = session.run(None, {"hidden_states": np.ascontiguousarray(hidden)})
    assert np.max(np.abs(key - expect_k)) < 1e-5
    assert np.max(np.abs(value - expect_v)) < 1e-5
    assert np.max(np.abs(projected - expect_h)) < 1e-5


def test_prefill_decode_handoff_matches_reference():
    from chunking_utils.onnx.phase_handoff import (
        reference_prefill_decode_handoff,
        run_prefill_decode_handoff,
    )

    got = run_prefill_decode_handoff(
        prompt_len=8,
        head_dim=64,
        steps=5,
        providers=["CPUExecutionProvider"],
    )
    expect = reference_prefill_decode_handoff(prompt_len=8, head_dim=64, steps=5)
    assert len(got["decode_hidden"]) == 5
    assert all(got["ptr_moved"])
    assert np.max(np.abs(got["prefill_hidden"] - expect["prefill_hidden"])) < 1e-4
    for left, right in zip(got["decode_hidden"], expect["decode_hidden"]):
        assert np.max(np.abs(left - right)) < 1e-4
    assert np.max(np.abs(got["final_key"] - expect["final_key"])) < 1e-4
    assert np.max(np.abs(got["final_value"] - expect["final_value"])) < 1e-4


def main():
    tests = [
        test_select_phase_ignores_cache_values,
        test_phase_graphs_share_weights_and_run,
        test_control_flow_is_rejected,
        test_session_manager_routes_phases,
        test_layer_series_is_not_a_phase_graph,
        test_prefill_attention_matches_reference,
        test_prefill_device_uses_measured_layout_only,
        test_multihead_handoff_matches_reference,
        test_multihead_prefill_matches_reference,
        test_prefill_decode_handoff_matches_reference,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: ok")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
