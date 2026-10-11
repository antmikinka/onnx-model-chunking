"""NPU readiness gaps that do not need a GPU or an NPU."""

import json
import os
import sys
import tempfile
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_io_bound_reuses_device_buffer():
    """IO-bound generation binds each component and passes the device buffer on."""
    from chunking_utils.onnx import inference
    from chunking_utils.onnx.inference import InferenceConfig, InferenceMode, SplitModelInferenceEngine

    class Port:
        def __init__(self, name, type_name):
            self.name = name
            self.type = type_name

    class Buffer:
        def __init__(self, array):
            self.ort_value = array
            self.shape = array.shape
            self._array = array

        def to_numpy(self):
            return self._array

    class Session:
        def __init__(self, runner, inputs, outputs):
            self._runner = runner
            self._inputs = inputs
            self._outputs = outputs

        def get_inputs(self):
            return self._inputs

        def get_outputs(self):
            return self._outputs

        def run(self, output_names, feeds):
            raise AssertionError("session.run")

        def produce(self, output_names, feeds):
            return self._runner(output_names, feeds)

    class Tokenizer:
        eos_token_id = None

        def __call__(self, prompt, return_tensors="np"):
            return {"input_ids": np.array([[4]], dtype=np.int64)}

        def decode(self, ids):
            if isinstance(ids, np.ndarray):
                ids = ids.tolist()
            return ",".join(str(int(item)) for item in ids)

    def embed_run(output_names, feeds):
        return [np.ones((1, 1, 4), dtype=np.float32)]

    def layer_run(output_names, feeds):
        hidden = np.asarray(feeds["hidden"])
        present = np.zeros((1, 1, 1, 4), dtype=np.float16)
        values = {"hidden": hidden, "present.0.key": present, "present.0.value": present.copy()}
        return [values[name] for name in output_names]

    def head_run(output_names, feeds):
        assert hasattr(feeds["hidden"], "shape")
        logits = np.zeros((1, 1, 4), dtype=np.float32)
        logits[0, -1, 3] = 5.0
        return [logits]

    sessions = {
        "embedding": Session(embed_run, [Port("input_ids", "tensor(int64)")], [Port("hidden", "tensor(float)")]),
        "layer_0": Session(
            layer_run,
            [
                Port("hidden", "tensor(float)"),
                Port("past_key_values.0.key", "tensor(float16)"),
                Port("past_key_values.0.value", "tensor(float16)"),
                Port("attention_mask", "tensor(int64)"),
                Port("position_ids", "tensor(int64)"),
            ],
            [
                Port("hidden", "tensor(float)"),
                Port("present.0.key", "tensor(float16)"),
                Port("present.0.value", "tensor(float16)"),
            ],
        ),
        "lm_head": Session(
            head_run,
            [Port("hidden", "tensor(float)")],
            [Port("logits", "tensor(float)")],
        ),
    }

    class Manager:
        def __init__(self, model_dir, config=None, model_info=None):
            self.config = config
            self.sessions = sessions

        def load_all(self):
            return {}

        def get_session(self, name):
            return self.sessions[name]

        def get_memory_usage(self):
            return {"total_mb": 1.0}

        def unload_all(self):
            return 0

    class Binder:
        def __init__(self, device_type="cpu", device_id=0, auto_reallocate=True, max_seq_len=4096):
            self.calls = 0
            self.inputs = []
            self.buffers = []

        def run_with_device_buffers(self, session, inputs, output_names=None):
            self.calls += 1
            self.inputs.append(dict(inputs))
            host = {}
            for name, value in inputs.items():
                host[name] = value.to_numpy() if hasattr(value, "to_numpy") else value
            names = list(output_names or [item.name for item in session.get_outputs()])
            produced = session.produce(names, host)
            result = {}
            for name, value in zip(names, produced):
                if "present" in name:
                    result[name] = value
                    continue
                buffer = Buffer(np.asarray(value))
                self.buffers.append(buffer)
                result[name] = buffer
            return result

        def run_with_binding(self, session, inputs):
            raise AssertionError("host copy binding was used")

        def deallocate_all(self):
            return 0

    original_manager = inference.SessionManager
    original_binding = inference.IOBindingManager
    original_session = inference.ort.InferenceSession
    try:
        inference.SessionManager = Manager
        inference.IOBindingManager = Binder
        inference.ort.InferenceSession = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("InferenceSession")
        )
        with tempfile.TemporaryDirectory() as root:
            info = {
                "num_layers": 1,
                "num_kv_heads": 1,
                "head_dim": 4,
                "components": {
                    "embedding": {
                        "file": "embedding.onnx",
                        "input_names": ["input_ids"],
                        "output_names": ["hidden"],
                    },
                    "layer_0": {
                        "file": "layer_0.onnx",
                        "input_names": [
                            "hidden",
                            "past_key_values.0.key",
                            "past_key_values.0.value",
                            "attention_mask",
                            "position_ids",
                        ],
                        "output_names": ["hidden", "present.0.key", "present.0.value"],
                    },
                    "lm_head": {
                        "file": "lm_head.onnx",
                        "input_names": ["hidden"],
                        "output_names": ["logits"],
                    },
                },
            }
            for name in ("embedding.onnx", "layer_0.onnx", "lm_head.onnx"):
                with open(os.path.join(root, name), "wb") as handle:
                    handle.write(b"onnx")
            with open(os.path.join(root, "model_info.json"), "w", encoding="utf-8") as handle:
                json.dump(info, handle)
            engine = SplitModelInferenceEngine(
                root,
                config=InferenceConfig(mode=InferenceMode.IO_BOUND, max_new_tokens=1),
                verbose=False,
            )
            engine.tokenizer = Tokenizer()
            heard = []
            text = engine.generate("hi", stream_callback=lambda index, token, elapsed: heard.append(token))
            binder = engine.io_manager
            assert binder.calls == 3
            assert binder.inputs[1]["hidden"] is binder.buffers[0]
            assert binder.inputs[2]["hidden"] is binder.buffers[1]
            assert text.endswith("3")
            assert heard == ["3"]
            assert engine.get_result().token_ids == [3]
            engine.cleanup()
    finally:
        inference.SessionManager = original_manager
        inference.IOBindingManager = original_binding
        inference.ort.InferenceSession = original_session


def test_preallocate_fixes_kv_shape():
    """A dynamic cache becomes a fixed buffer and later writes stay in that buffer."""
    from pipeline_utils.kv_cache import KVCacheManager

    cache = KVCacheManager(
        num_layers=1,
        num_kv_heads=2,
        head_dim=4,
        dtype=np.float16,
        batch_size=1,
    )
    assert cache.static is False
    assert cache.preallocate(6) == 6
    assert cache.static is True
    assert cache.max_seq_len == 6
    key = cache._cache[0]["key"]
    value = cache._cache[0]["value"]
    assert key.shape == (1, 2, 6, 4)
    assert value.shape == (1, 2, 6, 4)
    present = np.ones((1, 2, 3, 4), dtype=np.float16)
    cache.update_from_present(0, present, present.copy())
    assert cache._cache[0]["key"] is key
    assert cache._cache[0]["value"] is value
    assert key.shape == (1, 2, 6, 4)
    assert np.all(key[:, :, :3, :] == 1)
    assert cache.get_sequence_length() == 3


def test_phase_providers_differ():
    """Prefill and decode use different providers. The router stays off the expert provider."""
    from chunking_utils.common.types import ExecutionPlan as CommonPlan
    from chunking_utils.common.types import ExecutionPolicy, ExecutionStep
    from chunking_utils.onnx import inference
    from chunking_utils.onnx.inference import InferenceConfig, SplitModelInferenceEngine
    from pipeline_utils.session_manager import ExecutionPlan as SessionPlan

    assert CommonPlan is not SessionPlan
    plan = CommonPlan(
        steps=[
            ExecutionStep(
                component_name="backbone_prefill",
                phase="prefill",
                device="npu",
                provider="VitisAIExecutionProvider",
            ),
            ExecutionStep(
                component_name="backbone_decode",
                phase="decode",
                device="igpu",
                provider="DmlExecutionProvider",
            ),
            ExecutionStep(
                component_name="moe_router",
                phase="router",
                device="cpu",
                provider="CPUExecutionProvider",
            ),
            ExecutionStep(
                component_name="moe_expert",
                phase="expert",
                device="npu",
                provider="VitisAIExecutionProvider",
            ),
        ],
        policy=ExecutionPolicy.PHASE_SPLIT,
        prefill_device="npu",
        decode_device="igpu",
    )
    opened = []

    class Port:
        def __init__(self, name, type_name):
            self.name = name
            self.type = type_name

    class Session:
        def __init__(self, kind):
            self.kind = kind

        def get_inputs(self):
            if self.kind == "layer_0":
                return [
                    Port("hidden", "tensor(float)"),
                    Port("past_key_values.0.key", "tensor(float16)"),
                    Port("past_key_values.0.value", "tensor(float16)"),
                    Port("attention_mask", "tensor(int64)"),
                    Port("position_ids", "tensor(int64)"),
                ]
            if self.kind == "embedding":
                return [Port("input_ids", "tensor(int64)")]
            return [Port("hidden", "tensor(float)")]

        def get_outputs(self):
            if self.kind == "layer_0":
                return [
                    Port("hidden", "tensor(float)"),
                    Port("present.0.key", "tensor(float16)"),
                    Port("present.0.value", "tensor(float16)"),
                ]
            if self.kind == "lm_head":
                return [Port("logits", "tensor(float)")]
            return [Port("hidden", "tensor(float)")]

        def run(self, output_names, feeds):
            if self.kind == "embedding":
                seq = int(np.asarray(feeds["input_ids"]).shape[-1])
                return [np.ones((1, seq, 4), dtype=np.float32)]
            if self.kind == "layer_0":
                seq = int(np.asarray(feeds["attention_mask"]).shape[-1])
                hidden = np.ones((1, int(np.asarray(feeds["hidden"]).shape[1]), 4), dtype=np.float32)
                present = np.zeros((1, 1, seq, 4), dtype=np.float16)
                values = {
                    "hidden": hidden,
                    "present.0.key": present,
                    "present.0.value": present.copy(),
                }
                return [values[name] for name in output_names]
            if self.kind == "lm_head":
                seq = int(np.asarray(feeds["hidden"]).shape[1])
                logits = np.zeros((1, seq, 4), dtype=np.float32)
                logits[0, -1, 1] = 4.0
                return [logits]
            hidden = np.asarray(feeds["hidden"])
            return [hidden]

    def open_session(path, sess_options=None, providers=None):
        kind = os.path.basename(path).replace(".onnx", "")
        opened.append((kind, list(providers)))
        return Session(kind)

    class Tokenizer:
        eos_token_id = None

        def __call__(self, prompt, return_tensors="np"):
            return {"input_ids": np.array([[8, 9]], dtype=np.int64)}

        def decode(self, ids):
            return "ok"

    original = inference.ort.InferenceSession
    try:
        inference.ort.InferenceSession = open_session
        with tempfile.TemporaryDirectory() as root:
            components = {}
            for name, inputs, outputs in (
                ("embedding", ["input_ids"], ["hidden"]),
                (
                    "layer_0",
                    [
                        "hidden",
                        "past_key_values.0.key",
                        "past_key_values.0.value",
                        "attention_mask",
                        "position_ids",
                    ],
                    ["hidden", "present.0.key", "present.0.value"],
                ),
                ("moe_router", ["hidden"], ["hidden"]),
                ("moe_expert", ["hidden"], ["hidden"]),
                ("lm_head", ["hidden"], ["logits"]),
            ):
                components[name] = {
                    "file": name + ".onnx",
                    "input_names": inputs,
                    "output_names": outputs,
                }
                with open(os.path.join(root, name + ".onnx"), "wb") as handle:
                    handle.write(b"onnx")
            with open(os.path.join(root, "model_info.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {"num_layers": 1, "num_kv_heads": 1, "head_dim": 4, "components": components},
                    handle,
                )
            engine = SplitModelInferenceEngine(
                root,
                config=InferenceConfig(execution_plan=plan, max_new_tokens=2),
                verbose=False,
            )
            engine.tokenizer = Tokenizer()
            engine.generate("hi")
            trace = engine._provider_trace
            prefill = {name: providers for name, phase, providers in trace if phase == "prefill"}
            decode = {name: providers for name, phase, providers in trace if phase == "decode"}
            assert prefill["layer_0"][0] == "VitisAIExecutionProvider"
            assert decode["layer_0"][0] == "DmlExecutionProvider"
            assert prefill["layer_0"] != decode["layer_0"]
            assert prefill["moe_router"] == ["CPUExecutionProvider"]
            assert decode["moe_router"] == ["CPUExecutionProvider"]
            assert prefill["moe_expert"][0] == "VitisAIExecutionProvider"
            assert decode["moe_expert"][0] == "VitisAIExecutionProvider"
            assert opened[0][1][0] == "VitisAIExecutionProvider"
            engine.cleanup()
    finally:
        inference.ort.InferenceSession = original


def test_quantization_keeps_group_and_zero_point():
    """The quantization record keeps group size and a zero point from the graph."""
    from chunking_utils.onnx.quantization import (
        build_quantization_info,
        extract_quantization_info,
        extract_quantization_tensors,
    )

    class Attr:
        def __init__(self, name, value):
            self.name = name
            self.type = 2
            self.i = value

    class Node:
        def __init__(self):
            self.name = "q_proj"
            self.op_type = "MatMulNBits"
            self.domain = "com.microsoft"
            self.input = ["hidden", "q_weight", "q_scale", "q_zp"]
            self.attribute = [Attr("bits", 4), Attr("block_size", 32)]

    class Init:
        def __init__(self, name, values):
            self.name = name
            self.values = values
            self.data_type = 2
            self.dims = [len(values)]

    class Graph:
        def __init__(self):
            self.node = [Node()]
            self.initializer = [Init("q_scale", [0.5]), Init("q_zp", [8])]
            self.input = []
            self.output = []
            self.value_info = []

    graph = Graph()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        info = extract_quantization_info(graph)
        tensors = extract_quantization_tensors(graph)
        record = build_quantization_info(graph)
    assert info["group_size"] == 32
    assert info["has_zero_point"] is True
    assert "scales" not in info
    assert "zero_points" not in info
    assert tensors["scale_tensors"] == ["q_scale"]
    assert tensors["zero_point_tensors"] == ["q_zp"]
    assert tensors["zero_points"] == [8]
    assert record.group_size == 32
    assert record.has_zero_point is True
    assert record.zero_points == (8,)
    assert record.zero_point_tensors == ("q_zp",)
    assert record.scales == (0.5,)


def test_hybrid_name_is_classified():
    """A Gated DeltaNet or MoE name is classified and still belongs to a group."""
    from chunking_utils.common.graph_topology import (
        analyze_graph_topology,
        classify_hybrid_component,
    )

    assert classify_hybrid_component("model.layers.0.linear_attn.gated_deltanet") == "gated_deltanet"
    assert classify_hybrid_component("/model/layers.0/mlp/gate/MatMul") == "moe_router"
    assert classify_hybrid_component("/model/layers.0/mlp/experts/0/up_proj") == "moe_expert"
    assert classify_hybrid_component("/model/layers.0/self_attn/q_proj") is None
    assert classify_hybrid_component("model.layers.0.mlp.gate_proj") is None

    class Node:
        def __init__(self, name, inputs, outputs):
            self.name = name
            self.input = list(inputs)
            self.output = list(outputs)
            self.op_type = "MatMul"

    class Graph:
        def __init__(self, nodes):
            self.node = list(nodes)
            self.initializer = []
            self.output = []

    nodes = [
        Node("/model/embed_tokens/Gather", ["input_ids"], ["hidden"]),
        Node("model.layers.0.linear_attn.gated_deltanet", ["hidden"], ["h0"]),
        Node("/model/layers.0/mlp/experts/0/up_proj", ["h0"], ["h1"]),
        Node("/model/norm/Mul", ["h1"], ["hn"]),
        Node("/lm_head/MatMul", ["hn"], ["logits"]),
    ]
    topology = analyze_graph_topology(Graph(nodes))
    assert topology.node_roles[1] == "gated_deltanet"
    assert topology.node_roles[2] == "moe_expert"
    names = {
        group.name: [nodes[index].name for index in group.node_indices]
        for group in topology.groups
    }
    assert "model.layers.0.linear_attn.gated_deltanet" in names["layer_0"]
    assert "/model/layers.0/mlp/experts/0/up_proj" in names["layer_0"]
    assert 1 in topology.node_roles


def main():
    test_io_bound_reuses_device_buffer()
    test_preallocate_fixes_kv_shape()
    test_phase_providers_differ()
    test_quantization_keeps_group_and_zero_point()
    test_hybrid_name_is_classified()
    print("test_package_npu_readiness: OK")


if __name__ == "__main__":
    main()
