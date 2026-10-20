#!/usr/bin/env python3
"""Decode-only measurement for the baked Qwen3 split.

The graphs under the bake directory are query length 1. A 50-token prompt
does not fit. This script measures one-token decode steps, the host working
set, and the token distribution against one CPU forward of the packed
INT4 monolith on the same token ids.

Provider routing is fixed. The monolith session lists only
CPUExecutionProvider. Embedding and the final norm list only
CPUExecutionProvider. Each layer and the LM head list
VitisAIExecutionProvider, then CPUExecutionProvider, with the cache key
written by the layer batch for layers 1-35.

KV tensors are float32 [1, 8, 16, 128]. The present Slice keeps the newest
16 of 17 positions, so the attention bias is right-aligned. The swap uses
KVCacheManager.commit_present_swap on host buffers. VitisAIExecutionProvider
in this Ryzen AI 1.8 wheel has no NPU OrtValue device. The swap is host
memory.

static_seqlens_k is the constant 16. If a layer input is named
position_ids, each step feeds int64 [[step]] and that counter is not
capped at 16. past_valid is capped and is only the bias argument.
Graphs that omit position_ids keep the frozen seqlens row. The pass
check is the conjunction of top-1 agreement on at least 95 percent of
steps, top-5 overlap of at least 0.8 on every step, and KL below 0.05
on every step. Max absolute logit gap is recorded and is not the check.
The process exits 2 when that conjunction fails. Comparison stops at
step 16.

Do not point benchmark_onnx_split_vs_whole.py at these files.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline_utils.kv_cache import (  # noqa: E402
    KVCacheBackend,
    KVCacheManager,
    KVUpdateMode,
)

DEFAULT_SPLIT = r"C:\Users\antmi\qwen3-work\split-qwen-npu-full\model.onnx"
DEFAULT_CACHE = r"C:\Users\antmi\qwen3-work\split-qwen-npu-full\vitis-cache"
BATCH_LOCK = r"C:\Users\antmi\qwen3-work\vitis-pos-batch\RUNNING"
ISOLATION_PROMPT_IDS = [
    785,
    6722,
    315,
    9625,
    374,
    12095,
    13,
    362,
    4922,
    374,
    8063,
    9472,
    13,
    74856,
    70547,
    16919,
    1817,
]
CPU_FP32_FLOOR = {
    "source": r"C:\Users\antmi\qwen3-work\cpu-fp32-isolation-report.json",
    "prompt_token_ids": ISOLATION_PROMPT_IDS,
    "step_0_max_abs": 0.27833956480026245,
    "step_16_max_abs": 0.7901263236999512,
    "note": (
        "CPU MatMul of the stored bfloat16 weights, upcast to float32, "
        "on this prompt. This is the bake rounding measured on CPU. "
        "It is not an NPU gap."
    ),
}
PAST_SLOTS = 16
QUERY_LEN = 1
KEY_LEN = PAST_SLOTS + QUERY_LEN
KV_HEADS = 8
HEAD_DIM = 128
N_LAYERS = 36
VOCAB = 151936
MASK_VALUE = np.float32(-10000.0)


def layer_cache_dir(cache_root: str, index: int) -> str:
    return os.path.join(cache_root, f"layer_{index:02d}_pos")


def layer_cache_key(index: int) -> str:
    return f"fullbake_layer_{index:02d}_pos"


def right_aligned_decode_bias(past_valid: int, past_slots: int = PAST_SLOTS) -> np.ndarray:
    """Mask padding on the left. The current token is the last key position.

    The baked present Slice starts at query_len and keeps the newest
    past_slots of past_slots + query_len. Real past tokens therefore sit
    at the right of the window, and the token being decoded is the last
    column. static_attention_bias is left-aligned and is not this mask.
    """
    if not 0 <= int(past_valid) <= int(past_slots):
        raise ValueError(f"past_valid {past_valid} is outside 0..{past_slots}")
    key_len = int(past_slots) + QUERY_LEN
    bias = np.full((1, 1, QUERY_LEN, key_len), MASK_VALUE, dtype=np.float32)
    start = key_len - int(past_valid) - QUERY_LEN
    bias[..., start:] = np.float32(0)
    return bias


def _component_map(split_dir: str) -> dict:
    path = os.path.join(split_dir, "model_info.json")
    with open(path, "r", encoding="utf-8") as handle:
        info = json.load(handle)
    return info["components"], info.get("original_model")


def _names(component: dict, field: str) -> List[str]:
    return list(component[field])


def check_model_info(split_dir: str) -> List[str]:
    """Return wiring errors. An empty list means the decode chain connects."""
    components, _origin = _component_map(split_dir)
    errors = []
    required = ["embedding", "norm", "lm_head"] + [f"layer_{i}" for i in range(N_LAYERS)]
    for name in required:
        if name not in components:
            errors.append(f"missing component {name}")
    if errors:
        return errors
    embed_out = _names(components["embedding"], "output_names")
    if len(embed_out) != 1:
        errors.append(f"embedding outputs {embed_out}")
    layer0_in = _names(components["layer_0"], "input_names")
    if embed_out[0] not in layer0_in:
        errors.append("layer_0 does not consume the embedding output")
    layer0_out = _names(components["layer_0"], "output_names")
    if layer0_out[:2] != ["present.0.key", "present.0.value"]:
        errors.append(f"layer_0 outputs do not start with present KV: {layer0_out}")
    if len(_hidden_inputs(layer0_in)) != 1:
        errors.append(f"layer_0 hidden inputs {_hidden_inputs(layer0_in)}")
    previous_outs = set(layer0_out)
    for index in range(1, N_LAYERS):
        comp = components[f"layer_{index}"]
        hidden_in = [
            name
            for name in _names(comp, "input_names")
            if not _is_layer_side_input(name)
        ]
        if len(hidden_in) != 2:
            errors.append(f"layer_{index} hidden inputs {hidden_in}")
        for name in hidden_in:
            if name not in previous_outs:
                errors.append(f"layer_{index} input {name} is not produced by layer_{index - 1}")
        outs = _names(comp, "output_names")
        if outs[:2] != [f"present.{index}.key", f"present.{index}.value"]:
            errors.append(f"layer_{index} outputs do not start with present KV: {outs}")
        previous_outs = set(outs)
    norm_in = _names(components["norm"], "input_names")
    for name in norm_in:
        if name not in previous_outs:
            errors.append(f"norm input {name} is not produced by layer_35")
    norm_out = _names(components["norm"], "output_names")
    lm_in = _names(components["lm_head"], "input_names")
    if norm_out != lm_in:
        errors.append(f"norm outputs {norm_out} do not match lm_head inputs {lm_in}")
    return errors


def check_bias_and_swap() -> None:
    empty = right_aligned_decode_bias(0)
    assert empty.shape == (1, 1, 1, KEY_LEN)
    assert float(empty[0, 0, 0, -1]) == 0.0
    assert np.all(empty[0, 0, 0, :-1] == MASK_VALUE)
    one = right_aligned_decode_bias(1)
    assert float(one[0, 0, 0, -1]) == 0.0
    assert float(one[0, 0, 0, -2]) == 0.0
    assert np.all(one[0, 0, 0, :-2] == MASK_VALUE)
    full = right_aligned_decode_bias(PAST_SLOTS)
    assert np.all(full == 0)
    cache = KVCacheManager(
        num_layers=1,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=np.float32,
        batch_size=1,
        backend=KVCacheBackend.CPU,
        max_seq_len=PAST_SLOTS,
        update_mode=KVUpdateMode.CONCAT_SWAP,
    )
    shadow_key = cache._cache[0]["key_shadow"]
    shadow_key[...] = 1
    cache.commit_present_swap(0, 1)
    past_key, _past_value = cache.get_past_tensors(0)
    if not np.all(past_key == 1):
        raise AssertionError("commit_present_swap did not publish the shadow buffer")


def check_host_ort_write() -> None:
    """Prove a CPU session writes an output into a caller-owned NumPy buffer."""
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper

    value_in = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    value_out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "identity",
        [value_in],
        [value_out],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    path = os.path.join(os.environ.get("TEMP", "."), "npu_decode_identity.onnx")
    onnx.save(model, path)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
    buffer = np.zeros((2,), dtype=np.float32)
    held = ort.OrtValue.ortvalue_from_numpy(buffer, "cpu", 0)
    binding = session.io_binding()
    binding.bind_cpu_input("x", np.asarray([3.0, 4.0], dtype=np.float32))
    binding.bind_ortvalue_output("y", held)
    session.run_with_iobinding(binding)
    if buffer.tolist() != [3.0, 4.0]:
        raise AssertionError(f"host OrtValue output was {buffer.tolist()}")
    os.remove(path)


def self_check(split_dir: str) -> int:
    check_bias_and_swap()
    check_host_ort_write()
    empty = np.zeros((1, 8, 0, 128), dtype=np.float32)
    one = np.ones((1, 8, 1, 128), dtype=np.float32)
    grown = _extend_past(empty, one)
    if grown.shape[2] != 1:
        raise AssertionError(f"one-token present grew to {grown.shape}")
    full = np.full((1, 8, 2, 128), 2, dtype=np.float32)
    replaced = _extend_past(grown, full)
    if replaced.shape[2] != 2 or float(replaced[0, 0, 0, 0]) != 2.0:
        raise AssertionError("full present did not replace the past")
    errors = check_model_info(split_dir)
    if errors:
        for item in errors:
            print("SELF_CHECK_FAIL", item)
        return 1
    print("SELF_CHECK_OK")
    return 0


def _session_options(log_severity: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = int(log_severity)
    options.enable_profiling = False
    return options


def _cpu_session(path: str, log_severity: int):
    import onnxruntime as ort

    return ort.InferenceSession(
        path,
        sess_options=_session_options(log_severity),
        providers=["CPUExecutionProvider"],
    )


def _parse_layer_spec(spec: str, n_layers: int = N_LAYERS) -> set[int]:
    """Parse ``""``, ``"all"``, or comma-separated indices/ranges like ``"0,1,4-7"``."""
    cleaned = (spec or "").strip().lower()
    if not cleaned or cleaned == "none":
        return set()
    if cleaned == "all":
        return set(range(int(n_layers)))
    chosen: set[int] = set()
    for part in cleaned.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            lo_str, hi_str = token.split("-", 1)
            lo, hi = int(lo_str.strip()), int(hi_str.strip())
            if lo > hi or lo < 0 or hi >= int(n_layers):
                raise ValueError(f"layer range {token!r} out of 0..{n_layers - 1}")
            chosen.update(range(lo, hi + 1))
        else:
            idx = int(token)
            if idx < 0 or idx >= int(n_layers):
                raise ValueError(f"layer index {idx} out of 0..{n_layers - 1}")
            chosen.add(idx)
    return chosen


def _cpu_upcast_session(split_dir: str, filename: str, log_severity: int):
    """Load a baked bfloat16 component on CPUExecutionProvider by upcasting its MatMuls to float32."""
    import onnx
    import onnxruntime as ort
    import shutil
    from onnx import TensorProto
    from benchmark_cpu_fp32_isolation import upcast_baked_matmuls

    path = os.path.join(split_dir, filename)
    previous = os.getcwd()
    os.chdir(split_dir)
    try:
        probe = onnx.load(filename, load_external_data=False)
        has_bf16 = any(
            int(item.data_type) == int(TensorProto.BFLOAT16)
            for item in probe.graph.initializer
        )
        del probe
        if not has_bf16:
            return _cpu_session(path, log_severity)
        model = onnx.load(filename, load_external_data=True)
        upcast_baked_matmuls(model)
        total_bytes = sum(len(item.raw_data) for item in model.graph.initializer)
        if total_bytes >= 1_500_000_000:
            tmp_dir = os.path.join(
                split_dir,
                ".fp32-cpu-upcast-tmp",
                os.path.splitext(os.path.basename(filename))[0],
            )
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir, exist_ok=True)
            dst = os.path.join(tmp_dir, "model.onnx")
            onnx.save_model(
                model,
                dst,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="weights.data",
                size_threshold=1024,
            )
            del model
            gc.collect()
            os.chdir(tmp_dir)
            return ort.InferenceSession(
                "model.onnx",
                sess_options=_session_options(log_severity),
                providers=["CPUExecutionProvider"],
            )
        raw_proto = model.SerializeToString()
        del model
        gc.collect()
        return ort.InferenceSession(
            raw_proto,
            sess_options=_session_options(log_severity),
            providers=["CPUExecutionProvider"],
        )
    finally:
        os.chdir(previous)


def _vitis_session(path: str, cache_dir: str, cache_key: str, log_severity: int):
    import onnxruntime as ort

    os.makedirs(cache_dir, exist_ok=True)
    return ort.InferenceSession(
        os.path.basename(path),
        sess_options=_session_options(log_severity),
        providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
        provider_options=[{"cacheDir": cache_dir, "cacheKey": cache_key}, {}],
    )


def _is_layer_side_input(name: str) -> bool:
    return name.startswith("past_key_values.") or name in ("attention_bias", "position_ids")


def _hidden_inputs(input_names: List[str]) -> List[str]:
    return [name for name in input_names if not _is_layer_side_input(name)]


def _hidden_outputs(output_names: List[str], layer_idx: int) -> List[str]:
    present = {f"present.{layer_idx}.key", f"present.{layer_idx}.value"}
    return [name for name in output_names if name not in present]


class SplitDecode:
    """One resident decode pipeline. Sessions stay loaded for the timed loop."""

    def __init__(
        self,
        split_dir: str,
        cache_root: str,
        log_severity: int,
        *,
        cpu_lm_head: bool = False,
        cpu_layers: str = "",
    ):
        import onnxruntime as ort

        self.ort = ort
        self.split_dir = split_dir
        self.cpu_lm_head = bool(cpu_lm_head)
        self.cpu_layer_indices = _parse_layer_spec(cpu_layers, N_LAYERS)
        self.components, _origin = _component_map(split_dir)
        os.chdir(split_dir)
        load_started = time.perf_counter()
        embed = self.components["embedding"]
        self.embedding = _cpu_session(
            os.path.join(split_dir, embed["file"]), log_severity
        )
        self.embed_out = embed["output_names"][0]
        self.layers = []
        for index in range(N_LAYERS):
            comp = self.components[f"layer_{index}"]
            if index in self.cpu_layer_indices:
                session = _cpu_upcast_session(split_dir, comp["file"], log_severity)
            else:
                session = _vitis_session(
                    os.path.join(split_dir, comp["file"]),
                    layer_cache_dir(cache_root, index),
                    layer_cache_key(index),
                    log_severity,
                )
            inputs = [item.name for item in session.get_inputs()]
            outputs = [item.name for item in session.get_outputs()]
            self.layers.append(
                {
                    "session": session,
                    "inputs": inputs,
                    "outputs": outputs,
                    "hiddens": _hidden_inputs(inputs),
                    "hidden_outs": _hidden_outputs(outputs, index),
                    "past_key": f"past_key_values.{index}.key",
                    "past_value": f"past_key_values.{index}.value",
                    "present_key": f"present.{index}.key",
                    "present_value": f"present.{index}.value",
                    "providers": list(session.get_providers()),
                }
            )
        norm = self.components["norm"]
        self.norm = _cpu_session(os.path.join(split_dir, norm["file"]), log_severity)
        self.norm_inputs = list(norm["input_names"])
        self.norm_output = norm["output_names"][0]
        head = self.components["lm_head"]
        if self.cpu_lm_head:
            self.lm_head = _cpu_upcast_session(split_dir, head["file"], log_severity)
        else:
            self.lm_head = _vitis_session(
                os.path.join(split_dir, head["file"]),
                os.path.join(cache_root, "lm_head"),
                "fullbake_lmhead_bf16",
                log_severity,
            )
        self.lm_input = head["input_names"][0]
        self.lm_output = head["output_names"][0]
        self.load_s = time.perf_counter() - load_started
        self.cache = KVCacheManager(
            num_layers=N_LAYERS,
            num_kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            dtype=np.float32,
            batch_size=1,
            backend=KVCacheBackend.CPU,
            max_seq_len=PAST_SLOTS,
            update_mode=KVUpdateMode.CONCAT_SWAP,
        )
        self.io_binding = True
        self.past_valid = 0
        self.token_position = 0

    def _run_layer(self, index: int, values: Dict[str, np.ndarray], bias: np.ndarray, position: int) -> None:
        layer = self.layers[index]
        session = layer["session"]
        past_key, past_value = self.cache.get_past_tensors(index)
        if self.io_binding:
            try:
                self._run_layer_bound(index, values, bias, past_key, past_value, position)
                return
            except Exception as exc:
                print(f"IO_BINDING_FALLBACK layer={index} error={exc}")
                self.io_binding = False
        feeds = {
            layer["past_key"]: past_key,
            layer["past_value"]: past_value,
            "attention_bias": bias,
        }
        if "position_ids" in layer["inputs"]:
            feeds["position_ids"] = np.asarray([[int(position)]], dtype=np.int64)
        for name in layer["hiddens"]:
            feeds[name] = values[name]
        outputs = session.run(None, feeds)
        by_name = {
            name: np.array(array, copy=True)
            for name, array in zip(layer["outputs"], outputs)
        }
        shadow_key = self.cache._cache[index]["key_shadow"]
        shadow_value = self.cache._cache[index]["value_shadow"]
        np.copyto(shadow_key, by_name[layer["present_key"]])
        np.copyto(shadow_value, by_name[layer["present_value"]])
        for name in layer["hidden_outs"]:
            values[name] = by_name[name]
        self.cache.commit_present_swap(
            index,
            min(self.past_valid + 1, PAST_SLOTS),
            total_seen=self.token_position + 1,
            rolling=True,
        )

    def _run_layer_bound(self, index, values, bias, past_key, past_value, position: int) -> None:
        layer = self.layers[index]
        session = layer["session"]
        binding = session.io_binding()
        binding.bind_cpu_input(layer["past_key"], past_key)
        binding.bind_cpu_input(layer["past_value"], past_value)
        binding.bind_cpu_input("attention_bias", bias)
        if "position_ids" in layer["inputs"]:
            binding.bind_cpu_input(
                "position_ids", np.asarray([[int(position)]], dtype=np.int64)
            )
        for name in layer["hiddens"]:
            binding.bind_cpu_input(name, values[name])
        shadow_key_ov, shadow_value_ov = self.cache.shadow_ort_values(index)
        binding.bind_ortvalue_output(layer["present_key"], shadow_key_ov)
        binding.bind_ortvalue_output(layer["present_value"], shadow_value_ov)
        for name in layer["hidden_outs"]:
            binding.bind_output(name, "cpu")
        session.run_with_iobinding(binding)
        copied = binding.copy_outputs_to_cpu()
        # KV outputs stay in the shadow buffers. Hidden copies follow the
        # outputs that were bound with bind_output, in that same order.
        if len(copied) < len(layer["hidden_outs"]):
            raise RuntimeError(
                f"layer {index} returned {len(copied)} host outputs for "
                f"{len(layer['hidden_outs'])} hidden tensors"
            )
        hidden_copied = copied[-len(layer["hidden_outs"]) :]
        parsed = []
        for name, array in zip(layer["hidden_outs"], hidden_copied):
            array = np.array(array, copy=True)
            if array.shape != (1, 1, 4096):
                raise RuntimeError(
                    f"layer {index} output {name} shape {array.shape} is not [1, 1, 4096]"
                )
            parsed.append((name, array))
        for name, array in parsed:
            values[name] = array
        # Keep the OrtValues alive until the run has returned.
        del shadow_key_ov, shadow_value_ov
        self.cache.commit_present_swap(
            index,
            min(self.past_valid + 1, PAST_SLOTS),
            total_seen=self.token_position + 1,
            rolling=True,
        )

    def step(self, token_id: int) -> np.ndarray:
        bias = self.cache.attention_bias(query_len=QUERY_LEN, align="right")
        position = int(self.cache.position_ids(query_len=1)[0, 0])
        embed = self.embedding.run(
            None, {"input_ids": np.asarray([[int(token_id)]], dtype=np.int64)}
        )
        values = {self.embed_out: np.array(embed[0], copy=True)}
        for index in range(N_LAYERS):
            self._run_layer(index, values, bias, position)
        norm_feeds = {name: values[name] for name in self.norm_inputs}
        norm_out = self.norm.run(None, norm_feeds)
        values[self.norm_output] = np.array(norm_out[0], copy=True)
        logits = self.lm_head.run(None, {self.lm_input: values[self.norm_output]})
        self.token_position += 1
        self.past_valid = min(self.past_valid + 1, PAST_SLOTS)
        return np.array(logits[0], copy=True)


def _extend_past(past: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Accept either a full grown cache or a one-token present."""
    present = np.array(present, copy=True)
    if present.shape[2] == past.shape[2] + 1:
        return present
    if present.shape[2] == 1:
        if past.shape[2] == 0:
            return present
        return np.concatenate([past, present], axis=2)
    raise RuntimeError(
        f"present length {present.shape[2]} does not follow past length {past.shape[2]}"
    )


class MonolithDecode:
    """CPU-only packed INT4 decode. One new token per step, growing past."""

    def __init__(self, model_path: str, log_severity: int):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            model_path,
            sess_options=_session_options(log_severity),
            providers=["CPUExecutionProvider"],
        )
        self.past = [
            (
                np.zeros((1, KV_HEADS, 0, HEAD_DIM), dtype=np.float32),
                np.zeros((1, KV_HEADS, 0, HEAD_DIM), dtype=np.float32),
            )
            for _ in range(N_LAYERS)
        ]
        self.past_len = 0
        self.output_names = [item.name for item in self.session.get_outputs()]

    def close(self) -> None:
        self.session = None
        self.past = []

    def step(self, token_id: int) -> np.ndarray:
        feeds = {
            "input_ids": np.asarray([[int(token_id)]], dtype=np.int64),
            "attention_mask": np.ones((1, self.past_len + 1), dtype=np.int64),
        }
        for index, (key, value) in enumerate(self.past):
            feeds[f"past_key_values.{index}.key"] = key
            feeds[f"past_key_values.{index}.value"] = value
        outputs = self.session.run(None, feeds)
        by_name = {
            name: np.array(array, copy=True)
            for name, array in zip(self.output_names, outputs)
        }
        updated = []
        for index, (key, value) in enumerate(self.past):
            updated.append(
                (
                    _extend_past(key, by_name[f"present.{index}.key"]),
                    _extend_past(value, by_name[f"present.{index}.value"]),
                )
            )
        self.past = updated
        self.past_len += 1
        return by_name["logits"]


def _logit_gap(monolith: np.ndarray, split: np.ndarray) -> dict:
    if monolith.shape != split.shape:
        return {
            "compared": False,
            "reason": f"shapes {list(monolith.shape)} and {list(split.shape)} differ",
        }
    reference = monolith.astype(np.float32).reshape(-1)
    candidate = split.astype(np.float32).reshape(-1)
    delta = np.abs(reference - candidate)
    ref_top = np.argsort(reference)[-5:]
    cand_top = np.argsort(candidate)[-5:]
    overlap = len(set(ref_top.tolist()) & set(cand_top.tolist())) / 5.0
    ref_shift = reference - np.max(reference)
    cand_shift = candidate - np.max(candidate)
    ref_prob = np.exp(ref_shift)
    cand_prob = np.exp(cand_shift)
    ref_prob /= np.sum(ref_prob)
    cand_prob /= np.sum(cand_prob)
    mask = ref_prob > 0
    kl = float(
        np.sum(ref_prob[mask] * np.log(ref_prob[mask] / np.maximum(cand_prob[mask], 1e-30)))
    )
    return {
        "compared": True,
        "max_abs": float(np.max(delta)),
        "mean_abs": float(np.mean(delta)),
        "monolith_max_abs": float(np.max(np.abs(reference))),
        "split_max_abs": float(np.max(np.abs(candidate))),
        "top1_agree": int(int(np.argmax(reference)) == int(np.argmax(candidate))),
        "ref_top1": int(np.argmax(reference)),
        "split_top1": int(np.argmax(candidate)),
        "top5_overlap": float(overlap),
        "kl_mono_to_split": kl,
    }


def _distribution_pass(rows: List[dict]) -> bool:
    compared = [row for row in rows if row.get("compared")]
    if len(compared) != len(rows) or not compared:
        return False
    top1_rate = sum(int(row["top1_agree"]) for row in compared) / len(compared)
    top5_ok = all(float(row["top5_overlap"]) >= 0.8 for row in compared)
    kl_ok = all(float(row["kl_mono_to_split"]) < 0.05 for row in compared)
    return bool(top1_rate >= 0.95 and top5_ok and kl_ok)


def _rss_mb() -> float:
    import psutil

    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def run_decode(args: argparse.Namespace) -> int:
    if os.path.isfile(BATCH_LOCK) and not args.allow_overlap:
        print(f"REFUSING Vitis sessions while {BATCH_LOCK} exists")
        return 1
    errors = check_model_info(args.split_dir)
    if errors:
        for item in errors:
            print("WIRE_FAIL", item)
        return 1
    align_step = PAST_SLOTS
    if args.isolation_prompt:
        token_schedule = list(ISOLATION_PROMPT_IDS)
        n_steps = len(token_schedule)
    else:
        token_schedule = []
        n_steps = max(int(args.tokens), align_step + 1)
    if n_steps > align_step + 1 and not (args.skip_monolith or getattr(args, "allow_rolling", False)):
        print(f"REFUSING {n_steps} steps. The split window and the monolith diverge after step {align_step}.")
        return 1
    peak = _rss_mb()
    monolith_error = None
    monolith_peak = None
    monolith_saved: Dict[int, np.ndarray] = {}
    if not args.skip_monolith:
        monolith = None
        try:
            monolith = MonolithDecode(args.whole_model, args.log_severity)
            token = int(args.token_id)
            greedy: List[int] = []
            for step in range(n_steps):
                if token_schedule:
                    token = token_schedule[step]
                else:
                    greedy.append(token)
                logits = monolith.step(token)
                peak = max(peak, _rss_mb())
                monolith_saved[step] = np.array(logits, copy=True)
                flat = logits.reshape(-1, logits.shape[-1])[-1]
                if not token_schedule:
                    token = int(np.argmax(flat))
            if not token_schedule:
                token_schedule = greedy
            monolith_peak = round(peak, 2)
        except Exception as exc:
            monolith_error = str(exc)
            print("MONOLITH_FAIL", monolith_error)
            token_schedule = []
            monolith_saved = {}
        finally:
            if monolith is not None:
                monolith.close()
            del monolith
            gc.collect()
    released_rss = round(_rss_mb(), 2)
    pipeline = SplitDecode(
        args.split_dir,
        args.cache_root,
        args.log_severity,
        cpu_lm_head=bool(getattr(args, "cpu_lm_head", False)),
        cpu_layers=str(getattr(args, "cpu_layers", "") or ""),
    )
    split_peak = _rss_mb()
    # Session creation is outside the timed loop. Profiling is off.
    step_ms: List[float] = []
    generated = []
    split_saved: Dict[int, np.ndarray] = {}
    token = int(args.token_id)
    for step in range(n_steps):
        if token_schedule:
            token = token_schedule[step]
        started = time.perf_counter()
        logits = pipeline.step(token)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        step_ms.append(elapsed_ms)
        split_peak = max(split_peak, _rss_mb())
        split_saved[step] = np.array(logits, copy=True)
        flat = logits.reshape(-1, logits.shape[-1])[-1]
        generated.append(int(np.argmax(flat)))
        if not token_schedule:
            token = generated[-1]
    step_gaps = []
    for step in range(n_steps):
        if step in monolith_saved and step in split_saved:
            gap = _logit_gap(monolith_saved[step], split_saved[step])
            gap["step"] = step
            gap["token_id"] = int(token_schedule[step]) if step < len(token_schedule) else None
            step_gaps.append(gap)
    cold_gap = next((row for row in step_gaps if row["step"] == 0), None)
    aligned_gap = next((row for row in step_gaps if row["step"] == align_step), None)
    quality_pass = None
    if not args.skip_monolith:
        window_gaps = [row for row in step_gaps if row["step"] <= align_step]
        expected_compared = min(n_steps, align_step + 1)
        quality_pass = (
            bool(window_gaps)
            and len(window_gaps) == expected_compared
            and _distribution_pass(window_gaps)
        )
    total_s = sum(step_ms) / 1000.0
    report = {
        "split_dir": args.split_dir,
        "whole_model": None if args.skip_monolith else args.whole_model,
        "token_id": int(args.token_id),
        "tokens": n_steps,
        "shared_input_ids": token_schedule,
        "generated_ids": generated,
        "step_ms": [round(value, 3) for value in step_ms],
        "decode_tps": None if total_s <= 0 else round(len(step_ms) / total_s, 4),
        "session_load_s": round(pipeline.load_s, 3),
        "monolith_peak_rss_mb": monolith_peak,
        "rss_after_monolith_release_mb": released_rss,
        "split_peak_rss_mb": round(split_peak, 2),
        "io_binding": pipeline.io_binding,
        "kv": "host float32 [1, 8, 16, 128] commit_present_swap",
        "providers": {
            "embedding": list(pipeline.embedding.get_providers()),
            "norm": list(pipeline.norm.get_providers()),
            "lm_head": list(pipeline.lm_head.get_providers()),
            "layer_0": pipeline.layers[0]["providers"],
            "layer_35": pipeline.layers[35]["providers"],
            "cpu_lm_head": bool(pipeline.cpu_lm_head),
            "cpu_layer_indices": sorted(pipeline.cpu_layer_indices),
        },
        "monolith_error": monolith_error,
        "cold_start_gap": cold_gap,
        "aligned_gap": aligned_gap,
        "step_gaps": step_gaps,
        "aligned_step": align_step,
        "cpu_fp32_floor": CPU_FP32_FLOOR,
        "quality_gate": {
            "top1_rate_min": 0.95,
            "top5_overlap_min": 0.8,
            "kl_max": 0.05,
        },
        "quality_pass": quality_pass,
        "comparison_note": (
            "Both sides receive the same token ids. With --isolation-prompt "
            "those ids are the 17-token CPU isolation prompt. The check is "
            "top-1 on at least 95 percent of steps, top-5 overlap of at least "
            "0.8 on every step, and KL below 0.05 on every step. Max absolute "
            "gap is recorded and is not the check. Comparison stops at step 16."
        ),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    if quality_pass is False:
        return 2
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode TPS for the baked NPU split")
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--whole-model",
        default=r"C:\Users\antmi\qwen3-work\qwen3-8b-onnx\model.onnx",
    )
    parser.add_argument("--cache-root", default=DEFAULT_CACHE)
    parser.add_argument("--token-id", type=int, default=1)
    parser.add_argument("--isolation-prompt", action="store_true")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=0.5)
    parser.add_argument("--log-severity", type=int, default=2)
    parser.add_argument("--output", default="")
    parser.add_argument("--skip-monolith", action="store_true")
    parser.add_argument("--allow-rolling", action="store_true")
    parser.add_argument("--allow-overlap", action="store_true")
    parser.add_argument(
        "--cpu-lm-head",
        action="store_true",
        help="Run lm_head on CPUExecutionProvider (upcasting baked bf16 weights to float32) while keeping NPU layers on Vitis.",
    )
    parser.add_argument(
        "--cpu-layers",
        default="",
        help="Run selected layers on CPUExecutionProvider (e.g. 'all', '0,1', or '0-17') while keeping remaining layers/lm_head on Vitis.",
    )
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_check:
        return self_check(args.split_dir)
    return run_decode(args)


if __name__ == "__main__":
    sys.exit(main())
