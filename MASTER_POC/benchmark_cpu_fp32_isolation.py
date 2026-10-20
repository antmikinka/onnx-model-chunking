#!/usr/bin/env python3
"""CPU isolation for the stamped bake. This script does not open Vitis.

It writes position_ids onto the 36 baked layer protos, leaves the
external data files untouched, then teacher-forces one prompt through
the INT4 monolith and through the bake on CPUExecutionProvider.

Baked MatMul weights stay bfloat16 on disk. This process upcasts those
bytes to float32 and removes only the MatMul activation cast to
bfloat16 and the MatMul output cast back to float32. The norm casts
stay. The upcast still carries bfloat16 rounding of the stored weights.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmark_npu_decode import (  # noqa: E402
    DEFAULT_SPLIT,
    HEAD_DIM,
    KV_HEADS,
    N_LAYERS,
    PAST_SLOTS,
    MonolithDecode,
    _component_map,
    _hidden_inputs,
    _hidden_outputs,
    right_aligned_decode_bias,
)
from chunking_utils.onnx.static_shapes import _ensure_position_ids, _int_attr  # noqa: E402

TOKENIZER = r"C:\Users\antmi\qwen3-work\qwen3-8b-onnx\tokenizer.json"
PROMPT = (
    "The capital of France is Paris. A byte is eight bits. "
    "Rotary embeddings rotate each query and key by its absolute token index."
)


def _data_stamp(path: str) -> Dict[str, int]:
    if not os.path.isfile(path):
        return {}
    info = os.stat(path)
    return {"size": int(info.st_size), "mtime_ns": int(info.st_mtime_ns)}


def _external_locations(model) -> List[str]:
    found = []
    for item in model.graph.initializer:
        if int(item.data_location) != int(TensorProto.EXTERNAL):
            continue
        location = ""
        for entry in item.external_data:
            if entry.key == "location":
                location = entry.value
        found.append(location)
    return found


def stamp_position_ids(split_dir: str) -> Dict[str, object]:
    """Add int64 position_ids [1, 1] to each baked layer. Weights stay external."""
    components, _origin = _component_map(split_dir)
    stamped = []
    skipped = []
    for index in range(N_LAYERS):
        comp = components[f"layer_{index}"]
        path = os.path.join(split_dir, comp["file"])
        data_path = path + ".data"
        before = _data_stamp(data_path)
        model = onnx.load(path, load_external_data=False)
        before_locs = _external_locations(model)
        changed = False
        for node in model.graph.node:
            if node.op_type != "GroupQueryAttention":
                continue
            if _int_attr(node, "do_rotary", 0) != 1:
                raise RuntimeError(f"{path} GroupQueryAttention has do_rotary off")
            while len(node.input) <= 9:
                node.input.append("")
            if node.input[9] != "position_ids":
                if node.input[9]:
                    raise RuntimeError(f"{path} input 9 is {node.input[9]}")
                node.input[9] = "position_ids"
                changed = True
            _ensure_position_ids(model.graph, "position_ids", 1, 1)
        if "position_ids" not in comp["input_names"]:
            comp["input_names"].append("position_ids")
            changed = True
        if changed:
            temporary = path + ".stamp.tmp"
            onnx.save(model, temporary)
            if os.path.getsize(temporary) > 2_000_000:
                os.remove(temporary)
                raise RuntimeError(f"{path} stamp grew past 2 MB and may have inlined weights")
            reloaded = onnx.load(temporary, load_external_data=False)
            if _external_locations(reloaded) != before_locs:
                os.remove(temporary)
                raise RuntimeError(f"{path} stamp changed external data locations")
            if int(reloaded.ir_version) != 10:
                os.remove(temporary)
                raise RuntimeError(f"{path} ir_version became {reloaded.ir_version}")
            os.replace(temporary, path)
            stamped.append(comp["file"])
        else:
            skipped.append(comp["file"])
        after = _data_stamp(data_path)
        if after != before:
            raise RuntimeError(f"{data_path} changed during the stamp: {before} -> {after}")
        del model
    info_path = os.path.join(split_dir, "model_info.json")
    with open(info_path, "r", encoding="utf-8") as handle:
        info = json.load(handle)
    for index in range(N_LAYERS):
        names = info["components"][f"layer_{index}"]["input_names"]
        if "position_ids" not in names:
            names.append("position_ids")
    with open(info_path, "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)
        handle.write("\n")
    return {"stamped": stamped, "already": skipped}


def _bf16_to_f32(raw: bytes, dims) -> np.ndarray:
    units = np.frombuffer(raw, dtype=np.uint16)
    values = (units.astype(np.uint32) << 16).view(np.float32)
    return np.ascontiguousarray(values.reshape(tuple(int(dim) for dim in dims)))


def upcast_baked_matmuls(model) -> int:
    """Turn baked bf16 MatMuls into float32 MatMuls. Norm casts stay."""
    graph = model.graph
    matmul_inputs = set()
    for node in graph.node:
        if node.op_type == "MatMul" and len(node.input) > 1:
            matmul_inputs.add(node.input[1])
    replaced = []
    for item in list(graph.initializer):
        if int(item.data_type) != int(TensorProto.BFLOAT16):
            continue
        if item.name not in matmul_inputs:
            raise RuntimeError(f"bfloat16 initializer {item.name} is not a MatMul weight")
        if not item.raw_data:
            raise RuntimeError(f"{item.name} has no raw bfloat16 bytes")
        array = _bf16_to_f32(item.raw_data, item.dims)
        item.CopyFrom(numpy_helper.from_array(array, item.name))
        replaced.append(item.name)
    by_output = {}
    for node in graph.node:
        for name in node.output:
            if name:
                by_output[name] = node
    remove = set()
    for node in graph.node:
        if node.op_type != "Cast" or not node.name.endswith("/Cast_act_bf16"):
            continue
        to = next(attr.i for attr in node.attribute if attr.name == "to")
        if int(to) != int(TensorProto.BFLOAT16):
            raise RuntimeError(f"{node.name} is not a bfloat16 cast")
        source = node.input[0]
        dest = node.output[0]
        for consumer in graph.node:
            for index, name in enumerate(consumer.input):
                if name == dest:
                    consumer.input[index] = source
        remove.add(id(node))
    for node in graph.node:
        if node.op_type != "Cast" or not node.name.endswith("/Cast_out_f32"):
            continue
        producer = by_output.get(node.input[0])
        if producer is not None and producer.op_type == "MatMul":
            producer.output[0] = node.output[0]
            remove.add(id(node))
            continue
        if producer is None or producer.op_type != "Slice":
            raise RuntimeError(f"{node.name} follows {None if producer is None else producer.op_type}")
    for value in list(graph.value_info) + list(graph.output):
        if not value.type.HasField("tensor_type"):
            continue
        if int(value.type.tensor_type.elem_type) == int(TensorProto.BFLOAT16):
            value.type.tensor_type.elem_type = int(TensorProto.FLOAT)
    if remove:
        kept = [node for node in graph.node if id(node) not in remove]
        del graph.node[:]
        graph.node.extend(kept)
    leftover = [
        item.name
        for item in graph.initializer
        if int(item.data_type) == int(TensorProto.BFLOAT16)
    ]
    if leftover:
        raise RuntimeError(f"bfloat16 initializers remain: {leftover}")
    bf16_casts = [
        node.name
        for node in graph.node
        if node.op_type == "Cast"
        and any(attr.name == "to" and int(attr.i) == int(TensorProto.BFLOAT16) for attr in node.attribute)
    ]
    if bf16_casts:
        raise RuntimeError(f"bfloat16 casts remain: {bf16_casts}")
    return len(replaced)


def _session_options():
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    options.enable_profiling = False
    return options


def _session_bytes(model):
    import onnxruntime as ort

    return ort.InferenceSession(
        model.SerializeToString(),
        sess_options=_session_options(),
        providers=["CPUExecutionProvider"],
    )


def _session_file(filename: str):
    import onnxruntime as ort

    return ort.InferenceSession(
        os.path.basename(filename),
        sess_options=_session_options(),
        providers=["CPUExecutionProvider"],
    )


def _session_external(model, filename: str):
    """ORT rejects an in-memory proto larger than 2 GB. Keep those weights on disk."""
    import shutil

    directory = os.path.join(
        r"C:\Users\antmi\qwen3-work\fp32-isolation-tmp",
        os.path.splitext(os.path.basename(filename))[0],
    )
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    destination = os.path.join(directory, "model.onnx")
    onnx.save_model(
        model,
        destination,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.data",
        size_threshold=1024,
    )
    previous = os.getcwd()
    os.chdir(directory)
    try:
        return _session_file("model.onnx")
    finally:
        os.chdir(previous)


def _needs_upcast(filename: str) -> bool:
    model = onnx.load(filename, load_external_data=False)
    return any(int(item.data_type) == int(TensorProto.BFLOAT16) for item in model.graph.initializer)


def _initializer_bytes(model) -> int:
    return sum(len(item.raw_data) for item in model.graph.initializer)


def _load_component(split_dir: str, filename: str):
    path = os.path.join(split_dir, filename)
    os.chdir(split_dir)
    if not _needs_upcast(path):
        return _session_file(filename), 0
    model = onnx.load(filename, load_external_data=True)
    baked = upcast_baked_matmuls(model)
    if _initializer_bytes(model) >= 1_500_000_000:
        session = _session_external(model, filename)
    else:
        session = _session_bytes(model)
    del model
    gc.collect()
    return session, baked


def _prompt_ids() -> List[int]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(TOKENIZER)
    ids = list(tokenizer.encode(PROMPT).ids)
    if len(ids) < PAST_SLOTS + 1:
        raise RuntimeError(f"prompt produced {len(ids)} tokens")
    return ids[: PAST_SLOTS + 1]


def _working_set_mb() -> float:
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    process = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        process, ctypes.byref(counters), counters.cb
    )
    if not ok:
        return -1.0
    return float(counters.WorkingSetSize) / (1024.0 * 1024.0)


def _logits_of(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32).reshape(-1)
    return np.array(values, copy=True)


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, float]:
    gap = np.abs(reference - candidate)
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
    kl = float(np.sum(ref_prob[mask] * np.log(ref_prob[mask] / np.maximum(cand_prob[mask], 1e-30))))
    return {
        "max_abs": float(np.max(gap)),
        "mean_abs": float(np.mean(gap)),
        "top1_agree": int(int(np.argmax(reference)) == int(np.argmax(candidate))),
        "ref_top1": int(np.argmax(reference)),
        "split_top1": int(np.argmax(candidate)),
        "top5_overlap": float(overlap),
        "kl_mono_to_split": kl,
    }


class CpuSplit:
    def __init__(self, split_dir: str):
        components, _origin = _component_map(split_dir)
        self.components = components
        self.past = [
            (
                np.zeros((1, KV_HEADS, PAST_SLOTS, HEAD_DIM), dtype=np.float32),
                np.zeros((1, KV_HEADS, PAST_SLOTS, HEAD_DIM), dtype=np.float32),
            )
            for _ in range(N_LAYERS)
        ]
        print("LOAD embedding", flush=True)
        embed = components["embedding"]
        self.embedding, embed_baked = _load_component(split_dir, embed["file"])
        self.embed_out = embed["output_names"][0]
        if embed_baked != 0:
            raise RuntimeError(f"embedding baked {embed_baked} matmuls")
        self.layers = []
        for index in range(N_LAYERS):
            comp = components[f"layer_{index}"]
            print(f"LOAD layer {index} rss_mb={_working_set_mb():.0f}", flush=True)
            session, baked = _load_component(split_dir, comp["file"])
            if baked != 7:
                raise RuntimeError(f"layer {index} upcast {baked} matmuls")
            inputs = [item.name for item in session.get_inputs()]
            outputs = [item.name for item in session.get_outputs()]
            if "position_ids" not in inputs:
                raise RuntimeError(f"layer {index} has no position_ids input")
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
                }
            )
        print("LOAD norm", flush=True)
        norm = components["norm"]
        self.norm, norm_baked = _load_component(split_dir, norm["file"])
        if norm_baked != 0:
            raise RuntimeError(f"norm baked {norm_baked} matmuls")
        self.norm_inputs = list(norm["input_names"])
        self.norm_output = norm["output_names"][0]
        print("LOAD lm_head", flush=True)
        head = components["lm_head"]
        self.lm_head, head_baked = _load_component(split_dir, head["file"])
        if head_baked < 1:
            raise RuntimeError("lm_head upcast no matmuls")
        self.lm_input = head["input_names"][0]
        self.lm_output = head["output_names"][0]
        self.head_baked = head_baked
        print(f"LOAD done rss_mb={_working_set_mb():.0f} lm_head_matmuls={head_baked}", flush=True)

    def step(self, token_id: int, past_valid: int, position: int) -> np.ndarray:
        bias = right_aligned_decode_bias(past_valid)
        embed = self.embedding.run(
            None, {"input_ids": np.asarray([[int(token_id)]], dtype=np.int64)}
        )
        values = {self.embed_out: np.array(embed[0], copy=True)}
        position_ids = np.asarray([[int(position)]], dtype=np.int64)
        for index, layer in enumerate(self.layers):
            feeds = {
                layer["past_key"]: self.past[index][0],
                layer["past_value"]: self.past[index][1],
                "attention_bias": bias,
                "position_ids": position_ids,
            }
            for name in layer["hiddens"]:
                feeds[name] = values[name]
            outputs = layer["session"].run(None, feeds)
            by_name = {
                name: np.array(array, copy=True)
                for name, array in zip(layer["outputs"], outputs)
            }
            self.past[index] = (by_name[layer["present_key"]], by_name[layer["present_value"]])
            for name in layer["hidden_outs"]:
                values[name] = by_name[name]
        norm_feeds = {name: values[name] for name in self.norm_inputs}
        norm_out = self.norm.run(None, norm_feeds)
        values[self.norm_output] = np.array(norm_out[0], copy=True)
        logits = self.lm_head.run(None, {self.lm_input: values[self.norm_output]})
        return _logits_of(logits[0])


def _directory_bytes(split_dir: str) -> Dict[str, int]:
    layers = 0
    embeddings = 0
    lm_head = 0
    total = 0
    for name in os.listdir(split_dir):
        path = os.path.join(split_dir, name)
        if not os.path.isfile(path):
            continue
        size = os.path.getsize(path)
        total += size
        if name.startswith("layer_") and name.endswith(".onnx.data"):
            layers += size
        elif name == "embeddings.onnx.data":
            embeddings = size
        elif name == "lm_head.onnx.data":
            lm_head = size
    return {
        "layers_data": layers,
        "embeddings_data": embeddings,
        "lm_head_data": lm_head,
        "directory_total": total,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT)
    parser.add_argument("--stamp-only", action="store_true")
    parser.add_argument(
        "--output",
        default=r"C:\Users\antmi\qwen3-work\cpu-fp32-isolation-report.json",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    stamp = stamp_position_ids(args.split_dir)
    print(f"STAMP stamped={len(stamp['stamped'])} already={len(stamp['already'])}", flush=True)
    if args.stamp_only:
        return 0
    ids = _prompt_ids()
    print(f"PROMPT tokens={len(ids)} first={ids[0]}", flush=True)
    components, monolith_path = _component_map(args.split_dir)
    if not monolith_path:
        raise RuntimeError("model_info has no original model")
    monolith = MonolithDecode(monolith_path, 3)
    reference = []
    for step, token_id in enumerate(ids):
        reference.append(_logits_of(monolith.step(int(token_id))))
        print(f"MONO step {step} top1={int(np.argmax(reference[-1]))}", flush=True)
    monolith.close()
    del monolith
    gc.collect()
    print(f"MONO released rss_mb={_working_set_mb():.0f}", flush=True)
    split = CpuSplit(args.split_dir)
    rows = []
    for step, token_id in enumerate(ids):
        candidate = split.step(int(token_id), min(step, PAST_SLOTS), step)
        row = _metrics(reference[step], candidate)
        row["step"] = step
        row["token_id"] = int(token_id)
        rows.append(row)
        print(
            f"STEP {step} max={row['max_abs']:.6f} mean={row['mean_abs']:.6f} "
            f"top1={row['top1_agree']} kl={row['kl_mono_to_split']:.6f}",
            flush=True,
        )
    report = {
        "prompt": PROMPT,
        "token_ids": ids,
        "stamp": stamp,
        "steps": rows,
        "step_0": rows[0],
        "step_16": rows[PAST_SLOTS],
        "top1_agree_steps": sum(row["top1_agree"] for row in rows),
        "n_steps": len(rows),
        "lm_head_matmuls_upcast": split.head_baked,
        "bake_bytes": _directory_bytes(args.split_dir),
        "rss_mb": _working_set_mb(),
        "wall_s": round(time.perf_counter() - started, 3),
        "providers": ["CPUExecutionProvider"],
        "position_ids": "int64 [[step]]",
        "weight_note": (
            "CPU MatMul used float32 upcasts of the stored bfloat16 weights. "
            "This run did not open Vitis."
        ),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"REPORT {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
