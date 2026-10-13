"""Host handoff from the static prefill graph to the static decode graph.

The prefill length is the decode window. Prefill writes ``present`` key
and value for that whole window. The host copies those tensors into the
decode past. Each decode step drops index 0, appends one token, and
``commit_present_swap`` installs that full window. The decode bias still
blocks score index 0 only. These graphs return a projected hidden vector,
not logits. ``MEASURED_NPU_FASTER_PREFILL_PROMPT`` is 256 for heads 8
and head 128 only. ``select_prefill_device`` returns ``npu`` at that
prompt and ``cpu`` below it. Every other layout returns ``unmeasured``.
``select_execution_phase`` does not read the constant.

Measured on 2026-10-13 in conda ``ryzen-ai-1.8.0``, window 8, head 64,
five decode steps. CPU matched the chained float32 reference within
1e-4, and the past pointer changed on every step. Vitis reported
``VAIML 12`` for prefill and ``VAIML 15`` for decode, each subgraph
``NPU 1``, with one kernel each (output sizes 6144 and 4352). The NPU
differed from the chain by 0.0020 on the prefill output, at most 0.0012
on the five decode outputs, 0.0029 on the final key, and 0.0038 on the
final value. No ELF size was printed.

The same handoff at heads 8, head 128, and window 256 stayed finite.
Prefill differed from the float32 chain by 0.0077. The five decode
outputs differed by 0.0046, 0.0048, 0.0052, 0.0055, and 0.0045. The
final key differed by 0.0083 and the final value by 0.0097. The past
pointer moved on every step. Prefill reported ``VAIML 10`` and decode
``VAIML 15``, each subgraph ``NPU 1``, with one ``vitis_ai_ep`` kernel
each (output sizes 3145728 and 2101248). No ELF size was printed.
VirtualLock did not page-lock the 1048576-byte KV tensors. Host
OrtValues stayed on CPU.
"""

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from chunking_utils.onnx.golden_attention_subgraph import default_projection
from chunking_utils.onnx.masked_projected_attention import (
    attention_bias_f16,
    build_masked_projected_attention_model,
    reference_masked_projected_attention,
)
from chunking_utils.onnx.prefill_attention import (
    build_prefill_attention_model,
    causal_bias_f16,
    default_prompt,
    reference_prefill_attention,
)


def decode_tokens(steps: int, head_dim: int, num_heads: int = 1) -> np.ndarray:
    """``steps`` distinct tokens shaped ``[steps, 1, heads, 1, head]``."""
    heads = int(num_heads)
    count = int(steps) * int(head_dim) * heads
    values = np.arange(count, dtype=np.float32) / np.float32(count) - np.float32(0.5)
    return np.ascontiguousarray(values.reshape(int(steps), 1, heads, 1, int(head_dim)))


def reference_prefill_decode_handoff(
    prompt_len: int = 8,
    head_dim: int = 64,
    steps: int = 5,
    num_heads: int = 1,
) -> Dict[str, object]:
    """Float32 chain: one prefill, then ``steps`` full-window decode rolls."""
    prompt = int(prompt_len)
    head = int(head_dim)
    heads = int(num_heads)
    weights = [default_projection(seed, head, head) for seed in (1, 2, 3, 4)]
    key, value, prefill_hidden = reference_prefill_attention(
        default_prompt(prompt, head, heads),
        *weights,
        causal_bias_f16(prompt),
    )
    bias = attention_bias_f16(prompt)
    tokens = decode_tokens(int(steps), head, heads)
    decode_hidden: List[np.ndarray] = []
    for step in range(int(steps)):
        key, value, projected = reference_masked_projected_attention(
            key,
            value,
            tokens[step],
            *weights,
            bias,
        )
        decode_hidden.append(projected)
    return {
        "prefill_hidden": prefill_hidden,
        "decode_hidden": decode_hidden,
        "final_key": key,
        "final_value": value,
    }


def run_prefill_decode_handoff(
    prompt_len: int = 8,
    head_dim: int = 64,
    steps: int = 5,
    num_heads: int = 1,
    providers: Optional[Sequence[str]] = None,
    profile_dir: Optional[str] = None,
) -> Dict[str, object]:
    """Run prefill once, copy its KV into decode, and roll ``steps`` tokens.

    ``run_bound_session`` binds only the cache. This binds ``hidden_states``
    and ``hidden_out`` as well, then calls ``commit_present_swap``.
    """
    import onnxruntime as ort

    from pipeline_utils.kv_cache import KVCacheBackend, KVCacheManager

    prompt = int(prompt_len)
    head = int(head_dim)
    heads = int(num_heads)
    step_count = int(steps)
    if heads < 1:
        raise ValueError(f"num_heads must be positive, got {heads}")
    if prompt < 2:
        raise ValueError(f"prompt_len must match a full decode window, got {prompt}")
    if step_count < 1:
        raise ValueError(f"steps must be positive, got {step_count}")
    selected = list(providers or ["CPUExecutionProvider"])
    prefill_model = build_prefill_attention_model(
        prompt_len=prompt, head_dim=head, num_heads=heads
    )
    decode_model = build_masked_projected_attention_model(
        seq_len=prompt, head_dim=head, num_heads=heads
    )
    hidden = np.ascontiguousarray(default_prompt(prompt, head, heads))
    tokens = decode_tokens(step_count, head, heads)
    window = [1, heads, prompt, head]
    token_shape = [1, heads, 1, head]

    def session_options(name: str):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if profile_dir:
            options.log_severity_level = 1
            options.enable_profiling = True
            options.profile_file_prefix = os.path.join(profile_dir, name)
        return options

    prefill_session = ort.InferenceSession(
        prefill_model.SerializeToString(),
        sess_options=session_options("prefill"),
        providers=selected,
    )
    decode_session = ort.InferenceSession(
        decode_model.SerializeToString(),
        sess_options=session_options("decode"),
        providers=selected,
    )

    hidden_ort = ort.OrtValue.ortvalue_from_numpy(hidden, "cpu", 0)
    key_ort = ort.OrtValue.ortvalue_from_shape_and_type(window, np.float32, "cpu", 0)
    value_ort = ort.OrtValue.ortvalue_from_shape_and_type(window, np.float32, "cpu", 0)
    prefill_out = ort.OrtValue.ortvalue_from_shape_and_type(window, np.float32, "cpu", 0)
    binding = prefill_session.io_binding()
    binding.bind_ortvalue_input("hidden_states", hidden_ort)
    binding.bind_ortvalue_output("present.0.key", key_ort)
    binding.bind_ortvalue_output("present.0.value", value_ort)
    binding.bind_ortvalue_output("hidden_out", prefill_out)
    prefill_session.run_with_iobinding(binding)
    prefill_profile = prefill_session.end_profiling() if profile_dir else ""
    present_key = np.array(key_ort.numpy(), dtype=np.float32, copy=True)
    present_value = np.array(value_ort.numpy(), dtype=np.float32, copy=True)
    prefill_hidden = np.array(prefill_out.numpy(), dtype=np.float32, copy=True)

    cache = KVCacheManager(
        num_layers=1,
        num_kv_heads=heads,
        head_dim=head,
        dtype=np.float32,
        batch_size=1,
        backend=KVCacheBackend.VITISAI,
        max_seq_len=prompt,
    )
    stored_key, stored_value = cache.get_past_tensors(0)
    stored_key.ort_value.update_inplace(present_key)
    stored_value.ort_value.update_inplace(present_value)

    decode_hidden: List[np.ndarray] = []
    ptr_moved: List[bool] = []
    decode_profile = ""
    for step in range(step_count):
        token_ort = ort.OrtValue.ortvalue_from_numpy(
            np.ascontiguousarray(tokens[step]), "cpu", 0
        )
        hidden_out = ort.OrtValue.ortvalue_from_shape_and_type(
            token_shape, np.float32, "cpu", 0
        )
        before = int(cache.get_device_tensors(0)[0].data_ptr())
        binding = decode_session.io_binding()
        cache.bind_cache_to_session(decode_session, binding, 0)
        binding.bind_ortvalue_input("hidden_states", token_ort)
        binding.bind_ortvalue_output("hidden_out", hidden_out)
        decode_session.run_with_iobinding(binding)
        cache.commit_present_swap(0, prompt)
        ptr_moved.append(int(cache.get_device_tensors(0)[0].data_ptr()) != before)
        decode_hidden.append(np.array(hidden_out.numpy(), dtype=np.float32, copy=True))
        if profile_dir and step == 0:
            decode_profile = decode_session.end_profiling()

    final_key, final_value = cache.get_device_tensors(0)
    return {
        "prefill_hidden": prefill_hidden,
        "decode_hidden": decode_hidden,
        "final_key": np.array(final_key.numpy(), dtype=np.float32, copy=True),
        "final_value": np.array(final_value.numpy(), dtype=np.float32, copy=True),
        "ptr_moved": ptr_moved,
        "prefill_profile": prefill_profile,
        "decode_profile": decode_profile,
    }
