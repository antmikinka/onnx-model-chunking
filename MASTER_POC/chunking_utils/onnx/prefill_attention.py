"""Static prefill attention. No KV roll and no Transpose node.

Query length is the prompt length. ``K`` stays ``[1, 1, prompt, head]``.
Scores are a ``Gemm`` with ``transB=1``, which is ``Q @ K^T`` without a
Transpose node. The causal mask is a float16 initializer: 0 on and below
the diagonal, and -10000 above it. A Cast matches the float32 scores.

Measured on 2026-10-13 in conda ``ryzen-ai-1.8.0``, one head. At
``[1, 1, 8, 64]`` and ``[1, 1, 16, 64]`` the log said ``VAIML 12`` and
subgraph ``NPU 1``. The profiler recorded one ``vitis_ai_ep_1`` kernel.
Output sizes were 6144 and 12288 bytes. The compile instantiated
``GemmBfp16``. CPU matched this reference within 5e-8. The NPU differed
by at most 0.0028 and 0.0034 on the key, 0.0047 and 0.0034 on the value,
and 0.0020 and 0.0022 on the projected output. Thirty later calls took
0.413 ms and 0.386 ms, both slower than CPU. The log did not print an
ELF size. The causal mask moves the reference output by 0.121 and 0.130,
which is larger than the NPU gap.
"""

from typing import Dict, Sequence

import numpy as np
from onnx import ModelProto, TensorProto, helper, numpy_helper

from chunking_utils.onnx.attention_mask import ATTENTION_MASK_BIAS
from chunking_utils.onnx.golden_attention_subgraph import (
    IR_VERSION,
    OPSET,
    default_projection,
)

PREFILL_OPS = (
    "MatMul",
    "MatMul",
    "MatMul",
    "Reshape",
    "Reshape",
    "Gemm",
    "Reshape",
    "Cast",
    "Add",
    "Softmax",
    "MatMul",
    "MatMul",
)


def default_prompt(prompt_len: int, head_dim: int) -> np.ndarray:
    """Distinct tokens in ``[-0.5, 0.5)``, shaped ``[1, 1, prompt, head]``."""
    count = int(prompt_len) * int(head_dim)
    values = np.arange(count, dtype=np.float32) / np.float32(count) - np.float32(0.5)
    return values.reshape(1, 1, int(prompt_len), int(head_dim))


def causal_bias_f16(prompt_len: int) -> np.ndarray:
    """Float16 ``[1, 1, prompt, prompt]`` causal mask."""
    width = int(prompt_len)
    bias = np.zeros((1, 1, width, width), dtype=np.float16)
    index = np.arange(width)
    future = index[None, :] > index[:, None]
    bias[0, 0][future] = np.float16(ATTENTION_MASK_BIAS)
    return bias


def reference_prefill_attention(
    hidden: np.ndarray,
    weight_q: np.ndarray,
    weight_k: np.ndarray,
    weight_v: np.ndarray,
    weight_o: np.ndarray,
    bias_f16: np.ndarray,
) -> tuple:
    """Float32 reference for the static prefill graph."""
    query = np.matmul(hidden, weight_q)
    key = np.matmul(hidden, weight_k)
    value = np.matmul(hidden, weight_v)
    scores = np.matmul(query, np.swapaxes(key, -1, -2))
    biased = scores.astype(np.float64) + bias_f16.astype(np.float64)
    biased = biased - np.max(biased, axis=-1, keepdims=True)
    weights = np.exp(biased)
    probs = (weights / np.sum(weights, axis=-1, keepdims=True)).astype(np.float32)
    context = np.matmul(probs, value)
    projected = np.matmul(context, weight_o)
    return key, value, projected


def build_prefill_attention_model(prompt_len: int = 8, head_dim: int = 64) -> ModelProto:
    """Project one static prompt and apply a lower-triangular mask."""
    prompt = int(prompt_len)
    head = int(head_dim)
    if prompt < 2:
        raise ValueError(f"prompt_len must cover more than one token, got {prompt}")
    if head < 16 or head % 16 != 0:
        raise ValueError(f"head_dim must be a multiple of 16, got {head}")
    hidden_shape = [1, 1, prompt, head]
    score_shape = [1, 1, prompt, prompt]
    flat = [prompt, head]
    square = [prompt, prompt]
    weight_q = default_projection(1, head, head)
    weight_k = default_projection(2, head, head)
    weight_v = default_projection(3, head, head)
    weight_o = default_projection(4, head, head)

    def i64(name: str, values: Sequence[int]):
        return helper.make_tensor(name, TensorProto.INT64, [len(values)], list(values))

    nodes = [
        helper.make_node("MatMul", ["hidden_states", "Wq"], ["q"], name="MatMul_q"),
        helper.make_node("MatMul", ["hidden_states", "Wk"], ["present.0.key"], name="MatMul_k"),
        helper.make_node("MatMul", ["hidden_states", "Wv"], ["present.0.value"], name="MatMul_v"),
        helper.make_node("Reshape", ["q", "flat_shape"], ["q_2d"], name="Reshape_q"),
        helper.make_node(
            "Reshape", ["present.0.key", "flat_shape"], ["k_2d"], name="Reshape_k"
        ),
        helper.make_node(
            "Gemm", ["q_2d", "k_2d"], ["scores_2d"], transB=1, name="Gemm_scores"
        ),
        helper.make_node(
            "Reshape", ["scores_2d", "score_shape"], ["scores"], name="Reshape_scores"
        ),
        helper.make_node(
            "Cast", ["attn_bias_f16"], ["attn_bias"], to=TensorProto.FLOAT, name="Cast_bias"
        ),
        helper.make_node("Add", ["scores", "attn_bias"], ["biased"], name="Add_bias"),
        helper.make_node("Softmax", ["biased"], ["probs"], axis=-1, name="Softmax_probs"),
        helper.make_node(
            "MatMul", ["probs", "present.0.value"], ["context"], name="MatMul_context"
        ),
        helper.make_node("MatMul", ["context", "Wo"], ["hidden_out"], name="MatMul_output"),
    ]
    graph = helper.make_graph(
        nodes,
        "prefill_attention",
        [helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, hidden_shape)],
        [
            helper.make_tensor_value_info("present.0.key", TensorProto.FLOAT, hidden_shape),
            helper.make_tensor_value_info("present.0.value", TensorProto.FLOAT, hidden_shape),
            helper.make_tensor_value_info("hidden_out", TensorProto.FLOAT, hidden_shape),
        ],
        initializer=[
            i64("flat_shape", flat),
            i64("score_shape", score_shape),
            numpy_helper.from_array(weight_q, "Wq"),
            numpy_helper.from_array(weight_k, "Wk"),
            numpy_helper.from_array(weight_v, "Wv"),
            numpy_helper.from_array(weight_o, "Wo"),
            numpy_helper.from_array(causal_bias_f16(prompt), "attn_bias_f16"),
        ],
        value_info=[
            helper.make_tensor_value_info("q", TensorProto.FLOAT, hidden_shape),
            helper.make_tensor_value_info("q_2d", TensorProto.FLOAT, flat),
            helper.make_tensor_value_info("k_2d", TensorProto.FLOAT, flat),
            helper.make_tensor_value_info("scores_2d", TensorProto.FLOAT, square),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, score_shape),
            helper.make_tensor_value_info("attn_bias", TensorProto.FLOAT, score_shape),
            helper.make_tensor_value_info("biased", TensorProto.FLOAT, score_shape),
            helper.make_tensor_value_info("probs", TensorProto.FLOAT, score_shape),
            helper.make_tensor_value_info("context", TensorProto.FLOAT, hidden_shape),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    op_types = [node.op_type for node in model.graph.node]
    hostile = [op for op in op_types if op in {"Transpose", "Where", "If", "Loop", "Scan", "Gather"}]
    if hostile:
        raise RuntimeError(f"prefill attention graph contains {hostile}")
    return model


def prefill_attention_facts(model: ModelProto) -> Dict[str, object]:
    """Op list and the float16 causal-mask dtype."""
    bias = None
    for initializer in model.graph.initializer:
        if initializer.name == "attn_bias_f16":
            bias = initializer
    return {
        "ops": [node.op_type for node in model.graph.node],
        "ir_version": int(model.ir_version),
        "bias_dtype": None if bias is None else int(bias.data_type),
    }
