"""Projected attention plus the PR #6 additive mask and an output projection.

The cache roll is the one-axis Slice and Concat from the measured KV
graph. Scores are ``present_key @ q_column``, reshaped to
``[1, 1, 1, seq]`` without a Transpose. The mask is a float16
initializer: position 0 is ``ATTENTION_MASK_BIAS`` (-10000) and the
other positions are 0. A Cast makes that tensor match the float32
scores, because Add has to use one dtype. A uniform -10000 tensor is
not used.

Measured on 2026-10-13 in conda ``ryzen-ai-1.8.0``, one head, query
length 1. At ``[1, 1, 8, 64]`` and ``[1, 1, 512, 128]`` the log said
``VAIML 15`` and subgraph ``NPU 1``. The profiler recorded one
``vitis_ai_ep_1`` kernel, and the compile selected ``GemmBfp16``. CPU
matched this reference within 6e-7. The NPU differed by at most 0.0023
and 0.0034 on the key, 0.0039 on the value, and 0.018 and 0.016 on the
projected output. Thirty later calls took 0.575 ms and 0.725 ms. The
log did not print an ELF size. At sequence 8 the mask moves the
reference output by 0.079, which is larger than the NPU gap. At
sequence 512 that move is 0.003, smaller than the NPU gap.

Heads stay in the batch axis. There is still no Transpose. At heads 8,
cache ``[1, 8, 256, 128]``, and query ``[1, 8, 1, 128]`` the log said
``VAIML 15`` and subgraph ``NPU 1``. The profiler recorded one
``vitis_ai_ep_1`` kernel, output size 2101248, and the compile
instantiated ``GemmBfp16``. No ELF size was printed. CPU matched within
1.2e-6. The NPU differed by 0.0051 on the key, 0.0057 on the value, and
0.023 on the projected output. The one-position mask moves that
reference by 0.00096, which is smaller than the NPU gap, so this shape
does not separately show that the mask survived bfloat16. Thirty later
calls took 0.617 ms on CPU and 1.762 ms on Vitis. The session load took
34.0 s. VirtualLock did not page-lock the 1048576-byte cache tensors.
"""

from typing import Dict, Sequence

import numpy as np
from onnx import ModelProto, TensorProto, helper, numpy_helper

from chunking_utils.onnx.attention_mask import ATTENTION_MASK_BIAS
from chunking_utils.onnx.golden_attention_subgraph import (
    IR_VERSION,
    OPSET,
    default_projection,
    default_token,
)

MASKED_PROJECTED_OPS = (
    "MatMul",
    "Reshape",
    "MatMul",
    "MatMul",
    "Slice",
    "Concat",
    "MatMul",
    "Reshape",
    "Cast",
    "Add",
    "Softmax",
    "Slice",
    "Concat",
    "MatMul",
    "MatMul",
)


def attention_bias_f16(seq_len: int) -> np.ndarray:
    """Float16 ``[1, 1, 1, seq]`` mask. Index 0 is blocked."""
    bias = np.zeros((1, 1, 1, int(seq_len)), dtype=np.float16)
    bias[0, 0, 0, 0] = np.float16(ATTENTION_MASK_BIAS)
    return bias


def reference_masked_projected_attention(
    past_key: np.ndarray,
    past_value: np.ndarray,
    hidden_states: np.ndarray,
    weight_q: np.ndarray,
    weight_k: np.ndarray,
    weight_v: np.ndarray,
    weight_o: np.ndarray,
    bias_f16: np.ndarray,
) -> tuple:
    """Float32 reference. The float16 bias is cast before the add."""
    query_row = np.matmul(hidden_states, weight_q)
    query_column = query_row.reshape(query_row.shape[:-2] + (query_row.shape[-1], 1))
    new_key = np.matmul(hidden_states, weight_k)
    new_value = np.matmul(hidden_states, weight_v)
    present_key = np.concatenate([past_key[:, :, 1:, :], new_key], axis=2)
    present_value = np.concatenate([past_value[:, :, 1:, :], new_value], axis=2)
    raw_scores = np.matmul(present_key, query_column)
    scores = raw_scores.reshape(raw_scores.shape[0], raw_scores.shape[1], 1, raw_scores.shape[2])
    biased = scores.astype(np.float64) + bias_f16.astype(np.float64)
    biased = biased - np.max(biased, axis=-1, keepdims=True)
    weights = np.exp(biased)
    probs = (weights / np.sum(weights, axis=-1, keepdims=True)).astype(np.float32)
    context = np.matmul(probs, present_value)
    projected = np.matmul(context, weight_o)
    return present_key, present_value, projected


def build_masked_projected_attention_model(
    seq_len: int = 8,
    head_dim: int = 64,
    query_len: int = 1,
    num_heads: int = 1,
) -> ModelProto:
    """Decode block with weights, a static mask, and ``Wo``.

    Query length stays 1, so scores are a column ``MatMul``. Extra heads
    use that same layout and do not add a Transpose.
    """
    if int(query_len) != 1:
        raise ValueError(f"query_len 1 is the measured layout, got {query_len}")
    if int(seq_len) < 2:
        raise ValueError(f"seq_len must leave one token to drop, got {seq_len}")
    if int(head_dim) < 16 or int(head_dim) % 16 != 0:
        raise ValueError(f"head_dim must be a multiple of 16, got {head_dim}")
    heads = int(num_heads)
    if heads < 1:
        raise ValueError(f"num_heads must be positive, got {heads}")
    head = int(head_dim)
    seq = int(seq_len)
    shape = [1, heads, seq, head]
    hidden_shape = [1, heads, 1, head]
    kept = [1, heads, seq - 1, head]
    column = [1, heads, head, 1]
    score_column = [1, heads, seq, 1]
    score_row = [1, heads, 1, seq]
    bias_shape = [1, 1, 1, seq]
    weight_q = default_projection(1, head, head)
    weight_k = default_projection(2, head, head)
    weight_v = default_projection(3, head, head)
    weight_o = default_projection(4, head, head)

    def i64(name: str, values: Sequence[int]):
        return helper.make_tensor(name, TensorProto.INT64, [len(values)], list(values))

    nodes = [
        helper.make_node("MatMul", ["hidden_states", "Wq"], ["q_row"], name="MatMul_q"),
        helper.make_node("Reshape", ["q_row", "q_shape"], ["q_col"], name="Reshape_q"),
        helper.make_node("MatMul", ["hidden_states", "Wk"], ["new_k"], name="MatMul_k"),
        helper.make_node("MatMul", ["hidden_states", "Wv"], ["new_v"], name="MatMul_v"),
        helper.make_node(
            "Slice",
            ["past_key_values.0.key", "starts", "ends", "axes", "steps"],
            ["kept_k"],
            name="Slice_key",
        ),
        helper.make_node(
            "Concat", ["kept_k", "new_k"], ["present.0.key"], axis=2, name="Concat_key"
        ),
        helper.make_node(
            "MatMul", ["present.0.key", "q_col"], ["scores_col"], name="MatMul_scores"
        ),
        helper.make_node(
            "Reshape", ["scores_col", "score_shape"], ["scores"], name="Reshape_scores"
        ),
        helper.make_node(
            "Cast", ["attn_bias_f16"], ["attn_bias"], to=TensorProto.FLOAT, name="Cast_bias"
        ),
        helper.make_node("Add", ["scores", "attn_bias"], ["biased"], name="Add_bias"),
        helper.make_node("Softmax", ["biased"], ["probs"], axis=-1, name="Softmax_probs"),
        helper.make_node(
            "Slice",
            ["past_key_values.0.value", "starts", "ends", "axes", "steps"],
            ["kept_v"],
            name="Slice_value",
        ),
        helper.make_node(
            "Concat",
            ["kept_v", "new_v"],
            ["present.0.value"],
            axis=2,
            name="Concat_value",
        ),
        helper.make_node(
            "MatMul", ["probs", "present.0.value"], ["context"], name="MatMul_context"
        ),
        helper.make_node(
            "MatMul", ["context", "Wo"], ["hidden_out"], name="MatMul_output"
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "masked_projected_attention",
        [
            helper.make_tensor_value_info(
                "past_key_values.0.key", TensorProto.FLOAT, shape
            ),
            helper.make_tensor_value_info(
                "past_key_values.0.value", TensorProto.FLOAT, shape
            ),
            helper.make_tensor_value_info(
                "hidden_states", TensorProto.FLOAT, hidden_shape
            ),
        ],
        [
            helper.make_tensor_value_info("present.0.key", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("present.0.value", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("hidden_out", TensorProto.FLOAT, hidden_shape),
        ],
        initializer=[
            i64("starts", [1]),
            i64("ends", [seq]),
            i64("axes", [2]),
            i64("steps", [1]),
            i64("q_shape", column),
            i64("score_shape", score_row),
            numpy_helper.from_array(weight_q, "Wq"),
            numpy_helper.from_array(weight_k, "Wk"),
            numpy_helper.from_array(weight_v, "Wv"),
            numpy_helper.from_array(weight_o, "Wo"),
            numpy_helper.from_array(attention_bias_f16(seq), "attn_bias_f16"),
        ],
        value_info=[
            helper.make_tensor_value_info("q_row", TensorProto.FLOAT, hidden_shape),
            helper.make_tensor_value_info("q_col", TensorProto.FLOAT, column),
            helper.make_tensor_value_info("new_k", TensorProto.FLOAT, hidden_shape),
            helper.make_tensor_value_info("new_v", TensorProto.FLOAT, hidden_shape),
            helper.make_tensor_value_info("kept_k", TensorProto.FLOAT, kept),
            helper.make_tensor_value_info("scores_col", TensorProto.FLOAT, score_column),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, score_row),
            helper.make_tensor_value_info("attn_bias", TensorProto.FLOAT, bias_shape),
            helper.make_tensor_value_info("biased", TensorProto.FLOAT, score_row),
            helper.make_tensor_value_info("probs", TensorProto.FLOAT, score_row),
            helper.make_tensor_value_info("kept_v", TensorProto.FLOAT, kept),
            helper.make_tensor_value_info("context", TensorProto.FLOAT, hidden_shape),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    op_types = [node.op_type for node in model.graph.node]
    hostile = [
        op
        for op in op_types
        if op in {"Transpose", "Scatter", "ScatterND", "ScatterElements", "Where"}
    ]
    if hostile:
        raise RuntimeError(f"masked projected graph contains {hostile}")
    return model


def masked_projected_facts(model: ModelProto) -> Dict[str, object]:
    """Op list and the float16 bias initializer dtype."""
    bias = None
    for initializer in model.graph.initializer:
        if initializer.name == "attn_bias_f16":
            bias = initializer
    return {
        "ops": [node.op_type for node in model.graph.node],
        "ir_version": int(model.ir_version),
        "bias_dtype": None if bias is None else int(bias.data_type),
        "bias_name": None if bias is None else bias.name,
    }
