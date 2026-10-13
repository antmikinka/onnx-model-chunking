"""
Attention subgraph that Vitis AI EP ran as one NPU kernel.

Measured on 2026-10-13 in conda ``ryzen-ai-1.8.0``, shape ``[1, 1, 8, 64]``,
float32, opset 17, IR 10. The log said ``VAIML 9``, ``NPU 1``, and
``Actually running on NPU 1``. The profiler recorded one kernel,
``vitis_ai_ep_1``, on ``VitisAIExecutionProvider``, with ``output_size``
4096. The compile selected ``GemmBfp16``. Against the float32 reference
the key differed by at most 0.00195 and the value by at most 0.0204.
One hundred later calls took 0.405 ms on that session and 0.041 ms on
CPU. The log did not print ``Generated ELF size``. The cache file was a
365364-byte ``.rai`` with no ELF header.

This is not a Qwen layer. ``Q`` is a constant column, and there are no
projection weights. ``splitter.py`` does not emit this graph. Sequence
length 8 is the length that ran; it is not a multiple of 16. Only
``query_len`` 1 was measured, because the score reshape is the same
eight bytes only in that case.

``build_projected_attention_model`` keeps that cache contract and adds
``Wq``, ``Wk``, and ``Wv``. The present value stores V. ``context`` is
a separate output. On the same machine and day, Vitis reported
``VAIML 12``, ``NPU 1``, and one ``vitis_ai_ep_1`` kernel with
``output_size`` 4352. CPU matched the float32 reference within 1.2e-7.
The NPU differed by at most 0.0023 on the key, 0.0039 on the value, and
0.011 on the context. One hundred later calls took 0.556 ms on that
session and 0.101 ms on CPU. The log did not print an ELF size.
"""

from typing import Dict, List, Sequence

import numpy as np
from onnx import ModelProto, TensorProto, helper, numpy_helper

OPSET = 17
IR_VERSION = 10
MEASURED_SHAPE = (1, 1, 8, 64)
FORBIDDEN_OPS = (
    "Transpose",
    "Scatter",
    "ScatterND",
    "ScatterElements",
    "Where",
)
_MEASURED_OPS = (
    "Slice",
    "Concat",
    "MatMul",
    "Softmax",
    "Reshape",
    "Slice",
    "Concat",
    "MatMul",
    "Concat",
)


def _require_measured_layout(seq_len: int, head_dim: int, query_len: int) -> None:
    if int(query_len) != 1:
        raise ValueError(
            "query_len 1 is the subgraph that ran on the NPU. "
            f"Got {query_len}."
        )
    if int(seq_len) < 2:
        raise ValueError(f"seq_len must leave one token to drop, got {seq_len}")
    if int(head_dim) < 16 or int(head_dim) % 16 != 0:
        raise ValueError(
            "head_dim 4 was rejected by the partitioner. "
            f"Use a multiple of 16, got {head_dim}."
        )


def default_query_column(head_dim: int) -> np.ndarray:
    """Constant Q used by the measured graph, shaped ``[1, 1, head_dim, 1]``."""
    width = int(head_dim)
    return np.linspace(-0.25, 0.25, width, dtype=np.float32).reshape(1, 1, width, 1)


def default_new_key(head_dim: int) -> np.ndarray:
    """Constant new key token used by the measured graph."""
    width = int(head_dim)
    return np.linspace(0.05, 0.35, width, dtype=np.float32).reshape(1, 1, 1, width)


def reference_attention(
    past_key: np.ndarray,
    past_value: np.ndarray,
    query_column: np.ndarray,
    new_key: np.ndarray,
    query_len: int = 1,
) -> tuple:
    """Float32 result of the golden graph. ``query_len`` tokens are dropped."""
    query = int(query_len)
    present_key = np.concatenate([past_key[:, :, query:, :], new_key], axis=2)
    scores = np.matmul(present_key, query_column)
    scores = scores - np.max(scores, axis=2, keepdims=True)
    weights = np.exp(scores)
    probs = weights / np.sum(weights, axis=2, keepdims=True)
    seq_len = past_key.shape[2]
    probs = probs.reshape(past_key.shape[0], past_key.shape[1], 1, seq_len)
    kept_value = past_value[:, :, query:, :]
    zero_value = np.zeros_like(new_key)
    rolled_value = np.concatenate([kept_value, zero_value], axis=2)
    context = np.matmul(probs, rolled_value)
    present_value = np.concatenate([kept_value, context], axis=2)
    return present_key, present_value


def build_golden_attention_model(
    seq_len: int = 8,
    head_dim: int = 64,
    batch: int = 1,
    heads: int = 1,
    query_len: int = 1,
) -> ModelProto:
    """
    Build the measured graph.

    Past and present key and value stay ``[batch, heads, seq_len, head_dim]``.
    ``Q`` is a constant ``[1, 1, head_dim, 1]``, so the key concat feeds
    ``MatMul`` with no ``Transpose`` node. Slice bounds are int64
    initializers.
    """
    _require_measured_layout(seq_len, head_dim, query_len)
    if int(batch) < 1 or int(heads) < 1:
        raise ValueError(f"batch and heads must be positive, got {batch}, {heads}")
    shape = [int(batch), int(heads), int(seq_len), int(head_dim)]
    query = default_query_column(head_dim)
    new_key = default_new_key(head_dim)
    zero_value = np.zeros((1, 1, 1, int(head_dim)), dtype=np.float32)
    kept = [int(batch), int(heads), int(seq_len) - 1, int(head_dim)]
    score_column = [int(batch), int(heads), int(seq_len), 1]
    score_row = [int(batch), int(heads), 1, int(seq_len)]
    context = [int(batch), int(heads), 1, int(head_dim)]

    def i64(name: str, values: Sequence[int]):
        return helper.make_tensor(name, TensorProto.INT64, [len(values)], list(values))

    nodes = [
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
            "MatMul", ["present.0.key", "query_col"], ["scores"], name="MatMul_scores"
        ),
        helper.make_node(
            "Softmax", ["scores"], ["probs_col"], axis=2, name="Softmax_probs"
        ),
        helper.make_node(
            "Reshape", ["probs_col", "score_shape"], ["probs"], name="Reshape_probs"
        ),
        helper.make_node(
            "Slice",
            ["past_key_values.0.value", "starts", "ends", "axes", "steps"],
            ["kept_v"],
            name="Slice_value",
        ),
        helper.make_node(
            "Concat", ["kept_v", "zero_v"], ["rolled_v"], axis=2, name="Concat_rolled_v"
        ),
        helper.make_node(
            "MatMul", ["probs", "rolled_v"], ["context"], name="MatMul_context"
        ),
        helper.make_node(
            "Concat",
            ["kept_v", "context"],
            ["present.0.value"],
            axis=2,
            name="Concat_value",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "golden_attention",
        [
            helper.make_tensor_value_info(
                "past_key_values.0.key", TensorProto.FLOAT, shape
            ),
            helper.make_tensor_value_info(
                "past_key_values.0.value", TensorProto.FLOAT, shape
            ),
        ],
        [
            helper.make_tensor_value_info("present.0.key", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("present.0.value", TensorProto.FLOAT, shape),
        ],
        initializer=[
            i64("starts", [int(query_len)]),
            i64("ends", [int(seq_len)]),
            i64("axes", [2]),
            i64("steps", [1]),
            i64("score_shape", score_row),
            numpy_helper.from_array(query, "query_col"),
            numpy_helper.from_array(new_key, "new_k"),
            numpy_helper.from_array(zero_value, "zero_v"),
        ],
        value_info=[
            helper.make_tensor_value_info("kept_k", TensorProto.FLOAT, kept),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, score_column),
            helper.make_tensor_value_info("probs_col", TensorProto.FLOAT, score_column),
            helper.make_tensor_value_info("probs", TensorProto.FLOAT, score_row),
            helper.make_tensor_value_info("kept_v", TensorProto.FLOAT, kept),
            helper.make_tensor_value_info("rolled_v", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("context", TensorProto.FLOAT, context),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    violations = hostile_attention_ops([node.op_type for node in model.graph.node])
    if violations:
        raise RuntimeError(f"golden attention graph contains {violations}")
    return model


def hostile_attention_ops(op_types: Sequence[str]) -> List[str]:
    """Ops that were absent from the subgraph the NPU accepted."""
    return [op for op in op_types if op in FORBIDDEN_OPS]


_PROJECTED_OPS = (
    "MatMul",
    "Reshape",
    "MatMul",
    "MatMul",
    "Slice",
    "Concat",
    "MatMul",
    "Softmax",
    "Reshape",
    "Slice",
    "Concat",
    "MatMul",
)


def default_token(hidden: int) -> np.ndarray:
    """One token, shaped ``[1, 1, 1, hidden]``."""
    width = int(hidden)
    return np.linspace(-0.5, 0.5, width, dtype=np.float32).reshape(1, 1, 1, width)


def default_projection(seed: int, hidden: int, head_dim: int) -> np.ndarray:
    """A fixed ``[hidden, head_dim]`` weight. Scale keeps the softmax finite."""
    rng = np.random.default_rng(int(seed))
    return rng.standard_normal((int(hidden), int(head_dim))).astype(np.float32) * np.float32(0.05)


def reference_projected_attention(
    past_key: np.ndarray,
    past_value: np.ndarray,
    token: np.ndarray,
    weight_q: np.ndarray,
    weight_k: np.ndarray,
    weight_v: np.ndarray,
    query_len: int = 1,
) -> tuple:
    """Float32 Q, K, and V projection plus the static cache roll."""
    query = int(query_len)
    query_row = np.matmul(token, weight_q)
    query_column = query_row.reshape(1, 1, query_row.shape[-1], 1)
    new_key = np.matmul(token, weight_k)
    new_value = np.matmul(token, weight_v)
    present_key = np.concatenate([past_key[:, :, query:, :], new_key], axis=2)
    present_value = np.concatenate([past_value[:, :, query:, :], new_value], axis=2)
    scores = np.matmul(present_key, query_column)
    scores = scores - np.max(scores, axis=2, keepdims=True)
    weights = np.exp(scores)
    probs = weights / np.sum(weights, axis=2, keepdims=True)
    probs = probs.reshape(past_key.shape[0], past_key.shape[1], 1, past_key.shape[2])
    context = np.matmul(probs, present_value)
    return present_key, present_value, context


def build_projected_attention_model(
    seq_len: int = 8,
    head_dim: int = 64,
    hidden: int = 64,
    query_len: int = 1,
) -> ModelProto:
    """
    Project one token into K and V, roll the static cache, and score it.

    Past and present stay ``[1, 1, seq_len, head_dim]``. ``Q`` is reshaped
    from ``[1, 1, 1, head_dim]`` to ``[1, 1, head_dim, 1]``. For one token
    those are the same bytes, so the key concat feeds ``MatMul`` with no
    ``Transpose``. The present value is the cache, not the attention output.
    """
    _require_measured_layout(seq_len, head_dim, query_len)
    if int(hidden) != int(head_dim):
        raise ValueError(
            f"hidden must match head_dim for this probe, got {hidden} and {head_dim}"
        )
    shape = [1, 1, int(seq_len), int(head_dim)]
    token_shape = [1, 1, 1, int(hidden)]
    kept = [1, 1, int(seq_len) - 1, int(head_dim)]
    row = [1, 1, 1, int(head_dim)]
    column = [1, 1, int(head_dim), 1]
    score_column = [1, 1, int(seq_len), 1]
    score_row = [1, 1, 1, int(seq_len)]
    weight_q = default_projection(1, hidden, head_dim)
    weight_k = default_projection(2, hidden, head_dim)
    weight_v = default_projection(3, hidden, head_dim)

    def i64(name: str, values: Sequence[int]):
        return helper.make_tensor(name, TensorProto.INT64, [len(values)], list(values))

    nodes = [
        helper.make_node("MatMul", ["token", "Wq"], ["q_row"], name="MatMul_q"),
        helper.make_node("Reshape", ["q_row", "q_shape"], ["q_col"], name="Reshape_q"),
        helper.make_node("MatMul", ["token", "Wk"], ["new_k"], name="MatMul_k"),
        helper.make_node("MatMul", ["token", "Wv"], ["new_v"], name="MatMul_v"),
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
            "MatMul", ["present.0.key", "q_col"], ["scores"], name="MatMul_scores"
        ),
        helper.make_node(
            "Softmax", ["scores"], ["probs_col"], axis=2, name="Softmax_probs"
        ),
        helper.make_node(
            "Reshape", ["probs_col", "score_shape"], ["probs"], name="Reshape_probs"
        ),
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
            "MatMul",
            ["probs", "present.0.value"],
            ["context"],
            name="MatMul_context",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "projected_attention",
        [
            helper.make_tensor_value_info(
                "past_key_values.0.key", TensorProto.FLOAT, shape
            ),
            helper.make_tensor_value_info(
                "past_key_values.0.value", TensorProto.FLOAT, shape
            ),
            helper.make_tensor_value_info("token", TensorProto.FLOAT, token_shape),
        ],
        [
            helper.make_tensor_value_info("present.0.key", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("present.0.value", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("context", TensorProto.FLOAT, row),
        ],
        initializer=[
            i64("starts", [int(query_len)]),
            i64("ends", [int(seq_len)]),
            i64("axes", [2]),
            i64("steps", [1]),
            i64("q_shape", column),
            i64("score_shape", score_row),
            numpy_helper.from_array(weight_q, "Wq"),
            numpy_helper.from_array(weight_k, "Wk"),
            numpy_helper.from_array(weight_v, "Wv"),
        ],
        value_info=[
            helper.make_tensor_value_info("q_row", TensorProto.FLOAT, row),
            helper.make_tensor_value_info("q_col", TensorProto.FLOAT, column),
            helper.make_tensor_value_info("new_k", TensorProto.FLOAT, row),
            helper.make_tensor_value_info("new_v", TensorProto.FLOAT, row),
            helper.make_tensor_value_info("kept_k", TensorProto.FLOAT, kept),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, score_column),
            helper.make_tensor_value_info("probs_col", TensorProto.FLOAT, score_column),
            helper.make_tensor_value_info("probs", TensorProto.FLOAT, score_row),
            helper.make_tensor_value_info("kept_v", TensorProto.FLOAT, kept),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    violations = hostile_attention_ops([node.op_type for node in model.graph.node])
    if violations:
        raise RuntimeError(f"projected attention graph contains {violations}")
    return model


def golden_attention_facts(model: ModelProto) -> Dict[str, object]:
    """Op list, opset, IR version, and Slice initializer names."""
    opset = None
    for imported in model.opset_import:
        if imported.domain in ("", "ai.onnx"):
            opset = int(imported.version)
    return {
        "ops": [node.op_type for node in model.graph.node],
        "opset": opset,
        "ir_version": int(model.ir_version),
        "initializers": sorted(item.name for item in model.graph.initializer),
        "measured_ops": list(_MEASURED_OPS),
        "projected_ops": list(_PROJECTED_OPS),
    }
