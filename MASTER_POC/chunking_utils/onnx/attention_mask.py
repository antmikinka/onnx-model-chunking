"""
Additive attention masks for spatial NPUs.

Stephen Panaro's ``gpt2.py`` does not call ``masked_fill``. A stored
``[T, T]`` bias is copied into the model proto once per layer, and mask-fill
runs on the CPU. The form that stays on the accelerator is:

    att = att + attention_mask
    att = softmax(att)

Blocked positions in that mask are -1e4. That value is exact in float16.
``-inf`` and float16's minimum finite value (-65504) are not: the minimum
overflows when a causal mask and a padding mask are added, and ``-inf``
is a bad softmax input on a spatial NPU.

This module rewrites the attention-score form of that fill:

- ``Where(keep, 0, sentinel)`` becomes ``(1 - keep) * -10000``
- ``Where(blocked, sentinel, 0)`` becomes ``blocked * -10000``
- ``Where(blocked, sentinel, scores)`` becomes ``scores + blocked * -10000``
- ``Where(keep, scores, sentinel)`` becomes ``scores + (1 - keep) * -10000``
- ``MaskedFill(scores, blocked, sentinel)`` becomes the same additive form

``sentinel`` is a scalar at or below -1000, or a non-finite negative value.
The scalar may sit behind Identity, Cast, Expand, Unsqueeze, Squeeze, or
ConstantOfShape. The replacement uses that branch's floating dtype so Add
type-checks. Shape repairs, GELU ``IsInf`` selects, and the lm_head token
ban are left as Where ops. A GroupQueryAttention seqlens mask is also left
alone: that fused op does not consume an additive bias.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


# -1e4. Exact in float16, and exp(-10000) is 0 in float16 and float32.
ATTENTION_MASK_BIAS = -10000.0
_SENTINEL_FLOOR = -1000.0
_ATTENTION_TOKENS = ("attn", "attention", "causal", "mask")
_EXCLUDED_TOKENS = ("lm_head", "logit")
_PASSTHROUGH = {"Identity", "Expand", "Unsqueeze", "Squeeze"}
_FLOAT_DTYPES = {
    TensorProto.FLOAT,
    TensorProto.FLOAT16,
    TensorProto.BFLOAT16,
    TensorProto.DOUBLE,
}
# Consumers checked at this depth and at every shallower depth, so a mask
# Where a few ops before Softmax still counts. Fused MultiHeadAttention has
# no Softmax node; those masks are selected by name instead.
_SOFTMAX_DEPTH = 5


class _Scalar:
    __slots__ = ("value", "dtype")

    def __init__(self, value: float, dtype: int):
        self.value = value
        self.dtype = dtype


def _is_sentinel(value: float) -> bool:
    if not math.isfinite(value):
        return value < 0
    return value <= _SENTINEL_FLOOR


def _is_zero(value: float) -> bool:
    return math.isfinite(value) and value == 0.0


def _scalar_from_tensor(tensor) -> Optional[_Scalar]:
    if tensor.data_type == TensorProto.BFLOAT16 and tensor.raw_data:
        bits = np.frombuffer(tensor.raw_data, dtype=np.uint16)
        if bits.size != 1:
            return None
        value = float((bits.astype(np.uint32) << 16).view(np.float32)[0])
        return _Scalar(value, TensorProto.BFLOAT16)
    try:
        array = numpy_helper.to_array(tensor)
    except Exception:
        return None
    if array.size != 1:
        return None
    return _Scalar(float(array.reshape(-1)[0]), int(tensor.data_type))


def _producers(graph) -> Dict[str, onnx.NodeProto]:
    mapping: Dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for name in node.output:
            if name:
                mapping[name] = node
    return mapping


def _known_dtypes(graph) -> Dict[str, int]:
    found: Dict[str, int] = {}
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type:
            found[value.name] = int(tensor_type.elem_type)
    for initializer in graph.initializer:
        found[initializer.name] = int(initializer.data_type)
    for node in graph.node:
        if not node.output:
            continue
        if node.op_type == "Cast":
            for attribute in node.attribute:
                if attribute.name == "to" and attribute.i:
                    found[node.output[0]] = int(attribute.i)
        elif node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.name == "value" and attribute.HasField("t"):
                    found[node.output[0]] = int(attribute.t.data_type)
                elif attribute.name == "value_float":
                    found[node.output[0]] = TensorProto.FLOAT
        elif node.op_type == "ConstantOfShape":
            data_type = TensorProto.FLOAT
            for attribute in node.attribute:
                if attribute.name == "value" and attribute.HasField("t") and attribute.t.data_type:
                    data_type = int(attribute.t.data_type)
            found[node.output[0]] = data_type
    return found


def _direct_scalars(graph) -> Dict[str, _Scalar]:
    found: Dict[str, _Scalar] = {}
    for initializer in graph.initializer:
        scalar = _scalar_from_tensor(initializer)
        if scalar is not None:
            found[initializer.name] = scalar
    for node in graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        for attribute in node.attribute:
            scalar = None
            if attribute.name == "value" and attribute.HasField("t"):
                scalar = _scalar_from_tensor(attribute.t)
            elif attribute.name == "value_float":
                scalar = _Scalar(float(attribute.f), TensorProto.FLOAT)
            elif attribute.name == "value_floats" and len(attribute.floats) == 1:
                scalar = _Scalar(float(attribute.floats[0]), TensorProto.FLOAT)
            if scalar is not None:
                found[node.output[0]] = scalar
    return found


def _attribute_int(node, name: str) -> Optional[int]:
    for attribute in node.attribute:
        if attribute.name == name:
            return int(attribute.i)
    return None


def _resolve_scalar(
    name: str,
    direct: Dict[str, _Scalar],
    producers: Dict[str, onnx.NodeProto],
    seen: set,
    depth: int,
) -> Optional[_Scalar]:
    if not name or name in seen or depth < 0:
        return None
    if name in direct:
        return direct[name]
    node = producers.get(name)
    if node is None:
        return None
    seen.add(name)
    if node.op_type == "ConstantOfShape":
        for attribute in node.attribute:
            if attribute.name == "value" and attribute.HasField("t"):
                return _scalar_from_tensor(attribute.t)
        return _Scalar(0.0, TensorProto.FLOAT)
    if node.op_type == "Cast" and node.input:
        scalar = _resolve_scalar(node.input[0], direct, producers, seen, depth - 1)
        if scalar is None:
            return None
        cast_to = _attribute_int(node, "to")
        if cast_to:
            return _Scalar(scalar.value, cast_to)
        return scalar
    if node.op_type in _PASSTHROUGH and node.input:
        return _resolve_scalar(node.input[0], direct, producers, seen, depth - 1)
    return None


def _consumers(graph) -> Dict[str, List[onnx.NodeProto]]:
    mapping: Dict[str, List[onnx.NodeProto]] = {}
    for node in graph.node:
        for name in node.input:
            mapping.setdefault(name, []).append(node)
    return mapping


def _reaches_softmax(name: str, consumers: Dict[str, List[onnx.NodeProto]], depth: int) -> bool:
    if depth < 0 or not name:
        return False
    for node in consumers.get(name, []):
        if node.op_type == "Softmax":
            return True
        for output in node.output:
            if _reaches_softmax(output, consumers, depth - 1):
                return True
    return False


def _attention_where(node, consumers: Dict[str, List[onnx.NodeProto]]) -> bool:
    text = " ".join([node.name, *list(node.input), *list(node.output)]).lower()
    if any(token in text for token in _EXCLUDED_TOKENS):
        return False
    if any(token in text for token in _ATTENTION_TOKENS):
        return True
    if not node.output:
        return False
    return _reaches_softmax(node.output[0], consumers, _SOFTMAX_DEPTH)


def _numpy_dtype(dtype: int):
    if dtype == TensorProto.FLOAT16:
        return np.float16
    if dtype == TensorProto.DOUBLE:
        return np.float64
    return np.float32


def _scalar_tensor(name: str, value: float, dtype: int):
    if dtype == TensorProto.BFLOAT16:
        bits = (np.array(value, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
        tensor = TensorProto()
        tensor.name = name
        tensor.data_type = TensorProto.BFLOAT16
        tensor.raw_data = bits.tobytes()
        return tensor
    return numpy_helper.from_array(np.array(value, dtype=_numpy_dtype(dtype)), name=name)


def _masked_fill_value(node, resolved: Dict[str, _Scalar]) -> Optional[_Scalar]:
    for attribute in node.attribute:
        if attribute.name == "value" and attribute.HasField("f"):
            return _Scalar(float(attribute.f), TensorProto.FLOAT)
        if attribute.name == "value" and attribute.HasField("t"):
            return _scalar_from_tensor(attribute.t)
    if len(node.input) >= 3:
        return resolved.get(node.input[2])
    return None


def _where_plan(
    node,
    resolved: Dict[str, _Scalar],
    known: Dict[str, int],
    fallback_dtype: int,
) -> Optional[Tuple[str, str, bool, int]]:
    """
    Return ``(kind, scores_or_condition, blocked_when_condition_true, dtype)``.

    ``kind`` is ``mask`` when the Where builds the bias, and ``scores``
    when it selects between the scores and the sentinel.
    """
    if node.op_type == "MaskedFill":
        value = _masked_fill_value(node, resolved)
        if value is None or not _is_sentinel(value.value) or len(node.input) < 2:
            return None
        data_dtype = known.get(node.input[0], fallback_dtype)
        if data_dtype not in _FLOAT_DTYPES:
            data_dtype = value.dtype if value.dtype in _FLOAT_DTYPES else fallback_dtype
        return ("scores", node.input[0], True, data_dtype)

    if node.op_type != "Where" or len(node.input) < 3 or not node.output:
        return None
    condition, left, right = node.input[0], node.input[1], node.input[2]
    left_scalar = resolved.get(left)
    right_scalar = resolved.get(right)
    left_sentinel = left_scalar is not None and _is_sentinel(left_scalar.value)
    right_sentinel = right_scalar is not None and _is_sentinel(right_scalar.value)
    left_zero = left_scalar is not None and _is_zero(left_scalar.value)
    right_zero = right_scalar is not None and _is_zero(right_scalar.value)
    if left_zero and right_sentinel and right_scalar.dtype in _FLOAT_DTYPES:
        return ("mask", condition, False, right_scalar.dtype)
    if left_sentinel and right_zero and left_scalar.dtype in _FLOAT_DTYPES:
        return ("mask", condition, True, left_scalar.dtype)
    if left_sentinel and right_scalar is None and left_scalar.dtype in _FLOAT_DTYPES:
        return ("scores", right, True, _score_dtype(left_scalar.dtype, known.get(right)))
    if right_sentinel and left_scalar is None and right_scalar.dtype in _FLOAT_DTYPES:
        return ("scores", left, False, _score_dtype(right_scalar.dtype, known.get(left)))
    return None


def _score_dtype(sentinel_dtype: int, score_dtype: Optional[int]) -> int:
    """Add has to match the scores. The value written is still -10000."""
    if score_dtype in _FLOAT_DTYPES:
        return score_dtype
    return sentinel_dtype


def _clone(node) -> onnx.NodeProto:
    copied = onnx.NodeProto()
    copied.CopyFrom(node)
    return copied


def _nodes_for(
    node,
    plan: Tuple[str, str, bool, int],
    bias_name: str,
    one_name: str,
) -> Sequence[onnx.NodeProto]:
    kind, other, blocked_when_true, dtype = plan
    condition = node.input[1] if node.op_type == "MaskedFill" else node.input[0]
    stem = node.name or node.output[0]
    cast_out = f"{stem}/bias_cast_out"
    mul_out = f"{stem}/bias_mul_out"
    nodes = [
        helper.make_node("Cast", [condition], [cast_out], name=f"{stem}/bias_cast", to=dtype),
    ]
    factor = cast_out
    if not blocked_when_true:
        sub_out = f"{stem}/bias_blocked_out"
        nodes.append(
            helper.make_node("Sub", [one_name, cast_out], [sub_out], name=f"{stem}/bias_sub")
        )
        factor = sub_out
    if kind == "mask":
        nodes.append(
            helper.make_node("Mul", [factor, bias_name], list(node.output), name=f"{stem}/bias_mul")
        )
        return nodes
    nodes.append(helper.make_node("Mul", [factor, bias_name], [mul_out], name=f"{stem}/bias_mul"))
    nodes.append(
        helper.make_node("Add", [other, mul_out], list(node.output), name=f"{stem}/bias_add")
    )
    return nodes


def _captured_names(graph) -> set:
    """Outer tensors read by a subgraph. Those producers have to stay."""
    captured = set()

    def walk(container):
        for node in container.node:
            for attribute in node.attribute:
                subgraphs = []
                if attribute.HasField("g"):
                    subgraphs.append(attribute.g)
                subgraphs.extend(list(attribute.graphs))
                for subgraph in subgraphs:
                    produced = {output for inner in subgraph.node for output in inner.output if output}
                    produced.update(value.name for value in subgraph.input)
                    for inner in subgraph.node:
                        for name in inner.input:
                            if name and name not in produced:
                                captured.add(name)
                    walk(subgraph)

    walk(graph)
    return captured


def _drop_fill_chain(nodes, seeds, graph_outputs, captured):
    """Remove the unread fill that used to feed a replaced Where.

    A producer is removed only when every output is unread, so a shared
    constant stays. Unrelated dead nodes stay too. The returned input
    names are the initializers that no longer have a reader.
    """
    live = list(nodes)
    pending = {name for name in seeds if name}
    dropped_inputs = set()
    while pending:
        name = pending.pop()
        if name in graph_outputs or name in captured:
            continue
        if any(name in node.input for node in live):
            continue
        producer = next((node for node in live if name in node.output), None)
        if producer is None:
            dropped_inputs.add(name)
            continue
        outputs = [output for output in producer.output if output]
        if any(
            output in graph_outputs
            or output in captured
            or any(output in node.input for node in live)
            for output in outputs
        ):
            continue
        live.remove(producer)
        dropped_inputs.update(outputs)
        for input_name in producer.input:
            if input_name:
                pending.add(input_name)
    return live, dropped_inputs


def _rewrite_graph(graph, dtype: int) -> int:
    direct = _direct_scalars(graph)
    producers = _producers(graph)
    resolved: Dict[str, _Scalar] = {}
    for name in list(direct) + [output for node in graph.node for output in node.output]:
        if name in resolved:
            continue
        scalar = _resolve_scalar(name, direct, producers, set(), 4)
        if scalar is not None:
            resolved[name] = scalar
    known = _known_dtypes(graph)
    consumers = _consumers(graph)
    rebuilt = []
    replaced = 0
    dtypes = set()
    one_dtypes = set()
    fill_seeds = set()
    for node in graph.node:
        plan = _where_plan(node, resolved, known, dtype)
        if plan is None or not _attention_where(node, consumers):
            rebuilt.append(_clone(node))
            continue
        new_nodes = list(
            _nodes_for(
                node,
                plan,
                f"spatial_npu_attention_bias_{plan[3]}",
                f"spatial_npu_attention_one_{plan[3]}",
            )
        )
        new_inputs = {name for created in new_nodes for name in created.input}
        fill_seeds.update(name for name in node.input if name and name not in new_inputs)
        rebuilt.extend(new_nodes)
        dtypes.add(plan[3])
        if not plan[2]:
            one_dtypes.add(plan[3])
        replaced += 1
    if not replaced:
        return 0
    graph_outputs = {value.name for value in graph.output}
    rebuilt, dropped_inputs = _drop_fill_chain(
        rebuilt,
        fill_seeds,
        graph_outputs,
        _captured_names(graph),
    )
    existing = {initializer.name for initializer in graph.initializer}
    for used in sorted(dtypes):
        bias_name = f"spatial_npu_attention_bias_{used}"
        if bias_name not in existing:
            graph.initializer.append(_scalar_tensor(bias_name, ATTENTION_MASK_BIAS, used))
            existing.add(bias_name)
        if used in one_dtypes:
            one_name = f"spatial_npu_attention_one_{used}"
            if one_name not in existing:
                graph.initializer.append(_scalar_tensor(one_name, 1.0, used))
                existing.add(one_name)
    live_inputs = {name for node in rebuilt for name in node.input if name}
    input_names = {value.name for value in graph.input}
    kept_initializers = []
    for initializer in graph.initializer:
        unused_fill = (
            initializer.name in dropped_inputs
            and initializer.name not in live_inputs
            and initializer.name not in input_names
            and initializer.name not in graph_outputs
        )
        if unused_fill:
            continue
        copied = onnx.TensorProto()
        copied.CopyFrom(initializer)
        kept_initializers.append(copied)
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)
    del graph.node[:]
    graph.node.extend(rebuilt)
    return replaced


def rewrite_attention_mask_ops(model, dtype: int = TensorProto.FLOAT16) -> int:
    """
    Replace attention Where and MaskedFill with an additive -1e4 bias.

    Returns the number of nodes replaced. The sequence length is unchanged.
    A Where that does not match the sentinel pattern, or that belongs to
    the lm_head, stays in the graph. Zero replacements leave the graph
    untouched.
    """
    if dtype not in _FLOAT_DTYPES:
        dtype = TensorProto.FLOAT16
    return _rewrite_graph(model.graph, dtype)
