"""
Static shapes for an AOT NPU graph.

A symbolic axis or a Slice whose end comes from the token count cannot be
compiled onto a spatial array. The cache is allocated at ``max_seq_len``.
Positions past the valid prefix are an additive -10000 bias, not a shorter
view of the cache. ``session.disable_node_shape_check`` is left for a CPU
graph that is still dynamic.

A Qwen layer that still contains ``GroupQueryAttention`` is not the
decomposed MatMul graph measured on the NPU. ``apply_static_contract``
freezes that layer's symbolic axes, including ``kv_cache_dim`` when a
q_norm or k_norm weight declares the head size. It replaces the runtime
seqlens and total-sequence inputs with constants of the window and adds
an ``attention_bias`` input for the -10000 tail. That rewrite was checked
on the local Qwen3-8B ``layer_0_of_36.onnx``. No Vitis compile of that
layer was run.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

ADDITIVE_BIAS = -10000.0

_BATCH = {"batch", "batch_size"}
_QUERY = {
    "sequence_length",
    "seq_len",
    "query_length",
    "source_sequence_length",
    "seq",
}
_WINDOW = {
    "past_sequence_length",
    "total_sequence_length",
    "max_sequence_length",
    "max_seq_len",
    "target_sequence_length",
}
_STRUCTURAL = {
    "num_kv_heads",
    "num_heads",
    "num_attention_heads",
    "hidden",
    "hidden_size",
    "vocab",
    "vocab_size",
    "unk",
}
_HEAD = {"kv_cache_dim", "head_dim", "head_size"}
_AOT_PROVIDERS = ("vitisai", "ryzenai", "openvino", "coreml")


class StaticShapeError(ValueError):
    """A symbolic dimension or a runtime slice cannot be compiled onto an AOT NPU."""


def allow_shape_check_bypass(
    providers: Optional[Sequence[str]],
    disable_shape_check: bool,
    static_kv_cache: bool = False,
) -> bool:
    """
    Whether ORT may skip node shape checking.

    The key is a band-aid for a CPU graph that still has dynamic dims.
    An AOT provider and a preallocated KV cache keep the check on.
    """
    if not disable_shape_check or static_kv_cache:
        return False
    text = " ".join(str(provider) for provider in (providers or [])).lower()
    if any(token in text for token in _AOT_PROVIDERS):
        return False
    return True


def freeze_dim_name(
    dim: object,
    *,
    batch_size: int,
    query_len: int,
    max_seq_len: int,
    head_size: Optional[int] = None,
) -> Optional[int]:
    """
    Map a builder shape string to a concrete size.

    ``None`` means the string is not a batch or sequence axis and the caller
    should keep it. ``unk`` raises: that axis is a runtime rank.
    """
    if not isinstance(dim, str):
        return None
    key = dim.strip().lower()
    if key in _BATCH:
        return int(batch_size)
    if key in _QUERY:
        return int(query_len)
    if key in _WINDOW:
        return int(max_seq_len)
    if key == "head_dim / 2":
        if head_size is None:
            raise StaticShapeError("head_dim / 2 needs a concrete head size")
        return int(head_size) // 2
    if key == "unk":
        raise StaticShapeError("static_shapes cannot leave a symbolic 'unk' dimension")
    return None


def _role(name: str) -> str:
    lower = name.lower()
    if "attention_mask" in lower or lower.endswith("mask"):
        return "mask"
    if (
        "past_key" in lower
        or "present" in lower
        or "key_cache" in lower
        or "value_cache" in lower
        or lower.endswith(".key")
        or lower.endswith(".value")
    ):
        return "kv"
    return "query"


def _size_for(
    param: str,
    role: str,
    batch_size: int,
    query_len: int,
    max_seq_len: int,
    head_dim: Optional[int],
) -> int:
    text = param.strip().lower()
    if text in _STRUCTURAL or text == "":
        raise StaticShapeError(f"Cannot freeze symbolic dimension '{param}'")
    if text in _HEAD:
        if head_dim is None:
            raise StaticShapeError(f"Cannot freeze '{param}' without a concrete head dim")
        return int(head_dim)
    if text in _BATCH:
        return int(batch_size)
    if role == "kv" and (text in _QUERY or text in _WINDOW or "seq" in text):
        return int(max_seq_len)
    if role == "mask" and (text in _QUERY or text in _WINDOW or "seq" in text):
        return int(max_seq_len)
    if text in _WINDOW:
        return int(max_seq_len)
    if text in _QUERY or "seq" in text:
        return int(query_len)
    raise StaticShapeError(f"Cannot freeze symbolic dimension '{param}'")


def _freeze_value(
    value,
    *,
    batch_size: int,
    query_len: int,
    max_seq_len: int,
    head_dim: Optional[int],
) -> int:
    if not value.type.HasField("tensor_type"):
        return 0
    changed = 0
    role = _role(value.name)
    for dim in value.type.tensor_type.shape.dim:
        if dim.dim_param:
            size = _size_for(
                dim.dim_param, role, batch_size, query_len, max_seq_len, head_dim
            )
            dim.ClearField("dim_param")
            dim.dim_value = int(size)
            changed += 1
            continue
        if dim.dim_value > 0:
            continue
        raise StaticShapeError(
            f"Tensor '{value.name}' has an unknown axis and no symbolic name"
        )
    return changed


def infer_head_dim(model) -> int:
    """Head size declared by a Qwen q_norm or k_norm weight."""
    for item in model.graph.initializer:
        lower = item.name.lower()
        if not (
            lower.endswith("q_norm.layernorm.weight")
            or lower.endswith("k_norm.layernorm.weight")
        ):
            continue
        if len(item.dims) == 1 and int(item.dims[0]) >= 16:
            return int(item.dims[0])
    raise StaticShapeError(
        "kv_cache_dim needs head_dim, and no q_norm or k_norm weight declared it"
    )


def _graph_needs_head(model) -> bool:
    values = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    for value in values:
        if not value.type.HasField("tensor_type"):
            continue
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_param and dim.dim_param.strip().lower() in _HEAD:
                return True
    return False


def freeze_graph_io(
    model,
    *,
    batch_size: int,
    query_len: int,
    max_seq_len: int,
    head_dim: Optional[int] = None,
) -> int:
    """Replace symbolic input, value, and output axes with the static window."""
    if batch_size < 1 or query_len < 1 or max_seq_len < 1:
        raise StaticShapeError(
            f"batch, query, and max_seq must be positive, "
            f"got {batch_size}, {query_len}, {max_seq_len}"
        )
    if query_len > max_seq_len:
        raise StaticShapeError(
            f"query length {query_len} is longer than max_seq_len {max_seq_len}"
        )
    if head_dim is None and _graph_needs_head(model):
        head_dim = infer_head_dim(model)
    graph = model.graph
    changed = 0
    for value in list(graph.input) + list(graph.value_info) + list(graph.output):
        changed += _freeze_value(
            value,
            batch_size=batch_size,
            query_len=query_len,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
        )
    return changed


def _constant_names(graph) -> set:
    names = {item.name for item in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            names.update(out for out in node.output if out)
    return names


def _constant_vector(graph, name: str):
    if not name:
        return None
    for item in graph.initializer:
        if item.name == name:
            from onnx import numpy_helper

            return np.asarray(numpy_helper.to_array(item)).reshape(-1)
    for node in graph.node:
        if node.op_type != "Constant" or name not in node.output:
            continue
        from onnx import numpy_helper

        for attr in node.attribute:
            if attr.name == "value":
                return np.asarray(numpy_helper.to_array(attr.t)).reshape(-1)
    return None


def _is_kv(name: str) -> bool:
    return _role(name) == "kv"


def runtime_slices(model) -> List[str]:
    """Slice nodes whose starts or ends are not constants."""
    graph = model.graph
    constants = _constant_names(graph)
    found = []
    for node in graph.node:
        if node.op_type != "Slice" or len(node.input) < 3:
            continue
        starts = node.input[1]
        ends = node.input[2]
        if starts not in constants or ends not in constants:
            found.append(node.name or node.output[0] or node.input[0])
    return found


def rewrite_runtime_kv_slices(model, max_seq_len: int, seq_axis: int = 2) -> List[str]:
    """
    Replace ``cache[:, :valid, :]`` with a slice of the whole preallocated axis.

    The start must be open (constant 0 or itself dynamic). Any other runtime
    Slice raises. The additive bias, not this slice, hides the tail.
    """
    from onnx import numpy_helper

    if max_seq_len < 1:
        raise StaticShapeError(f"max_seq_len must be positive, got {max_seq_len}")
    graph = model.graph
    constants = _constant_names(graph)
    rewritten = []
    detached = []
    accepted_axes = {int(seq_axis), int(seq_axis) - 4}
    for node in graph.node:
        if node.op_type != "Slice" or len(node.input) < 3:
            continue
        data, starts, ends = node.input[0], node.input[1], node.input[2]
        if starts in constants and ends in constants:
            continue
        label = node.name or data
        if not _is_kv(data):
            raise StaticShapeError(
                f"Slice '{label}' uses a runtime index. "
                "Only a KV prefix slice can be widened to max_seq_len."
            )
        axes_name = node.input[3] if len(node.input) > 3 else ""
        axes = _constant_vector(graph, axes_name)
        axes_list = None if axes is None else [int(item) for item in axes]
        if axes_list not in ([axis] for axis in accepted_axes):
            raise StaticShapeError(
                f"Slice '{label}' axes {axes_list} are not the sequence axis {seq_axis}"
            )
        if len(node.input) > 4 and node.input[4]:
            steps = _constant_vector(graph, node.input[4]) if node.input[4] in constants else None
            if steps is None or any(int(item) != 1 for item in steps):
                raise StaticShapeError(f"Slice '{label}' has a non-unit step")
        if starts in constants:
            start_vec = _constant_vector(graph, starts)
            if start_vec is None or [int(item) for item in start_vec] != [0]:
                raise StaticShapeError(
                    f"Slice '{label}' does not start at 0, so it is not a prefix slice"
                )
        else:
            start_name = f"{label}_static_start"
            graph.initializer.append(
                numpy_helper.from_array(np.asarray([0], dtype=np.int64), start_name)
            )
            detached.append(starts)
            node.input[1] = start_name
            constants.add(start_name)
        end_name = f"{label}_static_end_{len(rewritten)}"
        graph.initializer.append(
            numpy_helper.from_array(np.asarray([int(max_seq_len)], dtype=np.int64), end_name)
        )
        detached.append(ends)
        node.input[2] = end_name
        constants.add(end_name)
        rewritten.append(label)
    if detached:
        used = set()
        for node in graph.node:
            used.update(item for item in node.input if item)
        kept = [value for value in graph.input if value.name not in detached or value.name in used]
        del graph.input[:]
        graph.input.extend(kept)
    return rewritten


def _drop_unused_inputs(graph, detached: Sequence[str]) -> None:
    if not detached:
        return
    used = set()
    for node in graph.node:
        used.update(item for item in node.input if item)
    removed = set(detached)
    kept = [value for value in graph.input if value.name not in removed or value.name in used]
    del graph.input[:]
    graph.input.extend(kept)


def rewrite_gqa_to_static_window(
    model,
    *,
    batch_size: int,
    query_len: int,
    max_seq_len: int,
) -> List[str]:
    """
    Point GroupQueryAttention at the static window.

    ``seqlens_k`` and ``total_sequence_length`` become constants of
    ``max_seq_len``. An empty ``attention_bias`` slot becomes a graph input
    of shape ``[1, 1, query_len, max_seq_len]``. The host fills that tensor
    with ``static_attention_bias``. The fused op is left in the graph.
    """
    from onnx import TensorProto, helper, numpy_helper

    graph = model.graph
    constants = _constant_names(graph)
    rewritten = []
    detached = []
    bias_name = "attention_bias"
    for node in graph.node:
        if node.op_type != "GroupQueryAttention":
            continue
        while len(node.input) <= 10:
            node.input.append("")
        seqlens_name = f"{node.name}/static_seqlens_k"
        total_name = f"{node.name}/static_total_seq_len"
        if node.input[5] not in constants:
            graph.initializer.append(
                numpy_helper.from_array(
                    np.full((int(batch_size),), int(max_seq_len), dtype=np.int32),
                    seqlens_name,
                )
            )
            if node.input[5]:
                detached.append(node.input[5])
            node.input[5] = seqlens_name
            constants.add(seqlens_name)
        if node.input[6] not in constants:
            graph.initializer.append(
                numpy_helper.from_array(
                    np.asarray(int(max_seq_len), dtype=np.int32),
                    total_name,
                )
            )
            if node.input[6]:
                detached.append(node.input[6])
            node.input[6] = total_name
            constants.add(total_name)
        if not node.input[10]:
            node.input[10] = bias_name
        rewritten.append(node.name or node.output[0])
    needs_bias = any(
        node.op_type == "GroupQueryAttention" and len(node.input) > 10 and node.input[10] == bias_name
        for node in graph.node
    )
    already = any(value.name == bias_name for value in graph.input) or bias_name in constants
    if needs_bias and not already:
        graph.input.append(
            helper.make_tensor_value_info(
                bias_name,
                TensorProto.FLOAT,
                [1, 1, int(query_len), int(max_seq_len)],
            )
        )
    _drop_unused_inputs(graph, detached)
    return rewritten


def apply_static_contract(
    model,
    *,
    batch_size: int,
    max_seq_len: int,
    query_len: int,
    seq_axis: int = 2,
    head_dim: Optional[int] = None,
) -> Dict[str, object]:
    """Freeze IO shapes, widen a KV prefix slice, and pin a Qwen GQA window."""
    if head_dim is None and _graph_needs_head(model):
        head_dim = infer_head_dim(model)
    frozen = freeze_graph_io(
        model,
        batch_size=batch_size,
        query_len=query_len,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
    )
    rewritten = rewrite_runtime_kv_slices(model, max_seq_len, seq_axis=seq_axis)
    attention = rewrite_gqa_to_static_window(
        model,
        batch_size=batch_size,
        query_len=query_len,
        max_seq_len=max_seq_len,
    )
    leftover = runtime_slices(model)
    if leftover:
        raise StaticShapeError(f"runtime slices remain: {leftover}")
    symbolic = []
    for value in list(model.graph.input) + list(model.graph.output):
        if not value.type.HasField("tensor_type"):
            continue
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_param:
                symbolic.append(f"{value.name}:{dim.dim_param}")
    if symbolic:
        raise StaticShapeError(f"symbolic axes remain: {symbolic}")
    return {
        "frozen_dims": frozen,
        "rewritten_slices": rewritten,
        "rewritten_attention": attention,
    }


def additive_length_bias(
    valid_len: int,
    max_seq_len: int,
    *,
    batch: int = 1,
    dtype=np.float16,
):
    """Rank-2 bias. Valid positions are 0. The unused tail is -10000."""
    if batch < 1 or max_seq_len < 1 or not 0 <= int(valid_len) <= int(max_seq_len):
        raise StaticShapeError(
            f"valid length {valid_len} does not fit batch {batch} and max_seq {max_seq_len}"
        )
    bias = np.zeros((int(batch), int(max_seq_len)), dtype=dtype)
    if int(valid_len) < int(max_seq_len):
        bias[:, int(valid_len):] = np.asarray(ADDITIVE_BIAS, dtype=dtype)
    return bias


def causal_additive_bias(query_len: int, key_len: int, *, value: float = ADDITIVE_BIAS, dtype=np.float32):
    """
    Left-aligned causal table of shape ``(1, 1, query, key)``.

    A query length of 1 is all zeros. Decode uses the length bias alone so
    the single token can still see the valid prefix.
    """
    if query_len < 1 or key_len < 1:
        raise StaticShapeError(f"query {query_len} and key {key_len} must be positive")
    bias = np.zeros((1, 1, int(query_len), int(key_len)), dtype=dtype)
    if query_len <= 1:
        return bias
    cols = np.arange(key_len)
    blocked = np.asarray(value, dtype=dtype)
    for row in range(int(query_len)):
        bias[0, 0, row, cols > row] = blocked
    return bias


def static_attention_bias(
    valid_len: int,
    query_len: int,
    key_len: int,
    *,
    causal: Optional[bool] = None,
    value: float = ADDITIVE_BIAS,
    dtype=np.float32,
):
    """Padding bias broadcast over queries, plus a constant causal table when requested."""
    if causal is None:
        causal = query_len > 1
    padding = additive_length_bias(valid_len, key_len, batch=1, dtype=dtype)[0]
    out = np.broadcast_to(padding, (1, 1, int(query_len), int(key_len))).copy()
    if causal:
        out = out + causal_additive_bias(query_len, key_len, value=value, dtype=dtype)
    return out
