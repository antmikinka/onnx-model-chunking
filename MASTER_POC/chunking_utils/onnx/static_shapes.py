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
q_norm or k_norm weight declares the head size. Fused attention appends
the query onto the past, so ``total_sequence_length`` is
``max_seq_len + query_len`` and a constant Slice on axis 2 returns the
graph boundary to ``max_seq_len``. ``seqlens`` stays the past window.
``attention_bias`` has length ``total_sequence_length``. When ``do_rotary``
is 1, the host supplies ``position_ids`` int64 ``[batch, query_len]`` and
the fused op indexes the existing rotary caches with it. An empty
position input would use the frozen ``seqlens`` row.
``SkipSimplifiedLayerNormalization`` and ``SimplifiedLayerNormalization``
become Add (for the skip) plus a full-width RMSNorm chain. The square is
cast to float32. ``ReduceSum`` divided by the weight length replaces
``ReduceMean``. That length is 4096 on the hidden norms and 128 on
``q_norm`` and ``k_norm``. The reciprocal is cast back to the activation
dtype. A ``ReduceMean`` chain with the same casts was re-fused into
SinglePassRMSNorm and reported CPU 53. The ``ReduceSum`` form kept the
CPU gaps at present key 6.104e-5, value 7.451e-8, residual 1.192e-7, and
down projection 7.749e-7. Its Vitis compile did not print the
SinglePassRMSNorm L1 line. aiecompiler then failed in
superkernel_GemmBf16Int4 because the generated weight buffer was bfloat16
where the kernel expects int8. The activation argument of that kernel is
already bfloat16. Algorithm-B node count was 57. No operator stat, no NPU
line, and no ELF line.
A graph that contains ``MatMulNBits`` keeps that ``ReduceSum`` path and
casts float graph inputs to bfloat16. Float graph outputs stay bfloat16.
The variance squares a float32 cast of the bfloat16 activation. Norm
scales are stored as bfloat16 initializers. Packed ``MatMulNBits``
weights stay on the external file. Their scales share the activation
type, so those scales are cast to bfloat16. ``GroupQueryAttention`` is
left in place; its float inputs are the bfloat16 casts. The CPU EP in
onnxruntime-vitisai 1.27 has no bfloat16 Mul kernel, so the
rewrite-versus-fused gaps were not measured. The Vitis compile of that
graph, file base ``3e42dccfc6c889d7ff6e67de7a73c9b2``, Algorithm-A
``1f7be0485299550546890a2e1bae8f60``, Algorithm-B
``a1873bce31449893f313510140258dc8``, node count 68, did not print
SinglePassRMSNorm or superkernel_GemmBf16Int4. It reported a StarNode
legalization error and ``WTS size greater than max wts memtile size``
on ``down_proj``. Operator stat CPU 68. No NPU line and no ELF line.
``GroupQueryAttention`` stays one node. A ``MatMulNBits`` whose ``K``
exceeds 4096, whose ``block_size`` divides that chunk, and whose packed
uint8 weight and float scales can be read, is split along ``K``. Partial
products are added. The live ``down_proj`` node is ``K`` 12288, ``N``
4096, ``bits`` 4, ``block_size`` 32, weight ``[4096, 384, 16]``, so the
chunk is 4096. The other four ``MatMulNBits`` stay one node. The external
data file is not rewritten. A float32 ORT session of that split matched
the unsplit ``down_proj`` with max absolute gap 2.122e-05 against a
reference max absolute value of 11.21. The Vitis compile, file base
``e671f01e271af234c16d124d6a33387e``, Algorithm-A
``25eb659ef84bd24baf43e5f815d6ee17``, Algorithm-B
``a1873bce31449893f313510140258dc8``, node count 77, did not print the
memtile line. StarNode was still the first legalization error. aiecompiler
then failed in ``superkernel_GemmBf16Int4`` because the weight buffer was
bfloat16 where the kernel expects int8. No operator stat, no NPU line, and
no ELF line. A
half-width split of the 4096 axis changed the present key by 3.039e-1
and the following Vitis compile failed in superkernel_GemmBf16Int4. The
unsplit constant-epsilon chain reported CPU 45. An epsilon product
reported CPU 49. The earlier window, with total frozen at ``max_seq_len``,
reported CPU 18.

``npu_bf16_fallback`` defaults off. When it is on, each 4-bit
``MatMulNBits`` is replaced by a Python-baked bfloat16 initializer and a
``MatMul``. The float32 norm body stays float32. An omitted zero point is
8, which is the value the CPU EP subtracts. ``DequantizeLinear`` is not
inserted. Ryzen AI 1.8.0 aborts in ``superkernel_GemmBf16Int4`` on the
packed op (https://github.com/amd/RyzenAI-SW/issues/407). A one-layer
probe of this node pattern, cache key ``pr9_bf16dense_fresh``, produced
two Vitis kernels and left ``GroupQueryAttention`` and two present slices
on CPU. That probe's weights omitted the zero point and its feeds were
zeros. Turn the flag off to keep ``MatMulNBits``. ``split_onnx`` leaves
the flag off, including when the target device is the NPU. When a caller
sets the flag, the bake runs on each saved component. The loaded master
graph is not baked. ``defer_matmul_bake`` keeps the packed ops and the
float32 norm body so that component bake has a graph to rewrite.
"""

import os
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


def _int_attr(node, name: str, default: int = 0) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return int(default)


def _float_attr(node, name: str, default: float) -> float:
    for attr in node.attribute:
        if attr.name == name:
            return float(attr.f)
    return float(default)


def _elem_type(graph, name: str, default: int) -> int:
    from onnx import TensorProto

    for value in list(graph.input) + list(graph.value_info) + list(graph.output):
        if value.name == name and value.type.HasField("tensor_type"):
            kind = value.type.tensor_type.elem_type
            if kind:
                return int(kind)
    for item in graph.initializer:
        if item.name == name and item.data_type:
            return int(item.data_type)
    return int(default if default else TensorProto.FLOAT)


def _put_initializer(graph, name: str, array) -> None:
    from onnx import numpy_helper

    kept = [item for item in graph.initializer if item.name != name]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    graph.initializer.append(numpy_helper.from_array(np.asarray(array), name))


def _value_by_name(graph, name: str):
    for value in list(graph.output) + list(graph.input) + list(graph.value_info):
        if value.name == name:
            return value
    return None


def _set_value_info(graph, name: str, elem_type: int, dims: Sequence[int]) -> None:
    from onnx import helper

    info = helper.make_tensor_value_info(name, elem_type, [int(dim) for dim in dims])
    for value in graph.value_info:
        if value.name == name:
            value.CopyFrom(info)
            return
    graph.value_info.append(info)


def _i64_initializer(graph, name: str, values: Sequence[int]) -> str:
    _put_initializer(graph, name, np.asarray(list(values), dtype=np.int64))
    return name


def rewrite_gqa_to_static_window(
    model,
    *,
    batch_size: int,
    query_len: int,
    max_seq_len: int,
) -> List[str]:
    """
    Point GroupQueryAttention at the static window.

    ``seqlens_k`` is the past window. ``total_sequence_length`` is that
    window plus ``query_len``, which is the present length the fused op
    writes. A constant Slice on axis 2 then keeps the newest
    ``max_seq_len`` tokens at the graph boundary. ``attention_bias`` has
    length ``total_sequence_length``. When ``do_rotary`` is 1, input 9
    is an int64 ``position_ids`` of shape ``[batch, query_len]``. The
    fused op indexes the cos and sin caches with that id. ``seqlens_k``
    stays the window length, so an empty position input rotates every
    token at ``max_seq_len``. ``do_rotary`` 0 does not add the input.
    Decode with ``query_len`` 1 feeds ``[[absolute_position]]``, and
    that index is not capped at ``max_seq_len``. Prefill feeds the
    whole query axis, ``[[0, 1, ..., query_len - 1]]``, not a scalar.
    """
    from onnx import TensorProto, helper

    if query_len < 1:
        raise StaticShapeError(f"query_len must be positive, got {query_len}")
    graph = model.graph
    constants = _constant_names(graph)
    rewritten = []
    detached = []
    bias_name = "attention_bias"
    bias_len = None
    rebuilt = []
    for node in graph.node:
        rebuilt.append(node)
        if node.op_type != "GroupQueryAttention":
            continue
        while len(node.input) <= 10:
            node.input.append("")
        shared = _int_attr(node, "past_present_share_buffer", 0) == 1
        total_len = int(max_seq_len) if shared else int(max_seq_len) + int(query_len)
        if bias_len is None:
            bias_len = total_len
        elif bias_len != total_len:
            raise StaticShapeError("GroupQueryAttention nodes disagree on the static window")
        label = node.name or node.output[0] or "GroupQueryAttention"
        seqlens_name = f"{label}/static_seqlens_k"
        total_name = f"{label}/static_total_seq_len"
        if node.input[5] not in constants:
            _put_initializer(
                graph,
                seqlens_name,
                np.full((int(batch_size),), int(max_seq_len), dtype=np.int32),
            )
            if node.input[5]:
                detached.append(node.input[5])
            node.input[5] = seqlens_name
            constants.add(seqlens_name)
        if node.input[6] not in constants:
            _put_initializer(graph, total_name, np.asarray(int(total_len), dtype=np.int32))
            if node.input[6]:
                detached.append(node.input[6])
            node.input[6] = total_name
            constants.add(total_name)
        if not node.input[10]:
            node.input[10] = bias_name
        if _int_attr(node, "do_rotary", 0) == 1:
            if not node.input[9]:
                node.input[9] = "position_ids"
            _ensure_position_ids(graph, node.input[9], int(batch_size), int(query_len))
        if not shared and len(node.output) >= 3 and node.output[1] and node.output[2]:
            rebuilt.extend(
                _trim_present_outputs(
                    graph,
                    node,
                    label=label,
                    query_len=int(query_len),
                    total_len=int(total_len),
                    max_seq_len=int(max_seq_len),
                )
            )
        rewritten.append(label)
    del graph.node[:]
    graph.node.extend(rebuilt)
    if bias_len is not None:
        _ensure_attention_bias(graph, bias_name, int(query_len), int(bias_len), constants)
    _drop_unused_inputs(graph, detached)
    return rewritten


def _ensure_position_ids(graph, name: str, batch_size: int, query_len: int) -> None:
    """Make ``name`` an int64 ``[batch, query]`` input unless a node already produces it."""
    from onnx import TensorProto, helper

    shape = [int(batch_size), int(query_len)]
    for value in graph.input:
        if value.name != name:
            continue
        if not value.type.HasField("tensor_type"):
            raise StaticShapeError(f"{name} is not a tensor")
        kind = int(value.type.tensor_type.elem_type)
        if kind not in (0, int(TensorProto.INT64)):
            raise StaticShapeError(
                f"{name} must be int64 so GroupQueryAttention can index RoPE, got {kind}"
            )
        value.CopyFrom(helper.make_tensor_value_info(name, TensorProto.INT64, shape))
        return
    if any(item.name == name for item in graph.initializer):
        return
    if any(name in node.output for node in graph.node):
        return
    graph.input.append(helper.make_tensor_value_info(name, TensorProto.INT64, shape))


def _ensure_attention_bias(graph, bias_name: str, query_len: int, bias_len: int, constants: set) -> None:
    from onnx import TensorProto, helper

    shape = [1, 1, int(query_len), int(bias_len)]
    for value in graph.input:
        if value.name != bias_name:
            continue
        dims = value.type.tensor_type.shape.dim
        if len(dims) == 4:
            dims[2].dim_value = int(query_len)
            dims[3].dim_value = int(bias_len)
            return
        value.CopyFrom(helper.make_tensor_value_info(bias_name, TensorProto.FLOAT, shape))
        return
    if bias_name in constants:
        return
    if not any(
        bias_name in node.input
        for node in graph.node
    ):
        return
    graph.input.append(helper.make_tensor_value_info(bias_name, TensorProto.FLOAT, shape))


def _tensor_dims(graph, name: str) -> Optional[List[int]]:
    value = _value_by_name(graph, name)
    if value is not None and value.type.HasField("tensor_type"):
        dims = list(value.type.tensor_type.shape.dim)
        if dims and all(dim.HasField("dim_value") and int(dim.dim_value) > 0 for dim in dims):
            return [int(dim.dim_value) for dim in dims]
    for item in graph.initializer:
        if item.name == name and item.dims:
            return [int(dim) for dim in item.dims]
    return None


def _rotary_half_dim(graph, cos_name: str, head_dim: int) -> int:
    dims = _tensor_dims(graph, cos_name)
    if dims and len(dims) >= 2 and int(dims[-1]) > 0:
        return int(dims[-1])
    return int(head_dim) // 2


def _rope_rotate_half_nodes(
    graph,
    *,
    label: str,
    prefix: str,
    tensor_in: str,
    cos_bcast: str,
    sin_bcast: str,
    batch_size: int,
    seq_len: int,
    heads: int,
    head_dim: int,
    rotary_half: int,
    elem_type: int,
):
    """Rotate the leading ``2 * rotary_half`` channels using half-split RoPE without Transpose."""
    from onnx import helper

    s0 = _i64_initializer(graph, f"{label}/{prefix}_rope_s0", [0])
    s1 = _i64_initializer(graph, f"{label}/{prefix}_rope_s1", [int(rotary_half)])
    s2 = _i64_initializer(graph, f"{label}/{prefix}_rope_s2", [int(rotary_half) * 2])
    ax = _i64_initializer(graph, f"{label}/{prefix}_rope_axis", [3])
    st = _i64_initializer(graph, f"{label}/{prefix}_rope_step", [1])

    half_shape = [int(batch_size), int(seq_len), int(heads), int(rotary_half)]
    x1 = f"{label}/{prefix}_x1"
    x2 = f"{label}/{prefix}_x2"
    x1_cos = f"{label}/{prefix}_x1_cos"
    x2_sin = f"{label}/{prefix}_x2_sin"
    x2_cos = f"{label}/{prefix}_x2_cos"
    x1_sin = f"{label}/{prefix}_x1_sin"
    rot1 = f"{label}/{prefix}_rot1"
    rot2 = f"{label}/{prefix}_rot2"
    rot_out = f"{label}/{prefix}_rotated"

    for name in (x1, x2, x1_cos, x2_sin, x2_cos, x1_sin, rot1, rot2):
        _set_value_info(graph, name, elem_type, half_shape)

    nodes = [
        helper.make_node("Slice", [tensor_in, s0, s1, ax, st], [x1], name=f"{label}/{prefix}_Slice_x1"),
        helper.make_node("Slice", [tensor_in, s1, s2, ax, st], [x2], name=f"{label}/{prefix}_Slice_x2"),
        helper.make_node("Mul", [x1, cos_bcast], [x1_cos], name=f"{label}/{prefix}_Mul_x1_cos"),
        helper.make_node("Mul", [x2, sin_bcast], [x2_sin], name=f"{label}/{prefix}_Mul_x2_sin"),
        helper.make_node("Sub", [x1_cos, x2_sin], [rot1], name=f"{label}/{prefix}_Sub_rot1"),
        helper.make_node("Mul", [x2, cos_bcast], [x2_cos], name=f"{label}/{prefix}_Mul_x2_cos"),
        helper.make_node("Mul", [x1, sin_bcast], [x1_sin], name=f"{label}/{prefix}_Mul_x1_sin"),
        helper.make_node("Add", [x2_cos, x1_sin], [rot2], name=f"{label}/{prefix}_Add_rot2"),
    ]
    concat_inputs = [rot1, rot2]
    if int(rotary_half) * 2 < int(head_dim):
        s3 = _i64_initializer(graph, f"{label}/{prefix}_rope_s3", [int(head_dim)])
        tail = f"{label}/{prefix}_tail"
        _set_value_info(
            graph,
            tail,
            elem_type,
            [int(batch_size), int(seq_len), int(heads), int(head_dim) - int(rotary_half) * 2],
        )
        nodes.append(
            helper.make_node("Slice", [tensor_in, s2, s3, ax, st], [tail], name=f"{label}/{prefix}_Slice_tail")
        )
        concat_inputs.append(tail)
    _set_value_info(
        graph,
        rot_out,
        elem_type,
        [int(batch_size), int(seq_len), int(heads), int(head_dim)],
    )
    nodes.append(
        helper.make_node("Concat", concat_inputs, [rot_out], axis=3, name=f"{label}/{prefix}_Concat_rope")
    )
    return nodes, rot_out


def _head_major_from_seq_major(
    graph,
    *,
    label: str,
    prefix: str,
    source: str,
    batch_size: int,
    query_len: int,
    heads: int,
    head_dim: int,
    elem_type: int,
):
    """Convert ``[batch, query_len, heads, head_dim]`` to ``[batch, heads, query_len, head_dim]`` without Transpose."""
    from onnx import helper

    target_shape = [int(batch_size), int(heads), int(query_len), int(head_dim)]
    out_name = f"{label}/{prefix}_head_major"
    _set_value_info(graph, out_name, elem_type, target_shape)
    if int(query_len) == 1 or int(heads) == 1:
        shape_name = _i64_initializer(graph, f"{label}/{prefix}_hm_shape", target_shape)
        return [
            helper.make_node("Reshape", [source, shape_name], [out_name], name=f"{label}/{prefix}_Reshape_hm")
        ], out_name

    ax = _i64_initializer(graph, f"{label}/{prefix}_hm_axis", [2])
    st = _i64_initializer(graph, f"{label}/{prefix}_hm_step", [1])
    one_shape_name = _i64_initializer(
        graph,
        f"{label}/{prefix}_hm_one_shape",
        [int(batch_size), 1, int(query_len), int(head_dim)],
    )
    nodes = []
    parts = []
    for head_idx in range(int(heads)):
        s_lo = _i64_initializer(graph, f"{label}/{prefix}_hm_s_{head_idx}", [head_idx])
        s_hi = _i64_initializer(graph, f"{label}/{prefix}_hm_e_{head_idx}", [head_idx + 1])
        sliced = f"{label}/{prefix}_hm_slice_{head_idx}"
        reshaped = f"{label}/{prefix}_hm_part_{head_idx}"
        _set_value_info(graph, sliced, elem_type, [int(batch_size), int(query_len), 1, int(head_dim)])
        _set_value_info(graph, reshaped, elem_type, [int(batch_size), 1, int(query_len), int(head_dim)])
        nodes.append(
            helper.make_node(
                "Slice",
                [source, s_lo, s_hi, ax, st],
                [sliced],
                name=f"{label}/{prefix}_Slice_head_{head_idx}",
            )
        )
        nodes.append(
            helper.make_node(
                "Reshape",
                [sliced, one_shape_name],
                [reshaped],
                name=f"{label}/{prefix}_Reshape_head_{head_idx}",
            )
        )
        parts.append(reshaped)
    nodes.append(
        helper.make_node("Concat", parts, [out_name], axis=1, name=f"{label}/{prefix}_Concat_heads")
    )
    return nodes, out_name


def _seq_major_from_head_major(
    graph,
    *,
    label: str,
    prefix: str,
    source: str,
    out_name: str,
    batch_size: int,
    query_len: int,
    heads: int,
    head_dim: int,
    elem_type: int,
):
    """Convert ``[batch, heads, query_len, head_dim]`` to ``[batch, query_len, heads * head_dim]`` without Transpose."""
    from onnx import helper

    flat_shape = [int(batch_size), int(query_len), int(heads) * int(head_dim)]
    if int(query_len) == 1 or int(heads) == 1:
        shape_name = _i64_initializer(graph, f"{label}/{prefix}_flat_shape", flat_shape)
        return [
            helper.make_node("Reshape", [source, shape_name], [out_name], name=f"{label}/{prefix}_Reshape_out")
        ]

    ax = _i64_initializer(graph, f"{label}/{prefix}_q_axis", [2])
    st = _i64_initializer(graph, f"{label}/{prefix}_q_step", [1])
    token_shape = _i64_initializer(
        graph,
        f"{label}/{prefix}_token_shape",
        [int(batch_size), 1, int(heads) * int(head_dim)],
    )
    nodes = []
    parts = []
    for q_idx in range(int(query_len)):
        s_lo = _i64_initializer(graph, f"{label}/{prefix}_q_s_{q_idx}", [q_idx])
        s_hi = _i64_initializer(graph, f"{label}/{prefix}_q_e_{q_idx}", [q_idx + 1])
        sliced = f"{label}/{prefix}_q_slice_{q_idx}"
        reshaped = f"{label}/{prefix}_q_part_{q_idx}"
        _set_value_info(graph, sliced, elem_type, [int(batch_size), int(heads), 1, int(head_dim)])
        _set_value_info(graph, reshaped, elem_type, [int(batch_size), 1, int(heads) * int(head_dim)])
        nodes.append(
            helper.make_node(
                "Slice",
                [source, s_lo, s_hi, ax, st],
                [sliced],
                name=f"{label}/{prefix}_Slice_q_{q_idx}",
            )
        )
        nodes.append(
            helper.make_node(
                "Reshape",
                [sliced, token_shape],
                [reshaped],
                name=f"{label}/{prefix}_Reshape_q_{q_idx}",
            )
        )
        parts.append(reshaped)
    nodes.append(
        helper.make_node("Concat", parts, [out_name], axis=1, name=f"{label}/{prefix}_Concat_out")
    )
    return nodes


def decompose_gqa_to_npu_primitives(
    model,
    *,
    batch_size: int,
    query_len: int,
    max_seq_len: int,
    sink_size: int = 0,
) -> List[str]:
    """
    Replace ``com.microsoft::GroupQueryAttention`` with standard ONNX primitives.

    The replacement subgraph uses only ``Reshape``, ``Gather``, ``Slice``,
    ``Concat``, ``Mul``, ``Sub``, ``Add``, ``MatMul``, and ``Softmax`` (no
    ``Transpose``, ``Where``, ``Scatter``, or contrib ops). Vitis AI
    compilation and NPU placement of this decomposed Qwen layer have not
    been measured.
    """
    from onnx import TensorProto, helper

    if int(query_len) < 1:
        raise StaticShapeError(f"query_len must be positive, got {query_len}")
    if int(sink_size) < 0 or int(sink_size) + int(query_len) > int(max_seq_len):
        raise StaticShapeError(
            f"sink_size {sink_size} + query_len {query_len} exceeds max_seq_len {max_seq_len}"
        )

    graph = model.graph
    constants = _constant_names(graph)
    rewritten: List[str] = []
    detached: List[str] = []
    bias_name = "attention_bias"
    bias_len: Optional[int] = None
    rebuilt = []

    for node in list(graph.node):
        if node.op_type != "GroupQueryAttention":
            rebuilt.append(node)
            continue

        label = node.name or _output_at(node, 0) or "GroupQueryAttention"
        num_heads = _int_attr(node, "num_heads", 0)
        kv_heads = _int_attr(node, "kv_num_heads", num_heads)
        if num_heads < 1 or kv_heads < 1 or num_heads % kv_heads != 0:
            raise StaticShapeError(
                f"{label} invalid head counts num_heads={num_heads}, kv_num_heads={kv_heads}"
            )
        group_size = num_heads // kv_heads
        q_in = _input_at(node, 0)
        k_in = _input_at(node, 1)
        v_in = _input_at(node, 2)
        past_k_in = _input_at(node, 3)
        past_v_in = _input_at(node, 4)
        seqlens_in = _input_at(node, 5)
        total_in = _input_at(node, 6)
        cos_in = _input_at(node, 7)
        sin_in = _input_at(node, 8)
        pos_in = _input_at(node, 9)
        node_bias_in = _input_at(node, 10) or bias_name
        out_attn = _output_at(node, 0)
        out_present_k = _output_at(node, 1)
        out_present_v = _output_at(node, 2)

        if seqlens_in and seqlens_in not in constants:
            detached.append(seqlens_in)
        if total_in and total_in not in constants:
            detached.append(total_in)

        past_dims = _tensor_dims(graph, past_k_in)
        q_dims = _tensor_dims(graph, q_in)
        if past_dims and len(past_dims) == 4:
            head_dim = int(past_dims[3])
        elif q_dims and len(q_dims) == 3 and int(q_dims[2]) % num_heads == 0:
            head_dim = int(q_dims[2]) // num_heads
        else:
            head_dim = infer_head_dim(model)

        elem_type = _elem_type(graph, q_in, TensorProto.FLOAT)
        scale_val = _float_attr(node, "scale", 0.0)
        if scale_val <= 0.0:
            scale_val = float(head_dim) ** -0.5

        shared = _int_attr(node, "past_present_share_buffer", 0) == 1
        total_len = int(max_seq_len) if shared else int(max_seq_len) + int(query_len)
        if bias_len is None:
            bias_len = total_len
        elif bias_len != total_len:
            raise StaticShapeError("GroupQueryAttention nodes disagree on the static window")

        q_4d_shape = [int(batch_size), int(query_len), int(num_heads), int(head_dim)]
        kv_4d_shape = [int(batch_size), int(query_len), int(kv_heads), int(head_dim)]
        q_4d_shape_name = _i64_initializer(graph, f"{label}/q_4d_shape", q_4d_shape)
        kv_4d_shape_name = _i64_initializer(graph, f"{label}/kv_4d_shape", kv_4d_shape)

        q_4d = f"{label}/q_4d"
        k_4d = f"{label}/k_4d"
        v_4d = f"{label}/v_4d"
        _set_value_info(graph, q_4d, elem_type, q_4d_shape)
        _set_value_info(graph, k_4d, elem_type, kv_4d_shape)
        _set_value_info(graph, v_4d, elem_type, kv_4d_shape)

        rebuilt.append(helper.make_node("Reshape", [q_in, q_4d_shape_name], [q_4d], name=f"{label}/Reshape_q"))
        rebuilt.append(helper.make_node("Reshape", [k_in, kv_4d_shape_name], [k_4d], name=f"{label}/Reshape_k"))
        rebuilt.append(helper.make_node("Reshape", [v_in, kv_4d_shape_name], [v_4d], name=f"{label}/Reshape_v"))

        do_rotary = _int_attr(node, "do_rotary", 0) == 1
        if do_rotary:
            if _int_attr(node, "rotary_interleaved", 0) != 0:
                raise StaticShapeError(f"{label} rotary_interleaved=1 is not supported by NPU decomposition")
            if not cos_in or not sin_in:
                raise StaticShapeError(f"{label} has do_rotary=1 without cos_cache/sin_cache")
            pos_name = pos_in or "position_ids"
            _ensure_position_ids(graph, pos_name, int(batch_size), int(query_len))
            rotary_half = _rotary_half_dim(graph, cos_in, head_dim)
            cos_gat = f"{label}/cos_gathered"
            sin_gat = f"{label}/sin_gathered"
            cos_bcast = f"{label}/cos_bcast"
            sin_bcast = f"{label}/sin_bcast"
            bcast_shape = [int(batch_size), int(query_len), 1, int(rotary_half)]
            bcast_shape_name = _i64_initializer(graph, f"{label}/rope_bcast_shape", bcast_shape)
            _set_value_info(graph, cos_gat, elem_type, [int(batch_size), int(query_len), int(rotary_half)])
            _set_value_info(graph, sin_gat, elem_type, [int(batch_size), int(query_len), int(rotary_half)])
            _set_value_info(graph, cos_bcast, elem_type, bcast_shape)
            _set_value_info(graph, sin_bcast, elem_type, bcast_shape)
            rebuilt.append(
                helper.make_node("Gather", [cos_in, pos_name], [cos_gat], axis=0, name=f"{label}/Gather_cos")
            )
            rebuilt.append(
                helper.make_node("Gather", [sin_in, pos_name], [sin_gat], axis=0, name=f"{label}/Gather_sin")
            )
            rebuilt.append(
                helper.make_node("Reshape", [cos_gat, bcast_shape_name], [cos_bcast], name=f"{label}/Reshape_cos")
            )
            rebuilt.append(
                helper.make_node("Reshape", [sin_gat, bcast_shape_name], [sin_bcast], name=f"{label}/Reshape_sin")
            )
            q_rope_nodes, q_4d = _rope_rotate_half_nodes(
                graph,
                label=label,
                prefix="q",
                tensor_in=q_4d,
                cos_bcast=cos_bcast,
                sin_bcast=sin_bcast,
                batch_size=int(batch_size),
                seq_len=int(query_len),
                heads=int(num_heads),
                head_dim=int(head_dim),
                rotary_half=int(rotary_half),
                elem_type=elem_type,
            )
            k_rope_nodes, k_4d = _rope_rotate_half_nodes(
                graph,
                label=label,
                prefix="k",
                tensor_in=k_4d,
                cos_bcast=cos_bcast,
                sin_bcast=sin_bcast,
                batch_size=int(batch_size),
                seq_len=int(query_len),
                heads=int(kv_heads),
                head_dim=int(head_dim),
                rotary_half=int(rotary_half),
                elem_type=elem_type,
            )
            rebuilt.extend(q_rope_nodes)
            rebuilt.extend(k_rope_nodes)

        q_hm_nodes, q_hm = _head_major_from_seq_major(
            graph,
            label=label,
            prefix="q",
            source=q_4d,
            batch_size=int(batch_size),
            query_len=int(query_len),
            heads=int(num_heads),
            head_dim=int(head_dim),
            elem_type=elem_type,
        )
        k_hm_nodes, new_k = _head_major_from_seq_major(
            graph,
            label=label,
            prefix="k",
            source=k_4d,
            batch_size=int(batch_size),
            query_len=int(query_len),
            heads=int(kv_heads),
            head_dim=int(head_dim),
            elem_type=elem_type,
        )
        v_hm_nodes, new_v = _head_major_from_seq_major(
            graph,
            label=label,
            prefix="v",
            source=v_4d,
            batch_size=int(batch_size),
            query_len=int(query_len),
            heads=int(kv_heads),
            head_dim=int(head_dim),
            elem_type=elem_type,
        )
        rebuilt.extend(q_hm_nodes)
        rebuilt.extend(k_hm_nodes)
        rebuilt.extend(v_hm_nodes)

        # Build full_k / full_v (for attention scoring) and present_k / present_v (for cache output).
        total_rows = int(group_size) * int(query_len)
        needs_kv_slice = bool(shared or int(sink_size) > 0 or total_rows > 1)
        ax2 = _i64_initializer(graph, f"{label}/kv_axis", [2]) if needs_kv_slice else ""
        st1 = _i64_initializer(graph, f"{label}/kv_step", [1]) if needs_kv_slice else ""
        if shared:
            # In-place window of length max_seq_len.
            if int(sink_size) > 0:
                s0 = _i64_initializer(graph, f"{label}/sink_start", [0])
                s_sink = _i64_initializer(graph, f"{label}/sink_end", [int(sink_size)])
                s_roll = _i64_initializer(
                    graph, f"{label}/roll_start", [int(sink_size) + int(query_len)]
                )
                s_max = _i64_initializer(graph, f"{label}/roll_end", [int(max_seq_len)])
                sink_k = f"{label}/sink_k"
                sink_v = f"{label}/sink_v"
                kept_k = f"{label}/kept_k"
                kept_v = f"{label}/kept_v"
                _set_value_info(
                    graph, sink_k, elem_type, [int(batch_size), int(kv_heads), int(sink_size), int(head_dim)]
                )
                _set_value_info(
                    graph, sink_v, elem_type, [int(batch_size), int(kv_heads), int(sink_size), int(head_dim)]
                )
                kept_len = int(max_seq_len) - int(sink_size) - int(query_len)
                _set_value_info(
                    graph, kept_k, elem_type, [int(batch_size), int(kv_heads), kept_len, int(head_dim)]
                )
                _set_value_info(
                    graph, kept_v, elem_type, [int(batch_size), int(kv_heads), kept_len, int(head_dim)]
                )
                rebuilt.extend(
                    [
                        helper.make_node("Slice", [past_k_in, s0, s_sink, ax2, st1], [sink_k], name=f"{label}/Slice_sink_k"),
                        helper.make_node("Slice", [past_k_in, s_roll, s_max, ax2, st1], [kept_k], name=f"{label}/Slice_kept_k"),
                        helper.make_node("Concat", [sink_k, kept_k, new_k], [out_present_k], axis=2, name=f"{label}/Concat_present_k"),
                        helper.make_node("Slice", [past_v_in, s0, s_sink, ax2, st1], [sink_v], name=f"{label}/Slice_sink_v"),
                        helper.make_node("Slice", [past_v_in, s_roll, s_max, ax2, st1], [kept_v], name=f"{label}/Slice_kept_v"),
                        helper.make_node("Concat", [sink_v, kept_v, new_v], [out_present_v], axis=2, name=f"{label}/Concat_present_v"),
                    ]
                )
            else:
                s_roll = _i64_initializer(graph, f"{label}/roll_start", [int(query_len)])
                s_max = _i64_initializer(graph, f"{label}/roll_end", [int(max_seq_len)])
                kept_k = f"{label}/kept_k"
                kept_v = f"{label}/kept_v"
                kept_len = int(max_seq_len) - int(query_len)
                _set_value_info(
                    graph, kept_k, elem_type, [int(batch_size), int(kv_heads), kept_len, int(head_dim)]
                )
                _set_value_info(
                    graph, kept_v, elem_type, [int(batch_size), int(kv_heads), kept_len, int(head_dim)]
                )
                rebuilt.extend(
                    [
                        helper.make_node("Slice", [past_k_in, s_roll, s_max, ax2, st1], [kept_k], name=f"{label}/Slice_kept_k"),
                        helper.make_node("Concat", [kept_k, new_k], [out_present_k], axis=2, name=f"{label}/Concat_present_k"),
                        helper.make_node("Slice", [past_v_in, s_roll, s_max, ax2, st1], [kept_v], name=f"{label}/Slice_kept_v"),
                        helper.make_node("Concat", [kept_v, new_v], [out_present_v], axis=2, name=f"{label}/Concat_present_v"),
                    ]
                )
            full_k = out_present_k
            full_v = out_present_v
        else:
            full_k = f"{label}/present_key_untrimmed"
            full_v = f"{label}/present_value_untrimmed"
            full_shape = [int(batch_size), int(kv_heads), int(total_len), int(head_dim)]
            _set_value_info(graph, full_k, elem_type, full_shape)
            _set_value_info(graph, full_v, elem_type, full_shape)
            rebuilt.append(
                helper.make_node("Concat", [past_k_in, new_k], [full_k], axis=2, name=f"{label}/Concat_full_k")
            )
            rebuilt.append(
                helper.make_node("Concat", [past_v_in, new_v], [full_v], axis=2, name=f"{label}/Concat_full_v")
            )
            if int(sink_size) > 0:
                s0 = _i64_initializer(graph, f"{label}/present_sink_start", [0])
                s_sink = _i64_initializer(graph, f"{label}/present_sink_end", [int(sink_size)])
                s_roll = _i64_initializer(
                    graph, f"{label}/present_slice_starts", [int(sink_size) + int(query_len)]
                )
                s_end = _i64_initializer(graph, f"{label}/present_slice_ends", [int(total_len)])
                sink_k = f"{label}/sink_k"
                sink_v = f"{label}/sink_v"
                kept_k = f"{label}/kept_k"
                kept_v = f"{label}/kept_v"
                recent_len = int(max_seq_len) - int(sink_size)
                _set_value_info(
                    graph, sink_k, elem_type, [int(batch_size), int(kv_heads), int(sink_size), int(head_dim)]
                )
                _set_value_info(
                    graph, sink_v, elem_type, [int(batch_size), int(kv_heads), int(sink_size), int(head_dim)]
                )
                _set_value_info(
                    graph, kept_k, elem_type, [int(batch_size), int(kv_heads), recent_len, int(head_dim)]
                )
                _set_value_info(
                    graph, kept_v, elem_type, [int(batch_size), int(kv_heads), recent_len, int(head_dim)]
                )
                rebuilt.extend(
                    [
                        helper.make_node("Slice", [full_k, s0, s_sink, ax2, st1], [sink_k], name=f"{label}/Slice_sink_key"),
                        helper.make_node("Slice", [full_k, s_roll, s_end, ax2, st1], [kept_k], name=f"{label}/Slice_recent_key"),
                        helper.make_node("Concat", [sink_k, kept_k], [out_present_k], axis=2, name=f"{label}/Concat_present_key"),
                        helper.make_node("Slice", [full_v, s0, s_sink, ax2, st1], [sink_v], name=f"{label}/Slice_sink_value"),
                        helper.make_node("Slice", [full_v, s_roll, s_end, ax2, st1], [kept_v], name=f"{label}/Slice_recent_value"),
                        helper.make_node("Concat", [sink_v, kept_v], [out_present_v], axis=2, name=f"{label}/Concat_present_value"),
                    ]
                )
            else:
                starts = _i64_initializer(graph, f"{label}/present_slice_starts", [int(query_len)])
                ends = _i64_initializer(graph, f"{label}/present_slice_ends", [int(total_len)])
                axes = _i64_initializer(graph, f"{label}/present_slice_axes", [2])
                steps = _i64_initializer(graph, f"{label}/present_slice_steps", [1])
                rebuilt.append(
                    helper.make_node(
                        "Slice",
                        [full_k, starts, ends, axes, steps],
                        [out_present_k],
                        name=f"{label}/Slice_present_key",
                    )
                )
                rebuilt.append(
                    helper.make_node(
                        "Slice",
                        [full_v, starts, ends, axes, steps],
                        [out_present_v],
                        name=f"{label}/Slice_present_value",
                    )
                )

        # Transpose-free attention scoring using column MatMul (full_k @ q_col).
        q_grouped_shape = [
            int(batch_size),
            int(kv_heads),
            int(group_size) * int(query_len),
            int(head_dim),
        ]
        q_grouped_shape_name = _i64_initializer(graph, f"{label}/q_grouped_shape", q_grouped_shape)
        q_grouped = f"{label}/q_grouped"
        _set_value_info(graph, q_grouped, elem_type, q_grouped_shape)
        rebuilt.append(
            helper.make_node("Reshape", [q_hm, q_grouped_shape_name], [q_grouped], name=f"{label}/Reshape_q_grouped")
        )

        col_shape = [int(batch_size), int(kv_heads), int(head_dim), 1]
        col_shape_name = _i64_initializer(graph, f"{label}/q_col_shape", col_shape)
        row_score_shape = [int(batch_size), int(kv_heads), 1, int(total_len)]
        row_score_shape_name = _i64_initializer(graph, f"{label}/score_row_shape", row_score_shape)
        total_rows = int(group_size) * int(query_len)
        score_rows = []
        for row_idx in range(total_rows):
            if total_rows == 1:
                q_row = q_grouped
            else:
                r_lo = _i64_initializer(graph, f"{label}/q_row_s_{row_idx}", [row_idx])
                r_hi = _i64_initializer(graph, f"{label}/q_row_e_{row_idx}", [row_idx + 1])
                q_row = f"{label}/q_row_{row_idx}"
                _set_value_info(graph, q_row, elem_type, [int(batch_size), int(kv_heads), 1, int(head_dim)])
                rebuilt.append(
                    helper.make_node(
                        "Slice",
                        [q_grouped, r_lo, r_hi, ax2, st1],
                        [q_row],
                        name=f"{label}/Slice_q_row_{row_idx}",
                    )
                )
            q_col = f"{label}/q_col_{row_idx}"
            score_col = f"{label}/score_col_{row_idx}"
            score_row = f"{label}/score_row_{row_idx}"
            _set_value_info(graph, q_col, elem_type, col_shape)
            _set_value_info(graph, score_col, elem_type, [int(batch_size), int(kv_heads), int(total_len), 1])
            _set_value_info(graph, score_row, elem_type, row_score_shape)
            rebuilt.append(
                helper.make_node("Reshape", [q_row, col_shape_name], [q_col], name=f"{label}/Reshape_q_col_{row_idx}")
            )
            rebuilt.append(
                helper.make_node("MatMul", [full_k, q_col], [score_col], name=f"{label}/MatMul_scores_{row_idx}")
            )
            rebuilt.append(
                helper.make_node(
                    "Reshape",
                    [score_col, row_score_shape_name],
                    [score_row],
                    name=f"{label}/Reshape_scores_{row_idx}",
                )
            )
            score_rows.append(score_row)

        if len(score_rows) == 1:
            scores_grouped = score_rows[0]
        else:
            scores_grouped = f"{label}/scores_grouped"
            _set_value_info(
                graph,
                scores_grouped,
                elem_type,
                [int(batch_size), int(kv_heads), total_rows, int(total_len)],
            )
            rebuilt.append(
                helper.make_node("Concat", score_rows, [scores_grouped], axis=2, name=f"{label}/Concat_scores")
            )

        scores_4d_shape = [int(batch_size), int(num_heads), int(query_len), int(total_len)]
        scores_4d_shape_name = _i64_initializer(graph, f"{label}/scores_4d_shape", scores_4d_shape)
        scores_4d = f"{label}/scores_4d"
        _set_value_info(graph, scores_4d, elem_type, scores_4d_shape)
        rebuilt.append(
            helper.make_node("Reshape", [scores_grouped, scores_4d_shape_name], [scores_4d], name=f"{label}/Reshape_scores_4d")
        )

        scale_name = f"{label}/attn_scale"
        _put_initializer(graph, scale_name, np.asarray(scale_val, dtype=np.float32))
        scaled_scores = f"{label}/scaled_scores"
        biased_scores = f"{label}/biased_scores"
        probs_4d = f"{label}/probs_4d"
        _set_value_info(graph, scaled_scores, elem_type, scores_4d_shape)
        _set_value_info(graph, biased_scores, elem_type, scores_4d_shape)
        _set_value_info(graph, probs_4d, elem_type, scores_4d_shape)
        rebuilt.append(
            helper.make_node("Mul", [scores_4d, scale_name], [scaled_scores], name=f"{label}/Mul_scale")
        )
        rebuilt.append(
            helper.make_node("Add", [scaled_scores, node_bias_in], [biased_scores], name=f"{label}/Add_bias")
        )
        rebuilt.append(
            helper.make_node("Softmax", [biased_scores], [probs_4d], axis=-1, name=f"{label}/Softmax_probs")
        )

        probs_grouped_shape = [int(batch_size), int(kv_heads), total_rows, int(total_len)]
        probs_grouped_shape_name = _i64_initializer(graph, f"{label}/probs_grouped_shape", probs_grouped_shape)
        probs_grouped = f"{label}/probs_grouped"
        context_grouped = f"{label}/context_grouped"
        context_hm = f"{label}/context_hm"
        context_hm_shape = [int(batch_size), int(num_heads), int(query_len), int(head_dim)]
        context_hm_shape_name = _i64_initializer(graph, f"{label}/context_hm_shape", context_hm_shape)
        _set_value_info(graph, probs_grouped, elem_type, probs_grouped_shape)
        _set_value_info(
            graph,
            context_grouped,
            elem_type,
            [int(batch_size), int(kv_heads), total_rows, int(head_dim)],
        )
        _set_value_info(graph, context_hm, elem_type, context_hm_shape)

        rebuilt.append(
            helper.make_node("Reshape", [probs_4d, probs_grouped_shape_name], [probs_grouped], name=f"{label}/Reshape_probs_grouped")
        )
        rebuilt.append(
            helper.make_node("MatMul", [probs_grouped, full_v], [context_grouped], name=f"{label}/MatMul_context")
        )
        rebuilt.append(
            helper.make_node("Reshape", [context_grouped, context_hm_shape_name], [context_hm], name=f"{label}/Reshape_context_hm")
        )
        rebuilt.extend(
            _seq_major_from_head_major(
                graph,
                label=label,
                prefix="context",
                source=context_hm,
                out_name=out_attn,
                batch_size=int(batch_size),
                query_len=int(query_len),
                heads=int(num_heads),
                head_dim=int(head_dim),
                elem_type=elem_type,
            )
        )
        rewritten.append(label)

    del graph.node[:]
    graph.node.extend(rebuilt)
    if bias_len is not None:
        _ensure_attention_bias(graph, bias_name, int(query_len), int(bias_len), constants)
    _drop_unused_inputs(graph, detached)
    return rewritten


def _trim_present_outputs(graph, node, *, label: str, query_len: int, total_len: int, max_seq_len: int):
    """Slice present key and value from total_len back to max_seq_len on axis 2."""
    from onnx import helper

    starts = _i64_initializer(graph, f"{label}/present_slice_starts", [int(query_len)])
    ends = _i64_initializer(graph, f"{label}/present_slice_ends", [int(total_len)])
    axes = _i64_initializer(graph, f"{label}/present_slice_axes", [2])
    steps = _i64_initializer(graph, f"{label}/present_slice_steps", [1])
    slices = []
    for slot, suffix in ((1, "key"), (2, "value")):
        boundary_name = node.output[slot]
        boundary = _value_by_name(graph, boundary_name)
        if boundary is None or not boundary.type.HasField("tensor_type"):
            raise StaticShapeError(f"{label} present output {boundary_name} has no static shape")
        dims = [int(dim.dim_value) for dim in boundary.type.tensor_type.shape.dim]
        if len(dims) != 4 or dims[2] != int(max_seq_len):
            raise StaticShapeError(
                f"{label} present {boundary_name} shape {dims} is not a rank-4 window of {max_seq_len}"
            )
        full_dims = list(dims)
        full_dims[2] = int(total_len)
        full_name = f"{label}/present_{suffix}_untrimmed"
        elem = boundary.type.tensor_type.elem_type
        _set_value_info(graph, full_name, elem, full_dims)
        node.output[slot] = full_name
        slices.append(
            helper.make_node(
                "Slice",
                [full_name, starts, ends, axes, steps],
                [boundary_name],
                name=f"{label}/Slice_present_{suffix}",
            )
        )
    return slices


def _model_opset(model, domain: str = "") -> int:
    for item in model.opset_import:
        if item.domain == domain:
            return int(item.version)
    return 0


def _reduce_sum(opset: int, *, data: str, reduced: str, axes_name: str, axis: int, name: str):
    """ReduceSum takes axes as an input from opset 11 onward."""
    from onnx import helper

    if int(opset) >= 11:
        return helper.make_node(
            "ReduceSum",
            [data, axes_name],
            [reduced],
            name=name,
            keepdims=1,
        )
    return helper.make_node(
        "ReduceSum",
        [data],
        [reduced],
        name=name,
        axes=[int(axis)],
        keepdims=1,
    )


def _rank1_length(graph, weight: str) -> int:
    """Rank-1 length of a weight, following a Cast back to its initializer."""
    seen = set()
    name = weight
    while name and name not in seen:
        seen.add(name)
        for item in graph.initializer:
            if item.name == name and len(item.dims) == 1 and int(item.dims[0]) > 0:
                return int(item.dims[0])
        producer = ""
        for node in graph.node:
            if node.op_type == "Cast" and node.output and node.output[0] == name and node.input:
                producer = node.input[0]
                break
        name = producer
    return 0


def _reduction_count(graph, weight: str, source: str, axis: int) -> int:
    """Length of the RMSNorm axis. A rank-1 weight is that length."""
    if weight:
        length = _rank1_length(graph, weight)
        if length > 0:
            return length
    value = _value_by_name(graph, source)
    if value is not None and value.type.HasField("tensor_type"):
        dims = list(value.type.tensor_type.shape.dim)
        if dims:
            index = int(axis) if int(axis) >= 0 else len(dims) + int(axis)
            if 0 <= index < len(dims) and dims[index].HasField("dim_value"):
                count = int(dims[index].dim_value)
                if count > 0:
                    return count
    raise StaticShapeError(f"{weight or source} has no static reduction length")


def _rms_nodes(
    graph,
    *,
    label: str,
    source: str,
    weight: str,
    output: str,
    epsilon: float,
    axis: int,
    elem_type: int,
    opset: int,
    mean_name: str = "",
    inv_name: str = "",
):
    """Full-width RMSNorm. Variance is the sum of squares divided by the count.

    The source is cast to float32 before the square, so a bfloat16
    activation is squared in float32. The count is the rank-1 weight
    length, so a 128-wide q_norm does not divide by 4096. A ReduceMean
    chain was re-fused into SinglePassRMSNorm.
    """
    from onnx import TensorProto, helper

    count = _reduction_count(graph, weight, source, axis)
    one = f"{label}/rms_one"
    eps = f"{label}/rms_epsilon"
    width = f"{label}/rms_count"
    axes = f"{label}/rms_axes"
    _put_initializer(graph, one, np.asarray(1, dtype=np.float32))
    _put_initializer(graph, eps, np.asarray(epsilon, dtype=np.float32))
    _put_initializer(graph, width, np.asarray(count, dtype=np.float32))
    if int(opset) >= 11:
        _put_initializer(graph, axes, np.asarray([int(axis)], dtype=np.int64))
    x_f32 = f"{label}/rms_x_f32"
    squared = f"{label}/rms_squared"
    summed = f"{label}/rms_sum"
    mean = mean_name or f"{label}/rms_mean"
    shifted = f"{label}/rms_shifted"
    std = f"{label}/rms_std"
    inv_f32 = f"{label}/rms_inv_f32"
    inv = inv_name or f"{label}/rms_inv"
    scaled = f"{label}/rms_scaled"
    nodes = [
        helper.make_node("Cast", [source], [x_f32], name=f"{label}/Cast_x", to=TensorProto.FLOAT),
        helper.make_node("Mul", [x_f32, x_f32], [squared], name=f"{label}/Mul_square"),
        _reduce_sum(
            opset,
            data=squared,
            reduced=summed,
            axes_name=axes,
            axis=int(axis),
            name=f"{label}/ReduceSum",
        ),
        helper.make_node("Div", [summed, width], [mean], name=f"{label}/Div_count"),
        helper.make_node("Add", [mean, eps], [shifted], name=f"{label}/Add_epsilon"),
        helper.make_node("Sqrt", [shifted], [std], name=f"{label}/Sqrt"),
        helper.make_node("Div", [one, std], [inv_f32], name=f"{label}/Div"),
        helper.make_node(
            "Cast",
            [inv_f32],
            [inv],
            name=f"{label}/Cast_inv",
            to=int(elem_type),
        ),
        helper.make_node("Mul", [source, inv], [scaled], name=f"{label}/Mul_inv"),
    ]
    if weight:
        nodes.append(helper.make_node("Mul", [scaled, weight], [output], name=f"{label}/Mul_weight"))
    else:
        nodes[-1].output[0] = output
    return nodes


def _input_at(node, index: int) -> str:
    if len(node.input) <= index:
        return ""
    return node.input[index]


def _output_at(node, index: int) -> str:
    if len(node.output) <= index:
        return ""
    return node.output[index]


def rewrite_fused_norms(model, activation_type: Optional[int] = None) -> List[str]:
    """
    Replace fused RMSNorm ops with primitive ONNX nodes.

    ``SkipSimplifiedLayerNormalization`` output 0 is the normalized sum and
    output 3 is that sum before normalization. ``SimplifiedLayerNormalization``
    is the same chain without the residual add. MatMulNBits and
    GroupQueryAttention nodes are copied through unchanged. The ReduceSum
    compile of the Qwen layer did not print the SinglePassRMSNorm L1 line
    and stopped in superkernel_GemmBf16Int4. ``activation_type`` is the
    dtype of the reciprocal cast. The variance stays float32.
    """
    opset = _model_opset(model, "")
    graph = model.graph
    rewritten = []
    rebuilt = []
    for node in list(graph.node):
        if node.op_type == "SkipSimplifiedLayerNormalization":
            rebuilt.extend(_decompose_skip_norm(graph, node, opset, activation_type))
            rewritten.append(node.name or _output_at(node, 0) or "SkipSimplifiedLayerNormalization")
            continue
        if node.op_type == "SimplifiedLayerNormalization":
            rebuilt.extend(_decompose_simple_norm(graph, node, opset, activation_type))
            rewritten.append(node.name or _output_at(node, 0) or "SimplifiedLayerNormalization")
            continue
        rebuilt.append(node)
    del graph.node[:]
    graph.node.extend(rebuilt)
    return rewritten


def _residual_nodes(node, label: str):
    """Add the skip, and a bias when the fused node has one."""
    from onnx import helper

    data = _input_at(node, 0)
    skip = _input_at(node, 1)
    bias = _input_at(node, 3)
    residual_name = _output_at(node, 3) or f"{label}/residual"
    nodes = []
    source = data
    if skip and bias:
        partial = f"{label}/skip_sum"
        nodes.append(helper.make_node("Add", [data, skip], [partial], name=f"{label}/Add_skip"))
        nodes.append(helper.make_node("Add", [partial, bias], [residual_name], name=f"{label}/Add_bias"))
        source = residual_name
    elif skip:
        nodes.append(helper.make_node("Add", [data, skip], [residual_name], name=f"{label}/Add_skip"))
        source = residual_name
    elif bias:
        nodes.append(helper.make_node("Add", [data, bias], [residual_name], name=f"{label}/Add_bias"))
        source = residual_name
    return nodes, source


def _decompose_skip_norm(graph, node, opset: int, activation_type: Optional[int] = None):
    from onnx import TensorProto

    label = node.name or "SkipSimplifiedLayerNormalization"
    nodes, source = _residual_nodes(node, label)
    output = _output_at(node, 0)
    if not output:
        return nodes
    weight = _input_at(node, 2)
    epsilon = _float_attr(node, "epsilon", 1e-5)
    axis = _int_attr(node, "axis", -1)
    elem = int(activation_type) if activation_type else _elem_type(graph, _input_at(node, 0), TensorProto.FLOAT)
    nodes.extend(
        _rms_nodes(
            graph,
            label=label,
            source=source,
            weight=weight,
            output=output,
            epsilon=epsilon,
            axis=axis,
            elem_type=elem,
            opset=opset,
            mean_name=_output_at(node, 1),
            inv_name=_output_at(node, 2),
        )
    )
    return nodes


def _decompose_simple_norm(graph, node, opset: int, activation_type: Optional[int] = None):
    from onnx import TensorProto, helper

    label = node.name or "SimplifiedLayerNormalization"
    source = _input_at(node, 0)
    weight = _input_at(node, 1)
    bias = _input_at(node, 2)
    output = _output_at(node, 0)
    if not output:
        return []
    epsilon = _float_attr(node, "epsilon", 1e-5)
    axis = _int_attr(node, "axis", -1)
    elem = int(activation_type) if activation_type else _elem_type(graph, source, TensorProto.FLOAT)
    normalized = f"{label}/normalized" if bias else output
    nodes = _rms_nodes(
        graph,
        label=label,
        source=source,
        weight=weight,
        output=normalized,
        epsilon=epsilon,
        axis=axis,
        elem_type=elem,
        opset=opset,
    )
    if bias:
        nodes.append(helper.make_node("Add", [normalized, bias], [output], name=f"{label}/Add_bias"))
    return nodes


def _f32_to_bf16_raw(array: np.ndarray) -> bytes:
    """Round float32 to bfloat16 bytes, nearest even."""
    values = np.ascontiguousarray(array, dtype=np.float32)
    bits = values.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = (bits + bias) >> np.uint32(16)
    return rounded.astype(np.uint16).tobytes()


def _read_f32_initializer(item, external_data_dir: Optional[str]):
    """Return a float32 copy of an initializer, or None when the file is unknown."""
    dims = tuple(int(dim) for dim in item.dims)
    count = 1
    for dim in dims or (1,):
        count *= int(dim)
    if item.raw_data:
        array = np.frombuffer(item.raw_data, dtype=np.float32).copy()
        return array.reshape(dims) if dims else array
    if item.float_data:
        return np.asarray(list(item.float_data), dtype=np.float32).reshape(dims)
    location = None
    offset = 0
    length = count * 4
    for entry in item.external_data:
        if entry.key == "location":
            location = entry.value
        elif entry.key == "offset":
            offset = int(entry.value)
        elif entry.key == "length":
            length = int(entry.value)
    if not location or not external_data_dir:
        return None
    path = location if os.path.isabs(location) else os.path.join(external_data_dir, location)
    with open(path, "rb") as handle:
        handle.seek(int(offset))
        raw = handle.read(int(length))
    array = np.frombuffer(raw, dtype=np.float32).copy()
    return array.reshape(dims) if dims else array


def _replace_initializer(graph, tensor) -> None:
    kept = [item for item in graph.initializer if item.name != tensor.name]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    graph.initializer.append(tensor)


def _norm_weight_names(graph) -> List[str]:
    names = []
    for node in graph.node:
        if node.op_type == "SkipSimplifiedLayerNormalization":
            weight = _input_at(node, 2)
        elif node.op_type == "SimplifiedLayerNormalization":
            weight = _input_at(node, 1)
        else:
            continue
        if weight and weight not in names:
            names.append(weight)
    return names


_CHUNK_K = 4096

_PROTO_NUMPY = None


def _numpy_dtype(data_type: int):
    """Numpy dtype for a tensor proto type this split can slice. None otherwise."""
    global _PROTO_NUMPY
    if _PROTO_NUMPY is None:
        from onnx import TensorProto

        _PROTO_NUMPY = {
            int(TensorProto.FLOAT): np.float32,
            int(TensorProto.UINT8): np.uint8,
            int(TensorProto.INT8): np.int8,
            int(TensorProto.INT32): np.int32,
            int(TensorProto.INT64): np.int64,
            int(TensorProto.FLOAT16): np.float16,
        }
    return _PROTO_NUMPY.get(int(data_type))


def _choose_chunk_k(k: int, block_size: int, preferred: int = _CHUNK_K) -> Optional[int]:
    """Largest block-aligned chunk that divides K and is at most ``preferred``.

    ``K`` of 4096 or less is left whole. 12288 with block 32 uses 4096.
    14336 with block 128 cannot use 4096; 3584 is the next divisor.
    """
    width = int(k)
    block = int(block_size)
    target = int(preferred)
    if block < 1 or target < block or width <= target:
        return None
    if target % block == 0 and width % target == 0:
        return target
    chunk = (min(target, width // 2) // block) * block
    while chunk >= block:
        if width % chunk == 0:
            return chunk
        chunk -= block
    return None


def _initializer_array(item, external_data_dir: Optional[str]):
    """Copy an initializer. None when the storage is missing or the wrong size.

    External bytes are read from ``external_data_dir``. The data file is
    not opened for writing and is not rewritten.
    """
    np_dtype = _numpy_dtype(int(item.data_type))
    if np_dtype is None:
        return None
    dims = tuple(int(dim) for dim in item.dims)
    count = 1
    for dim in dims or (1,):
        count *= int(dim)
    if count < 1:
        return None
    itemsize = int(np.dtype(np_dtype).itemsize)
    expected = count * itemsize
    if item.raw_data:
        if len(item.raw_data) != expected:
            return None
        array = np.frombuffer(item.raw_data, dtype=np_dtype).copy()
        return array.reshape(dims) if dims else array
    if np_dtype == np.float32 and item.float_data and len(item.float_data) == count:
        return np.asarray(list(item.float_data), dtype=np.float32).reshape(dims)
    if np_dtype == np.int32 and item.int32_data and len(item.int32_data) == count:
        return np.asarray(list(item.int32_data), dtype=np.int32).reshape(dims)
    if np_dtype == np.int64 and item.int64_data and len(item.int64_data) == count:
        return np.asarray(list(item.int64_data), dtype=np.int64).reshape(dims)
    location = None
    offset = 0
    for entry in item.external_data:
        if entry.key == "location":
            location = entry.value
        elif entry.key == "offset":
            offset = int(entry.value)
        elif entry.key == "length" and int(entry.value) != expected:
            return None
    if not location or not external_data_dir:
        return None
    path = location if os.path.isabs(location) else os.path.join(external_data_dir, location)
    try:
        with open(path, "rb") as handle:
            handle.seek(int(offset))
            raw = handle.read(expected)
    except OSError:
        return None
    if len(raw) != expected:
        return None
    array = np.frombuffer(raw, dtype=np_dtype).copy()
    return array.reshape(dims) if dims else array


def _required_int_attr(node, name: str) -> Optional[int]:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return None


def _balanced_adds(names: Sequence[str], final_name: str, label: str):
    """Add partial MatMul outputs. The root keeps the original tensor name."""
    from onnx import helper

    current = [name for name in names if name]
    nodes = []
    level = 0
    while len(current) > 1:
        nxt = []
        pair = 0
        index = 0
        while index < len(current):
            if index + 1 == len(current):
                nxt.append(current[index])
                break
            out = f"{label}/sum_{level}_{pair}"
            nodes.append(
                helper.make_node(
                    "Add",
                    [current[index], current[index + 1]],
                    [out],
                    name=f"{label}/Add_{level}_{pair}",
                )
            )
            nxt.append(out)
            index += 2
            pair += 1
        current = nxt
        level += 1
    if nodes:
        nodes[-1].output[0] = final_name
    return nodes


def _split_one_matmul_nbits(graph, node, external_data_dir: Optional[str], chunk_k: int):
    """Split one MatMulNBits on K, or return None to leave it unchanged.

    Scales and packed blocks stay aligned because the chunk is a multiple
    of ``block_size``. A bias is added on the last partial only. A g_idx
    input is left whole. Zero-points are split only when they have one
    value per block.
    """
    from onnx import TensorProto, helper

    if node.op_type != "MatMulNBits" or node.domain != "com.microsoft":
        return None
    output = _output_at(node, 0)
    activation = _input_at(node, 0)
    weight_name = _input_at(node, 1)
    scale_name = _input_at(node, 2)
    if not output or not activation or not weight_name or not scale_name:
        return None
    if _input_at(node, 4):
        return None
    k = _required_int_attr(node, "K")
    n = _required_int_attr(node, "N")
    bits = _required_int_attr(node, "bits")
    block = _required_int_attr(node, "block_size")
    if None in (k, n, bits, block):
        return None
    if int(bits) != 4 or int(n) < 1 or (int(block) * int(bits)) % 8 != 0:
        return None
    chunk = _choose_chunk_k(int(k), int(block), chunk_k)
    if chunk is None or int(k) % int(block) != 0:
        return None
    blob = int(block) * int(bits) // 8
    n_blocks = int(k) // int(block)
    by_name = {item.name: item for item in graph.initializer}
    weight = by_name.get(weight_name)
    scales = by_name.get(scale_name)
    if weight is None or scales is None:
        return None
    if int(weight.data_type) != int(TensorProto.UINT8):
        return None
    if int(scales.data_type) != int(TensorProto.FLOAT):
        return None
    if tuple(int(dim) for dim in weight.dims) != (int(n), n_blocks, blob):
        return None
    if tuple(int(dim) for dim in scales.dims) != (int(n), n_blocks):
        return None
    zp_name = _input_at(node, 3)
    bias_name = _input_at(node, 5)
    zp_item = by_name.get(zp_name) if zp_name else None
    if zp_name and zp_item is None:
        return None
    if bias_name:
        bias_item = by_name.get(bias_name)
        if bias_item is not None and tuple(int(dim) for dim in bias_item.dims) != (int(n),):
            return None
    weight_array = _initializer_array(weight, external_data_dir)
    scale_array = _initializer_array(scales, external_data_dir)
    if weight_array is None or scale_array is None:
        return None
    zp_array = None
    zp_packed_nibbles = False
    if zp_item is not None:
        zp_array = _initializer_array(zp_item, external_data_dir)
        if zp_array is None:
            return None
        zp_shape = tuple(int(dim) for dim in zp_array.shape)
        if zp_shape == (int(n), n_blocks):
            zp_packed_nibbles = False
        elif (
            zp_shape == (int(n), (n_blocks + 1) // 2)
            and (chunk // int(block)) % 2 == 0
            and n_blocks % 2 == 0
        ):
            zp_packed_nibbles = True
        else:
            return None
    blocks_per_chunk = chunk // int(block)
    label = node.name or output
    parts = []
    nodes = []
    dropped = [weight_name, scale_name]
    if zp_name:
        dropped.append(zp_name)
    accuracy = _required_int_attr(node, "accuracy_level")
    for block_start in range(0, n_blocks, blocks_per_chunk):
        block_end = block_start + blocks_per_chunk
        k_start = block_start * int(block)
        k_end = block_end * int(block)
        piece = f"{label}/k{k_start}"
        _put_initializer(graph, f"{piece}_starts", np.asarray([k_start], dtype=np.int64))
        _put_initializer(graph, f"{piece}_ends", np.asarray([k_end], dtype=np.int64))
        _put_initializer(graph, f"{piece}_axes", np.asarray([-1], dtype=np.int64))
        _put_initializer(graph, f"{piece}_steps", np.asarray([1], dtype=np.int64))
        activation_out = f"{piece}/activation"
        nodes.append(
            helper.make_node(
                "Slice",
                [activation, f"{piece}_starts", f"{piece}_ends", f"{piece}_axes", f"{piece}_steps"],
                [activation_out],
                name=f"{piece}/Slice",
            )
        )
        packed_name = f"{weight_name}/k{k_start}"
        scale_chunk = f"{scale_name}/k{k_start}"
        _put_initializer(graph, packed_name, np.ascontiguousarray(weight_array[:, block_start:block_end, :]))
        _put_initializer(graph, scale_chunk, np.ascontiguousarray(scale_array[:, block_start:block_end]))
        inputs = [activation_out, packed_name, scale_chunk]
        if zp_array is not None:
            zp_chunk = f"{zp_name}/k{k_start}"
            if zp_packed_nibbles:
                _put_initializer(
                    graph,
                    zp_chunk,
                    np.ascontiguousarray(zp_array[:, block_start // 2 : block_end // 2]),
                )
            else:
                _put_initializer(
                    graph,
                    zp_chunk,
                    np.ascontiguousarray(zp_array[:, block_start:block_end]),
                )
            inputs.append(zp_chunk)
        last = block_end == n_blocks
        if bias_name:
            if zp_array is None:
                inputs.append("")
            inputs.append("")
            inputs.append(bias_name if last else "")
        kwargs = {"K": int(chunk), "N": int(n), "bits": int(bits), "block_size": int(block)}
        if accuracy is not None:
            kwargs["accuracy_level"] = int(accuracy)
        partial = f"{piece}/output"
        nodes.append(
            helper.make_node(
                "MatMulNBits",
                inputs,
                [partial],
                name=piece,
                domain="com.microsoft",
                **kwargs,
            )
        )
        parts.append(partial)
    nodes.extend(_balanced_adds(parts, output, label))
    return nodes, dropped


def _split_large_matmul_nbits(
    model,
    external_data_dir: Optional[str],
    chunk_k: int = _CHUNK_K,
) -> List[str]:
    """Split long MatMulNBits reductions. No readable bytes means no split."""
    if not external_data_dir or _model_opset(model, "") < 10:
        return []
    graph = model.graph
    rebuilt = []
    dropped = []
    split = []
    for node in list(graph.node):
        pieces = _split_one_matmul_nbits(graph, node, external_data_dir, int(chunk_k))
        if pieces is None:
            rebuilt.append(node)
            continue
        nodes, names = pieces
        rebuilt.extend(nodes)
        dropped.extend(names)
        split.append(node.name or _output_at(node, 0) or "MatMulNBits")
    if not split:
        return []
    del graph.node[:]
    graph.node.extend(rebuilt)
    used = set()
    for node in graph.node:
        used.update(item for item in node.input if item)
    unused = {name for name in dropped if name and name not in used}
    if unused:
        kept = [item for item in graph.initializer if item.name not in unused]
        del graph.initializer[:]
        graph.initializer.extend(kept)
        _drop_unused_inputs(graph, list(unused))
    return split


def _unpack_uint4(packed: np.ndarray, count: int) -> np.ndarray:
    """Return ``[count, rows]`` nibbles, low nibble first."""
    rows = int(packed.shape[0])
    flat = np.ascontiguousarray(packed, dtype=np.uint8).reshape(rows, -1)
    vals = np.empty((rows, flat.shape[1] * 2), dtype=np.uint8)
    vals[:, 0::2] = flat & np.uint8(0x0F)
    vals[:, 1::2] = (flat >> np.uint8(4)) & np.uint8(0x0F)
    if int(vals.shape[1]) < int(count):
        raise StaticShapeError(
            f"uint4 packing has {vals.shape[1]} nibbles, need {count}"
        )
    return np.ascontiguousarray(vals[:, : int(count)].T)


def _block_zero_points(zp_array, n: int, n_blocks: int, label: str) -> np.ndarray:
    """Per-block zero points as ``[n_blocks, N]``. An absent input is 8."""
    if zp_array is None:
        return np.full((n_blocks, n), 8.0, dtype=np.float32)
    shape = tuple(int(dim) for dim in zp_array.shape)
    if shape == (n, n_blocks):
        per_block = np.ascontiguousarray(zp_array.T)
    elif shape == (n, (n_blocks + 1) // 2):
        per_block = _unpack_uint4(zp_array, n_blocks)
    else:
        raise StaticShapeError(f"{label} zero-point shape {shape} is not a 4-bit layout")
    return per_block.astype(np.float32)


def _dequant_matmul_nbits(weight, scales, zp, k: int, n: int, block: int, label: str) -> np.ndarray:
    """Float32 ``[K, N]`` matrix matching CPU ``MatMulNBits``."""
    n_blocks = int(k) // int(block)
    blob = int(block) * 4 // 8
    if tuple(int(dim) for dim in weight.shape) != (int(n), n_blocks, blob):
        raise StaticShapeError(f"{label} packed weight shape {tuple(weight.shape)}")
    if tuple(int(dim) for dim in scales.shape) != (int(n), n_blocks):
        raise StaticShapeError(f"{label} scale shape {tuple(scales.shape)}")
    if scales.dtype != np.float32:
        raise StaticShapeError(f"{label} scales are {scales.dtype}, expected float32")
    nibbles = _unpack_uint4(weight, int(k)).astype(np.float32)
    scale_full = np.repeat(np.ascontiguousarray(scales.T), int(block), axis=0)
    zp_full = np.repeat(_block_zero_points(zp, int(n), n_blocks, label), int(block), axis=0)
    return (nibbles - zp_full) * scale_full


def _bake_one_matmul_nbits(
    graph,
    node,
    by_name,
    external_data_dir: Optional[str],
    *,
    compensate: bool = False,
    row_scale: Optional[np.ndarray] = None,
    col_scale: Optional[np.ndarray] = None,
):
    """Replace one MatMulNBits with Cast, baked MatMul, Cast, and an optional Add.

    When ``compensate`` is True, also emits a second bfloat16 MatMul on the
    residual ``values - float32(bf16(values))`` (Ozaki-style dual-bf16 split)
    and sums the two float32 outputs before any bias add, recovering ~15 bits
    of effective weight mantissa precision through standard ``GemmBfp16`` ops.
    When ``row_scale`` (length ``K``) and/or ``col_scale`` (length ``N``) are
    provided, scales the float32 dequantized weight rows/columns prior to
    bfloat16 rounding.
    """
    from onnx import TensorProto, helper

    output = _output_at(node, 0)
    activation = _input_at(node, 0)
    weight_name = _input_at(node, 1)
    scale_name = _input_at(node, 2)
    label = node.name or output or "MatMulNBits"
    if not output or not activation or not weight_name or not scale_name:
        raise StaticShapeError(f"{label} is missing an activation, weight, or scale")
    if _input_at(node, 4):
        raise StaticShapeError(f"{label} has g_idx, which this fallback does not bake")
    k = _required_int_attr(node, "K")
    n = _required_int_attr(node, "N")
    bits = _required_int_attr(node, "bits")
    block = _required_int_attr(node, "block_size")
    if None in (k, n, bits, block) or int(bits) != 4 or int(block) < 1 or int(k) % int(block) != 0:
        raise StaticShapeError(f"{label} is not a 4-bit block-aligned MatMulNBits")
    weight_item = by_name.get(weight_name)
    scale_item = by_name.get(scale_name)
    if weight_item is None or scale_item is None:
        raise StaticShapeError(f"{label} weight or scale initializer is missing")
    if int(weight_item.data_type) != int(TensorProto.UINT8):
        raise StaticShapeError(f"{label} packed weight is not uint8")
    weight = _initializer_array(weight_item, external_data_dir)
    scales = _initializer_array(scale_item, external_data_dir)
    if weight is None or scales is None:
        raise StaticShapeError(f"{label} weight or scale bytes could not be read")
    zp_name = _input_at(node, 3)
    zp = None
    if zp_name:
        zp_item = by_name.get(zp_name)
        if zp_item is None:
            raise StaticShapeError(f"{label} zero-point initializer is missing")
        zp = _initializer_array(zp_item, external_data_dir)
        if zp is None:
            raise StaticShapeError(f"{label} zero-point bytes could not be read")
    values = _dequant_matmul_nbits(weight, scales, zp, int(k), int(n), int(block), label)
    if row_scale is not None:
        rs = np.ascontiguousarray(row_scale, dtype=np.float32).reshape(int(k), 1)
        values = np.ascontiguousarray(values * rs, dtype=np.float32)
    if col_scale is not None:
        cs = np.ascontiguousarray(col_scale, dtype=np.float32).reshape(1, int(n))
        values = np.ascontiguousarray(values * cs, dtype=np.float32)
    raw_hi = _f32_to_bf16_raw(values)
    weight_out = f"{label}/bf16_weight"
    tensor = helper.make_tensor(
        weight_out,
        TensorProto.BFLOAT16,
        [int(k), int(n)],
        raw_hi,
        raw=True,
    )
    tensors = [tensor]
    act_bf = f"{label}/act_bf16"
    mm_bf = f"{label}/mm_bf16"
    # WORKAROUND: Ryzen AI 1.8.0 Vitis EP aborts in superkernel_GemmBf16Int4
    # on MatMulNBits (https://github.com/amd/RyzenAI-SW/issues/407). The
    # uint4 weights are baked here. Disable npu_bf16_fallback to keep the
    # packed op when a later release accepts it.
    nodes = [
        helper.make_node(
            "Cast",
            [activation],
            [act_bf],
            name=f"{label}/Cast_act_bf16",
            to=TensorProto.BFLOAT16,
        ),
        helper.make_node(
            "MatMul",
            [act_bf, weight_out],
            [mm_bf],
            name=f"{label}/MatMul",
        ),
    ]
    bias_name = _input_at(node, 5)
    if compensate:
        hi_bits = np.frombuffer(raw_hi, dtype=np.uint16).astype(np.uint32)
        hi_f32 = (hi_bits << np.uint32(16)).view(np.float32).reshape(int(k), int(n))
        residual = np.ascontiguousarray(values - hi_f32, dtype=np.float32)
        raw_lo = _f32_to_bf16_raw(residual)
        weight_lo_out = f"{label}/bf16_weight_residual"
        tensors.append(
            helper.make_tensor(
                weight_lo_out,
                TensorProto.BFLOAT16,
                [int(k), int(n)],
                raw_lo,
                raw=True,
            )
        )
        cast_hi = f"{label}/cast_hi_f32"
        mm_lo = f"{label}/mm_lo_bf16"
        cast_lo = f"{label}/cast_lo_f32"
        comp_out = f"{label}/cast_f32" if bias_name else output
        nodes.extend(
            [
                helper.make_node(
                    "Cast",
                    [mm_bf],
                    [cast_hi],
                    name=f"{label}/Cast_out_f32",
                    to=TensorProto.FLOAT,
                ),
                helper.make_node(
                    "MatMul",
                    [act_bf, weight_lo_out],
                    [mm_lo],
                    name=f"{label}/MatMul_residual",
                ),
                helper.make_node(
                    "Cast",
                    [mm_lo],
                    [cast_lo],
                    name=f"{label}/Cast_residual_f32",
                    to=TensorProto.FLOAT,
                ),
                helper.make_node(
                    "Add",
                    [cast_hi, cast_lo],
                    [comp_out],
                    name=f"{label}/Add_residual",
                ),
            ]
        )
        if bias_name:
            nodes.append(helper.make_node("Add", [comp_out, bias_name], [output], name=f"{label}/Add_bias"))
    elif bias_name:
        cast_out = f"{label}/cast_f32"
        nodes.append(
            helper.make_node(
                "Cast",
                [mm_bf],
                [cast_out],
                name=f"{label}/Cast_out_f32",
                to=TensorProto.FLOAT,
            )
        )
        nodes.append(helper.make_node("Add", [cast_out, bias_name], [output], name=f"{label}/Add_bias"))
    else:
        nodes.append(
            helper.make_node(
                "Cast",
                [mm_bf],
                [output],
                name=f"{label}/Cast_out_f32",
                to=TensorProto.FLOAT,
            )
        )
    dropped = [weight_name, scale_name]
    if zp_name:
        dropped.append(zp_name)
    return nodes, tensors, dropped


def _resolve_matmul_nbits_consumers(
    tensor_name: str,
    consumers_by_tensor: Dict[str, List[tuple]],
    by_name: Dict[str, object],
    external_data_dir: Optional[str],
    width: int,
) -> Optional[List[tuple]]:
    """Trace ``tensor_name`` to consumer ``MatMulNBits`` nodes (directly or via K-axis ``Slice``).

    Returns a list of ``(matmul_node, k_start, k_stop)`` tuples when every
    consumer of ``tensor_name`` is either a 4-bit ``MatMulNBits`` at input 0
    or a last-axis ``Slice`` whose output is consumed exclusively by a 4-bit
    ``MatMulNBits`` at input 0, or ``None`` if any consumer is unsupported.
    """
    readers = consumers_by_tensor.get(tensor_name, [])
    if not readers:
        return None
    resolved: List[tuple] = []
    for reader, pos in readers:
        if reader.op_type == "MatMulNBits" and reader.domain == "com.microsoft" and pos == 0:
            k_attr = _required_int_attr(reader, "K")
            if k_attr != int(width):
                return None
            resolved.append((reader, 0, int(width)))
            continue
        if reader.op_type == "Slice" and pos == 0:
            starts_arr = _initializer_array(by_name.get(_input_at(reader, 1)), external_data_dir)
            ends_arr = _initializer_array(by_name.get(_input_at(reader, 2)), external_data_dir)
            axes_arr = (
                _initializer_array(by_name.get(_input_at(reader, 3)), external_data_dir)
                if _input_at(reader, 3)
                else None
            )
            steps_arr = (
                _initializer_array(by_name.get(_input_at(reader, 4)), external_data_dir)
                if _input_at(reader, 4)
                else None
            )
            if starts_arr is None or ends_arr is None or starts_arr.size != 1 or ends_arr.size != 1:
                return None
            if axes_arr is not None and (axes_arr.size != 1 or int(axes_arr.reshape(-1)[0]) not in (-1, 2)):
                return None
            if steps_arr is not None and (steps_arr.size != 1 or int(steps_arr.reshape(-1)[0]) != 1):
                return None
            k_start = int(starts_arr.reshape(-1)[0])
            k_stop = min(int(ends_arr.reshape(-1)[0]), int(width))
            if k_start < 0 or k_stop <= k_start or k_stop > int(width):
                return None
            slice_out = _output_at(reader, 0)
            sub_readers = consumers_by_tensor.get(slice_out, []) if slice_out else []
            if not sub_readers:
                return None
            for sub_reader, sub_pos in sub_readers:
                if (
                    sub_reader.op_type != "MatMulNBits"
                    or sub_reader.domain != "com.microsoft"
                    or sub_pos != 0
                    or _required_int_attr(sub_reader, "K") != (k_stop - k_start)
                ):
                    return None
                resolved.append((sub_reader, k_start, k_stop))
            continue
        return None
    return resolved if resolved else None


def _power_of_two_block_scale(
    num_vec: np.ndarray,
    den_vec: np.ndarray,
    *,
    block_len: int = 8,
    max_exp: int = 3,
) -> np.ndarray:
    """Compute exact power-of-two per-channel scales ``2^k`` within each ``block_len`` window.

    Multiplying IEEE float32/bfloat16 values by an exact power of two shifts only
    exponent bits without altering mantissa bits, while equilibrating dynamic range
    inside each 8-element ``bfp16ebs8`` shared-exponent block.
    """
    width = int(num_vec.size)
    s = np.ones((width,), dtype=np.float32)
    for start in range(0, width, int(block_len)):
        end = min(start + int(block_len), width)
        n_blk = np.maximum(np.abs(num_vec[start:end].astype(np.float32)), 1e-6)
        d_blk = np.maximum(np.abs(den_vec[start:end].astype(np.float32)), 1e-6)
        ratio_raw = np.sqrt(n_blk / d_blk)
        g_med = float(np.exp(np.mean(np.log(np.maximum(ratio_raw, 1e-6)))))
        if g_med <= 1e-6:
            continue
        exp_int = np.clip(np.rint(np.log2(np.maximum(ratio_raw / g_med, 1e-6))), -int(max_exp), int(max_exp))
        s[start:end] = np.ldexp(np.ones_like(exp_int, dtype=np.float32), exp_int.astype(np.int32))
    return s


def _matmul_nbits_row_amplitude(
    targets: List[tuple],
    by_name: Dict[str, object],
    external_data_dir: Optional[str],
    width: int,
) -> np.ndarray:
    """Compute per-row RMS scale amplitude across ``(matmul_node, k_start, k_stop)`` consumers."""
    row_amp = np.zeros((int(width),), dtype=np.float32)
    for reader, k_start, k_stop in targets:
        chunk_len = int(k_stop) - int(k_start)
        s_name = _input_at(reader, 2)
        block_sz = _required_int_attr(reader, "block_size") or 32
        s_item = by_name.get(s_name)
        s_arr = _initializer_array(s_item, external_data_dir) if s_item is not None else None
        if s_arr is not None:
            flat_s = np.nan_to_num(s_arr.astype(np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
            flat_s = np.clip(flat_s, -1e6, 1e6)
            n_col = _required_int_attr(reader, "N") or 0
            n_blocks = chunk_len // int(block_sz)
            if n_col > 0 and n_blocks > 0 and flat_s.size == n_col * n_blocks:
                grid = flat_s.reshape(n_col, n_blocks)
                col_rms = np.sqrt(np.mean(np.square(grid, dtype=np.float64), axis=0)).astype(np.float32)
                row_amp[int(k_start) : int(k_stop)] = np.maximum(
                    row_amp[int(k_start) : int(k_stop)],
                    np.repeat(col_rms, int(block_sz)),
                )
    if not np.any(row_amp > 1e-8):
        row_amp = np.ones((int(width),), dtype=np.float32)
    return row_amp


def _matmul_nbits_col_amplitude(
    nodes: List[object],
    by_name: Dict[str, object],
    external_data_dir: Optional[str],
) -> Optional[np.ndarray]:
    """Compute per-column RMS scale amplitude across ``MatMulNBits`` K-chunks."""
    if not nodes:
        return None
    n_col = _required_int_attr(nodes[0], "N")
    if not n_col or n_col < 1:
        return None
    sum_sq = np.zeros((int(n_col),), dtype=np.float64)
    total_blocks = 0
    for node in nodes:
        if _required_int_attr(node, "N") != int(n_col):
            return None
        k_dim = _required_int_attr(node, "K")
        block_sz = _required_int_attr(node, "block_size") or 32
        if not k_dim or int(k_dim) % int(block_sz) != 0:
            return None
        s_name = _input_at(node, 2)
        s_item = by_name.get(s_name) if s_name else None
        s_arr = _initializer_array(s_item, external_data_dir) if s_item is not None else None
        if s_arr is None:
            continue
        flat_s = np.nan_to_num(s_arr.astype(np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        flat_s = np.clip(flat_s, -1e6, 1e6)
        n_blocks = int(k_dim) // int(block_sz)
        if flat_s.size != int(n_col) * n_blocks:
            continue
        grid = flat_s.reshape(int(n_col), n_blocks)
        sum_sq += np.sum(np.square(grid, dtype=np.float64), axis=1)
        total_blocks += n_blocks
    if total_blocks <= 0 or not np.any(sum_sq > 1e-16):
        return np.ones((int(n_col),), dtype=np.float32)
    col_rms = np.sqrt(sum_sq / float(total_blocks)).astype(np.float32)
    return np.maximum(col_rms, 1e-6)


def _trace_unbiased_matmul_nbits_group(
    tensor_name: str,
    producer_by_tensor: Dict[str, object],
    consumers_by_tensor: Dict[str, List[tuple]],
) -> Optional[List[object]]:
    """Return the ``MatMulNBits`` leaf nodes whose K-split ``Add`` tree produces ``tensor_name``."""
    if not tensor_name or len(consumers_by_tensor.get(tensor_name, [])) != 1:
        return None
    prod = producer_by_tensor.get(tensor_name)
    if prod is None:
        return None
    if prod.op_type == "MatMulNBits" and prod.domain == "com.microsoft":
        return [prod] if not _input_at(prod, 5) else None
    if prod.op_type == "Add":
        left = _trace_unbiased_matmul_nbits_group(
            _input_at(prod, 0), producer_by_tensor, consumers_by_tensor
        )
        right = _trace_unbiased_matmul_nbits_group(
            _input_at(prod, 1), producer_by_tensor, consumers_by_tensor
        )
        if left and right:
            return left + right
    return None


def _norm_equilibration_scales(
    graph,
    by_name,
    external_data_dir: Optional[str],
) -> tuple:
    """Compute power-of-two per-channel equilibration scales for ``MatMulNBits`` nodes.

    Covers:
    1. ``RMSNorm -> (optional K-Slice ->) MatMulNBits`` (e.g. ``qkv_proj``, ``gate_proj``, ``up_proj``),
       folding ``1/s`` into the float32 RMSNorm ``gamma`` and ``s`` into consumer ``MatMulNBits`` rows.
    2. ``up_proj -> Mul -> (optional K-Slice ->) down_proj``, folding ``1/s_mlp`` into ``up_proj``
       columns and ``s_mlp[k_start:k_stop]`` into ``down_proj`` rows.

    Returns ``(row_scales_by_node_key, col_scales_by_node_key)``.
    """
    from onnx import TensorProto, numpy_helper

    def _node_key(n) -> str:
        return n.name or _output_at(n, 0) or ""

    consumers_by_tensor: Dict[str, List[tuple]] = {}
    producer_by_tensor: Dict[str, object] = {}
    for node in graph.node:
        for out in node.output:
            if out:
                producer_by_tensor[out] = node
        for idx, inp in enumerate(node.input):
            if inp:
                consumers_by_tensor.setdefault(inp, []).append((node, idx))
    graph_outputs = {out.name for out in graph.output}

    candidates: List[tuple] = []
    for node in graph.node:
        if node.op_type == "SimplifiedLayerNormalization":
            weight_name = _input_at(node, 1)
            out_name = _output_at(node, 0)
            if weight_name and out_name:
                candidates.append((out_name, weight_name))
        elif node.op_type == "SkipSimplifiedLayerNormalization":
            weight_name = _input_at(node, 2)
            out_name = _output_at(node, 0)
            if weight_name and out_name:
                candidates.append((out_name, weight_name))
        elif node.op_type == "Mul" and node.name.endswith("/Mul_weight"):
            weight_name = _input_at(node, 1)
            out_name = _output_at(node, 0)
            if weight_name and out_name:
                candidates.append((out_name, weight_name))

    row_scales_by_node: Dict[str, np.ndarray] = {}
    col_scales_by_node: Dict[str, np.ndarray] = {}

    # 1. Equilibrate RMSNorm -> (Slice ->) MatMulNBits
    for act_name, weight_name in candidates:
        if act_name in graph_outputs:
            continue
        gamma_item = by_name.get(weight_name)
        if gamma_item is None or int(gamma_item.data_type) != int(TensorProto.FLOAT):
            continue
        gamma = _read_f32_initializer(gamma_item, external_data_dir)
        if gamma is None or gamma.ndim != 1 or gamma.size < 1:
            continue
        width = int(gamma.size)
        targets = _resolve_matmul_nbits_consumers(
            act_name, consumers_by_tensor, by_name, external_data_dir, width
        )
        if not targets:
            continue
        row_amp = _matmul_nbits_row_amplitude(targets, by_name, external_data_dir, width)
        s = _power_of_two_block_scale(gamma, row_amp, block_len=8, max_exp=3)
        new_gamma = np.ascontiguousarray(gamma.astype(np.float32) / s, dtype=np.float32)
        new_tensor = numpy_helper.from_array(new_gamma, name=weight_name)
        _replace_initializer(graph, new_tensor)
        by_name[weight_name] = new_tensor
        for reader, k_start, k_stop in targets:
            key = _node_key(reader)
            if key:
                row_scales_by_node[key] = np.ascontiguousarray(
                    s[int(k_start) : int(k_stop)], dtype=np.float32
                )

    # 2. Equilibrate up_proj -> Mul(SwiGLU) -> (Slice ->) down_proj
    for node in graph.node:
        if node.op_type != "Mul" or node.name.endswith("/Mul_weight"):
            continue
        mul_out = _output_at(node, 0)
        if not mul_out or mul_out in graph_outputs:
            continue
        in0 = _input_at(node, 0)
        in1 = _input_at(node, 1)
        grp0 = _trace_unbiased_matmul_nbits_group(in0, producer_by_tensor, consumers_by_tensor)
        grp1 = _trace_unbiased_matmul_nbits_group(in1, producer_by_tensor, consumers_by_tensor)
        # Prefer the right-hand Mul input when both sides are linear MatMulNBits
        # groups (e.g. synthetic test without Silu), or the sole linear branch on
        # real SwiGLU graphs where in0 is Silu(gate_proj) and in1 is up_proj.
        up_nodes = grp1 if grp1 else grp0
        if not up_nodes:
            continue
        inter_width = _required_int_attr(up_nodes[0], "N")
        if not inter_width or inter_width < 8:
            continue
        down_targets = _resolve_matmul_nbits_consumers(
            mul_out, consumers_by_tensor, by_name, external_data_dir, int(inter_width)
        )
        if not down_targets:
            continue
        up_col_amp = _matmul_nbits_col_amplitude(up_nodes, by_name, external_data_dir)
        if up_col_amp is None or up_col_amp.size != int(inter_width):
            continue
        down_row_amp = _matmul_nbits_row_amplitude(
            down_targets, by_name, external_data_dir, int(inter_width)
        )
        s_mlp = _power_of_two_block_scale(up_col_amp, down_row_amp, block_len=8, max_exp=3)
        for u_node in up_nodes:
            up_key = _node_key(u_node)
            if up_key:
                col_scales_by_node[up_key] = np.ascontiguousarray(1.0 / s_mlp, dtype=np.float32)
        for reader, k_start, k_stop in down_targets:
            key = _node_key(reader)
            if key:
                row_scales_by_node[key] = np.ascontiguousarray(
                    s_mlp[int(k_start) : int(k_stop)], dtype=np.float32
                )

    return row_scales_by_node, col_scales_by_node


def _bake_matmul_nbits(
    model,
    external_data_dir: Optional[str],
    *,
    compensate: bool = False,
    equilibrate: bool = False,
) -> tuple:
    """Bake every 4-bit MatMulNBits. Returns ``(baked_names, equilibrated_names)``."""
    graph = model.graph
    by_name = {item.name: item for item in graph.initializer}
    row_scales, col_scales = (
        _norm_equilibration_scales(graph, by_name, external_data_dir)
        if equilibrate
        else ({}, {})
    )
    rebuilt = []
    made = []
    dropped = []
    baked = []
    equilibrated = []
    for node in list(graph.node):
        if node.op_type != "MatMulNBits" or node.domain != "com.microsoft":
            rebuilt.append(node)
            continue
        node_key = node.name or _output_at(node, 0) or ""
        r_scale = row_scales.get(node_key)
        c_scale = col_scales.get(node_key)
        nodes, tensors, names = _bake_one_matmul_nbits(
            graph,
            node,
            by_name,
            external_data_dir,
            compensate=bool(compensate),
            row_scale=r_scale,
            col_scale=c_scale,
        )
        rebuilt.extend(nodes)
        made.extend(tensors)
        dropped.extend(names)
        label = node.name or _output_at(node, 0) or "MatMulNBits"
        baked.append(label)
        if r_scale is not None or c_scale is not None:
            equilibrated.append(label)
    if not baked:
        return [], []
    del graph.node[:]
    graph.node.extend(rebuilt)
    graph.initializer.extend(made)
    used = set()
    for node in graph.node:
        used.update(item for item in node.input if item)
    unused = {name for name in dropped if name and name not in used}
    if unused:
        kept = [item for item in graph.initializer if item.name not in unused]
        del graph.initializer[:]
        graph.initializer.extend(kept)
    return baked, equilibrated


def _apply_bf16_activation_contract(model, external_data_dir: Optional[str] = None) -> List[str]:
    """Cast float activations to bfloat16 ahead of MatMulNBits.

    Graph inputs stay float32. The hidden state is also the skip residual,
    so one entry cast feeds both. Norm scales become inline bfloat16
    initializers when their bytes can be read. Packed MatMulNBits weights
    stay on the external file. Scales share that op's activation type, so
    they are cast to bfloat16. Float graph outputs stay bfloat16.
    """
    from onnx import TensorProto, helper

    graph = model.graph
    casts = []
    converted = []
    by_name = {item.name: item for item in graph.initializer}
    for weight in _norm_weight_names(graph):
        item = by_name.get(weight)
        if item is None or int(item.data_type) != int(TensorProto.FLOAT):
            continue
        values = _read_f32_initializer(item, external_data_dir)
        if values is None:
            cast_out = f"{weight}/bf16"
            casts.append(
                helper.make_node(
                    "Cast",
                    [weight],
                    [cast_out],
                    name=f"Cast_{weight}_bf16",
                    to=TensorProto.BFLOAT16,
                )
            )
            for node in graph.node:
                for index, name in enumerate(node.input):
                    if name == weight:
                        node.input[index] = cast_out
            continue
        tensor = helper.make_tensor(
            weight,
            TensorProto.BFLOAT16,
            list(np.ascontiguousarray(values, dtype=np.float32).shape),
            _f32_to_bf16_raw(values),
            raw=True,
        )
        _replace_initializer(graph, tensor)
        converted.append(weight)
    by_name = {item.name: item for item in graph.initializer}
    for value in list(graph.input):
        if not value.type.HasField("tensor_type"):
            continue
        if int(value.type.tensor_type.elem_type) != int(TensorProto.FLOAT):
            continue
        dims = []
        for dim in value.type.tensor_type.shape.dim:
            if not dim.HasField("dim_value") or int(dim.dim_value) < 1:
                raise StaticShapeError(f"{value.name} has no static shape for the bfloat16 cast")
            dims.append(int(dim.dim_value))
        cast_out = f"{value.name}/bf16"
        casts.append(
            helper.make_node(
                "Cast",
                [value.name],
                [cast_out],
                name=f"Cast_{value.name}_bf16",
                to=TensorProto.BFLOAT16,
            )
        )
        _set_value_info(graph, cast_out, TensorProto.BFLOAT16, dims)
        for node in graph.node:
            for index, name in enumerate(node.input):
                if name == value.name:
                    node.input[index] = cast_out
    for node in graph.node:
        if node.op_type != "GroupQueryAttention":
            continue
        for index, name in enumerate(list(node.input)):
            item = by_name.get(name)
            if item is None or int(item.data_type) != int(TensorProto.FLOAT):
                continue
            cast_out = f"{name}/bf16"
            if not any(cast.output[0] == cast_out for cast in casts):
                casts.append(
                    helper.make_node(
                        "Cast",
                        [name],
                        [cast_out],
                        name=f"Cast_{name}_bf16",
                        to=TensorProto.BFLOAT16,
                    )
                )
            node.input[index] = cast_out
    for node in graph.node:
        if node.op_type != "MatMulNBits":
            continue
        for index, name in enumerate(list(node.input)):
            if index <= 1 or not name:
                continue
            item = by_name.get(name)
            if item is None or int(item.data_type) != int(TensorProto.FLOAT):
                continue
            cast_out = f"{name}/bf16"
            if not any(cast.output[0] == cast_out for cast in casts):
                casts.append(
                    helper.make_node(
                        "Cast",
                        [name],
                        [cast_out],
                        name=f"Cast_{name}_bf16",
                        to=TensorProto.BFLOAT16,
                    )
                )
            node.input[index] = cast_out
    for value in list(graph.output) + list(graph.value_info):
        if not value.type.HasField("tensor_type"):
            continue
        if int(value.type.tensor_type.elem_type) == int(TensorProto.FLOAT):
            value.type.tensor_type.elem_type = int(TensorProto.BFLOAT16)
    if casts:
        rebuilt = list(casts) + list(graph.node)
        del graph.node[:]
        graph.node.extend(rebuilt)
    return converted


def homogenize_qdq_weights_and_split_k(
    model,
    external_data_dir: Optional[str] = None,
    *,
    chunk_k: int = _CHUNK_K,
) -> Dict[str, List[str]]:
    """Convert signed ``int8`` weight ``DequantizeLinear`` tensors to unsigned ``uint8`` and split ``K > chunk_k`` QDQ ``MatMul`` nodes.

    Resolves two VAIML / XDNA2 INT8 QDQ obstacles observed on layer 0:
    1. ``VAIML-FRONTEND 2008`` (``tosa.reshape inferred tensor<...xi8> against
       return tensor<...xui8>``): when activation QDQ uses ``uint8`` and weight
       ``DequantizeLinear`` uses signed ``int8``, shifting the weight bytes and
       its zero-point by ``+128`` into ``uint8`` preserves exact float32
       dequantized values ``(w_u8 - (zp_i8 + 128)) * scale == (w_i8 - zp_i8) * scale``
       while making both QDQ branches ``uint8`` (``ui8``).
    2. ``qlinear-conv2d`` tiling fallback on ``down_proj`` (``K=12288``):
       splitting any 2D QDQ ``MatMul`` whose ``K`` exceeds ``chunk_k`` (and is
       divisible by ``chunk_k``) into ``K // chunk_k`` sliced-input QDQ
       ``MatMul`` chunks summed in ``float32`` via ``_balanced_adds`` reduces
       ``12288 x 4096`` into three ``4096 x 4096`` QDQ ``MatMul`` blocks.
    """
    from onnx import TensorProto, helper, numpy_helper

    graph = model.graph
    by_name = {item.name: item for item in graph.initializer}
    dq_by_out = {
        _output_at(node, 0): node
        for node in graph.node
        if node.op_type == "DequantizeLinear" and _output_at(node, 0)
    }

    converted_weights: List[str] = []
    # 1. Convert signed int8 weight DequantizeLinear initializers feeding MatMul input[1] to uint8 (+128).
    for node in list(graph.node):
        if node.op_type != "MatMul":
            continue
        w_dq_out = _input_at(node, 1)
        dq_node = dq_by_out.get(w_dq_out)
        if dq_node is None:
            continue
        w_name = _input_at(dq_node, 0)
        s_name = _input_at(dq_node, 1)
        zp_name = _input_at(dq_node, 2)
        w_item = by_name.get(w_name)
        s_item = by_name.get(s_name)
        if w_item is None or s_item is None:
            continue
        if int(w_item.data_type) != int(TensorProto.INT8):
            continue
        w_i8 = _initializer_array(w_item, external_data_dir)
        if w_i8 is None:
            continue
        if zp_name:
            zp_item = by_name.get(zp_name)
            if zp_item is None:
                continue
            zp_arr = _initializer_array(zp_item, external_data_dir)
            if zp_arr is None:
                continue
            zp_u8 = (zp_arr.astype(np.int16) + 128).astype(np.uint8)
        else:
            zp_s = _initializer_array(s_item, external_data_dir)
            zp_shape = () if zp_s is None or zp_s.ndim == 0 else zp_s.shape
            zp_u8 = np.full(zp_shape, 128, dtype=np.uint8)
            zp_name = f"{w_name}_u8_zp"
            while len(dq_node.input) < 3:
                dq_node.input.append("")
            dq_node.input[2] = zp_name
        w_u8 = (w_i8.astype(np.int16) + 128).astype(np.uint8)
        w_tensor = numpy_helper.from_array(np.ascontiguousarray(w_u8), name=w_name)
        zp_tensor = numpy_helper.from_array(np.ascontiguousarray(zp_u8), name=zp_name)
        _replace_initializer(graph, w_tensor)
        _replace_initializer(graph, zp_tensor)
        by_name[w_name] = w_tensor
        by_name[zp_name] = zp_tensor
        converted_weights.append(w_name)

    # 2. Split QDQ MatMul nodes with K > chunk_k along K.
    split_nodes: List[str] = []
    if int(chunk_k) > 0:
        rebuilt = []
        removed_dq_ids = set()
        made_inits = []
        dropped_names = []
        for node in list(graph.node):
            if node.op_type != "MatMul":
                rebuilt.append(node)
                continue
            act_in = _input_at(node, 0)
            w_dq_out = _input_at(node, 1)
            out_name = _output_at(node, 0)
            dq_node = dq_by_out.get(w_dq_out)
            if not act_in or not out_name or dq_node is None:
                rebuilt.append(node)
                continue
            w_name = _input_at(dq_node, 0)
            s_name = _input_at(dq_node, 1)
            zp_name = _input_at(dq_node, 2)
            w_item = by_name.get(w_name)
            s_item = by_name.get(s_name)
            zp_item = by_name.get(zp_name) if zp_name else None
            if w_item is None or s_item is None or (zp_name and zp_item is None):
                rebuilt.append(node)
                continue
            w_arr = _initializer_array(w_item, external_data_dir)
            s_arr = _initializer_array(s_item, external_data_dir)
            zp_arr = _initializer_array(zp_item, external_data_dir) if zp_item is not None else None
            if w_arr is None or s_arr is None or w_arr.ndim != 2:
                rebuilt.append(node)
                continue
            k_dim, n_dim = int(w_arr.shape[0]), int(w_arr.shape[1])
            target_k = int(chunk_k)
            if k_dim <= target_k or k_dim % target_k != 0:
                rebuilt.append(node)
                continue
            # Ensure scale/zp are either scalar, per-column (n_dim,), or per-row (k_dim, ...).
            dq_axis = _int_attr(dq_node, "axis", 1)
            if s_arr.ndim not in (0, 1) or (
                s_arr.ndim == 1 and s_arr.shape[0] not in (1, n_dim, k_dim)
            ):
                rebuilt.append(node)
                continue
            # Check if activation comes from an activation Q->DQ pair so each K-slice gets its own Q->DQ,
            # or slice the activation tensor directly before the MatMul.
            act_dq = dq_by_out.get(act_in)
            q_by_out = {
                _output_at(cand, 0): cand
                for cand in graph.node
                if cand.op_type == "QuantizeLinear" and _output_at(cand, 0)
            }
            act_q = q_by_out.get(_input_at(act_dq, 0)) if act_dq is not None else None

            label = node.name or out_name
            axis_name = f"{label}/split_axis"
            step_name = f"{label}/split_step"
            made_inits.extend(
                [
                    numpy_helper.from_array(np.asarray([-1], dtype=np.int64), name=axis_name),
                    numpy_helper.from_array(np.asarray([1], dtype=np.int64), name=step_name),
                ]
            )
            partials = []
            for start in range(0, k_dim, target_k):
                stop = start + target_k
                tag = f"k{start}"
                s_lo = f"{label}/{tag}_start"
                s_hi = f"{label}/{tag}_stop"
                made_inits.extend(
                    [
                        numpy_helper.from_array(np.asarray([start], dtype=np.int64), name=s_lo),
                        numpy_helper.from_array(np.asarray([stop], dtype=np.int64), name=s_hi),
                    ]
                )
                if act_dq is not None and act_q is not None:
                    raw_act = _input_at(act_q, 0)
                    act_s = _input_at(act_q, 1)
                    act_zp = _input_at(act_q, 2)
                    sliced_raw = f"{label}/{tag}/act_slice"
                    q_out = f"{label}/{tag}/act_q"
                    dq_act_out = f"{label}/{tag}/act_dq"
                    rebuilt.append(
                        helper.make_node(
                            "Slice",
                            [raw_act, s_lo, s_hi, axis_name, step_name],
                            [sliced_raw],
                            name=f"{label}/{tag}/Slice",
                        )
                    )
                    q_inputs = [sliced_raw, act_s] + ([act_zp] if act_zp else [])
                    rebuilt.append(
                        helper.make_node(
                            "QuantizeLinear",
                            q_inputs,
                            [q_out],
                            name=f"{label}/{tag}/QuantizeLinear",
                        )
                    )
                    dq_inputs = [q_out, _input_at(act_dq, 1)] + (
                        [_input_at(act_dq, 2)] if _input_at(act_dq, 2) else []
                    )
                    rebuilt.append(
                        helper.make_node(
                            "DequantizeLinear",
                            dq_inputs,
                            [dq_act_out],
                            name=f"{label}/{tag}/ActDequantizeLinear",
                        )
                    )
                    chunk_act = dq_act_out
                    removed_dq_ids.add(id(act_dq))
                    removed_dq_ids.add(id(act_q))
                else:
                    sliced_act = f"{label}/{tag}/act_slice"
                    rebuilt.append(
                        helper.make_node(
                            "Slice",
                            [act_in, s_lo, s_hi, axis_name, step_name],
                            [sliced_act],
                            name=f"{label}/{tag}/Slice",
                        )
                    )
                    chunk_act = sliced_act

                w_piece = np.ascontiguousarray(w_arr[start:stop, :])
                w_piece_name = f"{w_name}/{tag}"
                made_inits.append(numpy_helper.from_array(w_piece, name=w_piece_name))
                if s_arr.ndim == 1 and s_arr.shape[0] == k_dim and int(dq_axis) == 0:
                    s_piece = np.ascontiguousarray(s_arr[start:stop])
                    s_piece_name = f"{s_name}/{tag}"
                    made_inits.append(numpy_helper.from_array(s_piece, name=s_piece_name))
                else:
                    s_piece_name = s_name
                if zp_arr is not None:
                    if zp_arr.ndim == 1 and zp_arr.shape[0] == k_dim and int(dq_axis) == 0:
                        zp_piece = np.ascontiguousarray(zp_arr[start:stop])
                        zp_piece_name = f"{zp_name}/{tag}"
                        made_inits.append(numpy_helper.from_array(zp_piece, name=zp_piece_name))
                    else:
                        zp_piece_name = zp_name
                else:
                    zp_piece_name = ""

                w_dq_piece_out = f"{w_dq_out}/{tag}"
                w_dq_inputs = [w_piece_name, s_piece_name] + ([zp_piece_name] if zp_piece_name else [])
                w_dq_kwargs = {}
                if any(attr.name == "axis" for attr in dq_node.attribute):
                    w_dq_kwargs["axis"] = int(dq_axis)
                rebuilt.append(
                    helper.make_node(
                        "DequantizeLinear",
                        w_dq_inputs,
                        [w_dq_piece_out],
                        name=f"{dq_node.name or w_dq_out}/{tag}",
                        **w_dq_kwargs,
                    )
                )
                partial_out = f"{label}/{tag}/out"
                rebuilt.append(
                    helper.make_node(
                        "MatMul",
                        [chunk_act, w_dq_piece_out],
                        [partial_out],
                        name=f"{label}/{tag}/MatMul",
                    )
                )
                partials.append(partial_out)
            rebuilt.extend(_balanced_adds(partials, out_name, label))
            removed_dq_ids.add(id(dq_node))
            dropped_names.extend([w_name, s_name] + ([zp_name] if zp_name else []))
            split_nodes.append(label)

        if split_nodes:
            kept_nodes = [n for n in rebuilt if id(n) not in removed_dq_ids]
            del graph.node[:]
            graph.node.extend(kept_nodes)
            graph.initializer.extend(made_inits)
            used = {inp for n in graph.node for inp in n.input if inp}
            unused = {name for name in dropped_names if name and name not in used}
            if unused:
                kept_inits = [item for item in graph.initializer if item.name not in unused]
                del graph.initializer[:]
                graph.initializer.extend(kept_inits)

    return {
        "converted_u8_weights": converted_weights,
        "split_qdq_matmuls": split_nodes,
    }


def apply_static_contract(
    model,
    *,
    batch_size: int,
    max_seq_len: int,
    query_len: int,
    seq_axis: int = 2,
    head_dim: Optional[int] = None,
    external_data_dir: Optional[str] = None,
    npu_bf16_fallback: bool = False,
    npu_bf16_compensate: bool = False,
    npu_bf16_equilibrate: bool = False,
    homogenize_qdq: bool = False,
    defer_matmul_bake: bool = False,
    decompose_gqa: bool = False,
    sink_size: int = 0,
    matmul_chunk_k: int = _CHUNK_K,
) -> Dict[str, object]:
    """Freeze IO shapes, widen a KV prefix slice, and pin a Qwen GQA window.

    A long ``MatMulNBits`` K axis is split before either the bfloat16 scale
    casts or the dense fallback, so each chunk is still a float initializer.
    ``npu_bf16_fallback`` defaults off. When it is on, each remaining 4-bit
    ``MatMulNBits`` becomes a baked bfloat16 ``MatMul`` and the norm body
    stays float32. ``defer_matmul_bake`` performs the same structural
    rewrites, leaves the packed ops in place, and keeps the norm body in
    float32. It is not combined with baking this graph. Ryzen AI 1.8.0
    aborts in ``superkernel_GemmBf16Int4``
    (https://github.com/amd/RyzenAI-SW/issues/407).
    """
    from onnx import TensorProto

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
    if decompose_gqa:
        attention = decompose_gqa_to_npu_primitives(
            model,
            batch_size=batch_size,
            query_len=query_len,
            max_seq_len=max_seq_len,
            sink_size=int(sink_size),
        )
    else:
        attention = rewrite_gqa_to_static_window(
            model,
            batch_size=batch_size,
            query_len=query_len,
            max_seq_len=max_seq_len,
        )
    qdq_report = (
        homogenize_qdq_weights_and_split_k(
            model, external_data_dir, chunk_k=int(matmul_chunk_k)
        )
        if homogenize_qdq
        else {"converted_u8_weights": [], "split_qdq_matmuls": []}
    )
    split = _split_large_matmul_nbits(model, external_data_dir, chunk_k=int(matmul_chunk_k))
    bf16_weights: List[str] = []
    baked: List[str] = []
    equilibrated: List[str] = []
    activation = None
    activation_name = "float32"
    if npu_bf16_fallback and defer_matmul_bake:
        raise StaticShapeError(
            "defer_matmul_bake keeps MatMulNBits packed and cannot bake this graph"
        )
    if npu_bf16_fallback:
        baked, equilibrated = _bake_matmul_nbits(
            model,
            external_data_dir,
            compensate=bool(npu_bf16_compensate),
            equilibrate=bool(npu_bf16_equilibrate),
        )
        if any(node.op_type == "MatMulNBits" for node in model.graph.node):
            raise StaticShapeError("MatMulNBits remains after npu_bf16_fallback")
        activation = TensorProto.FLOAT
    elif defer_matmul_bake:
        # Packed weights stay. The norm body stays float32 so a later
        # per-component bake does not inherit the bfloat16 activation contract.
        activation = TensorProto.FLOAT
    elif any(node.op_type == "MatMulNBits" for node in model.graph.node):
        bf16_weights = _apply_bf16_activation_contract(model, external_data_dir)
        activation = TensorProto.BFLOAT16
        activation_name = "bfloat16"
    norms = rewrite_fused_norms(model, activation_type=activation)
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
        "decomposed_gqa": bool(decompose_gqa),
        "sink_size": int(sink_size),
        "decomposed_norms": norms,
        "activation": activation_name,
        "bf16_norm_weights": bf16_weights,
        "split_matmuls": split,
        "matmul_chunk_k": int(matmul_chunk_k),
        "bf16_fallback": bool(npu_bf16_fallback),
        "bf16_compensate": bool(npu_bf16_compensate),
        "bf16_equilibrate": bool(equilibrated),
        "equilibrated_matmuls": equilibrated,
        "bf16_fallback_matmuls": baked,
        "matmul_bake_deferred": bool(defer_matmul_bake),
        "homogenized_qdq": qdq_report,
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


def right_aligned_attention_bias(
    past_valid: int,
    max_seq_len: int,
    query_len: int = 1,
    *,
    total_len: Optional[int] = None,
    sink_size: int = 0,
    causal: Optional[bool] = None,
    value: float = ADDITIVE_BIAS,
    dtype=np.float32,
):
    """
    Right-aligned additive attention bias of shape ``(1, 1, query_len, total_len)``.

    For static-window NPU decode graphs where ``present`` slices the trailing
    ``max_seq_len`` slots (or ``[0:sink_size]`` + ``[sink_size+Q:max_seq_len+Q]``),
    valid past tokens occupy the rightmost ``min(past_valid, max_seq_len)`` slots
    of the ``max_seq_len`` past window, and the new ``query_len`` tokens occupy
    the final ``query_len`` columns of ``total_len``.
    """
    past_cap = int(max_seq_len)
    query = int(query_len)
    if past_cap < 1 or query < 1:
        raise StaticShapeError(f"max_seq_len {max_seq_len} and query_len {query_len} must be positive")
    if int(past_valid) < 0:
        raise StaticShapeError(f"past_valid must be >= 0, got {past_valid}")
    if int(sink_size) < 0 or int(sink_size) > past_cap:
        raise StaticShapeError(f"sink_size {sink_size} must be in 0..{past_cap}")
    key_len = int(total_len) if total_len is not None else past_cap + query
    if key_len not in (past_cap, past_cap + query):
        raise StaticShapeError(
            f"total_len {key_len} must equal max_seq_len ({past_cap}) or max_seq_len + query_len ({past_cap + query})"
        )
    valid_past = min(int(past_valid), past_cap)
    if key_len == past_cap:
        valid_total = min(past_cap, valid_past + query)
        empty = past_cap - valid_total
    else:
        empty = past_cap - valid_past

    bias = np.zeros((1, 1, query, key_len), dtype=dtype)
    blocked = np.asarray(value, dtype=dtype)
    if empty > 0:
        if int(sink_size) > 0:
            bias[:, :, :, :past_cap] = blocked
            sink_keep = min(int(sink_size), valid_past)
            if sink_keep > 0:
                bias[:, :, :, :sink_keep] = np.asarray(0.0, dtype=dtype)
            recent_keep = max(0, valid_past - sink_keep)
            if recent_keep > 0:
                bias[:, :, :, past_cap - recent_keep : past_cap] = np.asarray(0.0, dtype=dtype)
        else:
            bias[:, :, :, :empty] = blocked
    if causal is None:
        causal = query > 1
    if causal and query > 1:
        query_start = key_len - query
        cols = np.arange(key_len)
        for row in range(query):
            bias[0, 0, row, cols > (query_start + row)] = blocked
    return bias
