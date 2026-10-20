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
        node.op_type == "GroupQueryAttention" and len(node.input) > 10 and node.input[10] == bias_name
        for node in graph.node
    ):
        return
    graph.input.append(helper.make_tensor_value_info(bias_name, TensorProto.FLOAT, shape))


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
    if zp_item is not None:
        zp_array = _initializer_array(zp_item, external_data_dir)
        if zp_array is None or tuple(zp_array.shape) != (int(n), n_blocks):
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
            _put_initializer(graph, zp_chunk, np.ascontiguousarray(zp_array[:, block_start:block_end]))
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


def _bake_one_matmul_nbits(graph, node, by_name, external_data_dir: Optional[str]):
    """Replace one MatMulNBits with Cast, baked MatMul, Cast, and an optional Add."""
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
    weight_out = f"{label}/bf16_weight"
    tensor = helper.make_tensor(
        weight_out,
        TensorProto.BFLOAT16,
        [int(k), int(n)],
        _f32_to_bf16_raw(values),
        raw=True,
    )
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
    if bias_name:
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
    return nodes, [tensor], dropped


def _bake_matmul_nbits(model, external_data_dir: Optional[str]) -> List[str]:
    """Bake every 4-bit MatMulNBits. The external data file is not rewritten."""
    graph = model.graph
    by_name = {item.name: item for item in graph.initializer}
    rebuilt = []
    made = []
    dropped = []
    baked = []
    for node in list(graph.node):
        if node.op_type != "MatMulNBits" or node.domain != "com.microsoft":
            rebuilt.append(node)
            continue
        nodes, tensors, names = _bake_one_matmul_nbits(graph, node, by_name, external_data_dir)
        rebuilt.extend(nodes)
        made.extend(tensors)
        dropped.extend(names)
        baked.append(node.name or _output_at(node, 0) or "MatMulNBits")
    if not baked:
        return []
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
    return baked


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
    defer_matmul_bake: bool = False,
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
    attention = rewrite_gqa_to_static_window(
        model,
        batch_size=batch_size,
        query_len=query_len,
        max_seq_len=max_seq_len,
    )
    split = _split_large_matmul_nbits(model, external_data_dir)
    bf16_weights: List[str] = []
    baked: List[str] = []
    activation = None
    activation_name = "float32"
    if npu_bf16_fallback and defer_matmul_bake:
        raise StaticShapeError(
            "defer_matmul_bake keeps MatMulNBits packed and cannot bake this graph"
        )
    if npu_bf16_fallback:
        baked = _bake_matmul_nbits(model, external_data_dir)
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
        "decomposed_norms": norms,
        "activation": activation_name,
        "bf16_norm_weights": bf16_weights,
        "split_matmuls": split,
        "bf16_fallback": bool(npu_bf16_fallback),
        "bf16_fallback_matmuls": baked,
        "matmul_bake_deferred": bool(defer_matmul_bake),
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
