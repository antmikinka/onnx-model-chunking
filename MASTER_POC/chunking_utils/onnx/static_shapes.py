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
``attention_bias`` has length ``total_sequence_length``.
``SkipSimplifiedLayerNormalization`` and ``SimplifiedLayerNormalization``
become Add (for the skip) plus the primitive RMSNorm chain. A default
Vitis compile of the earlier window, with total frozen at ``max_seq_len``,
reported CPU 18.
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


def _numpy_dtype(elem_type: int):
    from onnx import TensorProto

    mapping = {
        TensorProto.FLOAT: np.float32,
        TensorProto.FLOAT16: np.float16,
        TensorProto.DOUBLE: np.float64,
    }
    if elem_type not in mapping:
        raise StaticShapeError(f"RMSNorm rewrite does not support elem_type {elem_type}")
    return mapping[elem_type]


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
    length ``total_sequence_length``.
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


def _reduce_mean(opset: int, *, mean_input: str, mean_output: str, axes_name: str, axis: int, name: str):
    """ReduceMean uses an axes input from opset 18 onward."""
    from onnx import helper

    if int(opset) >= 18:
        return helper.make_node(
            "ReduceMean",
            [mean_input, axes_name],
            [mean_output],
            name=name,
            keepdims=1,
        )
    return helper.make_node(
        "ReduceMean",
        [mean_input],
        [mean_output],
        name=name,
        axes=[int(axis)],
        keepdims=1,
    )


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
    """Primitive RMSNorm: x * rsqrt(mean(x^2) + eps) * weight.

    VAIML re-fused both this Pow chain and a Mul(x, x) square into
    SinglePassRMSNorm, then failed L1 placement on the 4096-wide norms.
    """
    from onnx import helper

    dtype = _numpy_dtype(elem_type)
    two = f"{label}/rms_two"
    one = f"{label}/rms_one"
    eps = f"{label}/rms_epsilon"
    axes = f"{label}/rms_axes"
    _put_initializer(graph, two, np.asarray(2, dtype=dtype))
    _put_initializer(graph, one, np.asarray(1, dtype=dtype))
    _put_initializer(graph, eps, np.asarray(epsilon, dtype=dtype))
    if int(opset) >= 18:
        _put_initializer(graph, axes, np.asarray([int(axis)], dtype=np.int64))
    squared = f"{label}/rms_squared"
    mean = mean_name or f"{label}/rms_mean"
    shifted = f"{label}/rms_shifted"
    std = f"{label}/rms_std"
    inv = inv_name or f"{label}/rms_inv"
    scaled = f"{label}/rms_scaled"
    nodes = [
        helper.make_node("Pow", [source, two], [squared], name=f"{label}/Pow"),
        _reduce_mean(
            opset,
            mean_input=squared,
            mean_output=mean,
            axes_name=axes,
            axis=int(axis),
            name=f"{label}/ReduceMean",
        ),
        helper.make_node("Add", [mean, eps], [shifted], name=f"{label}/Add_epsilon"),
        helper.make_node("Sqrt", [shifted], [std], name=f"{label}/Sqrt"),
        helper.make_node("Div", [one, std], [inv], name=f"{label}/Div"),
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


def rewrite_fused_norms(model) -> List[str]:
    """
    Replace fused RMSNorm ops with primitive ONNX nodes.

    ``SkipSimplifiedLayerNormalization`` output 0 is the normalized sum and
    output 3 is that sum before normalization. ``SimplifiedLayerNormalization``
    is the same chain without the residual add. Qwen's input norm is hidden
    4096, the same width as the skip norm that failed SinglePassRMSNorm tiling.
    """
    opset = _model_opset(model, "")
    graph = model.graph
    rewritten = []
    rebuilt = []
    for node in list(graph.node):
        if node.op_type == "SkipSimplifiedLayerNormalization":
            rebuilt.extend(_decompose_skip_norm(graph, node, opset))
            rewritten.append(node.name or _output_at(node, 0) or "SkipSimplifiedLayerNormalization")
            continue
        if node.op_type == "SimplifiedLayerNormalization":
            rebuilt.extend(_decompose_simple_norm(graph, node, opset))
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


def _decompose_skip_norm(graph, node, opset: int):
    from onnx import TensorProto

    label = node.name or "SkipSimplifiedLayerNormalization"
    nodes, source = _residual_nodes(node, label)
    output = _output_at(node, 0)
    if not output:
        return nodes
    weight = _input_at(node, 2)
    epsilon = _float_attr(node, "epsilon", 1e-5)
    axis = _int_attr(node, "axis", -1)
    elem = _elem_type(graph, _input_at(node, 0), TensorProto.FLOAT)
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


def _decompose_simple_norm(graph, node, opset: int):
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
    elem = _elem_type(graph, source, TensorProto.FLOAT)
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
    norms = rewrite_fused_norms(model)
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
