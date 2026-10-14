"""Equal-tile LM head. Pad the vocabulary axis, then slice the pad off.

Embeddings keep the short tail from ``vocab_chunk_rows``. This rewrite
runs on one LM-head component graph. It does not bake the unsplit model.

``npu_bf16_fallback`` dequantizes a ``MatMulNBits`` in this graph first
(``(nibble - zp) * scale``, omitted zp 8) and then chunks the ``[K, N]``
matrix. The packed op stays when the flag is off. Ryzen AI 1.8.0 aborts
in ``superkernel_GemmBf16Int4``
(https://github.com/amd/RyzenAI-SW/issues/407). The bake does not place
the graph.

MatMul weights are ``[K, N]``. Each tile is ``[K, chunk]``. A table stored
as ``[N, K]`` is rejected. The last-token slice is inserted only on this
projection. A symbolic sequence uses a runtime slice, which AOT NPU
compilation rejects. A static sequence uses a constant index.
"""

from __future__ import annotations

import importlib.util
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from .static_shapes import (
    StaticShapeError,
    _dequant_matmul_nbits,
    _f32_to_bf16_raw,
    _initializer_array,
    _input_at,
    _required_int_attr,
)


_PLANNER = None


def _planner():
    global _PLANNER
    if _PLANNER is not None:
        return _PLANNER
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "pipeline_utils",
        "vocab_sram.py",
    )
    spec = importlib.util.spec_from_file_location("vocab_sram", os.path.abspath(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load vocabulary SRAM planner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _PLANNER = module
    return module


def _optional_int_attr(node, name: str) -> Optional[int]:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return None


def _value_by_name(graph, name: str):
    for collection in (graph.input, graph.value_info, graph.output):
        for item in collection:
            if item.name == name:
                return item
    return None


def _shape_of(graph, name: str):
    info = _value_by_name(graph, name)
    if info is None or not info.type.HasField("tensor_type"):
        return None
    dims = []
    for dim in info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param") and dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append("unk")
    return dims or None


def _elem_of(graph, name: str, default: int) -> int:
    info = _value_by_name(graph, name)
    if info is None or not info.type.HasField("tensor_type"):
        return int(default)
    elem = int(info.type.tensor_type.elem_type)
    return elem or int(default)


def _set_dim(info, index: int, value: int) -> None:
    if info is None or not info.type.HasField("tensor_type"):
        return
    dims = info.type.tensor_type.shape.dim
    if index < 0:
        index += len(dims)
    if index < 0 or index >= len(dims):
        return
    dim = dims[index]
    dim.ClearField("dim_param")
    dim.dim_value = int(value)


def _ensure_value_info(graph, name: str, elem: int, shape: Optional[Sequence]) -> None:
    if not name or shape is None:
        return
    if any(item.name == name for item in graph.input):
        return
    existing = _value_by_name(graph, name)
    if existing is None:
        graph.value_info.append(helper.make_tensor_value_info(name, int(elem), list(shape)))
        return
    tensor_shape = existing.type.tensor_type.shape
    del tensor_shape.dim[:]
    for dim in shape:
        entry = tensor_shape.dim.add()
        if isinstance(dim, str):
            entry.dim_param = dim
        else:
            entry.dim_value = int(dim)
    existing.type.tensor_type.elem_type = int(elem)


def _proj_shape(act_shape, columns: int, last_token: bool):
    if not act_shape:
        return None
    shape = list(act_shape)
    if last_token and len(shape) >= 2:
        shape[-2] = 1
    shape[-1] = int(columns)
    return shape


def _i64(name: str, values: Sequence[int]):
    return numpy_helper.from_array(np.asarray(list(values), dtype=np.int64), name)


def _producer(graph, name: str):
    for node in graph.node:
        if name in node.output:
            return node
    return None


def _consumer_count(graph, name: str) -> int:
    return sum(1 for node in graph.node if name in node.input)


def _find_projection(graph):
    nbits = [
        node for node in graph.node
        if node.op_type == "MatMulNBits" and node.domain == "com.microsoft"
    ]
    matmuls = [
        node for node in graph.node
        if node.op_type == "MatMul" and node.domain in ("", "ai.onnx")
    ]
    if len(nbits) == 1 and not matmuls:
        return "nbits", nbits[0]
    if len(matmuls) == 1 and not nbits:
        return "dense", matmuls[0]
    named = [node for node in nbits + matmuls if "lm_head" in (node.name or "")]
    if len(named) == 1:
        node = named[0]
        return ("nbits" if node.op_type == "MatMulNBits" else "dense"), node
    raise StaticShapeError(
        f"LM-head rewrite needs one projection, found "
        f"{len(nbits)} MatMulNBits and {len(matmuls)} MatMul"
    )


def _read_bytes(item, external_data_dir: Optional[str], itemsize: int) -> bytes:
    dims = tuple(int(dim) for dim in item.dims)
    count = 1
    for dim in dims or (1,):
        count *= int(dim)
    expected = count * int(itemsize)
    if item.raw_data:
        if len(item.raw_data) != expected:
            raise StaticShapeError(f"{item.name} has {len(item.raw_data)} bytes, expected {expected}")
        return bytes(item.raw_data)
    location = None
    offset = 0
    for entry in item.external_data:
        if entry.key == "location":
            location = entry.value
        elif entry.key == "offset":
            offset = int(entry.value)
        elif entry.key == "length" and int(entry.value) != expected:
            raise StaticShapeError(f"{item.name} external length is not {expected}")
    if not location or not external_data_dir:
        raise StaticShapeError(f"{item.name} bytes are not inline and have no external directory")
    path = location if os.path.isabs(location) else os.path.join(external_data_dir, location)
    with open(path, "rb") as handle:
        handle.seek(int(offset))
        raw = handle.read(expected)
    if len(raw) != expected:
        raise StaticShapeError(f"{item.name} external read returned {len(raw)} bytes")
    return raw


def _read_matrix(item, external_data_dir: Optional[str]):
    dtype = int(item.data_type)
    dims = tuple(int(dim) for dim in item.dims)
    if dtype == int(TensorProto.BFLOAT16):
        raw = _read_bytes(item, external_data_dir, 2)
        return np.frombuffer(raw, dtype=np.uint16).copy().reshape(dims), dtype
    array = _initializer_array(item, external_data_dir)
    if array is None:
        raise StaticShapeError(f"{item.name} could not be read")
    return array, dtype


def _store(name: str, array: np.ndarray, data_type: int):
    if int(data_type) == int(TensorProto.BFLOAT16):
        raw = np.ascontiguousarray(array, dtype=np.uint16).tobytes()
        return helper.make_tensor(name, TensorProto.BFLOAT16, list(array.shape), raw, raw=True)
    return numpy_helper.from_array(np.ascontiguousarray(array), name)


def _pad_columns(array: np.ndarray, pad: int) -> np.ndarray:
    """Zero-fill the vocab remainder.

    Byte-identical tiles plus this zero tail raised VAIML-BE-TPGEN 9032.
    Distinct per-tile constants with the same zero tail compiled
    (VAIML 153, NPU 1, ELF 5328344). The Slice drops these columns.
    """
    if pad <= 0:
        return array
    width = [(0, 0)] * array.ndim
    width[-1] = (0, int(pad))
    return np.pad(array, width, mode="constant")


def _pad_rows(array: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return array
    width = [(0, 0)] * array.ndim
    width[0] = (0, int(pad))
    return np.pad(array, width, mode="constant")


def _column_tile(array: np.ndarray, start: int, rows: int) -> np.ndarray:
    total = int(array.shape[-1])
    end = min(start + rows, total)
    piece = array[..., start:end]
    missing = rows - (end - start)
    if missing:
        piece = _pad_columns(piece, missing)
    return np.ascontiguousarray(piece)


def _row_tile(array: np.ndarray, start: int, rows: int) -> np.ndarray:
    total = int(array.shape[0])
    end = min(start + rows, total)
    piece = array[start:end]
    missing = rows - (end - start)
    if missing:
        piece = _pad_rows(piece, missing)
    return np.ascontiguousarray(piece)


def _last_token(activation: str, seq_len, prefix: str):
    sliced = f"{prefix}/last_hidden"
    if isinstance(seq_len, int):
        if seq_len < 1:
            raise StaticShapeError(f"{prefix} sequence {seq_len} is empty")
        inits = [
            _i64(f"{prefix}/last_starts", [int(seq_len) - 1]),
            _i64(f"{prefix}/last_ends", [int(seq_len)]),
            _i64(f"{prefix}/last_axes", [1]),
        ]
        nodes = [
            helper.make_node(
                "Slice",
                [activation, inits[0].name, inits[1].name, inits[2].name],
                [sliced],
                name=f"{prefix}/last_token",
            )
        ]
        return nodes, inits, sliced
    inits = [
        _i64(f"{prefix}/seq_index", [1]),
        _i64(f"{prefix}/seq_one", [1]),
        _i64(f"{prefix}/last_axes", [1]),
    ]
    shape = f"{prefix}/shape"
    seq = f"{prefix}/seq"
    starts = f"{prefix}/last_starts"
    nodes = [
        helper.make_node("Shape", [activation], [shape], name=f"{prefix}/Shape"),
        helper.make_node("Gather", [shape, inits[0].name], [seq], name=f"{prefix}/Seq", axis=0),
        helper.make_node("Sub", [seq, inits[1].name], [starts], name=f"{prefix}/SeqStart"),
        helper.make_node(
            "Slice",
            [activation, starts, seq, inits[2].name],
            [sliced],
            name=f"{prefix}/last_token",
        ),
    ]
    return nodes, inits, sliced


def _prepare_activation(graph, node, last_token: bool):
    """Slice the last token on the projection input. A lone Cast is moved after it."""
    activation = _input_at(node, 0)
    cast = None
    source = activation
    producer = _producer(graph, activation)
    if (
        producer is not None
        and producer.op_type == "Cast"
        and _consumer_count(graph, activation) == 1
    ):
        cast = producer
        source = _input_at(producer, 0)
    shape = _shape_of(graph, source)
    if not last_token or shape is None or len(shape) < 3:
        return activation, cast, [], [], shape, False
    seq = shape[-2]
    nodes, inits, sliced = _last_token(source, seq if isinstance(seq, int) else None, node.name or "lm_head")
    sliced_shape = list(shape)
    sliced_shape[-2] = 1
    source_elem = _elem_of(graph, source, TensorProto.FLOAT)
    _ensure_value_info(graph, sliced, source_elem, sliced_shape)
    if cast is None:
        return sliced, None, nodes, inits, sliced_shape, True
    cast.input[0] = sliced
    cast_shape = list(sliced_shape)
    if cast_shape:
        cast_shape[-1] = shape[-1]
    _ensure_value_info(graph, activation, _elem_of(graph, activation, source_elem), cast_shape)
    return activation, cast, nodes, inits, cast_shape, True


def _default_chunk(vocab: int, hidden: int, dtype_bytes: int) -> int:
    sizes = _planner().vocab_chunk_rows(int(vocab), int(hidden), int(dtype_bytes))
    return int(sizes[0])


def _sizes_for(vocab: int, hidden: int, chunk_rows: Optional[int], dtype_bytes: int) -> List[int]:
    chunk = int(chunk_rows) if chunk_rows is not None else _default_chunk(vocab, hidden, dtype_bytes)
    return _planner().uniform_lm_head_rows(int(vocab), chunk)


def _vocab_slice(wide: str, output: str, vocab: int, prefix: str):
    inits = [
        _i64(f"{prefix}/trim_starts", [0]),
        _i64(f"{prefix}/trim_ends", [int(vocab)]),
        _i64(f"{prefix}/trim_axes", [-1]),
    ]
    node = helper.make_node(
        "Slice",
        [wide, inits[0].name, inits[1].name, inits[2].name],
        [output],
        name=f"{prefix}/trim",
    )
    return node, inits


def _concat_or_single(parts: Sequence[str], wide: str, prefix: str):
    if len(parts) == 1:
        return parts[0], []
    return wide, [
        helper.make_node("Concat", list(parts), [wide], name=f"{prefix}/concat", axis=-1)
    ]


def _splice(graph, remove_node, cast_node, prelude, replacement) -> None:
    rebuilt = []
    pending = list(prelude)
    for node in graph.node:
        if cast_node is not None and node is cast_node:
            rebuilt.extend(pending)
            pending = []
            rebuilt.append(node)
        elif node is remove_node:
            if pending:
                rebuilt.extend(pending)
                pending = []
            rebuilt.extend(replacement)
        else:
            rebuilt.append(node)
    del graph.node[:]
    graph.node.extend(rebuilt)


def _drop_unused(graph, names: Sequence[str]) -> None:
    used = set()
    for node in graph.node:
        used.update(item for item in node.input if item)
    drop = {name for name in names if name and name not in used}
    if not drop:
        return
    kept = [item for item in graph.initializer if item.name not in drop]
    del graph.initializer[:]
    graph.initializer.extend(kept)


def _by_name(graph):
    return {item.name: item for item in graph.initializer}


def _chunk_nbits(graph, node, chunk_rows, external_data_dir, last_token: bool):
    label = node.name or _input_at(node, 0) or "lm_head"
    if _input_at(node, 4):
        raise StaticShapeError(f"{label} has g_idx, which the vocab split does not slice")
    k = _required_int_attr(node, "K")
    n = _required_int_attr(node, "N")
    bits = _required_int_attr(node, "bits")
    block = _required_int_attr(node, "block_size")
    if None in (k, n, bits, block) or int(bits) != 4 or int(block) < 1 or int(k) % int(block) != 0:
        raise StaticShapeError(f"{label} is not a 4-bit block-aligned MatMulNBits")
    stored = _by_name(graph)
    weight_name = _input_at(node, 1)
    scale_name = _input_at(node, 2)
    weight_item = stored.get(weight_name)
    scale_item = stored.get(scale_name)
    if weight_item is None or scale_item is None:
        raise StaticShapeError(f"{label} weight or scale initializer is missing")
    weight, weight_type = _read_matrix(weight_item, external_data_dir)
    scales, scale_type = _read_matrix(scale_item, external_data_dir)
    if int(weight_type) != int(TensorProto.UINT8):
        raise StaticShapeError(f"{label} packed weight is not uint8")
    zp_name = _input_at(node, 3)
    zp = None
    zp_type = None
    if zp_name:
        zp_item = stored.get(zp_name)
        if zp_item is None:
            raise StaticShapeError(f"{label} zero-point initializer is missing")
        zp, zp_type = _read_matrix(zp_item, external_data_dir)
    bias_name = _input_at(node, 5)
    sizes = _sizes_for(int(n), int(k), chunk_rows, 2)
    act, cast, prelude, prelude_inits, act_shape, did_slice = _prepare_activation(graph, node, last_token)
    output = node.output[0]
    out_elem = _elem_of(graph, output, TensorProto.FLOAT)
    accuracy = _optional_int_attr(node, "accuracy_level")
    parts = []
    nodes = []
    inits = list(prelude_inits)
    for index, rows in enumerate(sizes):
        start = index * int(rows)
        wname = f"{weight_name}.split_{index}"
        sname = f"{scale_name}.split_{index}"
        inputs = [act, wname, sname]
        inits.append(_store(wname, _row_tile(weight, start, int(rows)), weight_type))
        inits.append(_store(sname, _row_tile(scales, start, int(rows)), scale_type))
        if zp is not None:
            zname = f"{zp_name}.split_{index}"
            inputs.append(zname)
            inits.append(_store(zname, _row_tile(zp, start, int(rows)), zp_type))
        attrs = {"K": int(k), "N": int(rows), "bits": int(bits), "block_size": int(block)}
        if accuracy is not None:
            attrs["accuracy_level"] = int(accuracy)
        part = f"{label}/split_{index}/output_0"
        nodes.append(
            helper.make_node(
                "MatMulNBits",
                inputs,
                [part],
                name=f"{label}/split_{index}",
                domain="com.microsoft",
                **attrs,
            )
        )
        parts.append(part)
        _ensure_value_info(graph, part, out_elem, _proj_shape(act_shape, int(rows), did_slice))
    wide, concat_nodes = _concat_or_single(parts, f"{label}/concat", label)
    nodes.extend(concat_nodes)
    padded = int(sum(sizes))
    final = wide
    if padded != int(n):
        final = f"{label}/trimmed"
        slice_node, slice_inits = _vocab_slice(wide, final, int(n), label)
        nodes.append(slice_node)
        inits.extend(slice_inits)
        _ensure_value_info(graph, final, out_elem, _proj_shape(act_shape, int(n), did_slice))
    if bias_name:
        nodes.append(helper.make_node("Add", [final, bias_name], [output], name=f"{label}/Add_bias"))
    elif final != output:
        nodes[-1].output[0] = output
    _ensure_value_info(graph, output, out_elem, _proj_shape(act_shape, int(n), did_slice))
    if did_slice and cast is not None:
        _ensure_value_info(
            graph,
            cast.output[0],
            _elem_of(graph, cast.output[0], out_elem),
            _proj_shape(_shape_of(graph, _input_at(cast, 0)) or act_shape, int(k), True),
        )
    dropped = [weight_name, scale_name]
    if zp_name:
        dropped.append(zp_name)
    report = _report("matmul_nbits", sizes, int(n), did_slice)
    return prelude, cast, nodes, inits, dropped, report


def _fallback_chain(graph, node, chunk_rows, external_data_dir, last_token: bool):
    """Dequantize this node, then cut the float matrix into equal bf16 tiles."""
    label = node.name or "lm_head"
    if _input_at(node, 4):
        raise StaticShapeError(f"{label} has g_idx, which this fallback does not bake")
    k = _required_int_attr(node, "K")
    n = _required_int_attr(node, "N")
    bits = _required_int_attr(node, "bits")
    block = _required_int_attr(node, "block_size")
    if None in (k, n, bits, block) or int(bits) != 4 or int(block) < 1 or int(k) % int(block) != 0:
        raise StaticShapeError(f"{label} is not a 4-bit block-aligned MatMulNBits")
    stored = _by_name(graph)
    weight_name = _input_at(node, 1)
    scale_name = _input_at(node, 2)
    weight = _initializer_array(stored.get(weight_name), external_data_dir) if weight_name in stored else None
    scales = _initializer_array(stored.get(scale_name), external_data_dir) if scale_name in stored else None
    if weight is None or scales is None:
        raise StaticShapeError(f"{label} weight or scale bytes could not be read")
    zp_name = _input_at(node, 3)
    zp = None
    if zp_name:
        zp_item = stored.get(zp_name)
        if zp_item is None:
            raise StaticShapeError(f"{label} zero-point initializer is missing")
        zp = _initializer_array(zp_item, external_data_dir)
        if zp is None:
            raise StaticShapeError(f"{label} zero-point bytes could not be read")
    values = _dequant_matmul_nbits(weight, scales, zp, int(k), int(n), int(block), label)
    bias_name = _input_at(node, 5)
    sizes = _sizes_for(int(n), int(k), chunk_rows, 2)
    act, cast, prelude, prelude_inits, act_shape, did_slice = _prepare_activation(graph, node, last_token)
    output = node.output[0]
    cast_act = f"{label}/act_bf16"
    # A pre-existing Cast stays in front of the baked MatMul. The slice that
    # feeds it is spliced ahead of that Cast. Otherwise the slice is the
    # first new node and this Cast reads it.
    if cast is None:
        nodes = list(prelude)
        returned_prelude = []
    else:
        nodes = []
        returned_prelude = prelude
    inits = list(prelude_inits)
    nodes.append(
        helper.make_node("Cast", [act], [cast_act], name=f"{label}/Cast_act_bf16", to=TensorProto.BFLOAT16)
    )
    _ensure_value_info(graph, cast_act, TensorProto.BFLOAT16, act_shape)
    parts = []
    for index, rows in enumerate(sizes):
        tile = _column_tile(values, index * int(rows), int(rows))
        wname = f"{label}/bf16_weight.split_{index}"
        raw = _f32_to_bf16_raw(tile)
        inits.append(
            helper.make_tensor(wname, TensorProto.BFLOAT16, [int(k), int(rows)], raw, raw=True)
        )
        part = f"{label}/split_{index}/output_0"
        nodes.append(helper.make_node("MatMul", [cast_act, wname], [part], name=f"{label}/split_{index}"))
        parts.append(part)
        _ensure_value_info(graph, part, TensorProto.BFLOAT16, _proj_shape(act_shape, int(rows), did_slice))
    wide, concat_nodes = _concat_or_single(parts, f"{label}/concat", label)
    nodes.extend(concat_nodes)
    padded = int(sum(sizes))
    trimmed = wide
    if padded != int(n):
        trimmed = f"{label}/trimmed"
        slice_node, slice_inits = _vocab_slice(wide, trimmed, int(n), label)
        nodes.append(slice_node)
        inits.extend(slice_inits)
        _ensure_value_info(graph, trimmed, TensorProto.BFLOAT16, _proj_shape(act_shape, int(n), did_slice))
    if bias_name:
        cast_out = f"{label}/cast_f32"
        nodes.append(helper.make_node("Cast", [trimmed], [cast_out], name=f"{label}/Cast_out_f32", to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Add", [cast_out, bias_name], [output], name=f"{label}/Add_bias"))
    else:
        nodes.append(helper.make_node("Cast", [trimmed], [output], name=f"{label}/Cast_out_f32", to=TensorProto.FLOAT))
    _ensure_value_info(graph, output, TensorProto.FLOAT, _proj_shape(act_shape, int(n), did_slice))
    dropped = [weight_name, scale_name]
    if zp_name:
        dropped.append(zp_name)
    return returned_prelude, cast, nodes, inits, dropped, _report("bf16_fallback", sizes, int(n), did_slice)


def _chunk_dense(graph, node, chunk_rows, external_data_dir, last_token: bool):
    label = node.name or "lm_head"
    weight_name = _input_at(node, 1)
    if not weight_name:
        raise StaticShapeError(f"{label} has no weight")
    stored = _by_name(graph)
    item = stored.get(weight_name)
    if item is None:
        raise StaticShapeError(f"{label} weight initializer is missing")
    weight, weight_type = _read_matrix(item, external_data_dir)
    if weight.ndim != 2:
        raise StaticShapeError(f"{label} weight rank is {weight.ndim}")
    output = node.output[0]
    out_shape = _shape_of(graph, output)
    vocab = int(weight.shape[1])
    hidden = int(weight.shape[0])
    if out_shape and isinstance(out_shape[-1], int) and int(out_shape[-1]) != vocab:
        raise StaticShapeError(
            f"{label} weight is [{hidden}, {vocab}] and the output vocab is {out_shape[-1]}"
        )
    act_shape_now = _shape_of(graph, _input_at(node, 0))
    if act_shape_now and isinstance(act_shape_now[-1], int) and int(act_shape_now[-1]) != hidden:
        raise StaticShapeError(
            f"{label} weight K {hidden} does not match activation {act_shape_now[-1]}"
        )
    dtype_bytes = 2 if int(weight_type) in (int(TensorProto.FLOAT16), int(TensorProto.BFLOAT16)) else 4
    sizes = _sizes_for(vocab, hidden, chunk_rows, dtype_bytes)
    act, cast, prelude, inits, act_shape, did_slice = _prepare_activation(graph, node, last_token)
    elem = int(weight_type)
    parts = []
    nodes = []
    for index, rows in enumerate(sizes):
        tile = _column_tile(weight, index * int(rows), int(rows))
        wname = f"{weight_name}.split_{index}"
        inits.append(_store(wname, tile, weight_type))
        part = f"{label}/split_{index}/output_0"
        nodes.append(helper.make_node("MatMul", [act, wname], [part], name=f"{label}/split_{index}"))
        parts.append(part)
        _ensure_value_info(graph, part, elem, _proj_shape(act_shape, int(rows), did_slice))
    wide, concat_nodes = _concat_or_single(parts, f"{label}/concat", label)
    nodes.extend(concat_nodes)
    padded = int(sum(sizes))
    if padded != vocab:
        slice_node, slice_inits = _vocab_slice(wide, output, vocab, label)
        nodes.append(slice_node)
        inits.extend(slice_inits)
    elif wide != output:
        if concat_nodes:
            concat_nodes[-1].output[0] = output
        else:
            nodes[-1].output[0] = output
    out_elem = _elem_of(graph, output, elem)
    _ensure_value_info(graph, output, out_elem, _proj_shape(act_shape, vocab, did_slice))
    return prelude, cast, nodes, inits, [weight_name], _report("dense", sizes, vocab, did_slice)


def _report(mode: str, sizes: Sequence[int], vocab: int, last_token: bool):
    return {
        "mode": mode,
        "chunks": len(sizes),
        "chunk_rows": int(sizes[0]),
        "vocab": int(vocab),
        "padded_vocab": int(sum(sizes)),
        "last_token": bool(last_token),
    }


def rewrite_uniform_lm_head(
    model,
    *,
    chunk_rows: Optional[int] = None,
    npu_bf16_fallback: bool = False,
    external_data_dir: Optional[str] = None,
    last_token: bool = True,
):
    """Chunk the one LM-head projection into equal vocabulary tiles.

    When ``npu_bf16_fallback`` is true and the projection is still
    ``MatMulNBits``, that node is baked before it is sliced. A graph that
    is already a dense MatMul is sliced as ``[K, N]``. The external data
    file is not rewritten.
    """
    graph = model.graph
    kind, node = _find_projection(graph)
    if kind == "nbits" and npu_bf16_fallback:
        prelude, cast, nodes, inits, dropped, report = _fallback_chain(
            graph, node, chunk_rows, external_data_dir, last_token
        )
    elif kind == "nbits":
        prelude, cast, nodes, inits, dropped, report = _chunk_nbits(
            graph, node, chunk_rows, external_data_dir, last_token
        )
    else:
        prelude, cast, nodes, inits, dropped, report = _chunk_dense(
            graph, node, chunk_rows, external_data_dir, last_token
        )
    _splice(graph, node, cast, prelude, nodes)
    _drop_unused(graph, dropped)
    graph.initializer.extend(inits)
    output = model.graph.output
    if output and report["last_token"]:
        _set_dim(output[0], -2, 1)
    if output:
        _set_dim(output[0], -1, report["vocab"])
    return report
