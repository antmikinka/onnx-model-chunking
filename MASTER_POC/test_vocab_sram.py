"""SRAM vocabulary splits: row planner, masked gather, concatenated LM head."""

import ast
import importlib.util
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def _planner():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_utils", "vocab_sram.py")
    spec = importlib.util.spec_from_file_location("vocab_sram", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLANNER = _planner()
BUDGET = int(16 * 1024 * 1024 * 0.5)


def _check(name, fn):
    try:
        fn()
    except Exception as exc:
        print(f"  {name}: FAIL {exc}")
        return False
    print(f"  {name}: PASS")
    return True


def test_gpt2_rows():
    sizes = PLANNER.vocab_chunk_rows(50257, 768, 2, tile_sram_mb=16, headroom=0.5)
    assert sizes == [5440] * 9 + [1297], sizes
    assert sum(sizes) == 50257
    assert max(sizes) < 14000
    for rows in sizes:
        assert rows * 768 * 2 <= BUDGET


def test_small_table_stays_whole():
    assert PLANNER.vocab_chunk_rows(128, 64, 2) == [128]
    assert PLANNER.vocab_chunk_rows(100, 32, 4, tile_sram_mb=16, headroom=0.5) == [100]


def test_split_rows_override():
    forced = PLANNER.vocab_chunk_rows(50257, 768, 2, split_rows=14000)
    assert forced == [14000, 14000, 14000, 8257]
    assert PLANNER.vocab_chunk_rows(50257, 768, 2, split_rows=0) == [50257]
    assert PLANNER.vocab_chunk_rows(50257, 768, 2, split_rows=-1) == [50257]
    assert PLANNER.vocab_chunk_rows(100, 8, 2, split_rows=100) == [100]
    try:
        PLANNER.vocab_chunk_rows(32, 8, 2, headroom=0)
    except ValueError:
        return
    raise AssertionError("headroom 0 was accepted")


def test_numpy_embedding_matches_lookup():
    rng = np.random.default_rng(0)
    table = rng.standard_normal((10, 4)).astype(np.float32)
    sizes = [4, 4, 2]
    ids = np.array([[0, 3, 4, 9, -1], [8, 7, 10, 1, 5]], dtype=np.int64)
    got = PLANNER.apply_split_embedding(table, ids, sizes)
    expected = np.zeros(ids.shape + (4,), dtype=np.float32)
    valid = (ids >= 0) & (ids < table.shape[0])
    expected[valid] = table[ids[valid]]
    np.testing.assert_allclose(got, expected)
    assert np.all(got[0, 4] == 0)
    assert np.all(got[1, 2] == 0)
    np.testing.assert_allclose(got[0, 2], table[4])


def test_numpy_lm_head_matches_matmul():
    rng = np.random.default_rng(1)
    weight = rng.standard_normal((10, 4)).astype(np.float32)
    hidden = rng.standard_normal((2, 3, 4)).astype(np.float32)
    sizes = [4, 6]
    got = PLANNER.apply_split_lm_head(weight, hidden, sizes)
    np.testing.assert_allclose(got, hidden @ weight.T, rtol=1e-5, atol=1e-5)

    weight16 = weight.astype(np.float16)
    hidden16 = hidden.astype(np.float16)
    got16 = PLANNER.apply_split_lm_head(weight16, hidden16, sizes)
    np.testing.assert_array_equal(got16, hidden16 @ np.swapaxes(weight16, -1, -2))


def _const(nodes, made, value):
    if isinstance(value, (list, tuple)):
        literal = "[" + ",".join(str(int(item)) for item in value) + "]"
        array = np.asarray(list(value), dtype=np.int64)
    else:
        literal = str(int(value))
        array = np.asarray(int(value), dtype=np.int64)
    name = f"/model/constants/INT64/{literal}"
    if name not in made:
        made.add(name)
        nodes.append(
            helper.make_node(
                "Constant",
                [],
                [name],
                name=name.replace("constants", "constant_nodes"),
                value=numpy_helper.from_array(array),
            )
        )
    return name


def _model(nodes, inputs, outputs, initializers):
    graph = helper.make_graph(nodes, "vocab_sram", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    return model


def _run(model, feeds):
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(None, feeds)


def test_onnx_embedding_matches_numpy():
    rng = np.random.default_rng(2)
    table = rng.standard_normal((10, 4)).astype(np.float32)
    sizes = [4, 4, 2]
    ids = np.array([[0, 3, 4, 9, -1], [8, 7, 10, 1, 5]], dtype=np.int64)
    nodes = []
    made = set()
    initializers = []
    running = None
    start = 0
    for index, rows in enumerate(sizes):
        weight_name = f"model.embed_tokens.weight.split_{index}"
        initializers.append(numpy_helper.from_array(np.ascontiguousarray(table[start:start + rows]), weight_name))
        prefix = f"/model/embed_tokens/Gather/split_{index}"
        nodes.append(helper.make_node("Sub", ["input_ids", _const(nodes, made, start)], [f"{prefix}/Sub/output_0"], name=f"{prefix}/Sub"))
        nodes.append(
            helper.make_node(
                "Clip",
                [f"{prefix}/Sub/output_0", _const(nodes, made, 0), _const(nodes, made, rows - 1)],
                [f"{prefix}/Clip/output_0"],
                name=f"{prefix}/Clip",
            )
        )
        nodes.append(
            helper.make_node(
                "Gather",
                [weight_name, f"{prefix}/Clip/output_0"],
                [f"{prefix}/Gather/output_0"],
                name=f"{prefix}/Gather",
                axis=0,
            )
        )
        nodes.append(
            helper.make_node(
                "GreaterOrEqual",
                ["input_ids", _const(nodes, made, start)],
                [f"{prefix}/GreaterOrEqual/output_0"],
                name=f"{prefix}/GreaterOrEqual",
            )
        )
        nodes.append(
            helper.make_node(
                "Less",
                ["input_ids", _const(nodes, made, start + rows)],
                [f"{prefix}/Less/output_0"],
                name=f"{prefix}/Less",
            )
        )
        nodes.append(
            helper.make_node(
                "And",
                [f"{prefix}/GreaterOrEqual/output_0", f"{prefix}/Less/output_0"],
                [f"{prefix}/And/output_0"],
                name=f"{prefix}/And",
            )
        )
        nodes.append(
            helper.make_node(
                "Cast",
                [f"{prefix}/And/output_0"],
                [f"{prefix}/Cast/output_0"],
                name=f"{prefix}/Cast",
                to=TensorProto.FLOAT,
            )
        )
        nodes.append(
            helper.make_node(
                "Unsqueeze",
                [f"{prefix}/Cast/output_0", _const(nodes, made, [-1])],
                [f"{prefix}/Unsqueeze/output_0"],
                name=f"{prefix}/Unsqueeze",
            )
        )
        nodes.append(
            helper.make_node(
                "Mul",
                [f"{prefix}/Gather/output_0", f"{prefix}/Unsqueeze/output_0"],
                [f"{prefix}/Mul/output_0"],
                name=f"{prefix}/Mul",
            )
        )
        term = f"{prefix}/Mul/output_0"
        if running is None:
            running = term
        else:
            add_name = "/model/embed_tokens/Gather" if index == len(sizes) - 1 else f"{prefix}/Add"
            nodes.append(helper.make_node("Add", [running, term], [f"{add_name}/output_0"], name=add_name))
            running = f"{add_name}/output_0"
        start += rows

    output_name = "/model/embed_tokens/Gather/output_0"
    assert running == output_name
    assert not any(node.op_type == "If" for node in nodes)
    model = _model(
        nodes,
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [2, 5])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [2, 5, 4])],
        initializers,
    )
    (got,) = _run(model, {"input_ids": ids})
    np.testing.assert_allclose(got, PLANNER.apply_split_embedding(table, ids, sizes))


def test_onnx_lm_head_matches_numpy():
    rng = np.random.default_rng(3)
    weight = rng.standard_normal((10, 4)).astype(np.float32)
    hidden = rng.standard_normal((2, 3, 4)).astype(np.float32)
    sizes = [4, 6]
    nodes = []
    initializers = []
    parts = []
    start = 0
    for index, rows in enumerate(sizes):
        chunk = np.ascontiguousarray(np.swapaxes(weight[start:start + rows], 0, 1))
        weight_name = f"lm_head.MatMul.weight.split_{index}"
        initializers.append(numpy_helper.from_array(chunk, weight_name))
        part = f"/lm_head/MatMul/split_{index}/output_0"
        nodes.append(
            helper.make_node(
                "MatMul",
                ["hidden_states", weight_name],
                [part],
                name=f"/lm_head/MatMul/split_{index}",
            )
        )
        parts.append(part)
        start += rows
    output_name = "/lm_head/MatMul/output_0"
    nodes.append(helper.make_node("Concat", parts, [output_name], name="/lm_head/MatMul", axis=-1))
    assert not any(node.op_type == "If" for node in nodes)
    model = _model(
        nodes,
        [helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [2, 3, 10])],
        initializers,
    )
    (got,) = _run(model, {"hidden_states": hidden})
    np.testing.assert_allclose(got, PLANNER.apply_split_lm_head(weight, hidden, sizes))
    logits = helper.make_node("Concat", parts, ["logits"], name="/lm_head/MatMul", axis=-1)
    assert logits.output == ["logits"]


def test_builder_wires_the_split():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "builder.py")
    source = open(path, encoding="utf-8").read()
    tree = ast.parse(source)
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model")
    names = {node.name for node in model.body if isinstance(node, ast.FunctionDef)}
    required = {
        "_io_dtype_bytes",
        "_vocab_chunk_rows_for",
        "_row_slice",
        "_make_split_embedding",
        "_should_split_vocab_projection",
        "_make_split_vocab_matmul",
    }
    missing = required - names
    assert not missing, missing
    assert "if self._should_split_vocab_projection(matmul, name):" in source
    assert "gather_output = self._make_split_embedding(basename, embedding, sizes)" in source
    assert "GatherBlockQuantized" in source
    assert '"split_vocab"' in source
    embedding = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "_make_split_embedding")
    ops = [node.func.attr for node in ast.walk(embedding) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    for op in ("make_sub", "make_clip", "make_gather", "make_greater_or_equal", "make_less", "make_and", "make_cast", "make_unsqueeze", "make_mul", "make_add"):
        assert op in ops, op
    assert "If" not in ops
    head = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "_make_split_vocab_matmul")
    head_src = ast.get_source_segment(source, head)
    assert 'kwargs.get("logits", False)' in head_src
    assert 'axis=-1' in head_src
    assert "qweight" in ast.get_source_segment(
        source,
        next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "_should_split_vocab_projection"),
    )


def main():
    checks = [
        ("gpt2_rows", test_gpt2_rows),
        ("small_table", test_small_table_stays_whole),
        ("split_rows_override", test_split_rows_override),
        ("numpy_embedding", test_numpy_embedding_matches_lookup),
        ("numpy_lm_head", test_numpy_lm_head_matches_matmul),
        ("onnx_embedding", test_onnx_embedding_matches_numpy),
        ("onnx_lm_head", test_onnx_lm_head_matches_numpy),
        ("builder_wires_split", test_builder_wires_the_split),
    ]
    return all(_check(name, fn) for name, fn in checks)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
