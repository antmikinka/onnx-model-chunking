"""Three-node MatMulNBits reproducer for the Ryzen AI MatMulNBits compile.

The default tests only check the ONNX contract. ``--compile`` runs Vitis.
Synthetic packed weights stay inline UINT8. The Qwen external data file
is not read or written.

The 2026-10-13 bfloat16 control, file base
3ed3332ca96856cb29ca15f046827ad6, failed in superkernel_GemmBf16Int4 at
superkernels.cpp:69. Argument 2 was a bfloat16 buffer. The candidate wants
int8. No operator stat, no NPU line, and no ELF line were printed.

Later compiles of this harness, conda ryzen-ai-1.8.0, ORT_DISABLE_ALL:

- ``fp16``: file base d90b18a0698846c28f03a8d93b070947, Algorithm-A
  070a111eee132db3401f71cb3862f06c, Algorithm-B
  cf985f7266446e9999ce66aec7deb7f4, node count 3. VAIML-COMPILE 1048,
  "No failsafe requirements are not satisfied." Operator stat CPU 3.
  The profile assigned MatMulNBits to CPUExecutionProvider. The run
  returned a finite float16 output. No NPU line and no ELF line.
- ``block128``: file base 9f5cac83b0e527301ecb4f9e824ab078, Algorithm-A
  355f3467875eb6376135ef68a0a90fa5. Peano failed in the same
  superkernel_GemmBf16Int4 weight-slot mismatch. Exit -1. No ELF line.
- ``zp``: file base 835e450996b1d3f07567dfe66f0b3025. Same weight-slot
  mismatch. Exit -1. No ELF line. Its Algorithm-A string matched the
  fp16 probe, and the log still said the cache was missing.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper

K = 4096
N = 6144
BITS = 4
ACCURACY_LEVEL = 4
TEMP = os.environ.get("TEMP", r"C:\Users\antmi\AppData\Local\Temp")


def _packed_shape(block_size: int) -> tuple[int, int, int]:
    if K % block_size != 0 or (block_size * BITS) % 8 != 0:
        raise ValueError(f"block_size {block_size} does not tile K={K} at {BITS} bits")
    n_blocks = K // block_size
    blob = block_size * BITS // 8
    return (N, n_blocks, blob)


def _zero_point_shape(block_size: int) -> tuple[int, int]:
    """ORT QOperator 4-bit zero points: [N, ceil(n_blocks / 2)]."""
    n_blocks = K // block_size
    return (N, (n_blocks + 1) // 2)


def build_probe(
    variant: str = "fp16",
    *,
    block_size: int | None = None,
    activation=None,
    zero_points: bool = False,
):
    """Return a 3-node model. One variant flag differs from the bf16 control."""
    if variant == "bf16":
        block_size = 32 if block_size is None else block_size
        activation = TensorProto.BFLOAT16 if activation is None else activation
    elif variant == "fp16":
        block_size = 32 if block_size is None else block_size
        activation = TensorProto.FLOAT16 if activation is None else activation
    elif variant == "block128":
        block_size = 128 if block_size is None else block_size
        activation = TensorProto.BFLOAT16 if activation is None else activation
    elif variant == "zp":
        block_size = 32 if block_size is None else block_size
        activation = TensorProto.BFLOAT16 if activation is None else activation
        zero_points = True
    else:
        raise ValueError(f"unknown variant {variant}")

    packed_shape = _packed_shape(block_size)
    scale_shape = (N, packed_shape[1])
    weight = helper.make_tensor(
        "qkv_weight",
        TensorProto.UINT8,
        list(packed_shape),
        np.zeros(packed_shape, dtype=np.uint8).tobytes(),
        raw=True,
    )
    scales = helper.make_tensor(
        "qkv_scales",
        TensorProto.FLOAT,
        list(scale_shape),
        np.ones(scale_shape, dtype=np.float32).tobytes(),
        raw=True,
    )
    initializers = [weight, scales]
    scale_out = "qkv_scales/cast"
    matmul_inputs = ["activation/cast", "qkv_weight", scale_out]
    if zero_points:
        zp_shape = _zero_point_shape(block_size)
        initializers.append(
            helper.make_tensor(
                "qkv_zero_points",
                TensorProto.UINT8,
                list(zp_shape),
                np.zeros(zp_shape, dtype=np.uint8).tobytes(),
                raw=True,
            )
        )
        matmul_inputs.append("qkv_zero_points")
    graph = helper.make_graph(
        [
            helper.make_node(
                "Cast",
                ["activation"],
                ["activation/cast"],
                name="Cast_activation",
                to=activation,
            ),
            helper.make_node(
                "Cast",
                ["qkv_scales"],
                [scale_out],
                name="Cast_scales",
                to=activation,
            ),
            helper.make_node(
                "MatMulNBits",
                matmul_inputs,
                ["qkv_out"],
                name="qkv_MatMulNBits",
                domain="com.microsoft",
                K=K,
                N=N,
                bits=BITS,
                block_size=block_size,
                accuracy_level=ACCURACY_LEVEL,
            ),
        ],
        f"matmulnbits_{variant}",
        [helper.make_tensor_value_info("activation", TensorProto.FLOAT, [1, 1, K])],
        [helper.make_tensor_value_info("qkv_out", activation, [1, 1, N])],
        initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 22),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 10
    return model


def contract_summary(model) -> str:
    graph = model.graph
    matmul = next(node for node in graph.node if node.op_type == "MatMulNBits")
    attrs = {attr.name: int(attr.i) for attr in matmul.attribute}
    by_name = {item.name: item for item in graph.initializer}
    weight = by_name[matmul.input[1]]
    casts = [
        (node.name, int(node.attribute[0].i), node.input[0])
        for node in graph.node
        if node.op_type == "Cast"
    ]
    zp = by_name.get(matmul.input[3]) if len(matmul.input) > 3 and matmul.input[3] else None
    zp_text = "none" if zp is None else f"UINT8 {list(zp.dims)}"
    return (
        f"nodes={len(graph.node)} K={attrs.get('K')} N={attrs.get('N')} "
        f"bits={attrs.get('bits')} block={attrs.get('block_size')} "
        f"acc={attrs.get('accuracy_level')} "
        f"weight=UINT8 {list(weight.dims)} location={int(weight.data_location)} "
        f"casts={casts} zp={zp_text}"
    )


class TestMatMulNBitsProbe(unittest.TestCase):
    def test_fp16_keeps_uint8_weights(self):
        model = build_probe("fp16")
        matmul = next(node for node in model.graph.node if node.op_type == "MatMulNBits")
        weight = next(item for item in model.graph.initializer if item.name == matmul.input[1])
        casts = [node for node in model.graph.node if node.op_type == "Cast"]
        self.assertEqual(weight.data_type, TensorProto.UINT8)
        self.assertEqual(list(weight.dims), [6144, 128, 16])
        self.assertEqual(len(matmul.input), 3)
        self.assertEqual([int(node.attribute[0].i) for node in casts], [TensorProto.FLOAT16, TensorProto.FLOAT16])
        self.assertEqual(model.graph.output[0].type.tensor_type.elem_type, TensorProto.FLOAT16)

    def test_block128_layout(self):
        model = build_probe("block128")
        weight = next(item for item in model.graph.initializer if item.name == "qkv_weight")
        scales = next(item for item in model.graph.initializer if item.name == "qkv_scales")
        self.assertEqual(list(weight.dims), [6144, 32, 64])
        self.assertEqual(list(scales.dims), [6144, 32])
        self.assertEqual(weight.data_type, TensorProto.UINT8)

    def test_zero_point_packs_two_per_byte(self):
        model = build_probe("zp")
        zp = next(item for item in model.graph.initializer if item.name == "qkv_zero_points")
        self.assertEqual(zp.data_type, TensorProto.UINT8)
        self.assertEqual(list(zp.dims), [6144, 64])


def compile_probe(variant: str) -> str:
    import onnxruntime as ort

    path = os.path.join(TEMP, f"pr9_matmulnbits_{variant}.onnx")
    model = build_probe(variant)
    onnx.save(model, path)
    print("CONTRACT", contract_summary(model), flush=True)
    print("SAVED", path, os.path.getsize(path), flush=True)
    os.chdir(TEMP)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 1
    options.enable_profiling = True
    options.profile_file_prefix = os.path.join(TEMP, f"pr9_matmulnbits_{variant}")
    try:
        session = ort.InferenceSession(
            path,
            sess_options=options,
            providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
        )
    except Exception:
        print("PROBE_FAIL", flush=True)
        print(traceback.format_exc()[-2000:], flush=True)
        return path
    print("PROVIDERS", session.get_providers(), flush=True)
    try:
        outputs = session.run(None, {"activation": np.zeros((1, 1, K), dtype=np.float32)})
        array = np.asarray(outputs[0])
        print(
            f"RUN shape={list(array.shape)} dtype={array.dtype} finite={bool(np.isfinite(array).all())}",
            flush=True,
        )
    except Exception:
        print("RUN_FAIL", flush=True)
        print(traceback.format_exc()[-2000:], flush=True)
    print("PROFILE", session.end_profiling(), flush=True)
    print("PROBE_DONE", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--variant", default="fp16", choices=("bf16", "fp16", "block128", "zp"))
    args = parser.parse_args()
    if not args.compile:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestMatMulNBitsProbe)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    path = ""
    try:
        path = compile_probe(args.variant)
    finally:
        if path and os.path.basename(path) == f"pr9_matmulnbits_{args.variant}.onnx" and os.path.exists(path):
            os.remove(path)
            print("PROBE_DELETED", flush=True)


if __name__ == "__main__":
    main()
