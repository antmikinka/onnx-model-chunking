#!/usr/bin/env python3
"""Inspect MatMulNBits nodes in split vs original model."""
import onnx

def inspect_matmul(model_path, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    
    model = onnx.load(model_path)
    
    # Find MatMulNBits nodes
    matmul_count = 0
    for node in model.graph.node:
        if 'MatMul' in node.op_type:
            matmul_count += 1
            if matmul_count <= 2:  # Show first 2
                print(f"\nNode: {node.name}")
                print(f"  OpType: {node.op_type}")
                print(f"  Domain: '{node.domain}'")
                for attr in node.attribute:
                    val = attr.i if attr.HasField('i') else list(attr.ints) if attr.ints else attr.s
                    print(f"  Attr '{attr.name}': {val}")
    
    print(f"\nTotal MatMul-like nodes: {matmul_count}")

# Inspect split component
inspect_matmul('output-final/model_jit.onnx/layer_0_of_36.onnx', 'SPLIT LAYER 0')

# Inspect original
inspect_matmul('C:/Users/antmi/Downloads/Qwen3-8B-awq-quant-onnx-ryzenai-1.7-hybrid/model_jit.onnx', 'ORIGINAL MODEL')
