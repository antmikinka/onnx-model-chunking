
import onnx
import sys
import os

model_path = 'output-final/model_jit.onnx/lm_head.onnx'
if not os.path.exists(model_path):
    print(f"Error: {model_path} does not exist")
    sys.exit(1)

model = onnx.load(model_path, load_external_data=False)
print(f"File: {model_path}")
print(f"Inputs: {[i.name for i in model.graph.input]}")
print(f"Outputs: {[o.name for o in model.graph.output]}")
for node in model.graph.node:
    print(f"  Node: {node.name} (Op: {node.op_type})")
    print(f"    Ins: {list(node.input)}")
    print(f"    Outs: {list(node.output)}")
