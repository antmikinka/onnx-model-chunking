import onnx
import sys

model_path = r"C:\Users\antmi\Downloads\olive-recipes-main\olive-recipes-main\Qwen-Qwen2.5-Coder-7B-Instruct\VitisAI\kat-dev-bf16-onnx\model.onnx"
model = onnx.load(model_path, load_external_data=False)

# Run shape inference
print("Running shape inference...")
model = onnx.shape_inference.infer_shapes(model)

target_names = [
    "/model/constants/TensorProto.INT64/1D/0, -1, 128",
    "/model/constants/TensorProto.INT64/1D/0, -1, 8192",
    "/model/constants/TensorProto.INT64/1D/0, 0, 8, -1"
]

for name in target_names:
    print(f"\nTensor: {name}")
    # Check graph.value_info
    found = False
    for vi in model.graph.value_info:
        if vi.name == name:
            print(f"  Found in value_info: {vi}")
            found = True
            break
            
    if not found:
        # Check initializers
        for init in model.graph.initializer:
            if init.name == name:
                print(f"  Found in initializers: Dims {init.dims}")
                found = True
                break
                
    if not found:
        print("  NOT FOUND in master graph")
