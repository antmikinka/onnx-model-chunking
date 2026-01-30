import onnx
import sys

model_path = r"split_output\kat-dev-v3\model.onnx\layer_0_of_65.onnx"
model = onnx.load(model_path, load_external_data=False)

print(f"Inspecting {model_path}")
print(f"Inputs: {len(model.graph.input)}")

# Find Reshape nodes
for node in model.graph.node:
    if node.op_type == "Reshape":
        print(f"\nNode: {node.name}")
        print(f"  Inputs: {list(node.input)}")
        shape_input_name = node.input[1]
        
        # Check if shape_input is in graph.input
        for vi in model.graph.input:
            if vi.name == shape_input_name:
                print(f"  Shape Input '{shape_input_name}' is a GRAPH INPUT")
                print(f"  ValueInfo: {vi}")
                break
        else:
            # Check if shape_input is an initializer
            for init in model.graph.initializer:
                if init.name == shape_input_name:
                    print(f"  Shape Input '{shape_input_name}' is an INITIALIZER")
                    print(f"  Dims: {init.dims}")
                    break
            else:
                # Check if it's an intermediate output of another node
                print(f"  Shape Input '{shape_input_name}' is an INTERMEDIATE OUTPUT")
