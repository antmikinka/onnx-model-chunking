import onnx
import onnx.helper as helper
import os

# Original model path
ORIG_MODEL = 'C:/Users/antmi/Downloads/Qwen3-8B-awq-quant-onnx-ryzenai-1.7-hybrid/model_jit.onnx'
ORIG_BIN = 'model_jit.pb.bin' # Relative name in original
ABS_BIN = 'C:/Users/antmi/Downloads/Qwen3-8B-awq-quant-onnx-ryzenai-1.7-hybrid/model_jit.pb.bin'

print("Loading original model structure...")
m = onnx.load(ORIG_MODEL, load_external_data=False)

# Extract Layer 0 nodes and initializers
print("Extracting Layer 0...")
layer_nodes = []
layer_inits = []
layer_inputs = []
layer_outputs = []

# For simplicity, let's just grab the first MatMulNBits and its inputs
target_node = None
for n in m.graph.node:
    if 'MatMulNBits' in n.op_type:
        target_node = n
        break

print(f"Target Node: {target_node.name}")

# Find inputs
needed_inputs = set(target_node.input)
for init in m.graph.initializer:
    if init.name in needed_inputs:
        # CRITICAL: Modify this initializer to point to ABSOLUTE PATH
        new_init = onnx.TensorProto()
        new_init.CopyFrom(init)
        
        # If it was external, point it to the absolute path
        if new_init.data_location == 1:
            # Clear existing metadata
            del new_init.external_data[:]
            # Add new location
            loc = new_init.external_data.add()
            loc.key = 'location'
            loc.value = ABS_BIN
            
            # Preserve offset/length/checksum if they exist
            # (assuming they are correct for the original file)
            for d in init.external_data:
                if d.key in ['offset', 'length', 'checksum']:
                    new_loc = new_init.external_data.add()
                    new_loc.key = d.key
                    new_loc.value = d.value
        
        layer_inits.append(new_init)

# Create a graph with just this node
# We realize we need inputs/outputs definitions
# For this verify test, we will just create a dummy graph with this node
# But inputs that represent activations need ValueInfos
input_vis = []
output_vis = []

# Mock inputs that are NOT initializers
for inp in target_node.input:
    if inp not in [i.name for i in layer_inits]:
        # Make a dummy float32 value info
        vi = helper.make_tensor_value_info(inp, onnx.TensorProto.FLOAT16, [1, 1, 4096])
        input_vis.append(vi)

# Mock output
for out in target_node.output:
    vi = helper.make_tensor_value_info(out, onnx.TensorProto.FLOAT16, [1, 1, 4096])
    output_vis.append(vi)

graph = helper.make_graph(
    [target_node],
    "test_link_graph",
    input_vis,
    output_vis,
    layer_inits
)

model = helper.make_model(
    graph,
    producer_name=m.producer_name,
    producer_version=m.producer_version,
    opset_imports=m.opset_import,
    functions=m.functions,
    doc_string=m.doc_string
)

if hasattr(m, "metadata_props"):
    model.metadata_props.extend(m.metadata_props)

output_path = "test_vitis_link.onnx"
print(f"Saving to {output_path}...")
onnx.save(model, output_path)

print("Done. Now try loading with VitisAI EP.")
