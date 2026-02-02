import onnx

def check_domain(path, label):
    print(f"\n--- {label} ---")
    m = onnx.load(path)
    found_sln = False
    found_matmul = False
    
    for n in m.graph.node:
        if 'SimplifiedLayerNormalization' in n.op_type and not found_sln:
            print(f"Node: {n.name}")
            print(f"  OpType: {n.op_type}")
            print(f"  Domain: '{n.domain}'")
            found_sln = True
            
        if 'MatMulNBits' in n.op_type and not found_matmul:
            print(f"Node: {n.name}")
            print(f"  OpType: {n.op_type}")
            print(f"  Domain: '{n.domain}'")
            found_matmul = True
            
        if found_sln and found_matmul:
            break

check_domain('output-final/model_jit.onnx/layer_0_of_36.onnx', 'SPLIT')
check_domain('C:/Users/antmi/Downloads/Qwen3-8B-awq-quant-onnx-ryzenai-1.7-hybrid/model_jit.onnx', 'ORIGINAL')
