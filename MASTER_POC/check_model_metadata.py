#!/usr/bin/env python3
"""Check what's in the original model that needs to be preserved."""
import onnx

print("Loading original model...")
m = onnx.load("C:/Users/antmi/Downloads/Qwen3-8B-awq-quant-onnx-ryzenai-1.7-hybrid/model_jit.onnx")

print(f"\nModel IR Version: {m.ir_version}")
print(f"Producer: {m.producer_name} v{m.producer_version}")

print(f"\nOpsets ({len(m.opset_import)}):")
for op in m.opset_import:
    print(f"  domain='{op.domain}' version={op.version}")

print(f"\nFunctions: {len(m.functions) if m.functions else 0}")
for f in (m.functions or []):
    print(f"  {f.domain}:{f.name}")

print(f"\nMetadata props ({len(m.metadata_props)}):")
for p in m.metadata_props:
    print(f"  {p.key}: {p.value[:50]}...")

# Check split model
print("\n" + "="*60)
print("Loading split layer 0...")
s = onnx.load("output-final/model_jit.onnx/layer_0_of_36.onnx")

print(f"\nSplit IR Version: {s.ir_version}")
print(f"Split Producer: {s.producer_name} v{s.producer_version}")

print(f"\nSplit Opsets ({len(s.opset_import)}):")
for op in s.opset_import:
    print(f"  domain='{op.domain}' version={op.version}")

print(f"\nSplit Functions: {len(s.functions) if s.functions else 0}")
