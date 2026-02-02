
import os
import json
import numpy as np
import onnxruntime as ort
from grok_inference_split_model_pipeline import SequentialModelPipeline

def debug_inference():
    model_dir = "output-final/model_jit.onnx"
    pipeline = SequentialModelPipeline(model_dir, providers=["CPUExecutionProvider"], verbose=True)
    
    prompt = "Hello"
    print(f"\n--- DEBUG START: {prompt} ---")
    
    # Manually run parts of generate to see what's happening
    inputs = pipeline.tokenizer(prompt, return_tensors="np")
    prompt_ids = inputs["input_ids"]
    pipeline.tensor_registry["input_ids"] = prompt_ids
    pipeline.tensor_registry["attention_mask"] = np.ones(prompt_ids.shape, dtype=np.int64)
    
    print("Registry keys before start:", list(pipeline.tensor_registry.keys()))
    
    components_to_run = ["embedding", "other"]
    for i in range(36):
        components_to_run.append(f"layer_{i}")
    components_to_run.append("lm_head")
    
    for comp in components_to_run:
        print(f"\n> Running {comp}...")
        try:
            pipeline._run_component(comp)
            print(f"  Success. Registry keys: {[k for k in pipeline.tensor_registry.keys() if '/' not in k or comp in k or 'logits' in k]}")
        except Exception as e:
            print(f"  FAILED {comp}: {e}")
            import traceback
            traceback.print_exc()
            break
            
    if "logits" in pipeline.tensor_registry:
        print("\nSUCCESS: 'logits' found in registry!")
        logits = pipeline.tensor_registry["logits"]
        print(f"Logits shape: {logits.shape}")
    else:
        print("\nFAILURE: 'logits' NOT found in registry.")

if __name__ == "__main__":
    debug_inference()
