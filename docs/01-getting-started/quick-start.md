# Quick Start Guide

**Outcome Statement:** In 5 minutes, you will split a model and run inference using the memory-optimized sequential mode.

---

## Prerequisites Check

Before starting, verify:

```bash
python --version  # Should be 3.8+
pip show onnx onnxruntime  # Should show installed packages
```

---

## Step 1: Obtain a Model

**Option A: Download from HuggingFace**

```bash
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
model.save_pretrained("./qwen2.5-0.5b", safe_serialization=True)
```

**Option B: Use Existing ONNX Model**

```bash
# If you already have an ONNX model, note its path
MODEL_PATH="C:/models/qwen2.5-0.5b/model.onnx"
```

---

## Step 2: Split the Model (2 minutes)

```bash
cd MASTER_POC

python grok_onnx_split.py \
    --model "$MODEL_PATH" \
    --output "split_output/qwen2.5-0.5b" \
    --split-only
```

**What happens:**
1. Architecture is auto-detected (Qwen2 in this case)
2. Model is decomposed into 31 components:
   - `embeddings.onnx` (519 MB)
   - `layer_0.onnx` through `layer_27.onnx` (16 MB each)
   - `norm.onnx` (<1 MB)
   - `lm_head.onnx` (70 MB)

**Expected output:**
```
[INFO] Architecture detected: Qwen2 (28 layers)
[INFO] Split complete: 31 components created
[INFO] Peak memory reduction: 780MB -> 519MB (33% savings)
```

---

## Step 3: Run Inference (2 minutes)

```bash
python grok_inference_split_model_pipeline.py \
    --model-dir "split_output/qwen2.5-0.5b" \
    --prompt "The future of artificial intelligence" \
    --tokens 30
```

**Expected output:**
```
[INFO] Loading model components...
[INFO] Processing prompt...
[INFO] Generated text:
"The future of artificial intelligence lies in creating systems that can
augment human capabilities rather than replace them entirely..."
```

---

## Step 4: Verify Memory Savings

```python
import psutil
import os

def get_directory_size(path):
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024 * 1024)  # MB

# Original model size
original_size = get_directory_size("C:/models/qwen2.5-0.5b")
print(f"Original model: {original_size:.1f} MB")

# Split components total
split_size = get_directory_size("split_output/qwen2.5-0.5b")
print(f"Split components: {split_size:.1f} MB")

# Peak memory during inference
process = psutil.Process()
peak_mem = process.memory_info().rss / (1024 * 1024)
print(f"Peak memory (sequential mode): {peak_mem:.1f} MB")
```

---

## What You Accomplished

| Metric | Monolithic | Split (Sequential) | Savings |
|--------|------------|-------------------|---------|
| Peak RAM | ~1050 MB | ~519 MB | 51% reduction |
| Load Time | Full load | Component load | Distributed |
| Hardware Required | High-end GPU | CPU-only capable | Democratized |

---

## Next Steps

1. **Try different execution modes:** See [Inference Pipelines](../04-inference-pipelines/README.md)
2. **Optimize for your hardware:** See [Hardware Deployment](../05-hardware-deployment/README.md)
3. **Understand the architecture:** See [Architecture Overview](../02-architecture-overview/README.md)

---

## Troubleshooting Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| Model not found | Use absolute path, not relative |
| Out of memory during split | Close other applications, use `--split-only` |
| Slow inference | Try `--persist` flag for persistent mode |
| Provider error | Verify ORT installation with `python -c "import onnxruntime; print(onnxruntime.get_available_providers())"` |

---

*For comprehensive usage, see [Full Getting Started](./README.md).*
