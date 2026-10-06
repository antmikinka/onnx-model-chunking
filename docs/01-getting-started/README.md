# Getting Started with ONNX Model Chunking

**Outcome Statement:** After completing this section, you will have the system installed, understand the core concepts, and be able to split and run inference on a model within 5 minutes.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start (5 Minutes)](#quick-start-5-minutes)
3. [Core Concepts](#core-concepts)
4. [Next Steps](#next-steps)

---

## Installation

### Prerequisites

**Required:**
- Python 3.8 or higher
- pip package manager
- 10GB free disk space for model files

**Recommended:**
- CUDA-capable GPU with 8GB+ VRAM (for GPU acceleration)
- AMD Ryzen AI NPU (for NPU acceleration)
- 16GB+ system RAM

### Step 1: Install Core Dependencies

```bash
pip install onnx onnxruntime numpy psutil
```

### Step 2: Install Optional Acceleration

**For GPU Acceleration (CUDA):**
```bash
pip install onnxruntime-gpu
# Or for specific CUDA version:
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

**For AMD NPU Acceleration:**
```bash
pip install onnxruntime-vitisai
# Standard onnxruntime includes VitisAI on supported hardware
```

**For HuggingFace Integration:**
```bash
pip install transformers safetensors huggingface_hub
```

### Step 3: Verify Installation

```python
import onnx
import onnxruntime as ort

print(f"ONNX version: {onnx.__version__}")
print(f"ORT version: {ort.__version__}")
print(f"Available providers: {ort.get_available_providers()}")
```

Expected output:
```
ONNX version: 1.15.0
ORT version: 1.17.0
Available providers: ['CPUExecutionProvider', 'CUDAExecutionProvider']
```

---

## Quick Start (5 Minutes)

### Step 1: Split a Model

**Goal:** Decompose a monolithic ONNX model into semantic components.

```bash
cd MASTER_POC

python grok_onnx_split.py \
    --model "path/to/model.onnx" \
    --output "split_output/my_model" \
    --split-only
```

**Expected Output:**
```
[INFO] Detecting architecture...
[INFO] Found: Qwen2 architecture (28 layers)
[INFO] Extracting components...
[INFO] Created: embeddings.onnx (519 MB)
[INFO] Created: layer_0.onnx (16 MB)
...
[INFO] Created: layer_27.onnx (16 MB)
[INFO] Created: norm.onnx (<1 MB)
[INFO] Created: lm_head.onnx (70 MB)
[INFO] Split complete: split_output/my_model/
```

### Step 2: Link External Data (If Required)

If your model uses external weight files:

```powershell
# Windows PowerShell
New-Item -ItemType HardLink `
    -Path "split_output\my_model\model.onnx.data" `
    -Target "path\to\original\model.onnx.data"
```

```bash
# Linux/Mac
ln -s /path/to/original/model.onnx.data split_output/my_model/model.onnx.data
```

### Step 3: Run Inference

```bash
python grok_inference_split_model_pipeline.py \
    --model-dir "split_output/my_model" \
    --prompt "Explain machine learning" \
    --tokens 50
```

**Expected Output:**
```
[INFO] Loading model from: split_output/my_model
[INFO] Found 31 components
[INFO] Using providers: ['CPUExecutionProvider']
[INFO] Processing prompt: Explain machine learning
[INFO] Generated: Explain machine learning in simple terms.
Machine learning is a subset of artificial intelligence...
```

### Step 4: Python API Usage

```python
from grok_onnx_split import split_onnx_model
from grok_inference_split_model_pipeline import SequentialModelPipeline
import argparse

# Split the model
params = argparse.Namespace(
    model="path/to/model.onnx",
    output="split_output/my_model",
    split_only=True
)
result = split_onnx_model(params)
print(f"Split into {result.num_layers} layers")

# Run inference (memory-optimized mode)
pipeline = SequentialModelPipeline(
    model_path="split_output/my_model",
    providers=["CPUExecutionProvider"],
    persist_sessions=False  # Sequential loading = minimum memory
)

output = pipeline.generate("What is AI?", max_new_tokens=50)
print(output)

pipeline.cleanup()
```

---

## Core Concepts

### Concept 1: Why Splitting Works

**Outcome:** Understand the mathematical foundation of semantic splitting.

A neural network is a **chain of function compositions**:

```
output = LM_Head(Norm(Layer_N(...Layer_1(Layer_0(Embedding(input)))...)))
```

**Key Insight:** You don't need all functions loaded simultaneously. You need:
1. Current function in memory
2. Input from previous function
3. Output buffer for next function

This reduces peak memory from "entire model" to "largest single component."

### Concept 2: Semantic vs Arbitrary Splitting

| Approach | Description | Memory Efficiency |
|----------|-------------|-------------------|
| **Semantic Split** | Split at architectural boundaries (per layer) | Peak = max(component) |
| **Arbitrary Split** | Split at any graph node | Variable, may break optimizations |
| **Bisection** | Binary split at weight midpoint | ~50% reduction |

**This system uses semantic splitting** - each transformer layer becomes a standalone ONNX model.

### Concept 3: Execution Modes

| Mode | Memory | Speed | Best For |
|------|--------|-------|----------|
| **Sequential** | Minimum (1 component at a time) | Slowest (disk I/O) | Memory-constrained CPU |
| **Persistent** | All components loaded | Fast (no disk I/O) | GPU with sufficient VRAM |
| **IO-Bound** | All + device buffers | Fastest (zero-copy) | NPU/GPU production |

### Concept 4: Memory Savings Calculation

**Example: Qwen2.5-0.5B (INT4 quantized)**

| Component | Size | Cumulative (Monolithic) | Peak (Sequential) |
|-----------|------|------------------------|-------------------|
| Embeddings | 519 MB | 519 MB | 519 MB |
| Layer 0 | 16 MB | 535 MB | 16 MB |
| Layer 1 | 16 MB | 551 MB | 16 MB |
| ... (26 more layers) | 16 MB each | ... | 16 MB |
| Norm | <1 MB | ~980 MB | <1 MB |
| LM Head | 70 MB | ~1050 MB | 70 MB |

**Result:** Monolithic requires ~1GB+ loaded. Sequential requires ~519MB peak (embeddings is largest).

---

## Configuration Basics

### Basic Configuration

```python
extra_options = {
    "enable_chunking": True,
    "chunk_output_dir": "./split_output",
    "verify_chunks": True,
    "min_psnr_threshold": 35.0
}
```

### Unified Configuration (Recommended)

```python
extra_options = {
    "chunking": {
        "method": "onnx_graph_level",
        "onnx_graph_level": {
            "strategy": "weight_based",
            "distribution": "equal_memory",
            "num_chunks": 2
        },
        "common": {
            "output_directory": "./split_output",
            "verify_chunks": True,
            "min_psnr_threshold": 40.0
        }
    }
}
```

---

## Verification

### Verify Split Correctness

**Goal:** Ensure split models produce identical outputs to the original.

```python
from builder import Model
import numpy as np

# Create model with verification enabled
model = Model(config, "float16", "float16", "cpu", "./cache", {
    "enable_chunking": True,
    "verify_chunks": True,
    "min_psnr_threshold": 35.0
})

model.make_model("path/to/model.onnx")

# PSNR > 35dB indicates good accuracy
# PSNR > 40dB indicates production-ready accuracy
```

### Interpret PSNR Results

| PSNR Value | Quality | Action |
|------------|---------|--------|
| > 40 dB | Production-ready | Deploy with confidence |
| 35-40 dB | Good accuracy | Acceptable for most use cases |
| 30-35 dB | Moderate degradation | Review split configuration |
| < 30 dB | Poor accuracy | Re-verify split points |

---

## Next Steps

| Topic | Location | Purpose |
|-------|----------|---------|
| Architecture Overview | [02-architecture-overview](../02-architecture-overview/README.md) | Understand system design |
| Chunking Methods | [03-chunking-methods](../03-chunking-methods/README.md) | Learn splitting techniques |
| Inference Pipelines | [04-inference-pipelines](../04-inference-pipelines/README.md) | Run split models |
| Hardware Deployment | [05-hardware-deployment](../05-hardware-deployment/README.md) | Target specific hardware |

---

## Common First-Run Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Provider not found" | Missing acceleration package | Install appropriate ORT package |
| "No suitable split point" | Model has non-linear topology | Try layer-based splitting |
| "PSNR verification failed" | Numerical precision issues | Lower threshold or check model |
| "Out of memory during split" | Both original + split in memory | Use split-only mode |

---

*For detailed API reference, see [Appendix: API Reference](../07-appendix/api-reference.md).*
