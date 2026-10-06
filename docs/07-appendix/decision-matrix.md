# Decision Matrix - Quick Reference

**Outcome Statement:** Quick reference guide for selecting the optimal configuration based on hardware and constraints.

---

## Hardware × Configuration Matrix

### CPU Deployment

| Memory Constraint | Model Size | Split Method | Execution Mode | Providers |
|-------------------|------------|--------------|----------------|-----------|
| **Severe** (< 8 GB) | Any | Semantic | Sequential | CPUExecutionProvider |
| **Moderate** (8-16 GB) | < RAM | Semantic or None | Persistent | CPUExecutionProvider |
| **Relaxed** (> 16 GB) | < RAM/2 | None | Monolithic | CPUExecutionProvider |

**CPU Command Examples:**

```bash
# Memory-constrained (Sequential)
python grok_inference_split_model_pipeline.py \
    --model-dir "split_output/my_model" \
    --providers "CPUExecutionProvider"

# Relaxed memory (Persistent)
python grok_inference_split_model_pipeline.py \
    --model-dir "split_output/my_model" \
    --providers "CPUExecutionProvider" \
    --persist
```

---

### GPU Deployment

| VRAM | Model Size | Split Method | Execution Mode | Providers |
|------|------------|--------------|----------------|-----------|
| **4-8 GB** | < VRAM | None | Persistent + IO-Bound | CUDA/DmlExecutionProvider |
| **4-8 GB** | > VRAM | Semantic | Persistent + IO-Bound | CUDA/DmlExecutionProvider |
| **12-24 GB** | < VRAM | None | Persistent + IO-Bound | CUDAExecutionProvider |
| **12-24 GB** | > VRAM | Semantic | Persistent + IO-Bound | CUDAExecutionProvider |
| **40-80 GB** | Any < VRAM | None | Persistent + IO-Bound | CUDAExecutionProvider |

**GPU Command Examples:**

```bash
# Model fits in VRAM (Monolithic)
python grok_inference_split_model_pipeline.py \
    --model "model.onnx" \
    --providers "CUDAExecutionProvider" \
    --persist

# Model exceeds VRAM (Split)
python grok_inference_split_model_pipeline.py \
    --model-dir "split_output/my_model" \
    --providers "CUDAExecutionProvider" \
    --persist
```

---

### NPU Deployment (AMD Ryzen AI)

| NPU Memory | Model Size | Action | Configuration |
|------------|------------|--------|---------------|
| **Any** | < NPU Memory | **Use MONOLITHIC** | VitisAIExecutionProvider + persist |
| **Any** | > NPU Memory | Quantize or fall back | See below |

**⚠️ CRITICAL: NPU requires monolithic models. Never use split models on NPU.**

**NPU Command Examples:**

```bash
# CORRECT: Monolithic model on NPU
python grok_inference_split_model_pipeline.py \
    --model "model.onnx" \
    --providers "VitisAIExecutionProvider" \
    --persist

# INCORRECT: DO NOT USE - 5-100× slower!
# python grok_inference_split_model_pipeline.py \
#     --model-dir "split_output/my_model" \
#     --providers "VitisAIExecutionProvider"
```

---

## Quick Configuration Generator

### Step 1: Identify Your Hardware

```bash
# Check available providers
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Expected outputs:
- `['CPUExecutionProvider']` → CPU only
- `['CUDAExecutionProvider', 'CPUExecutionProvider']` → NVIDIA GPU
- `['DmlExecutionProvider', 'CPUExecutionProvider']` → DirectML (AMD/Intel GPU)
- `['VitisAIExecutionProvider', 'CPUExecutionProvider']` → AMD NPU

---

### Step 2: Check Model Size

```bash
# Check model size
du -sh model.onnx  # Linux/Mac
dir /s model.onnx  # Windows

# Or in Python
import os
size_mb = os.path.getsize("model.onnx") / (1024 * 1024)
print(f"Model size: {size_mb:.1f} MB")
```

---

### Step 3: Select Configuration

| Your Situation | Configuration |
|----------------|---------------|
| CPU only, < 8 GB RAM, large model | Split + Sequential |
| CPU only, > 8 GB RAM, small model | Monolithic or Split + Persistent |
| NVIDIA GPU, model fits in VRAM | Monolithic + Persistent |
| NVIDIA GPU, model exceeds VRAM | Split + Persistent + IO-Bound |
| AMD GPU (DirectML) | Same as NVIDIA, use DmlExecutionProvider |
| AMD NPU available, model fits | **Monolithic ONLY** + Persistent |
| AMD NPU available, model too large | Quantize or use GPU/CPU |

---

## Configuration Templates

### Template 1: Memory-Constrained CPU (Sequential)

```python
from grok_inference_split_model_pipeline import SequentialModelPipeline

pipeline = SequentialModelPipeline(
    model_path="split_output/my_model",  # Split model required
    tokenizer_path="Qwen/Qwen2.5-0.5B",
    providers=["CPUExecutionProvider"],
    persist_sessions=False  # Sequential loading = minimum memory
)
```

**When to use:**
- RAM < 8 GB
- Model size > available RAM
- Latency not critical

---

### Template 2: GPU with Sufficient VRAM (Persistent)

```python
from grok_inference_split_model_pipeline import SequentialModelPipeline

pipeline = SequentialModelPipeline(
    model_path="model.onnx",  # Monolithic if fits in VRAM
    tokenizer_path="Qwen/Qwen2.5-0.5B",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    persist_sessions=True  # Pre-load all sessions
)
```

**When to use:**
- GPU with sufficient VRAM
- Low-latency requirements
- Repeated inference (chat, batch)

---

### Template 3: NPU Deployment (Monolithic Required)

```python
from grok_inference_split_model_pipeline import SequentialModelPipeline
from pipeline_utils import SessionConfig, NPUConfig

# NPU-specific configuration
npu_config = NPUConfig(
    graph_capture=True,
    weight_preload=True,
    compilation_mode="default"
)

pipeline = SequentialModelPipeline(
    model_path="model.onnx",  # MUST be monolithic for NPU!
    tokenizer_path="Qwen/Qwen2.5-0.5B",
    providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
    persist_sessions=True  # REQUIRED for NPU
)
```

**When to use:**
- AMD Ryzen AI NPU available
- Model fits in NPU memory
- Power-efficient inference needed

---

### Template 4: Model Exceeds VRAM (Split + GPU)

```python
from grok_inference_split_model_pipeline import SequentialModelPipeline

pipeline = SequentialModelPipeline(
    model_path="split_output/my_model",  # Split to fit in VRAM
    tokenizer_path="Qwen/Qwen2.5-0.5B",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    persist_sessions=True  # Pre-load all for performance
)
```

**When to use:**
- Model larger than GPU VRAM
- GPU acceleration desired
- Accept memory overhead of persistent mode

---

## Performance Expectations

### By Hardware

| Hardware | Configuration | Expected Time/Token |
|----------|---------------|---------------------|
| CPU | Sequential | 2-5 seconds |
| CPU | Persistent | 500ms - 1 second |
| GPU (NVIDIA) | Monolithic + Persistent | 20-50 ms |
| GPU (NVIDIA) | Split + Persistent | 50-100 ms |
| GPU (DirectML) | Monolithic + Persistent | 50-100 ms |
| NPU | Monolithic + Persistent | 20-100 ms |
| NPU | Split (DO NOT USE) | 500ms - 5 seconds |

---

## Memory Expectations

### By Configuration

| Configuration | Peak Memory |
|---------------|-------------|
| Sequential (Split) | Largest component (~50% of model) |
| Persistent (Split) | All components (100% of model) |
| Monolithic | Full model size |
| IO-Bound | Model + device buffers (~10-20% overhead) |

**Example: Qwen2.5-0.5B (780 MB on disk, ~1050 MB loaded)**

| Configuration | Peak Memory |
|---------------|-------------|
| Sequential (Split) | ~519 MB (embeddings only) |
| Persistent (Split) | ~1050 MB (all components) |
| Monolithic | ~1050 MB |
| IO-Bound + Persistent | ~1200 MB (includes buffers) |

---

## Decision Flowchart

```
START
│
▼
What is your target hardware?
│
├── CPU only
│   │
│   └── Is memory severely constrained?
│       ├── YES → Split + Sequential mode
│       └── NO → Monolithic or Split + Persistent
│
├── GPU (CUDA/DirectML)
│   │
│   └── Does model fit in VRAM?
│       ├── YES → Monolithic + Persistent + IO-Bound
│       └── NO → Split + Persistent + IO-Bound
│
└── NPU (AMD Ryzen AI)
    │
    └── Does model fit in NPU memory?
        ├── YES → MONOLITHIC + Persistent (REQUIRED)
        └── NO → Quantize to fit, OR fall back to GPU/CPU
```

---

## Common Mistakes to Avoid

| Mistake | Consequence | Correction |
|---------|-------------|------------|
| Using split model on NPU | 5-100× slowdown | Use monolithic on NPU |
| Sequential mode on GPU | Unnecessary disk I/O | Use persistent mode |
| Not enabling persist on NPU | Compilation on every token | Enable persist_sessions |
| Using monolithic when model > RAM | Out of memory errors | Use split + sequential |
| Wrong provider priority | Falls back to CPU | Put fastest provider first |

---

*For detailed hardware guidance, see [Hardware Deployment](../05-hardware-deployment/README.md).*
