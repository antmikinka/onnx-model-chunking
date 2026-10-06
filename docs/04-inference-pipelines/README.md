# Inference Pipelines

**Outcome Statement:** After reading this section, you will understand all execution modes and be able to choose the optimal pipeline configuration for your hardware.

---

## Execution Modes Overview

| Mode | Memory | Speed | I/O | Best For |
|------|--------|-------|-----|----------|
| **Sequential** | Minimum (1 component at a time) | Slowest | Disk every step | Memory-constrained CPU |
| **Persistent** | Maximum (all loaded) | Fast | Only at startup | GPU with sufficient VRAM |
| **IO-Bound** | All + device buffers | Fastest | Zero CPU copies | NPU/GPU production |

---

## Table of Contents

1. [Sequential Mode](#sequential-mode-memory-optimized)
2. [Persistent Mode](#persistent-mode-latency-optimized)
3. [IO-Bound Mode](#io-bound-mode-zero-copy)
4. [Mode Selection Guide](#mode-selection-guide)

---

## Sequential Mode (Memory-Optimized)

**Outcome:** Understand how to use sequential mode for minimum memory inference.

### What Is Sequential Mode?

Sequential mode loads, runs, and unloads each component one at a time. Only one component resides in memory at any moment.

```
For each token:
│
├── Load embeddings.onnx → Run → Unload  (519 MB peak)
│
├── Load layer_0.onnx → Run → Unload     (16 MB peak)
├── Load layer_1.onnx → Run → Unload     (16 MB peak)
├── ...
├── Load layer_27.onnx → Run → Unload    (16 MB peak)
│
├── Load norm.onnx → Run → Unload        (<1 MB peak)
│
└── Load lm_head.onnx → Run → Unload     (70 MB peak)

Overall peak: 519 MB (embeddings)
```

### When to Use Sequential Mode

| Scenario | Recommendation |
|----------|----------------|
| Severe memory constraints | ✅ Strongly recommended |
| CPU-only inference | ✅ Default choice |
| Running models > 2× RAM size | ✅ Only viable option |
| Maximum throughput needed | ❌ Use persistent mode |
| GPU with sufficient VRAM | ❌ Use persistent mode |

### How to Use

**Command Line:**
```bash
python grok_inference_split_model_pipeline.py \
    --model-dir "split_output/my_model" \
    --prompt "Explain machine learning" \
    --tokens 50 \
    --sequential  # Explicit sequential mode
```

**Python API:**
```python
from grok_inference_split_model_pipeline import SequentialModelPipeline

# Memory-optimized configuration
pipeline = SequentialModelPipeline(
    model_path="split_output/my_model",
    tokenizer_path="Qwen/Qwen2.5-0.5B",
    providers=["CPUExecutionProvider"],
    persist_sessions=False  # Critical: sequential loading
)

output = pipeline.generate("What is AI?", max_new_tokens=50)
print(output)

pipeline.cleanup()
```

### Memory Profile

```
Memory Usage Over Time (Sequential Mode):

519 MB │█████████████████████░░░░░░░░░░░░░░░░░░░░░  Embeddings
       │░░░░░░░░░░░░░░░░░░░░░█████████░░░░░░░░░░░░  Layer 0
       │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█████████░░░░  Layer 1
       │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░...   ...
       │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█████  Norm
       │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░██████  LM Head
       └────────────────────────────────────────────
       Time →

Peak: 519 MB
Average: ~100 MB
```

### Performance Characteristics

| Metric | Value |
|--------|-------|
| Peak Memory | 519 MB (Qwen2.5-0.5B example) |
| Average Memory | ~100 MB |
| Disk I/O | 31 loads per token (for 28-layer model) |
| Time per Token | ~2-5 seconds (depends on hardware) |
| First Token Latency | ~5-10 seconds (full prefill) |

### Optimization Tips

1. **Use SSD storage** - Faster component loading reduces latency
2. **Enable OS caching** - Frequently accessed files stay in RAM
3. **Reduce sequence length** - Smaller activations = less memory
4. **Close other applications** - Maximize available RAM for OS caching

---

## Persistent Mode (Latency-Optimized)

**Outcome:** Understand how to use persistent mode for low-latency inference.

### What Is Persistent Mode?

Persistent mode pre-loads all components at startup and keeps them resident in memory. No disk I/O occurs during inference.

```
Startup:
│
├── Load embeddings.onnx → Keep resident  (519 MB)
├── Load layer_0.onnx → Keep resident     (16 MB)
├── Load layer_1.onnx → Keep resident     (16 MB)
│   ...
├── Load layer_27.onnx → Keep resident    (16 MB)
├── Load norm.onnx → Keep resident        (<1 MB)
└── Load lm_head.onnx → Keep resident     (70 MB)

Total: 1050 MB loaded once

Inference (per token):
│
├── Run embeddings session (already loaded)
├── Run layer_0 session (already loaded)
├── ...
└── Run lm_head session (already loaded)

No disk I/O during inference!
```

### When to Use Persistent Mode

| Scenario | Recommendation |
|----------|----------------|
| GPU with sufficient VRAM | ✅ Strongly recommended |
| Low-latency requirements | ✅ Best choice |
| Repeated inference (chat, batch) | ✅ Amortizes startup cost |
| System RAM < model size | ❌ Use sequential mode |
| One-time inference | ❌ Startup overhead not worth it |

### How to Use

**Command Line:**
```bash
python grok_inference_split_model_pipeline.py \
    --model-dir "split_output/my_model" \
    --prompt "Explain machine learning" \
    --tokens 50 \
    --persist  # Enable persistent mode
```

**Python API:**
```python
from grok_inference_split_model_pipeline import SequentialModelPipeline

# Latency-optimized configuration
pipeline = SequentialModelPipeline(
    model_path="split_output/my_model",
    tokenizer_path="Qwen/Qwen2.5-0.5B",
    providers=["CUDAExecutionProvider"],  # GPU acceleration
    persist_sessions=True  # Critical: pre-load all
)

# Warm up (optional, amortizes startup cost)
_ = pipeline.generate("warmup", max_new_tokens=1)

# Actual inference
output = pipeline.generate("What is AI?", max_new_tokens=50)
print(output)

pipeline.cleanup()
```

### Memory Profile

```
Memory Usage Over Time (Persistent Mode):

1050 MB│████████████████████████████████████████████  All components loaded
       │
       │
       │
       │
       │
       │
       └────────────────────────────────────────────
       Startup        Inference         Cleanup
       (load all)     (no I/O)          (unload all)

Peak: 1050 MB (constant during inference)
Startup: ~2-5 seconds (one-time cost)
Per-token: ~50-200ms (no disk I/O)
```

### Performance Characteristics

| Metric | Value |
|--------|-------|
| Peak Memory | 1050 MB (all components) |
| Startup Time | ~2-5 seconds |
| Disk I/O | Only at startup |
| Time per Token | ~50-200ms (GPU: ~20-50ms) |
| First Token Latency | ~100-500ms (after warmup) |

### Optimization Tips

1. **Pre-load at application startup** - Amortize cost over many inferences
2. **Use GPU providers** - CUDA/ROCm/DirectML for faster execution
3. **Warm up before actual use** - Run one dummy inference to prime sessions
4. **Keep pipeline resident** - For chat/batch scenarios, don't cleanup between requests

---

## IO-Bound Mode (Zero-Copy)

**Outcome:** Understand how to use IO binding for maximum throughput.

### What Is IO-Bound Mode?

IO-bound mode combines persistent sessions with device-resident buffers. Component outputs stay on the device (GPU/NPU) without copying back to CPU.

```
Without IO Binding:
Component A output
    ↓ (Device → CPU copy: 10-50 μs)
CPU numpy array
    ↓ (CPU → Device copy: 10-50 μs)
Component B input


With IO Binding:
Component A output
    ↓ (stays on device: 0 μs)
Device Buffer
    ↓ (zero-copy!)
Component B input
```

### When to Use IO-Bound Mode

| Scenario | Recommendation |
|----------|----------------|
| Production GPU deployment | ✅ Strongly recommended |
| Maximum throughput required | ✅ Best choice |
| NPU acceleration | ✅ Use with monolithic models |
| Memory-constrained systems | ❌ Use sequential mode |
| CPU-only inference | ❌ No benefit (no device memory) |

### How to Use

**Python API with IO Binding:**
```python
from grok_inference_split_model_pipeline import SequentialModelPipeline
from pipeline_utils import IOBindingManager

# Initialize IO binding manager
binding = IOBindingManager(device_type="cuda")

# Create pipeline with persistent sessions
pipeline = SequentialModelPipeline(
    model_path="split_output/my_model",
    tokenizer_path="Qwen/Qwen2.5-0.5B",
    providers=["CUDAExecutionProvider"],
    persist_sessions=True  # Required for IO binding
)

# IO binding is automatically used when:
# 1. persist_sessions=True
# 2. GPU/NPU provider is available
# 3. Device buffers can be allocated

output = pipeline.generate("What is AI?", max_new_tokens=50)
print(output)

pipeline.cleanup()
```

### Memory Profile

```
Memory Usage Over Time (IO-Bound Mode):

1200 MB│█████████████████████████████████████████████  Components + buffers
       │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  Device buffers (~150 MB)
       │
       │
       │
       │
       │
       └────────────────────────────────────────────
       Startup        Inference         Cleanup

Peak: 1200 MB (1050 MB components + 150 MB device buffers)
Device Buffers: Pre-allocated for zero-copy transfers
Per-token: ~20-100ms (zero-copy between components)
```

### Performance Characteristics

| Metric | Value |
|--------|-------|
| Peak Memory | 1200 MB (components + buffers) |
| Device Memory | ~150 MB (buffer pool) |
| CPU ↔ Device Copies | Zero between components |
| Time per Token | ~20-100ms (depends on model size) |
| Throughput | ~10-50 tokens/second |

### Zero-Copy Implementation

```python
class IOBindingManager:
    def __init__(self, device_type="cuda"):
        self.device_type = device_type
        self.device_buffers = {}

    def bind_output(self, session, name, shape, dtype):
        """
        Bind session output to device-resident buffer.

        Eliminates Device → CPU → Device round-trip.
        """
        # Allocate device buffer
        buffer = allocate_device_buffer(shape, dtype, self.device_type)

        # Bind to session output
        io_binding = session.io_binding()
        io_binding.bind_output(name, buffer)

        # Store for cross-component passing
        self.device_buffers[name] = buffer

        return io_binding

    def get_device_buffer(self, name):
        """Return device-resident buffer (no CPU copy)."""
        return self.device_buffers.get(name)
```

---

## Mode Selection Guide

**Outcome:** Choose the optimal execution mode for your scenario.

### Decision Tree

```
START: What is your primary constraint?
│
├── Memory (RAM/VRAM limited)?
│   │
│   ├── YES, severely constrained → SEQUENTIAL mode
│   │   - Peak memory = largest component
│   │   - Accept disk I/O overhead
│   │
│   └── NO, sufficient memory → Continue...
│
├── Latency (fast response needed)?
│   │
│   ├── YES, interactive/chat → PERSISTENT mode
│   │   - Pre-load all sessions
│   │   - No per-token disk I/O
│   │
│   └── NO, batch/offline → Continue...
│
└── Throughput (max tokens/second)?
    │
    └── YES, production → IO-BOUND mode
        - Zero-copy device buffers
        - Maximum throughput
```

### Comparison Matrix

| Criterion | Sequential | Persistent | IO-Bound |
|-----------|------------|------------|----------|
| **Peak Memory** | 519 MB | 1050 MB | 1200 MB |
| **Startup Time** | Instant | ~2-5s | ~2-5s |
| **Per-Token Latency** | ~2-5s | ~50-200ms | ~20-100ms |
| **Disk I/O** | Every token | Startup only | Startup only |
| **Device Copies** | CPU only | CPU ↔ Device | Zero-copy |
| **Best Hardware** | CPU | GPU | GPU/NPU |

### Recommended Configurations

**Memory-Constrained Laptop (8GB RAM):**
```python
pipeline = SequentialModelPipeline(
    model_path="split_output/my_model",
    providers=["CPUExecutionProvider"],
    persist_sessions=False  # Sequential
)
```

**Desktop with GPU (16GB VRAM):**
```python
pipeline = SequentialModelPipeline(
    model_path="split_output/my_model",
    providers=["CUDAExecutionProvider"],
    persist_sessions=True  # Persistent
)
```

**Production Server (AMD Ryzen AI NPU):**
```python
# NPU requires monolithic model - do NOT split
pipeline = SequentialModelPipeline(
    model_path="model.onnx",  # Monolithic
    providers=["VitisAIExecutionProvider"],
    persist_sessions=True  # Required for NPU
)
```

---

*For hardware-specific deployment guidance, see [Hardware Deployment](../05-hardware-deployment/README.md).*
