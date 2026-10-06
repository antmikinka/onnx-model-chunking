# Hardware Deployment Guide

**Outcome Statement:** After reading this section, you will understand how to deploy on CPU, GPU, and NPU hardware with optimal configurations for each platform.

---

## Critical Hardware Decision Matrix

```
START: What is your target hardware?
│
├── CPU only?
│   └── Is memory severely constrained?
│       ├── YES → SPLIT + SEQUENTIAL mode
│       └── NO  → MONOLITHIC or SPLIT + PERSISTENT
│
├── GPU (CUDA/ROCm/DirectML)?
│   └── Does model fit in VRAM?
│       ├── YES → MONOLITHIC preferred
│       └── NO  → SPLIT + PERSISTENT + IO_BOUND
│
└── NPU (AMD Ryzen AI)?
    └── Does model fit in NPU memory?
        ├── YES → MONOLITHIC (REQUIRED)
        └── NO  → Quantize/prune to fit, OR fall back to GPU/CPU
```

---

## Table of Contents

1. [CPU Deployment](#cpu-deployment)
2. [GPU Deployment](#gpu-deployment)
3. [NPU Deployment (Critical)](#npu-deployment-critical)
4. [Hardware Comparison](#hardware-comparison)

---

## CPU Deployment

**Outcome:** Understand how to deploy for CPU-only inference.

### CPU Characteristics

| Characteristic | Value |
|----------------|-------|
| Memory Type | System RAM (DDR5/LPDDR5X) |
| Typical RAM | 8-64 GB |
| Memory Bandwidth | ~50-100 GB/s |
| Compute | General-purpose, flexible |
| Compilation | None (native execution) |
| Context Switch | ~1 μs |

### When to Use CPU

| Scenario | Recommendation |
|----------|----------------|
| No GPU/NPU available | ✅ Only option |
| Memory severely constrained | ✅ Sequential mode enables larger models |
| Development/testing | ✅ Universal compatibility |
| Production with latency requirements | ❌ Use GPU/NPU |

### CPU Deployment Configuration

**Memory-Constrained (Sequential Mode):**
```bash
python grok_inference_split_model_pipeline.py \
    --model-dir "split_output/my_model" \
    --providers "CPUExecutionProvider" \
    --prompt "Explain quantum computing" \
    --tokens 50
```

**Python API:**
```python
from grok_inference_split_model_pipeline import SequentialModelPipeline

# Minimum memory configuration
pipeline = SequentialModelPipeline(
    model_path="split_output/my_model",
    providers=["CPUExecutionProvider"],
    persist_sessions=False  # Sequential loading
)

output = pipeline.generate("What is AI?", max_new_tokens=50)
```

**Relaxed Memory (Persistent Mode):**
```python
# If RAM allows, use persistent mode for better latency
pipeline = SequentialModelPipeline(
    model_path="split_output/my_model",
    providers=["CPUExecutionProvider"],
    persist_sessions=True  # Pre-load all
)
```

### CPU Optimization Tips

1. **Use SSD storage** - Faster component loading for sequential mode
2. **Enable OS file caching** - Keep frequently accessed files in RAM
3. **Close other applications** - Maximize available RAM
4. **Consider INT4 quantization** - Reduces model size ~4×

### Expected Performance (CPU)

| Model | Mode | Peak Memory | Time/Token |
|-------|------|-------------|------------|
| Qwen2.5-0.5B | Sequential | 519 MB | ~2-5s |
| Qwen2.5-0.5B | Persistent | 1050 MB | ~500ms-1s |
| Llama-7B | Sequential | ~4 GB | ~10-20s |

---

## GPU Deployment

**Outcome:** Understand how to deploy for GPU-accelerated inference.

### GPU Characteristics

| Characteristic | Value |
|----------------|-------|
| Memory Type | GDDR6/HBM (VRAM) |
| Typical VRAM | 4-24 GB (consumer), 40-80 GB (datacenter) |
| Memory Bandwidth | ~200-900 GB/s |
| Compute | Massively parallel |
| Compilation | JIT (~10 μs) |
| Context Switch | ~10 μs |

### GPU Provider Options

| Provider | Platform | Package |
|----------|----------|---------|
| **CUDA** | NVIDIA | `onnxruntime-gpu` |
| **ROCm** | AMD (pro GPUs) | `onnxruntime-rocm` |
| **DirectML** | AMD/NVIDIA/Intel (consumer) | `onnxruntime-directml` |

### When to Use GPU

| Scenario | Recommendation |
|----------|----------------|
| Model fits in VRAM | ✅ Strongly recommended |
| Low-latency requirements | ✅ Best choice |
| High-throughput production | ✅ Maximum performance |
| Model exceeds VRAM | ✅ Use split + persistent + IO-bound |

### GPU Deployment Configuration

**Model Fits in VRAM (Monolithic):**
```bash
python grok_inference_split_model_pipeline.py \
    --model "model.onnx" \
    --providers "CUDAExecutionProvider" \
    --prompt "Explain quantum computing" \
    --tokens 50 \
    --persist  # Keep resident
```

**Model Exceeds VRAM (Split + Persistent):**
```bash
python grok_inference_split_model_pipeline.py \
    --model-dir "split_output/my_model" \
    --providers "CUDAExecutionProvider" \
    --prompt "Explain quantum computing" \
    --tokens 50 \
    --persist  # Critical: pre-load all
```

**Python API with GPU:**
```python
from grok_inference_split_model_pipeline import SequentialModelPipeline

# GPU-optimized configuration
pipeline = SequentialModelPipeline(
    model_path="split_output/my_model",  # or monolithic .onnx
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],  # Fallback to CPU
    persist_sessions=True  # Critical for GPU performance
)

output = pipeline.generate("What is AI?", max_new_tokens=50)
```

### GPU Optimization Tips

1. **Use persistent mode** - Avoid per-token disk I/O
2. **Enable IO binding** - Zero-copy between components
3. **Pre-allocate device buffers** - Reduce allocation overhead
4. **Warm up before use** - Prime CUDA kernels with dummy inference
5. **Use fp16/bf16** - Half precision for 2× memory savings

### Expected Performance (GPU)

| Model | Mode | VRAM | Time/Token |
|-------|------|------|------------|
| Qwen2.5-0.5B | Monolithic | 1 GB | ~20-50ms |
| Qwen2.5-0.5B | Split + Persistent | 1.2 GB | ~50-100ms |
| Llama-7B | Monolithic | 14 GB | ~50-100ms |
| Llama-7B | Split + IO-Bound | ~16 GB | ~100-200ms |

---

## NPU Deployment (Critical)

**Outcome:** Understand why NPU requires monolithic models and how to deploy correctly.

---

### ⚠️ CRITICAL: NPU Requires Monolithic Models

**Outcome Statement:** NPUs (AMD XDNA, VitisAI) are fundamentally incompatible with sequential load/unload patterns. Always use monolithic models on NPU.

---

### Why NPU Is Different

| Aspect | GPU | NPU |
|--------|-----|-----|
| **Compilation** | JIT, ~10 μs | AOT, ~5-50 ms |
| **Context Switch** | ~10 μs | ~5-50 ms |
| **Memory Model** | Shared, addressable | Isolated, program-specific |
| **Graph Assumption** | Any subgraph OK | Entire graph required |
| **Kernel Loading** | Fast, flexible | Slow, fixed pipeline |

**Critical Insight:** NPU context switches are **500-5000× more expensive** than GPU.

### The NPU Monolithic Requirement

```
┌─────────────────────────────────────────────────────────────────┐
│              WHY NPU REQUIRES MONOLITHIC MODE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. AHEAD-OF-TIME COMPILATION                                   │
│     └── NPU compiles ENTIRE graph before execution             │
│     └── Split models break whole-graph optimizations            │
│                                                                  │
│  2. MEMORY MOVEMENT OVERHEAD                                    │
│     └── Every chunk boundary: NPU → DDR → CPU → DDR → NPU      │
│     └── Catastrophic bandwidth overhead                         │
│                                                                  │
│  3. CONTEXT SWITCH COST                                         │
│     └── NPU program loading: 5-50 ms                            │
│     └── GPU kernel switch: ~10 μs                               │
│     └── 500-5000× more expensive!                               │
│                                                                  │
│  4. QUANTIZATION COHERENCE                                      │
│     └── NPU models need globally calibrated INT8/INT4           │
│     └── Split models break quantization pipeline                │
│                                                                  │
│  5. KV-CACHE ISOLATION                                          │
│     └── Split sessions have isolated memory                     │
│     └── KV-cache cannot be shared without expensive transfers   │
│                                                                  │
│  6. ARCHITECTURE MISMATCH                                       │
│     └── NPUs are fixed dataflow pipelines                       │
│     └── GPUs are flexible kernel dispatchers                    │
│                                                                  │
│  BOTTOM LINE: Running split models on NPU results in            │
│               5-100× performance regression                     │
└─────────────────────────────────────────────────────────────────┘
```

### NPU Deployment Decision Tree

```
START: Do you have an AMD XDNA NPU?
│
├── NO → Use GPU or CPU deployment
│
└── YES → Does model fit in NPU memory?
    │
    ├── YES → Use MONOLITHIC mode (required)
    │   │
    │   └── python grok_inference_split_model_pipeline.py \
    │           --model "model.onnx" \
    │           --device npu \
    │           --persist
    │
    └── NO → Quantize or fall back to GPU/CPU
        │
        ├── Option 1: Quantize to fit
        │   └── Use INT4 quantization (~75% size reduction)
        │
        └── Option 2: Fall back to GPU/CPU
            └── Use split + sequential on CPU/GPU
```

### Correct NPU Configuration

**✅ CORRECT - Monolithic Model:**
```bash
python grok_inference_split_model_pipeline.py \
    --model "model.onnx" \
    --providers "VitisAIExecutionProvider" \
    --prompt "Explain quantum computing" \
    --tokens 50 \
    --persist  # Critical for NPU
```

**❌ INCORRECT - Split Model on NPU:**
```bash
# DO NOT DO THIS - 5-100× slower!
python grok_inference_split_model_pipeline.py \
    --model-dir "split_output/my_model" \
    --providers "VitisAIExecutionProvider" \
    --device npu  # WRONG: NPU + split = catastrophic performance
```

### NPU Configuration Options

```python
from pipeline_utils import SessionConfig, NPUConfig

# NPU-specific configuration
npu_config = NPUConfig(
    graph_capture=True,           # Cache compiled graph (required)
    weight_preload=True,          # Preload to Tile SRAM (if fits)
    compilation_mode="default",   # "fast", "default", "optimized"
    enable_profiling=False,       # Enable performance profiling
    debug_mode=False,             # Verbose NPU logging
    memory_limit_mb=4096,         # Cap NPU memory usage
    num_cores=-1                  # Use all available cores
)

# Session configuration
session_config = SessionConfig(
    providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
    optimization_level="DISABLE",  # Required for dynamic shapes
    inter_op_threads=0,
    intra_op_threads=0,
    memory_limit_mb=4096,
    disable_shape_check=True,
    npu_graph_capture=True,
    npu_weight_preload=True,
    npu_compilation_mode="default"
)
```

### NPU Compilation Modes

| Mode | Compilation Time | Runtime Performance | Use Case |
|------|------------------|---------------------|----------|
| `fast` | ~1s | Lower | Development |
| `default` | ~5s | Balanced | General inference |
| `optimized` | ~30s | Best | Production (amortized) |

### NPU Performance Expectations

**Monolithic Model (Correct):**
| Metric | Value |
|--------|-------|
| Compilation | One-time: 5-30s |
| First Token | ~100-500ms (after compilation) |
| Per Token | ~20-100ms |
| Power | <5W typical |

**Split Model (Incorrect - DO NOT USE):**
| Metric | Value |
|--------|-------|
| Compilation | Per component: 5-50ms × 31 = 2.5-25s |
| First Token | ~5-15s |
| Per Token | ~500ms-5s |
| Power | Higher due to inefficiency |

### NPU Availability Check

```python
from pipeline_utils import is_npu_available, get_npu_capabilities

if is_npu_available():
    print("NPU detected!")
    caps = get_npu_capabilities()
    print(f"  Driver: {caps.driver_version}")
    print(f"  Cores: {caps.num_cores}")
    print(f"  Memory: {caps.memory_size_mb} MB")
    print(f"  Tile SRAM: {caps.tile_sram_mb} MB")
else:
    print("NPU not available - falling back to CPU/GPU")
```

### Hybrid NPU + iGPU Dispatch

**Outcome:** Understand AMD's recommended hybrid approach.

AMD Ryzen AI systems have both NPU and integrated GPU. The recommended flow:

```
┌─────────────────────────────────────────────────────────────────┐
│              AMD HYBRID LLM FLOW                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PREFILL PHASE (Compute-Intensive):                            │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐                   │
│  │Embedding│────▶│ Layers  │────▶│  Norm   │                   │
│  │  (iGPU) │     │  (NPU)  │     │  (CPU)  │                   │
│  └─────────┘     └─────────┘     └─────────┘                   │
│                                           │                      │
│                                           ▼                      │
│  DECODE PHASE (Memory-Bandwidth):         │LM Head (iGPU)      │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐                   │
│  │Embedding│────▶│ Layers  │────▶│  Norm   │                   │
│  │  (iGPU) │     │  (NPU)  │     │  (CPU)  │                   │
│  └─────────┘     └─────────┘     └─────────┘                   │
│                                                                  │
│  NPU: Compute-heavy prompt processing (prefill)                 │
│  iGPU: Memory-bandwidth token generation (decode)               │
└─────────────────────────────────────────────────────────────────┘
```

**Note:** Hybrid dispatch operates on **monolithic models**. Our semantic splitting is for memory-constrained CPU/GPU scenarios.

---

## Hardware Comparison

### Summary Comparison

| Hardware | Memory | Speed | Power | Best For |
|----------|--------|-------|-------|----------|
| **CPU** | 8-64 GB | Slow | Medium | Universal, memory-constrained |
| **GPU** | 4-80 GB | Fast | High | Low-latency, high-throughput |
| **NPU** | 4-16 GB | Fast | <5W | Efficient inference (monolithic only) |

### Recommended Configurations Summary

| Hardware | Model Size | Configuration |
|----------|------------|---------------|
| CPU (8GB) | < 2 GB | Split + Sequential |
| CPU (16GB) | < 8 GB | Split + Persistent |
| GPU (8GB VRAM) | < 6 GB | Monolithic + Persistent |
| GPU (16GB VRAM) | < 14 GB | Monolithic + IO-Bound |
| NPU (8GB) | < 6 GB | Monolithic + Persistent |

---

*For detailed NPU explanation, see [NPU Monolithic Requirement](../07-appendix/npu-monolithic-requirement.md).*
