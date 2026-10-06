# NPU Monolithic Model Requirement

**Critical Technical Guidance**

---

## Outcome Statement

**NPUs (AMD XDNA, VitisAI) are fundamentally incompatible with sequential load/unload patterns. Using split models on NPU results in 5-100× performance regression. Always use monolithic models on NPU.**

---

## Executive Summary

| Hardware | Split Model Performance | Monolithic Model Performance | Recommendation |
|----------|------------------------|------------------------------|----------------|
| **CPU** | ~2-5× slower than monolithic | Baseline | Use split for memory constraints |
| **GPU** | ~1.5-3× slower than monolithic | Baseline | Use split if model exceeds VRAM |
| **NPU** | **5-100× slower** | Baseline | **NEVER use split - monolithic required** |

---

## Why NPU Requires Monolithic Models

### Reason 1: Ahead-of-Time Compilation

**GPU Approach (Just-in-Time):**
```
Kernel dispatch model:
1. Submit subgraph to GPU driver
2. JIT compile (~10 microseconds)
3. Execute kernel
4. Discard compiled kernel
5. Next kernel: repeat from step 1

Total overhead per component: ~10 μs
For 31 components: ~310 μs = 0.31 ms
```

**NPU Approach (Ahead-of-Time):**
```
Fixed dataflow model:
1. Submit ENTIRE computation graph to NPU driver
2. AOT compile entire graph (5-50 milliseconds)
3. Load compiled program to NPU
4. Execute dataflow pipeline
5. Any graph change: repeat from step 1

Total overhead per component: 5-50 ms
For 31 components: 155-1550 ms = 0.15-1.55 seconds PER TOKEN
```

**Impact:** Splitting breaks whole-graph optimizations. Each component requires separate compilation, multiplying overhead by number of components.

---

### Reason 2: Memory Movement Overhead

**GPU Memory Model (Unified):**
```
┌─────────────────────────────────────────────────────────┐
│                    GPU VRAM                              │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Component A output → GPU Buffer                │    │
│  │                      ↓ (zero-copy)              │    │
│  │  Component B input ← GPU Buffer                 │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘

Memory movement: None (zero-copy possible)
```

**NPU Memory Model (Isolated):**
```
┌─────────────────────────────────────────────────────────┐
│                     System RAM                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │                  NPU Device Memory               │    │
│  │  ┌───────────────────────────────────────────┐  │    │
│  │  │          Tile SRAM (16-32 MB)              │  │    │
│  │  │  Component A output → Tile SRAM           │  │    │
│  │  └───────────────────────────────────────────┘  │    │
│  │         ↓ (NPU → DDR: ~100-500 μs)              │    │
│  │     DDR RAM (system memory)                     │    │
│  │         ↓ (DDR → CPU: ~50-100 μs)               │    │
│  │     CPU Memory                                  │    │
│  │         ↓ (CPU → DDR: ~50-100 μs)               │    │
│  │     DDR RAM (system memory)                     │    │
│  │         ↓ (DDR → NPU: ~100-500 μs)              │    │
│  │  ┌───────────────────────────────────────────┐  │    │
│  │  │          Tile SRAM (16-32 MB)              │  │    │
│  │  │  Component B input ← Tile SRAM            │  │    │
│  │  └───────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘

Memory movement per component boundary: ~300-1200 μs
For 31 components: 9.3-37.2 ms PER TOKEN just for memory transfers
```

**Impact:** Every component boundary forces expensive NPU→DDR→CPU→DDR→NPU transfers.

---

### Reason 3: Context Switch Cost

| Operation | Latency | Ratio |
|-----------|---------|-------|
| GPU kernel launch | ~10 μs | 1× |
| NPU program load | ~5-50 ms | 500-5000× |
| GPU memory transfer (device→host) | ~10-50 μs | 1-5× |
| NPU memory transfer (device→host) | ~100-500 μs | 10-50× |

**Sequential Mode Overhead Calculation:**

For a 28-layer model (31 components total):

**GPU:**
```
Per-token overhead:
31 components × 10 μs = 310 μs = 0.31 ms

Acceptable overhead for memory savings.
```

**NPU:**
```
Per-token overhead:
31 components × 5-50 ms = 155-1550 ms = 0.15-1.55 seconds

Catastrophic overhead. Makes inference 5-100× slower.
```

---

### Reason 4: Quantization Coherence

**Monolithic Quantization (Correct for NPU):**
```
Entire model → Global calibration → Consistent INT8/INT4 scales
│
├── Layer 0: scale = 0.0078 (INT8)
├── Layer 1: scale = 0.0078 (INT8)  ← Same scale
├── ...
└── Layer 27: scale = 0.0078 (INT8)  ← Same scale

Result: Coherent quantization across all layers
```

**Split Quantization (Breaks NPU Requirements):**
```
Component 0 → Local calibration → scale = 0.0078
Component 1 → Local calibration → scale = 0.0082  ← Different!
Component 2 → Local calibration → scale = 0.0075  ← Different!
...

Result: Inconsistent scales cause accuracy degradation
```

**Impact:** NPU quantization requires globally calibrated scales. Split models break this calibration.

---

### Reason 5: KV-Cache Isolation

**Monolithic KV-Cache (Correct):**
```
Unified memory space:
Layer 0 KV → Shared memory → Layer 1 reads (zero-copy)
Layer 1 KV → Shared memory → Layer 2 reads (zero-copy)
...

Transfer overhead: ~0 μs (same address space)
```

**Split KV-Cache (Catastrophic):**
```
Isolated session memory:
Layer 0 KV → Session 0 memory → CPU → Session 1 memory → Layer 1
Layer 1 KV → Session 1 memory → CPU → Session 2 memory → Layer 2
...

Transfer overhead per layer: ~100-500 μs
For 28 layers: 2.8-14 ms PER TOKEN
```

---

### Reason 6: Architecture Mismatch

| Characteristic | GPU | NPU |
|----------------|-----|-----|
| **Execution Model** | Kernel dispatch | Fixed dataflow |
| **Graph Flexibility** | Any subgraph OK | Entire graph required |
| **Memory Addressing** | Unified, shared | Isolated, program-specific |
| **Best Workload** | Flexible, dynamic | Fixed, predictable |
| **Compilation** | JIT per-kernel | AOT whole-graph |

**Impact:** NPUs are designed for fixed, monolithic inference graphs. The flexible load/unload pattern of sequential inference fundamentally mismatches NPU architecture.

---

## Performance Comparison

### Monolithic on NPU (CORRECT)

```
Configuration:
  providers=["VitisAIExecutionProvider"]
  persist_sessions=True  # Required for NPU
  model="model.onnx"  # Monolithic, NOT split

Performance:
  Compilation: One-time 5-30 seconds
  First Token: 100-500 ms (after compilation)
  Per Token: 20-100 ms
  Throughput: ~10-50 tokens/second
  Power: <5W typical
```

### Split on NPU (INCORRECT - DO NOT USE)

```
Configuration:
  providers=["VitisAIExecutionProvider"]
  persist_sessions=False  # Sequential loading
  model_dir="split_output/my_model"  # Split components

Performance (CATASTROPHIC):
  Per-token Compilation: 31 × 5-50ms = 155-1550 ms
  Memory Transfers: 31 × (NPU→DDR→CPU→DDR→NPU) = 100-500 ms
  KV-Cache Transfers: 28 × 100-500μs = 2.8-14 ms
  Total Per Token: 500 ms - 5 seconds

  This is 5-100× SLOWER than monolithic!
  Power: Higher due to inefficiency
```

---

## Recommended NPU Deployment Flow

### Decision Tree

```
START: Do you have an AMD XDNA NPU?
│
├── NO → Use GPU or CPU deployment (see Hardware Deployment guide)
│
└── YES → Does model fit in NPU memory?
    │
    ├── YES → Use MONOLITHIC mode (REQUIRED)
    │   │
    │   └── python grok_inference_split_model_pipeline.py \
    │           --model "model.onnx" \
    │           --providers "VitisAIExecutionProvider" \
    │           --persist
    │
    └── NO → Quantize to fit, or fall back to GPU/CPU
        │
        ├── Option 1: INT4 Quantization
        │   └── Reduces model size ~75%
        │       Example: 1 GB → 250 MB
        │
        └── Option 2: Fall Back to GPU/CPU
            └── Use split + sequential on CPU/GPU
                python grok_inference_split_model_pipeline.py \
                    --model-dir "split_output/my_model" \
                    --providers "CUDAExecutionProvider" \
                    --sequential
```

### Correct NPU Configuration

```python
from grok_inference_split_model_pipeline import SequentialModelPipeline
from pipeline_utils import SessionConfig, NPUConfig

# NPU-specific configuration
npu_config = NPUConfig(
    graph_capture=True,           # Cache compiled graph (REQUIRED)
    weight_preload=True,          # Preload to Tile SRAM (if fits)
    compilation_mode="default",   # "fast", "default", "optimized"
    enable_profiling=False,
    debug_mode=False,
    memory_limit_mb=4096,
    num_cores=-1
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

# Create pipeline with MONOLITHIC model
pipeline = SequentialModelPipeline(
    model_path="model.onnx",  # NOT split_output/ - must be monolithic!
    tokenizer_path="Qwen/Qwen2.5-0.5B",
    providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
    persist_sessions=True  # REQUIRED for NPU
)

output = pipeline.generate("What is AI?", max_new_tokens=50)
```

---

## NPU Compilation Modes

| Mode | Compilation Time | Runtime Performance | Use Case |
|------|------------------|---------------------|----------|
| `fast` | ~1-5 seconds | Lower | Development, testing |
| `default` | ~5-15 seconds | Balanced | General inference |
| `optimized` | ~15-60 seconds | Best | Production (amortized over many runs) |

**Recommendation:** Use `fast` during development, `optimized` for production deployment.

---

## NPU Availability Check

```python
from pipeline_utils import is_npu_available, get_npu_capabilities

if is_npu_available():
    print("✓ NPU detected!")
    caps = get_npu_capabilities()
    print(f"  Driver: {caps.driver_version}")
    print(f"  Cores: {caps.num_cores}")
    print(f"  Memory: {caps.memory_size_mb} MB")
    print(f"  Tile SRAM: {caps.tile_sram_mb} MB")

    # Decide if model fits
    model_size_mb = 1024  # Example: 1 GB model
    if model_size_mb <= caps.memory_size_mb:
        print(f"  ✓ Model fits in NPU memory - use MONOLITHIC")
    else:
        print(f"  ✗ Model exceeds NPU memory - quantize or use GPU/CPU")
else:
    print("✗ NPU not available - falling back to CPU/GPU")
```

---

## Summary: Critical Rules for NPU Deployment

1. **ALWAYS use monolithic models on NPU** - Never split for NPU
2. **ALWAYS enable persist_sessions** - Required for NPU performance
3. **ALWAYS enable graph_capture** - Caches compiled program
4. **NEVER use sequential mode on NPU** - 5-100× slowdown
5. **If model doesn't fit, quantize or fall back** - Don't split

---

## Related Documentation

- [Hardware Deployment Guide](../05-hardware-deployment/README.md) - CPU, GPU, NPU deployment strategies
- [Decision Matrix](./decision-matrix.md) - Quick reference for configuration selection
- [API Reference](./api-reference.md) - Complete API documentation

---

*Document generated: 2026-03-21 | Critical NPU Guidance v1.0*
