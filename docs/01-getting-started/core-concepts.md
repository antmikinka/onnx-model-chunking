# Core Concepts

**Outcome Statement:** After reading this section, you will understand the mathematical and architectural foundations that make semantic model splitting possible.

---

## Concept 1: The Memory Problem

### Why Large Models Don't Fit on Consumer Hardware

**Outcome:** Understand the memory requirements of monolithic model inference.

When running an LLM, memory is consumed by:

| Component | Memory Required | Example (0.5B model) |
|-----------|-----------------|---------------------|
| Model Weights | 2-4 bytes per parameter | 1-2 GB (FP16) |
| Activations | Batch × Seq_Len × Hidden_Dim | 500 MB |
| KV Cache | 2 × Layers × KV_Heads × Head_Dim × Seq_Len | 256 MB @ 2048 seq |
| **Total** | **2-4× model size** | **~2-4 GB** |

**The fundamental problem:** All weights must be loaded simultaneously in monolithic inference.

```
Monolithic Loading:
┌─────────────────────────────────────────────────────────────┐
│  [Embed][L0][L1][L2]...[L27][Norm][LM_Head]  = 1050 MB     │
│                     ALL IN RAM AT ONCE                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Concept 2: The Splitting Solution

### Semantic Decomposition

**Outcome:** Understand how splitting at architectural boundaries preserves correctness while reducing peak memory.

A transformer model is a **composition of functions**:

```
output = f_LM_Head(f_Norm(f_L27(...f_L1(f_L0(f_Embed(input)))...)))
```

**Key Insight:** Function composition is inherently sequential. You only need:
1. Current function loaded
2. Input from previous function
3. Output buffer for next function

```
Semantic Splitting:
┌────────┐   ┌─────┐   ┌─────┐         ┌─────┐   ┌──────┐   ┌──────┐
│ Embed  │ → │ L0  │ → │ L1  │ → ... → │ L27 │ → │ Norm │ → │ LM_Head│
│ 519 MB │   │16 MB│   │16 MB│         │16 MB│   │<1 MB │   │ 70 MB │
└────────┘   └─────┘   └─────┘         └─────┘   └──────┘   └──────┘
     ↓           ↓           ↓             ↓           ↓           ↓
  Load→Run    Load→Run    Load→Run      Load→Run    Load→Run    Load→Run
  →Unload     →Unload     →Unload       →Unload     →Unload     →Keep
```

**Peak Memory:** 519 MB (largest single component) vs 1050 MB (monolithic)

---

## Concept 3: Why Semantic Boundaries Matter

### Arbitrary vs Semantic Splitting

**Outcome:** Understand why splitting at layer boundaries is superior to arbitrary graph cuts.

| Split Type | Description | Pros | Cons |
|------------|-------------|------|------|
| **Arbitrary** | Cut graph at any node | Simple algorithm | May break operator fusion, unclear interfaces |
| **Weight-Based Bisection** | Split at 50% weight point | Balanced sizes | May cut through middle of layer |
| **Semantic** | Split at architectural boundaries | Clean interfaces, preserves layer integrity | Requires architecture detection |

**This system uses semantic splitting** because:

1. **Clean Interfaces:** Each component has well-defined input/output tensors
2. **Type Safety:** Boundary tensors are explicitly typed (no ORT auto-cast errors)
3. **Standalone Validity:** Each split is a complete, valid ONNX model
4. **Metadata Inheritance:** Each component carries shape/rank metadata

### Example: Qwen2 Architecture Boundaries

```
Qwen2 Model Structure:
┌──────────────────────────────────────────────────────────────┐
│  model.embed_tokens  →  [batch, seq] → [batch, seq, hidden] │
├──────────────────────────────────────────────────────────────┤
│  model.layers.0  →  [hidden] + [KV_cache] → [hidden] + [KV] │
│  model.layers.1  →  [hidden] + [KV_cache] → [hidden] + [KV] │
│  ...                                                         │
│  model.layers.27 →  [hidden] + [KV_cache] → [hidden] + [KV] │
├──────────────────────────────────────────────────────────────┤
│  model.norm  →  [hidden] → [normalized_hidden]               │
├──────────────────────────────────────────────────────────────┤
│  lm_head  →  [normalized_hidden] → [vocab_logits]            │
└──────────────────────────────────────────────────────────────┘
```

Each boundary has explicit tensor contracts:

| Boundary | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| Embed → Layer0 | `input_ids: int64[batch, seq]` | `hidden: float32[batch, seq, hidden]` |
| Layer N → Layer N+1 | `hidden: float32[batch, seq, hidden]` | `hidden: float32[batch, seq, hidden]` |
| Last Layer → Norm | `hidden: float32[batch, seq, hidden]` | `hidden: float32[batch, seq, hidden]` |
| Norm → LM Head | `normalized: float32[batch, seq, hidden]` | `logits: float32[batch, seq, vocab]` |

---

## Concept 4: Execution Modes

### Three Strategies for Running Split Models

**Outcome:** Choose the appropriate execution mode for your hardware constraints.

| Mode | Strategy | Memory | Speed | I/O | Best For |
|------|----------|--------|-------|-----|----------|
| **Sequential** | Load → Run → Unload per component | Minimum | Slowest | Disk every step | Memory-constrained CPU |
| **Persistent** | Load all at startup, keep resident | Maximum | Fast | Only at startup | GPU with sufficient VRAM |
| **IO-Bound** | Persistent + device buffers | All + buffers | Fastest | Zero CPU copies | NPU/GPU production |

### Sequential Mode (Memory-Optimized)

```python
pipeline = SequentialModelPipeline(
    model_path="split_output/my_model",
    persist_sessions=False  # Default: sequential loading
)
```

**Memory Profile:**
```
Time 0: Load embeddings (519 MB) → Run → Unload
Time 1: Load layer_0 (16 MB) → Run → Unload
Time 2: Load layer_1 (16 MB) → Run → Unload
...
Peak: 519 MB (largest component)
```

**Tradeoff:** Minimum memory, but disk I/O on every component load.

### Persistent Mode (Latency-Optimized)

```python
pipeline = SequentialModelPipeline(
    model_path="split_output/my_model",
    persist_sessions=True  # Pre-load all sessions
)
```

**Memory Profile:**
```
Startup: Load all 31 components (1050 MB total)
Inference: All sessions resident, no disk I/O
Peak: 1050 MB (all components)
```

**Tradeoff:** Higher memory, but no per-inference disk I/O.

### IO-Bound Mode (Zero-Copy)

```python
from pipeline_utils import IOBindingManager

binding = IOBindingManager(device_type="cuda")
# Use with persistent sessions for zero-copy between components
```

**Memory Profile:**
```
All components loaded + device buffers allocated
Component outputs stay on device (GPU/NPU memory)
No CPU ↔ Device copies between components
```

**Tradeoff:** Highest memory, but maximum throughput.

---

## Concept 5: Accuracy Verification (PSNR)

### Peak Signal-to-Noise Ratio

**Outcome:** Understand how to verify split models produce identical outputs.

PSNR measures similarity between original and split model outputs:

```
PSNR = 10 × log10(MAX² / MSE)

Where:
- MAX = Maximum possible value (e.g., 1.0 for normalized tensors)
- MSE = Mean Squared Error between original and split outputs
```

| PSNR Value | Quality | Interpretation |
|------------|---------|----------------|
| > 40 dB | Production-ready | Identical for practical purposes |
| 35-40 dB | Good accuracy | Acceptable for most use cases |
| 30-35 dB | Moderate degradation | Review configuration |
| < 30 dB | Poor accuracy | Re-verify split |

**Example:**
```
Original output: [0.1234, 0.5678, 0.9012, ...]
Split output:    [0.1234, 0.5678, 0.9012, ...]
PSNR: 45.2 dB → Excellent match
```

### Verification Code

```python
from builder import Model

model = Model(config, "float16", "float16", "cpu", "./cache", {
    "enable_chunking": True,
    "verify_chunks": True,
    "min_psnr_threshold": 35.0  # Fail if PSNR < 35 dB
})

model.make_model("path/to/model.onnx")
# Verification runs automatically, raises error if PSNR too low
```

---

## Concept 6: The Transient Session Pattern

### How Sequential Inference Works

**Outcome:** Understand the runtime pattern that enables memory-constrained inference.

```
For each token generated:
    registry = {}  # Shared tensor memory on CPU

    # 1. Embedding phase
    session = load_session("embeddings.onnx")
    outputs = session.run({"input_ids": token_ids})
    registry["hidden_states"] = outputs["hidden_states"]
    unload_session("embeddings.onnx")

    # 2. Layer-by-layer processing
    for layer_id in range(num_layers):
        session = load_session(f"layer_{layer_id}.onnx")
        outputs = session.run({
            "hidden_states": registry["hidden_states"],
            "attention_mask": registry["attention_mask"],
            **registry.get_kv_inputs(layer_id)
        })
        registry["hidden_states"] = outputs["hidden_states"]
        registry.store_kv_outputs(layer_id, outputs)
        unload_session(f"layer_{layer_id}.onnx")

    # 3. Normalization phase
    session = load_session("norm.onnx")
    outputs = session.run({"hidden_states": registry["hidden_states"]})
    registry["normalized"] = outputs["normalized_hidden"]
    unload_session("norm.onnx")

    # 4. LM Head phase
    session = load_session("lm_head.onnx")
    outputs = session.run({"normalized": registry["normalized"]})
    logits = outputs["logits"]
    next_token = argmax(logits)
    unload_session("lm_head.onnx")

    # Repeat until generation complete
```

**Key Innovation:** The `TensorRegistry` acts as shared memory, passing data between transiently-loaded components.

---

## Concept 7: Hardware Decision Matrix

### Choosing the Right Strategy

**Outcome:** Select the optimal configuration for your hardware.

```
START: What is your target hardware?
    │
    ├── CPU only?
    │   └── Is memory severely constrained?
    │       ├── YES → SPLIT + SEQUENTIAL mode
    │       └── NO  → MONOLITHIC or SPLIT + PERSISTENT
    │
    ├── GPU (CUDA/ROCm/DML)?
    │   └── Does model fit in VRAM?
    │       ├── YES → MONOLITHIC preferred
    │       └── NO  → SPLIT + PERSISTENT + IO_BOUND
    │
    └── NPU (AMD Ryzen AI)?
        └── Does model fit in NPU memory?
            ├── YES → MONOLITHIC (REQUIRED)
            └── NO  → Quantize/prune to fit, OR fall back to GPU/CPU
```

**Critical Note on NPU:** NPUs require monolithic models due to ahead-of-time compilation. See [NPU Deployment](../05-hardware-deployment/npu-deployment.md) for detailed explanation.

---

## Summary

| Concept | Key Takeaway |
|---------|--------------|
| Memory Problem | Monolithic loading requires 2-4× model size in RAM |
| Splitting Solution | Semantic decomposition reduces peak to largest component |
| Semantic Boundaries | Layer-level splits preserve model integrity |
| Execution Modes | Sequential (min memory) ↔ Persistent (max speed) |
| PSNR Verification | Quantitative accuracy validation |
| Transient Sessions | Load/run/unload pattern enables memory savings |
| Hardware Matrix | Match strategy to hardware constraints |

---

*For detailed splitting techniques, see [Chunking Methods](../03-chunking-methods/README.md).*
