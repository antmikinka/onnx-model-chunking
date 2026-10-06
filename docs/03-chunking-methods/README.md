# Chunking Methods

**Outcome Statement:** After reading this section, you will understand all available splitting techniques, their tradeoffs, and when to use each method.

---

## Overview of Splitting Techniques

| Method | Output | Memory Savings | Quality | Speed | Best For |
|--------|--------|----------------|---------|-------|----------|
| **Semantic Splitting** | N+4 files (per component) | 60-95% | Exact | ~2-5× slower | Memory-constrained devices |
| **Graph Bisection** | 2 chunks | ~50% | PSNR verified | ~1.5× slower | Simple 2-way splits |
| **SafeTensors Splitting** | Per-layer weight files | Distribution only | Exact | None (format only) | Model distribution |

---

## Method Comparison Matrix

```
                        Memory    Quality    Speed     Complexity    Hardware
                        Savings   Preserved  Impact                Required
┌─────────────────────┬──────────┬──────────┬─────────┬────────────┬──────────┐
│ Semantic Splitting  │  ⭐⭐⭐⭐⭐  │  ⭐⭐⭐⭐⭐ │  ⭐⭐     │  Medium    │  CPU+    │
│ Graph Bisection     │  ⭐⭐⭐     │  ⭐⭐⭐⭐   │  ⭐⭐⭐    │  Low       │  CPU+    │
│ SafeTensors Split   │  ⭐       │  ⭐⭐⭐⭐⭐ │  ⭐⭐⭐⭐⭐ │  Low       │  Any     │
└─────────────────────┴──────────┴──────────┴─────────┴────────────┴──────────┘

Legend:
⭐⭐⭐⭐⭐ = Excellent
⭐⭐⭐⭐ = Good
⭐⭐⭐ = Moderate
⭐⭐ = Fair
⭐ = Poor
```

---

## Table of Contents

1. [Semantic Splitting (Recommended)](#semantic-splitting-recommended)
2. [Graph Bisection](#graph-bisection)
3. [SafeTensors Splitting](#safetensors-splitting)
4. [Method Selection Guide](#method-selection-guide)

---

## Semantic Splitting (Recommended)

**Outcome:** Understand how to use semantic splitting for maximum memory savings.

### What Is Semantic Splitting?

Semantic splitting decomposes a model at **architectural boundaries** - each transformer layer becomes a standalone ONNX file.

```
Monolithic Model:
┌─────────────────────────────────────────────────────────────────┐
│ [Embed][L0][L1][L2]...[L27][Norm][LM_Head] = 1050 MB total    │
└─────────────────────────────────────────────────────────────────┘

After Semantic Split:
┌────────┐ ┌─────┐ ┌─────┐     ┌─────┐ ┌──────┐ ┌──────┐
│ Embed  │→│ L0  │→│ L1  │→...→│ L27 │→│ Norm │→│ LM_Head│
│ 519 MB │ │16 MB│ │16 MB│     │16 MB│ │<1 MB │ │ 70 MB │
└────────┘ └─────┘ └─────┘     └─────┘ └──────┘ └──────┘

Peak memory during inference: 519 MB (largest component)
Memory savings: 51% vs monolithic
```

### When to Use Semantic Splitting

| Scenario | Recommendation |
|----------|----------------|
| Memory-constrained CPU inference | ✅ Strongly recommended |
| Running models larger than RAM | ✅ Only viable option |
| GPU with limited VRAM | ✅ Use with persistent mode |
| NPU acceleration | ❌ Use monolithic (NPU requirement) |
| Maximum throughput required | ❌ Use monolithic if it fits |

### How to Use

**Step 1: Run the Splitter**

```bash
cd MASTER_POC

python grok_onnx_split.py \
    --model "path/to/model.onnx" \
    --output "split_output/my_model" \
    --split-only
```

**Step 2: Review Output**

```
split_output/my_model/
├── embeddings.onnx          # Token embeddings
├── layer_0.onnx             # Transformer layer 0
├── layer_1.onnx             # Transformer layer 1
├── ...                      # More layer files
├── layer_27.onnx            # Transformer layer 27
├── norm.onnx                # Final normalization
├── lm_head.onnx             # Language model head
└── model_info.json          # Metadata
```

**Step 3: Run Inference**

```bash
python grok_inference_split_model_pipeline.py \
    --model-dir "split_output/my_model" \
    --prompt "Explain quantum computing" \
    --tokens 50 \
    --sequential  # Use sequential mode for minimum memory
```

### Supported Architectures

| Architecture | Status | Layer Pattern |
|--------------|--------|---------------|
| **LLaMA Family** | ✅ Fully supported | `model.layers.\d+` |
| **Qwen/Qwen2** | ✅ Fully supported | `model.layers.\d+` |
| **GPT-2** | ✅ Fully supported | `transformer.h.\d+` |
| **Phi** | ✅ Fully supported | `model.layers.\d+` |
| **Mistral** | ✅ Fully supported | `model.layers.\d+` |
| **Gemma** | ✅ Fully supported | `model.layers.\d+` |
| **Mixture of Experts** | 🔧 Infrastructure-ready | Requires custom handling |

### Technical Details

**Architecture Detection:**
```python
def detect_architecture_from_onnx(model: onnx.ModelProto) -> ArchitectureInfo:
    """
    Scan node names to identify architecture type.

    Returns:
        ArchitectureInfo with:
        - arch_type: Llama, Qwen, GPT2, etc.
        - num_layers: Count of transformer layers
        - component_prefixes: Names for embed, layers, norm, lm_head
    """
    node_names = [node.name for node in model.graph.node]

    # Pattern matching
    if any("model/layers." in n for n in node_names):
        return ArchitectureInfo(arch_type="llama", ...)
    elif any("transformer/h." in n for n in node_names):
        return ArchitectureInfo(arch_type="gpt2", ...)
    # ... more patterns
```

**Boundary Tensor Typing:**
```python
# Critical: Explicitly type all boundary tensors
def create_boundary_tensor(name, dtype, shape):
    return onnx.ValueInfoProto(
        name=name,
        type=onnx.TensorTypeProto(
            elem_type=dtype,
            shape=onnx.TensorShapeProto(
                dim=[onnx.TensorShapeProto.Dimension(dim_value=s)
                     for s in shape]
            )
        )
    )

# Example usage for hidden states boundary
hidden_tensor = create_boundary_tensor(
    name="hidden_states",
    dtype=TensorProto.FLOAT,
    shape=["batch", "sequence", 896]  # 896 = hidden_dim for Qwen2.5-0.5B
)
```

### Memory Savings Calculation

**Example: Qwen2.5-0.5B (INT4 quantized, 780 MB total)**

| Component | Size | Cumulative | Peak (Sequential) |
|-----------|------|------------|-------------------|
| Embeddings | 519 MB | 519 MB | **519 MB** |
| Layer 0 | 16 MB | 535 MB | 16 MB |
| Layer 1 | 16 MB | 551 MB | 16 MB |
| Layers 2-26 | 16 MB each | ~950 MB | 16 MB |
| Layer 27 | 16 MB | ~966 MB | 16 MB |
| Norm | <1 MB | ~966 MB | <1 MB |
| LM Head | 70 MB | ~1036 MB | 70 MB |

**Result:**
- Monolithic peak: ~1036 MB (all loaded at once)
- Sequential peak: 519 MB (embeddings only)
- **Savings: 50% reduction**

---

## Graph Bisection

**Outcome:** Understand weight-based binary splitting.

### What Is Graph Bisection?

Graph bisection splits the ONNX computation graph at the **weight distribution midpoint** - creating two roughly equal chunks.

```
Monolithic Model:
┌─────────────────────────────────────────────────────────┐
│              [All Nodes] = 1000 MB                      │
└─────────────────────────────────────────────────────────┘

After Graph Bisection:
┌──────────────────────┐    ┌──────────────────────┐
│   First Chunk        │ →  │   Second Chunk       │
│   ~500 MB            │    │   ~500 MB            │
│   Nodes 0-245        │    │   Nodes 246-625      │
└──────────────────────┘    └──────────────────────┘
```

### When to Use Graph Bisection

| Scenario | Recommendation |
|----------|----------------|
| Simple 2-way split needed | ✅ Good choice |
| Model has unclear architecture | ✅ Works without architecture detection |
| Need balanced chunk sizes | ✅ Automatically balanced |
| Maximum memory savings needed | ❌ Use semantic splitting |
| More than 2 chunks needed | ❌ Use semantic or layer-based |

### How to Use

**Using builder.py (Root-Level System):**

```python
from builder import Model

extra_options = {
    "enable_chunking": True,
    "chunk_output_dir": "./chunked_models",
    "verify_chunks": True,
    "chunk_split_ratio": 0.5,  # 50/50 split
    "min_psnr_threshold": 35.0
}

model = Model(config, "float16", "float16", "cpu", "./cache", extra_options)
model.make_model("path/to/model.onnx")

# Output: model_first_chunk.onnx, model_second_chunk.onnx
```

**Using local_model_splitter.py:**

```bash
python split_local_model.py \
    "path/to/model" \
    --num-chunks 2 \
    --strategy weight_based
```

### Technical Details

**Split Point Calculation:**
```python
def _get_op_idx_split_location(model, split_ratio=0.5):
    """
    Find the operator index that bisects model by weight size.

    Algorithm:
    1. Calculate cumulative weight size for each node
    2. Find node where cumulative ≈ split_ratio × total
    3. Ensure split point is at clean boundary (non-constant, single output)
    """
    # Analyze initializer sizes
    initializer_sizes = {}
    for init in model.graph.initializer:
        initializer_sizes[init.name] = get_tensor_size(init)

    # Calculate cumulative sizes
    cumulative = 0
    total = sum(initializer_sizes.values())
    target = total * split_ratio

    for i, node in enumerate(model.graph.node):
        for input_name in node.input:
            if input_name in initializer_sizes:
                cumulative += initializer_sizes[input_name]

        if cumulative >= target:
            # Found approximate split point
            if is_clean_boundary(node):
                return i

    return len(model.graph.node) // 2  # Fallback to middle
```

**Boundary Tensor Identification:**
```python
def _get_first_chunk_outputs(model, split_node_idx):
    """
    Identify tensors that cross from first to second chunk.

    These become:
    - Outputs of first chunk
    - Additional inputs to second chunk
    """
    first_chunk_nodes = model.graph.node[:split_node_idx]
    second_chunk_nodes = model.graph.node[split_node_idx:]

    # Tensors produced by first chunk
    first_outputs = set()
    for node in first_chunk_nodes:
        first_outputs.update(node.output)

    # Tensors consumed by second chunk
    second_inputs = set()
    for node in second_chunk_nodes:
        second_inputs.update(node.input)

    # Boundary tensors = intersection
    boundary = first_outputs & second_inputs

    # Add model outputs that cross boundary
    model_outputs = {o.name for o in model.graph.output}
    boundary |= (first_outputs & model_outputs)

    return list(boundary)
```

### PSNR Verification

**How PSNR Works:**
```python
def calculate_psnr(original, chunked):
    """
    Peak Signal-to-Noise Ratio between original and chunked outputs.

    PSNR = 10 × log10(MAX² / MSE)

    Where:
    - MAX = Maximum possible value (e.g., 1.0)
    - MSE = Mean Squared Error
    """
    mse = np.mean((original - chunked) ** 2)
    if mse == 0:
        return float('inf')  # Identical

    max_val = 1.0  # Normalized range
    psnr = 10 * np.log10(max_val ** 2 / mse)
    return psnr

# Interpretation:
# PSNR > 40 dB: Production-ready (identical for practical purposes)
# PSNR 35-40 dB: Good accuracy
# PSNR < 35 dB: Accuracy degradation detected
```

---

## SafeTensors Splitting

**Outcome:** Understand HuggingFace weight splitting for distribution.

### What Is SafeTensors Splitting?

SafeTensors splitting decomposes **PyTorch weight files** into per-layer components - useful for distribution and inspection, but requires merging for inference.

```
model.safetensors (988 MB)
         ↓
├── embeddings.safetensors      (519 MB)
├── layer_0_of_24.safetensors   (16 MB)
├── layer_1_of_24.safetensors   (16 MB)
│   ...
├── layer_23_of_24.safetensors  (16 MB)
├── norm.safetensors            (<1 MB)
└── lm_head.safetensors         (70 MB)
```

### When to Use SafeTensors Splitting

| Scenario | Recommendation |
|----------|----------------|
| Model distribution | ✅ Excellent for incremental downloads |
| Layer inspection/debugging | ✅ Each layer is separate file |
| PyTorch inference | ✅ Native HF format |
| ONNX sequential inference | ❌ Must merge back first |
| Runtime memory savings | ❌ No runtime benefit |

### How to Use

```bash
cd MASTER_POC

python grok_hf_split.py \
    --model "Qwen/Qwen2.5-0.5B" \
    --output "split_weights/qwen2.5-0.5b"
```

### Output Structure

```
split_weights/qwen2.5-0.5b/
├── embeddings.safetensors
├── layer_0_of_24.safetensors
├── layer_1_of_24.safetensors
├── ...
├── layer_23_of_24.safetensors
├── norm.safetensors
├── lm_head.safetensors
└── metadata.json
```

---

## Method Selection Guide

**Outcome:** Choose the right splitting method for your use case.

### Decision Tree

```
START: What is your primary goal?
│
├── Run model on memory-constrained hardware?
│   │
│   └── YES → Do you have NPU?
│       │
│       ├── YES → Use MONOLITHIC (NPU requirement)
│       │         See: Hardware Deployment → NPU
│       │
│       └── NO → Use SEMANTIC SPLITTING + Sequential mode
│                See: Semantic Splitting section
│
├── Distribute model files efficiently?
│   │
│   └── YES → Use SAFETENSORS SPLITTING
│             Enables incremental downloads
│
├── Simple 2-way split for basic memory savings?
│   │
│   └── YES → Use GRAPH BISECTION
│             Simpler setup, moderate savings
│
└── Maximum performance with no memory constraints?
    │
    └── YES → Use MONOLITHIC (no splitting)
              Load entire model, fastest inference
```

### Comparison Table

| Criterion | Semantic | Bisection | SafeTensors | Monolithic |
|-----------|----------|-----------|-------------|------------|
| **Memory Savings** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐ |
| **Quality Preserved** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Speed Impact** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Setup Complexity** | Medium | Low | Low | None |
| **NPU Compatible** | ❌ | ❌ | ❌ | ✅ |
| **Distribution** | Good | Fair | Excellent | Poor |

### Recommended Configurations

**Memory-Constrained CPU:**
```python
# Semantic splitting + sequential mode
python grok_onnx_split.py --model model.onnx --output splits/ --split-only
python grok_inference_split_model_pipeline.py --model-dir splits/ --sequential
```

**GPU with Sufficient VRAM:**
```python
# Monolithic + persistent sessions
python grok_inference_split_model_pipeline.py --model-dir model.onnx --persist
```

**NPU (AMD Ryzen AI):**
```python
# Monolithic required - do NOT split
python grok_inference_split_model_pipeline.py --model model.onnx --device npu
```

**Model Distribution:**
```python
# SafeTensors splitting for incremental download
python grok_hf_split.py --model Qwen/Qwen2.5-0.5B --output distributed/
```

---

*For inference pipeline details, see [Inference Pipelines](../04-inference-pipelines/README.md).*
