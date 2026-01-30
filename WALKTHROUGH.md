# Local ONNX Model Splitter - Walkthrough

## Overview

Successfully implemented a modular, scalable local ONNX model splitting system that supports **two splitting strategies**:

1. **Weight-Based Splitting** - Equal memory distribution
2. **Layer-Based Splitting** - Architecture-aware layer grouping (adapted from `grok_hf_split.py`)

## Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `local_model_splitter.py` | Core `LocalONNXModelSplitter` class | ~1,600 |
| `split_local_model.py` | CLI interface for general chunking | ~260 |
| `grok_onnx_split.py` | True Semantic Splitter (1 file per layer) | ~650 |
| `grok_inference_split_model_pipeline.py` | Sequential Inference Pipeline | ~150 |

---

## Layer-Based Splitting (from grok_hf_split.py)

### Key Methods Added

| Method | Description |
|--------|-------------|
| `get_model_prefixes()` | Returns architecture-aware prefixes (embed, layer, norm, lm_head) |
| `_detect_model_type_from_nodes()` | Detects llama/qwen/gpt2 from node names |
| `_get_layer_components()` | Extracts embedding, layers, norm, lm_head nodes |
| `_get_layer_based_split_points()` | Calculates split points at layer boundaries |
| `split_by_layers()` | Main entry point for layer-based splitting |
| `save_layer_chunks()` | Saves with descriptive layer-based filenames |

### Architecture Prefixes Supported

| Model Type | Embed | Layers | Norm | LM Head |
|------------|-------|--------|------|---------|
| llama/qwen | model/embed | model/layers. | model/norm | lm_head |
| gpt2 | transformer/wte | transformer/h. | transformer/ln_f | lm_head |

---

## Test Results

### Weight-Based Splitting (2 chunks)

```bash
python split_local_model.py "path/to/model" --num-chunks 2
```

| Chunk | Nodes | Memory |
|-------|-------|--------|
| chunk_0.onnx | 29 | 318.9 MB |
| chunk_1.onnx | 548 | 335.6 MB |

### Layer-Based Splitting (7 layers per chunk)

```bash
python split_local_model.py "path/to/model" --strategy layer_based --layers-per-chunk 7
```

| Chunk | Layers | Memory |
|-------|--------|--------|
| chunk_layers_0_6.onnx | 0-6 | 533.2 MB |
| chunk_1.onnx | 7-13 | 33.8 MB |
| chunk_2.onnx | 14-20 | 33.8 MB |
| chunk_lm_head.onnx | 21-27 | 33.8 MB |

---

## Usage

### Weight-Based Splitting

```bash
# Basic 2-chunk split
python split_local_model.py "C:\path\to\model" --num-chunks 2

# 4-chunk split
python split_local_model.py "C:\path\to\model" -n 4
```

### Layer-Based Splitting

```bash
# 7 layers per chunk (default)
python split_local_model.py "C:\path\to\model" --strategy layer_based

# 1 layer per file (like grok_hf_split.py)
python split_local_model.py "C:\path\to\model" --strategy layer_based --layers-per-chunk 1

# 14 layers per chunk
python split_local_model.py "C:\path\to\model" -s layer_based -l 14
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--strategy, -s` | `weight_based` or `layer_based` |
| `--num-chunks, -n` | Number of chunks (weight-based) |
| `--layers-per-chunk, -l` | Layers per chunk (layer-based, default: 7) |
| `--output-dir, -o` | Custom output directory |
| `--verify` | Verify chunk correctness |
| `--analyze-only` | Only analyze, don't split |
| `--verbose, -v` | Enable verbose output |

---

## Programmatic Usage

```python
from local_model_splitter import LocalONNXModelSplitter

splitter = LocalONNXModelSplitter("path/to/model")

# Weight-based splitting
result = splitter.split_model(output_dir="./chunks", num_chunks=2)

# Layer-based splitting  
result = splitter.split_by_layers(output_dir="./layer_chunks", layers_per_chunk=7)

print(f"Created {len(result.chunk_models)} chunks")
for info in result.chunk_info:
    print(f"  Chunk {info.chunk_idx}: {info.node_count} nodes, {info.memory_mb:.1f} MB")
```

---

---

## Sequential Inference Pipeline

The `grok_inference_split_model_pipeline.py` enables memory-constrained inference by loading only one component at a time.

### Workflow
1. Load **Model Info** metadata.
2. Initialize **Tensor Registry** on CPU.
3. For each token:
   - Load component (e.g., `layer_0.onnx`) -> Execute -> Unload.
   - Sweep activations through the pipeline until `lm_head.onnx`.
4. Greedy-sample next token and repeat.

### Usage
```bash
python grok_inference_split_model_pipeline.py --model-dir "./split_output/my_model" --prompt "Once upon a time"
```

---

## Tool Comparison

| Tool | Format | Strategy | Inference Mode |
|------|--------|----------|----------------|
| `grok_hf_split.py` | SafeTensor | Semantic (Per Layer) | Merged Inference |
| `grok_onnx_split.py` | ONNX | Semantic (Per Layer) | Pipeline Supportive |
| `grok_inference_..._pipeline.py` | ONNX | Sequential | **Layer-by-Layer Swapping** |
| `local_model_splitter.py` | ONNX | Weight-based / Layer-Group | N/A |
