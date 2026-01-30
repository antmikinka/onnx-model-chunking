# MASTER POC: Semantic ONNX Model Splitting & Multi-Provider Benchmarking

This directory contains the core implementation and analysis tools for the **Semantic ONNX Model Splitting** Proof of Concept. This architecture enables running Large Language Models (LLMs) on resource-constrained hardware by executing them as a sequence of independent, low-memory components.

## 🚀 Key Innovation: Sequential Model Execution
Instead of loading a multi-gigabyte monolith into RAM, our pipeline:
1.  **Decomposes** the model into semantic subgraphs (Embeddings, Transformer Layers, LM Head).
2.  **Loads/Unloads** each component on-the-fly during a single token generation step.
3.  **Minimizes Memory Footprint** to the size of the largest single component (typically Word Embeddings).

---

## 🛠️ Core Components

| File | Description |
| :--- | :--- |
| **[grok_onnx_split.py](./grok_onnx_split.py)** | The splitter engine. **Now supports Universal Transformer Architectures** (Llama, GPT2, Qwen, etc.) via dynamic regex detection. Purges redundant data and creates valid ONNX subgraphs. |
| **[grok_inference_split_model_pipeline.py](./grok_inference_split_model_pipeline.py)** | The sequential inference engine with dynamic hardware support. Manages KV cache statefully across component loads. |
| **[benchmark_onnx_split_vs_whole.py](./benchmark_onnx_split_vs_whole.py)** | The comprehensive performance analysis suite. **Features "Fair comparison" logic**: Upgraded Monolithic runner to support **Stateful Execution (KV Cache)**, ensuring latency comparisons against the Split model are valid (Stateful vs Stateful). |

---

## 📖 Deployment Guide

### 1. Generate the Split Model
Run the splitter on your local monolithic ONNX file. It auto-detects architecture (Llama, GPT2, Qwen).
```bash
python MASTER_POC/grok_onnx_split.py --model "path/to/model.onnx" --output "split_output/qwen2.5-0.5b" --split-only
```

### 2. Multi-Provider Benchmarking
Compare performance, memory, and quality across different hardware engines (CPU, CUDA, VitisAI, etc.).
```bash
python MASTER_POC/benchmark_onnx_split_vs_whole.py \
    --whole-model "path/to/monolith.onnx" \
    --split-dir "split_output/qwen2.5-0.5b/model.onnx" \
    --tokenizer-path "path/to/tokenizer" \
    --providers "CPUExecutionProvider" \
    --runs 3 \
    --tokens 10 \
    --output "MASTER_POC/final_bench.json"
```

### 3. Standalone Sequential Inference
Run only the split model inference loop.
```bash
python MASTER_POC/grok_inference_split_model_pipeline.py \
    --model-dir "split_output/qwen2.5-0.5b/model.onnx" \
    --prompt "What is the future of AI?" \
    --tokens 20
```

---

## 📊 Quality & Performance Metrics

### Matrix Comparison Table
The benchmark suite generates a matrix in the terminal and a Markdown report (`*_report.md`) that tracks:
- **TTFT (Latency)**: Time to first token.
- **TPS (Throughput)**: Tokens generated per second.
- **Peak RAM**: Highest RSS memory usage during the run.
- **Quality Match**: 1:1 text comparison between split and monolithic outputs (Greedy Equivalence).

### Verification methodology
- **Triple Check Logic**: The benchmark script includes explicit verification steps to ensure metrics like `total_time` and `decode_time` are calculated correctly.
- **Stateful Baseline**: Monolithic benchmarks are run with full KV-cache support to ensure the speedup metrics for Split models reflect *loading efficiency* gains vs *loading overhead* losses, rather than algorithmic differences.

## ⚙️ Requirements
- `onnxruntime` (or `onnxruntime-gpu`, `onnxruntime-vitisai`)
- `transformers`
- `psutil`
- `numpy`
