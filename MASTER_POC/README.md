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
| **[grok_onnx_split.py](file:///C:/Users/antmi/Downloads/onnx-model-chunking/MASTER_POC/grok_onnx_split.py)** | The splitter engine. Purges redundant data and creates valid ONNX subgraphs. |
| **[grok_inference_split_model_pipeline.py](file:///C:/Users/antmi/Downloads/onnx-model-chunking/MASTER_POC/grok_inference_split_model_pipeline.py)** | The sequential inference engine with dynamic hardware support. |
| **[benchmark_onnx_split_vs_whole.py](file:///C:/Users/antmi/Downloads/onnx-model-chunking/MASTER_POC/benchmark_onnx_split_vs_whole.py)** | The multi-provider performance and quality analysis suite. |

---

## 📖 Deployment Guide

### 1. Generate the Split Model
Run the splitter on your local monolithic ONNX file.
```bash
python MASTER_POC/grok_onnx_split.py --model "path/to/model.onnx" --output "split_output/qwen2.5-0.5b" --split-only
```

### 2. Multi-Provider Benchmarking (CPU vs VitisAI)
Compare performance, memory, and quality across different hardware engines.
```bash
python MASTER_POC/benchmark_onnx_split_vs_whole.py \
    --whole-model "path/to/monolith.onnx" \
    --split-dir "split_output/qwen2.5-0.5b/model.onnx" \
    --tokenizer-path "path/to/tokenizer" \
    --providers "CPUExecutionProvider,VitisAIExecutionProvider" \
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
- **TTFT (Latency)**: Time to first token (includes loading time).
- **TPS (Throughput)**: Tokens generated per second.
- **Peak RAM**: Highest RSS memory usage during the run.
- **Quality Match**: 1:1 text comparison between split and monolithic outputs (Greedy Equivalence).

### Verification Reports
- **[WALKTHROUGH_MULTI_PROVIDER.md](file:///C:/Users/antmi/Downloads/onnx-model-chunking/MASTER_POC/WALKTHROUGH_MULTI_PROVIDER.md)**: Detailed analysis of multi-provider results.
- **[verification_results_report.md](file:///C:/Users/antmi/Downloads/onnx-model-chunking/MASTER_POC/verification_results_report.md)**: Recorded audit of output integrity between Monolith and Split architectures.

## ⚙️ Requirements
- `onnxruntime` (or `onnxruntime-vitisai` for hardware acceleration)
- `transformers`
- `psutil`
- `numpy`
