# ONNX Benchmark Report: 2026-01-29 23:53:01

**Model ID:** `None`

## Configuration
- **Split Directory:** `C:\Users\antmi\Downloads\onnx-model-chunking\split_output\qwen2.5-0.5b-v8\model.onnx`
- **Monolithic Model:** `C:\Users\antmi\.cache\huggingface\hub\models--amd--Qwen2.5-0.5B-Instruct-quantized_int4-float16-cpu-onnx\snapshots\7d826c831528165f5289bbb0975894b078f1da7b\model.onnx`
- **Runs per Prompt:** 2
- **Tokens generated:** 5

## Model Structure
- **Total ONNX Files:** 28
- **Total Model Size:** 34.1 MB

| Component Type | Count |
| :--- | :--- |
| Embeddings | 1 |
| Layers | 25 |
| Norm | 0 |
| Lm_head | 1 |
| Other | 1 |

## Results Summary

### Prompt: "The nature of artificial intelligence is"

| Metric | Monolith (Avg) | Split (Avg) | Improvement |
| :--- | :--- | :--- | :--- |
| Load Time | 0.2669s | 0.0020s | **+99.2%** |
| Peak Memory (RSS) | 128.3 MB | 29.7 MB | **+76.9%** |
| Total Latency | 0.00s | 2.42s | **-369026.5%** |
| Time To First Token | 0.1000s | 0.4702s | **-370.2%** |
| Overall TPS | 0.00 t/s | 2.07 t/s | **+0.0%** |
| Prefill Throughput | 60.0 t/s | 12.8 t/s | **-78.7%** |
| Decode Throughput | 0.0 t/s | 2.1 t/s | **+0.0%** |
| **Max Component Mem** | 128.3 MB | 65.5 MB | **+48.9%** |

> **Note:** Largest split component is `lm_head`.
