# Glossary - Technical Terminology

**Outcome Statement:** Understand the technical terminology used throughout this documentation.

---

## A

### AOT (Ahead-of-Time Compilation)
Compilation that occurs before program execution, producing a complete executable image. NPU uses AOT compilation, which requires the entire computation graph upfront. **See:** [NPU Monolithic Requirement](./npu-monolithic-requirement.md)

### Architecture Detection
The process of automatically identifying model architecture type (LLaMA, Qwen, GPT-2, etc.) by scanning ONNX node names for characteristic patterns.

### Autoregressive
Property of language models where each token is generated based on previously generated tokens. Requires KV-cache for efficient inference.

---

## B

### Batch Size
Number of sequences processed in parallel. Most memory-constrained scenarios use batch_size=1.

### Boundary Tensor
Tensors that cross component boundaries during sequential inference. Must be explicitly typed to prevent ONNX Runtime auto-cast errors.

---

## C

### Component
A semantic unit of a split model (e.g., embeddings, layer_0, norm, lm_head). Each component is a standalone, valid ONNX model.

### Context Switch
The overhead of switching from one execution context to another. GPU: ~10 μs, NPU: ~5-50 ms (500-5000× more expensive).

### CUDA
NVIDIA's parallel computing platform and API. Supported via `onnxruntime-gpu` package.

---

## D

### Decode Phase
The token-by-token generation phase of LLM inference. Memory-bandwidth intensive, benefits from KV-cache.

### DirectML
Microsoft's hardware-accelerated computation API. Supports AMD, NVIDIA, and Intel GPUs via `onnxruntime-directml`.

---

## E

### Embedding
The first component of a transformer model. Converts token IDs to dense vector representations (hidden states).

---

## F

### FP16
16-bit floating point (half precision). Reduces memory usage by 50% vs FP32 with minimal accuracy loss.

### FP32
32-bit floating point (single precision). Standard precision for most inference.

---

## G

### Graph Bisection
Splitting technique that divides the ONNX graph at the weight distribution midpoint, creating two roughly equal chunks.

### GPU (Graphics Processing Unit)
Massively parallel processor optimized for graphics and compute workloads. Ideal for LLM inference due to high memory bandwidth and JIT compilation.

### GGUF
Quantized model format used by llama.cpp. Not directly compatible with ONNX Runtime.

---

## H

### Hidden Size
The dimension of transformer hidden states (e.g., 896 for Qwen2.5-0.5B). Determines memory usage per token.

### HuggingFace
Popular platform and library for transformer models. Models can be exported to ONNX format.

---

## I

### INT4
4-bit integer quantization. Reduces model size by ~75% vs FP16 with minimal accuracy loss.

### IO-Bound Mode
Execution mode that uses device-resident buffers for zero-copy tensor passing between components. Fastest mode, requires GPU/NPU.

### IO Binding
Technique for binding ONNX session inputs/outputs to device buffers, eliminating CPU↔Device copies.

---

## J

### JIT (Just-in-Time Compilation)
Compilation that occurs during program execution. GPU uses JIT compilation, enabling flexible subgraph execution.

---

## K

### KV-Cache
Key-Value cache for autoregressive attention. Stores attention keys and values from previous tokens to avoid recomputation. Memory usage: ~2 × layers × kv_heads × head_dim × sequence_length bytes.

---

## L

### Layer
A transformer block containing attention and feed-forward sublayers. Typical models have 12-80 layers.

### LM Head (Language Model Head)
The final component that converts hidden states to vocabulary logits. Includes linear projection and optional softmax.

### LLaMA
Meta's family of open language models. Supported architecture in this system.

---

## M

### Memory Mode
Execution strategy for memory management:
- **Sequential:** Load/run/unload per component (minimum memory)
- **Persistent:** All sessions pre-loaded (low latency)
- **IO-Bound:** Persistent + device buffers (zero-copy)

### Model
A neural network represented as an ONNX computation graph.

### Monolithic
A model that is NOT split - entire graph in single file. Required for NPU deployment.

---

## N

### Node
An operator in an ONNX graph (e.g., MatMul, Add, Softmax).

### Norm (Normalization)
Final normalization layer before LM head. Typically RMSNorm or LayerNorm.

### NPU (Neural Processing Unit)
Specialized AI accelerator. AMD XDNA NPU (Ryzen AI) uses VitisAI execution provider. **Requires monolithic models.**

---

## O

### ONNX (Open Neural Network eXchange)
Open format for representing machine learning models. Enables cross-framework interoperability.

### ONNX Runtime (ORT)
High-performance inference engine for ONNX models. Supports CPU, GPU, and NPU execution providers.

---

## P

### Persistent Mode
Execution mode where all component sessions are pre-loaded at startup and kept resident. Faster than sequential, requires more memory.

### Prefill Phase
The initial phase of LLM inference where the full prompt is processed. Compute-intensive, benefits from parallel execution.

### Provider (Execution Provider)
Hardware backend for ONNX Runtime. Options: CPU, CUDA, ROCm, DirectML, VitisAI.

### PSNR (Peak Signal-to-Noise Ratio)
Metric for measuring accuracy between original and split model outputs. >35 dB = good, >40 dB = production-ready.

---

## Q

### Qwen
Alibaba's family of open language models. Fully supported architecture in this system.

### Quantization
Technique for reducing model precision (FP32→FP16→INT8→INT4) to reduce size and improve performance.

---

## R

### Registry (Tensor Registry)
Shared memory structure that passes tensors between components during sequential inference.

---

## S

### SafeTensors
HuggingFace's weight format. Can be split per-layer for distribution, but requires merging for ONNX inference.

### Semantic Splitting
Splitting technique that decomposes models at architectural boundaries (per-layer). Provides maximum memory savings.

### Sequential Mode
Execution mode where components are loaded, run, and unloaded one at a time. Minimum memory, slower speed.

### Session
An ONNX Runtime inference session. Represents a loaded, ready-to-run model.

### Split Ratio
For bisection: the fraction of total weight size for the first chunk (0.0-1.0). Default: 0.5 (50/50 split).

---

## T

### Tile SRAM
On-chip NPU memory (16-32 MB on AMD XDNA). Ultra-fast, but limited capacity.

### Token
The basic unit of text for LLMs (subword units). Typical English text: ~1.3 tokens per word.

### Tokenizer
Converts between text and token IDs. HuggingFace tokenizers are commonly used.

---

## V

### VitisAI
AMD's execution provider for NPU acceleration. Requires monolithic models.

### VRAM (Video RAM)
GPU-dedicated memory. Typical consumer GPUs: 4-24 GB. High-end/datacenter: 40-80 GB.

---

## Z

### Zero-Copy
Data transfer technique where memory is accessed directly without intermediate copies. IO-binding enables zero-copy between components on GPU/NPU.

---

## Abbreviations

| Abbreviation | Meaning |
|--------------|---------|
| AOT | Ahead-of-Time (compilation) |
| API | Application Programming Interface |
| CPU | Central Processing Unit |
| DDR | Double Data Rate (RAM type) |
| DML | DirectML (Microsoft's API) |
| GPU | Graphics Processing Unit |
| HF | HuggingFace |
| IO | Input/Output |
| JIT | Just-in-Time (compilation) |
| KV | Key-Value (cache) |
| LLM | Large Language Model |
| MoE | Mixture of Experts |
| NPU | Neural Processing Unit |
| ONNX | Open Neural Network eXchange |
| ORT | ONNX Runtime |
| PSNR | Peak Signal-to-Noise Ratio |
| RAM | Random Access Memory |
| RMSNorm | Root Mean Square Layer Normalization |
| SRAM | Static RAM (on-chip memory) |
| VRAM | Video RAM |
| XDNA | AMD's NPU architecture |

---

*For usage examples, see related documentation sections throughout this system.*
