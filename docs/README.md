# ONNX Model Chunking System - Technical Documentation

**Version:** 2.0.0
**Last Updated:** 2026-03-21
**Repository:** https://github.com/antmikinka/onnx-model-chunking

---

## Outcome Statement

This documentation system provides a comprehensive, technically accurate reference for deploying large language models on memory-constrained hardware through semantic model splitting and sequential inference. After reading, you will understand how to run models 2-4x larger than available physical RAM while maintaining mathematical correctness.

---

## Quick Navigation

| Section | Purpose | Key Content |
|---------|---------|-------------|
| [01-getting-started](./01-getting-started/README.md) | Installation and first use | Setup, quick start, basic concepts |
| [02-architecture-overview](./02-architecture-overview/README.md) | System design | Architecture diagrams, component relationships |
| [03-chunking-methods](./03-chunking-methods/README.md) | Splitting techniques | Semantic splitting, bisection, SafeTensors |
| [04-inference-pipelines](./04-inference-pipelines/README.md) | Running split models | Sequential, persistent, and IO-bound modes |
| [05-hardware-deployment](./05-hardware-deployment/README.md) | Hardware targeting | CPU, GPU, NPU deployment strategies |
| [06-analytics-reporting](./06-analytics-reporting/README.md) | Performance monitoring | Metrics, benchmarking, reports |
| [07-appendix](./07-appendix/README.md) | Technical reference | APIs, troubleshooting, detailed specs |

---

## Executive Summary

### The Core Problem

Running a Large Language Model (LLM) normally requires loading the **entire model into RAM simultaneously**:

```
Load ENTIRE model (500MB - 100GB) → Compute → Generate output
Required RAM: 2-4x model size
```

**Example:** A 0.5B parameter model needs ~2-4GB RAM for monolithic inference. A 7B model needs 14-28GB. Most consumer devices have 8-16GB total.

### The Solution: Semantic Model Splitting

This system decomposes monolithic models into **semantic components** that execute sequentially:

```
Monolithic: [Embed + Layer_0 + Layer_1 + ... + Layer_N + Norm + LM_Head]
                 Peak RAM: 3-4GB

Split: [Embed] → [Layer_0] → [Layer_1] → ... → [Norm] → [LM_Head]
       Peak RAM: ~100-200MB (largest single component)
```

### Key Achievement

**Run a 62GB model on 32GB RAM** with mathematical correctness preserved (PSNR verified).

---

## System Capabilities

| Capability | Description | Status |
|------------|-------------|--------|
| **Semantic Splitting** | Decompose models by architectural component (per-layer) | Production-ready |
| **Graph Bisection** | Binary split at weight distribution midpoint | Production-ready |
| **Sequential Inference** | Load/run/unload components one at a time | Production-ready |
| **Persistent Mode** | Pre-load all sessions for low-latency inference | Production-ready |
| **IO Binding** | Zero-copy device-resident tensor passing | Production-ready |
| **Multi-Backend** | ONNX Runtime, llama.cpp, PyTorch abstraction | Infrastructure-ready |
| **NPU Acceleration** | AMD XDNA NPU support with hybrid dispatch | Production-ready |
| **Analytics Suite** | Real-time monitoring, timing, reports | Production-ready |

---

## Memory Savings by Method

| Method | Memory Reduction | Best Use Case |
|--------|-----------------|---------------|
| ONNX Bisection | ~50% | Simple 2-way splits |
| Semantic Split | 60-95% (peak = largest component) | Memory-constrained devices |
| INT4 Quantization | ~75% size reduction | Combined with splitting |
| **Combined (INT4 + Semantic)** | **Run 62GB on 32GB RAM** | Maximum efficiency |

---

## Supported Model Architectures

| Architecture Family | Specific Models | Status |
|---------------------|-----------------|--------|
| **LLaMA Family** | LLaMA, Mistral, Phi, Gemma | Fully supported |
| **Qwen Family** | Qwen, Qwen2, Qwen2.5, Qwen3 | Fully supported |
| **GPT-2 Family** | GPT-2, GPT-J | Fully supported |
| **Mixture of Experts** | Mixtral, Grok | Infrastructure-ready |

Architecture detection is automatic via pattern matching on ONNX node names.

---

## Documentation Structure

```
docs/
├── README.md                              # This index file
├── 01-getting-started/
│   ├── README.md                          # Installation and setup
│   ├── quick-start.md                     # 5-minute tutorial
│   └── core-concepts.md                   # Fundamental concepts
├── 02-architecture-overview/
│   ├── README.md                          # System design overview
│   ├── component-relationships.md         # Module interactions
│   └── data-flow.md                       # Tensor flow diagrams
├── 03-chunking-methods/
│   ├── README.md                          # Splitting techniques index
│   ├── semantic-splitting.md              # Per-layer decomposition
│   ├── graph-bisection.md                 # Binary weight-based splitting
│   └── safetensors-splitting.md           # HuggingFace weight splitting
├── 04-inference-pipelines/
│   ├── README.md                          # Pipeline modes index
│   ├── sequential-mode.md                 # Memory-optimized inference
│   ├── persistent-mode.md                 # Latency-optimized inference
│   └── io-bound-mode.md                   # Zero-copy device inference
├── 05-hardware-deployment/
│   ├── README.md                          # Hardware targeting index
│   ├── cpu-deployment.md                  # CPU-only deployment
│   ├── gpu-deployment.md                  # CUDA/ROCm/DirectML
│   └── npu-deployment.md                  # AMD XDNA NPU (critical guidance)
├── 06-analytics-reporting/
│   ├── README.md                          # Analytics system index
│   ├── performance-metrics.md             # Timing and resource metrics
│   └── report-generation.md               # JSON/CSV/HTML reports
└── 07-appendix/
    ├── README.md                          # Appendix index
    ├── api-reference.md                   # Complete API documentation
    ├── npu-monolithic-requirement.md      # Why NPU requires monolithic models
    ├── decision-matrix.md                 # Hardware selection guide
    ├── troubleshooting.md                 # Common issues and solutions
    └── glossary.md                        # Technical terminology
```

---

## Related Documentation

Legacy documentation files (still valid but superseded by this consolidated system):

| File | Purpose |
|------|---------|
| `MASTER_FINDINGS.md` | Comprehensive findings and tradeoff analysis |
| `MASTER_POC/MASTER_REFERENCE.md` | Original technical reference (now with graphs and glossary) |
| `MASTER_POC/NPU_GUIDE.md` | NPU integration guide (source for hardware section) |
| `CHUNKING_DOCUMENTATION.md` | Original chunking guide |
| `IMPLEMENTATION_SUMMARY.md` | Phase 1 implementation details |
| `ANALYTICS_SYSTEM_SUMMARY.md` | Analytics system documentation |
| `WALKTHROUGH.md` | Original usage walkthrough |

---

## Quick Glossary

**Outcome:** Understand key technical terms used throughout this documentation.

| Term | Definition |
|------|------------|
| **AOT Compilation** | Ahead-of-Time compilation (5-50ms) - NPU uses this, making split models slow |
| **Boundary Tensor** | Input/output tensors at component interfaces, explicitly typed to prevent errors |
| **Component** | Semantic unit (embedding, layer, norm, lm_head) as standalone ONNX model |
| **Context Switch** | Overhead when switching sessions: GPU ~10μs, NPU ~5-50ms (500-5000× slower) |
| **Hybrid Dispatch** | Routing different components/phases to different accelerators (NPU+iGPU) |
| **IO Binding** | Zero-copy tensor transfers between components on same device |
| **KV Cache** | Cached attention key/value tensors for autoregressive inference (~64MB typical) |
| **Monolithic** | Traditional approach loading entire model at once |
| **NPU** | Neural Processing Unit (AMD XDNA) - requires monolithic models only |
| **Persistent Mode** | All sessions pre-loaded at startup for low-latency inference |
| **PSNR** | Peak Signal-to-Noise Ratio - accuracy verification (>40dB = production-ready) |
| **Semantic Splitting** | Per-layer decomposition at architectural boundaries (the proven approach) |
| **Sequential Mode** | Load/run/unload one component at a time - minimum memory, higher latency |
| **Tensor Registry** | Shared dictionary passing tensors between split components |
| **TPS** | Tokens Per Second - inference throughput metric |
| **TTFT** | Time To First Token - latency metric including prefill phase |

> For complete glossary with detailed definitions, see: [`MASTER_POC/MASTER_REFERENCE.md`](./MASTER_POC/MASTER_REFERENCE.md#glossary)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2026-03-21 | Consolidated documentation system with outcome statements |
| 1.5.0 | 2026-02-26 | NPU hybrid dispatch and multi-device support |
| 1.0.0 | 2026-02-17 | Initial semantic splitting POC |

---

*Documentation generated by the ONNX Model Chunking System project.*
