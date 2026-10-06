# Architecture Overview

**Outcome Statement:** After reading this section, you will understand the complete system architecture, component relationships, and data flow through the splitting and inference pipeline.

---

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT: Model Source                              │
│        HuggingFace Hub  │  Local Safetensors  │  Local ONNX File        │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      SPLITTING ENGINE (Choose One)                      │
├─────────────────────────────────┬───────────────────────────────────────┤
│   ROOT-LEVEL BISECTION          │   MASTER_POC SEMANTIC SPLIT           │
│   builder.py                    │   grok_onnx_split.py                  │
│   - Weight-based split point    │   - Architecture detection            │
│   - Binary chunking             │   - Per-layer extraction              │
│   - PSNR verification           │   - Typed boundary handshakes         │
│   - 2 chunks output             │   - N+4 files output                  │
└─────────────────────────────────┴───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        SPLIT OUTPUT STRUCTURE                           │
│  ┌────────────┐ ┌─────────┐ ┌─────────────┐ ┌──────────┐ ┌──────────┐  │
│  │ embeddings │→│ layer_0 │→│ layer_1..N  │→│  norm    │→│ lm_head  │  │
│  │  (519 MB)  │ │(16 MB)  │ │ (16 MB each)│ │ (<1 MB)  │ │ (70 MB)  │  │
│  └────────────┘ └─────────┘ └─────────────┘ └──────────┘ └──────────┘  │
│                        + model_info.json (metadata)                     │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      INFERENCE PIPELINE                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  SequentialModelPipeline (grok_inference_split_model_pipeline) │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐ │   │
│  │  │ SessionManager  │  │  KVCacheManager  │  │ TensorRegistry │ │   │
│  │  │ - Load/Unload   │  │  - Prefill       │  │ - Shared state │ │   │
│  │  │ - Persistent    │  │  - Decode        │  │ - Zero-copy    │ │   │
│  │  │ - Lazy loading  │  │  - Update        │  │ - Cross-component││   │
│  │  └─────────────────┘  └──────────────────┘  └────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      EXECUTION MODES                                    │
├─────────────────┬─────────────────┬─────────────────────────────────────┤
│  🔄 SEQUENTIAL  │  📌 PERSISTENT  │  ⚡ IO_BOUND (Hybrid)               │
│  - Min memory   │  - All loaded   │  - Device buffers                  │
│  - Load/Unload │  - No I/O       │  - Zero-copy between components    │
│  - Disk I/O per │  - Fastest      │  - NPU/GPU optimized               │
└─────────────────┴─────────────────┴─────────────────────────────────────┘
```

---

## Component Layers

### Layer 1: Configuration

**Outcome:** Understand how configuration drives system behavior.

```
┌────────────────────────────────────────────────────────────────────────┐
│                     CONFIGURATION LAYER                                 │
│  chunking_config.py: UnifiedChunkingConfig, ConfigurationProcessor     │
│  - Method selection (pytorch_level vs onnx_graph_level)                │
│  - Analytics configuration                                              │
│  - Legacy format backward compatibility                                │
└────────────────────────────────────────────────────────────────────────┘
```

| Component | Purpose | Key Classes |
|-----------|---------|-------------|
| `chunking_config.py` | Unified configuration system | `UnifiedChunkingConfig`, `ConfigurationProcessor` |
| Configuration Options | Control splitting strategy | `ONNXGraphLevelConfig`, `PyTorchLevelConfig` |
| Validation | Ensure valid configurations | Type checking, range validation |

**Example Configuration:**
```python
extra_options = {
    "chunking": {
        "method": "onnx_graph_level",
        "onnx_graph_level": {
            "strategy": "weight_based",
            "distribution": "equal_memory",
            "num_chunks": 2
        },
        "common": {
            "output_directory": "./model_chunks",
            "verify_chunks": True,
            "min_psnr_threshold": 40.0
        }
    }
}
```

---

### Layer 2: Analytics

**Outcome:** Understand the monitoring and metrics infrastructure.

```
┌────────────────────────────────────────────────────────────────────────┐
│                     ANALYTICS LAYER                                     │
│  chunking_analytics.py: ChunkingAnalytics                              │
│  ├── SystemMonitor (CPU, memory, disk, GPU)                            │
│  ├── PerformanceTimer (phase-by-phase timing)                          │
│  ├── ModelAnalyzer (efficiency scoring)                                │
│  └── AlertManager (threshold-based alerts)                             │
└────────────────────────────────────────────────────────────────────────┘
```

| Component | Purpose | Metrics Collected |
|-----------|---------|-------------------|
| `SystemMonitor` | Real-time resource monitoring | CPU%, RAM GB, Disk I/O, GPU utilization |
| `PerformanceTimer` | High-precision timing | Phase durations, operation timing |
| `ModelAnalyzer` | Model-specific metrics | Size ratios, efficiency scores |
| `AlertManager` | Threshold-based alerts | Memory warnings, performance issues |

---

### Layer 3: Building

**Outcome:** Understand how ONNX models are constructed from source.

```
┌────────────────────────────────────────────────────────────────────────┐
│                     BUILDING LAYER                                      │
│  builder.py: Model (QwenModel, Qwen3Model)                             │
│  - ONNX graph construction from HF config                              │
│  - Layer-by-layer graph building                                       │
│  - Automatic chunking integration                                      │
│  - PSNR verification                                                   │
└────────────────────────────────────────────────────────────────────────┘
```

| Component | Purpose | Key Methods |
|-----------|---------|-------------|
| `builder.py` | Core model builder | `make_model()`, `create_chunked_models()` |
| Architecture Support | Model-specific builders | `QwenModel`, `Qwen3Model`, `LlamaModel` |
| Chunking Integration | Automatic chunking | `apply_chunking_configuration()` |

---

### Layer 4: Splitting

**Outcome:** Understand how models are decomposed into components.

```
┌────────────────────────────────────────────────────────────────────────┐
│                     SPLITTING LAYER                                     │
│  MASTER_POC/grok_onnx_split.py: split_onnx()                           │
│  ├── Architecture detection                                            │
│  ├── Component extraction (embed, layers, norm, lm_head)               │
│  ├── Subgraph creation with typed boundaries                          │
│  └── Metadata generation (model_info.json)                             │
└────────────────────────────────────────────────────────────────────────┘
```

| Component | Purpose | Key Functions |
|-----------|---------|---------------|
| `grok_onnx_split.py` | ONNX semantic splitter | `split_onnx_model()`, `detect_architecture()` |
| `grok_hf_split.py` | SafeTensors splitter | `split_hf_weights()` |
| `chunking_utils/` | Modular utilities | `splitter.py`, `merger.py`, `loader.py` |

---

### Layer 5: Runtime

**Outcome:** Understand the inference pipeline infrastructure.

```
┌────────────────────────────────────────────────────────────────────────┐
│                     RUNTIME LAYER                                       │
│  pipeline_utils/                                                       │
│  ├── session_manager.py: SessionManager, SessionConfig                 │
│  ├── kv_cache.py: KVCacheManager                                       │
│  ├── io_binding.py: IOBindingManager                                   │
│  ├── dtypes.py: Type handling                                          │
│  └── backend.py: Multi-backend abstraction                             │
└────────────────────────────────────────────────────────────────────────┘
```

| Component | Purpose | Key Classes |
|-----------|---------|-------------|
| `session_manager.py` | ONNX session lifecycle | `SessionManager`, `SessionConfig` |
| `kv_cache.py` | Autoregressive state | `KVCacheManager` |
| `io_binding.py` | Zero-copy I/O | `IOBindingManager`, `MemoryMode` |
| `dtypes.py` | Type conversion | INT4, FP16, BF16 handlers |
| `backend.py` | Backend abstraction | ONNX Runtime, llama.cpp, PyTorch |

---

## Data Flow: Splitting Pipeline

**Outcome:** Trace data flow from source model to split components.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Load Source Model                                              │
│                                                                         │
│  HuggingFace Safetensors ──→ Load We ━━━━━━━━┓                        │
│                                    eight      ┃                        │
│  Local ONNX File ──────────────→ Load Graph ━━╋──→ Model Object        │
│                                    + Weights  ┃                        │
└─────────────────────────────────────────────────┻━━━━━━━━━━━━━━━━━━━━━━━┛
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Detect Architecture                                            │
│                                                                         │
│  Scan node names for patterns:                                          │
│  - "model/embed" or "transformer/wte" → Embedding layer                │
│  - "model/layers." or "transformer/h." → Transformer blocks            │
│  - "model/norm" or "transformer/ln_f" → Final normalization            │
│  - "lm_head" → Language model head                                      │
│                                                                         │
│  Result: Architecture type (Llama/Qwen/GPT2) + Layer count             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Extract Components                                             │
│                                                                         │
│  For each semantic component:                                           │
│  1. Identify nodes belonging to component                               │
│  2. Collect initializer tensors (weights)                               │
│  3. Determine input/output boundary tensors                             │
│  4. Create ComponentInfo with metadata                                  │
│                                                                         │
│  Result: List of ComponentInfo objects                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 4: Create Subgraphs                                               │
│                                                                         │
│  For each ComponentInfo:                                                │
│  1. Create new ONNX graph with component nodes                          │
│  2. Add boundary tensors as inputs/outputs                              │
│  3. Copy initializers (weights)                                         │
│  4. Validate graph structure                                            │
│  5. Save to disk as standalone .onnx file                               │
│                                                                         │
│  Result: Directory of .onnx files + model_info.json                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Inference Pipeline

**Outcome:** Trace data flow from prompt to generated text.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PREFILL PHASE (Process Full Prompt)                                    │
│                                                                         │
│  Input: "The future of AI is"                                          │
│     │                                                                   │
│     ▼                                                                   │
│  ┌─────────────┐                                                       │
│  │ Tokenizer   │ → input_ids: [1, 5823, 29871, 3975, 3186]            │
│  └─────────────┘                                                       │
│     │                                                                   │
│     ▼                                                                   │
│  ┌─────────────┐                                                       │
│  │ Embeddings  │ → hidden: [batch, seq, hidden_dim]                   │
│  │ (Load→Run→Unload)                                                   │
│  └─────────────┘                                                       │
│     │                                                                   │
│     ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ For each layer (0 to N-1):                                       │   │
│  │   ┌─────────┐                                                   │   │
│  │   │ Layer N │ → hidden + KV_cache                              │   │
│  │   │ (Load→Run→Unload)                                           │   │
│  │   └─────────┘                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│     │                                                                   │
│     ▼                                                                   │
│  ┌─────────────┐                                                       │
│  │  Norm       │ → normalized_hidden                                  │
│  └─────────────┘                                                       │
│     │                                                                   │
│     ▼                                                                   │
│  ┌─────────────┐                                                       │
│  │  LM Head    │ → logits: [batch, seq, vocab]                        │
│  └─────────────┘                                                       │
│     │                                                                   │
│     ▼                                                                   │
│  Select last token logits → argmax → first generated token             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  DECODE PHASE (Generate Tokens One-by-One)                              │
│                                                                         │
│  For each new token:                                                    │
│  1. Use KV cache from prefill (no recomputation)                        │
│  2. Run layers with single-token input                                  │
│  3. Update KV cache with new keys/values                                │
│  4. Sample next token from logits                                       │
│  5. Repeat until max_tokens or EOS                                      │
│                                                                         │
│  Output: "The future of AI is [bright] [and] [holds] [great] [promise]"│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Interactions

### Session Manager and Tensor Registry

**Outcome:** Understand how components share data during inference.

```
┌────────────────────────────────────────────────────────────────────────┐
│                     TENSOR REGISTRY (Shared Memory)                     │
│                                                                         │
│  registry = {                                                           │
│      "hidden_states": numpy_array,      # Passed between layers        │
│      "attention_mask": numpy_array,     # Constant across layers       │
│      "position_ids": numpy_array,       # Position information         │
│      "kv_layer_0_key": numpy_array,     # KV cache for layer 0         │
│      "kv_layer_0_value": numpy_array,   │
│      ...                                                                │
│  }                                                                      │
└────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   SESSION MANAGER (Lifecycle Control)                   │
│                                                                         │
│  Sequential Mode:                                                       │
│  ├── get_session("layer_0") → Load from disk → Return session         │
│  ├── run_session(session, inputs) → Execute → Return outputs          │
│  └── unload_session("layer_0") → Free memory                           │
│                                                                         │
│  Persistent Mode:                                                       │
│  ├── load_all() → Pre-load all sessions at startup                     │
│  ├── get_session(name) → Return cached session (no load)              │
│  └── unload_all() → Free all sessions at once                          │
└────────────────────────────────────────────────────────────────────────┘
```

---

## File Organization

**Outcome:** Navigate the codebase structure.

```
onnx-model-chunking/
├── docs/                               # Consolidated documentation
├── MASTER_POC/                         # Core semantic split implementation
│   ├── grok_onnx_split.py              # ONNX semantic splitter
│   ├── grok_inference_split_model_pipeline.py  # Inference pipeline
│   ├── chunking_utils/                 # Modular splitting library
│   │   ├── common/                     # Shared utilities
│   │   ├── onnx/                       # ONNX operations
│   │   └── safetensors/                # SafeTensors operations
│   └── pipeline_utils/                 # Runtime infrastructure
│       ├── session_manager.py
│       ├── io_binding.py
│       ├── kv_cache.py
│       └── dtypes.py
├── chunking_config.py                  # Configuration system
├── chunking_analytics.py               # Analytics suite
└── builder.py                          # Model builder
```

---

*For detailed splitting techniques, see [Chunking Methods](../03-chunking-methods/README.md).*
