# Component Relationships

**Outcome Statement:** After reading this section, you will understand how each module interacts with others and where to extend the system.

---

## Module Dependency Graph

```
                                    ┌─────────────────┐
                                    │   Application   │
                                    │     Layer       │
                                    └────────┬────────┘
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              │                              │                              │
              ▼                              ▼                              ▼
    ┌─────────────────┐            ┌─────────────────┐            ┌─────────────────┐
    │    Splitting    │            │    Inference    │            │    Analytics    │
    │     Engine      │            │     Pipeline    │            │     System      │
    └────────┬────────┘            └────────┬────────┘            └────────┬────────┘
             │                              │                              │
             ▼                              ▼                              ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        Core Utilities Layer                             │
    │  chunking_utils/  │  pipeline_utils/  │  chunking_config.py  │  dtypes │
    └─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        External Dependencies                            │
    │    ONNX  │  ONNX Runtime  │  NumPy  │  Transformers  │  Safetensors   │
    └─────────────────────────────────────────────────────────────────────────┘
```

---

## Splitting Engine Modules

### grok_onnx_split.py

**Dependencies:** None (uses standard ONNX library)
**Dependents:** Inference pipeline, benchmarking tools

```
┌─────────────────────────────────────────────────────────────────┐
│                   grok_onnx_split.py                            │
├─────────────────────────────────────────────────────────────────┤
│  split_onnx_model(params)                                       │
│  │                                                               │
│  ├── detect_architecture_from_onnx(model)                      │
│  │   └── Uses: onnx library                                     │
│  │                                                               │
│  ├── extract_components(model, architecture)                   │
│  │   ├── create_component_info()                               │
│  │   └── identify_boundary_tensors()                           │
│  │                                                               │
│  ├── create_component_onnx(component_info)                     │
│  │   ├── build_subgraph()                                      │
│  │   ├── set_boundary_tensors()                                │
│  │   └── validate_and_save()                                   │
│  │                                                               │
│  └── Returns: SplitResult                                      │
│      ├── output_dir                                             │
│      ├── num_layers                                             │
│      ├── component_files                                        │
│      └── model_info                                             │
└─────────────────────────────────────────────────────────────────┘
```

**Integration Points:**
- Output directory is input to `SequentialModelPipeline`
- `model_info.json` is read by `SessionManager` for configuration

---

### chunking_utils/ Module

**Outcome:** Understand the modular splitting library structure.

```
chunking_utils/
├── __init__.py                 # Package exports
│
├── common/                     # Shared utilities
│   ├── __init__.py
│   ├── architecture.py         # Architecture detection (278 lines)
│   │   └── detect_architecture(model_path) → ArchitectureType
│   ├── types.py                # Shared data types (190 lines)
│   │   ├── ComponentInfo
│   │   ├── SplitResult
│   │   └── ArchitectureType enum
│   ├── path_utils.py           # Path sanitization
│   │   └── sanitize_path(), get_model_path()
│   └── hf_integration.py       # HuggingFace integration
│       └── load_hf_model(), save_hf_weights()
│
├── onnx/                       # ONNX-specific operations
│   ├── __init__.py
│   ├── types.py                # ONNX-specific types
│   ├── loader.py               # Model loading
│   │   ├── load_onnx_model()
│   │   └── load_external_data()
│   ├── splitter.py             # Core splitting logic (561 lines)
│   │   ├── ONNXSplitter class
│   │   ├── extract_subgraph()
│   │   └── create_component_model()
│   ├── merger.py               # Component merging (215 lines)
│   │   ├── merge_components()
│   │   └── reconcile_outputs()
│   └── inference.py            # ONNX inference helpers
│       └── run_onnx_inference()
│
└── safetensors/                # SafeTensors operations
    ├── __init__.py
    ├── config.py               # SafeTensors config
    ├── loader.py               # Weight loading
    ├── splitter.py             # SafeTensors splitting
    ├── merger.py               # SafeTensors merging
    └── inference.py            # SafeTensors inference
```

**Key Integration Pattern:**
```python
from chunking_utils import split_onnx_model, load_onnx_model
from chunking_utils.common import detect_architecture

# High-level API
result = split_onnx_model(model_path, output_dir)

# Or use modular components
model = load_onnx_model(model_path)
arch = detect_architecture(model)
# ... custom processing
```

---

## Inference Pipeline Modules

### SequentialModelPipeline

**Dependencies:** `SessionManager`, `KVCacheManager`, `TensorRegistry`
**Dependents:** Application code, benchmarking tools

```
┌──────────────────────────────────────────────────────────────────────┐
│              SequentialModelPipeline                                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  __init__(model_path, tokenizer_path, providers, persist_sessions)  │
│  │                                                                    │
│  ├── session_manager = SessionManager(model_path, config)           │
│  ├── kv_cache = KVCacheManager(num_layers, num_heads, head_dim)     │
│  ├── tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)      │
│  └── registry = {}  # Tensor registry                                │
│                                                                      │
│  generate(prompt, max_new_tokens, temperature)                       │
│  │                                                                    │
│  ├── tokenize_prompt(prompt) → input_ids                            │
│  │                                                                    │
│  ├── prefill(input_ids)                                             │
│  │   ├── hidden = run_component("embeddings", {"input_ids"})        │
│  │   ├── for layer in layers:                                       │
│  │   │       hidden, kv = run_component(f"layer_{layer}", ...)      │
│  │   │       kv_cache.update(layer, kv)                             │
│  │   ├── normalized = run_component("norm", {"hidden"})             │
│  │   └── logits = run_component("lm_head", {"normalized"})          │
│  │                                                                    │
│  └── decode(max_new_tokens)                                          │
│      └── Loop: generate one token at a time using KV cache          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

### SessionManager

**Dependencies:** ONNX Runtime, `SessionConfig`
**Dependents:** `SequentialModelPipeline`

```
┌──────────────────────────────────────────────────────────────────────┐
│                    SessionManager                                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  __init__(model_dir, session_config)                                 │
│  │                                                                    │
│  ├── load_component_metadata() → component_names, shapes            │
│  └── sessions = {}  # name → InferenceSession                       │
│                                                                      │
│  get_session(name)                                                   │
│  │                                                                    │
│  ├── Sequential Mode (persist=False):                               │
│  │   ├── Load from disk                                             │
│  │   ├── Create InferenceSession                                    │
│  │   ├── Cache temporarily                                          │
│  │   └── Return session                                             │
│  │                                                                    │
│  └── Persistent Mode (persist=True):                                │
│      └── Return cached session (pre-loaded at startup)              │
│                                                                      │
│  unload_session(name)                                                │
│  │                                                                    │
│  └── Delete session, free memory                                     │
│                                                                      │
│  get_memory_usage() → dict                                           │
│  │                                                                    │
│  └── Return: {loaded_count, total_count, total_mb}                  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Integration with TensorRegistry:**
```python
# Session manager handles session lifecycle
session = manager.get_session("layer_0")

# Tensor registry handles data passing
registry["hidden_states"] = session.run(inputs)["hidden_states"]

# Session can be unloaded after use
manager.unload_session("layer_0")
# Registry retains data for next component
```

---

### KVCacheManager

**Dependencies:** NumPy
**Dependents:** `SequentialModelPipeline`

```
┌──────────────────────────────────────────────────────────────────────┐
│                   KVCacheManager                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  __init__(num_layers, num_kv_heads, head_dim, dtype, batch_size)    │
│  │                                                                    │
│  └── cache = {                                                        │
│      layer_idx: {                                                     │
│          "key": numpy_array,     # [batch, seq, kv_heads, head_dim] │
│          "value": numpy_array                                         │
│      } for layer_idx in range(num_layers)                            │
│  }                                                                    │
│                                                                      │
│  update(layer_idx, key_tensor, value_tensor)                         │
│  │                                                                    │
│  └── Append new KV to existing cache for layer                      │
│                                                                      │
│  get(layer_idx) → (key, value)                                       │
│  │                                                                    │
│  └── Return KV cache for layer                                      │
│                                                                      │
│  reset()                                                             │
│  │                                                                    │
│  └── Clear all cached KV tensors                                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

### IOBindingManager

**Outcome:** Understand zero-copy device memory management.

```
┌──────────────────────────────────────────────────────────────────────┐
│                  IOBindingManager                                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  __init__(device_type="cuda")                                        │
│  │                                                                    │
│  └── device_buffers = {}  # name → device buffer                    │
│                                                                      │
│  bind_output(session, name, shape, dtype)                            │
│  │                                                                    │
│  ├── Allocate device buffer                                           │
│  ├── Bind to session output                                           │
│  └── Return binding                                                   │
│                                                                      │
│  get_device_buffer(name) → device_buffer                             │
│  │                                                                    │
│  └── Return device-resident buffer (no CPU copy)                    │
│                                                                      │
│  copy_to_device(numpy_array, name)                                   │
│  │                                                                    │
│  └── Copy numpy array to device buffer                              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Zero-Copy Flow:**
```
Without IO Binding:                  With IO Binding:
Component A output                   Component A output
    ↓ (Device → CPU copy)                ↓ (stays on device)
CPU numpy array                    Device Buffer
    ↓ (CPU → Device copy)                ↓ (zero-copy!)
Component B input                  Component B input
```

---

## Configuration System Relationships

### UnifiedChunkingConfig

```
┌──────────────────────────────────────────────────────────────────────┐
│               UnifiedChunkingConfig                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  method: ChunkingMethod  # onnx_graph_level | pytorch_level         │
│                                                                      │
│  onnx_graph_level: ONNXGraphLevelConfig                             │
│  │                                                                    │
│  ├── strategy: "weight_based" | "layer_based"                       │
│  ├── distribution: "equal_memory" | "equal_layers" | "custom"       │
│  ├── num_chunks: int                                                 │
│  ├── respect_layer_boundaries: bool                                  │
│  └── custom_chunk_sizes: List[int]                                   │
│                                                                      │
│  common: CommonChunkingConfig                                        │
│  │                                                                    │
│  ├── output_directory: str                                           │
│  ├── verify_chunks: bool                                             │
│  ├── min_psnr_threshold: float                                       │
│  └── save_metadata: bool                                             │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Configuration Processing:**
```python
from chunking_config import ConfigurationProcessor

processor = ConfigurationProcessor()

# Process extra_options from user
config = processor.process(extra_options)

# Validate configuration
is_valid, errors = processor.validate(config)

# Apply to model builder
model.apply_chunking_configuration(config)
```

---

## Analytics System Relationships

### ChunkingAnalytics Orchestrator

```
┌──────────────────────────────────────────────────────────────────────┐
│                 ChunkingAnalytics                                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  __init__(config)                                                    │
│  │                                                                    │
│  ├── system_monitor = SystemMonitor()                               │
│  ├── performance_timer = PerformanceTimer()                         │
│  ├── model_analyzer = ModelAnalyzer()                               │
│  └── alert_manager = AlertManager()                                 │
│                                                                      │
│  start_session()                                                     │
│  │                                                                    │
│  ├── Start background monitoring thread                              │
│  └── Initialize metrics collection                                   │
│                                                                      │
│  monitor_operation(name, phase)                                      │
│  │                                                                    │
│  ├── Context manager for timing                                      │
│  └── Collect metrics during operation                                │
│                                                                      │
│  end_session()                                                       │
│  │                                                                    │
│  ├── Stop monitoring                                                  │
│  ├── Generate reports                                                │
│  └── Save to output directory                                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

*For data flow diagrams, see [Data Flow](./data-flow.md).*
