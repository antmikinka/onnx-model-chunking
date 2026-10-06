# MASTER DOCUMENT SPECIFICATION
## ONNX Model Chunking Repository - Comprehensive Technical Analysis

**Document Version:** 1.0.0
**Generated:** 2026-03-17
**Repository:** https://github.com/antmikinka/onnx-model-chunking.git
**Analysis Scope:** Complete codebase deep-dive with NPU integration gap analysis

---

## TABLE OF CONTENTS

1. [Executive Summary](#executive-summary)
2. [Repository Structure](#repository-structure)
3. [Architecture Overview](#architecture-overview)
4. [Code Path Analysis](#code-path-analysis)
5. [Component Specification](#component-specification)
6. [Data Flow Documentation](#data-flow-documentation)
7. [API Reference](#api-reference)
8. [Class Hierarchies](#class-hierarchies)
9. [Known Limitations and Gaps](#known-limitations-and-gaps)
10. [NPU Integration Gap Analysis](#npu-integration-gap-analysis)
11. [Priority Ranking](#priority-ranking)

---

## EXECUTIVE SUMMARY

### What This Repository Does

This repository implements **semantic model splitting** - the ability to decompose large neural network models into smaller, independently executable components. The core insight is that a neural network is a chain of mathematical operations that don't all need to be in memory simultaneously.

**Key Achievement:** Proven ability to run models 2-4x larger than available physical RAM by loading components sequentially.

### Two Distinct Codebases

This repository contains **two fundamentally different approaches** to model chunking:

| Aspect | Root-Level (Bisection) | MASTER_POC (Semantic Split) |
|--------|------------------------|----------------------------|
| **Approach** | Binary graph bisection at weight midpoint | Semantic decomposition by layer/component |
| **Location** | Root directory files | `MASTER_POC/` subdirectory |
| **Output** | 2 chunks (first_chunk, second_chunk) | N+4 files (embed, layer_0..N, norm, lm_head) |
| **Split Logic** | Weight distribution analysis | Architecture-aware pattern matching |
| **Inference** | Sequential: chunk1 -> chunk2 | Sequential: embed -> layers -> norm -> head |
| **Verification** | PSNR-based accuracy checking | Mathematical equivalence by design |
| **Status** | Working, tested | Working POC with full inference pipeline |

### Supported Architectures

- **LLaMA family:** LLaMA, Mistral, Phi, Gemma
- **Qwen family:** Qwen, Qwen2, Qwen3
- **GPT-2 family:** GPT-2
- **Extensible:** Architecture detection system allows easy addition

### Memory Savings Achieved

| Method | Memory Reduction | Best For |
|--------|-----------------|----------|
| ONNX Bisection | ~50% | Simple 2-way split |
| Semantic Split | 60-95% (peak = largest component) | Memory-constrained inference |
| INT4 Quantization | ~75% size reduction | Combined with splitting |
| Combined (INT4 + Semantic) | Run 62GB model on 32GB RAM | Maximum memory efficiency |

---

## REPOSITORY STRUCTURE

```
onnx-model-chunking/
├── 📄 README.md                          # Project overview
├── 📄 MASTER_FINDINGS.md                 # Comprehensive findings document
├── 📄 MASTER-POC.md                      # POC working concept
├── 📄 CHUNKING_DOCUMENTATION.md          # Root-level chunking guide
├── 📄 IMPLEMENTATION_SUMMARY.md          # Phase 1 implementation summary
├── 📄 ANALYTICS_SYSTEM_SUMMARY.md        # Analytics system documentation
├── 📄 UNIFIED_CONFIG_GUIDE.md            # Configuration system guide
├── 📄 ANALYTICS_REPORTING_README.md      # Analytics reporting guide
├── 📄 QWEN2_COMPLETE_ANALYTICS_EXECUTION_SUMMARY.md
├── 📄 ENHANCED_ANALYTICS_IMPLEMENTATION_COMPLETE.md
├── 📄 ONNX_IR_COMPATIBILITY_FIXES.md     # Compatibility fixes
├── 📄 QUALITY_ASSESSMENT_REPORT.md       # Quality assessment
├── 📄 provider_test_report.md            # Provider testing results
├── 📄 verification_results_report.md     # Verification results
│
├── 🔧 ROOT-LEVEL (ORIGINAL BISECTION SYSTEM)
│   ├── builder.py                        # Core model builder (352KB, ~9000 lines)
│   ├── chunking_config.py                # Unified configuration (776 lines)
│   ├── chunking_analytics.py             # Performance analytics (1172 lines)
│   ├── chunk_mlprogram.py                # CoreML chunking reference (464 lines)
│   ├── quality_assurance_system.py       # QA verification (1100 lines)
│   ├── analytics_reporting.py            # Report generation (950 lines)
│   ├── enhanced_analytics_integration.py # Analytics orchestrator (1400 lines)
│   ├── local_model_splitter.py           # Local model splitting (1600 lines)
│   │
│   ├── 🧪 Test Files
│   │   ├── test_chunking_functionality.py    # Comprehensive test suite (1800 lines)
│   │   ├── test_unified_config.py            # Config testing (300 lines)
│   │   ├── test_analytics_system.py          # Analytics testing (550 lines)
│   │   ├── test_enhanced_analytics.py        # Enhanced analytics tests
│   │   ├── test_enhanced_integration.py      # Integration tests
│   │   ├── test_onnx_ir_compatibility.py     # Compatibility tests
│   │   ├── simple_analytics_test.py          # Basic analytics tests
│   │   ├── simple_config_demo.py             # Config demonstration
│   │   ├── simple_enhanced_test.py           # Simple enhancement tests
│   │   ├── simple_test.py                    # Basic functionality tests
│   │   ├── demo_unified_config.py            # Unified config demo
│   │   ├── demo_enhanced_analytics.py        # Analytics demo
│   │   ├── demo_complete_enhanced_analytics.py  # Complete demo
│   │   ├── example_chunking_usage.py         # Usage examples
│   │   ├── enhanced_quality_validation.py    # Quality validation
│   │   └── verify_chunk_structure.py         # Structure verification
│   │
│   └── 🧪 Benchmark/Execution Files
│       ├── benchmark_qwen_conversion.py      # Qwen benchmark (700 lines)
│       ├── qwen2_real_conversion_analytics.py  # Qwen2 execution
│       ├── qwen2_simplified_conversion_analytics.py  # Simplified execution
│       ├── corrected_test_chunking.py        # Corrected tests
│       └── test_chunking_fix.py              # Fix validation
│
└── 🏆 MASTER_POC/ (SEMANTIC SPLIT SYSTEM)
    ├── 📄 README.md                          # MASTER_POC overview
    ├── 📄 WALKTHROUGH.md                     # Code walkthrough
    ├── 📄 WALKTHROUGH_MULTI_PROVIDER.md      # Multi-provider guide
    │
    ├── 🔧 Core Splitting Engines
    │   ├── grok_onnx_split.py                # ONNX semantic splitter (1029 lines)
    │   ├── grok_hf_split.py                  # HuggingFace/SafeTensors splitter (414 lines)
    │   ├── grok_split_v3.py                  # Iterative version 3 (850 lines)
    │   ├── grok_split_v4.py                  # Iterative version 4 (920 lines)
    │   └── split_qwen3_8b_awq.py             # Qwen3 AWQ-specific splitter
    │
    ├── 🔧 Inference Pipeline
    │   ├── grok_inference_split_model_pipeline.py  # Sequential inference (638 lines)
    │   ├── test_pipeline_logic.py            # Pipeline logic tests
    │   ├── test_pipeline_utils.py            # Pipeline utility tests
    │   └── test_single_step.py               # Single-step tests
    │
    ├── 🔧 Benchmarking
    │   ├── benchmark_onnx_split_vs_whole.py  # Split vs whole benchmark (946 lines)
    │   ├── check_model_metadata.py           # Metadata inspection
    │   ├── check_domain.py                   # Domain checking
    │   ├── inspect_lm_head.py                # LM head inspection
    │   └── inspect_matmul.py                 # MatMul inspection
    │
    ├── 📦 chunking_utils/ (Modular Library)
    │   ├── __init__.py                       # Package exports
    │   ├── common/                           # Shared utilities
    │   │   ├── __init__.py                   # Common exports
    │   │   ├── architecture.py               # Architecture detection (278 lines)
    │   │   ├── types.py                      # Shared data types (190 lines)
    │   │   ├── path_utils.py                 # Path sanitization
    │   │   └── hf_integration.py             # HuggingFace integration
    │   │
    │   ├── onnx/                             # ONNX-specific operations
    │   │   ├── __init__.py                   # ONNX exports
    │   │   ├── types.py                      # ONNX-specific types
    │   │   ├── loader.py                     # Model loading
    │   │   ├── splitter.py                   # ONNX splitting (561 lines)
    │   │   ├── merger.py                     # ONNX merging (215 lines)
    │   │   └── inference.py                  # ONNX inference
    │   │
    │   └── safetensors/                      # SafeTensors operations
    │       ├── __init__.py                   # SafeTensors exports
    │       ├── config.py                     # SafeTensors config
    │       ├── loader.py                     # Weight loading
    │       ├── splitter.py                   # SafeTensors splitting
    │       ├── merger.py                     # SafeTensors merging
    │       └── inference.py                  # SafeTensors inference
    │
    └── 📦 pipeline_utils/ (Runtime Infrastructure)
        ├── __init__.py                       # Package exports
        ├── session_manager.py                # Session lifecycle (501 lines)
        ├── io_binding.py                     # Zero-copy I/O binding
        ├── kv_cache.py                       # KV cache management (263 lines)
        ├── dtypes.py                         # Type handling (200+ lines)
        ├── backend.py                        # Multi-backend abstraction
        ├── config_loader.py                  # Model config loading
        └── kv_cache.py                       # KV cache utilities
```

### File Statistics

| Category | File Count | Total Lines (est.) |
|----------|------------|-------------------|
| Root-level core | 7 | ~6,500 |
| Root-level tests | 15 | ~5,000 |
| MASTER_POC core | 5 | ~3,500 |
| chunking_utils | 15 | ~2,000 |
| pipeline_utils | 7 | ~1,500 |
| Documentation | 20+ | ~3,000 |
| **TOTAL** | **~70** | **~21,500+** |

---

## ARCHITECTURE OVERVIEW

### System Architecture Diagram

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

### Component Relationships

```
┌────────────────────────────────────────────────────────────────────────┐
│                     CONFIGURATION LAYER                                 │
│  chunking_config.py: UnifiedChunkingConfig, ConfigurationProcessor     │
│  - Method selection (pytorch_level vs onnx_graph_level)                │
│  - Analytics configuration                                              │
│  - Legacy format backward compatibility                                │
└────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     ANALYTICS LAYER                                     │
│  chunking_analytics.py: ChunkingAnalytics                              │
│  ├── SystemMonitor (CPU, memory, disk, GPU)                            │
│  ├── PerformanceTimer (phase-by-phase timing)                          │
│  ├── ModelAnalyzer (efficiency scoring)                                │
│  └── AlertManager (threshold-based alerts)                             │
└────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     BUILDING LAYER                                      │
│  builder.py: Model (QwenModel, Qwen3Model)                             │
│  - ONNX graph construction from HF config                              │
│  - Layer-by-layer graph building                                       │
│  - Automatic chunking integration                                      │
│  - PSNR verification                                                   │
└────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     SPLITTING LAYER                                     │
│  MASTER_POC/grok_onnx_split.py: split_onnx()                           │
│  ├── Architecture detection                                            │
│  ├── Component extraction (embed, layers, norm, lm_head)               │
│  ├── Subgraph creation with typed boundaries                          │
│  └── Metadata generation (model_info.json)                             │
└────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     RUNTIME LAYER                                       │
│  pipeline_utils/                                                       │
│  ├── session_manager.py: SessionManager, SessionConfig                 │
│  ├── kv_cache.py: KVCacheManager                                       │
│  ├── io_binding.py: IOBindingManager                                   │
│  ├── dtypes.py: Type handling utilities                                │
│  └── config_loader.py: Model configuration loading                     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## CODE PATH ANALYSIS

### Execution Flow: Root-Level Bisection

```
User calls builder.py with extra_options
         │
         ▼
Model.__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
         │
         ├── Line 88: self.config_processor = ConfigurationProcessor()
         │
         ├── Line 92: self.chunking_config = self.config_processor.process_configuration(extra_options)
         │              │
         │              ├── ConfigurationProcessor.process_configuration()
         │              │   ├── _is_legacy_format() → legacy_mode flag
         │              │   ├── _process_legacy_configuration() OR
         │              │   └── _process_unified_configuration()
         │              │       ├── _process_pytorch_level_config()
         │              │       ├── _process_onnx_graph_level_config()
         │              │       ├── _process_common_config()
         │              │       ├── _process_analytics_config()
         │              │       └── _validate_configuration()
         │              │
         │              └── Returns: UnifiedChunkingConfig
         │
         ├── Line 93: self.enable_chunking = self._should_enable_chunking()
         │
         ├── Line 101-102: method_name, method_config = get_method_specific_config()
         │                  _apply_method_configuration(method_name, method_config)
         │
         └── Line 105: self._initialize_analytics()
                        │
                        ├── ChunkingAnalytics(config)
                        ├── SystemMonitor, PerformanceTimer, ModelAnalyzer, AlertManager
                        └── Optional: EnhancedAnalyticsIntegrator

Model.make_model() called
         │
         ├── Build ONNX graph (embedding → layers → norm → lm_head)
         │   ├── make_embedding()
         │   ├── make_layer(i) for each layer
         │   ├── make_layernorm()
         │   └── make_lm_head()
         │
         └── If enable_chunking=True:
             Model.save_model(out_dir)
                  │
                  ├── Line 586-720: save_model()
                  │   ├── _get_model_characteristics()
                  │   ├── analytics.start_session()
                  │   ├── with analytics.monitor_operation("model_preparation"):
                  │   │   └── if INT4: to_int4() → MatMulNBitsQuantizer
                  │   │   └── else: use self.model
                  │   │
                  │   ├── with analytics.monitor_operation("graph_topological_sort"):
                  │   │   └── model.graph.sort()
                  │   │
                  │   ├── with analytics.monitor_operation("model_saving"):
                  │   │   └── ir.save(model, out_path, external_data=...)
                  │   │
                  │   ├── _trigger_enhanced_analytics_reporting()
                  │   ├── analytics.end_session()
                  │   └── analytics.save_report() / save_enhanced_reports()
                  │
                  └── If chunking enabled:
                      Model.create_chunked_models(output_dir)
                           │
                           ├── _get_op_idx_split_location()
                           │   └── Analyze initializer sizes, find midpoint
                           │
                           ├── _get_first_chunk_outputs(split_node_idx)
                           │   └── Trace graph connectivity, find boundary tensors
                           │
                           ├── _make_first_chunk_prog(split_node_idx)
                           │   └── Create ONNX graph with nodes 0..split
                           │
                           ├── _make_second_chunk_prog(split_node_idx, boundary_tensors)
                           │   └── Create ONNX graph with nodes split..end
                           │
                           └── _verify_output_correctness_of_chunks()
                               └── PSNR comparison: original vs chunked outputs
```

### Execution Flow: MASTER_POC Semantic Split

```
User runs: python grok_onnx_split.py --model <path> --output <dir>
         │
         ▼
main() in grok_onnx_split.py
         │
         ├── argparse: Parse command-line arguments
         │
         └── split_onnx_model(params)
              │
              ├── Line 179: model = load_onnx_model(params.model)
              │   ├── onnx.load(model_file, load_external_data=True/False)
              │   └── Returns: onnx.ModelProto
              │
              ├── Line 185-188: shape_inference.infer_shapes(model)
              │   └── Static shape inference for boundary tensor types
              │
              ├── Line 191: components = extract_components(model)
              │   │
              │   ├── detect_architecture_from_onnx(graph)
              │   │   └── Pattern match node names → ArchitectureType
              │   │
              │   ├── get_onnx_model_prefixes(arch)
              │   │   └── Return (embed_prefix, layer_prefix, norm_prefix, lm_prefix)
              │   │
              │   └── For each node in graph.node:
              │       ├── Match node.name against prefixes
              │       ├── Extract layer_id from pattern "layers.N" or "h.N"
              │       ├── Track initializers used by each component
              │       └── Build ComponentInfo for: embedding, layer_0..N, norm, lm_head, other
              │
              ├── Line 194-197: Create output directory structure
              │
              └── For each component in sorted(components):
                  │
                  ├── create_component_onnx(model, component, component_name)
                  │   │
                  │   ├── Collect nodes for this component
                  │   ├── Identify boundary tensors (inputs/outputs crossing component)
                  │   ├── Separate initializer inputs from boundary inputs
                  │   ├── Create input ValueInfoProtos with inferred shapes
                  │   ├── Create output ValueInfoProtos
                  │   └── helper.make_graph() → helper.make_model()
                  │
                  ├── onnx.save(component_model, filepath, save_as_external_data=True)
                  │
                  └── Update component_files dict

              │
              └── Save model_info.json with:
                  - num_layers, num_components
                  - architecture, ir_version, producer
                  - Per-component: type, file, num_nodes, input_names, output_names

═══════════════════════════════════════════════════════════════════════

INFERENCE PIPELINE EXECUTION

User runs: python grok_inference_split_model_pipeline.py --model-dir <dir> --prompt <text>
         │
         ▼
main()
         │
         ├── SequentialModelPipeline(model_path, providers, persist_sessions=False)
         │   │
         │   ├── __init__():
         │   │   ├── Load model_info.json
         │   │   ├── Load tokenizer via AutoTokenizer
         │   │   ├── config = load_model_config(model_dir)
         │   │   │   └── Try config.json → model_info.json → defaults
         │   │   │
         │   │   ├── kv_cache = KVCacheManager(num_layers, num_kv_heads, head_dim, dtype)
         │   │   │   └── Initialize per-layer KV cache tensors [batch, heads, 0, head_dim]
         │   │   │
         │   │   └── If persist_sessions=True:
         │   │       └── _init_persistent_sessions()
         │   │           ├── SessionManager(model_dir, config, model_info)
         │   │           └── session_manager.load_all()
         │   │
         │   └── tensor_registry = {}  # Shared state between components
         │
         ├── pipeline.generate(prompt, max_new_tokens)
         │   │
         │   ├── Tokenize: input_ids = tokenizer.encode(prompt)
         │   │
         │   ├── PREFILL PHASE (process full prompt):
         │   │   │
         │   │   ├── Run embeddings.onnx
         │   │   │   ├── session = load_component_session("embedding")
         │   │   │   ├── outputs = session.run(None, {"input_ids": input_ids})
         │   │   │   └── tensor_registry["hidden_states"] = outputs[0]
         │   │   │
         │   │   ├── For layer_idx in 0..num_layers-1:
         │   │   │   │
         │   │   │   ├── session = load_component_session(f"layer_{layer_idx}")
         │   │   │   │
         │   │   │   ├── Prepare inputs:
         │   │   │   │   ├── hidden_states from registry
         │   │   │   │   └── kv_cache.inject_into_registry(registry)  # past_key_values
         │   │   │   │
         │   │   │   ├── outputs = session.run(None, inputs)
         │   │   │   │
         │   │   │   └── Update registry and cache:
         │   │   │       ├── tensor_registry["hidden_states"] = outputs["hidden_states"]
         │   │   │       └── kv_cache.update_from_present(layer_idx, present_key, present_value)
         │   │   │
         │   │   ├── Run norm.onnx
         │   │   │   └── tensor_registry["hidden_states"] = norm_output
         │   │   │
         │   │   └── Run lm_head.onnx
         │   │       └── next_token_logits = lm_head_output
         │   │
         │   ├── DECODE PHASE (generate new tokens):
         │   │   │
         │   │   └── For _ in range(max_new_tokens):
         │   │       │
         │   │       ├── Select token: next_token = argmax(logits)
         │   │       │
         │   │       ├── Check EOS: if next_token == eos_token_id: break
         │   │       │
         │   │       └── Repeat PREFILL with single token + updated KV cache
         │   │
         │   └── Return: tokenizer.decode(generated_tokens)
         │
         └── pipeline.cleanup()
             └── If sequential mode: session_manager.unload_all()
```

### Execution Flow: Analytics System

```
ChunkingAnalytics.__init__(config)
         │
         ├── self.system_monitor = SystemMonitor(sampling_interval, max_history)
         │   ├── Background thread for continuous monitoring
         │   └── Collects: CPU, memory, disk, GPU, network, threads
         │
         ├── self.performance_timer = PerformanceTimer()
         │   └── High-precision timing with context manager support
         │
         ├── self.model_analyzer = ModelAnalyzer()
         │   └── Model-specific metrics and efficiency scoring
         │
         ├── self.alert_manager = AlertManager()
         │   └── Threshold-based alerting with suggestions
         │
         └── Optional: self.graph_generator, self.markdown_reporter
             └── Enhanced reporting with matplotlib

ChunkingAnalytics.start_session()
         │
         └── self.system_monitor.start_monitoring()
             └── Start background sampling thread

@contextmanager: analytics.monitor_operation(name, phase, **metadata)
         │
         ├── Pre-operation: Check alerts (memory, CPU, disk thresholds)
         │
         ├── During: performance_timer.time_operation(name, phase)
         │   └── Record: start_time, end_time, duration_ms, phase, metadata
         │
         └── Post-operation: Capture any error metrics

analytics.add_model_analysis(**kwargs)
         │
         └── model_analyzer.analyze_model(
             model_name, original_size_mb, chunk_count, chunk_sizes_mb,
             parameter_count, memory_footprint_mb,
             conversion_time_ms, chunking_time_ms, method
             )
         │
         └── Calculate:
             - compression_ratio = original_size / total_chunk_size
             - efficiency_score = (memory_eff + time_eff + size_eff) / 3

analytics.get_comprehensive_report()
         │
         ├── session_info: start_time, end_time, duration, total_operations
         │
         ├── system_resources:
         │   ├── peak_usage: peak_cpu, peak_memory, peak_disk
         │   └── average_usage: avg_cpu, avg_memory
         │
         ├── performance_timing:
         │   ├── summary: per-operation count, total, min, max, avg
         │   └── phase_breakdown: per-phase totals
         │
         ├── model_analysis:
         │   ├── models_analyzed: efficiency_score, compression_ratio
         │   └── comparisons: method vs method
         │
         ├── alerts:
         │   ├── summary: count by level (info, warning, error, critical)
         │   └── recent_alerts: last 5 alerts with timestamps
         │
         └── recommendations: Generated actionable insights

analytics.save_report(output_dir, formats=['json', 'csv', 'summary'])
         │
         ├── JSON: Complete report with all metrics
         ├── CSV: Flattened data for spreadsheet analysis
         └── Summary: Human-readable text report

analytics.save_enhanced_reports(output_dir)
         │
         ├── GraphGenerator.create_system_resource_dashboard()
         │   └── Time-series plots: CPU, memory, disk over session
         │
         ├── GraphGenerator.create_bar_chart(phase_breakdown)
         │   └── Phase timing comparison
         │
         └── MarkdownReporter.generate_report()
             └── Professional markdown with embedded graphs
```

---

## COMPONENT SPECIFICATION

### 1. Configuration System (`chunking_config.py`)

**Purpose:** Unified configuration management for dual chunking approaches

**Classes:**

#### `ChunkingMethod` (Enum)
```python
PYTORCH_LEVEL = "pytorch_level"
ONNX_GRAPH_LEVEL = "onnx_graph_level"
```

#### `ChunkStrategy` (Enum)
```python
BY_LAYERS = "by_layers"
BY_WEIGHTS = "by_weights"
LAYER_BASED = "layer_based"
WEIGHT_BASED = "weight_based"
```

#### `DistributionType` (Enum)
```python
EQUAL_LAYERS = "equal_layers"
EQUAL_MEMORY = "equal_memory"
EQUAL_WEIGHTS = "equal_weights"
CUSTOM = "custom"
```

#### `CommonChunkingConfig` (dataclass)
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| output_directory | str | "./chunked_models" | Output directory for chunks |
| save_original_model | bool | True | Preserve original model |
| verify_chunks | bool | False | Enable PSNR verification |
| min_psnr_threshold | float | 35.0 | Minimum PSNR for validation |
| save_metadata | bool | True | Save chunking metadata |
| verbose | bool | True | Enable verbose output |

#### `AnalyticsConfig` (dataclass)
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| enable_monitoring | bool | True | Enable system monitoring |
| sampling_interval | float | 1.0 | Monitoring frequency (seconds) |
| max_history | int | 3600 | Maximum samples to retain |
| system_resources | List[str] | ["cpu","memory","disk","gpu"] | Resources to monitor |
| timing_phases | List[str] | ["loading","conversion","chunking","saving"] | Phases to track |
| detailed_metrics | bool | True | Enable detailed collection |
| real_time_updates | bool | True | Show real-time updates |
| output_format | List[str] | ["json","csv","summary"] | Report formats |
| enable_alerts | bool | True | Enable alert system |
| save_snapshots | bool | False | Save resource snapshots |
| snapshot_interval | int | 10 | Snapshot frequency |
| performance_alerts | Dict[str,float] | {...} | Alert thresholds |

#### `PyTorchLevelConfig` (dataclass)
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| chunk_strategy | str | "by_layers" | by_layers or by_weights |
| layer_distribution | Optional[List[int]] | None | Layers per chunk |
| include_embedding_in_first | bool | True | Include embedding in chunk 1 |
| include_lm_head_in_last | bool | True | Include LM head in last chunk |
| preserve_intermediate_outputs | bool | True | Preserve intermediate tensors |
| chunk_overlap_ratio | float | 0.0 | Overlap between chunks |
| balance_memory_usage | bool | True | Balance memory across chunks |

#### `ONNXGraphLevelConfig` (dataclass)
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| strategy | str | "weight_based" | layer_based or weight_based |
| distribution | str | "equal_memory" | equal_memory, equal_layers, custom |
| num_chunks | int | 2 | Number of chunks (>= 2) |
| respect_layer_boundaries | bool | True | Force splits at layer boundaries |
| custom_chunk_sizes | Optional[List[float]] | None | Custom size distribution |
| chunk_split_ratio | float | 0.5 | Split ratio for 2 chunks |

#### `UnifiedChunkingConfig` (dataclass)
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| method | str | "onnx_graph_level" | Chunking method |
| pytorch_level | PyTorchLevelConfig | default | PyTorch-specific options |
| onnx_graph_level | ONNXGraphLevelConfig | default | ONNX-specific options |
| common | CommonChunkingConfig | default | Shared options |
| analytics | AnalyticsConfig | default | Analytics configuration |

#### `ConfigurationProcessor` (class)

**Methods:**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `__init__` | `()` | Initialize processor |
| `process_configuration` | `(extra_options: Dict) -> UnifiedChunkingConfig` | Process and validate config |
| `_is_legacy_format` | `(extra_options: Dict) -> bool` | Detect legacy format |
| `_process_legacy_configuration` | `(extra_options: Dict) -> UnifiedChunkingConfig` | Process legacy format |
| `_process_unified_configuration` | `(extra_options: Dict) -> UnifiedChunkingConfig` | Process unified format |
| `_validate_configuration` | `() -> None` | Validate processed config |
| `is_legacy_mode` | `() -> bool` | Check legacy mode status |
| `get_legacy_style_config` | `() -> Dict` | Convert to legacy format |
| `save_configuration` | `(file_path: str) -> None` | Save config to JSON |
| `get_method_specific_config` | `() -> Tuple[str, Any]` | Get method-specific config |

**Validation Rules:**
- `num_chunks` must be >= 2
- `chunk_split_ratio` must be in (0.0, 1.0)
- `chunk_overlap_ratio` must be in [0.0, 1.0)
- `custom_chunk_sizes` length must equal `num_chunks`
- `layer_distribution` must sum to positive number when using by_layers strategy

---

### 2. Analytics System (`chunking_analytics.py`)

**Purpose:** Comprehensive performance monitoring and benchmarking

**Core Classes:**

#### `MetricType` (Enum)
```python
TIMING = "timing"
SYSTEM_RESOURCES = "system_resources"
MODEL_ANALYTICS = "model_analytics"
BENEFIT_ANALYSIS = "benefit_analysis"
PERFORMANCE = "performance"
```

#### `AlertLevel` (Enum)
```python
INFO = "info"
WARNING = "warning"
ERROR = "error"
CRITICAL = "critical"
```

#### `ResourceSnapshot` (dataclass)
| Field | Type | Description |
|-------|------|-------------|
| timestamp | datetime | Snapshot timestamp |
| cpu_percent | float | CPU utilization |
| cpu_freq_mhz | float | CPU frequency |
| memory_mb | float | Memory usage (MB) |
| memory_percent | float | Memory percentage |
| disk_usage_percent | float | Disk usage percentage |
| disk_io_read_mb | float | Cumulative disk read (MB) |
| disk_io_write_mb | float | Cumulative disk write (MB) |
| network_io_sent_mb | float | Cumulative network sent (MB) |
| network_io_recv_mb | float | Cumulative network received (MB) |
| gpu_info | List[Dict] | GPU metrics per device |
| thread_count | int | Process thread count |
| process_count | int | System process count |

#### `SystemMonitor` (class)

**Constructor:**
```python
SystemMonitor(sampling_interval: float = 1.0, max_history: int = 3600)
```

**Methods:**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `start_monitoring` | `()` | Start background thread |
| `stop_monitoring` | `()` | Stop background thread |
| `get_current_snapshot` | `() -> Optional[ResourceSnapshot]` | Get latest snapshot |
| `get_peak_usage` | `() -> Dict[str, float]` | Get peak resource usage |
| `get_average_usage` | `() -> Dict[str, float]` | Get average usage |

**Internal:**
- `_monitor_loop()`: Background monitoring loop
- `_collect_snapshot()`: Collect system snapshot

#### `PerformanceTimer` (class)

**Methods:**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `time_operation` | `(name, phase, **metadata)` | Context manager for timing |
| `start_timer` | `(name, phase, **metadata) -> str` | Start named timer |
| `end_timer` | `(timer_id: str) -> Optional[float]` | End timer, return duration |
| `get_timing_summary` | `() -> Dict` | Get timing statistics |
| `get_phase_breakdown` | `() -> Dict` | Get per-phase totals |

#### `ModelAnalyzer` (class)

**Methods:**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `analyze_model` | `(**kwargs) -> ModelMetric` | Analyze chunked model |
| `compare_methods` | `(model_name: str) -> Optional[Dict]` | Compare chunking methods |

**ModelMetric Fields:**
- model_name, original_size_mb, chunk_count
- chunk_sizes_mb, parameter_count, memory_footprint_mb
- conversion_time_ms, chunking_time_ms, method
- compression_ratio, efficiency_score

#### `AlertManager` (class)

**Methods:**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `set_threshold` | `(metric_name, threshold_value)` | Set alert threshold |
| `check_thresholds` | `(metrics, timestamp) -> List[PerformanceAlert]` | Check and generate alerts |
| `get_alert_summary` | `() -> Dict[str, int]` | Get alert counts by level |

**Default Thresholds:**
- memory_percent: 80.0
- cpu_percent: 90.0
- disk_usage_percent: 90.0
- conversion_time_seconds: 300.0
- memory_usage_gb: 8.0

#### `ChunkingAnalytics` (class) - Main Orchestrator

**Constructor:**
```python
ChunkingAnalytics(config: Optional[Dict[str, Any]] = None)
```

**Methods:**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `start_session` | `()` | Start monitoring session |
| `end_session` | `()` | End monitoring session |
| `monitor_operation` | `(name, phase, **metadata)` | Context manager for operation |
| `add_model_analysis` | `(**kwargs)` | Add model analysis |
| `get_comprehensive_report` | `() -> Dict` | Generate full report |
| `save_report` | `(output_dir, formats) -> Dict[str, str]` | Save reports |
| `save_enhanced_reports` | `(output_dir) -> Dict[str, str]` | Save enhanced reports with graphs |

**Internal:**
- `_get_default_config()`: Get default analytics config
- `_handle_alert(alert)`: Handle performance alert
- `_generate_recommendations()`: Generate actionable insights
- `_prepare_enhanced_report_data()`: Prepare data for visualization

---

### 3. Builder (`builder.py`)

**Purpose:** ONNX model construction with integrated chunking

**Main Class: `Model`**

**Constructor Parameters:**
```python
Model(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
```

**Key Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| context_length | int | Maximum context length |
| hidden_size | int | Hidden dimension |
| num_kv_heads | int | Key-value attention heads |
| num_attn_heads | int | Query attention heads |
| head_size | int | Head dimension |
| num_layers | int | Number of transformer layers |
| vocab_size | int | Vocabulary size |
| model_type | str | Architecture type |
| io_dtype | ir.DataType | I/O tensor data type |
| onnx_dtype | ir.DataType | Weight data type |
| ep | str | Execution provider |
| config_processor | ConfigurationProcessor | Config processor |
| chunking_config | UnifiedChunkingConfig | Processed config |
| enable_chunking | bool | Chunking enabled flag |
| analytics | ChunkingAnalytics | Analytics instance |
| layer_boundaries | Dict | Architecture-aware boundaries |

**Model Building Methods:**

| Method | Purpose |
|--------|---------|
| `make_model(model_name_or_path, **kwargs)` | Build complete ONNX model |
| `make_embedding()` | Create embedding layer |
| `make_layer(layer_id)` | Create transformer layer |
| `make_layernorm()` | Create layer normalization |
| `make_lm_head()` | Create language model head |
| `make_attention(layer_id)` | Create attention mechanism |
| `make_mlp(layer_id)` | Create MLP block |

**Graph Operation Methods (make_*):**
```python
make_node(op_type, inputs, outputs, name, **kwargs)
make_value(name, dtype, shape)
make_inputs_and_outputs()
make_initializer(tensor, name, to_dtype)
make_constant(name)
make_gather/name, inputs, dtype, shape, axis)
make_reshape(name, inputs, dtype, shape)
make_shape(name, root_input, shape)
make_constant_of_shape(name, root_input, value, dtype, shape)
make_unsqueeze(name, inputs, dtype, shape)
make_squeeze(name, inputs, dtype, shape)
make_concat(name, inputs, dtype, shape, axis)
make_tile(name, inputs, dtype, shape)
make_equal(name, inputs, shape)
make_greater(name, inputs, shape)
make_greater_or_equal(name, inputs, shape)
make_isinf(name, root_input, shape)
make_clip(name, inputs, dtype, shape)
make_where(name, inputs, dtype, shape)
make_expand(name, inputs, dtype, shape)
make_reduce_sum(name, inputs, dtype, shape, keepdims)
make_reduce_max(name, inputs, dtype, shape, keepdims)
make_reduce_mean(name, inputs, dtype, shape, keepdims)
make_sqrt(name, inputs, dtype, shape)
make_cast(name, root_input, dtype, shape)
make_add(name, inputs, dtype, shape)
make_sub(name, inputs, dtype, shape)
make_less(name, inputs)
make_range(name, inputs)
make_slice(name, inputs, dtype, shape)
make_mul(name, inputs, dtype, shape)
make_transpose(name, root_input, dtype, shape, perm)
make_div(name, inputs, dtype, shape)
make_tanh(name, root_input, dtype, shape)
make_softmax(name, root_input, dtype, shape, axis)
make_sigmoid(name, root_input, dtype, shape)
make_matmul(matmul, basename, root_input, **kwargs)
```

**Chunking Methods:**

| Method | Purpose |
|--------|---------|
| `_should_enable_chunking()` | Determine if chunking should be enabled |
| `_apply_method_configuration(method_name, method_config)` | Apply method settings |
| `_initialize_analytics()` | Initialize analytics system |
| `_set_default_chunking_config()` | Set default chunking config |
| `_get_op_idx_split_location()` | Find optimal split point |
| `_get_first_chunk_outputs(split_node_idx)` | Identify boundary tensors |
| `_make_first_chunk_prog(split_node_idx)` | Create first chunk model |
| `_make_second_chunk_prog(split_node_idx, boundary_tensors)` | Create second chunk model |
| `_verify_output_correctness_of_chunks()` | PSNR verification |
| `get_chunking_info()` | Get current chunking configuration |
| `save_chunking_config(file_path)` | Save config to file |

**Save Methods:**

| Method | Purpose |
|--------|---------|
| `save_model(out_dir)` | Save ONNX model with analytics |
| `save_processing(model_name_or_path, extra_kwargs, out_dir)` | Save tokenizer and config |
| `make_genai_config(model_name_or_path, extra_kwargs, out_dir)` | Generate GenAI config |

**Quantization Methods:**

| Method | Purpose |
|--------|---------|
| `make_int4_algo_config(quant_method)` | Create INT4 quantization config |
| `to_int4()` | Convert model to INT4 |
| `make_matmul_int4(...)`, `make_matmul_int4_qdq(...)`, `make_matmul_float(...)` | MatMul variants |

**Derived Classes:**
- `QwenModel(Model)`: Qwen-specific architecture
- `Qwen3Model(QwenModel)`: Qwen3 with q_norm/k_norm

---

### 4. MASTER_POC: ONNX Semantic Splitter (`grok_onnx_split.py`, `chunking_utils/onnx/splitter.py`)

**Purpose:** Decompose ONNX models into per-component files

**Main Function:** `split_onnx(model, output, config, split_only, **kwargs) -> OnnxSplitResult`

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| model | str | Yes* | Path to ONNX model |
| output | str | Yes* | Output directory |
| config | SplitConfig | No | Configuration object |
| split_only | bool | No | Skip inference validation |

*Required if config not provided

**Internal Functions:**

| Function | Purpose |
|----------|---------|
| `load_onnx_model(model_path)` | Load ONNX model with external data handling |
| `detect_architecture_from_onnx(graph)` | Detect architecture from node patterns |
| `get_onnx_prefixes(model_type)` | Get node name prefixes for architecture |
| `extract_components(model)` | Extract semantic components |
| `create_component_onnx(model, component, component_name)` | Create standalone ONNX for component |

**Component Types:**
- `embedding`: Token embedding layer
- `layer_0` through `layer_N`: Individual transformer layers
- `norm`: Final layer normalization
- `lm_head`: Language model head
- `other`: Miscellaneous nodes

**Output Files:**
```
output_dir/
├── embeddings.onnx
├── layer_0_of_N.onnx
├── layer_1_of_N.onnx
│   ...
├── layer_N_of_N.onnx
├── norm.onnx
├── lm_head.onnx
└── model_info.json
```

**model_info.json Structure:**
```json
{
  "original_model": "path/to/original.onnx",
  "num_layers": 24,
  "num_components": 28,
  "architecture": "llama",
  "ir_version": 10,
  "producer": "onnxruntime-genai",
  "components": {
    "embedding": {
      "type": "embedding",
      "layer_id": null,
      "num_nodes": 15,
      "num_initializers": 8,
      "input_names": ["input_ids"],
      "output_names": ["hidden_states"],
      "file": "embeddings.onnx"
    },
    "layer_0": {
      "type": "layer",
      "layer_id": 0,
      "num_nodes": 45,
      "num_initializers": 24,
      "input_names": ["hidden_states", "past_key_values.0.key", ...],
      "output_names": ["hidden_states", "present.0.key", ...],
      "file": "layer_0_of_24.onnx"
    },
    ...
  }
}
```

---

### 5. MASTER_POC: Inference Pipeline (`grok_inference_split_model_pipeline.py`)

**Purpose:** Sequential layer-by-layer inference with memory management

**Main Class:** `SequentialModelPipeline`

**Constructor:**
```python
SequentialModelPipeline(
    model_path: str,
    tokenizer_path: Optional[str] = None,
    providers: Optional[List[str]] = None,
    verbose: bool = True,
    persist_sessions: bool = False
)
```

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| model_path | str | Path to split model directory |
| is_monolith | bool | Whether using monolithic model |
| model_dir | str | Model directory |
| model_info | Dict | Loaded model_info.json |
| tokenizer | AutoTokenizer | Tokenizer instance |
| num_layers | int | Number of transformer layers |
| config | ModelConfig | Model configuration |
| kv_cache | KVCacheManager | KV cache manager |
| session_manager | Optional[SessionManager] | Session manager (persistent mode) |
| tensor_registry | Dict[str, np.ndarray] | Shared tensor storage |
| providers | List[str] | Execution providers |
| persist_sessions | bool | Persistent session mode flag |

**Methods:**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `_log` | `(msg: str)` | Log message if verbose |
| `_init_persistent_sessions` | `()` | Initialize persistent sessions |
| `load_component_session` | `(component_name) -> Optional[Session]` | Load component session |
| `_detect_kv_format` | `() -> Tuple[str, str]` | Detect KV cache naming format |
| `run_component` | `(component_name, inputs) -> Dict` | Run single component |
| `prefill` | `(input_ids) -> np.ndarray` | Process full prompt |
| `decode_step` | `(input_ids) -> int` | Generate single token |
| `generate` | `(prompt, max_new_tokens, **kwargs) -> str` | Generate text |
| `cleanup` | `()` | Unload sessions and free memory |

**Execution Modes:**

| Mode | persist_sessions | Memory | Speed | Use Case |
|------|-----------------|--------|-------|----------|
| Sequential | False | Minimal (1 component) | Slow (disk I/O) | Memory-constrained |
| Persistent | True | All components | Fast (no I/O) | NPU/GPU with enough VRAM |
| IO_BOUND | True + device buffers | All + device | Fastest | Production NPU/GPU |

---

### 6. Session Manager (`pipeline_utils/session_manager.py`)

**Purpose:** ONNX session lifecycle management with persistence support

**Classes:**

#### `SessionConfig` (dataclass)
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| providers | List[str] | ["CPUExecutionProvider"] | Execution providers |
| optimization_level | str | "DISABLE" | Graph optimization level |
| inter_op_threads | int | 0 | Inter-op parallelism |
| intra_op_threads | int | 0 | Intra-op parallelism |
| memory_limit_mb | Optional[int] | None | Arena memory limit |
| enable_profiling | bool | False | Enable ORT profiling |
| disable_shape_check | bool | True | Disable shape checking |

**Methods:**
- `get_optimization_level_enum() -> ort.GraphOptimizationLevel`
- `create_session_options() -> ort.SessionOptions`

#### `SessionInfo` (dataclass)
| Field | Type | Description |
|-------|------|-------------|
| name | str | Component name |
| file_path | str | Path to ONNX file |
| file_size_bytes | int | File size in bytes |
| is_loaded | bool | Load status |
| input_names | List[str] | Input tensor names |
| output_names | List[str] | Output tensor names |
| input_types | Dict[str, str] | Input tensor types |
| output_types | Dict[str, str] | Output tensor types |
| provider_used | Optional[str] | Actual provider used |

**Methods:**
- `to_dict() -> Dict[str, Any]`

#### `SessionManager` (class)

**Constructor:**
```python
SessionManager(
    model_dir: str,
    config: Optional[SessionConfig] = None,
    model_info: Optional[Dict[str, Any]] = None
)
```

**Methods:**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `load_all` | `() -> Dict[str, SessionInfo]` | Load all sessions eagerly |
| `get_session` | `(name: str) -> Optional[Session]` | Get session (lazy load if needed) |
| `unload_session` | `(name: str) -> bool` | Unload specific session |
| `unload_all` | `() -> int` | Unload all sessions |
| `get_session_info` | `(name: str) -> Optional[SessionInfo]` | Get session info |
| `get_all_session_info` | `() -> Dict[str, SessionInfo]` | Get all session info |
| `get_loaded_sessions` | `() -> List[str]` | List loaded sessions |
| `get_memory_usage` | `() -> Dict[str, Any]` | Get memory statistics |
| `get_component_names` | `() -> List[str]` | Get component names in order |

**Internal:**
- `_discover_sessions()`: Discover from model_info.json
- `_load_single_session(name)`: Load individual session

**Utility Functions:**
- `get_available_providers() -> List[str]`
- `get_provider_for_device(device_type) -> List[str]`

**Supported Providers:**
```python
SUPPORTED_PROVIDERS = [
    'VitisAIExecutionProvider',    # AMD XDNA NPU
    'RyzenAIExecutionProvider',    # AMD Ryzen AI
    'CUDAExecutionProvider',       # NVIDIA GPU
    'ROCmExecutionProvider',       # AMD GPU
    'DmlExecutionProvider',        # DirectML
    'CoreMLExecutionProvider',     # Apple Neural Engine
    'TensorrtExecutionProvider',   # NVIDIA TensorRT
    'OpenVINOExecutionProvider',   # Intel NPU/GPU
    'MIGraphXExecutionProvider',   # AMD MIGraphX
    'CPUExecutionProvider',        # CPU fallback
]
```

---

### 7. KV Cache Manager (`pipeline_utils/kv_cache.py`)

**Purpose:** Manage KV cache state for incremental decoding

**Main Class:** `KVCacheManager`

**Constructor:**
```python
KVCacheManager(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: Any = np.float16,
    batch_size: int = 1
)
```

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| num_layers | int | Number of transformer layers |
| num_kv_heads | int | Key-value attention heads |
| head_dim | int | Head dimension |
| dtype | Any | Numpy dtype |
| batch_size | int | Batch size |
| _cache | Dict[int, Dict[str, np.ndarray]] | Internal cache storage |

**Methods:**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `reset` | `()` | Reset cache to empty |
| `get_sequence_length` | `() -> int` | Get cached sequence length |
| `get_past_tensors` | `(layer_idx) -> Tuple[np.ndarray, np.ndarray]` | Get past K/V |
| `update_from_present` | `(layer_idx, present_key, present_value, validate_shape)` | Update cache |
| `inject_into_registry` | `(registry, key_format, value_format)` | Inject cache into registry |
| `extract_from_registry` | `(registry, present_key_format, present_value_format)` | Extract from registry |
| `get_memory_usage` | `() -> Dict[str, Any]` | Get memory statistics |

**Cache Tensor Shape:**
```
[batch_size, num_kv_heads, sequence_length, head_dim]
```

**KV Cache Flow:**
```
PREFILL:
  For each layer:
    - Input: hidden_states, past_key_values (empty initially)
    - Output: hidden_states, present.{layer}.key, present.{layer}.value
    - Cache: kv_cache.update_from_present(layer, present_key, present_value)

DECODE:
  For each new token:
    - Input: new_token_embed, past_key_values (from cache)
    - Output: logits, new present.{layer}.key/value
    - Cache: kv_cache.update_from_present() with concatenated tensors
```

---

## DATA FLOW DOCUMENTATION

### Root-Level Bisection Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│ INPUT: extra_options Dict                                               │
│ {                                                                       │
│   "enable_chunking": True,                                              │
│   "chunk_strategy": "layer_based",                                      │
│   "num_chunks": 2,                                                      │
│   "chunk_split_ratio": 0.5,                                             │
│   "verify_chunks": True,                                                │
│   "min_psnr_threshold": 35.0,                                           │
│   "chunk_output_dir": "./chunked_models"                                │
│ }                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ ConfigurationProcessor.process_configuration(extra_options)             │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Legacy Format Detection                                         │   │
│  │ - "enable_chunking" in extra_options → legacy_mode = True       │   │
│  │ - Map to UnifiedChunkingConfig                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  │                                                                     │
│  └─────────────────────────────────────────────────────────────────┐   │
│  │ Unified Format Processing                                       │   │
│  │ - "chunking": { "method": "onnx_graph_level", ... }             │   │
│  │ - Process: pytorch_level, onnx_graph_level, common, analytics   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ UnifiedChunkingConfig                                                   │
│ {                                                                       │
│   "method": "onnx_graph_level",                                         │
│   "onnx_graph_level": {                                                 │
│     "strategy": "layer_based",                                          │
│     "distribution": "equal_layers",                                     │
│     "num_chunks": 2,                                                    │
│     "chunk_split_ratio": 0.5                                            │
│   },                                                                    │
│   "common": {                                                           │
│     "output_directory": "./chunked_models",                             │
│     "verify_chunks": True,                                              │
│     "min_psnr_threshold": 35.0                                          │
│   },                                                                    │
│   "analytics": { ... }                                                  │
│ }                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Model.build() → Model.save_model()                                      │
│                                                                         │
│ Data transformations:                                                   │
│ 1. PyTorch weights → ONNX initializers                                  │
│ 2. Graph construction: nodes added layer by layer                       │
│ 3. If INT4: MatMulNBitsQuantizer processes weights                      │
│ 4. Topological sort ensures valid execution order                       │
│ 5. ir.save() writes: model.onnx + model.onnx.data                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ create_chunked_models(output_dir)                                       │
│                                                                         │
│ Split Analysis:                                                         │
│ 1. _get_op_idx_split_location():                                        │
│    - Analyze initializer sizes: [size_0, size_1, ..., size_N]           │
│    - Calculate cumulative: cumsum = [s0, s0+s1, ...]                    │
│    - Find index where cumsum[i] / total ≈ chunk_split_ratio             │
│    - Return split_node_idx                                              │
│                                                                         │
│ 2. _get_first_chunk_outputs(split_node_idx):                            │
│    - Trace forward from split_node_idx                                  │
│    - Find tensors consumed by nodes > split_node_idx                    │
│    - Return boundary_tensors list                                       │
│                                                                         │
│ 3. _make_first_chunk_prog():                                            │
│    - Copy nodes[0:split_node_idx]                                       │
│    - Add boundary_tensors as graph outputs                              │
│    - Copy required initializers                                         │
│    - Return first_chunk_model                                           │
│                                                                         │
│ 4. _make_second_chunk_prog():                                           │
│    - Copy nodes[split_node_idx:end]                                     │
│    - Add boundary_tensors as graph inputs                               │
│    - Copy original outputs                                              │
│    - Copy required initializers                                         │
│    - Return second_chunk_model                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Output Files:                                                           │
│ ./chunked_models/                                                       │
│ ├── model_first_chunk.onnx                                              │
│ ├── model_second_chunk.onnx                                             │
│ └── chunking_metadata.json                                              │
│     {                                                                   │
│       "split_node_index": 245,                                          │
│       "cumulative_size_mb": 1280.5,                                     │
│       "total_size_mb": 2560.0,                                          │
│       "boundary_tensors": ["hidden_states_12", "attention_mask_12"],    │
│       "first_chunk_nodes": 246,                                         │
│       "second_chunk_nodes": 380,                                        │
│       "split_ratio": 0.5                                                │
│     }                                                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ _verify_output_correctness_of_chunks()                                  │
│                                                                         │
│ PSNR Verification:                                                      │
│ 1. Run original model: output_orig = original(input_sample)             │
│ 2. Run chunked pipeline:                                                │
│    - chunk1_out = first_chunk(input_sample)                             │
│    - chunk2_in = {**input_sample, **chunk1_out}                         │
│    - output_chunked = second_chunk(chunk2_in)                           │
│ 3. Calculate PSNR:                                                      │
│    - mse = mean((output_orig - output_chunked)^2)                       │
│    - psnr = 20 * log10(max_val / sqrt(mse))                             │
│ 4. Validate: psnr >= min_psnr_threshold                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### MASTER_POC Semantic Split Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│ INPUT: grok_onnx_split.py command line                                  │
│ python grok_onnx_split.py --model <path> --output <dir> --split-only    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ argparse.Namespace                                                      │
│ {                                                                       │
│   "model": "path/to/model.onnx",                                        │
│   "output": "split_output/my_model",                                    │
│   "split_only": True                                                    │
│ }                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ split_onnx_model(params)                                                │
│                                                                         │
│ Step 1: load_onnx_model(params.model)                                   │
│ ┌─────────────────────────────────────────────────────────────────┐    │
│ │ Check for external data: model.onnx.data                        │    │
│ │ If external data exists:                                        │    │
│ │   - If > 10GB: onnx.load(path, load_external_data=False)        │    │
│ │   - Else: onnx.load(path, load_external_data=True)              │    │
│ │ Else: onnx.load(path, load_external_data=True)                  │    │
│ │                                                                 │    │
│ │ Returns: onnx.ModelProto with:                                  │    │
│ │   - graph.node: List[NodeProto]                                 │    │
│ │   - graph.initializer: List[TensorProto]                        │    │
│ │   - graph.input: List[ValueInfoProto]                           │    │
│ │   - graph.output: List[ValueInfoProto]                          │    │
│ │   - graph.value_info: List[ValueInfoProto]                      │    │
│ │   - opset_import: List[OperatorSetIdProto]                      │    │
│ │   - ir_version: int                                             │    │
│ └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ shape_inference.infer_shapes(model)                                     │
│                                                                         │
│ Purpose: Populate type/shape information for intermediate tensors       │
│ Input: model with graph.value_info (may be incomplete)                  │
│ Output: model with inferred shapes for all tensors                      │
│                                                                         │
│ Note: May fail for dynamic shapes - continuation is OK                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ extract_components(model) -> Dict[str, ComponentInfo]                   │
│                                                                         │
│ Architecture Detection:                                                 │
│ 1. Sample node names (first 100 nodes)                                  │
│ 2. Pattern matching:                                                    │
│    - "/model/layers." in names → LLaMA                                  │
│    - "q_norm" or "k_norm" in names → Qwen3                              │
│    - "/transformer/h." in names → GPT-2                                 │
│ 3. Return ArchitectureType                                              │
│                                                                         │
│ Component Classification:                                               │
│ For each node in graph.node:                                            │
│   node_name = node.name.lower()                                         │
│                                                                         │
│   if "/model/embed" in node_name or "embed_tokens" in node_name:        │
│     components["embedding"].node_indices.append(i)                      │
│     components["embedding"].initializer_names.update(node.input)        │
│                                                                         │
│   elif match r'layers[./](\d+)' in node_name:                           │
│     layer_id = int(match.group(1))                                      │
│     components[f"layer_{layer_id}"].node_indices.append(i)              │
│     components[f"layer_{layer_id}"].initializer_names.update(...)       │
│                                                                         │
│   elif "/model/norm" in node_name and "layers" not in node_name:        │
│     components["norm"].node_indices.append(i)                           │
│                                                                         │
│   elif "/lm_head" in node_name:                                         │
│     components["lm_head"].node_indices.append(i)                        │
│                                                                         │
│   else:                                                                 │
│     components["other"].node_indices.append(i)                          │
│                                                                         │
│ Returns:                                                                │
│ {                                                                       │
│   "embedding": ComponentInfo(component_type="embedding",                │
│                              node_indices=[0,1,2,...],                  │
│                              initializer_names={"embed.weight",...}),   │
│   "layer_0": ComponentInfo(component_type="layer",                     │
│                            layer_id=0,                                  │
│                            node_indices=[3,4,5,...],...),               │
│   ...                                                                   │
│   "norm": ComponentInfo(...),                                           │
│   "lm_head": ComponentInfo(...)                                         │
│ }                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ For each component in sorted(components):                               │
│   create_component_onnx(model, component, component_name)               │
│                                                                         │
│ Component Graph Creation:                                               │
│ 1. Collect nodes for component:                                         │
│    component_nodes = [all_nodes[i] for i in component.node_indices]     │
│                                                                         │
│ 2. Identify boundary tensors:                                           │
│    - component_outputs = {output for node in component_nodes}           │
│    - required_inputs = {input for node in component_nodes               │
│                         if input not in component_outputs}              │
│    - boundary_inputs = required_inputs - initializer_names              │
│    - component_initializers = [init for name in required_inputs         │
│                               if name in initializer_names]             │
│                                                                         │
│ 3. Determine outputs:                                                   │
│    - output_tensors = {output consumed by nodes outside component}      │
│    - OR graph.output tensors                                            │
│    - OR last node's outputs (fallback)                                  │
│                                                                         │
│ 4. Create ValueInfoProtos:                                              │
│    - Input: from value_info_dict or heuristic                           │
│    - Output: from value_info_dict or heuristic                          │
│                                                                         │
│ 5. helper.make_graph():                                                 │
│    - nodes=component_nodes                                              │
│    - inputs=boundary_inputs                                             │
│    - outputs=output_tensors                                             │
│    - initializer=component_initializers                                 │
│                                                                         │
│ 6. helper.make_model():                                                 │
│    - graph=component_graph                                              │
│    - opset_imports=model.opset_import                                   │
│    - ir_version=model.ir_version                                        │
│                                                                         │
│ Returns: onnx.ModelProto (standalone valid model)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ onnx.save(component_model, filepath, save_as_external_data=True)        │
│                                                                         │
│ Output:                                                                 │
│ - component.onnx (graph structure)                                      │
│ - component.onnx.data (weights, if > 1MB threshold)                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Save model_info.json:                                                   │
│ {                                                                       │
│   "original_model": "path/to/model.onnx",                               │
│   "num_layers": 24,                                                     │
│   "num_components": 28,                                                 │
│   "architecture": "llama",                                              │
│   "ir_version": 10,                                                     │
│   "producer": "onnxruntime-genai",                                      │
│   "components": {                                                       │
│     "embedding": {                                                      │
│       "type": "embedding",                                              │
│       "layer_id": null,                                                 │
│       "num_nodes": 15,                                                  │
│       "num_initializers": 8,                                            │
│       "input_names": ["input_ids"],                                     │
│       "output_names": ["hidden_states"],                                │
│       "file": "embeddings.onnx"                                         │
│     },                                                                  │
│     "layer_0": {                                                        │
│       "type": "layer",                                                  │
│       "layer_id": 0,                                                    │
│       "num_nodes": 45,                                                  │
│       "num_initializers": 24,                                           │
│       "input_names": ["hidden_states", "past_key_values.0.key",         │
│                      "past_key_values.0.value", "attention_mask"],      │
│       "output_names": ["hidden_states", "present.0.key",                │
│                       "present.0.value"],                               │
│       "file": "layer_0_of_24.onnx"                                      │
│     }                                                                   │
│   }                                                                     │
│ }                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

### Inference Pipeline Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│ INPUT: User prompt                                                      │
│ "The future of AI is"                                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ tokenizer.encode(prompt) -> input_ids                                   │
│ input_ids = [1, 5247, 318, 8765, 29889]  # Example token IDs            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PREFILL PHASE                                                           │
│ Process full prompt through all components                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               │
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. embeddings.onnx                                                      │
│    Input: {"input_ids": input_ids}                                      │
│    Output: hidden_states [batch, seq_len, hidden_size]                  │
│    Action: tensor_registry["hidden_states"] = hidden_states             │
│    Session: Load → Run → Unload (sequential mode)                       │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. For layer_idx in 0..num_layers-1:                                    │
│                                                                         │
│    a) Load layer_{layer_idx}.onnx                                       │
│                                                                         │
│    b) Prepare inputs from tensor_registry:                              │
│       {                                                                 │
│         "hidden_states": tensor_registry["hidden_states"],              │
│         "past_key_values.{layer}.key": kv_cache[layer].key,             │
│         "past_key_values.{layer}.value": kv_cache[layer].value,         │
│         "attention_mask": attention_mask                                │
│       }                                                                 │
│                                                                         │
│    c) Run session: outputs = session.run(None, inputs)                  │
│                                                                         │
│    d) Update tensor_registry:                                           │
│       tensor_registry["hidden_states"] = outputs["hidden_states"]       │
│                                                                         │
│    e) Update KV cache:                                                  │
│       kv_cache.update_from_present(                                     │
│         layer_idx,                                                      │
│         outputs["present.{layer}.key"],                                 │
│         outputs["present.{layer}.value"]                                │
│       )                                                                 │
│                                                                         │
│    f) Unload session (sequential mode)                                  │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. norm.onnx                                                            │
│    Input: {"hidden_states": tensor_registry["hidden_states"]}           │
│    Output: normalized_hidden_states                                     │
│    Action: tensor_registry["hidden_states"] = normalized_hidden_states  │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. lm_head.onnx                                                         │
│    Input: {"hidden_states": tensor_registry["hidden_states"]}           │
│    Output: logits [batch, seq_len, vocab_size]                          │
│    Action: next_token_logits = logits[:, -1, :]                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ DECODE PHASE                                                            │
│ Generate new tokens autoregressively                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ For _ in range(max_new_tokens):                                         │
│                                                                         │
│   a) Select token: next_token = argmax(next_token_logits)               │
│                                                                         │
│   b) Check EOS: if next_token == eos_token_id: break                    │
│                                                                         │
│   c) Append to generated: generated_tokens.append(next_token)           │
│                                                                         │
│   d) Create new input: new_input_ids = [[next_token]]                   │
│                                                                         │
│   e) Repeat PREFILL steps with:                                         │
│      - Single token input                                               │
│      - Populated KV cache (past_sequence_length > 0)                    │
│                                                                         │
│   f) Update next_token_logits for next iteration                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ OUTPUT: Generated text                                                  │
│ tokenizer.decode(generated_tokens)                                      │
│ "The future of AI is [generated text...]"                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## API REFERENCE

### Public API Exports

#### Root-Level Exports

```python
# chunking_config.py
from chunking_config import (
    # Enums
    ChunkingMethod,
    ChunkStrategy,
    DistributionType,

    # Config dataclasses
    CommonChunkingConfig,
    AnalyticsConfig,
    PyTorchLevelConfig,
    ONNXGraphLevelConfig,
    UnifiedChunkingConfig,

    # Processor
    ConfigurationProcessor,

    # Utilities
    create_sample_configurations,
    print_configuration_examples,
)

# chunking_analytics.py
from chunking_analytics import (
    # Enums
    MetricType,
    AlertLevel,

    # Data classes
    ResourceSnapshot,
    TimingMetric,
    ModelMetric,
    PerformanceAlert,

    # Core classes
    SystemMonitor,
    PerformanceTimer,
    ModelAnalyzer,
    AlertManager,
    ChunkingAnalytics,

    # Convenience functions
    create_analytics_session,
    monitor_chunking_performance,
)

# builder.py
from builder import (
    Model,
    QwenModel,
    Qwen3Model,
)
```

#### MASTER_POC Exports

```python
# chunking_utils/__init__.py
from chunking_utils import (
    # Common
    common.architecture,
    common.types,
    common.path_utils,

    # ONNX
    onnx.splitter,
    onnx.merger,
    onnx.loader,
    onnx.inference,

    # SafeTensors
    safetensors.splitter,
    safetensors.merger,
    safetensors.loader,
    safetensors.inference,
)

# chunking_utils/common/__init__.py
from chunking_utils.common import (
    ArchitectureType,
    ArchitecturePrefixes,
    ModelFormat,
    TensorDtype,
    MemoryMode,
    ComponentType,
    SplitConfig,
    SplitResult,
    ComponentInfo,
    MergeResult,
    get_prefixes,
    get_prefixes_tuple,
    architecture_from_string,
    SAFETENSORS_PREFIXES,
    ONNX_PREFIXES,
    LLAMA_STYLE_ARCHITECTURES,
)

# chunking_utils/onnx/__init__.py
from chunking_utils.onnx import (
    split_onnx,
    merge_onnx,
    merge_onnx_with_result,
    get_merge_info,
    load_onnx_model,
    ComponentInfo,  # ONNX-specific
    OnnxSplitResult,
    OnnxMergeResult,
)

# chunking_utils/safetensors/__init__.py
from chunking_utils.safetensors import (
    split_safetensors,
    merge_safetensors,
    load_safetensors,
    SafeTensorsSplitResult,
)

# pipeline_utils/__init__.py
from pipeline_utils import (
    # Session management
    SessionManager,
    SessionConfig,
    SessionInfo,
    get_available_providers,
    get_provider_for_device,
    SUPPORTED_PROVIDERS,
    PROVIDER_CATEGORIES,

    # KV cache
    KVCacheManager,

    # Types
    ModelConfig,
    load_model_config,

    # dtype handling
    get_numpy_dtype,
    safe_cast,

    # IO binding
    IOBindingManager,

    # Backend
    Backend,
    get_backend,
)
```

### Function Signatures

#### Configuration Functions

```python
def process_configuration(extra_options: Dict[str, Any]) -> UnifiedChunkingConfig:
    """Process and validate configuration."""

def create_sample_configurations() -> Dict[str, Dict[str, Any]]:
    """Create sample configurations for different scenarios."""

def save_configuration(file_path: str) -> None:
    """Save configuration to JSON file."""
```

#### Analytics Functions

```python
def create_analytics_session(config: Optional[Dict[str, Any]] = None) -> ChunkingAnalytics:
    """Create new analytics session."""

@contextmanager
def monitor_chunking_performance(config: Optional[Dict[str, Any]] = None):
    """Context manager for monitoring chunking performance."""

def get_comprehensive_report() -> Dict[str, Any]:
    """Generate comprehensive analytics report."""

def save_report(output_dir: str, formats: List[str] = None) -> Dict[str, str]:
    """Save analytics report in specified formats."""
```

#### Splitting Functions

```python
def split_onnx(
    model: Optional[str] = None,
    output: Optional[str] = None,
    config: Optional[SplitConfig] = None,
    split_only: bool = False,
    **kwargs
) -> OnnxSplitResult:
    """Split ONNX model into per-component files."""

def split_safetensors(
    model: str,
    output: str,
    config: Optional[SplitConfig] = None,
    **kwargs
) -> SafeTensorsSplitResult:
    """Split SafeTensors model into per-layer files."""

def merge_onnx(
    output_dir: str,
    save_path: Optional[str] = None,
) -> onnx.ModelProto:
    """Merge component ONNX files back into single model."""
```

#### Inference Functions

```python
def generate(
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.9,
    do_sample: bool = True
) -> str:
    """Generate text from prompt."""

def prefill(input_ids: np.ndarray) -> np.ndarray:
    """Process full prompt through pipeline."""

def decode_step(input_ids: np.ndarray) -> int:
    """Generate single token."""
```

#### Session Management Functions

```python
def load_all() -> Dict[str, SessionInfo]:
    """Load all sessions eagerly."""

def get_session(name: str) -> Optional[ort.InferenceSession]:
    """Get session by name (lazy load if needed)."""

def unload_all() -> int:
    """Unload all sessions."""

def get_memory_usage() -> Dict[str, Any]:
    """Get memory usage statistics."""
```

---

## CLASS HIERARCHIES

### Inheritance Tree

```
object
├── Enum
│   ├── ChunkingMethod
│   ├── ChunkStrategy
│   ├── DistributionType
│   ├── MetricType
│   ├── AlertLevel
│   ├── ModelFormat
│   ├── TensorDtype
│   ├── MemoryMode
│   ├── ComponentType
│   └── ArchitectureType
│
├── Model (builder.py)
│   ├── QwenModel
│   │   └── Qwen3Model
│   └── (other model types if implemented)
│
├── ConfigurationProcessor
├── ChunkingAnalytics
├── SystemMonitor
├── PerformanceTimer
├── ModelAnalyzer
├── AlertManager
├── SessionManager
├── KVCacheManager
└── IOBindingManager
```

### Dataclass Hierarchy

```
@dataclass
├── CommonChunkingConfig
├── AnalyticsConfig
├── PyTorchLevelConfig
├── ONNXGraphLevelConfig
├── UnifiedChunkingConfig
├── ResourceSnapshot
├── TimingMetric
├── ModelMetric
├── PerformanceAlert
├── ArchitecturePrefixes
├── SplitConfig
├── SplitResult
├── ComponentInfo
├── MergeResult
├── SessionConfig
└── SessionInfo
```

### Composition Relationships

```
ChunkingAnalytics
├── SystemMonitor
├── PerformanceTimer
├── ModelAnalyzer
├── AlertManager
├── Optional[GraphGenerator]
└── Optional[MarkdownReporter]

Model
├── ConfigurationProcessor
├── Optional[UnifiedChunkingConfig]
├── Optional[ChunkingAnalytics]
└── Optional[EnhancedAnalyticsIntegrator]

SequentialModelPipeline
├── KVCacheManager
├── Optional[SessionManager]
├── AutoTokenizer
├── ModelConfig
└── tensor_registry (Dict)

SessionManager
├── SessionConfig
├── _sessions (Dict[str, InferenceSession])
└── _session_info (Dict[str, SessionInfo])
```

---

## KNOWN LIMITATIONS AND GAPS

### Root-Level Bisection Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| Only supports 2-way split | Cannot split into more than 2 chunks | Use MASTER_POC semantic split |
| Weight-based split may cut mid-layer | Suboptimal for memory efficiency | Use layer-based semantic split |
| PSNR verification requires sample input | May fail for edge cases | Increase threshold or skip verification |
| No native PyTorch-level chunking | Infrastructure ready, not implemented | Implement in Phase 2 |
| Limited architecture support | Only tested on Qwen, LLaMA | Extend architecture detection |

### MASTER_POC Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| Sequential mode has disk I/O overhead | 2-5x slower per token | Use persistent mode if memory allows |
| NPU incompatible with splitting | 5-100x slower on NPU | Use monolithic model on NPU |
| MoE models need special handling | Expert routing not implemented | Manual expert grouping |
| External data hard-linking required | Manual step for large models | Automate in pipeline |
| Limited shape inference for dynamic shapes | Some boundaries may have unknown shapes | Use `disable_shape_check` |
| No automatic architecture detection for all models | Some models fail detection | Manual architecture specification |

### Analytics System Gaps

| Gap | Impact | Status |
|-----|--------|--------|
| GPU monitoring requires GPUtil | May fail if GPUtil not installed | Optional, graceful degradation |
| Enhanced reporting requires matplotlib | Graph generation fails without it | Optional, text reports work |
| Real-time monitoring has overhead | <1% CPU but measurable | Configurable sampling |
| No ML-based optimization recommendations | Manual analysis required | Future enhancement |

### Testing Gaps

| Gap | Risk | Mitigation |
|-----|------|------------|
| Limited end-to-end tests for all architectures | Undiscovered bugs in edge cases | Expand test matrix |
| No NPU hardware in CI | NPU issues discovered late | Manual NPU testing |
| PSNR verification only for root-level | Semantic split relies on design correctness | Add output comparison |
| No multi-GPU tests | Multi-GPU issues undiscovered | Add multi-GPU test configs |

### Documentation Gaps

| Gap | Impact |
|-----|--------|
| No API reference documentation | Hard to discover features |
| Limited troubleshooting guide | Users stuck on common issues |
| No performance tuning guide | Suboptimal configurations |
| Missing model architecture guide | Hard to add new architectures |

---

## NPU INTEGRATION GAP ANALYSIS

### Current State

The repository has **partial NPU awareness** but lacks full NPU integration:

| Feature | Current Status | NPU Ready |
|---------|---------------|-----------|
| VitisAI EP support | Listed in SUPPORTED_PROVIDERS | ✅ Yes |
| Session management | SessionManager supports all EPs | ✅ Yes |
| KV cache management | Hardware-agnostic | ✅ Yes |
| IO binding | Device buffer support exists | ✅ Yes |
| Hybrid dispatch | Not implemented | ❌ No |
| NPU-specific optimizations | Not implemented | ❌ No |
| AMD hybrid (NPU+iGPU) flow | Documented, not implemented | ❌ No |

### NPU Integration Challenges

#### 1. Split Models Incompatible with NPU Compilation

**Problem:** NPUs compile the entire graph ahead-of-time. Splitting breaks this contract.

**Impact:** 5-100x slowdown when using split models on NPU.

**Root Cause:**
- NPU compilation happens once at session creation
- Each split requires separate compilation
- Context switch overhead: 5-50ms per chunk (vs GPU's 10μs)

**Mitigation:**
```python
# For NPU: Use monolithic model
pipeline = SequentialModelPipeline(
    model_path="monolithic_model.onnx",
    providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
    persist_sessions=True  # Required for NPU
)
```

#### 2. No Per-Component Provider Routing

**Current Code:**
```python
# All components use same providers
pipeline = SequentialModelPipeline(
    model_path="split_output",
    providers=["CPUExecutionProvider"]  # Same for all
)
```

**Required for Hybrid Dispatch:**
```python
pipeline = SequentialModelPipeline(
    model_path="split_output",
    component_providers={
        "embedding": ["DmlExecutionProvider"],      # iGPU
        "layer_*":   ["VitisAIExecutionProvider"],  # NPU
        "norm":      ["CPUExecutionProvider"],      # CPU
        "lm_head":   ["DmlExecutionProvider"],      # iGPU
    }
)
```

**Code Changes Required:**

**File: `grok_inference_split_model_pipeline.py`**

```python
# Current (line ~62-68):
def __init__(
    self,
    model_path: str,
    tokenizer_path: Optional[str] = None,
    providers: Optional[List[str]] = None,  # Single list for all
    ...
):
    self.providers = providers or ['CPUExecutionProvider']

# Required change:
def __init__(
    self,
    model_path: str,
    tokenizer_path: Optional[str] = None,
    providers: Optional[List[str]] = None,
    component_providers: Optional[Dict[str, List[str]]] = None,  # NEW
    ...
):
    self.providers = providers or ['CPUExecutionProvider']
    self.component_providers = component_providers or {}  # NEW

def load_component_session(self, component_name: str):
    # Current (line ~226):
    # session = ort.InferenceSession(filepath, ..., providers=self.providers)

    # Required change:
    providers = self._get_component_providers(component_name)  # NEW
    session = ort.InferenceSession(filepath, ..., providers=providers)

def _get_component_providers(self, component_name: str) -> List[str]:
    """Get providers for specific component."""
    for pattern, provs in self.component_providers.items():
        if pattern == component_name or (
            pattern.endswith('*') and component_name.startswith(pattern[:-1])
        ):
            return provs
    return self.providers  # Fallback to default
```

#### 3. No Phase-Based Hybrid Dispatch

**AMD's Hybrid Flow:**
- PREFILL → NPU (compute-intensive)
- DECODE → iGPU (memory-bandwidth-intensive)

**Not Implemented:**
```python
# Required API:
pipeline = SequentialModelPipeline(
    model_path="split_output",
    phase_providers={
        "prefill": ["VitisAIExecutionProvider"],
        "decode": ["DmlExecutionProvider"],
    }
)
```

**Code Changes Required:**

**File: `grok_inference_split_model_pipeline.py`**

```python
# Add phase tracking:
def __init__(self, ..., phase_providers: Optional[Dict[str, List[str]]] = None):
    self.phase = "prefill"  # or "decode"
    self.phase_providers = phase_providers or {}

def _get_current_providers(self) -> List[str]:
    return self.phase_providers.get(self.phase, self.providers)

def prefill(self, input_ids):
    self.phase = "prefill"
    # ... run with prefill providers

def decode_step(self, input_ids):
    self.phase = "decode"
    # ... run with decode providers
```

#### 4. No NPU-Specific Optimizations

**Missing Optimizations:**

| Optimization | Description | Implementation Effort |
|--------------|-------------|----------------------|
| Graph capture | Pre-compile NPU graph for reuse | Medium |
| Weight preloading | Preload weights to NPU tile SRAM | High |
| Zero-copy I/O | Minimize CPU↔NPU transfers | Medium |
| Batch optimization | Optimize for NPU batch sizes | Low |
| Dynamic shape handling | NPU-specific shape compilation | High |

**Required Changes:**

**File: `pipeline_utils/session_manager.py`**

```python
@dataclass
class SessionConfig:
    providers: List[str]
    # Add NPU-specific options:
    npu_graph_capture: bool = True  # Enable NPU graph capture
    npu_weight_preload: bool = True  # Preload weights to tile SRAM
    npu_compilation_mode: str = "default"  # "default", "fast", "optimized"

def create_session_options(self) -> ort.SessionOptions:
    opts = super().create_session_options()

    # Add NPU-specific config:
    if "VitisAIExecutionProvider" in self.providers:
        opts.add_session_config_entry("vitisai.graph_capture",
                                       "1" if self.npu_graph_capture else "0")
        opts.add_session_config_entry("vitisai.weight_preload",
                                       "1" if self.npu_weight_preload else "0")
        opts.add_session_config_entry("vitisai.compilation_mode",
                                       self.npu_compilation_mode)

    return opts
```

#### 5. No Memory Offloading Strategy

**Problem:** NPU has limited on-chip memory. Large models need CPU↔NPU offloading.

**Required:**
```python
@dataclass
class NPUOffloadConfig:
    # How much of model fits in NPU memory
    npu_memory_limit_mb: int = 4096

    # Offloading strategy
    offload_strategy: str = "layer_by_layer"  # or "pipeline_parallel"

    # Prefetch configuration
    prefetch_layers: int = 1  # How many layers to prefetch

    # Double buffering
    enable_double_buffering: bool = True
```

### What Would Need Modification for NPU Integration

#### Files Requiring Changes

| File | Changes Required | Priority |
|------|-----------------|----------|
| `grok_inference_split_model_pipeline.py` | Add per-component and phase-based provider routing | 🔴 High |
| `pipeline_utils/session_manager.py` | Add NPU-specific session options | 🔴 High |
| `pipeline_utils/io_binding.py` | Optimize zero-copy for NPU | 🟡 Medium |
| `pipeline_utils/dtypes.py` | Add NPU-specific INT4/AWQ types | 🟡 Medium |
| `builder.py` | Add NPU graph capture during build | 🟡 Medium |
| NEW: `pipeline_utils/npu_config.py` | NPU-specific configuration | 🟡 Medium |

#### What Can Be Used As-Is

| Component | NPU Compatible | Notes |
|-----------|---------------|-------|
| `chunking_config.py` | ✅ Yes | Configuration is hardware-agnostic |
| `chunking_analytics.py` | ✅ Yes | Monitoring works on all hardware |
| `KVCacheManager` | ✅ Yes | KV cache format is hardware-agnostic |
| `SessionManager` (basic) | ✅ Yes | Session loading works, just needs NPU options |
| Architecture detection | ✅ Yes | Architecture detection is independent of hardware |

---

## PRIORITY RANKING

### Priority 1: Critical for NPU Integration (Week 1-2)

| Task | Files | Effort | Impact |
|------|-------|--------|--------|
| **P1.1: Per-component provider routing** | `grok_inference_split_model_pipeline.py` | 4 hours | Enables hybrid dispatch |
| **P1.2: NPU session options** | `pipeline_utils/session_manager.py` | 2 hours | Enables NPU optimizations |
| **P1.3: Phase-based dispatch** | `grok_inference_split_model_pipeline.py` | 3 hours | Enables AMD hybrid flow |
| **P1.4: Documentation update** | README, NEW: NPU_GUIDE.md | 4 hours | User guidance |

**Total P1:** ~13 hours

### Priority 2: Important Enhancements (Week 3-4)

| Task | Files | Effort | Impact |
|------|-------|--------|--------|
| **P2.1: Zero-copy I/O optimization** | `pipeline_utils/io_binding.py` | 8 hours | Reduces CPU↔NPU transfers |
| **P2.2: NPU graph capture** | `builder.py`, `session_manager.py` | 6 hours | Faster NPU compilation |
| **P2.3: Memory offloading** | NEW: `npu_offload.py` | 12 hours | Enable larger models |
| **P2.4: Testing framework** | NEW: `test_npu_integration.py` | 8 hours | Validation |

**Total P2:** ~34 hours

### Priority 3: Nice-to-Have (Week 5+)

| Task | Files | Effort | Impact |
|------|-------|--------|--------|
| **P3.1: Automatic provider selection** | `pipeline_utils/backend.py` | 6 hours | Better UX |
| **P3.2: Lemonade SDK integration** | NEW: `lemonade_backend.py` | 16 hours | Ecosystem integration |
| **P3.3: PyPI packaging** | NEW: `pyproject.toml` | 4 hours | Distribution |
| **P3.4: Comprehensive documentation** | Multiple | 16 hours | User adoption |

**Total P3:** ~42 hours

### Summary: NPU Integration Roadmap

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **Phase 1** | Week 1-2 | Per-component routing, NPU options, phase dispatch |
| **Phase 2** | Week 3-4 | Zero-copy I/O, graph capture, memory offloading |
| **Phase 3** | Week 5+ | Auto-selection, Lemonade integration, packaging |

**Total Estimated Effort:** ~89 hours (~2.5 weeks full-time)

---

## APPENDICES

### A. File Locations Quick Reference

| File | Absolute Path |
|------|---------------|
| Root config | `C:\Users\antmi\onnx-model-chunking\chunking_config.py` |
| Root analytics | `C:\Users\antmi\onnx-model-chunking\chunking_analytics.py` |
| Root builder | `C:\Users\antmi\onnx-model-chunking\builder.py` |
| Semantic splitter | `C:\Users\antmi\onnx-model-chunking\MASTER_POC\grok_onnx_split.py` |
| HF splitter | `C:\Users\antmi\onnx-model-chunking\MASTER_POC\grok_hf_split.py` |
| Inference pipeline | `C:\Users\antmi\onnx-model-chunking\MASTER_POC\grok_inference_split_model_pipeline.py` |
| Session manager | `C:\Users\antmi\onnx-model-chunking\MASTER_POC\pipeline_utils\session_manager.py` |
| KV cache | `C:\Users\antmi\onnx-model-chunking\MASTER_POC\pipeline_utils\kv_cache.py` |
| Architecture | `C:\Users\antmi\onnx-model-chunking\MASTER_POC\chunking_utils\common\architecture.py` |
| Types | `C:\Users\antmi\onnx-model-chunking\MASTER_POC\chunking_utils\common\types.py` |
| ONNX splitter | `C:\Users\antmi\onnx-model-chunking\MASTER_POC\chunking_utils\onnx\splitter.py` |
| ONNX merger | `C:\Users\antmi\onnx-model-chunking\MASTER_POC\chunking_utils\onnx\merger.py` |

### B. Key Constants and Defaults

```python
# PSNR Thresholds
MIN_PSNR_THRESHOLD = 35.0  # Minimum acceptable
PRODUCTION_PSNR_THRESHOLD = 40.0  # Production-ready

# Default Config Values
DEFAULT_CHUNKING_METHOD = "onnx_graph_level"
DEFAULT_NUM_CHUNKS = 2
DEFAULT_CHUNK_SPLIT_RATIO = 0.5
DEFAULT_VERIFY_CHUNKS = False

# Analytics Defaults
DEFAULT_SAMPLING_INTERVAL = 1.0  # seconds
DEFAULT_MAX_HISTORY = 3600  # samples
DEFAULT_ENABLE_MONITORING = True

# Session Defaults
DEFAULT_OPTIMIZATION_LEVEL = "DISABLE"
DEFAULT_DISABLE_SHAPE_CHECK = True

# NPU Constants
NPU_CONTEXT_SWITCH_OVERHEAD_MS = 5-50  # vs GPU's 0.01ms
NPU_COMPILATION_TIME = 30-300 seconds  # per graph
```

### C. Glossary

| Term | Definition |
|------|------------|
| **Bisection** | Splitting graph into two equal-weight halves |
| **Semantic Split** | Decomposing by architectural components (layers) |
| **PSNR** | Peak Signal-to-Noise Ratio, accuracy metric |
| **KV Cache** | Key-Value cache for autoregressive decoding |
| **Prefill** | Processing full prompt before generation |
| **Decode** | Generating tokens one at a time |
| **VitisAI EP** | AMD's NPU execution provider for ONNX Runtime |
| **IO Binding** | Zero-copy tensor transfer between devices |
| **Graph Capture** | Pre-compiling computation graph for reuse |

---

**Document End**

*This specification captures the complete technical understanding of the onnx-model-chunking repository as of 2026-03-17. All code paths, class hierarchies, data flows, and NPU integration requirements are documented.*
