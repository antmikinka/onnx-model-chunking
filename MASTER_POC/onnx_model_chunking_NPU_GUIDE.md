# NPU Execution Guide

**AMD XDNA NPU Integration for ONNX Model Inference**

*Version: 1.0 | Last Updated: 2026-02-17*

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [NPU Architecture Overview](#2-npu-architecture-overview)
3. [Configuration Guide](#3-configuration-guide)
4. [Performance Optimization](#4-performance-optimization)
5. [Hybrid Dispatch Patterns](#5-hybrid-dispatch-patterns)
6. [Troubleshooting](#6-troubleshooting)
7. [API Reference](#7-api-reference)
8. [Examples](#8-examples)

---

## 1. Quick Start

This guide covers NPU-accelerated inference using AMD XDNA NPUs (Ryzen AI) with ONNX Runtime and the VitisAI Execution Provider.

### 1.1 Prerequisites

Ensure you have the following installed:

```bash
# ONNX Runtime with VitisAI provider
pip install onnxruntime-vitisai

# Or standard onnxruntime (includes VitisAI on supported hardware)
pip install onnxruntime

# Required dependencies
pip install transformers numpy psutil
```

### 1.2 Verify NPU Availability

Before running NPU inference, verify your system has an AMD XDNA NPU:

```python
from pipeline_utils import is_npu_available, get_npu_capabilities

if is_npu_available():
    print("NPU detected!")
    caps = get_npu_capabilities()
    print(f"  Driver: {caps.driver_version}")
    print(f"  Cores: {caps.num_cores}")
    print(f"  Memory: {caps.memory_size_mb}MB")
    print(f"  Tile SRAM: {caps.tile_sram_mb}MB")
else:
    print("NPU not available - falling back to CPU/GPU")
```

### 1.3 Basic NPU Inference

The simplest way to run inference on NPU:

```bash
python grok_inference_split_model_pipeline.py \
    --model-dir "split_output/qwen2.5-0.5b" \
    --prompt "Explain quantum computing" \
    --tokens 50 \
    --device npu \
    --persist
```

**Key flags:**
- `--device npu`: Selects NPU as primary execution provider
- `--persist`: Enables session persistence (critical for NPU performance)

### 1.4 Quick Example - Programmatic Usage

```python
from pipeline_utils import SessionManager, SessionConfig

# Configure for NPU
config = SessionConfig(
    providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
    npu_graph_capture=True,
    npu_weight_preload=True,
    npu_compilation_mode="default"
)

# Initialize and load
manager = SessionManager("split_output/my_model", config)
manager.load_all()

# Session is now ready for inference
session = manager.get_session("layer_0")
```

---

## 2. NPU Architecture Overview

### 2.1 AMD XDNA NPU Hardware

AMD XDNA (eXtensible Deep Learning Neural Network) is AMD's AI engine architecture, integrated into Ryzen AI processors.

| Feature | Specification |
|---------|---------------|
| Architecture | XDNA / XDNA2 |
| Compute Cores | 2-4 AI Engine tiles |
| Peak Performance | Up to 16+ TOPS (INT8) |
| On-Chip SRAM | 16-32 MB Tile SRAM |
| System Memory | Shared DDR5/LPDDR5X |
| Power Envelope | <5W typical |

### 2.2 NPU vs GPU vs CPU

| Characteristic | CPU | GPU | NPU |
|----------------|-----|-----|-----|
| **Best For** | Control logic, fallback | Parallel compute, training | Dedicated AI inference |
| **Memory Model** | System RAM | VRAM + System | Tile SRAM + System |
| **Compilation** | None (native) | JIT (~10us) | AOT (~5-50ms) |
| **Context Switch** | ~1us | ~10us | ~5-50ms |
| **Power Efficiency** | Low | Medium | **High** |

### 2.3 NPU Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                     SYSTEM MEMORY (DDR5)                     │
│                    8GB - 64GB (Shared)                       │
│  ┌───────────────────────────────────────────────────────┐   │
│  │              NPU DEVICE MEMORY                         │   │
│  │              512MB - 8GB (Allocated)                   │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │            TILE SRAM (On-Chip)                   │   │   │
│  │  │            16-32 MB Ultra-Fast                   │   │   │
│  │  │            [Weights] [Activations]               │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  └───────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 Why NPU Requires Special Handling

**Critical Insight:** NPUs are fundamentally different from GPUs:

1. **Ahead-of-Time Compilation**: The NPU compiles the entire computation graph before execution
2. **Context Switch Overhead**: Loading a new program takes 5-50ms (vs 10us for GPU)
3. **Memory Isolation**: Each session has isolated memory - no sharing between components
4. **Weight Locality**: Performance depends on keeping weights resident in Tile SRAM

**Implication:** For NPU, always use `--persist` mode to avoid repeated compilation overhead.

---

## 3. Configuration Guide

### 3.1 Provider Routing Fundamentals

The system supports two provider routing strategies:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Per-Component** | Different providers for different model components | Hybrid NPU+iGPU execution |
| **Phase-Based** | Different providers for prefill vs decode phases | Optimizing compute vs bandwidth |

### 3.2 Per-Component Provider Routing

Route specific model components to specific execution providers:

```python
from grok_inference_split_model_pipeline import SequentialModelPipeline

pipeline = SequentialModelPipeline(
    model_path="split_output/qwen2.5-0.5b",
    component_providers={
        "embedding": ["DmlExecutionProvider"],      # iGPU for embeddings
        "layer_*":   ["VitisAIExecutionProvider"],  # NPU for all transformer layers
        "norm":      ["CPUExecutionProvider"],      # CPU for normalization
        "lm_head":   ["DmlExecutionProvider"],      # iGPU for output projection
    },
    persist_sessions=True
)
```

**Pattern Matching Rules:**

| Pattern | Matches | Example |
|---------|---------|---------|
| `"embedding"` | Exact match only | `embedding` |
| `"layer_*"` | Wildcard prefix | `layer_0`, `layer_1`, ..., `layer_31` |
| `"*head"` | Wildcard suffix | `lm_head`, `mlp_head` |
| `"*"` | Universal fallback | All components |

**Provider Selection Priority:**
1. Exact component name match
2. Wildcard pattern match (first match wins)
3. Default providers fallback

### 3.3 Phase-Based Provider Dispatch

Different phases of LLM inference have different characteristics:

| Phase | Description | Characteristics | Recommended Provider |
|-------|-------------|-----------------|---------------------|
| **Prefill** | Process full prompt | Compute-intensive, one-time | NPU |
| **Decode** | Generate tokens one-by-one | Memory-bandwidth-intensive, repeated | iGPU/CPU |

```python
pipeline = SequentialModelPipeline(
    model_path="split_output/qwen2.5-0.5b",
    phase_providers={
        "prefill": ["VitisAIExecutionProvider"],  # NPU for heavy compute
        "decode":  ["DmlExecutionProvider"],      # iGPU for bandwidth
    },
    persist_sessions=True
)
```

### 3.4 Combining Both Strategies

You can combine component and phase routing:

```python
pipeline = SequentialModelPipeline(
    model_path="split_output/qwen2.5-0.5b",
    providers=["CPUExecutionProvider"],  # Global fallback
    component_providers={
        "embedding": ["DmlExecutionProvider"],
        "layer_*":   ["VitisAIExecutionProvider"],
        "norm":      ["CPUExecutionProvider"],
        "lm_head":   ["DmlExecutionProvider"],
    },
    phase_providers={
        "prefill": ["VitisAIExecutionProvider"],
        "decode":  ["DmlExecutionProvider"],
    },
    persist_sessions=True
)
```

**Resolution Order:**
1. Component-specific providers take precedence
2. Phase providers apply when no component match
3. Default providers as final fallback

### 3.5 SessionConfig Options

The `SessionConfig` class provides NPU-specific configuration:

```python
from pipeline_utils import SessionConfig, NPUCompilationMode

config = SessionConfig(
    # Provider configuration
    providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],

    # Graph optimization
    optimization_level="DISABLE",  # DISABLE, BASIC, EXTENDED, ALL
    inter_op_threads=0,            # 0 = auto
    intra_op_threads=0,            # 0 = auto

    # Memory management
    memory_limit_mb=4096,
    disable_shape_check=True,      # Required for dynamic sequence lengths

    # NPU-specific options (when using VitisAIExecutionProvider)
    npu_graph_capture=True,        # Cache computation graph
    npu_weight_preload=True,       # Load weights to Tile SRAM
    npu_compilation_mode="default" # default, fast, optimized
)
```

### 3.6 NPU Session Options Reference

| Option | Default | Description | When to Change |
|--------|---------|-------------|----------------|
| `npu_graph_capture` | `True` | Enables graph capture for reuse | Keep `True` for inference |
| `npu_weight_preload` | `True` | Preloads weights to Tile SRAM | `False` if model exceeds SRAM |
| `npu_compilation_mode` | `"default"` | Compilation strategy | See table below |

**Compilation Mode Options:**

| Mode | Compilation Time | Runtime Performance | Use Case |
|------|------------------|---------------------|----------|
| `"fast"` | Fastest | Lower | Development, testing |
| `"default"` | Balanced | Balanced | General inference |
| `"optimized"` | Slowest | Best | Production deployment |

### 3.7 NPUConfig Dataclass

For advanced NPU configuration, use `NPUConfig`:

```python
from pipeline_utils.npu_config import NPUConfig, NPUCompilationMode

config = NPUConfig(
    graph_capture=True,           # Enable graph caching
    weight_preload=True,          # Preload to Tile SRAM
    compilation_mode=NPUCompilationMode.OPTIMIZED,
    enable_profiling=False,       # Enable performance metrics
    debug_mode=False,             # Verbose NPU logging
    memory_limit_mb=4096,         # Cap NPU memory usage
    num_cores=-1                  # -1 = use all cores
)

# Create ONNX Runtime SessionOptions
session_opts = config.create_session_options()
```

### 3.8 Configuration Validation

Validate your NPU configuration before use:

```python
from pipeline_utils.npu_config import (
    NPUConfig,
    get_npu_capabilities,
    validate_npu_config
)

config = NPUConfig(memory_limit_mb=8192)
caps = get_npu_capabilities()

is_valid, warnings = validate_npu_config(config, caps)

if not is_valid:
    print("Configuration issues:")
    for warning in warnings:
        print(f"  - {warning}")
```

---

## 4. Performance Optimization

### 4.1 Provider Selection Guide

| Scenario | Recommended Provider | Rationale |
|----------|---------------------|-----------|
| Model fits in Tile SRAM (<32MB) | NPU | Maximum efficiency |
| Model fits in NPU memory (<8GB) | NPU with weight_preload=False | Good performance |
| Model exceeds NPU memory | iGPU (DML/CUDA) | Avoid excessive offloading |
| Latency-critical decode | iGPU or CPU | Lower context switch |
| Power-efficient inference | NPU | Best TOPS/Watt |

### 4.2 Memory Management Strategies

#### Strategy 1: Full Residency (Recommended)

Keep all sessions loaded in memory:

```python
pipeline = SequentialModelPipeline(
    model_path="split_output/my_model",
    persist_sessions=True  # Critical for NPU
)
```

**Pros:**
- No reload overhead
- Weights stay in Tile SRAM
- Best for repeated inference

**Cons:**
- Higher memory footprint
- Requires sufficient RAM

#### Strategy 2: Sequential Loading

Load/unload components one at a time:

```python
pipeline = SequentialModelPipeline(
    model_path="split_output/my_model",
    persist_sessions=False  # Default
)
```

**Pros:**
- Minimal memory footprint
- Works on memory-constrained systems

**Cons:**
- High reload overhead
- **NOT recommended for NPU** (5-50ms per load)

#### Strategy 3: Hybrid Approach

Keep critical components resident, load others on-demand:

```python
# Custom implementation using SessionManager
manager = SessionManager(model_dir, config)

# Load frequently-used components
manager.load_all()  # Or selectively load

# Unload rarely-used components after use
manager.unload_session("lm_head")
```

### 4.3 Compilation Time vs Execution Time

Understanding the tradeoff:

```
┌──────────────────────────────────────────────────────────────┐
│                    COMPILATION MODE TRADEOFF                  │
├──────────────────────────────────────────────────────────────┤
│  Mode      │ Compilation │ 1st Token │ Avg Token │ Total    │
│  fast      │     1s      │   50ms    │   20ms    │ ~2.5s    │
│  default   │     5s      │   40ms    │   15ms    │ ~2.0s    │
│  optimized │    30s      │   30ms    │   10ms    │ ~1.6s    │
└──────────────────────────────────────────────────────────────┘
```

**Recommendation:**
- Use `fast` during development
- Use `optimized` for production (amortized over many runs)

### 4.4 KV Cache Optimization

Proper KV cache management is critical for autoregressive inference:

```python
from pipeline_utils import KVCacheManager

# Initialize KV cache
kv_cache = KVCacheManager(
    num_layers=32,
    num_kv_heads=8,
    head_dim=128,
    dtype="float32",
    batch_size=1
)

# During prefill: inject KV into registry
kv_cache.inject_into_registry(tensor_registry)

# After layer execution: extract KV from outputs
kv_cache.extract_from_registry(
    tensor_registry,
    present_key_format="present.{layer}.key",
    present_value_format="present.{layer}.value"
)
```

**KV Cache Memory:**
- ~2 x num_layers x num_kv_heads x head_dim x sequence_length bytes
- Example: 32 layers, 8 heads, 128 dim, 1024 seq = ~64MB

### 4.5 IO Binding for Zero-Copy Transfers

For maximum performance, use I/O binding to avoid CPU-GPU-NPU copies:

```python
from pipeline_utils import IOBindingManager

# Initialize with device-resident buffers
binding = IOBindingManager(device_type="npu")

# Bind session outputs to device buffers
binding.bind_output(session, "hidden_states")

# Subsequent components read directly from device memory
# No CPU round-trip!
```

**Performance Impact:**
- Without I/O binding: Device -> CPU -> Device copies
- With I/O binding: Zero-copy between components
- Speedup: 1.5-2x for GPU, less impact on NPU

---

## 5. Hybrid Dispatch Patterns

### 5.1 AMD Recommended NPU+iGPU Flow

AMD recommends a hybrid approach leveraging both NPU and integrated GPU:

```
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID DISPATCH FLOW                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PREFILL PHASE (Compute-Intensive):                         │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐               │
│  │Embedding│────▶│ Layers  │────▶│  Norm   │               │
│  │  (iGPU) │     │  (NPU)  │     │  (CPU)  │               │
│  └─────────┘     └─────────┘     └─────────┘               │
│                                         │                    │
│                                         ▼                    │
│  DECODE PHASE (Memory-Bandwidth):       │LM Head (iGPU)     │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐               │
│  │Embedding│────▶│ Layers  │────▶│  Norm   │               │
│  │  (iGPU) │     │  (NPU)  │     │  (CPU)  │               │
│  └─────────┘     └─────────┘     └─────────┘               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Implementing Hybrid Dispatch

```python
from grok_inference_split_model_pipeline import SequentialModelPipeline

# Configure hybrid dispatch
pipeline = SequentialModelPipeline(
    model_path="split_output/qwen2.5-0.5b",

    # Per-component routing
    component_providers={
        "embedding": ["DmlExecutionProvider"],      # iGPU
        "layer_*":   ["VitisAIExecutionProvider"],  # NPU
        "norm":      ["CPUExecutionProvider"],      # CPU
        "lm_head":   ["DmlExecutionProvider"],      # iGPU
    },

    # Phase-based optimization
    phase_providers={
        "prefill": ["VitisAIExecutionProvider"],    # NPU compute
        "decode":  ["DmlExecutionProvider"],        # iGPU bandwidth
    },

    persist_sessions=True
)
```

### 5.3 When to Use Hybrid Dispatch

| Scenario | Recommended Pattern | Rationale |
|----------|---------------------|-----------|
| Large prompt, few tokens | NPU for prefill, iGPU decode | Compute-heavy prefill |
| Chat/interactive | iGPU throughout | Lower latency per token |
| Batch processing | NPU throughout | Higher throughput |
| Memory-constrained | Component-specific | Distribute memory load |

### 5.4 Multi-Device Load Balancing

For systems with multiple acceleration devices:

```python
# Distribute layers across devices
pipeline = SequentialModelPipeline(
    model_path="split_output/qwen2.5-0.5b",
    component_providers={
        "layer_0":   ["VitisAIExecutionProvider"],  # NPU
        "layer_1":   ["VitisAIExecutionProvider"],  # NPU
        "layer_2":   ["VitisAIExecutionProvider"],  # NPU
        "layer_3":   ["VitisAIExecutionProvider"],  # NPU
        # Middle layers...
        "layer_28":  ["DmlExecutionProvider"],      # iGPU overflow
        "layer_29":  ["DmlExecutionProvider"],      # iGPU overflow
        "layer_30":  ["DmlExecutionProvider"],      # iGPU overflow
        "layer_31":  ["DmlExecutionProvider"],      # iGPU overflow
        "lm_head":   ["DmlExecutionProvider"],
    },
    persist_sessions=True
)
```

---

## 6. Troubleshooting

### 6.1 Common NPU Issues

#### Issue 1: "VitisAIExecutionProvider not available"

**Symptoms:**
```
ValueError: Provider VitisAIExecutionProvider not found
```

**Causes:**
1. ONNX Runtime VitisAI not installed
2. NPU driver not installed
3. Hardware not present

**Solutions:**
```bash
# Install VitisAI provider
pip install onnxruntime-vitisai

# Verify driver installation
xclinfo  # Should show NPU devices

# Check Windows Device Manager for "AMD XDNA NPU"
```

#### Issue 2: "Graph compilation failed"

**Symptoms:**
```
RuntimeError: VitisAI graph compilation failed
```

**Causes:**
1. Model incompatible with NPU
2. Unsupported operators
3. Memory limit exceeded

**Solutions:**
```python
# Try with reduced optimization
config = SessionConfig(
    npu_compilation_mode="fast",
    npu_weight_preload=False
)

# Or fall back to iGPU for problematic components
component_providers={
    "layer_*": ["DmlExecutionProvider"]  # Use iGPU instead
}
```

#### Issue 3: "Out of NPU memory"

**Symptoms:**
```
RuntimeError: NPU memory allocation failed
```

**Solutions:**
```python
# Set explicit memory limit
config = NPUConfig(memory_limit_mb=2048)

# Disable weight preloading for large models
config = NPUConfig(weight_preload=False)

# Use component-specific routing to distribute load
component_providers={
    "embedding": ["DmlExecutionProvider"],  # Offload to iGPU
    "layer_0":   ["VitisAIExecutionProvider"],
    # ... only some layers on NPU
    "lm_head":   ["DmlExecutionProvider"],
}
```

#### Issue 4: "Slow first inference"

**Symptoms:**
- First token takes 5-30 seconds
- Subsequent tokens fast

**Cause:** NPU graph compilation happens on first run

**Solutions:**
```python
# Enable graph capture for reuse
config = NPUConfig(graph_capture=True)

# Use cached compilation
config = NPUConfig(compilation_mode="optimized")  # One-time cost

# Warm up the pipeline before actual use
pipeline.generate("warmup", max_new_tokens=1)
```

### 6.2 Debug Mode

Enable detailed NPU logging:

```python
from pipeline_utils.npu_config import NPUConfig

config = NPUConfig(
    debug_mode=True,
    enable_profiling=True
)

# This outputs detailed VitisAI logs
```

### 6.3 Performance Profiling

```python
from pipeline_utils.npu_config import NPUConfig

config = NPUConfig(enable_profiling=True)

# After inference, check profiling output
# Profile data written to: ./vitisai_profile.json
```

### 6.4 Fallback Strategy

Always have a fallback plan:

```python
from pipeline_utils import is_npu_available, get_provider_for_device

def get_optimal_providers():
    """Get best available providers with fallback."""
    if is_npu_available():
        return ["VitisAIExecutionProvider", "CPUExecutionProvider"]

    available = get_provider_for_device("auto")
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "DmlExecutionProvider" in available:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]

    return ["CPUExecutionProvider"]

providers = get_optimal_providers()
print(f"Using providers: {providers}")
```

---

## 7. API Reference

### 7.1 Core Classes

#### NPUConfig

```python
@dataclass
class NPUConfig:
    """Comprehensive NPU configuration."""

    graph_capture: bool = True
        """Enable graph capture for reuse (default: True)"""

    weight_preload: bool = True
        """Preload weights to Tile SRAM (default: True)"""

    compilation_mode: NPUCompilationMode = NPUCompilationMode.DEFAULT
        """Compilation strategy: fast, default, optimized"""

    enable_profiling: bool = False
        """Enable NPU performance profiling"""

    debug_mode: bool = False
        """Enable debug logging"""

    memory_limit_mb: Optional[int] = None
        """Maximum NPU memory allocation"""

    num_cores: int = -1
        """Number of NPU cores (-1 = all)"""

    def create_session_options() -> ort.SessionOptions
    def to_dict() -> Dict[str, Any]
    @classmethod
    def from_dict(data: Dict[str, Any]) -> NPUConfig
```

#### SessionConfig

```python
@dataclass
class SessionConfig:
    """ONNX Runtime session configuration."""

    providers: List[str] = ["CPUExecutionProvider"]
        """Execution providers in priority order"""

    optimization_level: str = "DISABLE"
        """Graph optimization: DISABLE, BASIC, EXTENDED, ALL"""

    inter_op_threads: int = 0
        """Threads for parallel op execution (0 = auto)"""

    intra_op_threads: int = 0
        """Threads within single op (0 = auto)"""

    memory_limit_mb: Optional[int] = None
        """Arena allocator memory limit"""

    disable_shape_check: bool = True
        """Disable node shape checking for dynamic shapes"""

    # NPU-specific (when VitisAIExecutionProvider used)
    npu_graph_capture: bool = True
    npu_weight_preload: bool = True
    npu_compilation_mode: str = "default"

    def create_session_options() -> ort.SessionOptions
```

#### NPUCompilationMode

```python
class NPUCompilationMode(str, Enum):
    DEFAULT = "default"      # Balanced
    FAST = "fast"            # Fast compilation
    OPTIMIZED = "optimized"  # Best runtime
```

#### NPUOffloadStrategy

```python
class NPUOffloadStrategy(str, Enum):
    LAYER_BY_LAYER = "layer_by_layer"     # Minimal memory
    PIPELINE_PARALLEL = "pipeline_parallel" # Overlap compute/transfer
    STATIC_PARTITION = "static_partition"  # Pre-allocated
```

### 7.2 Utility Functions

#### is_npu_available()

```python
def is_npu_available() -> bool:
    """Check if AMD XDNA NPU is available."""
    # Returns True if VitisAIExecutionProvider is registered
```

#### get_npu_capabilities()

```python
def get_npu_capabilities() -> NPUCapabilities:
    """Query NPU hardware capabilities."""
    # Returns:
    #   NPUCapabilities(
    #       available: bool,
    #       driver_version: str,
    #       num_cores: int,
    #       memory_size_mb: int,
    #       tile_sram_mb: int
    #   )
```

#### validate_npu_config()

```python
def validate_npu_config(
    config: NPUConfig,
    capabilities: NPUCapabilities
) -> Tuple[bool, List[str]]:
    """
    Validate NPU configuration.

    Returns:
        (is_valid, warnings_list)
    """
```

#### get_recommended_npu_config()

```python
def get_recommended_npu_config(
    model_size_mb: float
) -> NPUConfig:
    """
    Get recommended NPU config based on model size.

    Args:
        model_size_mb: Model size in megabytes

    Returns:
        NPUConfig with optimal settings
    """
```

### 7.3 SequentialModelPipeline Parameters

```python
SequentialModelPipeline(
    model_path: str,
        """Path to split model directory or monolithic ONNX file"""

    tokenizer_path: Optional[str] = None,
        """Path to tokenizer directory"""

    providers: Optional[List[str]] = None,
        """Default execution providers"""

    component_providers: Optional[Dict[str, List[str]]] = None,
        """Per-component provider routing (supports wildcards)"""

    phase_providers: Optional[Dict[str, List[str]]] = None,
        """Phase-based routing: prefill/decode"""

    verbose: bool = True,
        """Enable verbose logging"""

    persist_sessions: bool = False
        """Keep sessions loaded (required for NPU performance)"""
)
```

---

## 8. Examples

### 8.1 Basic NPU Inference

```python
#!/usr/bin/env python3
"""Simple NPU inference example."""

from grok_inference_split_model_pipeline import SequentialModelPipeline
from pipeline_utils import is_npu_available

# Check NPU availability
if not is_npu_available():
    print("NPU not available, falling back to CPU")
    providers = ["CPUExecutionProvider"]
else:
    print("Using NPU acceleration")
    providers = ["VitisAIExecutionProvider", "CPUExecutionProvider"]

# Create pipeline with persistence (critical for NPU)
pipeline = SequentialModelPipeline(
    model_path="split_output/qwen2.5-0.5b",
    tokenizer_path="Qwen/Qwen2.5-0.5B",
    providers=providers,
    persist_sessions=True,  # Keep sessions loaded
    verbose=True
)

# Generate text
output = pipeline.generate(
    prompt="What is machine learning?",
    max_new_tokens=50
)

print("\nGenerated text:")
print(output)

# Cleanup
pipeline.cleanup()
```

### 8.2 Hybrid NPU+iGPU Dispatch

```python
#!/usr/bin/env python3
"""Hybrid dispatch example using NPU for layers, iGPU for I/O."""

from grok_inference_split_model_pipeline import SequentialModelPipeline

pipeline = SequentialModelPipeline(
    model_path="split_output/qwen2.5-0.5b",
    tokenizer_path="Qwen/Qwen2.5-0.5B",

    # Component-specific provider routing
    component_providers={
        "embedding": ["DmlExecutionProvider"],      # iGPU for token embed
        "layer_*":   ["VitisAIExecutionProvider"],  # NPU for transformer
        "norm":      ["CPUExecutionProvider"],      # CPU is fine for norm
        "lm_head":   ["DmlExecutionProvider"],      # iGPU for output
    },

    # Phase-based optimization
    phase_providers={
        "prefill": ["VitisAIExecutionProvider"],    # NPU compute power
        "decode":  ["DmlExecutionProvider"],        # iGPU bandwidth
    },

    persist_sessions=True,
    verbose=True
)

output = pipeline.generate("Explain neural networks", max_new_tokens=30)
print(output)
pipeline.cleanup()
```

### 8.3 Advanced NPU Configuration

```python
#!/usr/bin/env python3
"""Advanced NPU configuration with custom settings."""

from pipeline_utils import (
    SessionManager,
    SessionConfig,
    NPUConfig,
    NPUCompilationMode,
    get_npu_capabilities,
    validate_npu_config
)

# Query hardware capabilities
caps = get_npu_capabilities()
print(f"NPU: {caps.num_cores} cores, {caps.memory_size_mb}MB memory")
print(f"Tile SRAM: {caps.tile_sram_mb}MB")

# Create advanced NPU configuration
npu_config = NPUConfig(
    graph_capture=True,
    weight_preload=True if caps.tile_sram_mb >= 32 else False,
    compilation_mode=NPUCompilationMode.OPTIMIZED,
    enable_profiling=True,
    debug_mode=False,
    memory_limit_mb=min(4096, caps.memory_size_mb),
    num_cores=-1  # Use all available cores
)

# Validate configuration
is_valid, warnings = validate_npu_config(npu_config, caps)
if warnings:
    print("Configuration warnings:")
    for w in warnings:
        print(f"  - {w}")

# Create session config with NPU options
session_config = SessionConfig(
    providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
    optimization_level="DISABLE",
    inter_op_threads=0,
    intra_op_threads=0,
    memory_limit_mb=4096,
    disable_shape_check=True,
    npu_graph_capture=npu_config.graph_capture,
    npu_weight_preload=npu_config.weight_preload,
    npu_compilation_mode=npu_config.compilation_mode.value
)

# Initialize session manager
manager = SessionManager("split_output/my_model", session_config)

# Load all sessions (eager loading for NPU)
manager.load_all()

# Get memory usage
mem = manager.get_memory_usage()
print(f"Loaded: {mem['loaded_count']}/{mem['total_count']} sessions")
print(f"Memory: {mem['total_mb']}MB")

# Use sessions for inference...
session = manager.get_session("layer_0")

# Cleanup when done
manager.unload_all()
```

### 8.4 Multi-Model NPU Server

```python
#!/usr/bin/env python3
"""Example: Serving multiple models with NPU."""

from typing import Dict, Optional
from pipeline_utils import SessionManager, SessionConfig, is_npu_available

class NPUModelServer:
    """Simple multi-model server with NPU acceleration."""

    def __init__(self, npu_optimized: bool = True):
        self.managers: Dict[str, SessionManager] = {}
        self.npu_available = is_npu_available()

        if npu_optimized and self.npu_available:
            self.config = SessionConfig(
                providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
                npu_graph_capture=True,
                npu_weight_preload=True,
                npu_compilation_mode="default"
            )
            print("NPU optimization enabled")
        else:
            self.config = SessionConfig(
                providers=["CPUExecutionProvider"]
            )
            print("Using CPU execution")

    def load_model(self, model_id: str, model_dir: str) -> None:
        """Load a model into the server."""
        print(f"Loading model '{model_id}' from {model_dir}")

        manager = SessionManager(model_dir, self.config)
        manager.load_all()  # Eager load for NPU

        self.managers[model_id] = manager
        print(f"  Model loaded: {manager}")

    def get_session(self, model_id: str, component: str):
        """Get a session for a specific model component."""
        if model_id not in self.managers:
            raise ValueError(f"Model '{model_id}' not loaded")

        return self.managers[model_id].get_session(component)

    def unload_model(self, model_id: str) -> None:
        """Unload a model to free memory."""
        if model_id in self.managers:
            self.managers[model_id].unload_all()
            del self.managers[model_id]
            print(f"Model '{model_id}' unloaded")

    def get_memory_usage(self) -> Dict[str, any]:
        """Get memory usage across all loaded models."""
        return {
            model_id: manager.get_memory_usage()
            for model_id, manager in self.managers.items()
        }

    def shutdown(self) -> None:
        """Shutdown server and release all resources."""
        for model_id in list(self.managers.keys()):
            self.unload_model(model_id)
        print("Server shutdown complete")


# Usage example
if __name__ == "__main__":
    server = NPUModelServer(npu_optimized=True)

    # Load models
    server.load_model("qwen-0.5b", "split_output/qwen2.5-0.5b")
    server.load_model("gpt2-small", "split_output/gpt2")

    # Get memory stats
    mem = server.get_memory_usage()
    print(f"\nMemory Usage:")
    for model_id, stats in mem.items():
        print(f"  {model_id}: {stats['total_mb']}MB")

    # Use for inference...
    session = server.get_session("qwen-0.5b", "layer_0")

    # Cleanup
    server.shutdown()
```

### 8.5 Command-Line NPU Benchmarking

```bash
#!/bin/bash
# benchmark_npu.sh - Compare NPU vs CPU vs iGPU performance

MODEL_DIR="split_output/qwen2.5-0.5b"
PROMPT="Explain the concept of attention in neural networks"
TOKENS=50

echo "=============================================="
echo "  NPU Performance Benchmark"
echo "=============================================="

echo -e "\n[1/3] Testing NPU (VitisAI)..."
time python grok_inference_split_model_pipeline.py \
    --model-dir "$MODEL_DIR" \
    --prompt "$PROMPT" \
    --tokens $TOKENS \
    --device npu \
    --persist \
    --quiet

echo -e "\n[2/3] Testing iGPU (DirectML)..."
time python grok_inference_split_model_pipeline.py \
    --model-dir "$MODEL_DIR" \
    --prompt "$PROMPT" \
    --tokens $TOKENS \
    --device gpu \
    --quiet

echo -e "\n[3/3] Testing CPU..."
time python grok_inference_split_model_pipeline.py \
    --model-dir "$MODEL_DIR" \
    --prompt "$PROMPT" \
    --tokens $TOKENS \
    --device cpu \
    --quiet

echo -e "\n=============================================="
echo "  Benchmark Complete"
echo "=============================================="
```

### 8.6 Dynamic Provider Selection

```python
#!/usr/bin/env python3
"""Dynamic provider selection based on model size and hardware."""

from pipeline_utils import (
    is_npu_available,
    get_npu_capabilities,
    get_available_providers,
    NPUConfig,
    get_recommended_npu_config
)
import os

def get_model_size_mb(model_dir: str) -> float:
    """Calculate total model size in MB."""
    total_bytes = 0
    for root, dirs, files in os.walk(model_dir):
        for f in files:
            if f.endswith('.onnx'):
                total_bytes += os.path.getsize(os.path.join(root, f))
    return total_bytes / (1024 * 1024)

def select_optimal_providers(model_dir: str) -> tuple:
    """
    Select optimal execution providers based on model and hardware.

    Returns:
        (providers_list, persist_sessions_bool, config_dict)
    """
    model_size = get_model_size_mb(model_dir)
    print(f"Model size: {model_size:.1f}MB")

    available = get_available_providers()
    print(f"Available providers: {available}")

    # Check NPU capabilities
    if is_npu_available():
        caps = get_npu_capabilities()
        print(f"NPU: {caps.memory_size_mb}MB total, {caps.tile_sram_mb}MB SRAM")

        # Get recommended config
        config = get_recommended_npu_config(model_size)

        if model_size <= caps.tile_sram_mb:
            print("Model fits in Tile SRAM - optimal NPU performance!")
            return (
                ["VitisAIExecutionProvider", "CPUExecutionProvider"],
                True,  # persist
                config.to_dict()
            )
        elif model_size <= caps.memory_size_mb:
            print("Model fits in NPU memory - good performance")
            return (
                ["VitisAIExecutionProvider", "CPUExecutionProvider"],
                True,  # persist
                config.to_dict()
            )
        else:
            print("Model exceeds NPU memory - consider hybrid or iGPU")
            # Hybrid: NPU for layers, iGPU for large I/O components
            return (
                ["VitisAIExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"],
                True,  # persist
                {"weight_preload": False}  # Can't preload all weights
            )

    # NPU not available - check for GPU
    if "CUDAExecutionProvider" in available:
        print("Using CUDA GPU acceleration")
        return (["CUDAExecutionProvider", "CPUExecutionProvider"], True, {})

    if "DmlExecutionProvider" in available:
        print("Using DirectML GPU acceleration")
        return (["DmlExecutionProvider", "CPUExecutionProvider"], True, {})

    # Fallback to CPU
    print("Using CPU execution")
    return (["CPUExecutionProvider"], False, {})  # No need to persist on CPU

# Usage
if __name__ == "__main__":
    providers, persist, config = select_optimal_providers("split_output/qwen2.5-0.5b")
    print(f"\nSelected providers: {providers}")
    print(f"Persist sessions: {persist}")
    print(f"NPU config: {config}")
```

---

## Appendix A: Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────┐
│                      NPU QUICK REFERENCE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ALWAYS use --persist for NPU inference                         │
│                                                                  │
│  Provider Priority: VitisAI > DML/CUDA > CPU                    │
│                                                                  │
│  SessionConfig NPU Options:                                     │
│    npu_graph_capture=True     # Cache compiled graph            │
│    npu_weight_preload=True    # Load to Tile SRAM               │
│    npu_compilation_mode="default"  # Balance compile/runtime    │
│                                                                  │
│  Pattern Matching: "layer_*" matches all layer_X components     │
│                                                                  │
│  Phase Dispatch: prefill=NPU (compute), decode=iGPU (bandwidth) │
│                                                                  │
│  Debug: NPUConfig(debug_mode=True, enable_profiling=True)       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Appendix B: Related Documentation

- [MASTER_REFERENCE.md](./MASTER_REFERENCE.md) - Complete system architecture
- [README.md](./README.md) - Project overview
- [WALKTHROUGH.md](./WALKTHROUGH.md) - Usage guide
- [WALKTHROUGH_MULTI_PROVIDER.md](./WALKTHROUGH_MULTI_PROVIDER.md) - Multi-provider setup

---

*Document generated: 2026-02-17 | AMD XDNA NPU Integration Guide v1.0*
