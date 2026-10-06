# Appendix - Technical Reference

**Outcome Statement:** This appendix provides detailed technical reference material including API documentation, troubleshooting guides, and critical architectural decisions.

---

## Table of Contents

1. [API Reference](#api-reference)
2. [NPU Monolithic Requirement](#npu-monolithic-requirement)
3. [Decision Matrix](#decision-matrix)
4. [Troubleshooting](#troubleshooting)
5. [Glossary](#glossary)

---

## API Reference

**Outcome:** Complete API documentation for all public interfaces.

### Splitting API

#### split_onnx_model

```python
def split_onnx_model(params: argparse.Namespace) -> SplitResult:
    """
    Split an ONNX model into semantic components.

    Args:
        params: Command-line parameters or namespace with:
            - model: Path to source ONNX model
            - output: Output directory for split components
            - split_only: If True, skip inference testing

    Returns:
        SplitResult with:
            - output_dir: Directory containing split files
            - num_layers: Number of transformer layers
            - component_files: Dict[name → path]
            - model_info: Metadata dictionary

    Example:
        >>> import argparse
        >>> params = argparse.Namespace(
        ...     model="qwen2.5-0.5b.onnx",
        ...     output="split_output/qwen2.5-0.5b",
        ...     split_only=True
        ... )
        >>> result = split_onnx_model(params)
        >>> print(f"Created {result.num_layers} layers")
    """
```

#### SplitResult

```python
@dataclass
class SplitResult:
    """Result of semantic model splitting."""

    output_dir: str
        """Directory containing split component files."""

    num_layers: int
        """Number of transformer layers extracted."""

    component_files: Dict[str, str]
        """Mapping of component name to file path.
        Example: {
            'embeddings': 'split_output/embeddings.onnx',
            'layer_0': 'split_output/layer_0.onnx',
            ...
            'lm_head': 'split_output/lm_head.onnx'
        }"""

    model_info: Dict[str, Any]
        """Metadata including:
            - architecture: Detected architecture type
            - hidden_size: Model hidden dimension
            - vocab_size: Vocabulary size
            - num_heads: Number of attention heads
        """
```

---

### Inference API

#### SequentialModelPipeline

```python
class SequentialModelPipeline:
    """
    Orchestrates sequential execution of split model components.

    Args:
        model_path: Path to split components directory or monolithic ONNX
        tokenizer_path: Path to tokenizer directory (for HF tokenizers)
        providers: List of ONNX Runtime execution providers
        persist_sessions: If True, pre-load all sessions (recommended for GPU/NPU)
        verbose: Enable verbose logging

    Example:
        >>> # Memory-optimized mode (CPU)
        >>> pipeline = SequentialModelPipeline(
        ...     model_path="split_output/my_model",
        ...     providers=["CPUExecutionProvider"],
        ...     persist_sessions=False
        ... )
        >>>
        >>> # Latency-optimized mode (GPU)
        >>> pipeline = SequentialModelPipeline(
        ...     model_path="split_output/my_model",
        ...     providers=["CUDAExecutionProvider"],
        ...     persist_sessions=True
        ... )
    """

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text given a prompt.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (1.0 = greedy)
            top_p: Nucleus sampling parameter

        Returns:
            Generated text (prompt + completion)
        """

    def cleanup(self) -> None:
        """Release all resources (sessions, memory, etc.)."""
```

---

### Session Management API

#### SessionManager

```python
class SessionManager:
    """
    Manages ONNX Runtime session lifecycle.

    Args:
        model_dir: Directory containing split component files
        config: SessionConfig with provider and optimization settings

    Example:
        >>> config = SessionConfig(
        ...     providers=["CUDAExecutionProvider"],
        ...     optimization_level="DISABLE"
        ... )
        >>> manager = SessionManager("split_output/my_model", config)
        >>> manager.load_all()  # Eager loading for GPU
        >>> session = manager.get_session("layer_0")
        >>> # ... use session ...
        >>> manager.unload_all()
    """

    def get_session(self, name: str) -> ort.InferenceSession:
        """
        Get session for a component.

        In sequential mode: Loads from disk if not cached
        In persistent mode: Returns pre-loaded session
        """

    def load_all(self) -> None:
        """Load all sessions (eager loading)."""

    def unload_all(self) -> None:
        """Unload all sessions."""

    def get_memory_usage(self) -> Dict[str, Any]:
        """Return memory usage statistics."""
```

#### SessionConfig

```python
@dataclass
class SessionConfig:
    """
    ONNX Runtime session configuration.

    Attributes:
        providers: Execution providers in priority order
            Example: ["CUDAExecutionProvider", "CPUExecutionProvider"]

        optimization_level: Graph optimization level
            Options: "DISABLE", "BASIC", "EXTENDED", "ALL"

        inter_op_threads: Threads for parallel op execution (0 = auto)
        intra_op_threads: Threads within single op (0 = auto)

        memory_limit_mb: Arena allocator memory limit (None = unlimited)
        disable_shape_check: Disable node shape checking (required for dynamic shapes)

        # NPU-specific (when using VitisAIExecutionProvider)
        npu_graph_capture: Enable graph caching (default: True)
        npu_weight_preload: Preload weights to Tile SRAM (default: True)
        npu_compilation_mode: "fast", "default", or "optimized"
    """
```

---

### Utility Functions

#### is_npu_available

```python
def is_npu_available() -> bool:
    """
    Check if AMD XDNA NPU is available.

    Returns:
        True if VitisAIExecutionProvider is registered and NPU hardware detected

    Example:
        >>> if is_npu_available():
        ...     print("NPU acceleration available")
        ... else:
        ...     print("Falling back to CPU/GPU")
    """
```

#### get_npu_capabilities

```python
def get_npu_capabilities() -> NPUCapabilities:
    """
    Query NPU hardware capabilities.

    Returns:
        NPUCapabilities dataclass with:
            - available: bool
            - driver_version: str
            - num_cores: int
            - memory_size_mb: int
            - tile_sram_mb: int

    Example:
        >>> caps = get_npu_capabilities()
        >>> print(f"NPU: {caps.num_cores} cores, {caps.memory_size_mb}MB")
        >>> print(f"Tile SRAM: {caps.tile_sram_mb}MB")
    """
```

---

## NPU Monolithic Requirement

**Outcome:** Understand the technical reasons why NPUs require monolithic models.

### Critical Finding

**NPUs (AMD XDNA, VitisAI) are fundamentally incompatible with sequential load/unload patterns. Using split models on NPU results in 5-100× performance regression.**

---

### Technical Explanation

#### 1. Ahead-of-Time Compilation

**GPU Approach (JIT):**
```
GPU receives subgraph → JIT compile (~10 μs) → Execute → Discard
```

**NPU Approach (AOT):**
```
NPU receives entire graph → AOT compile entire graph (~5-50 ms) → Execute
```

**Impact:** Splitting breaks whole-graph optimizations. Each component would require separate compilation, multiplying compilation overhead by number of components (31× for a 28-layer model).

---

#### 2. Memory Movement Overhead

**GPU Memory Model:**
```
Unified memory address space
Component A output → GPU memory → Component B input (zero-copy possible)
```

**NPU Memory Model:**
```
Isolated memory per program
Component A output → NPU SRAM → DDR RAM → CPU → DDR RAM → NPU SRAM → Component B input
```

**Impact:** Every component boundary forces expensive memory transfers. For a 28-layer model: 28 × (NPU→DDR→CPU→DDR→NPU) transfers = catastrophic bandwidth overhead.

---

#### 3. Context Switch Cost

| Operation | GPU | NPU | Ratio |
|-----------|-----|-----|-------|
| Kernel launch | ~10 μs | - | - |
| Program load | - | ~5-50 ms | 500-5000× |
| Memory transfer | ~10-50 μs | ~100-500 μs | 10-50× |

**Impact:** Sequential mode requires loading 31 components per token. On NPU: 31 × 5-50ms = 155-1550ms per token vs GPU: 31 × 10μs = 0.31ms per token.

---

#### 4. Quantization Coherence

**Monolithic Quantization:**
```
Entire model → Global calibration → Consistent INT8/INT4 quantization
```

**Split Quantization:**
```
Component 0 → Local calibration → Quantized
Component 1 → Local calibration → Quantized (different scale!)
...
Result: Inconsistent quantization scales across components
```

**Impact:** NPU quantization requires globally calibrated scales. Split models break this calibration, causing accuracy degradation.

---

#### 5. KV-Cache Isolation

**Monolithic KV-Cache:**
```
All layers share unified KV-cache memory
Layer N reads KV from Layer N-1 (zero-copy, same address space)
```

**Split KV-Cache:**
```
Each session has isolated memory
Layer N KV-cache → Session N memory → CPU → Session N+1 memory
```

**Impact:** KV-cache transfer overhead adds 100-500μs per layer per token. For 28 layers: 2.8-14ms overhead per token.

---

#### 6. Architecture Mismatch

| Characteristic | GPU | NPU |
|----------------|-----|-----|
| **Execution Model** | Kernel dispatch | Fixed dataflow |
| **Graph Flexibility** | Any subgraph OK | Entire graph required |
| **Memory Addressing** | Unified, shared | Isolated, program-specific |
| **Best Workload** | Flexible, dynamic | Fixed, predictable |

**Impact:** NPUs are designed for fixed, monolithic inference graphs. The flexible load/unload pattern of sequential inference fundamentally mismatches NPU architecture.

---

### Performance Comparison

**Monolithic on NPU (Correct):**
```
Compilation: One-time 5-30s
First Token: 100-500ms (after compilation)
Per Token: 20-100ms
Power: <5W
```

**Split on NPU (INCORRECT - DO NOT USE):**
```
Per-token Compilation: 31 components × 5-50ms = 155-1550ms
Memory Transfers: 31 × (NPU→DDR→CPU→DDR→NPU) = 100-500ms
KV-Cache Transfers: 28 layers × 100-500μs = 2.8-14ms
Total Per Token: 500ms-5s (5-100× slower!)
Power: Higher due to inefficiency
```

---

### Recommended NPU Flow

```
1. Check if model fits in NPU memory
   │
   ├── YES → Use MONOLITHIC mode
   │   └── python inference.py --model model.onnx --device npu --persist
   │
   └── NO → Quantize to fit, or fall back to GPU/CPU
       ├── Option A: INT4 quantization (~75% size reduction)
       └── Option B: Use GPU/CPU with split + sequential mode
```

---

## Decision Matrix

**Outcome:** Quick reference for configuration selection.

### Hardware × Mode Selection

| Hardware | Memory Constraint | Model Size | Recommended Mode |
|----------|-------------------|------------|------------------|
| CPU | Tight | Any | Split + Sequential |
| CPU | Relaxed | < RAM | Monolithic or Split + Persistent |
| GPU | Fits in VRAM | < VRAM | Monolithic + Persistent |
| GPU | Exceeds VRAM | > VRAM | Split + Persistent + IO-Bound |
| NPU | Any | < NPU memory | **Monolithic ONLY** |
| NPU | Any | > NPU memory | Quantize or fall back to GPU/CPU |

### Configuration Quick Reference

```
CPU, Memory-Constrained:
  providers=["CPUExecutionProvider"]
  persist_sessions=False

CPU, Relaxed Memory:
  providers=["CPUExecutionProvider"]
  persist_sessions=True

GPU, Model Fits VRAM:
  providers=["CUDAExecutionProvider"]
  persist_sessions=True

GPU, Model Exceeds VRAM:
  providers=["CUDAExecutionProvider"]
  persist_sessions=True
  # Use split model

NPU (AMD Ryzen AI):
  providers=["VitisAIExecutionProvider"]
  persist_sessions=True
  # MUST use monolithic model!
```

---

## Troubleshooting

**Outcome:** Resolve common issues.

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| **"Provider not found"** | Missing acceleration package | Install appropriate ORT package (e.g., `pip install onnxruntime-gpu`) |
| **"NPU not available"** | Driver not installed or hardware absent | Verify with `xclinfo` or check Device Manager |
| **"No suitable split point"** | Model has non-linear topology | Try layer-based splitting or adjust split ratio |
| **"PSNR verification failed"** | Numerical precision issues | Lower threshold or check model integrity |
| **"Out of memory during split"** | Both original + split in memory | Use `--split-only` mode, close other apps |
| **"Graph compilation failed" (NPU)** | Model incompatible with NPU | Try `fast` compilation mode or fall back to GPU |
| **Slow first inference (NPU)** | Graph compilation on first run | Enable `graph_capture=True` for caching |
| **5-100× slowdown on NPU** | Using split model on NPU | **Switch to monolithic model immediately** |

---

## Glossary

| Term | Definition |
|------|------------|
| **AOT (Ahead-of-Time)** | Compilation before execution (NPU approach) |
| **JIT (Just-in-Time)** | Compilation during execution (GPU approach) |
| **KV-Cache** | Key-Value cache for autoregressive attention |
| **Monolithic** | Entire model in single file/execution |
| **PSNR** | Peak Signal-to-Noise Ratio (accuracy metric) |
| **Semantic Splitting** | Decomposition at architectural boundaries |
| **Sequential Mode** | Load/run/unload components one at a time |
| **Persistent Mode** | Pre-load all sessions, keep resident |
| **IO-Bound** | Zero-copy device-resident tensor passing |
| **Tile SRAM** | On-chip NPU memory (16-32 MB on AMD XDNA) |
| **VitisAI** | AMD's NPU execution provider for ONNX Runtime |

---

*End of Appendix. For main documentation, see [docs/README.md](../README.md).*
