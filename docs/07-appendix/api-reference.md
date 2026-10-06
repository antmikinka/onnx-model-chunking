# API Reference

**Outcome Statement:** Complete API documentation for all public interfaces in the ONNX Model Chunking system.

---

## Table of Contents

1. [Splitting API](#splitting-api)
2. [Inference API](#inference-api)
3. [Session Management API](#session-management-api)
4. [Utility Functions](#utility-functions)
5. [Data Classes](#data-classes)

---

## Splitting API

### split_onnx_model

```python
def split_onnx_model(params: argparse.Namespace) -> SplitResult:
    """
    Split an ONNX model into semantic components.

    Args:
        params: Namespace with splitting parameters:
            - model (str): Path to source ONNX model
            - output (str): Output directory for split components
            - split_only (bool): If True, skip inference testing

    Returns:
        SplitResult containing split information

    Example:
        >>> import argparse
        >>> params = argparse.Namespace(
        ...     model="qwen2.5-0.5b.onnx",
        ...     output="split_output/qwen2.5-0.5b",
        ...     split_only=True
        ... )
        >>> result = split_onnx_model(params)
        >>> print(f"Created {result.num_layers} layers")
        28
        >>> print(f"Components: {list(result.component_files.keys())}")
        ['embeddings', 'layer_0', ..., 'layer_27', 'norm', 'lm_head']
    """
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | Required | Path to source ONNX model |
| `output` | str | Required | Output directory for split components |
| `split_only` | bool | False | Skip inference testing if True |

**Returns:**
- `SplitResult` dataclass with split information

**Raises:**
- `FileNotFoundError`: Model file not found
- `ValueError`: Invalid model or unsupported architecture
- `RuntimeError`: Splitting failed

---

### detect_architecture

```python
def detect_architecture_from_onnx(model: onnx.ModelProto) -> ArchitectureInfo:
    """
    Detect model architecture from ONNX node names.

    Args:
        model: Loaded ONNX model

    Returns:
        ArchitectureInfo with detected architecture type and layer count

    Supported Architectures:
        - LLaMA family (LLaMA, Mistral, Phi, Gemma)
        - Qwen family (Qwen, Qwen2, Qwen2.5)
        - GPT-2 family
    """
```

---

## Inference API

### SequentialModelPipeline

```python
class SequentialModelPipeline:
    """
    Orchestrates sequential execution of split model components.

    Args:
        model_path: Path to split components directory or monolithic ONNX file
        tokenizer_path: Path to tokenizer directory (for HuggingFace tokenizers)
        providers: List of ONNX Runtime execution providers (default: auto-detect)
        persist_sessions: If True, pre-load all sessions (default: False)
        component_providers: Optional dict for per-component provider routing
        phase_providers: Optional dict for phase-based provider routing
        verbose: Enable verbose logging (default: True)

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
        >>>
        >>> # Generate text
        >>> output = pipeline.generate("What is AI?", max_new_tokens=50)
        >>> print(output)
    """
```

#### generate

```python
def generate(
    self,
    prompt: str,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1
) -> str:
    """
    Generate text given a prompt.

    Args:
        prompt: Input text prompt
        max_new_tokens: Maximum tokens to generate (default: 20)
        temperature: Sampling temperature (default: 1.0)
            - 1.0: Standard sampling
            - <1.0: More deterministic
            - >1.0: More random
        top_p: Nucleus sampling parameter (default: 0.9)
            - 0.9: Sample from top 90% of probability mass
        repetition_penalty: Penalty for token repetition (default: 1.1)
            - 1.0: No penalty
            - >1.0: Discourage repetition

    Returns:
        Generated text (prompt + completion)

    Example:
        >>> output = pipeline.generate(
        ...     "Explain machine learning",
        ...     max_new_tokens=100,
        ...     temperature=0.7,
        ...     top_p=0.9
        ... )
        >>> print(output)
        "Explain machine learning: Machine learning is..."
    """
```

#### cleanup

```python
def cleanup(self) -> None:
    """
    Release all resources (sessions, memory, device buffers).

    Should be called when pipeline is no longer needed.
    """
```

---

## Session Management API

### SessionManager

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
```

#### get_session

```python
def get_session(self, name: str) -> ort.InferenceSession:
    """
    Get session for a component.

    Args:
        name: Component name (e.g., "layer_0", "embeddings")

    Returns:
        ONNX Runtime InferenceSession

    Behavior:
        - Sequential mode: Loads from disk if not cached
        - Persistent mode: Returns pre-loaded session
    """
```

#### load_all

```python
def load_all(self) -> None:
    """
    Load all sessions (eager loading).

    Use for persistent mode to pre-load all components at startup.
    """
```

#### unload_all

```python
def unload_all(self) -> None:
    """
    Unload all sessions.

    Releases all loaded sessions and frees memory.
    """
```

#### get_memory_usage

```python
def get_memory_usage(self) -> Dict[str, Any]:
    """
    Return memory usage statistics.

    Returns:
        Dict with:
            - loaded_count: Number of loaded sessions
            - total_count: Total number of sessions
            - total_mb: Total memory usage in MB
    """
```

---

### SessionConfig

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
            Note: "DISABLE" required for dynamic shapes

        inter_op_threads: Threads for parallel op execution
            0 = auto-detect, >0 = explicit count

        intra_op_threads: Threads within single op
            0 = auto-detect, >0 = explicit count

        memory_limit_mb: Arena allocator memory limit
            None = unlimited, int = limit in MB

        disable_shape_check: Disable node shape checking
            True = required for dynamic shapes

        # NPU-specific (when using VitisAIExecutionProvider)
        npu_graph_capture: Enable graph caching (default: True)
        npu_weight_preload: Preload weights to Tile SRAM (default: True)
        npu_compilation_mode: Compilation strategy
            Options: "fast", "default", "optimized"
    """
```

---

### NPUConfig

```python
@dataclass
class NPUConfig:
    """
    AMD XDNA NPU-specific configuration.

    Attributes:
        graph_capture: Enable graph caching (default: True)
            Required for NPU performance

        weight_preload: Preload weights to Tile SRAM (default: True)
            Disable if model exceeds Tile SRAM capacity

        compilation_mode: Compilation strategy (default: "default")
            - "fast": Fastest compilation, lower runtime performance
            - "default": Balanced compilation and runtime
            - "optimized": Slowest compilation, best runtime performance

        enable_profiling: Enable NPU performance profiling (default: False)
            Profile data written to ./vitisai_profile.json

        debug_mode: Enable debug logging (default: False)
            Verbose NPU/ VitisAI output

        memory_limit_mb: Maximum NPU memory allocation (default: None)
            None = use all available, int = limit in MB

        num_cores: Number of NPU cores to use (default: -1)
            -1 = use all available cores
    """
```

---

## Utility Functions

### is_npu_available

```python
def is_npu_available() -> bool:
    """
    Check if AMD XDNA NPU is available.

    Returns:
        True if VitisAIExecutionProvider is registered and NPU detected

    Example:
        >>> if is_npu_available():
        ...     print("NPU acceleration available")
        ... else:
        ...     print("Falling back to CPU/GPU")
    """
```

---

### get_npu_capabilities

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

### get_available_providers

```python
def get_available_providers() -> List[str]:
    """
    Get list of available ONNX Runtime execution providers.

    Returns:
        List of provider names in priority order

    Example:
        >>> providers = get_available_providers()
        >>> print(providers)
        ['CUDAExecutionProvider', 'CPUExecutionProvider']
    """
```

---

### validate_npu_config

```python
def validate_npu_config(
    config: NPUConfig,
    capabilities: NPUCapabilities
) -> Tuple[bool, List[str]]:
    """
    Validate NPU configuration against hardware capabilities.

    Args:
        config: NPUConfig to validate
        capabilities: NPU hardware capabilities

    Returns:
        Tuple of (is_valid, warnings_list)

    Example:
        >>> config = NPUConfig(memory_limit_mb=8192)
        >>> caps = get_npu_capabilities()
        >>> is_valid, warnings = validate_npu_config(config, caps)
        >>> if not is_valid:
        ...     for w in warnings:
        ...         print(f"Warning: {w}")
    """
```

---

### get_recommended_npu_config

```python
def get_recommended_npu_config(
    model_size_mb: float
) -> NPUConfig:
    """
    Get recommended NPU config based on model size.

    Args:
        model_size_mb: Model size in megabytes

    Returns:
        NPUConfig with optimal settings for model size

    Logic:
        - If model fits in Tile SRAM: weight_preload=True
        - If model fits in NPU memory: weight_preload=True
        - If model exceeds NPU memory: weight_preload=False
    """
```

---

## Data Classes

### SplitResult

```python
@dataclass
class SplitResult:
    """
    Result of semantic model splitting.

    Attributes:
        output_dir: Directory containing split component files
        num_layers: Number of transformer layers extracted
        component_files: Dict mapping component name to file path
        model_info: Metadata dictionary with model information
    """
```

### ArchitectureInfo

```python
@dataclass
class ArchitectureInfo:
    """
    Information about detected model architecture.

    Attributes:
        arch_type: Architecture type (llama, qwen, gpt2, etc.)
        num_layers: Number of transformer layers
        embed_name: Node name pattern for embeddings
        layer_prefix: Node name pattern for layers
        norm_name: Node name pattern for normalization
        lm_head_name: Node name pattern for LM head
    """
```

### NPUCapabilities

```python
@dataclass
class NPUCapabilities:
    """
    NPU hardware capabilities.

    Attributes:
        available: Whether NPU is available
        driver_version: NPU driver version string
        num_cores: Number of NPU cores
        memory_size_mb: Total NPU memory in MB
        tile_sram_mb: On-chip Tile SRAM in MB
    """
```

---

## Component Providers Routing

### Wildcard Pattern Matching

Component providers support wildcard pattern matching:

```python
component_providers = {
    "embedding": ["DmlExecutionProvider"],      # Exact match
    "layer_*":   ["VitisAIExecutionProvider"],  # Matches layer_0, layer_1, etc.
    "*head":     ["DmlExecutionProvider"],      # Matches lm_head, mlp_head
    "norm":      ["CPUExecutionProvider"],      # Exact match
}
```

**Pattern Resolution Order:**
1. Exact component name match
2. Wildcard pattern match (first match wins)
3. Default providers as fallback

---

## Error Classes

### SplittingError

```python
class SplittingError(Exception):
    """Raised when model splitting fails."""
```

### InferenceError

```python
class InferenceError(Exception):
    """Raised when inference fails."""
```

### ConfigurationError

```python
class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
```

---

*For usage examples, see [Getting Started](../01-getting-started/README.md) and [Hardware Deployment](../05-hardware-deployment/README.md).*
