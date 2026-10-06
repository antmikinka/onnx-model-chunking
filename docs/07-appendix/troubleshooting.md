# Troubleshooting Guide

**Outcome Statement:** Resolve common issues with model splitting, inference, and hardware deployment.

---

## Installation Issues

### "Provider not found" Error

**Symptoms:**
```
ValueError: Provider CUDAExecutionProvider not found
```

**Causes:**
1. ONNX Runtime GPU package not installed
2. CUDA drivers not installed
3. Hardware not present

**Solutions:**

```bash
# For NVIDIA GPU
pip install onnxruntime-gpu

# For AMD GPU (DirectML)
pip install onnxruntime-directml

# For AMD NPU
pip install onnxruntime-vitisai

# Verify installation
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

---

### "VitisAIExecutionProvider not available" (NPU)

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

# Verify driver installation (Linux)
xclinfo  # Should show NPU devices

# Verify driver installation (Windows)
# Check Device Manager for "AMD XDNA NPU"

# Reinstall drivers if needed
# Download from: https://www.amd.com/en/support
```

---

## Model Splitting Issues

### "No suitable split point found"

**Symptoms:**
```
ValueError: No suitable split point found in graph
```

**Causes:**
1. Model has non-linear topology (MoE, multi-branch)
2. Architecture not recognized
3. Split ratio places cut in invalid location

**Solutions:**

```python
# Try different split ratio
extra_options = {
    "enable_chunking": True,
    "chunk_split_ratio": 0.4,  # Try 0.3, 0.4, 0.5, 0.6
    "verify_chunks": True
}

# Or use layer-based splitting
python split_local_model.py \
    "path/to/model" \
    --strategy layer_based \
    --layers-per-chunk 7
```

**For unsupported architectures:**
```python
# Manual architecture registration
from chunking_utils.common import register_architecture

register_architecture(
    name="custom_arch",
    layer_pattern="model/custom_layers.\\d+",
    embed_name="model/custom_embed",
    norm_name="model/custom_norm",
    lm_head_name="custom_head"
)
```

---

### "PSNR verification failed"

**Symptoms:**
```
ValueError: PSNR verification failed: 28.5 dB < threshold 35.0 dB
```

**Causes:**
1. Numerical precision issues (FP16 vs FP32)
2. Split point cuts through critical operation
3. Model corruption

**Solutions:**

```python
# Lower PSNR threshold for testing
extra_options = {
    "enable_chunking": True,
    "min_psnr_threshold": 30.0,  # Temporarily lower for testing
    "verify_chunks": True
}

# Use FP32 for verification
model = Model(config, "float32", "float32", "cpu", "./cache", extra_options)

# Check model integrity
import onnx
model = onnx.load("path/to/model.onnx")
onnx.checker.check_model(model)  # Should pass without errors
```

**If PSNR still low:**
- Review split point location
- Check for architecture-specific issues
- Consider using semantic splitting instead of bisection

---

### "Out of memory during split"

**Symptoms:**
```
MemoryError: Unable to allocate 2.5 GB
```

**Causes:**
1. Both original model and split components in memory simultaneously
2. Insufficient system RAM
3. Other applications consuming memory

**Solutions:**

```bash
# Use split-only mode (no inference testing)
python grok_onnx_split.py \
    --model "path/to/model.onnx" \
    --output "split_output/my_model" \
    --split-only  # Skip inference, reduces memory usage

# Close other applications
# Restart Python interpreter

# Use disk-based processing (if available)
# Some splitters support streaming to disk
```

---

## Inference Issues

### "Slow inference on NPU" (CRITICAL)

**Symptoms:**
- First token takes 5-30 seconds
- Per-token latency: 500ms - 5 seconds
- Power consumption higher than expected

**Cause: Using split model on NPU (INCORRECT)**

**Solution: Switch to monolithic model immediately**

```bash
# INCORRECT - DO NOT USE
# python grok_inference_split_model_pipeline.py \
#     --model-dir "split_output/my_model" \
#     --providers "VitisAIExecutionProvider"

# CORRECT - Use monolithic model
python grok_inference_split_model_pipeline.py \
    --model "model.onnx" \
    --providers "VitisAIExecutionProvider" \
    --persist
```

**See:** [NPU Monolithic Requirement](./npu-monolithic-requirement.md)

---

### "Slow first inference" (NPU/GPU)

**Symptoms:**
- First token takes 5-30 seconds
- Subsequent tokens are fast

**Cause:** Graph compilation happens on first run

**Solutions:**

```python
# Enable graph capture for reuse
npu_config = NPUConfig(
    graph_capture=True,  # Cache compiled graph
    compilation_mode="default"
)

# Warm up before actual use
_ = pipeline.generate("warmup", max_new_tokens=1)

# Use optimized compilation for production
npu_config = NPUConfig(
    compilation_mode="optimized"  # Slower compile, faster runtime
)
```

---

### "Out of memory during inference"

**Symptoms:**
```
RuntimeError: CUDA out of memory
# or
MemoryError: Unable to allocate tensor
```

**Causes:**
1. Model exceeds available VRAM/RAM
2. Batch size too large
3. Sequence length too long
4. Other processes consuming memory

**Solutions:**

```bash
# Reduce sequence length
python grok_inference_split_model_pipeline.py \
    --model-dir "split_output/my_model" \
    --prompt "Your prompt" \
    --max-sequence-length 512  # Reduce from default

# Use split model if using monolithic
python grok_inference_split_model_pipeline.py \
    --model-dir "split_output/my_model" \
    --providers "CUDAExecutionProvider" \
    --persist

# Or reduce batch size
# (Most inference is batch_size=1 anyway)
```

---

### "Graph compilation failed" (NPU)

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
    npu_compilation_mode="fast",  # Faster compilation
    npu_weight_preload=False  # Don't preload if memory tight
)

# Or fall back to iGPU for problematic components
component_providers={
    "layer_*": ["DmlExecutionProvider"]  # Use iGPU instead
}

# Check operator support
from pipeline_utils.npu_config import check_npu_operator_support
supported, unsupported = check_npu_operator_support("model.onnx")
print(f"Unsupported operators: {unsupported}")
```

---

### "KV cache error"

**Symptoms:**
```
RuntimeError: KV cache shape mismatch
# or
IndexError:KV cache index out of bounds
```

**Causes:**
1. Incorrect KV cache initialization
2. Sequence length exceeds cache capacity
3. Layer count mismatch

**Solutions:**

```python
# Ensure correct KV cache configuration
from pipeline_utils import KVCacheManager

kv_cache = KVCacheManager(
    num_layers=28,  # Must match model
    num_kv_heads=8,  # Must match model config
    head_dim=128,  # Must match model config
    dtype="float32",
    batch_size=1
)

# Check model config
from transformers import AutoConfig
config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
print(f"Num layers: {config.num_hidden_layers}")
print(f"Num KV heads: {config.num_key_value_heads}")
print(f"Head dim: {config.hidden_size // config.num_attention_heads}")
```

---

## Provider Routing Issues

### "Provider fallback to CPU"

**Symptoms:**
- Inference runs on CPU despite GPU/NPU being available
- Warning messages about provider fallback

**Causes:**
1. Provider not listed in session providers
2. Operator not supported on accelerator
3. Memory limit exceeded on accelerator

**Solutions:**

```python
# Ensure correct provider priority
pipeline = SequentialModelPipeline(
    model_path="split_output/my_model",
    providers=[
        "CUDAExecutionProvider",  # Fastest first
        "CPUExecutionProvider"    # Fallback
    ]
)

# Check operator placement
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
print(f"Provider for each node:")
for node in session._sess._providers_per_node:
    print(f"  {node}")
```

---

### "Hybrid dispatch not working"

**Symptoms:**
- All components run on same provider despite hybrid config
- Component-specific providers not applied

**Solutions:**

```python
# Ensure correct component provider configuration
pipeline = SequentialModelPipeline(
    model_path="split_output/my_model",
    component_providers={
        "embedding": ["DmlExecutionProvider"],
        "layer_*":   ["VitisAIExecutionProvider"],
        "norm":      ["CPUExecutionProvider"],
        "lm_head":   ["DmlExecutionProvider"],
    },
    persist_sessions=True
)

# Verify provider routing
print(f"Component providers: {pipeline.component_providers}")
```

---

## Performance Issues

### "Higher than expected latency"

**Troubleshooting Steps:**

1. **Check execution mode:**
   ```bash
   # Sequential mode is slower but uses less memory
   # Persistent mode is faster but uses more memory
   ```

2. **Check provider:**
   ```python
   print(f"Active providers: {session.get_providers()}")
   ```

3. **Check memory usage:**
   ```python
   import psutil
   process = psutil.Process()
   print(f"Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
   ```

4. **Enable profiling:**
   ```python
   npu_config = NPUConfig(enable_profiling=True)
   # Check vitisai_profile.json for detailed timing
   ```

---

## Quick Diagnostic Commands

```bash
# Check available providers
python -c "import onnxruntime as ort; print(ort.get_available_providers())"

# Check NPU availability
python -c "from pipeline_utils import is_npu_available; print(is_npu_available())"

# Check model structure
python -c "import onnx; m=onnx.load('model.onnx'); print(f'Nodes: {len(m.graph.node)}')"

# Check model size
python -c "import os; print(os.path.getsize('model.onnx')/1024/1024, 'MB')"

# Verify NPU driver (Linux)
xclinfo | grep -A 5 'XDNA'

# Verify NPU driver (Windows)
# Device Manager → Look for 'AMD XDNA NPU'
```

---

## Getting Help

If issues persist:

1. **Check logs** - Enable verbose logging for detailed output
2. **Review documentation** - See relevant sections in this documentation
3. **Check GitHub issues** - Known issues may already be documented
4. **Provide diagnostic info** when asking for help:
   - Hardware (CPU/GPU/NPU model)
   - Python version
   - ONNX Runtime version
   - Model being used
   - Full error message
   - Configuration used

---

*For more technical details, see [API Reference](./api-reference.md) and [NPU Monolithic Requirement](./npu-monolithic-requirement.md).*
