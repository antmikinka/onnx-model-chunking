# ONNX Model Chunking Documentation

## Overview

The ONNX model chunking functionality allows you to split large ONNX models into smaller, more manageable chunks. This is particularly useful for:

- **Memory optimization**: Running models with limited memory resources
- **Model distribution**: Distributing model parts across different devices or servers
- **Faster loading**: Loading only necessary parts of the model for specific tasks
- **Pipeline execution**: Creating inference pipelines where different chunks run on different hardware

## Architecture

The chunking system is built into the `Model` class in `builder.py` and follows the same principles as the Core ML chunking implementation from `chunk_mlprogram.py`. It maintains:

- **Graph integrity**: Ensures all connections between chunks are properly preserved
- **Accuracy**: Uses PSNR (Peak Signal-to-Noise Ratio) verification to maintain model accuracy
- **Flexibility**: Configurable split ratios and verification thresholds

## Core Functions

### 1. `_get_op_idx_split_location()`
Finds the optimal split point in the ONNX graph based on weight distribution.
- Analyzes initializer (weight) sizes across the graph
- Finds a node that approximately bisects the model by weight size
- Ensures the split point is at a clean boundary (non-constant node with single output)

### 2. `_get_first_chunk_outputs(split_node_idx)`
Identifies tensors that cross the boundary between the first and second chunks.
- Analyzes graph connectivity
- Returns a list of tensor names that need to be outputs of the first chunk and inputs to the second chunk

### 3. `_make_first_chunk_prog(split_node_idx)`
Creates the first chunk model containing all nodes up to the split point.
- Creates a new ONNX graph with appropriate inputs and outputs
- Copies relevant initializers and nodes
- Sets boundary tensors as model outputs

### 4. `_make_second_chunk_prog(split_node_idx, boundary_tensors)`
Creates the second chunk model containing all nodes after the split point.
- Creates a new ONNX graph with boundary tensors as inputs
- Preserves original model outputs
- Copies relevant initializers and nodes

### 5. `_verify_output_correctness_of_chunks()`
Verifies that chunked models produce the same outputs as the original model using PSNR.
- Runs inference on both the original and chunked models
- Computes PSNR values to measure accuracy
- Validates against configurable threshold

## Configuration Options

The chunking functionality can be configured through `extra_options` in the Model constructor:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_chunking` | bool | `False` | Enable/disable model chunking |
| `chunk_output_dir` | str | `None` | Directory to save chunked models |
| `verify_chunks` | bool | `False` | Enable PSNR verification of chunks |
| `chunk_split_ratio` | float | `0.5` | Fraction of total weight size for first chunk (0.0-1.0) |
| `min_psnr_threshold` | float | `35.0` | Minimum PSNR value for verification to pass |

## Usage Examples

### Basic Usage

```python
# Example configuration for model chunking
extra_options = {
    "enable_chunking": True,
    "chunk_output_dir": "./chunked_models",
    "verify_chunks": True,
    "chunk_split_ratio": 0.6,  # Split at 60% of weight size
    "min_psnr_threshold": 40.0  # Higher accuracy requirement
}

# Create model with chunking enabled
model = Model(config, "float32", "float32", "cpu", "./cache", extra_options)

# Build the model - chunking happens automatically
model.make_model("path/to/model")

# Save the main model
model.save_model("./output")
```

### Advanced Usage with Custom Verification

```python
import numpy as np

# Configure for memory-constrained deployment
extra_options = {
    "enable_chunking": True,
    "chunk_output_dir": "./optimized_chunks",
    "verify_chunks": True,
    "chunk_split_ratio": 0.3,  # First chunk is 30% (smaller for faster loading)
    "min_psnr_threshold": 35.0,  # Standard accuracy requirement
    "filename": "optimized_model.onnx"
}

model = Model(config, "float16", "float16", "cuda", "./cache", extra_options)

# The chunking process will:
# 1. Find optimal split point at 30% of weight size
# 2. Create first_chunk_model.onnx and second_chunk_model.onnx
# 3. Generate chunking_metadata.json with split information
# 4. Verify accuracy with PSNR if verification is enabled
# 5. Save metadata about boundary tensors and chunk sizes

model.make_model("path/to/transformer/model")
model.save_model("./models")
```

### Manual Chunking Control

```python
# If you want more control over the chunking process
model = Model(config, "float32", "float32", "cpu", "./cache", {})

# Build the model first
model.make_model("path/to/model")

# Then manually create chunks with custom settings
first_chunk, second_chunk, boundaries = model.create_chunked_models(
    output_dir="./manual_chunks",
    verify_chunks=True,
    sample_input=model._create_sample_input_for_verification()
)

print(f"Created {len(first_chunk.graph.nodes)} nodes in first chunk")
print(f"Created {len(second_chunk.graph.nodes)} nodes in second chunk")
print(f"Boundary tensors: {boundaries}")
```

## Output Files

When chunking is enabled with an output directory, the following files are generated:

1. **`model_first_chunk.onnx`** - First chunk of the model
2. **`model_second_chunk.onnx`** - Second chunk of the model
3. **`chunking_metadata.json`** - Metadata about the chunking process

### Metadata Structure

```json
{
  "split_node_index": 245,
  "cumulative_size_mb": 1280.5,
  "total_size_mb": 2560.0,
  "boundary_tensors": [
    "hidden_states_12",
    "attention_mask_12",
    "position_ids_12"
  ],
  "first_chunk_nodes": 246,
  "second_chunk_nodes": 380,
  "split_ratio": 0.5
}
```

## Pipeline Usage

The chunked models can be used in an inference pipeline:

```python
import onnxruntime as ort

# Load chunked models
first_session = ort.InferenceSession("model_first_chunk.onnx")
second_session = ort.InferenceSession("model_second_chunk.onnx")

# Prepare inputs
inputs = {
    "input_ids": np.array([[1, 2, 3, 4, 5]], dtype=np.int64),
    "attention_mask": np.array([[1, 1, 1, 1, 1]], dtype=np.int64),
    "position_ids": np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
}

# Run first chunk
first_outputs = first_session.run(None, inputs)

# Prepare inputs for second chunk (combine original inputs with first chunk outputs)
second_inputs = inputs.copy()
for boundary_tensor in boundary_tensors:
    second_inputs[boundary_tensor] = first_outputs[boundary_tensor]

# Run second chunk
final_outputs = second_session.run(None, second_inputs)
```

## Accuracy Verification

The PSNR-based verification ensures that chunked models maintain the same accuracy as the original model:

- **PSNR > 35 dB**: Considered good accuracy (default threshold)
- **PSNR > 40 dB**: High accuracy suitable for production use
- **PSNR < 35 dB**: Accuracy degradation detected - verification fails

## Limitations and Considerations

1. **Graph Structure**: Works best with models that have a clear linear structure. Highly branched graphs may not split optimally.

2. **Memory Usage**: During chunking, both the original model and chunked models exist in memory simultaneously.

3. **Verification Overhead**: PSNR verification requires running inference, which adds processing time.

4. **ONNX Compatibility**: Uses the `onnx_ir` library for graph manipulation, which must be compatible with your ONNX model version.

## Troubleshooting

### Common Issues

1. **"No suitable split point found"**
   - The model structure may not have clean boundaries
   - Try adjusting `chunk_split_ratio`
   - Check if the model has too many constant operations

2. **"PSNR verification failed"**
   - Try a higher `min_psnr_threshold` for testing
   - Verify that the model has proper numerical precision
   - Check if boundary tensors have correct shapes and types

3. **"Chunk creation failed"**
   - Ensure `onnx_ir` library is properly installed
   - Check model compatibility with the ONNX version
   - Verify that the model graph is not corrupted

### Debug Information

Enable verbose output to see detailed chunking information:

```python
import logging
logging.basicConfig(level=logging.INFO)

# This will show:
# - Split point analysis
# - Boundary tensor identification
# - Chunk creation progress
# - PSNR verification results
```

## Integration with Existing Workflow

The chunking functionality integrates seamlessly with the existing model creation workflow:

1. No changes needed to existing model creation code
2. Chunking is optional and controlled by configuration
3. Regular model creation continues even if chunking fails
4. All existing model types (Llama, Mistral, etc.) support chunking

This ensures backward compatibility while adding powerful new chunking capabilities.