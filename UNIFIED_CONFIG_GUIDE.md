# Unified Configuration System for Dual Chunking Approach

This guide explains how to use the unified configuration system that supports both PyTorch-level and ONNX graph-level chunking methods with full backward compatibility.

## Overview

The unified configuration system provides:
- **Dual Method Support**: Both PyTorch-level and ONNX graph-level chunking
- **Backward Compatibility**: All existing legacy configurations continue to work
- **Comprehensive Validation**: Robust error checking and helpful error messages
- **Flexible Configuration**: Method-specific and common configuration options
- **Configuration Persistence**: Save and load configuration files

## Configuration Methods

### 1. Legacy Format (Backward Compatible)

All existing configurations continue to work without modification:

```python
# Simple legacy configuration
extra_options = {
    "enable_chunking": True,
    "chunk_strategy": "layer_based",
    "num_chunks": 3
}

# Advanced legacy configuration
extra_options = {
    "enable_chunking": True,
    "chunk_strategy": "weight_based",
    "chunk_distribution": "equal_memory",
    "num_chunks": 4,
    "respect_layer_boundaries": True,
    "chunk_output_dir": "./chunked_models",
    "verify_chunks": True,
    "min_psnr_threshold": 40.0
}
```

**Legacy Options:**
- `enable_chunking`: Boolean to enable/disable chunking
- `chunk_strategy`: "layer_based" or "weight_based"
- `chunk_distribution`: "equal_layers", "equal_memory", or "custom"
- `num_chunks`: Number of chunks to create (>= 2)
- `respect_layer_boundaries`: Force splits at layer boundaries
- `chunk_output_dir`: Output directory for chunked models
- `verify_chunks`: Enable chunk verification
- `min_psnr_threshold`: PSNR threshold for verification
- `custom_chunk_sizes`: Custom chunk distribution
- `chunk_split_ratio`: Split ratio for 2-chunk scenarios

### 2. Unified Format (Recommended)

The new unified format provides more explicit configuration and better organization:

```python
# ONNX Graph-level chunking
extra_options = {
    "chunking": {
        "method": "onnx_graph_level",
        "onnx_graph_level": {
            "strategy": "weight_based",
            "distribution": "equal_memory",
            "num_chunks": 3,
            "respect_layer_boundaries": True
        },
        "common": {
            "output_directory": "./chunked_models",
            "verify_chunks": True,
            "min_psnr_threshold": 35.0,
            "save_metadata": True
        }
    }
}

# PyTorch-level chunking (Phase 2 - Infrastructure Ready)
extra_options = {
    "chunking": {
        "method": "pytorch_level",
        "pytorch_level": {
            "chunk_strategy": "by_layers",
            "layer_distribution": [8, 8, 8],
            "include_embedding_in_first": True,
            "include_lm_head_in_last": True,
            "balance_memory_usage": True
        },
        "common": {
            "output_directory": "./torch_chunks",
            "verify_chunks": True
        }
    }
}
```

## Configuration Sections

### Method Selection
- `"method": "onnx_graph_level"` - Use ONNX graph-level chunking (currently implemented)
- `"method": "pytorch_level"` - Use PyTorch-level chunking (Phase 2 implementation)

### ONNX Graph-level Configuration

```python
"onnx_graph_level": {
    "strategy": "weight_based",           # "weight_based" or "layer_based"
    "distribution": "equal_memory",       # "equal_memory", "equal_layers", "custom"
    "num_chunks": 3,                      # Number of chunks (>= 2)
    "respect_layer_boundaries": True,     # Force splits at layer boundaries
    "custom_chunk_sizes": [0.3, 0.4, 0.3], # Custom distribution (if distribution="custom")
    "chunk_split_ratio": 0.5              # Split ratio for 2-chunk scenarios
}
```

**Strategy Options:**
- `weight_based`: Split based on model weight distribution
- `layer_based`: Split at layer boundaries

**Distribution Options:**
- `equal_memory`: Balance memory usage across chunks
- `equal_layers`: Distribute layers equally across chunks
- `custom`: Use custom chunk sizes

### PyTorch-level Configuration

```python
"pytorch_level": {
    "chunk_strategy": "by_layers",           # "by_layers" or "by_weights"
    "layer_distribution": [8, 8, 8],         # Number of layers per chunk
    "include_embedding_in_first": True,      # Include embedding in first chunk
    "include_lm_head_in_last": True,         # Include LM head in last chunk
    "preserve_intermediate_outputs": True,   # Preserve intermediate layer outputs
    "chunk_overlap_ratio": 0.0,              # Overlap ratio between chunks (0.0-1.0)
    "balance_memory_usage": True             # Balance memory across chunks
}
```

**Strategy Options:**
- `by_layers`: Distribute layers across chunks
- `by_weights`: Distribute based on weight sizes

### Common Configuration

```python
"common": {
    "output_directory": "./chunked_models",  # Output directory
    "save_original_model": True,             # Save original model
    "verify_chunks": False,                  # Enable chunk verification
    "min_psnr_threshold": 35.0,             # PSNR threshold for verification
    "save_metadata": True,                   # Save chunking metadata
    "verbose": True                          # Verbose output
}
```

## Usage Examples

### Example 1: Basic ONNX Graph-level Chunking

```python
from builder import Model

# Using legacy format
extra_options = {
    "enable_chunking": True,
    "chunk_strategy": "layer_based",
    "num_chunks": 3
}

# Using unified format (equivalent)
extra_options = {
    "chunking": {
        "method": "onnx_graph_level",
        "onnx_graph_level": {
            "strategy": "layer_based",
            "distribution": "equal_layers",
            "num_chunks": 3
        }
    }
}

model = Model(config, "float16", "float16", "cpu", "./cache", extra_options)
```

### Example 2: Advanced ONNX Graph-level Chunking

```python
extra_options = {
    "chunking": {
        "method": "onnx_graph_level",
        "onnx_graph_level": {
            "strategy": "weight_based",
            "distribution": "custom",
            "num_chunks": 4,
            "respect_layer_boundaries": True,
            "custom_chunk_sizes": [0.2, 0.3, 0.3, 0.2]
        },
        "common": {
            "output_directory": "./custom_chunks",
            "verify_chunks": True,
            "min_psnr_threshold": 40.0,
            "save_metadata": True
        }
    }
}
```

### Example 3: PyTorch-level Chunking (Future)

```python
extra_options = {
    "chunking": {
        "method": "pytorch_level",
        "pytorch_level": {
            "chunk_strategy": "by_layers",
            "layer_distribution": [6, 6, 6, 6],
            "include_embedding_in_first": True,
            "include_lm_head_in_last": True,
            "balance_memory_usage": True
        },
        "common": {
            "output_directory": "./torch_chunks",
            "verify_chunks": True
        }
    }
}
```

## Configuration Validation

The system provides comprehensive validation with helpful error messages:

### Common Validation Errors

- **Invalid chunking method**: "Unknown chunking method: xxx. Supported: pytorch_level, onnx_graph_level"
- **Invalid num_chunks**: "num_chunks must be an integer >= 2, got 1"
- **Missing layer distribution**: "layer_distribution is required when using 'by_layers' strategy"
- **Invalid PSNR threshold**: "min_psnr_threshold must be a positive number"
- **Invalid custom sizes**: "custom_chunk_sizes must be a list of length {num_chunks}"

### Method-Specific Validation

#### ONNX Graph-level
- Strategy must be "weight_based" or "layer_based"
- Distribution must be valid type
- Custom chunk sizes must match num_chunks if provided

#### PyTorch-level
- Layer distribution required for "by_layers" strategy
- Chunk overlap ratio must be between 0.0 and 1.0
- Layer distribution must sum to positive number

## Integration with Existing Code

The unified configuration system integrates seamlessly with existing code:

### Automatic Detection
- Automatically detects legacy vs unified format
- Applies appropriate processing and validation
- Maintains full backward compatibility

### Configuration Information
```python
# Get current chunking configuration
chunking_info = model.get_chunking_info()
print(chunking_info)
# Output: {'enabled': True, 'method': 'onnx_graph_level', 'legacy_mode': False, ...}

# Save configuration to file
model.save_chunking_config("./output_dir")
# Creates: chunking_configuration.json
```

### Error Handling
```python
try:
    model = Model(config, dtype, dtype, ep, cache, extra_options)
    # Configuration processed successfully
except ValueError as e:
    print(f"Configuration error: {e}")
    # Handle configuration errors
```

## Migration Guide

### From Legacy to Unified Format

**Legacy:**
```python
extra_options = {
    "enable_chunking": True,
    "chunk_strategy": "layer_based",
    "chunk_distribution": "equal_layers",
    "num_chunks": 3,
    "verify_chunks": True
}
```

**Unified (Equivalent):**
```python
extra_options = {
    "chunking": {
        "method": "onnx_graph_level",
        "onnx_graph_level": {
            "strategy": "layer_based",
            "distribution": "equal_layers",
            "num_chunks": 3
        },
        "common": {
            "verify_chunks": True
        }
    }
}
```

**Migration Steps:**
1. Wrap legacy options in `"chunking": {...}` structure
2. Add `"method": "onnx_graph_level"`
3. Move ONNX-specific options to `"onnx_graph_level"`
4. Move common options to `"common"`
5. Test with existing models to ensure compatibility

## Best Practices

### 1. Use Unified Format
- Use the unified format for new projects
- Provides better organization and clarity
- Enables future PyTorch-level chunking

### 2. Specify All Options Explicitly
- Avoid relying on defaults when possible
- Makes configuration more maintainable
- Reduces ambiguity

### 3. Validate Configuration Early
- Test configuration with small models first
- Check validation messages for guidance
- Use configuration saving for debugging

### 4. Use Method-Specific Options
- Take advantage of method-specific configurations
- ONNX graph-level: focus on graph structure and memory
- PyTorch-level: focus on layer distribution and model components

### 5. Document Your Configuration
- Include comments explaining configuration choices
- Save configuration files with models
- Maintain version control for configuration changes

## Troubleshooting

### Common Issues

1. **Configuration Not Working**
   - Check for syntax errors in JSON/dict structure
   - Verify all required options are provided
   - Use validation error messages for guidance

2. **Legacy Configuration Broken**
   - Legacy configurations should continue to work
   - Check for typos in option names
   - Verify option values are within valid ranges

3. **PyTorch-level Not Working**
   - PyTorch-level chunking is infrastructure-only in Phase 1
   - Will be fully implemented in Phase 2
   - Use ONNX graph-level for current chunking needs

4. **Performance Issues**
   - Chunking method doesn't affect model performance
   - Check model size and memory requirements
   - Use appropriate distribution strategy

### Getting Help

1. **Check Validation Messages**: The system provides detailed error messages
2. **Use Test Suite**: Run `python test_unified_config.py` for verification
3. **Review Examples**: See configuration examples in this guide
4. **Examine Output**: Check verbose output during model building

## File Structure

```
onnx-model-chunking/
├── chunking_config.py          # Core configuration system
├── builder.py                  # Model builder with integrated config
├── test_unified_config.py      # Test suite for configuration
├── UNIFIED_CONFIG_GUIDE.md     # This documentation
└── example_configs/            # Example configuration files
    ├── basic_onnx.json
    ├── advanced_onnx.json
    ├── pytorch_level.json
    └── legacy_example.json
```

This unified configuration system provides a solid foundation for the dual chunking approach while maintaining full backward compatibility and preparing for future enhancements.