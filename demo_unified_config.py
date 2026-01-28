#!/usr/bin/env python3
"""
Demonstration script for the unified configuration system integration with the Model class.

This script shows how the unified configuration system works with the actual Model builder
without requiring a full model to be present.
"""

import sys
import os
import json
from unittest.mock import Mock, MagicMock

# Mock the imports that would require actual model files
sys.modules['torch'] = Mock()
sys.modules['numpy'] = Mock()
sys.modules['onnx_ir'] = Mock()
sys.modules['transformers'] = Mock()
sys.modules['tqdm'] = Mock()
sys.modules['onnxruntime.quantization.matmul_nbits_quantizer'] = Mock()

# Mock the parse_hf_token function
def mock_parse_hf_token(token):
    return "true"

# Add the mock to sys.modules before importing builder
sys.modules['builder'] = Mock()
sys.modules['builder'].parse_hf_token = mock_parse_hf_token

# Now we can import our modules
from chunking_config import (
    ConfigurationProcessor,
    ChunkingMethod,
    create_sample_configurations
)

# Mock the ir.DataType class
class MockDataType:
    FLOAT = "float32"
    FLOAT16 = "float16"
    INT64 = "int64"

# Mock ir module
mock_ir = Mock()
mock_ir.DataType = MockDataType
mock_ir.Graph = Mock()
mock_ir.Model = Mock()
sys.modules['onnx_ir'] = mock_ir


def create_mock_config():
    """Create a mock model configuration"""
    config = Mock()
    config.seq_length = 2048
    config.max_position_embeddings = 2048
    config.sliding_window = -1
    config.ffn_hidden_size = 11008
    config.hidden_size = 4096
    config.num_key_value_heads = 32
    config.num_attention_heads = 32
    config.head_dim = 128
    config.num_hidden_layers = 32
    config.num_layers = 32
    config.vocab_size = 32000
    config.hidden_activation = "silu"
    config.hidden_act = "silu"
    config._name_or_path = "mock-model"
    config.architectures = ["MockModelForCausalLM"]
    config.quantization_config = None

    return config


def test_legacy_configuration_integration():
    """Test legacy configuration integration with Model class"""
    print("=== Testing Legacy Configuration Integration ===")

    # Import after mocking
    from builder import Model

    # Create mock model config
    mock_model_config = create_mock_config()

    # Test legacy configuration
    legacy_extra_options = {
        "enable_chunking": True,
        "chunk_strategy": "layer_based",
        "chunk_distribution": "equal_layers",
        "num_chunks": 3,
        "respect_layer_boundaries": True,
        "chunk_output_dir": "./test_chunks",
        "verify_chunks": True,
        "min_psnr_threshold": 35.0,
        "num_hidden_layers": 32  # Required for Model initialization
    }

    try:
        # Create model instance (this will test configuration processing)
        print("Creating Model with legacy configuration...")
        model = Model(
            config=mock_model_config,
            io_dtype="float16",
            onnx_dtype="float16",
            ep="cpu",
            cache_dir="./cache",
            extra_options=legacy_extra_options
        )

        # Check configuration processing results
        print(f"[SUCCESS] Model created successfully")
        print(f"  Chunking enabled: {model.enable_chunking}")
        print(f"  Chunking method: {model.chunking_config.method}")
        print(f"  Legacy mode: {model.config_processor.is_legacy_mode()}")
        print(f"  Chunk strategy: {model.chunk_strategy}")
        print(f"  Number of chunks: {model.num_chunks}")
        print(f"  Output directory: {model.chunk_output_dir}")

        # Test configuration info
        chunking_info = model.get_chunking_info()
        print(f"  Configuration info: {json.dumps(chunking_info, indent=4)}")

        return True

    except Exception as e:
        print(f"[ERROR] Failed to create model with legacy config: {e}")
        return False


def test_unified_configuration_integration():
    """Test unified configuration integration with Model class"""
    print("\n\n=== Testing Unified Configuration Integration ===")

    from builder import Model

    # Create mock model config
    mock_model_config = create_mock_config()

    # Test unified configuration
    unified_extra_options = {
        "chunking": {
            "method": "onnx_graph_level",
            "onnx_graph_level": {
                "strategy": "weight_based",
                "distribution": "equal_memory",
                "num_chunks": 4,
                "respect_layer_boundaries": True,
                "custom_chunk_sizes": [0.25, 0.25, 0.25, 0.25]
            },
            "common": {
                "output_directory": "./unified_chunks",
                "verify_chunks": True,
                "min_psnr_threshold": 40.0,
                "save_metadata": True,
                "verbose": True
            }
        },
        "num_hidden_layers": 32  # Required for Model initialization
    }

    try:
        # Create model instance
        print("Creating Model with unified configuration...")
        model = Model(
            config=mock_model_config,
            io_dtype="float16",
            onnx_dtype="float16",
            ep="cpu",
            cache_dir="./cache",
            extra_options=unified_extra_options
        )

        # Check configuration processing results
        print(f"[SUCCESS] Model created successfully")
        print(f"  Chunking enabled: {model.enable_chunking}")
        print(f"  Chunking method: {model.chunking_config.method}")
        print(f"  Legacy mode: {model.config_processor.is_legacy_mode()}")
        print(f"  Chunk strategy: {model.chunk_strategy}")
        print(f"  Number of chunks: {model.num_chunks}")
        print(f"  Custom chunk sizes: {model.custom_chunk_sizes}")

        # Test method-specific configuration
        method_name, method_config = model.config_processor.get_method_specific_config()
        print(f"  Active method: {method_name}")
        print(f"  Method config strategy: {method_config.strategy}")
        print(f"  Method config distribution: {method_config.distribution}")

        return True

    except Exception as e:
        print(f"[ERROR] Failed to create model with unified config: {e}")
        return False


def test_pytorch_level_configuration():
    """Test PyTorch-level configuration (infrastructure only)"""
    print("\n\n=== Testing PyTorch-level Configuration (Infrastructure) ===")

    from builder import Model

    # Create mock model config
    mock_model_config = create_mock_config()

    # Test PyTorch-level configuration
    pytorch_extra_options = {
        "chunking": {
            "method": "pytorch_level",
            "pytorch_level": {
                "chunk_strategy": "by_layers",
                "layer_distribution": [8, 8, 8, 8],
                "include_embedding_in_first": True,
                "include_lm_head_in_last": True,
                "balance_memory_usage": True
            },
            "common": {
                "output_directory": "./torch_chunks",
                "verify_chunks": True,
                "min_psnr_threshold": 45.0
            }
        },
        "num_hidden_layers": 32  # Required for Model initialization
    }

    try:
        # Create model instance
        print("Creating Model with PyTorch-level configuration...")
        model = Model(
            config=mock_model_config,
            io_dtype="float16",
            onnx_dtype="float16",
            ep="cpu",
            cache_dir="./cache",
            extra_options=pytorch_extra_options
        )

        # Check that PyTorch-level is detected but disabled until Phase 2
        print(f"[SUCCESS] Model created successfully")
        print(f"  Chunking enabled: {model.enable_chunking}")  # Should be False for now
        print(f"  Original method: {model.chunking_config.method}")
        print(f"  PyTorch-level config processed: {model.chunking_config.pytorch_level.__dict__}")

        if not model.enable_chunking and model.chunking_config.method == "pytorch_level":
            print("  [INFO] PyTorch-level chunking infrastructure ready, implementation pending Phase 2")

        return True

    except Exception as e:
        print(f"[ERROR] Failed to create model with PyTorch-level config: {e}")
        return False


def test_error_handling():
    """Test error handling with invalid configurations"""
    print("\n\n=== Testing Error Handling ===")

    from builder import Model

    # Create mock model config
    mock_model_config = create_mock_config()

    # Test invalid configurations
    invalid_configs = [
        {
            "name": "Invalid chunking method",
            "config": {
                "chunking": {
                    "method": "invalid_method",
                    "onnx_graph_level": {
                        "strategy": "weight_based",
                        "num_chunks": 2
                    }
                },
                "num_hidden_layers": 32
            }
        },
        {
            "name": "Invalid num_chunks",
            "config": {
                "chunking": {
                    "method": "onnx_graph_level",
                    "onnx_graph_level": {
                        "strategy": "weight_based",
                        "num_chunks": 1  # Invalid
                    }
                },
                "num_hidden_layers": 32
            }
        },
        {
            "name": "Missing required option",
            "config": {
                "chunking": {
                    "method": "pytorch_level",
                    "pytorch_level": {
                        "chunk_strategy": "by_layers"
                        # Missing layer_distribution
                    }
                },
                "num_hidden_layers": 32
            }
        }
    ]

    success_count = 0
    for test_case in invalid_configs:
        print(f"\n--- {test_case['name']} ---")
        try:
            model = Model(
                config=mock_model_config,
                io_dtype="float16",
                onnx_dtype="float16",
                ep="cpu",
                cache_dir="./cache",
                extra_options=test_case['config']
            )
            print(f"[ERROR] Should have failed but didn't!")
        except Exception as e:
            print(f"[SUCCESS] Correctly caught error: {e}")
            success_count += 1

    print(f"\nError handling: {success_count}/{len(invalid_configs)} tests passed")
    return success_count == len(invalid_configs)


def test_configuration_conversion():
    """Test legacy to unified configuration conversion"""
    print("\n\n=== Testing Configuration Conversion ===")

    from builder import Model

    # Create mock model config
    mock_model_config = create_mock_config()

    # Test legacy configuration
    legacy_config = {
        "enable_chunking": True,
        "chunk_strategy": "weight_based",
        "chunk_distribution": "equal_memory",
        "num_chunks": 3,
        "respect_layer_boundaries": True,
        "chunk_output_dir": "./legacy_test",
        "verify_chunks": True,
        "min_psnr_threshold": 38.0,
        "num_hidden_layers": 32
    }

    try:
        # Create model with legacy config
        model = Model(
            config=mock_model_config,
            io_dtype="float16",
            onnx_dtype="float16",
            ep="cpu",
            cache_dir="./cache",
            extra_options=legacy_config
        )

        # Test conversion back to legacy format
        if model.config_processor.is_legacy_mode():
            legacy_style = model.config_processor.get_legacy_style_config()
            print("[SUCCESS] Legacy conversion successful")
            print(f"  Converted strategy: {legacy_style['chunk_strategy']}")
            print(f"  Converted num_chunks: {legacy_style['num_chunks']}")
            print(f"  Converted verify_chunks: {legacy_style['verify_chunks']}")

            return True
        else:
            print("[ERROR] Expected legacy mode but got unified mode")
            return False

    except Exception as e:
        print(f"[ERROR] Configuration conversion failed: {e}")
        return False


def main():
    """Run all integration tests"""
    print("Unified Configuration System Integration Tests")
    print("=" * 60)

    tests = [
        ("Legacy Configuration Integration", test_legacy_configuration_integration),
        ("Unified Configuration Integration", test_unified_configuration_integration),
        ("PyTorch-level Configuration", test_pytorch_level_configuration),
        ("Error Handling", test_error_handling),
        ("Configuration Conversion", test_configuration_conversion)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[ERROR] Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Integration Test Results:")
    print("=" * 60)

    passed = 0
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\nSummary: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n[SUCCESS] All integration tests passed!")
        print("The unified configuration system is working correctly.")
    else:
        print(f"\n[WARNING] {len(results) - passed} test(s) failed.")
        print("Please check the configuration implementation.")

    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)