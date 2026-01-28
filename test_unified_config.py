#!/usr/bin/env python3
"""
Test script for the unified chunking configuration system.

This script tests both legacy and unified configuration formats
to ensure backward compatibility and proper functionality.
"""

import sys
import os
import json
from chunking_config import (
    ConfigurationProcessor,
    UnifiedChunkingConfig,
    ChunkingMethod,
    create_sample_configurations,
    print_configuration_examples
)


def test_legacy_configuration():
    """Test legacy configuration format processing"""
    print("=== Testing Legacy Configuration Format ===")

    processor = ConfigurationProcessor()

    # Test legacy configuration examples
    legacy_configs = [
        {
            "name": "Basic legacy config",
            "config": {
                "enable_chunking": True,
                "chunk_strategy": "layer_based",
                "num_chunks": 3,
                "verify_chunks": True
            }
        },
        {
            "name": "Advanced legacy config",
            "config": {
                "enable_chunking": True,
                "chunk_strategy": "weight_based",
                "chunk_distribution": "equal_memory",
                "num_chunks": 4,
                "respect_layer_boundaries": True,
                "chunk_output_dir": "./test_chunks",
                "verify_chunks": True,
                "min_psnr_threshold": 40.0
            }
        },
        {
            "name": "Legacy with custom sizes",
            "config": {
                "enable_chunking": True,
                "chunk_strategy": "layer_based",
                "chunk_distribution": "custom",
                "num_chunks": 3,
                "custom_chunk_sizes": [0.3, 0.4, 0.3]
            }
        }
    ]

    for test_case in legacy_configs:
        print(f"\n--- {test_case['name']} ---")
        try:
            config = processor.process_configuration(test_case['config'])
            print(f"[SUCCESS] Successfully processed legacy configuration")
            print(f"  Method: {config.method}")
            print(f"  Strategy: {config.onnx_graph_level.strategy}")
            print(f"  Num chunks: {config.onnx_graph_level.num_chunks}")
            print(f"  Legacy mode: {processor.is_legacy_mode()}")

            # Test legacy conversion
            legacy_style = processor.get_legacy_style_config()
            print(f"  Legacy conversion: [SUCCESS]")

        except Exception as e:
            print(f"[ERROR] {e}")


def test_unified_configuration():
    """Test unified configuration format processing"""
    print("\n\n=== Testing Unified Configuration Format ===")

    processor = ConfigurationProcessor()

    # Test unified configuration examples
    unified_configs = [
        {
            "name": "ONNX graph-level config",
            "config": {
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
                        "min_psnr_threshold": 35.0
                    }
                }
            }
        },
        {
            "name": "PyTorch-level config",
            "config": {
                "chunking": {
                    "method": "pytorch_level",
                    "pytorch_level": {
                        "chunk_strategy": "by_layers",
                        "layer_distribution": [8, 8, 8],
                        "include_embedding_in_first": True,
                        "include_lm_head_in_last": True
                    },
                    "common": {
                        "output_directory": "./torch_chunks",
                        "verify_chunks": True,
                        "min_psnr_threshold": 40.0
                    }
                }
            }
        },
        {
            "name": "Comprehensive config with both methods",
            "config": {
                "chunking": {
                    "method": "onnx_graph_level",
                    "pytorch_level": {
                        "chunk_strategy": "by_layers",
                        "layer_distribution": [6, 6, 6, 6],
                        "balance_memory_usage": True
                    },
                    "onnx_graph_level": {
                        "strategy": "layer_based",
                        "distribution": "equal_layers",
                        "num_chunks": 4,
                        "respect_layer_boundaries": True,
                        "custom_chunk_sizes": [0.25, 0.25, 0.25, 0.25]
                    },
                    "common": {
                        "output_directory": "./dual_chunks",
                        "verify_chunks": True,
                        "min_psnr_threshold": 45.0,
                        "save_metadata": True,
                        "verbose": True
                    }
                }
            }
        }
    ]

    for test_case in unified_configs:
        print(f"\n--- {test_case['name']} ---")
        try:
            config = processor.process_configuration(test_case['config'])
            print(f"[SUCCESS] Successfully processed unified configuration")
            print(f"  Method: {config.method}")
            print(f"  Legacy mode: {processor.is_legacy_mode()}")

            # Get method-specific configuration
            method_name, method_config = processor.get_method_specific_config()
            print(f"  Active method: {method_name}")
            print(f"  Method config: {method_config.__dict__}")

        except Exception as e:
            print(f"[ERROR] {e}")


def test_error_handling():
    """Test error handling for invalid configurations"""
    print("\n\n=== Testing Error Handling ===")

    processor = ConfigurationProcessor()

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
                }
            }
        },
        {
            "name": "Invalid num_chunks",
            "config": {
                "chunking": {
                    "method": "onnx_graph_level",
                    "onnx_graph_level": {
                        "strategy": "weight_based",
                        "num_chunks": 1  # Must be >= 2
                    }
                }
            }
        },
        {
            "name": "Invalid chunk strategy",
            "config": {
                "enable_chunking": True,
                "chunk_strategy": "invalid_strategy",
                "num_chunks": 3
            }
        },
        {
            "name": "Missing layer distribution",
            "config": {
                "chunking": {
                    "method": "pytorch_level",
                    "pytorch_level": {
                        "chunk_strategy": "by_layers",
                        # Missing layer_distribution
                    }
                }
            }
        },
        {
            "name": "Invalid PSNR threshold",
            "config": {
                "chunking": {
                    "method": "onnx_graph_level",
                    "onnx_graph_level": {
                        "strategy": "weight_based",
                        "num_chunks": 2
                    },
                    "common": {
                        "min_psnr_threshold": -10.0  # Must be positive
                    }
                }
            }
        }
    ]

    for test_case in invalid_configs:
        print(f"\n--- {test_case['name']} ---")
        try:
            config = processor.process_configuration(test_case['config'])
            print(f"[ERROR] Should have failed but didn't!")
        except Exception as e:
            print(f"[SUCCESS] Correctly caught error: {e}")


def test_configuration_saving():
    """Test configuration saving functionality"""
    print("\n\n=== Testing Configuration Saving ===")

    processor = ConfigurationProcessor()

    # Test configuration
    test_config = {
        "chunking": {
            "method": "onnx_graph_level",
            "onnx_graph_level": {
                "strategy": "weight_based",
                "distribution": "equal_memory",
                "num_chunks": 3
            },
            "common": {
                "output_directory": "./test_save",
                "verify_chunks": True
            }
        }
    }

    try:
        config = processor.process_configuration(test_config)

        # Save to temporary file
        temp_dir = "temp_test_dir"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, "temp_test_config.json")
        processor.save_configuration(temp_file)

        # Verify file was created and contains valid JSON
        if os.path.exists(temp_file):
            with open(temp_file, 'r') as f:
                saved_data = json.load(f)

            print("[SUCCESS] Configuration saved successfully")
            print(f"  Saved method: {saved_data['chunking']['method']}")
            print(f"  Legacy mode: {saved_data['metadata']['legacy_mode']}")

            # Clean up
            os.remove(temp_file)
            os.rmdir(temp_dir)
            print("[SUCCESS] Temporary file cleaned up")
        else:
            print("[ERROR] Configuration file was not created")

    except Exception as e:
        print(f"[ERROR] Error during save test: {e}")


def main():
    """Run all tests"""
    print("Unified Chunking Configuration System Test Suite")
    print("=" * 60)

    try:
        test_legacy_configuration()
        test_unified_configuration()
        test_error_handling()
        test_configuration_saving()

        print("\n\n" + "=" * 60)
        print("All tests completed!")

        # Print sample configurations
        print("\n\n=== Sample Configurations for Reference ===")
        print_configuration_examples()

    except Exception as e:
        print(f"Test suite failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()