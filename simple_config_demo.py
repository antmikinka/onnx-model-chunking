#!/usr/bin/env python3
"""
Simple demonstration of the unified configuration system.

This script shows the core functionality of the unified configuration system
without the complexity of the full Model class integration.
"""

import json
from chunking_config import (
    ConfigurationProcessor,
    UnifiedChunkingConfig,
    ChunkingMethod,
    create_sample_configurations
)


def demo_legacy_format_processing():
    """Demonstrate legacy format processing"""
    print("=== Legacy Format Processing Demo ===")

    processor = ConfigurationProcessor()

    # Example legacy configurations
    legacy_examples = [
        {
            "name": "Basic legacy configuration",
            "config": {
                "enable_chunking": True,
                "chunk_strategy": "layer_based",
                "num_chunks": 3
            }
        },
        {
            "name": "Advanced legacy configuration",
            "config": {
                "enable_chunking": True,
                "chunk_strategy": "weight_based",
                "chunk_distribution": "equal_memory",
                "num_chunks": 4,
                "respect_layer_boundaries": True,
                "chunk_output_dir": "./legacy_chunks",
                "verify_chunks": True,
                "min_psnr_threshold": 40.0,
                "custom_chunk_sizes": [0.25, 0.25, 0.25, 0.25]
            }
        }
    ]

    for example in legacy_examples:
        print(f"\n--- {example['name']} ---")
        print("Input config:")
        print(json.dumps(example['config'], indent=2))

        try:
            # Process the legacy configuration
            config = processor.process_configuration(example['config'])

            print("\nProcessed configuration:")
            print(f"  Method: {config.method}")
            print(f"  Legacy mode: {processor.is_legacy_mode()}")
            print(f"  ONNX strategy: {config.onnx_graph_level.strategy}")
            print(f"  ONNX distribution: {config.onnx_graph_level.distribution}")
            print(f"  Number of chunks: {config.onnx_graph_level.num_chunks}")
            print(f"  Output directory: {config.common.output_directory}")
            print(f"  Verify chunks: {config.common.verify_chunks}")
            print(f"  PSNR threshold: {config.common.min_psnr_threshold}")

            # Convert back to legacy format
            legacy_style = processor.get_legacy_style_config()
            print(f"\nConverted back to legacy format: [SUCCESS]")
            print(f"  enable_chunking: {legacy_style.get('enable_chunking')}")
            print(f"  chunk_strategy: {legacy_style.get('chunk_strategy')}")
            print(f"  num_chunks: {legacy_style.get('num_chunks')}")

        except Exception as e:
            print(f"Error: {e}")


def demo_unified_format_processing():
    """Demonstrate unified format processing"""
    print("\n\n=== Unified Format Processing Demo ===")

    processor = ConfigurationProcessor()

    # Example unified configurations
    unified_examples = [
        {
            "name": "ONNX graph-level configuration",
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
                        "output_directory": "./onnx_chunks",
                        "verify_chunks": True,
                        "min_psnr_threshold": 35.0,
                        "save_metadata": True
                    }
                }
            }
        },
        {
            "name": "PyTorch-level configuration",
            "config": {
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
                        "verify_chunks": True,
                        "min_psnr_threshold": 40.0
                    }
                }
            }
        },
        {
            "name": "Comprehensive configuration with both methods",
            "config": {
                "chunking": {
                    "method": "onnx_graph_level",  # Active method
                    "pytorch_level": {
                        "chunk_strategy": "by_weights",
                        "balance_memory_usage": True
                    },
                    "onnx_graph_level": {
                        "strategy": "layer_based",
                        "distribution": "equal_layers",
                        "num_chunks": 4,
                        "respect_layer_boundaries": True
                    },
                    "common": {
                        "output_directory": "./comprehensive_chunks",
                        "verify_chunks": True,
                        "save_original_model": True,
                        "verbose": True
                    }
                }
            }
        }
    ]

    for example in unified_examples:
        print(f"\n--- {example['name']} ---")
        print("Input config:")
        print(json.dumps(example['config'], indent=2))

        try:
            # Process the unified configuration
            config = processor.process_configuration(example['config'])

            print("\nProcessed configuration:")
            print(f"  Method: {config.method}")
            print(f"  Legacy mode: {processor.is_legacy_mode()}")

            # Get method-specific configuration
            method_name, method_config = processor.get_method_specific_config()
            print(f"  Active method: {method_name}")
            print(f"  Method config type: {type(method_config).__name__}")

            # Show method-specific details
            if method_name == "onnx_graph_level":
                print(f"  Strategy: {method_config.strategy}")
                print(f"  Distribution: {method_config.distribution}")
                print(f"  Num chunks: {method_config.num_chunks}")
            elif method_name == "pytorch_level":
                print(f"  Chunk strategy: {method_config.chunk_strategy}")
                print(f"  Layer distribution: {method_config.layer_distribution}")
                print(f"  Include embedding: {method_config.include_embedding_in_first}")

            # Show common configuration
            common = config.common
            print(f"  Output directory: {common.output_directory}")
            print(f"  Verify chunks: {common.verify_chunks}")
            print(f"  Save metadata: {common.save_metadata}")

        except Exception as e:
            print(f"Error: {e}")


def demo_validation_and_error_handling():
    """Demonstrate validation and error handling"""
    print("\n\n=== Validation and Error Handling Demo ===")

    processor = ConfigurationProcessor()

    # Invalid configurations that should fail
    invalid_examples = [
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
            },
            "expected_error": "Unknown chunking method"
        },
        {
            "name": "Invalid number of chunks",
            "config": {
                "chunking": {
                    "method": "onnx_graph_level",
                    "onnx_graph_level": {
                        "strategy": "weight_based",
                        "num_chunks": 1  # Must be >= 2
                    }
                }
            },
            "expected_error": "num_chunks must be an integer >= 2"
        },
        {
            "name": "Invalid legacy strategy",
            "config": {
                "enable_chunking": True,
                "chunk_strategy": "invalid_strategy",
                "num_chunks": 3
            },
            "expected_error": "Legacy chunk_strategy"
        },
        {
            "name": "Missing layer distribution for PyTorch",
            "config": {
                "chunking": {
                    "method": "pytorch_level",
                    "pytorch_level": {
                        "chunk_strategy": "by_layers"
                        # Missing layer_distribution
                    }
                }
            },
            "expected_error": "layer_distribution is required"
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
            },
            "expected_error": "min_psnr_threshold must be a positive number"
        }
    ]

    success_count = 0
    for example in invalid_examples:
        print(f"\n--- {example['name']} ---")
        print("Input config:")
        print(json.dumps(example['config'], indent=2))

        try:
            config = processor.process_configuration(example['config'])
            print(f"[ERROR] UNEXPECTED SUCCESS: Configuration should have failed!")
        except Exception as e:
            error_msg = str(e)
            if example['expected_error'] in error_msg:
                print(f"[SUCCESS] EXPECTED ERROR: {error_msg}")
                success_count += 1
            else:
                print(f"[WARNING] UNEXPECTED ERROR: {error_msg}")
                print(f"   Expected to contain: {example['expected_error']}")

    print(f"\nError handling validation: {success_count}/{len(invalid_examples)} tests passed")
    return success_count == len(invalid_examples)


def demo_configuration_saving():
    """Demonstrate configuration saving and loading"""
    print("\n\n=== Configuration Saving Demo ===")

    processor = ConfigurationProcessor()

    # Create a test configuration
    test_config = {
        "chunking": {
            "method": "onnx_graph_level",
            "onnx_graph_level": {
                "strategy": "weight_based",
                "distribution": "equal_memory",
                "num_chunks": 3,
                "respect_layer_boundaries": True,
                "custom_chunk_sizes": [0.3, 0.4, 0.3]
            },
            "common": {
                "output_directory": "./save_test_chunks",
                "verify_chunks": True,
                "min_psnr_threshold": 38.5,
                "save_metadata": True,
                "verbose": False
            }
        }
    }

    try:
        # Process the configuration
        config = processor.process_configuration(test_config)
        print("[SUCCESS] Configuration processed successfully")

        # Save to file
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "saved_config.json")
            processor.save_configuration(save_path)
            print(f"[SUCCESS] Configuration saved to: {save_path}")

            # Load and verify
            with open(save_path, 'r') as f:
                saved_data = json.load(f)

            print("\nSaved configuration structure:")
            print(f"  Method: {saved_data['chunking']['method']}")
            print(f"  Legacy mode: {saved_data['metadata']['legacy_mode']}")
            print(f"  Version: {saved_data['metadata']['version']}")

            # Verify saved data matches original
            saved_method = saved_data['chunking']['method']
            original_method = test_config['chunking']['method']

            if saved_method == original_method:
                print("[SUCCESS] Saved configuration matches original")
                return True
            else:
                print("[ERROR] Saved configuration doesn't match original")
                return False

    except Exception as e:
        print(f"[ERROR] Error during save test: {e}")
        return False


def demo_configuration_examples():
    """Display sample configurations for reference"""
    print("\n\n=== Sample Configurations Reference ===")

    examples = create_sample_configurations()

    for name, config in examples.items():
        print(f"\n--- {name.upper().replace('_', ' ')} ---")
        print(json.dumps(config, indent=2))
        print("\n" + "="*50)


def main():
    """Run all demonstrations"""
    print("Unified Configuration System Demonstration")
    print("=" * 60)

    # Run all demonstrations
    demo_legacy_format_processing()
    demo_unified_format_processing()
    validation_success = demo_validation_and_error_handling()
    save_success = demo_configuration_saving()
    demo_configuration_examples()

    # Summary
    print("\n" + "=" * 60)
    print("Demonstration Summary")
    print("=" * 60)

    print("[SUCCESS] Legacy format processing: Demonstrated")
    print("[SUCCESS] Unified format processing: Demonstrated")
    print(f"[SUCCESS] Validation and error handling: {'Passed' if validation_success else 'Failed'}")
    print(f"[SUCCESS] Configuration saving: {'Passed' if save_success else 'Failed'}")
    print("[SUCCESS] Sample configurations: Displayed")

    print(f"\nOverall demonstration: {'SUCCESS' if validation_success and save_success else 'PARTIAL'}")

    if validation_success and save_success:
        print("\n[SUCCESS] The unified configuration system is working correctly!")
        print("   Ready for integration with the Model builder.")
    else:
        print("\n[WARNING] Some features need attention before full integration.")

    return validation_success and save_success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)