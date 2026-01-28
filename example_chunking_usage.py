#!/usr/bin/env python3
"""
Example script demonstrating ONNX model chunking functionality.

This script shows how to:
1. Configure model chunking options
2. Create chunked models
3. Verify chunk accuracy
4. Use chunked models in an inference pipeline

Usage:
    python example_chunking_usage.py --model-path <path_to_model> --output-dir <output_directory>
"""

import argparse
import json
import os
import sys
import numpy as np
import onnxruntime as ort

# Add the current directory to the path to import the builder module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_sample_config():
    """Create a sample configuration for demonstration"""
    class SampleConfig:
        def __init__(self):
            self.seq_length = 2048
            self.max_position_embeddings = 2048
            self.ffn_hidden_size = 11008
            self.hidden_size = 4096
            self.num_key_value_heads = 32
            self.num_attention_heads = 32
            self.head_dim = 128
            self.num_hidden_layers = 32
            self.vocab_size = 32000
            self.hidden_activation = "silu"
            self._name_or_path = "meta-llama/Llama-2-7b-hf"
            self.architectures = ["LlamaForCausalLM"]
            self.quantization_config = None

    return SampleConfig()

def demonstrate_chunking_configuration():
    """Demonstrate different chunking configurations"""
    print("=== ONNX Model Chunking Configuration Examples ===\n")

    # Example 1: Basic chunking for memory optimization
    print("1. Basic Memory Optimization:")
    basic_config = {
        "enable_chunking": True,
        "chunk_output_dir": "./chunks_memory_optimized",
        "verify_chunks": False,  # Skip verification for faster processing
        "chunk_split_ratio": 0.6,  # Larger first chunk for better caching
        "min_psnr_threshold": 35.0
    }
    print(f"   Configuration: {json.dumps(basic_config, indent=2)}")
    print()

    # Example 2: High-accuracy chunking with verification
    print("2. High-Accuracy Production Setup:")
    production_config = {
        "enable_chunking": True,
        "chunk_output_dir": "./chunks_production",
        "verify_chunks": True,  # Enable verification for production
        "chunk_split_ratio": 0.5,  # Balanced split
        "min_psnr_threshold": 40.0,  # Higher accuracy requirement
        "filename": "production_model.onnx"
    }
    print(f"   Configuration: {json.dumps(production_config, indent=2)}")
    print()

    # Example 3: Memory-constrained deployment
    print("3. Memory-Constrained Deployment:")
    memory_config = {
        "enable_chunking": True,
        "chunk_output_dir": "./chunks_constrained",
        "verify_chunks": True,
        "chunk_split_ratio": 0.3,  # Small first chunk for faster loading
        "min_psnr_threshold": 35.0,
        "filename": "constrained_model.onnx"
    }
    print(f"   Configuration: {json.dumps(memory_config, indent=2)}")
    print()

def simulate_chunking_workflow():
    """Simulate the chunking workflow without requiring actual models"""
    print("=== Simulated Chunking Workflow ===\n")

    # This would normally be done with a real model
    print("Step 1: Loading model configuration...")
    config = create_sample_config()
    print(f"   Model: {config.architectures[0]}")
    print(f"   Layers: {config.num_hidden_layers}")
    print(f"   Hidden size: {config.hidden_size}")
    print()

    print("Step 2: Configuring chunking options...")
    extra_options = {
        "enable_chunking": True,
        "chunk_output_dir": "./example_chunks",
        "verify_chunks": True,
        "chunk_split_ratio": 0.5,
        "min_psnr_threshold": 35.0
    }
    print(f"   Chunk split ratio: {extra_options['chunk_split_ratio']}")
    print(f"   Verification enabled: {extra_options['verify_chunks']}")
    print(f"   PSNR threshold: {extra_options['min_psnr_threshold']} dB")
    print()

    print("Step 3: Analyzing model graph...")
    # Simulate graph analysis
    total_nodes = 626  # Typical for Llama 7B
    total_size_mb = 13000  # 13GB model
    split_idx = int(total_nodes * 0.5)
    first_chunk_size = total_size_mb * 0.5
    boundary_tensors = ["hidden_states_16", "attention_mask_16", "position_ids_16"]

    print(f"   Total nodes: {total_nodes}")
    print(f"   Total size: {total_size_mb:.1f} MB")
    print(f"   Split at node: {split_idx}")
    print(f"   First chunk size: {first_chunk_size:.1f} MB")
    print(f"   Boundary tensors: {len(boundary_tensors)}")
    print()

    print("Step 4: Creating chunked models...")
    print(f"   First chunk: {split_idx} nodes")
    print(f"   Second chunk: {total_nodes - split_idx} nodes")
    print()

    print("Step 5: Verifying chunk accuracy...")
    # Simulate PSNR verification
    psnr_values = {
        "logits": 42.1,
        "hidden_states": 45.3,
        "attention_weights": 38.7
    }
    for output_name, psnr in psnr_values.items():
        status = "✓ PASSED" if psnr >= 35.0 else "✗ FAILED"
        print(f"   {output_name}: {psnr:.1f} dB {status}")
    print()

    print("Step 6: Saving chunked models...")
    output_files = [
        "example_chunks/model_first_chunk.onnx",
        "example_chunks/model_second_chunk.onnx",
        "example_chunks/chunking_metadata.json"
    ]
    for file_path in output_files:
        print(f"   Saved: {file_path}")
    print()

def demonstrate_inference_pipeline():
    """Demonstrate how to use chunked models in an inference pipeline"""
    print("=== Inference Pipeline Example ===\n")

    print("Using chunked models for inference:")
    print()

    # Step 1: Load chunked models
    print("1. Load chunked models:")
    print("   first_session = ort.InferenceSession('model_first_chunk.onnx')")
    print("   second_session = ort.InferenceSession('model_second_chunk.onnx')")
    print()

    # Step 2: Prepare inputs
    print("2. Prepare input data:")
    sample_inputs = {
        "input_ids": "np.array([[1, 2, 3, 4, 5]], dtype=np.int64)",
        "attention_mask": "np.array([[1, 1, 1, 1, 1]], dtype=np.int64)",
        "position_ids": "np.array([[0, 1, 2, 3, 4]], dtype=np.int64)"
    }
    for name, value in sample_inputs.items():
        print(f"   {name}: {value}")
    print()

    # Step 3: Run first chunk
    print("3. Execute first chunk:")
    print("   first_outputs = first_session.run(None, inputs)")
    print("   # Returns: hidden_states_16, attention_mask_16, position_ids_16")
    print()

    # Step 4: Prepare second chunk inputs
    print("4. Prepare inputs for second chunk:")
    print("   second_inputs = inputs.copy()")
    print("   for boundary_tensor in boundary_tensors:")
    print("       second_inputs[boundary_tensor] = first_outputs[boundary_tensor]")
    print()

    # Step 5: Run second chunk
    print("5. Execute second chunk:")
    print("   final_outputs = second_session.run(None, second_inputs)")
    print("   # Returns: logits (final model output)")
    print()

def show_chunking_benefits():
    """Show the benefits of model chunking"""
    print("=== Benefits of Model Chunking ===\n")

    benefits = [
        ("Memory Optimization", "Run larger models on memory-constrained devices by loading only necessary chunks"),
        ("Faster Loading", "Load only the first chunk for initial processing, reducing startup time"),
        ("Pipeline Parallelism", "Run different chunks on different devices for better throughput"),
        ("Selective Execution", "Skip chunks when only partial model functionality is needed"),
        ("Distribution Flexibility", "Distribute model chunks across multiple servers or edge devices"),
        ("Incremental Updates", "Update individual chunks without redistributing the entire model")
    ]

    for i, (title, description) in enumerate(benefits, 1):
        print(f"{i}. {title}")
        print(f"   {description}")
        print()

def main():
    parser = argparse.ArgumentParser(description="ONNX Model Chunking Example")
    parser.add_argument("--model-path", type=str, help="Path to the model (for demonstration)")
    parser.add_argument("--output-dir", type=str, default="./example_output",
                       help="Output directory for chunked models")

    args = parser.parse_args()

    print("ONNX Model Chunking Example")
    print("=" * 50)
    print()

    # Show configuration examples
    demonstrate_chunking_configuration()

    # Show simulated workflow
    simulate_chunking_workflow()

    # Show inference pipeline
    demonstrate_inference_pipeline()

    # Show benefits
    show_chunking_benefits()

    print("=" * 50)
    print("Example completed successfully!")
    print()
    print("To use chunking with real models:")
    print("1. Set enable_chunking: True in extra_options")
    print("2. Configure chunk_output_dir for saving chunks")
    print("3. Set verify_chunks: True for production use")
    print("4. Adjust chunk_split_ratio and min_psnr_threshold as needed")
    print()
    print("See CHUNKING_DOCUMENTATION.md for detailed information.")

if __name__ == "__main__":
    main()