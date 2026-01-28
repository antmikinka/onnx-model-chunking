#!/usr/bin/env python3
"""
Simple test to verify that the architectural chunking issue is fixed.
This test demonstrates that chunked models now have actual nodes.
"""

import os
import sys
import onnx
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from builder import create_model

def test_chunking_fix():
    """Test that chunking now works with proper model creation"""

    print("Testing ONNX Model Chunking Architecture Fix")
    print("=" * 50)

    # Setup directories
    output_dir = Path("test_chunking_fix_output")
    chunk_dir = output_dir / "chunks"
    output_dir.mkdir(exist_ok=True)
    chunk_dir.mkdir(exist_ok=True)

    try:
        print("\n1. Creating chunked models using proper workflow...")

        # Use the correct workflow: create_model with chunking options
        create_model(
            model_name="Qwen/Qwen2.5-0.5B",
            input_path="Qwen/Qwen2.5-0.5B",
            output_dir=str(chunk_dir),
            precision="fp16",
            execution_provider="cpu",
            cache_dir=str(output_dir / "cache"),
            num_hidden_layers=24,
            # Enable chunking with layer-based strategy
            enable_chunking=True,
            chunk_strategy="layer_based",
            chunk_distribution="equal_memory",
            num_chunks=3,
            respect_layer_boundaries=True,
            chunk_output_dir=str(chunk_dir),
            verify_chunks=False
        )

        print("\n2. Analyzing created chunk models...")

        # Find chunk files
        chunk_files = list(chunk_dir.glob("*.onnx"))
        chunk_files.sort()

        if not chunk_files:
            print("❌ FAILED: No chunk files were created")
            return False

        print(f"✓ Found {len(chunk_files)} chunk files")

        # Analyze each chunk
        total_nodes = 0
        all_chunks_valid = True

        for i, chunk_file in enumerate(chunk_files):
            try:
                print(f"\nAnalyzing chunk {i+1}: {chunk_file.name}")

                # Load ONNX model
                chunk_model = onnx.load(str(chunk_file))

                # Check model validity
                onnx.checker.check_model(chunk_model)

                node_count = len(chunk_model.graph.node)
                total_nodes += node_count

                print(f"  ✓ Valid ONNX model")
                print(f"  ✓ Nodes: {node_count}")
                print(f"  ✓ Inputs: {len(chunk_model.graph.input)}")
                print(f"  ✓ Outputs: {len(chunk_model.graph.output)}")
                print(f"  ✓ Size: {chunk_file.stat().st_size / (1024**2):.2f}MB")

                if node_count == 0:
                    print(f"  ❌ FAILED: Chunk has 0 nodes!")
                    all_chunks_valid = False
                else:
                    print(f"  ✅ SUCCESS: Chunk has actual nodes!")

            except Exception as e:
                print(f"  ❌ FAILED: Error analyzing chunk: {e}")
                all_chunks_valid = False

        print(f"\n3. Final Results:")
        print(f"   Total chunks created: {len(chunk_files)}")
        print(f"   Total nodes across all chunks: {total_nodes}")
        print(f"   Average nodes per chunk: {total_nodes // len(chunk_files)}")

        if all_chunks_valid and total_nodes > 0:
            print(f"\n🎉 SUCCESS: Architectural issue is FIXED!")
            print(f"   ✓ Chunked models have actual nodes")
            print(f"   ✓ Total {total_nodes} nodes distributed across {len(chunk_files)} chunks")
            print(f"   ✓ Chunking integrates properly with model creation workflow")
            return True
        else:
            print(f"\n❌ FAILED: Chunking still has issues")
            return False

    except Exception as e:
        print(f"\n❌ FAILED: Error during chunking test: {e}")
        return False

if __name__ == "__main__":
    success = test_chunking_fix()
    sys.exit(0 if success else 1)