#!/usr/bin/env python3
"""
Split Local ONNX Model CLI

Command-line interface for splitting local ONNX models into chunks using
weight-based or layer-based strategies.

Usage:
    python split_local_model.py <model_path> [options]

Example:
    python split_local_model.py "C:\\path\\to\\model" --num-chunks 2 --strategy weight_based
    python split_local_model.py "C:\\path\\to\\model" --strategy layer_based --layers-per-chunk 7
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Split local ONNX models into coherent chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split a model in a directory (looks for model.onnx)
  python split_local_model.py "C:\\path\\to\\model_folder"
  
  # Split a specific ONNX file into 3 chunks (weight-based)
  python split_local_model.py "C:\\path\\to\\model.onnx" --num-chunks 3
  
  # Layer-based splitting: 7 layers per chunk  
  python split_local_model.py "C:\\path\\to\\model" --strategy layer_based --layers-per-chunk 7
  
  # Layer-based splitting: one layer per file (like grok_hf_split.py)
  python split_local_model.py "C:\\path\\to\\model" --strategy layer_based --layers-per-chunk 1
  
  # Example with the Qwen3 model:
  python split_local_model.py "C:\\Users\\antmi\\Downloads\\...\\Qwen3-0.6B-...\\model" --strategy layer_based
"""
    )
    
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to ONNX model file or directory containing model.onnx"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for chunk models (default: <model_path>_chunks)"
    )
    
    parser.add_argument(
        "--num-chunks", "-n",
        type=int,
        default=2,
        help="Number of chunks for weight-based splitting (default: 2)"
    )
    
    parser.add_argument(
        "--strategy", "-s",
        type=str,
        choices=["weight_based", "layer_based"],
        default="weight_based",
        help="Splitting strategy (default: weight_based)"
    )
    
    parser.add_argument(
        "--layers-per-chunk", "-l",
        type=int,
        default=7,
        help="Number of layers per chunk for layer_based strategy (default: 7, use 1 for one layer per file)"
    )
    
    parser.add_argument(
        "--distribution", "-d",
        type=str,
        choices=["equal_memory", "equal_layers", "custom"],
        default="equal_memory",
        help="Distribution type for chunks (default: equal_memory)"
    )
    
    parser.add_argument(
        "--split-ratio", "-r",
        type=float,
        default=0.5,
        help="Split ratio for 2-chunk weight-based splitting (default: 0.5)"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify chunk correctness after splitting"
    )
    
    parser.add_argument(
        "--no-onnx-ir",
        action="store_true",
        help="Use standard onnx instead of onnx_ir for graph manipulation"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze the model without splitting"
    )
    
    args = parser.parse_args()
    
    # Import here to avoid import errors if dependencies are missing
    try:
        from local_model_splitter import LocalONNXModelSplitter, split_local_model, split_local_model_by_layers
    except ImportError as e:
        print(f"Error: Failed to import local_model_splitter: {e}")
        print("\nMake sure you have the required dependencies installed:")
        print("  pip install onnx numpy")
        print("\nOptionally for better performance:")
        print("  pip install onnx_ir")
        sys.exit(1)
    
    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if model_path.is_file():
            output_dir = model_path.parent / f"{model_path.stem}_chunks"
        else:
            if args.strategy == "layer_based":
                output_dir = model_path / "layer_chunks"
            else:
                output_dir = model_path / "chunks"
    
    print("=" * 60)
    print("Local ONNX Model Splitter")
    print("=" * 60)
    print(f"\nModel path: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Strategy: {args.strategy}")
    
    if args.strategy == "weight_based":
        print(f"Number of chunks: {args.num_chunks}")
        print(f"Distribution: {args.distribution}")
        if args.num_chunks == 2:
            print(f"Split ratio: {args.split_ratio}")
    else:  # layer_based
        print(f"Layers per chunk: {args.layers_per_chunk}")
    
    # Create configuration
    config = {
        "num_chunks": args.num_chunks,
        "chunk_strategy": args.strategy,
        "chunk_distribution": args.distribution,
        "chunk_split_ratio": args.split_ratio,
    }
    
    # Create splitter
    use_onnx_ir = not args.no_onnx_ir
    splitter = LocalONNXModelSplitter(
        model_path=str(model_path),
        config=config,
        use_onnx_ir=use_onnx_ir
    )
    
    start_time = time.time()
    
    try:
        # Load and analyze model
        print("\n" + "-" * 40)
        splitter.load_model()
        analysis = splitter.analyze_model()
        
        if args.analyze_only:
            print("\n" + "=" * 60)
            print("Analysis complete (--analyze-only specified)")
            print("=" * 60)
            return 0
        
        # Split the model based on strategy
        print("\n" + "-" * 40)
        
        if args.strategy == "layer_based":
            result = splitter.split_by_layers(
                output_dir=str(output_dir),
                layers_per_chunk=args.layers_per_chunk,
                verify_chunks=args.verify
            )
        else:
            result = splitter.split_model(
                output_dir=str(output_dir),
                num_chunks=args.num_chunks,
                strategy=args.strategy,
                verify_chunks=args.verify
            )
        
        elapsed_time = time.time() - start_time
        
        # Print summary
        print("\n" + "=" * 60)
        print("Splitting Complete!")
        print("=" * 60)
        print(f"\nTime elapsed: {elapsed_time:.2f} seconds")
        print(f"Total chunks created: {len(result.chunk_models)}")
        print(f"Total memory: {result.total_memory_mb:.2f} MB")
        
        print(f"\nChunk details:")
        for info in result.chunk_info:
            print(f"  Chunk {info.chunk_idx}:")
            print(f"    Nodes: {info.start_node_idx} to {info.end_node_idx} ({info.node_count} nodes)")
            print(f"    Memory: {info.memory_mb:.2f} MB")
            
            # Show layers for layer-based splitting
            if "chunk_layer_ranges" in result.metadata:
                layers = result.metadata["chunk_layer_ranges"][info.chunk_idx]
                if layers:
                    print(f"    Layers: {min(layers)}-{max(layers)} ({len(layers)} layers)")
            
            print(f"    Boundary inputs: {len(info.boundary_inputs)}")
            print(f"    Boundary outputs: {len(info.boundary_outputs)}")
        
        print(f"\nOutput files saved to: {output_dir}")
        if "chunk_paths" in result.metadata:
            for path in result.metadata["chunk_paths"]:
                print(f"  - {Path(path).name}")
        else:
            for i in range(len(result.chunk_models)):
                print(f"  - chunk_{i}.onnx")
        print(f"  - chunking_metadata.json")
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

