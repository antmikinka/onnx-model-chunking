#!/usr/bin/env python3
"""
Split Qwen3-8B AWQ ONNX model for RyzenAI NPU modular inference.

This script splits the Qwen3-8B AWQ quantized ONNX model into per-component
files for modular inference on RyzenAI NPU hardware.

Model: Qwen3-8B-awq-quant-onnx-ryzenai-1.7-hybrid
- 36 transformer layers
- AWQ INT4 quantization
- Hybrid CPU/NPU execution

Usage:
    python split_qwen3_8b_awq.py [--model-path PATH] [--output-dir PATH]
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from chunking_utils import split_onnx


def main():
    parser = argparse.ArgumentParser(
        description="Split Qwen3-8B AWQ ONNX model for NPU inference"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=r"C:\Users\antmi\Downloads\Qwen3-8B-awq-quant-onnx-ryzenai-1.7-hybrid",
        help="Path to the Qwen3-8B model directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"./output",
        help="Output directory for split model"
    )
    parser.add_argument(
        "--copy-configs",
        action="store_true",
        default=True,
        help="Copy tokenizer and config files to output"
    )
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    
    # Validate model path
    onnx_file = model_path / "model_jit.onnx"
    if not onnx_file.exists():
        print(f"ERROR: ONNX model not found at {onnx_file}")
        print("Expected model_jit.onnx in the model directory.")
        return 1
    
    print("=" * 60)
    print("Qwen3-8B AWQ Model Splitter for RyzenAI NPU")
    print("=" * 60)
    print(f"\nModel path: {model_path}")
    print(f"ONNX file: {onnx_file}")
    print(f"Output directory: {output_dir}")
    
    # Check for external data
    data_file = model_path / "model_jit.onnx.data"
    if data_file.exists():
        data_size_gb = data_file.stat().st_size / (1024**3)
        print(f"External weights: {data_file.name} ({data_size_gb:.2f} GB)")
    
    # Check for pre-compiled NPU binary
    bin_file = model_path / "model_jit.bin"
    if bin_file.exists():
        bin_size_gb = bin_file.stat().st_size / (1024**3)
        print(f"Pre-compiled NPU binary: {bin_file.name} ({bin_size_gb:.2f} GB)")
        print("  NOTE: This pre-compiled binary is NOT being split.")
    
    print("\nStarting split operation...")
    print("-" * 60)
    
    # Perform the split
    try:
        result = split_onnx(
            model=str(onnx_file),
            output=str(output_dir),
            split_only=True,  # Skip inference validation
        )
    except Exception as e:
        print(f"\nERROR during split: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("-" * 60)
    print(f"\nSplit complete!")
    print(f"  Output directory: {result.output_dir}")
    print(f"  Number of layers: {result.num_layers}")
    print(f"  Components created: {len(result.component_files)}")
    
    # Copy config files
    if args.copy_configs:
        print("\nCopying configuration files...")
        config_files = [
            "genai_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
        ]
        
        copied = 0
        for config_file in config_files:
            src = model_path / config_file
            if src.exists():
                dst = Path(result.output_dir) / config_file
                shutil.copy(str(src), str(dst))
                print(f"  Copied {config_file}")
                copied += 1
        
        print(f"  Copied {copied} config files")
    
    # Summary
    print("\n" + "=" * 60)
    print("OUTPUT STRUCTURE")
    print("=" * 60)
    
    output_path = Path(result.output_dir)
    if output_path.exists():
        files = sorted(output_path.iterdir())
        total_size = 0
        for f in files:
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                total_size += size_mb
                print(f"  {f.name}: {size_mb:.2f} MB")
        print(f"\nTotal size: {total_size:.2f} MB")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Use SessionManager to load split components")
    print("2. Configure RyzenAIExecutionProvider for NPU inference")
    print("3. Run sequential inference through layer pipeline")
    print("\nExample:")
    print("  from pipeline_utils import SessionManager, SessionConfig")
    print("  from pipeline_utils import get_provider_for_device")
    print()
    print("  providers = get_provider_for_device('npu')")
    print("  config = SessionConfig(providers=providers)")
    print(f"  manager = SessionManager('{result.output_dir}', config)")
    print("  manager.load_all()")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
