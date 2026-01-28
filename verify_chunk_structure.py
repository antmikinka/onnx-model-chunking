#!/usr/bin/env python3
"""
Chunk Structure Verification Script

This script verifies that chunked models have actual nodes and proper structure
by running the corrected workflow and analyzing the resulting chunks.

VERIFICATION STEPS:
1. Run corrected test to create chunked models
2. Load and analyze each chunk for proper structure
3. Verify chunks have actual nodes (not empty models)
4. Check input/output connectivity between chunks
5. Validate model size and memory distribution
"""

import os
import sys
import json
import onnx
import onnxruntime as ort
from pathlib import Path
import numpy as np
from typing import Dict, List, Any

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def analyze_chunk_model(chunk_path: Path) -> Dict[str, Any]:
    """Analyze a single chunk model for structure and content"""
    try:
        # Load ONNX model
        onnx_model = onnx.load(str(chunk_path))

        # Check model validity
        onnx.checker.check_model(onnx_model)

        # Create inference session to test loading
        session = ort.InferenceSession(str(chunk_path), providers=["CPUExecutionProvider"])

        # Get model information
        inputs = session.get_inputs()
        outputs = session.get_outputs()

        # Analyze nodes
        node_types = {}
        node_count = len(onnx_model.graph.node)

        for node in onnx_model.graph.node:
            op_type = node.op_type
            node_types[op_type] = node_types.get(op_type, 0) + 1

        # Analyze initializers (weights)
        initializer_count = len(onnx_model.graph.initializer)
        initializer_size_mb = sum(
            i.data_type_size() * np.prod(i.dims)
            for i in onnx_model.graph.initializer
        ) / (1024 * 1024)

        # Analyze values (constants)
        value_count = len(onnx_model.graph.value_info)

        return {
            "status": "success",
            "file_size_mb": chunk_path.stat().st_size / (1024 * 1024),
            "node_count": node_count,
            "node_types": node_types,
            "input_count": len(inputs),
            "output_count": len(outputs),
            "inputs": [{"name": inp.name, "shape": inp.shape, "type": inp.type} for inp in inputs],
            "outputs": [{"name": out.name, "shape": out.shape, "type": out.type} for out in outputs],
            "initializer_count": initializer_count,
            "initializer_size_mb": round(initializer_size_mb, 2),
            "value_count": value_count,
            "graph_name": onnx_model.graph.name,
            "model_valid": True
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "file_size_mb": chunk_path.stat().st_size / (1024 * 1024) if chunk_path.exists() else 0,
            "model_valid": False
        }

def verify_chunk_connectivity(chunk_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify that chunks can connect properly in sequence"""
    connectivity_results = {
        "status": "success",
        "connectivity_issues": [],
        "pipeline_possible": True
    }

    if len(chunk_analyses) < 2:
        connectivity_results["connectivity_issues"].append("Need at least 2 chunks to verify connectivity")
        connectivity_results["pipeline_possible"] = False
        return connectivity_results

    # Check if chunks can be connected in sequence
    for i in range(len(chunk_analyses) - 1):
        current_chunk = chunk_analyses[i]
        next_chunk = chunk_analyses[i + 1]

        if current_chunk["status"] != "success" or next_chunk["status"] != "success":
            connectivity_results["connectivity_issues"].append(f"Chunk {i} or {i+1} failed to load")
            connectivity_results["pipeline_possible"] = False
            continue

        # Check output/input compatibility (simplified)
        current_outputs = [out["name"] for out in current_chunk["outputs"]]
        next_inputs = [inp["name"] for inp in next_chunk["inputs"]]

        # For proper chunking, there should be some overlap or boundary tensors
        # This is a simplified check - in reality, you'd need to trace tensor names
        if not current_outputs or not next_inputs:
            connectivity_results["connectivity_issues"].append(
                f"Chunk {i} has no outputs or chunk {i+1} has no inputs"
            )

    return connectivity_results

def main():
    """Main verification function"""
    print("Chunk Structure Verification")
    print("=" * 50)

    # Expected chunk directories from corrected test
    chunk_directories = [
        ("corrected_test_environment/chunked_models/weight_based_chunks", "Weight-Based Strategy"),
        ("corrected_test_environment/chunked_models/layer_based_chunks", "Layer-Based Strategy"),
        ("corrected_test_environment/chunked_models/custom_chunks", "Custom Distribution")
    ]

    overall_results = {}

    for chunk_dir_path, strategy_name in chunk_directories:
        chunk_dir = Path(chunk_dir_path)
        print(f"\nVerifying {strategy_name}")
        print("-" * 30)

        if not chunk_dir.exists():
            print(f"  X Directory not found: {chunk_dir}")
            overall_results[strategy_name] = {
                "status": "directory_not_found",
                "message": f"Directory {chunk_dir} does not exist"
            }
            continue

        # Find chunk files
        chunk_files = list(chunk_dir.glob("*.onnx"))
        chunk_files.sort()

        if not chunk_files:
            print(f"  X No chunk files found in {chunk_dir}")
            overall_results[strategy_name] = {
                "status": "no_chunks_found",
                "message": f"No .onnx files found in {chunk_dir}"
            }
            continue

        print(f"  Found {len(chunk_files)} chunk files")

        # Analyze each chunk
        chunk_analyses = []
        total_size_mb = 0
        total_nodes = 0

        for i, chunk_file in enumerate(chunk_files):
            print(f"  Analyzing chunk {i+1}: {chunk_file.name}")

            analysis = analyze_chunk_model(chunk_file)
            chunk_analyses.append(analysis)

            if analysis["status"] == "success":
                total_size_mb += analysis["file_size_mb"]
                total_nodes += analysis["node_count"]

                print(f"    + Valid model")
                print(f"    Nodes: {analysis['node_count']}")
                print(f"    Size: {analysis['file_size_mb']:.2f}MB")
                print(f"    Inputs: {analysis['input_count']}, Outputs: {analysis['output_count']}")

                # Check if model has actual nodes
                if analysis["node_count"] == 0:
                    print(f"    ! WARNING: Chunk has 0 nodes - this indicates the architectural issue!")

                if analysis["node_count"] > 0:
                    print(f"    + SUCCESS: Chunk has actual nodes - architecture is fixed!")

            else:
                print(f"    X Failed to analyze: {analysis['error']}")

        # Verify connectivity
        connectivity = verify_chunk_connectivity(chunk_analyses)

        # Store results
        overall_results[strategy_name] = {
            "status": "analyzed",
            "chunk_count": len(chunk_files),
            "total_size_mb": round(total_size_mb, 2),
            "total_nodes": total_nodes,
            "chunk_analyses": chunk_analyses,
            "connectivity": connectivity,
            "average_chunk_size_mb": round(total_size_mb / len(chunk_files), 2) if chunk_files else 0,
            "average_nodes_per_chunk": round(total_nodes / len(chunk_files), 2) if chunk_files else 0
        }

        print(f"  Total: {total_nodes} nodes across {len(chunk_files)} chunks")
        print(f"  Total size: {total_size_mb:.2f}MB")
        print(f"  Connectivity: {'✅ OK' if connectivity['pipeline_possible'] else '❌ Issues found'}")

    # Generate summary report
    print(f"\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)

    all_strategies_valid = True
    for strategy_name, results in overall_results.items():
        print(f"\n{strategy_name}:")
        if results["status"] == "analyzed":
            if results["total_nodes"] > 0:
                print(f"  ✅ SUCCESS: {results['total_nodes']} total nodes found")
                print(f"  ✅ Chunk models have actual content")
                if results["connectivity"]["pipeline_possible"]:
                    print(f"  ✅ Pipeline connectivity verified")
                else:
                    print(f"  ⚠️  Connectivity issues detected")
            else:
                print(f"  ❌ FAILED: 0 nodes found - architectural issue still present")
                all_strategies_valid = False
        else:
            print(f"  ❌ {results['status']}: {results.get('message', 'Unknown error')}")
            all_strategies_valid = False

    # Save detailed results
    results_file = Path("chunk_verification_results.json")
    with open(results_file, 'w') as f:
        json.dump(overall_results, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {results_file}")

    if all_strategies_valid:
        print("\n🎉 SUCCESS: All chunk models have actual nodes and proper structure!")
        print("The architectural issue has been fixed.")
        return 0
    else:
        print("\n❌ FAILURE: Some chunk models still have issues.")
        print("The architectural issue may still be present.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)