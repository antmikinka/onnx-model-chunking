"""
Comprehensive Qwen2-0.5B Benchmark with Full Analytics

This script demonstrates the complete analytics and benchmarking system
by converting a Qwen2-0.5B model with detailed performance monitoring,
system resource tracking, and benefit analysis.

Features:
- Real-time system resource monitoring
- Detailed performance timing analysis
- Model-specific analytics and comparisons
- Professional report generation
- Both chunking methods comparison
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Import analytics and chunking systems
from chunking_analytics import ChunkingAnalytics, monitor_chunking_performance
from chunking_config import ConfigurationProcessor, create_sample_configurations
import builder


def create_benchmark_configurations() -> Dict[str, Dict[str, Any]]:
    """
    Create benchmark configurations for comprehensive testing.

    Returns:
        Dictionary of configuration names to their configurations
    """
    base_config = {
        "cache_dir": "./model_cache",
        "output_folder": "./qwen_benchmark_output",
        "model_path": "Qwen/Qwen2-0.5B-Instruct",
        "filename": "qwen2-0.5b-instruct.onnx",
        "precision": "fp32",
        "conv_type": "q4_g128",
        "io_dtype": "fp32",
        "extra_options": {
            "num_hidden_layers": 24,  # Qwen2-0.5B has 24 layers
            "use_external_data_format": True,
            "verbose": 1
        }
    }

    configurations = {
        "baseline_no_chunking": {
            **base_config,
            "extra_options": {
                **base_config["extra_options"],
                # No chunking configuration - baseline
            }
        },

        "onnx_graph_level_2_chunks": {
            **base_config,
            "extra_options": {
                **base_config["extra_options"],
                "chunking": {
                    "method": "onnx_graph_level",
                    "onnx_graph_level": {
                        "strategy": "weight_based",
                        "distribution": "equal_memory",
                        "num_chunks": 2,
                        "respect_layer_boundaries": True
                    },
                    "common": {
                        "output_directory": "./qwen_benchmark_output/onnx_graph_2_chunks",
                        "verify_chunks": True,
                        "verbose": True
                    },
                    "analytics": {
                        "enable_monitoring": True,
                        "sampling_interval": 0.5,
                        "max_history": 7200,
                        "system_resources": ["cpu", "memory", "disk", "gpu"],
                        "timing_phases": ["loading", "conversion", "chunking", "saving"],
                        "detailed_metrics": True,
                        "real_time_updates": True,
                        "output_format": ["json", "csv", "summary"],
                        "enable_alerts": True,
                        "performance_alerts": {
                            "memory_percent": 80.0,
                            "cpu_percent": 90.0,
                            "conversion_time_seconds": 600.0,
                            "memory_usage_gb": 12.0
                        }
                    }
                }
            }
        },

        "onnx_graph_level_3_chunks": {
            **base_config,
            "extra_options": {
                **base_config["extra_options"],
                "chunking": {
                    "method": "onnx_graph_level",
                    "onnx_graph_level": {
                        "strategy": "weight_based",
                        "distribution": "equal_memory",
                        "num_chunks": 3,
                        "respect_layer_boundaries": True
                    },
                    "common": {
                        "output_directory": "./qwen_benchmark_output/onnx_graph_3_chunks",
                        "verify_chunks": True,
                        "verbose": True
                    },
                    "analytics": {
                        "enable_monitoring": True,
                        "sampling_interval": 0.5,
                        "max_history": 7200,
                        "system_resources": ["cpu", "memory", "disk", "gpu"],
                        "timing_phases": ["loading", "conversion", "chunking", "saving"],
                        "detailed_metrics": True,
                        "real_time_updates": True,
                        "output_format": ["json", "csv", "summary"],
                        "enable_alerts": True,
                        "performance_alerts": {
                            "memory_percent": 80.0,
                            "cpu_percent": 90.0,
                            "conversion_time_seconds": 600.0,
                            "memory_usage_gb": 12.0
                        }
                    }
                }
            }
        },

        "layer_based_2_chunks": {
            **base_config,
            "extra_options": {
                **base_config["extra_options"],
                "chunking": {
                    "method": "onnx_graph_level",
                    "onnx_graph_level": {
                        "strategy": "layer_based",
                        "distribution": "equal_layers",
                        "num_chunks": 2,
                        "respect_layer_boundaries": True
                    },
                    "common": {
                        "output_directory": "./qwen_benchmark_output/layer_based_2_chunks",
                        "verify_chunks": True,
                        "verbose": True
                    },
                    "analytics": {
                        "enable_monitoring": True,
                        "sampling_interval": 0.5,
                        "max_history": 7200,
                        "system_resources": ["cpu", "memory", "disk", "gpu"],
                        "timing_phases": ["loading", "conversion", "chunking", "saving"],
                        "detailed_metrics": True,
                        "real_time_updates": True,
                        "output_format": ["json", "csv", "summary"],
                        "enable_alerts": True,
                        "performance_alerts": {
                            "memory_percent": 80.0,
                            "cpu_percent": 90.0,
                            "conversion_time_seconds": 600.0,
                            "memory_usage_gb": 12.0
                        }
                    }
                }
            }
        },

        "high_performance_monitoring": {
            **base_config,
            "extra_options": {
                **base_config["extra_options"],
                "chunking": {
                    "method": "onnx_graph_level",
                    "onnx_graph_level": {
                        "strategy": "weight_based",
                        "distribution": "equal_memory",
                        "num_chunks": 2,
                        "respect_layer_boundaries": True
                    },
                    "common": {
                        "output_directory": "./qwen_benchmark_output/high_performance",
                        "verify_chunks": True,
                        "verbose": True
                    },
                    "analytics": {
                        "enable_monitoring": True,
                        "sampling_interval": 0.1,  # High frequency monitoring
                        "max_history": 10000,      # Extended history
                        "system_resources": ["cpu", "memory", "disk", "gpu"],
                        "timing_phases": ["loading", "conversion", "chunking", "saving"],
                        "detailed_metrics": True,
                        "real_time_updates": True,
                        "output_format": ["json", "csv", "summary"],
                        "enable_alerts": True,
                        "save_snapshots": True,
                        "snapshot_interval": 5,
                        "performance_alerts": {
                            "memory_percent": 75.0,  # More sensitive thresholds
                            "cpu_percent": 85.0,
                            "disk_usage_percent": 80.0,
                            "conversion_time_seconds": 300.0,
                            "memory_usage_gb": 10.0
                        }
                    }
                }
            }
        }
    }

    return configurations


def run_single_benchmark(config_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single benchmark configuration with comprehensive analytics.

    Args:
        config_name: Name of the benchmark configuration
        config: Configuration dictionary

    Returns:
        Benchmark results dictionary
    """
    print(f"\n{'='*80}")
    print(f"STARTING BENCHMARK: {config_name}")
    print(f"{'='*80}")
    print(f"Model: {config['model_path']}")
    print(f"Output: {config['output_folder']}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Initialize analytics for the entire benchmark
    benchmark_analytics = ChunkingAnalytics({
        "enable_monitoring": True,
        "sampling_interval": 1.0,
        "max_history": 5000,
        "system_resources": ["cpu", "memory", "disk", "gpu"],
        "detailed_metrics": True,
        "real_time_updates": True,
        "output_format": ["json", "summary"],
        "enable_alerts": True
    })

    benchmark_results = {
        "config_name": config_name,
        "start_time": datetime.now().isoformat(),
        "success": False,
        "error": None,
        "metrics": {}
    }

    try:
        # Start benchmark monitoring
        benchmark_analytics.start_session()

        # Model loading phase
        print(f"\n--- Phase 1: Model Loading ---")
        with benchmark_analytics.monitor_operation("model_loading", "loading", model=config['model_path']):
            try:
                # Create model instance
                model = builder.create_model(
                    config['model_path'],
                    config['precision'],
                    config['io_dtype'],
                    config.get('ep', 'cpu'),
                    config['cache_dir'],
                    config['extra_options']
                )
                print(f"✓ Model loaded successfully")
                print(f"  - Model type: {type(model).__name__}")
                print(f"  - Hidden size: {model.hidden_size}")
                print(f"  - Num layers: {model.num_layers}")
                print(f"  - Num attention heads: {model.num_attn_heads}")
                print(f"  - Vocabulary size: {model.vocab_size}")

            except Exception as e:
                raise RuntimeError(f"Model loading failed: {e}")

        # Model conversion phase
        print(f"\n--- Phase 2: Model Conversion ---")
        with benchmark_analytics.monitor_operation("model_conversion", "conversion",
                                                 precision=config['precision'],
                                                 conv_type=config.get('conv_type')):
            try:
                # Convert to ONNX
                onnx_model_path = os.path.join(config['output_folder'], config['filename'])
                os.makedirs(config['output_folder'], exist_ok=True)

                # Create sample input for verification
                sample_input = model._create_sample_input_for_verification()

                # Convert model
                model.save_model(onnx_model_path)
                print(f"✓ Model converted to ONNX: {onnx_model_path}")

                # Check file size
                if os.path.exists(onnx_model_path):
                    model_size_mb = os.path.getsize(onnx_model_path) / (1024 * 1024)
                    print(f"  - Model size: {model_size_mb:.2f} MB")
                    benchmark_results["metrics"]["model_size_mb"] = model_size_mb

            except Exception as e:
                raise RuntimeError(f"Model conversion failed: {e}")

        # Chunking phase (if enabled)
        chunking_enabled = "chunking" in config.get("extra_options", {})
        if chunking_enabled:
            print(f"\n--- Phase 3: Model Chunking ---")
            with benchmark_analytics.monitor_operation("model_chunking", "chunking"):
                try:
                    # Get chunking configuration
                    chunking_info = model.get_chunking_info()
                    print(f"Chunking configuration: {chunking_info}")

                    # Create chunked models
                    output_dir = chunking_info.get('output_directory',
                                                 os.path.join(config['output_folder'], 'chunked_models'))

                    chunk_models = model.create_chunked_models(
                        output_dir=output_dir,
                        verify_chunks=config['extra_options']['chunking']['common'].get('verify_chunks', False),
                        sample_input=sample_input if chunking_info else None
                    )

                    if chunk_models:
                        print(f"✓ Model chunking completed successfully")
                        if isinstance(chunk_models, tuple) and len(chunk_models) >= 2:
                            # 2-chunk scenario
                            first_chunk, second_chunk, boundary_tensors = chunk_models[:3]
                            print(f"  - Created 2 chunks")
                            if hasattr(first_chunk, 'graph') and hasattr(second_chunk, 'graph'):
                                print(f"  - First chunk: {len(first_chunk.graph)} nodes")
                                print(f"  - Second chunk: {len(second_chunk.graph)} nodes")
                        elif isinstance(chunk_models, list) and len(chunk_models) > 0:
                            # Multi-chunk scenario
                            print(f"  - Created {len(chunk_models)} chunks")
                            for i, chunk in enumerate(chunk_models):
                                if hasattr(chunk, 'graph'):
                                    print(f"  - Chunk {i}: {len(chunk.graph)} nodes")

                        # Collect chunk file sizes
                        chunk_sizes = []
                        chunk_dir = output_dir
                        if os.path.exists(chunk_dir):
                            for file in os.listdir(chunk_dir):
                                if file.endswith('.onnx'):
                                    file_path = os.path.join(chunk_dir, file)
                                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                                    chunk_sizes.append(size_mb)
                                    print(f"  - {file}: {size_mb:.2f} MB")

                        benchmark_results["metrics"]["chunk_sizes_mb"] = chunk_sizes
                        benchmark_results["metrics"]["num_chunks"] = len(chunk_sizes)

                except Exception as e:
                    raise RuntimeError(f"Model chunking failed: {e}")

        # Performance validation phase
        print(f"\n--- Phase 4: Performance Validation ---")
        try:
            # Validate memory usage
            current_snapshot = benchmark_analytics.system_monitor.get_current_snapshot()
            if current_snapshot:
                print(f"Current system resources:")
                print(f"  - Memory usage: {current_snapshot.memory_mb:.1f} MB ({current_snapshot.memory_percent:.1f}%)")
                print(f"  - CPU usage: {current_snapshot.cpu_percent:.1f}%")
                if current_snapshot.gpu_info:
                    for i, gpu in enumerate(current_snapshot.gpu_info):
                        print(f"  - GPU {i}: {gpu['name']} ({gpu['memory_used_mb']:.1f} MB used)")

            # Get peak usage statistics
            peak_usage = benchmark_analytics.system_monitor.get_peak_usage()
            avg_usage = benchmark_analytics.system_monitor.get_average_usage()
            benchmark_results["metrics"]["peak_resources"] = peak_usage
            benchmark_results["metrics"]["avg_resources"] = avg_usage

            print(f"Peak resource usage:")
            print(f"  - Peak memory: {peak_usage.get('peak_memory_mb', 0):.1f} MB")
            print(f"  - Peak CPU: {peak_usage.get('peak_cpu_percent', 0):.1f}%")

        except Exception as e:
            print(f"Warning: Performance validation failed: {e}")

        # Mark benchmark as successful
        benchmark_results["success"] = True
        benchmark_results["end_time"] = datetime.now().isoformat()

        print(f"\n✓ BENCHMARK COMPLETED SUCCESSFULLY: {config_name}")

    except Exception as e:
        benchmark_results["error"] = str(e)
        benchmark_results["end_time"] = datetime.now().isoformat()
        print(f"\n✗ BENCHMARK FAILED: {config_name}")
        print(f"Error: {e}")

    finally:
        # End analytics session and generate report
        benchmark_analytics.end_session()

        # Generate comprehensive report
        print(f"\n--- Generating Benchmark Report ---")
        try:
            report = benchmark_analytics.get_comprehensive_report()
            benchmark_results["analytics_report"] = report

            # Save benchmark results
            benchmark_dir = os.path.join(config['output_folder'], 'benchmark_results')
            os.makedirs(benchmark_dir, exist_ok=True)

            # Save benchmark summary
            benchmark_file = os.path.join(benchmark_dir, f"benchmark_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(benchmark_file, 'w') as f:
                json.dump(benchmark_results, f, indent=2, default=str)

            # Save analytics reports
            analytics_files = benchmark_analytics.save_report(benchmark_dir)
            print(f"Analytics reports saved:")
            for format_type, file_path in analytics_files.items():
                print(f"  - {format_type.upper()}: {file_path}")

            print(f"  - Benchmark summary: {benchmark_file}")

        except Exception as e:
            print(f"Warning: Failed to save benchmark report: {e}")

    return benchmark_results


def run_comparative_benchmark(configurations: Dict[str, Dict[str, Any]],
                            selected_configs: List[str] = None) -> Dict[str, Any]:
    """
    Run comparative benchmark across multiple configurations.

    Args:
        configurations: Dictionary of all available configurations
        selected_configs: List of configuration names to run (None = all)

    Returns:
        Comparative benchmark results
    """
    if selected_configs is None:
        selected_configs = list(configurations.keys())

    print(f"\n{'='*100}")
    print(f"COMPREHENSIVE QWEN2-0.5B BENCHMARK SUITE")
    print(f"{'='*100}")
    print(f"Selected configurations: {', '.join(selected_configs)}")
    print(f"Total configurations: {len(selected_configs)}")
    print(f"Start time: {datetime.now().isoformat()}")

    comparative_results = {
        "benchmark_suite": "qwen2_0.5b_comprehensive",
        "start_time": datetime.now().isoformat(),
        "configurations_tested": selected_configs,
        "individual_results": {},
        "comparative_analysis": {},
        "recommendations": []
    }

    # Run each benchmark
    for config_name in selected_configs:
        if config_name not in configurations:
            print(f"Warning: Configuration '{config_name}' not found, skipping")
            continue

        try:
            result = run_single_benchmark(config_name, configurations[config_name])
            comparative_results["individual_results"][config_name] = result

        except Exception as e:
            print(f"Critical error running benchmark '{config_name}': {e}")
            comparative_results["individual_results"][config_name] = {
                "config_name": config_name,
                "success": False,
                "error": str(e),
                "start_time": datetime.now().isoformat()
            }

    # Generate comparative analysis
    print(f"\n{'='*80}")
    print(f"COMPARATIVE ANALYSIS")
    print(f"{'='*80}")

    successful_results = {name: result for name, result in comparative_results["individual_results"].items()
                        if result.get("success", False)}

    if successful_results:
        # Compare model sizes
        model_sizes = {name: result.get("metrics", {}).get("model_size_mb", 0)
                      for name, result in successful_results.items()}
        if model_sizes:
            print(f"\nModel Size Comparison (MB):")
            for name, size in sorted(model_sizes.items(), key=lambda x: x[1], reverse=True):
                print(f"  {name}: {size:.2f} MB")

        # Compare chunking performance
        chunking_results = {name: result for name, result in successful_results.items()
                          if "chunk_sizes_mb" in result.get("metrics", {})}
        if chunking_results:
            print(f"\nChunking Performance:")
            for name, result in chunking_results.items():
                metrics = result["metrics"]
                num_chunks = metrics.get("num_chunks", 0)
                chunk_sizes = metrics.get("chunk_sizes_mb", [])
                total_chunk_size = sum(chunk_sizes)
                compression_ratio = metrics.get("model_size_mb", 0) / total_chunk_size if total_chunk_size > 0 else 0

                print(f"  {name}:")
                print(f"    - Chunks: {num_chunks}")
                print(f"    - Total size: {total_chunk_size:.2f} MB")
                print(f"    - Compression ratio: {compression_ratio:.3f}")
                if len(chunk_sizes) > 1:
                    print(f"    - Chunk sizes: {[f'{s:.2f}' for s in chunk_sizes]} MB")

        # Compare resource usage
        resource_comparison = {}
        for name, result in successful_results.items():
            peak_resources = result.get("metrics", {}).get("peak_resources", {})
            if peak_resources:
                resource_comparison[name] = {
                    "peak_memory_mb": peak_resources.get("peak_memory_mb", 0),
                    "peak_cpu_percent": peak_resources.get("peak_cpu_percent", 0)
                }

        if resource_comparison:
            print(f"\nPeak Resource Usage:")
            for name, resources in resource_comparison.items():
                print(f"  {name}:")
                print(f"    - Peak memory: {resources['peak_memory_mb']:.1f} MB")
                print(f"    - Peak CPU: {resources['peak_cpu_percent']:.1f}%")

        # Generate recommendations
        print(f"\nRECOMMENDATIONS:")

        # Most memory efficient
        if resource_comparison:
            most_memory_efficient = min(resource_comparison.items(),
                                      key=lambda x: x[1]["peak_memory_mb"])
            print(f"1. Most memory efficient: {most_memory_efficient[0]} "
                  f"({most_memory_efficient[1]['peak_memory_mb']:.1f} MB peak)")

        # Fastest chunking (if available)
        chunking_times = {}
        for name, result in successful_results.items():
            analytics_report = result.get("analytics_report", {})
            timing_summary = analytics_report.get("performance_timing", {}).get("summary", {})
            chunking_time = 0
            for key, value in timing_summary.items():
                if "chunking" in key.lower():
                    chunking_time += value.get("total_ms", 0)
            if chunking_time > 0:
                chunking_times[name] = chunking_time

        if chunking_times:
            fastest_chunking = min(chunking_times.items(), key=lambda x: x[1])
            print(f"2. Fastest chunking: {fastest_chunking[0]} "
                  f"({fastest_chunking[1]:.1f} ms total)")

        # Best compression ratio
        if chunking_results:
            best_compression = None
            best_ratio = 0
            for name, result in chunking_results.items():
                metrics = result["metrics"]
                model_size = metrics.get("model_size_mb", 0)
                chunk_sizes = metrics.get("chunk_sizes_mb", [])
                total_chunk_size = sum(chunk_sizes)
                if total_chunk_size > 0:
                    ratio = model_size / total_chunk_size
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_compression = name

            if best_compression:
                print(f"3. Best compression: {best_compression} (ratio: {best_ratio:.3f})")

    else:
        print("No successful benchmarks to compare")

    comparative_results["end_time"] = datetime.now().isoformat()

    # Save comparative results
    try:
        output_dir = "./qwen_benchmark_output/comparative_results"
        os.makedirs(output_dir, exist_ok=True)

        comparative_file = os.path.join(output_dir,
                                      f"comparative_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(comparative_file, 'w') as f:
            json.dump(comparative_results, f, indent=2, default=str)

        print(f"\nComparative benchmark results saved: {comparative_file}")

    except Exception as e:
        print(f"Warning: Failed to save comparative results: {e}")

    return comparative_results


def main():
    """Main function to run the comprehensive benchmark suite"""
    parser = argparse.ArgumentParser(description="Comprehensive Qwen2-0.5B Benchmark with Analytics")
    parser.add_argument("--configs", nargs="+",
                       choices=["baseline_no_chunking", "onnx_graph_level_2_chunks",
                               "onnx_graph_level_3_chunks", "layer_based_2_chunks",
                               "high_performance_monitoring"],
                       help="Specific configurations to run (default: all chunking configs)")
    parser.add_argument("--output-dir", default="./qwen_benchmark_output",
                       help="Output directory for benchmark results")
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct",
                       help="Model to benchmark")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")

    args = parser.parse_args()

    # Create configurations
    configurations = create_benchmark_configurations()

    # Update base configurations with command line arguments
    for config in configurations.values():
        config["output_folder"] = args.output_dir
        config["model_path"] = args.model

    # Select configurations to run
    selected_configs = args.configs if args.configs else [
        "onnx_graph_level_2_chunks",
        "onnx_graph_level_3_chunks",
        "layer_based_2_chunks"
    ]

    print(f"Qwen2-0.5B Comprehensive Benchmark Suite")
    print(f"========================================")
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Configurations: {', '.join(selected_configs)}")
    print(f"Verbose mode: {args.verbose}")

    # Run comparative benchmark
    try:
        results = run_comparative_benchmark(configurations, selected_configs)

        print(f"\n{'='*100}")
        print(f"BENCHMARK SUITE COMPLETED")
        print(f"{'='*100}")

        successful_count = sum(1 for result in results["individual_results"].values()
                            if result.get("success", False))
        total_count = len(results["individual_results"])

        print(f"Successful benchmarks: {successful_count}/{total_count}")
        print(f"Results directory: {args.output_dir}")
        print(f"End time: {datetime.now().isoformat()}")

        return 0

    except KeyboardInterrupt:
        print("\nBenchmark suite interrupted by user")
        return 1
    except Exception as e:
        print(f"\nCritical error in benchmark suite: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())