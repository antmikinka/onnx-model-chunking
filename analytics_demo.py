"""
Comprehensive Analytics System Demo

This script demonstrates the full capabilities of the analytics and benchmarking system
for model chunking, showing how to use it with real-world scenarios.

Usage Examples:
1. Basic analytics usage
2. Advanced configuration
3. Custom monitoring
4. Report generation
5. Integration with chunking system
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List

# Import analytics and configuration systems
from chunking_analytics import ChunkingAnalytics, monitor_chunking_performance
from chunking_config import ConfigurationProcessor, create_sample_configurations


def demo_basic_usage():
    """Demonstrate basic analytics usage"""
    print("=== Demo 1: Basic Analytics Usage ===")

    # Create analytics with default configuration
    analytics = ChunkingAnalytics()
    analytics.start_session()

    print("Analytics session started with default configuration")

    # Monitor some operations
    with analytics.monitor_operation("model_loading", "loading", model="demo_model"):
        time.sleep(0.2)  # Simulate model loading

    with analytics.monitor_operation("model_conversion", "conversion", format="onnx"):
        time.sleep(0.3)  # Simulate conversion

    with analytics.monitor_operation("model_chunking", "chunking", chunks=2):
        time.sleep(0.25)  # Simulate chunking

    # Add model analysis
    analytics.add_model_analysis(
        model_name="demo_model",
        original_size_mb=500.0,
        chunk_count=2,
        chunk_sizes_mb=[260.0, 240.0],
        parameter_count=125000000,
        memory_footprint_mb=480.0,
        conversion_time_ms=12000.0,
        chunking_time_ms=4000.0,
        method="onnx_graph_level"
    )

    # End session and generate report
    analytics.end_session()

    # Save reports
    output_dir = "./analytics_demo_output/basic"
    os.makedirs(output_dir, exist_ok=True)

    saved_files = analytics.save_report(output_dir)
    print(f"Basic demo completed. Reports saved to {output_dir}:")
    for format_type, file_path in saved_files.items():
        print(f"  - {format_type}: {file_path}")


def demo_advanced_configuration():
    """Demonstrate advanced analytics configuration"""
    print("\n=== Demo 2: Advanced Configuration ===")

    # Advanced configuration with custom settings
    config = {
        "enable_monitoring": True,
        "sampling_interval": 0.1,  # High-frequency monitoring
        "max_history": 1000,
        "system_resources": ["cpu", "memory", "disk", "gpu"],
        "timing_phases": ["loading", "preprocessing", "conversion", "chunking", "postprocessing", "saving"],
        "detailed_metrics": True,
        "real_time_updates": True,
        "output_format": ["json", "csv", "summary"],
        "enable_alerts": True,
        "save_snapshots": True,
        "snapshot_interval": 5,
        "performance_alerts": {
            "memory_percent": 70.0,
            "cpu_percent": 85.0,
            "disk_usage_percent": 90.0,
            "conversion_time_seconds": 120.0,
            "memory_usage_gb": 16.0
        }
    }

    analytics = ChunkingAnalytics(config)
    analytics.start_session()

    print("Advanced analytics session started with custom configuration")
    print(f"  - Sampling interval: {config['sampling_interval']}s")
    print(f"  - Monitoring resources: {config['system_resources']}")
    print(f"  - Output formats: {config['output_format']}")
    print(f"  - Alerts enabled: {config['enable_alerts']}")

    # Simulate complex workflow
    phases = [
        ("data_preprocessing", "preprocessing"),
        ("model_optimization", "conversion"),
        ("layer_analysis", "chunking"),
        ("memory_profiling", "chunking"),
        ("quality_validation", "postprocessing"),
        ("final_export", "saving")
    ]

    for phase_name, phase_category in phases:
        with analytics.monitor_operation(phase_name, phase_category):
            time.sleep(0.15)  # Simulate work
            print(f"  Completed phase: {phase_name}")

    # Add multiple model analyses for comparison
    models_data = [
        {"name": "model_a", "size": 400, "chunks": 2, "method": "weight_based"},
        {"name": "model_a", "size": 400, "chunks": 3, "method": "weight_based"},
        {"name": "model_a", "size": 400, "chunks": 2, "method": "layer_based"},
    ]

    for model_data in models_data:
        chunk_sizes = [model_data["size"] / model_data["chunks"]] * model_data["chunks"]
        analytics.add_model_analysis(
            model_name=model_data["name"],
            original_size_mb=model_data["size"],
            chunk_count=model_data["chunks"],
            chunk_sizes_mb=chunk_sizes,
            parameter_count=100000000,
            memory_footprint_mb=model_data["size"] * 0.95,
            conversion_time_ms=8000.0,
            chunking_time_ms=3000.0,
            method=model_data["method"]
        )

    analytics.end_session()

    # Generate comprehensive report
    output_dir = "./analytics_demo_output/advanced"
    os.makedirs(output_dir, exist_ok=True)

    saved_files = analytics.save_report(output_dir, formats=["json", "csv", "summary"])
    print(f"Advanced demo completed. Reports saved to {output_dir}:")

    # Show method comparison
    comparison = analytics.model_analyzer.compare_methods("model_a")
    if comparison:
        print(f"  Method comparison completed:")
        print(f"    - Best method: {comparison['best_method']}")
        if 'comparisons' in comparison:
            for comp_name, comp_data in comparison['comparisons'].items():
                print(f"    - {comp_name}: {comp_data.get('efficiency_improvement_percent', 0):.1f}% efficiency improvement")


def demo_context_manager_usage():
    """Demonstrate context manager usage"""
    print("\n=== Demo 3: Context Manager Usage ===")

    # Use the convenience context manager
    config = {
        "enable_monitoring": True,
        "real_time_updates": True,
        "output_format": ["json", "summary"],
        "performance_alerts": {
            "memory_percent": 80.0,
            "cpu_percent": 90.0
        }
    }

    with monitor_chunking_performance(config) as analytics:
        print("Analytics session started via context manager")

        # Monitor multiple operations
        operations = [
            ("setup_environment", "loading"),
            ("load_model_weights", "loading"),
            ("convert_to_onnx", "conversion"),
            ("analyze_model_structure", "chunking"),
            ("create_chunks", "chunking"),
            ("validate_chunks", "postprocessing"),
            ("save_results", "saving")
        ]

        for op_name, op_phase in operations:
            with analytics.monitor_operation(op_name, op_phase):
                time.sleep(0.1)  # Simulate work
                print(f"  Operation completed: {op_name}")

        # The context manager automatically handles session cleanup
    print("Context manager automatically ended analytics session")

    # Reports are automatically saved by context manager
    print("Context manager demo completed")


def demo_custom_monitoring():
    """Demonstrate custom monitoring scenarios"""
    print("\n=== Demo 4: Custom Monitoring ===")

    analytics = ChunkingAnalytics({
        "enable_monitoring": True,
        "sampling_interval": 0.05,  # Very high frequency
        "detailed_metrics": True,
        "enable_alerts": True,
        "performance_alerts": {
            "memory_percent": 60.0,  # Sensitive thresholds
            "cpu_percent": 75.0
        }
    })

    analytics.start_session()

    print("Custom monitoring with high-frequency sampling and sensitive thresholds")

    # Simulate resource-intensive operations
    for i in range(20):
        with analytics.monitor_operation(f"intensive_task_{i}", "processing", iteration=i):
            # Vary operation duration to simulate different loads
            time.sleep(0.02 + (i % 3) * 0.01)

        if i % 5 == 0:
            print(f"  Completed {i+1}/20 intensive tasks")

    # Monitor system resource patterns
    print("  Monitoring system resource patterns...")

    # Simulate memory usage spike
    with analytics.monitor_operation("memory_spike_simulation", "processing"):
        time.sleep(0.3)

    analytics.end_session()

    # Get detailed statistics
    report = analytics.get_comprehensive_report()

    print(f"Custom monitoring completed:")
    print(f"  - Total operations: {report['session_info']['total_operations']}")
    print(f"  - Session duration: {report['session_info']['duration_seconds']:.2f}s")
    print(f"  - Peak memory: {report['system_resources']['peak_usage'].get('peak_memory_mb', 0):.1f} MB")
    print(f"  - Total alerts: {report['alerts']['total_alerts']}")

    # Save custom monitoring report
    output_dir = "./analytics_demo_output/custom"
    os.makedirs(output_dir, exist_ok=True)
    analytics.save_report(output_dir)
    print(f"Custom monitoring report saved to {output_dir}")


def demo_integration_with_configuration():
    """Demonstrate integration with unified configuration system"""
    print("\n=== Demo 5: Configuration System Integration ===")

    # Get sample configurations
    sample_configs = create_sample_configurations()

    # Test different analytics configurations
    config_names = ["analytics_detailed", "onnx_graph_level_weight_based", "minimal_analytics"]

    for config_name in config_names:
        if config_name not in sample_configs:
            continue

        print(f"\nTesting configuration: {config_name}")

        config_data = sample_configs[config_name]
        processor = ConfigurationProcessor()
        config = processor.process_configuration(config_data)

        if config and config.analytics:
            analytics_config = config.analytics.__dict__
            print(f"  Configuration details:")
            print(f"    - Monitoring enabled: {analytics_config.get('enable_monitoring')}")
            print(f"    - Sampling interval: {analytics_config.get('sampling_interval')}")
            print(f"    - Output formats: {analytics_config.get('output_format')}")
            print(f"    - Real-time updates: {analytics_config.get('real_time_updates')}")

            # Create analytics with this configuration
            analytics = ChunkingAnalytics(analytics_config)
            analytics.start_session()

            # Run sample operations
            with analytics.monitor_operation("config_test", "testing"):
                time.sleep(0.1)

            analytics.end_session()

            # Save with configuration-specific naming
            output_dir = f"./analytics_demo_output/config_{config_name}"
            os.makedirs(output_dir, exist_ok=True)
            analytics.save_report(output_dir)

            print(f"  Configuration test completed successfully")
        else:
            print(f"  No analytics configuration found in {config_name}")


def demo_alert_system():
    """Demonstrate alert system functionality"""
    print("\n=== Demo 6: Alert System Demo ===")

    # Create analytics with sensitive alert thresholds
    analytics = ChunkingAnalytics({
        "enable_monitoring": True,
        "enable_alerts": True,
        "real_time_updates": True,
        "performance_alerts": {
            "memory_percent": 50.0,  # Very sensitive
            "cpu_percent": 60.0,
            "conversion_time_seconds": 5.0
        }
    })

    analytics.start_session()

    print("Alert system demo with sensitive thresholds")
    print("  - Memory alert threshold: 50%")
    print("  - CPU alert threshold: 60%")
    print("  - Time alert threshold: 5s")

    # Simulate operations that might trigger alerts
    with analytics.monitor_operation("memory_intensive_task", "processing"):
        time.sleep(0.2)  # Might trigger time alert

    with analytics.monitor_operation("cpu_intensive_task", "processing"):
        time.sleep(0.15)  # Might trigger CPU alert

    analytics.end_session()

    # Check alerts generated
    report = analytics.get_comprehensive_report()
    alert_summary = report['alerts']
    total_alerts = alert_summary['total_alerts']
    recent_alerts = alert_summary['recent_alerts']

    print(f"Alert system demo completed:")
    print(f"  - Total alerts generated: {total_alerts}")
    print(f"  - Recent alerts:")
    for alert in recent_alerts:
        print(f"    - {alert['level']}: {alert['message']}")

    # Save alert report
    output_dir = "./analytics_demo_output/alerts"
    os.makedirs(output_dir, exist_ok=True)
    analytics.save_report(output_dir)


def main():
    """Run all analytics demonstrations"""
    print("Comprehensive Analytics System Demonstration")
    print("=" * 60)
    print(f"Start time: {datetime.now().isoformat()}")

    # Create main output directory
    main_output_dir = "./analytics_demo_output"
    os.makedirs(main_output_dir, exist_ok=True)

    try:
        # Run all demonstrations
        demo_basic_usage()
        demo_advanced_configuration()
        demo_context_manager_usage()
        demo_custom_monitoring()
        demo_integration_with_configuration()
        demo_alert_system()

        print(f"\n{'=' * 60}")
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print(f"{'=' * 60}")

        # Summary of generated files
        print(f"\nGenerated files in {main_output_dir}:")
        for root, dirs, files in os.walk(main_output_dir):
            level = root.replace(main_output_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")

        print(f"\nDemo completed successfully!")
        print(f"Check the generated files to see comprehensive analytics reports.")

        return 0

    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())