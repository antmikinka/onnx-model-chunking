"""
Test Script for Analytics System

This script validates the comprehensive analytics and benchmarking system
without requiring full model downloads or conversions.
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any

# Import analytics and configuration systems
from chunking_analytics import ChunkingAnalytics, SystemMonitor, PerformanceTimer, ModelAnalyzer, AlertManager
from chunking_config import ConfigurationProcessor, create_sample_configurations


def test_system_monitor():
    """Test system resource monitoring functionality"""
    print("Testing System Monitor...")
    print("-" * 40)

    try:
        # Create system monitor
        monitor = SystemMonitor(sampling_interval=0.5, max_history=20)

        # Start monitoring
        monitor.start_monitoring()
        print("[OK] System monitoring started")

        # Collect some samples
        print("Collecting samples for 3 seconds...")
        time.sleep(3)

        # Get current snapshot
        current = monitor.get_current_snapshot()
        if current:
            print(f"[OK] Current snapshot collected:")
            print(f"  - CPU: {current.cpu_percent:.1f}%")
            print(f"  - Memory: {current.memory_mb:.1f} MB ({current.memory_percent:.1f}%)")
            print(f"  - Disk usage: {current.disk_usage_percent:.1f}%")
            if current.gpu_info:
                for i, gpu in enumerate(current.gpu_info):
                    print(f"  - GPU {i}: {gpu['load_percent']:.1f}% load, {gpu['memory_used_mb']:.1f} MB memory")
        else:
            print("⚠ No snapshot collected")

        # Get statistics
        peak_usage = monitor.get_peak_usage()
        avg_usage = monitor.get_average_usage()
        print(f"✓ Peak usage: {peak_usage}")
        print(f"✓ Average usage: {avg_usage}")

        # Stop monitoring
        monitor.stop_monitoring()
        print("✓ System monitoring stopped")

        return True

    except Exception as e:
        print(f"✗ System monitor test failed: {e}")
        return False


def test_performance_timer():
    """Test performance timing functionality"""
    print("\nTesting Performance Timer...")
    print("-" * 40)

    try:
        timer = PerformanceTimer()

        # Test context manager
        with timer.time_operation("test_operation", "test_phase", test_param="value"):
            time.sleep(0.1)  # Simulate work

        print("✓ Context manager timing completed")

        # Test manual timing
        timer_id = timer.start_timer("manual_operation", "manual_phase")
        time.sleep(0.05)
        duration = timer.end_timer(timer_id)
        print(f"✓ Manual timing completed: {duration:.1f} ms")

        # Get summaries
        summary = timer.get_timing_summary()
        breakdown = timer.get_phase_breakdown()
        print(f"✓ Timing summary: {len(summary)} operations")
        print(f"✓ Phase breakdown: {breakdown}")

        return True

    except Exception as e:
        print(f"✗ Performance timer test failed: {e}")
        return False


def test_model_analyzer():
    """Test model analysis functionality"""
    print("\nTesting Model Analyzer...")
    print("-" * 40)

    try:
        analyzer = ModelAnalyzer()

        # Add sample model analysis
        metric = analyzer.analyze_model(
            model_name="test_model",
            original_size_mb=100.0,
            chunk_count=2,
            chunk_sizes_mb=[52.0, 48.0],
            parameter_count=500000000,
            memory_footprint_mb=95.0,
            conversion_time_ms=5000.0,
            chunking_time_ms=2000.0,
            method="onnx_graph_level"
        )

        print(f"✓ Model analysis completed:")
        print(f"  - Model: {metric.model_name}")
        print(f"  - Efficiency score: {metric.efficiency_score:.1f}/100")
        print(f"  - Compression ratio: {metric.compression_ratio:.3f}")

        # Test another model for comparison
        analyzer.analyze_model(
            model_name="test_model",
            original_size_mb=100.0,
            chunk_count=3,
            chunk_sizes_mb=[35.0, 33.0, 32.0],
            parameter_count=500000000,
            memory_footprint_mb=98.0,
            conversion_time_ms=4500.0,
            chunking_time_ms=1800.0,
            method="onnx_graph_level"
        )

        # Test comparison
        comparison = analyzer.compare_methods("test_model")
        if comparison:
            print(f"✓ Method comparison completed:")
            print(f"  - Best method: {comparison['best_method']}")
            if 'comparisons' in comparison and comparison['comparisons']:
                for comp_name, comp_data in comparison['comparisons'].items():
                    print(f"  - {comp_name}: {comp_data}")

        return True

    except Exception as e:
        print(f"✗ Model analyzer test failed: {e}")
        return False


def test_alert_manager():
    """Test alert management functionality"""
    print("\nTesting Alert Manager...")
    print("-" * 40)

    try:
        alert_manager = AlertManager()

        # Set custom thresholds
        alert_manager.set_threshold("memory_percent", 50.0)
        alert_manager.set_threshold("cpu_percent", 70.0)
        print("✓ Custom thresholds set")

        # Test threshold checking (should generate alerts)
        test_metrics = {
            "memory_percent": 75.0,  # Above threshold
            "cpu_percent": 80.0,     # Above threshold
            "disk_usage_percent": 60.0  # Below threshold
        }

        alerts = alert_manager.check_thresholds(test_metrics)
        print(f"✓ Generated {len(alerts)} alerts:")
        for alert in alerts:
            print(f"  - {alert.level.value}: {alert.message}")

        # Get alert summary
        summary = alert_manager.get_alert_summary()
        print(f"✓ Alert summary: {summary}")

        return True

    except Exception as e:
        print(f"✗ Alert manager test failed: {e}")
        return False


def test_full_analytics_session():
    """Test complete analytics session integration"""
    print("\nTesting Full Analytics Session...")
    print("-" * 40)

    try:
        # Create analytics system
        analytics = ChunkingAnalytics({
            "enable_monitoring": True,
            "sampling_interval": 0.2,
            "max_history": 50,
            "system_resources": ["cpu", "memory"],
            "detailed_metrics": True,
            "real_time_updates": True,
            "output_format": ["json", "summary"],
            "enable_alerts": True,
            "performance_alerts": {
                "memory_percent": 60.0,
                "cpu_percent": 80.0
            }
        })

        print("✓ Analytics system created")

        # Start session
        analytics.start_session()
        print("✓ Analytics session started")

        # Monitor some operations
        with analytics.monitor_operation("test_loading", "loading", model="test"):
            time.sleep(0.1)

        with analytics.monitor_operation("test_conversion", "conversion", precision="fp32"):
            time.sleep(0.15)

        with analytics.monitor_operation("test_chunking", "chunking", num_chunks=2):
            time.sleep(0.2)

        print("✓ Operations monitored")

        # Add model analysis
        analytics.add_model_analysis(
            model_name="integration_test_model",
            original_size_mb=200.0,
            chunk_count=2,
            chunk_sizes_mb=[105.0, 95.0],
            parameter_count=1000000000,
            memory_footprint_mb=190.0,
            conversion_time_ms=8000.0,
            chunking_time_ms=3000.0,
            method="onnx_graph_level"
        )

        print("✓ Model analysis added")

        # End session
        analytics.end_session()
        print("✓ Analytics session ended")

        # Generate report
        report = analytics.get_comprehensive_report()
        print(f"✓ Report generated with {len(report)} sections")

        # Save report
        output_dir = "./test_analytics_output"
        os.makedirs(output_dir, exist_ok=True)

        saved_files = analytics.save_report(output_dir)
        print(f"✓ Reports saved:")
        for format_type, file_path in saved_files.items():
            print(f"  - {format_type.upper()}: {file_path}")

        return True

    except Exception as e:
        print(f"✗ Full analytics session test failed: {e}")
        return False


def test_configuration_integration():
    """Test configuration system integration with analytics"""
    print("\nTesting Configuration Integration...")
    print("-" * 40)

    try:
        # Get sample configurations
        sample_configs = create_sample_configurations()
        print(f"✓ Retrieved {len(sample_configs)} sample configurations")

        # Test analytics configuration
        analytics_config = sample_configs.get("analytics_detailed", {})
        if analytics_config:
            chunking_config = analytics_config.get("chunking", {})
            analytics_settings = chunking_config.get("analytics", {})

            if analytics_settings:
                print("✓ Found analytics configuration:")
                print(f"  - Enable monitoring: {analytics_settings.get('enable_monitoring')}")
                print(f"  - Sampling interval: {analytics_settings.get('sampling_interval')}")
                print(f"  - Output formats: {analytics_settings.get('output_format')}")

                # Test configuration processing
                processor = ConfigurationProcessor()
                config = processor.process_configuration(analytics_config)

                if config and config.analytics:
                    print("✓ Configuration processed successfully")
                    print(f"  - Analytics enabled: {config.analytics.enable_monitoring}")
                    print(f"  - Real-time updates: {config.analytics.real_time_updates}")
                    return True
                else:
                    print("⚠ Configuration processed but analytics not found")
                    return False
            else:
                print("⚠ No analytics settings found in sample configuration")
                return False
        else:
            print("⚠ Analytics sample configuration not found")
            return False

    except Exception as e:
        print(f"✗ Configuration integration test failed: {e}")
        return False


def test_performance_benchmarks():
    """Test performance benchmarking capabilities"""
    print("\nTesting Performance Benchmarks...")
    print("-" * 40)

    try:
        # Create high-performance analytics configuration
        analytics = ChunkingAnalytics({
            "enable_monitoring": True,
            "sampling_interval": 0.05,  # Very high frequency
            "max_history": 100,
            "system_resources": ["cpu", "memory", "disk"],
            "detailed_metrics": True,
            "real_time_updates": True,
            "output_format": ["json"],
            "enable_alerts": True
        })

        analytics.start_session()

        # Simulate high-frequency operations
        for i in range(10):
            with analytics.monitor_operation(f"operation_{i}", "benchmark", iteration=i):
                time.sleep(0.01)  # Very short operation

        analytics.end_session()

        # Check metrics
        report = analytics.get_comprehensive_report()
        session_info = report.get("session_info", {})
        timing_info = report.get("performance_timing", {})

        print(f"✓ High-frequency benchmark completed:")
        print(f"  - Total operations: {session_info.get('total_operations', 0)}")
        print(f"  - Session duration: {session_info.get('duration_seconds', 0):.2f} seconds")
        print(f"  - Timing metrics collected: {len(timing_info.get('summary', {}))}")

        return True

    except Exception as e:
        print(f"✗ Performance benchmarks test failed: {e}")
        return False


def main():
    """Run all analytics system tests"""
    print("Analytics System Test Suite")
    print("=" * 50)
    print(f"Start time: {datetime.now().isoformat()}")

    tests = [
        ("System Monitor", test_system_monitor),
        ("Performance Timer", test_performance_timer),
        ("Model Analyzer", test_model_analyzer),
        ("Alert Manager", test_alert_manager),
        ("Configuration Integration", test_configuration_integration),
        ("Full Analytics Session", test_full_analytics_session),
        ("Performance Benchmarks", test_performance_benchmarks),
    ]

    results = {}
    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
            results[test_name] = False

    # Summary
    print(f"\n{'=' * 50}")
    print("TEST SUMMARY")
    print(f"{'=' * 50}")
    print(f"Passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    print(f"End time: {datetime.now().isoformat()}")

    # Detailed results
    print(f"\nDetailed Results:")
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")

    # Check if output files were created
    test_output_dir = "./test_analytics_output"
    if os.path.exists(test_output_dir):
        files = os.listdir(test_output_dir)
        print(f"\nGenerated test files ({len(files)}):")
        for file in files:
            print(f"  - {file}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())