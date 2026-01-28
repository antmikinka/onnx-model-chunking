"""
Simple Analytics System Test

This script validates the comprehensive analytics and benchmarking system
without unicode characters for Windows compatibility.
"""

import json
import os
import sys
import time
from datetime import datetime

# Import analytics and configuration systems
try:
    from chunking_analytics import ChunkingAnalytics, SystemMonitor, PerformanceTimer, ModelAnalyzer
    from chunking_config import ConfigurationProcessor, create_sample_configurations
    print("[OK] Analytics modules imported successfully")
except ImportError as e:
    print(f"[ERROR] Failed to import analytics modules: {e}")
    sys.exit(1)


def test_basic_functionality():
    """Test basic analytics functionality"""
    print("\n=== Testing Basic Analytics Functionality ===")

    try:
        # Test system monitor
        print("\n1. Testing System Monitor...")
        monitor = SystemMonitor(sampling_interval=0.5, max_history=10)
        monitor.start_monitoring()
        print("[OK] System monitoring started")

        time.sleep(2)  # Collect samples

        current = monitor.get_current_snapshot()
        if current:
            print(f"[OK] Collected snapshot - Memory: {current.memory_mb:.1f}MB, CPU: {current.cpu_percent:.1f}%")

        monitor.stop_monitoring()
        print("[OK] System monitoring stopped")

        # Test performance timer
        print("\n2. Testing Performance Timer...")
        timer = PerformanceTimer()

        with timer.time_operation("test_operation", "test_phase"):
            time.sleep(0.1)

        summary = timer.get_timing_summary()
        print(f"[OK] Performance timing completed - {len(summary)} operations recorded")

        # Test model analyzer
        print("\n3. Testing Model Analyzer...")
        analyzer = ModelAnalyzer()

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

        print(f"[OK] Model analysis completed - Efficiency: {metric.efficiency_score:.1f}/100")

        return True

    except Exception as e:
        print(f"[ERROR] Basic functionality test failed: {e}")
        return False


def test_analytics_session():
    """Test complete analytics session"""
    print("\n=== Testing Complete Analytics Session ===")

    try:
        # Create analytics system
        analytics = ChunkingAnalytics({
            "enable_monitoring": True,
            "sampling_interval": 0.2,
            "max_history": 50,
            "system_resources": ["cpu", "memory"],
            "detailed_metrics": True,
            "real_time_updates": False,  # Disable for cleaner output
            "output_format": ["json", "summary"],
            "enable_alerts": True
        })

        print("[OK] Analytics system created")

        # Start session
        analytics.start_session()
        print("[OK] Analytics session started")

        # Monitor operations
        with analytics.monitor_operation("test_loading", "loading"):
            time.sleep(0.1)

        with analytics.monitor_operation("test_conversion", "conversion"):
            time.sleep(0.1)

        with analytics.monitor_operation("test_chunking", "chunking"):
            time.sleep(0.1)

        print("[OK] Operations monitored")

        # Add model analysis
        analytics.add_model_analysis(
            model_name="session_test_model",
            original_size_mb=200.0,
            chunk_count=2,
            chunk_sizes_mb=[105.0, 95.0],
            parameter_count=1000000000,
            memory_footprint_mb=190.0,
            conversion_time_ms=8000.0,
            chunking_time_ms=3000.0,
            method="onnx_graph_level"
        )

        print("[OK] Model analysis added")

        # End session
        analytics.end_session()
        print("[OK] Analytics session ended")

        # Generate report
        report = analytics.get_comprehensive_report()
        print(f"[OK] Report generated with {len(report)} sections")

        # Save report
        output_dir = "./test_analytics_output"
        os.makedirs(output_dir, exist_ok=True)

        saved_files = analytics.save_report(output_dir)
        print(f"[OK] Reports saved:")
        for format_type, file_path in saved_files.items():
            print(f"    - {format_type.upper()}: {file_path}")

        return True

    except Exception as e:
        print(f"[ERROR] Analytics session test failed: {e}")
        return False


def test_configuration_integration():
    """Test configuration system integration"""
    print("\n=== Testing Configuration Integration ===")

    try:
        # Get sample configurations
        sample_configs = create_sample_configurations()
        print(f"[OK] Retrieved {len(sample_configs)} sample configurations")

        # Test analytics configuration
        if "analytics_detailed" in sample_configs:
            config_data = sample_configs["analytics_detailed"]
            chunking_config = config_data.get("chunking", {})
            analytics_settings = chunking_config.get("analytics", {})

            if analytics_settings:
                print("[OK] Found analytics configuration:")
                print(f"    - Enable monitoring: {analytics_settings.get('enable_monitoring')}")
                print(f"    - Sampling interval: {analytics_settings.get('sampling_interval')}")
                print(f"    - Output formats: {analytics_settings.get('output_format')}")

                # Test configuration processing
                processor = ConfigurationProcessor()
                config = processor.process_configuration(config_data)

                if config and config.analytics:
                    print("[OK] Configuration processed successfully")
                    print(f"    - Analytics enabled: {config.analytics.enable_monitoring}")
                    return True
                else:
                    print("[WARNING] Configuration processed but analytics not found")
                    return False
            else:
                print("[WARNING] No analytics settings found in sample configuration")
                return False
        else:
            print("[WARNING] Analytics sample configuration not found")
            return False

    except Exception as e:
        print(f"[ERROR] Configuration integration test failed: {e}")
        return False


def test_alert_system():
    """Test alert management system"""
    print("\n=== Testing Alert System ===")

    try:
        from chunking_analytics import AlertManager

        alert_manager = AlertManager()

        # Set thresholds
        alert_manager.set_threshold("memory_percent", 50.0)
        alert_manager.set_threshold("cpu_percent", 70.0)
        print("[OK] Alert thresholds set")

        # Test threshold checking
        test_metrics = {
            "memory_percent": 75.0,  # Above threshold
            "cpu_percent": 80.0,     # Above threshold
            "disk_usage_percent": 60.0  # Below threshold
        }

        alerts = alert_manager.check_thresholds(test_metrics)
        print(f"[OK] Generated {len(alerts)} alerts:")
        for alert in alerts:
            print(f"    - {alert.level.value}: {alert.message}")

        # Get alert summary
        summary = alert_manager.get_alert_summary()
        print(f"[OK] Alert summary: {summary}")

        return True

    except Exception as e:
        print(f"[ERROR] Alert system test failed: {e}")
        return False


def main():
    """Run all analytics tests"""
    print("Simple Analytics System Test")
    print("=" * 50)
    print(f"Start time: {datetime.now().isoformat()}")

    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Analytics Session", test_analytics_session),
        ("Configuration Integration", test_configuration_integration),
        ("Alert System", test_alert_system),
    ]

    results = {}
    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'-' * 20} {test_name} {'-' * 20}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
                print(f"[PASS] {test_name}")
            else:
                print(f"[FAIL] {test_name}")
        except Exception as e:
            print(f"[ERROR] {test_name}: {e}")
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