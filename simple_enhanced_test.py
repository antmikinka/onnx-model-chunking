"""
Simple Test for Enhanced Analytics Reporting System

Basic validation of the enhanced analytics functionality.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

from chunking_analytics import ChunkingAnalytics


def test_enhanced_analytics():
    """Test enhanced analytics functionality"""
    print("Testing Enhanced Analytics Reporting System")
    print("=" * 50)

    # Create configuration
    config = {
        "enable_monitoring": True,
        "sampling_interval": 0.5,
        "max_history": 100,
        "enhanced_reporting": {
            "enabled": True,
            "output_dir": "simple_test_reports",
            "include_graphs": True,
            "executive_summary": True,
            "report_formats": ["markdown", "json"],
            "dpi": 150
        }
    }

    # Ensure output directory exists
    test_dir = Path("simple_test_reports")
    test_dir.mkdir(exist_ok=True)

    try:
        # Initialize analytics
        analytics = ChunkingAnalytics(config)
        print("[OK] Analytics system initialized")

        # Start session
        analytics.start_session()
        print("[OK] Analytics session started")

        # Simulate some operations
        with analytics.monitor_operation("test_operation", "testing"):
            time.sleep(1)

        # Add model analysis
        analytics.add_model_analysis(
            model_name="test_model",
            original_size_mb=256.0,
            chunk_count=4,
            chunk_sizes_mb=[64.0, 64.0, 64.0, 64.0],
            parameter_count=1000000,
            memory_footprint_mb=200.0,
            conversion_time_ms=1000.0,
            chunking_time_ms=500.0,
            method="graph_level"
        )
        print("[OK] Test operations completed")

        # Let monitoring collect data
        time.sleep(1)

        # End session
        analytics.end_session()
        print("[OK] Analytics session ended")

        # Generate reports
        try:
            saved_files = analytics.save_enhanced_reports()
            print("[OK] Enhanced reports generated")

            for format_type, filepath in saved_files.items():
                if Path(filepath).exists():
                    print(f"[OK] {format_type.upper()}: {Path(filepath).name}")
                else:
                    print(f"[WARN] {format_type.upper()}: File not found")

            return True

        except Exception as e:
            print(f"[WARN] Enhanced reporting failed: {e}")
            return False

    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False


def test_graph_generation():
    """Test basic graph generation"""
    print("\nTesting Graph Generation")
    print("=" * 30)

    try:
        from analytics.reporting import GraphGenerator

        # Ensure output directory exists
        test_dir = Path("simple_graph_test")
        test_dir.mkdir(exist_ok=True)

        # Create test data
        test_data = {
            "CPU": 75.5,
            "Memory": 1024.0,
            "Disk": 512.0
        }

        # Initialize graph generator
        graph_gen = GraphGenerator(output_dir=str(test_dir.absolute()), dpi=150)
        print("[OK] Graph generator initialized")

        # Create a simple bar chart
        bar_path = graph_gen.create_bar_chart(
            data=test_data,
            title="Test Resource Usage",
            xlabel="Resource Type",
            ylabel="Usage"
        )

        if Path(bar_path).exists():
            print(f"[OK] Bar chart generated: {Path(bar_path).name}")
            return True
        else:
            print("[WARN] Bar chart file not found")
            return False

    except ImportError as e:
        print(f"[WARN] Graph generation skipped: {e}")
        return False
    except Exception as e:
        print(f"[WARN] Graph generation failed: {e}")
        return False


def test_markdown_reporting():
    """Test basic markdown reporting"""
    print("\nTesting Markdown Reporting")
    print("=" * 35)

    try:
        from analytics.reporting import MarkdownReporter

        # Ensure output directory exists
        test_dir = Path("simple_markdown_test")
        test_dir.mkdir(exist_ok=True)

        # Create test data
        test_data = {
            "session_info": {
                "start_time": datetime.now().isoformat(),
                "duration_seconds": 5.0,
                "total_operations": 2
            },
            "system_resources": {
                "peak_usage": {
                    "peak_memory_mb": 1024.0,
                    "peak_cpu_percent": 75.0
                }
            },
            "performance_timing": {
                "phase_breakdown": {
                    "loading": {"total_ms": 1000, "avg_ms": 1000, "count": 1},
                    "processing": {"total_ms": 2000, "avg_ms": 2000, "count": 1}
                },
                "total_operation_time_ms": 3000
            },
            "model_analysis": {
                "models_analyzed": {
                    "test_model": {
                        "method": "graph_level",
                        "efficiency_score": 85.0,
                        "compression_ratio": 1.2,
                        "chunk_count": 4
                    }
                }
            },
            "recommendations": [
                "Test recommendation 1",
                "Test recommendation 2"
            ]
        }

        # Initialize markdown reporter
        reporter = MarkdownReporter(output_dir=str(test_dir.absolute()))
        print("[OK] Markdown reporter initialized")

        # Generate markdown report
        markdown_path = reporter.generate_report(test_data)

        if Path(markdown_path).exists():
            print(f"[OK] Markdown report generated: {Path(markdown_path).name}")

            # Check content
            with open(markdown_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if "# Model Chunking Analytics Report" in content:
                    print("[OK] Report contains title")
                if "## Executive Summary" in content:
                    print("[OK] Report contains executive summary")
                if "## System Resource Analysis" in content:
                    print("[OK] Report contains system analysis")

            return True
        else:
            print("[WARN] Markdown report file not found")
            return False

    except ImportError as e:
        print(f"[WARN] Markdown reporting skipped: {e}")
        return False
    except Exception as e:
        print(f"[WARN] Markdown reporting failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Enhanced Analytics Simple Test Suite")
    print("=" * 40)

    results = []

    # Test enhanced analytics
    results.append(("Enhanced Analytics", test_enhanced_analytics()))

    # Test graph generation
    results.append(("Graph Generation", test_graph_generation()))

    # Test markdown reporting
    results.append(("Markdown Reporting", test_markdown_reporting()))

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\nPassed: {passed}/{total} tests")

    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        print("\nGenerated files can be found in:")
        print("- simple_test_reports")
        print("- simple_graph_test")
        print("- simple_markdown_test")
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed or skipped")

    return passed == total


if __name__ == "__main__":
    main()