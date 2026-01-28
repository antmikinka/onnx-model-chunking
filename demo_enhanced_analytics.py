"""
Enhanced Analytics Reporting System Demo

Comprehensive demonstration of the professional graph generation and
markdown reporting capabilities with realistic chunking scenarios.

This demo shows:
- Professional graph generation
- Markdown report creation
- Real-time system monitoring
- Model analysis and comparisons
- Executive summary generation
"""

import time
import random
from datetime import datetime, timedelta
from pathlib import Path

from chunking_analytics import ChunkingAnalytics


def demo_model_chunking_scenarios():
    """Demonstrate realistic model chunking scenarios"""
    print("="*60)
    print("ENHANCED ANALYTICS REPORTING SYSTEM DEMO")
    print("="*60)
    print("Demonstrating professional graph generation and markdown reporting")
    print()

    # Configuration for demo
    config = {
        "enable_monitoring": True,
        "sampling_interval": 0.5,
        "max_history": 200,
        "real_time_updates": True,
        "enhanced_reporting": {
            "enabled": True,
            "output_dir": "demo_reports",
            "include_graphs": True,
            "executive_summary": True,
            "report_formats": ["markdown", "json"],
            "dpi": 200,
            "graph_style": "professional"
        },
        "performance_alerts": {
            "memory_percent": 85.0,
            "cpu_percent": 90.0,
            "disk_usage_percent": 98.0,
            "conversion_time_seconds": 20.0
        }
    }

    # Ensure output directory exists
    demo_dir = Path("demo_reports")
    demo_dir.mkdir(exist_ok=True)

    # Initialize analytics
    analytics = ChunkingAnalytics(config)
    print("[INIT] Enhanced Analytics System Initialized")
    print(f"[CONFIG] Output directory: {demo_dir.absolute()}")
    print(f"[CONFIG] Graph resolution: {config['enhanced_reporting']['dpi']} DPI")
    print()

    # Start monitoring session
    analytics.start_session()
    print("[START] Analytics monitoring session started")
    print()

    # Scenario 1: Qwen2-0.5B Model Processing
    print("SCENARIO 1: Processing Qwen2-0.5B Model")
    print("-" * 40)

    # Model Loading Phase
    with analytics.monitor_operation("load_qwen2_0.5b", "loading",
                                   model_name="qwen2-0.5b", size="0.5B"):
        print("[PHASE] Loading model...")
        time.sleep(1.2)  # Simulate loading time

    # ONNX Conversion Phase
    with analytics.monitor_operation("convert_to_onnx", "conversion",
                                   model_name="qwen2-0.5b", target_format="onnx"):
        print("[PHASE] Converting to ONNX format...")
        time.sleep(3.5)  # Simulate conversion time

    # Graph-Level Chunking
    with analytics.monitor_operation("graph_level_chunking", "chunking",
                                   model_name="qwen2-0.5b", method="graph_level",
                                   chunk_size=1000):
        print("[PHASE] Graph-level chunking...")
        time.sleep(2.8)  # Simulate chunking time

    # Saving Chunks
    with analytics.monitor_operation("save_chunks_graph", "saving",
                                   model_name="qwen2-0.5b", method="graph_level",
                                   chunk_count=8):
        print("[PHASE] Saving graph-level chunks...")
        time.sleep(1.0)  # Simulate saving time

    # Add model analysis
    analytics.add_model_analysis(
        model_name="qwen2-0.5b",
        original_size_mb=512.0,
        chunk_count=8,
        chunk_sizes_mb=[64.5, 63.8, 65.2, 64.1, 63.9, 64.7, 64.3, 64.0],
        parameter_count=467000000,
        memory_footprint_mb=380.5,
        conversion_time_ms=3500.0,
        chunking_time_ms=2800.0,
        method="graph_level"
    )

    print("[COMPLETE] Graph-level chunking completed")
    print()

    # Let monitoring collect some data
    print("[MONITOR] Collecting system metrics...")
    time.sleep(2)

    # Scenario 2: PyTorch-Level Chunking (Alternative Method)
    print("SCENARIO 2: PyTorch-Level Chunking (Alternative)")
    print("-" * 40)

    # Model Loading for PyTorch method
    with analytics.monitor_operation("load_pytorch_model", "loading",
                                   model_name="qwen2-0.5b-pytorch", size="0.5B"):
        print("[PHASE] Loading PyTorch model...")
        time.sleep(1.5)

    # PyTorch-Level Chunking
    with analytics.monitor_operation("pytorch_level_chunking", "chunking",
                                   model_name="qwen2-0.5b", method="pytorch_level",
                                   chunk_size=800):
        print("[PHASE] PyTorch-level chunking...")
        time.sleep(4.2)  # Slower chunking for PyTorch method

    # Saving PyTorch Chunks
    with analytics.monitor_operation("save_chunks_pytorch", "saving",
                                   model_name="qwen2-0.5b", method="pytorch_level",
                                   chunk_count=6):
        print("[PHASE] Saving PyTorch-level chunks...")
        time.sleep(1.3)

    # Add second model analysis for comparison
    analytics.add_model_analysis(
        model_name="qwen2-0.5b",
        original_size_mb=512.0,
        chunk_count=6,
        chunk_sizes_mb=[85.3, 84.9, 85.7, 85.1, 84.8, 85.2],
        parameter_count=467000000,
        memory_footprint_mb=420.3,  # Higher memory usage
        conversion_time_ms=4200.0,  # Slower conversion
        chunking_time_ms=4100.0,    # Slower chunking
        method="pytorch_level"
    )

    print("[COMPLETE] PyTorch-level chunking completed")
    print()

    # Scenario 3: Hybrid Approach (Best of Both)
    print("SCENARIO 3: Hybrid Approach (Optimized)")
    print("-" * 40)

    with analytics.monitor_operation("hybrid_chunking", "chunking",
                                   model_name="qwen2-0.5b", method="hybrid",
                                   optimization="enabled"):
        print("[PHASE] Hybrid chunking with optimizations...")
        time.sleep(2.2)  # Optimized processing time

    with analytics.monitor_operation("save_hybrid_chunks", "saving",
                                   model_name="qwen2-0.5b", method="hybrid"):
        print("[PHASE] Saving optimized chunks...")
        time.sleep(0.8)

    # Add hybrid method analysis
    analytics.add_model_analysis(
        model_name="qwen2-0.5b",
        original_size_mb=512.0,
        chunk_count=7,
        chunk_sizes_mb=[73.2, 73.5, 72.8, 73.1, 73.0, 73.3, 73.1],
        parameter_count=467000000,
        memory_footprint_mb=390.1,
        conversion_time_ms=3800.0,
        chunking_time_ms=2200.0,
        method="hybrid"
    )

    print("[COMPLETE] Hybrid chunking completed")
    print()

    # Additional monitoring for better graphs
    print("[MONITOR] Final data collection...")
    time.sleep(3)

    # End monitoring session
    analytics.end_session()
    print("[END] Analytics monitoring session ended")
    print()

    # Generate comprehensive enhanced reports
    print("GENERATING ENHANCED REPORTS")
    print("=" * 40)

    try:
        saved_files = analytics.save_enhanced_reports()
        print("[SUCCESS] Enhanced reports generated successfully!")
        print()

        for format_type, filepath in saved_files.items():
            path_obj = Path(filepath)
            if path_obj.exists():
                size_kb = path_obj.stat().st_size / 1024
                print(f"[REPORT] {format_type.upper()}: {path_obj.name} ({size_kb:.1f} KB)")
            else:
                print(f"[WARN] {format_type.upper()}: File not found")

        print()
        print("[INFO] Reports include:")
        print("- Executive summary with key findings")
        print("- System resource analysis with graphs")
        print("- Performance timing breakdowns")
        print("- Model efficiency comparisons")
        print("- Actionable recommendations")
        print("- Professional visualizations")

        # Show generated graph files
        graphs_dir = demo_dir / "graphs"
        if graphs_dir.exists():
            graph_files = list(graphs_dir.glob("*.png"))
            if graph_files:
                print()
                print("[GRAPHS] Generated visualizations:")
                for graph_file in graph_files:
                    size_kb = graph_file.stat().st_size / 1024
                    print(f"  - {graph_file.name} ({size_kb:.1f} KB)")

    except Exception as e:
        print(f"[ERROR] Report generation failed: {e}")
        print("This might be due to missing matplotlib or other dependencies")

    return saved_files if 'saved_files' in locals() else {}


def demonstrate_direct_graph_generation():
    """Demonstrate direct graph generation capabilities"""
    print()
    print("DIRECT GRAPH GENERATION DEMO")
    print("=" * 40)

    try:
        from analytics.reporting import GraphGenerator

        # Create output directory
        graph_dir = Path("demo_direct_graphs")
        graph_dir.mkdir(exist_ok=True)

        # Initialize graph generator
        graph_gen = GraphGenerator(output_dir=str(graph_dir.absolute()), dpi=200)
        print("[INIT] Graph generator initialized")

        # Create sample time-series data
        base_time = datetime.now()
        time_points = [base_time - timedelta(minutes=i) for i in range(10, 0, -1)]

        # CPU usage simulation
        cpu_data = [(tp, 20 + random.uniform(0, 30)) for tp in time_points]

        # Memory usage simulation
        memory_data = [(tp, 1500 + random.uniform(-200, 500)) for tp in time_points]

        # Generate system dashboard
        dashboard_path = graph_gen.create_system_resource_dashboard(
            cpu_data=cpu_data,
            memory_data=memory_data,
            title="Demo: System Resource Utilization"
        )
        print(f"[GRAPH] System dashboard: {Path(dashboard_path).name}")

        # Create efficiency comparison chart
        efficiency_data = {
            "Graph-Level": 85.2,
            "PyTorch-Level": 72.8,
            "Hybrid Approach": 89.5,
            "Baseline": 65.0
        }

        efficiency_path = graph_gen.create_bar_chart(
            data=efficiency_data,
            title="Demo: Method Efficiency Comparison",
            xlabel="Chunking Method",
            ylabel="Efficiency Score (/100)",
            orientation="horizontal"
        )
        print(f"[GRAPH] Efficiency comparison: {Path(efficiency_path).name}")

        # Create timing breakdown pie chart
        timing_data = {
            "Model Loading": 15,
            "ONNX Conversion": 35,
            "Chunking Process": 30,
            "File I/O": 20
        }

        timing_path = graph_gen.create_pie_chart(
            data=timing_data,
            title="Demo: Processing Time Distribution"
        )
        print(f"[GRAPH] Timing distribution: {Path(timing_path).name}")

        print()
        print("[SUCCESS] Direct graph generation completed!")
        print(f"[INFO] Graphs saved to: {graph_dir.absolute()}")

    except ImportError as e:
        print(f"[SKIP] Graph generation demo skipped: {e}")
    except Exception as e:
        print(f"[ERROR] Graph generation failed: {e}")


def show_report_preview():
    """Show a preview of generated report content"""
    print()
    print("REPORT PREVIEW")
    print("=" * 40)

    demo_report = Path("demo_reports/chunking_analytics_report_*.md")
    report_files = list(Path("demo_reports").glob("chunking_analytics_report_*.md"))

    if report_files:
        latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
        print(f"[PREVIEW] Showing content from: {latest_report.name}")
        print()

        try:
            with open(latest_report, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Show first 30 lines
            for i, line in enumerate(lines[:30], 1):
                print(f"{i:2d}: {line.rstrip()}")

            if len(lines) > 30:
                print(f"... ({len(lines) - 30} more lines)")

            print()
            print(f"[INFO] Full report: {latest_report.absolute()}")

        except Exception as e:
            print(f"[ERROR] Could not read report: {e}")
    else:
        print("[INFO] No markdown reports found for preview")


def main():
    """Run the complete demonstration"""
    print("Enhanced Analytics Reporting System")
    print("Professional Graph Generation & Markdown Reporting")
    print("Demo Version 1.0")
    print()

    start_time = time.time()

    # Run main demonstration
    saved_files = demo_model_chunking_scenarios()

    # Demonstrate direct graph generation
    demonstrate_direct_graph_generation()

    # Show report preview
    show_report_preview()

    # Final summary
    print()
    print("DEMO SUMMARY")
    print("=" * 40)

    duration = time.time() - start_time
    print(f"[TIME] Total demo duration: {duration:.2f} seconds")

    if saved_files:
        print(f"[REPORTS] Generated {len(saved_files)} report files")

        # Check file sizes
        total_size = 0
        for format_type, filepath in saved_files.items():
            if Path(filepath).exists():
                size = Path(filepath).stat().st_size
                total_size += size
                print(f"  - {format_type.upper()}: {size:,} bytes")

        print(f"[SIZE] Total report size: {total_size:,} bytes ({total_size/1024:.1f} KB)")

    print()
    print("[OUTPUT] Generated files:")
    print("  - demo_reports/: Enhanced analytics reports")
    print("  - demo_direct_graphs/: Direct graph generation")
    print()
    print("[FEATURES] Demonstrated:")
    print("  + Real-time system monitoring")
    print("  + Professional graph generation")
    print("  + Markdown report creation")
    print("  + Executive summary generation")
    print("  + Model analysis and comparison")
    print("  + Professional styling and formatting")
    print("  + Multiple output formats")
    print()
    print("[NEXT] Review generated reports and graphs in the output directories")
    print("[NEXT] Customize configuration for your specific requirements")
    print("[NEXT] Integrate with your chunking workflow")


if __name__ == "__main__":
    main()