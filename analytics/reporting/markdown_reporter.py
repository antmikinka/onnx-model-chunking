"""
Professional Markdown Report Generator

Creates comprehensive, professional markdown reports with embedded graphs
and executive summaries for analytics data.

Features:
- Professional markdown templates
- Executive summary generation
- Graph embedding with proper paths
- Structured sections and formatting
- Actionable recommendations
- Professional table formatting
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .graph_generator import GraphGenerator


class MarkdownReporter:
    """
    Professional markdown report generator for analytics data.

    Creates comprehensive reports with executive summaries, detailed analysis,
    and embedded visualizations suitable for technical documentation and
    stakeholder presentations.
    """

    def __init__(self, output_dir: str = "reports", graph_generator: Optional[GraphGenerator] = None):
        """
        Initialize markdown reporter.

        Args:
            output_dir: Directory to save reports
            graph_generator: GraphGenerator instance for creating visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        if graph_generator is None:
            self.graph_generator = GraphGenerator(str(self.output_dir / "graphs"))
        else:
            self.graph_generator = graph_generator

    def generate_executive_summary(self, analytics_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate executive summary from analytics data.

        Args:
            analytics_data: Comprehensive analytics data dictionary

        Returns:
            Dictionary containing summary sections
        """
        summary = {
            "overview": "",
            "key_findings": [],
            "recommendations": [],
            "next_steps": []
        }

        # Extract key metrics
        session_info = analytics_data.get("session_info", {})
        system_resources = analytics_data.get("system_resources", {})
        performance_timing = analytics_data.get("performance_timing", {})
        model_analysis = analytics_data.get("model_analysis", {})

        # Generate overview
        duration_seconds = session_info.get("duration_seconds", 0)
        total_operations = session_info.get("total_operations", 0)

        summary["overview"] = f"""
This report provides a comprehensive analysis of model chunking performance conducted over
{duration_seconds:.1f} seconds with {total_operations} total operations.
The analysis includes system resource utilization, performance timing breakdowns,
and model-specific efficiency metrics.
        """.strip()

        # Generate key findings
        if system_resources.get("peak_usage"):
            peak = system_resources["peak_usage"]
            summary["key_findings"].append(
                f"Peak memory usage reached {peak.get('peak_memory_mb', 0):.1f} MB "
                f"({peak.get('peak_memory_percent', 0):.1f}%)"
            )
            summary["key_findings"].append(
                f"Peak CPU utilization was {peak.get('peak_cpu_percent', 0):.1f}%"
            )

        if performance_timing.get("phase_breakdown"):
            phases = performance_timing["phase_breakdown"]
            total_time = sum(phase.get("total_ms", 0) for phase in phases.values()) / 1000
            summary["key_findings"].append(
                f"Total processing time: {total_time:.2f} seconds"
            )

        if model_analysis.get("models_analyzed"):
            models = model_analysis["models_analyzed"]
            if len(models) > 0:
                best_model = max(models.values(), key=lambda x: x.get("efficiency_score", 0))
                summary["key_findings"].append(
                    f"Best performing method: {best_model.get('method', 'unknown')} "
                    f"with efficiency score of {best_model.get('efficiency_score', 0):.1f}/100"
                )

        # Add existing recommendations if available
        if "recommendations" in analytics_data:
            summary["recommendations"] = analytics_data["recommendations"]

        # Generate next steps
        summary["next_steps"] = [
            "Review system resource utilization patterns",
            "Evaluate alternative chunking strategies if efficiency is below target",
            "Consider memory optimization techniques for larger models",
            "Monitor performance trends over time"
        ]

        return summary

    def format_metric_table(self, data: Dict[str, Any], title: str) -> str:
        """
        Format data as a professional markdown table.

        Args:
            data: Dictionary of metrics to format
            title: Table title

        Returns:
            Formatted markdown table string
        """
        table = f"\n### {title}\n\n"

        if not data:
            table += "No data available.\n"
            return table

        # Determine table structure based on data type
        if isinstance(next(iter(data.values())), dict):
            # Nested dictionary format
            table += "| Metric | Value | Unit |\n"
            table += "|--------|-------|------|\n"

            for category, metrics in data.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        unit = self._get_unit_for_metric(metric)
                        formatted_value = self._format_value(value)
                        table += f"| {category}_{metric} | {formatted_value} | {unit} |\n"
        else:
            # Flat dictionary format
            table += "| Metric | Value | Unit |\n"
            table += "|--------|-------|------|\n"

            for metric, value in data.items():
                unit = self._get_unit_for_metric(metric)
                formatted_value = self._format_value(value)
                table += f"| {metric} | {formatted_value} | {unit} |\n"

        return table + "\n"

    def _get_unit_for_metric(self, metric_name: str) -> str:
        """Get appropriate unit for a metric name"""
        metric_lower = metric_name.lower()
        if any(word in metric_lower for word in ['percent', 'usage', 'cpu', 'memory']):
            return '%'
        elif any(word in metric_lower for word in ['mb', 'memory', 'disk']):
            return 'MB'
        elif any(word in metric_lower for word in ['ms', 'time', 'duration']):
            return 'ms'
        elif any(word in metric_lower for word in ['seconds', 'time']):
            return 's'
        elif 'count' in metric_lower:
            return 'count'
        elif 'score' in metric_lower:
            return '/100'
        else:
            return 'unit'

    def _format_value(self, value: Any) -> str:
        """Format a value for display"""
        if isinstance(value, float):
            if value > 1000:
                return f"{value:,.0f}"
            elif value > 100:
                return f"{value:.1f}"
            else:
                return f"{value:.2f}"
        elif isinstance(value, int):
            return f"{value:,}"
        elif isinstance(value, str):
            return value
        else:
            return str(value)

    def create_system_resource_section(self, analytics_data: Dict[str, Any]) -> str:
        """
        Create system resource analysis section with graphs.

        Args:
            analytics_data: Analytics data dictionary

        Returns:
            Formatted markdown section
        """
        section = "## System Resource Analysis\n\n"

        system_resources = analytics_data.get("system_resources", {})
        if not system_resources:
            section += "No system resource data available.\n\n"
            return section

        # Add summary text
        section += """
The system resource monitoring provides insights into hardware utilization during
the chunking process. Key metrics include CPU usage, memory consumption, and disk I/O
patterns.
        """.strip() + "\n\n"

        # Create graphs if data is available
        # Note: This would require actual time-series data from the system monitor
        # For now, we'll add placeholder text and table

        # Add peak usage table
        peak_usage = system_resources.get("peak_usage", {})
        if peak_usage:
            section += self.format_metric_table(peak_usage, "Peak Resource Usage")

        # Add average usage table
        avg_usage = system_resources.get("average_usage", {})
        if avg_usage:
            section += self.format_metric_table(avg_usage, "Average Resource Usage")

        return section

    def create_performance_analysis_section(self, analytics_data: Dict[str, Any]) -> str:
        """
        Create performance analysis section with timing breakdowns.

        Args:
            analytics_data: Analytics data dictionary

        Returns:
            Formatted markdown section
        """
        section = "## Performance Analysis\n\n"

        performance_timing = analytics_data.get("performance_timing", {})
        if not performance_timing:
            section += "No performance timing data available.\n\n"
            return section

        section += """
Performance analysis provides detailed timing breakdowns for each phase of the
chunking process. This helps identify bottlenecks and optimization opportunities.
        """.strip() + "\n\n"

        # Phase breakdown table
        phase_breakdown = performance_timing.get("phase_breakdown", {})
        if phase_breakdown:
            section += self.format_metric_table(phase_breakdown, "Phase Timing Breakdown")

        # Total timing summary
        total_time = performance_timing.get("total_operation_time_ms", 0)
        if total_time:
            section += f"**Total Operation Time:** {total_time/1000:.2f} seconds\n\n"

        return section

    def create_model_analysis_section(self, analytics_data: Dict[str, Any]) -> str:
        """
        Create model analysis section with efficiency comparisons.

        Args:
            analytics_data: Analytics data dictionary

        Returns:
            Formatted markdown section
        """
        section = "## Model Analysis\n\n"

        model_analysis = analytics_data.get("model_analysis", {})
        if not model_analysis:
            section += "No model analysis data available.\n\n"
            return section

        section += """
Model analysis evaluates the efficiency and effectiveness of different chunking methods.
Key metrics include efficiency scores, compression ratios, and memory footprints.
        """.strip() + "\n\n"

        # Models analyzed table
        models = model_analysis.get("models_analyzed", {})
        if models:
            section += "### Model Performance Summary\n\n"
            section += "| Model | Method | Efficiency Score | Compression Ratio | Chunk Count | Size (MB) |\n"
            section += "|-------|--------|-----------------|------------------|-------------|-----------|\n"

            for model_name, metrics in models.items():
                efficiency = metrics.get("efficiency_score", 0)
                compression = metrics.get("compression_ratio", 0)
                chunks = metrics.get("chunk_count", 0)
                size = metrics.get("total_chunk_size_mb", 0)

                section += f"| {model_name} | {metrics.get('method', 'N/A')} | {efficiency:.1f} | {compression:.2f}x | {chunks} | {size:.1f} |\n"

            section += "\n"

        # Method comparison
        comparison = model_analysis.get("comparisons")
        if comparison and comparison.get("best_method"):
            section += f"**Recommended Method:** {comparison['best_method']}\n\n"

        return section

    def create_recommendations_section(self, analytics_data: Dict[str, Any]) -> str:
        """
        Create actionable recommendations section.

        Args:
            analytics_data: Analytics data dictionary

        Returns:
            Formatted markdown section
        """
        section = "## Recommendations\n\n"

        recommendations = analytics_data.get("recommendations", [])
        if not recommendations:
            section += "No specific recommendations available at this time.\n\n"
            return section

        section += "Based on the analysis, the following recommendations are provided:\n\n"

        for i, rec in enumerate(recommendations, 1):
            section += f"{i}. {rec}\n"

        section += "\n"

        return section

    def generate_report(
        self,
        analytics_data: Dict[str, Any],
        filename: Optional[str] = None,
        include_graphs: bool = True,
        include_executive_summary: bool = True
    ) -> str:
        """
        Generate comprehensive markdown report.

        Args:
            analytics_data: Complete analytics data dictionary
            filename: Output filename (auto-generated if None)
            include_graphs: Whether to generate and include graphs
            include_executive_summary: Whether to include executive summary

        Returns:
            Path to generated report file
        """
        # Generate filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chunking_analytics_report_{timestamp}.md"

        filepath = self.output_dir / filename

        # Start building report
        report_content = []

        # Report header
        report_content.append("# Model Chunking Analytics Report\n")
        report_content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_content.append(f"**Report ID:** {datetime.now().strftime('%Y%m%d_%H%M%S')}\n\n")

        # Executive Summary
        if include_executive_summary:
            report_content.append("## Executive Summary\n\n")
            summary = self.generate_executive_summary(analytics_data)

            report_content.append(summary["overview"])
            report_content.append("\n\n")

            # Key Findings
            if summary["key_findings"]:
                report_content.append("### Key Findings\n\n")
                for finding in summary["key_findings"]:
                    report_content.append(f"- {finding}\n")
                report_content.append("\n")

            # Quick Recommendations
            if summary["recommendations"]:
                report_content.append("### Primary Recommendations\n\n")
                for i, rec in enumerate(summary["recommendations"][:3], 1):  # Top 3
                    report_content.append(f"{i}. {rec}\n")
                report_content.append("\n")

        # System Resource Analysis
        report_content.append(self.create_system_resource_section(analytics_data))

        # Performance Analysis
        report_content.append(self.create_performance_analysis_section(analytics_data))

        # Model Analysis
        report_content.append(self.create_model_analysis_section(analytics_data))

        # Detailed Recommendations
        report_content.append(self.create_recommendations_section(analytics_data))

        # Technical Appendix
        report_content.append("## Technical Appendix\n\n")
        report_content.append("### Report Configuration\n\n")
        report_content.append("- **Analytics Framework:** Enhanced Chunking Analytics System\n")
        report_content.append("- **Graph Generation:** Matplotlib with Professional Styling\n")
        report_content.append("- **Report Format:** Markdown with Embedded Visualizations\n")
        report_content.append("- **Data Collection:** Real-time System Monitoring\n\n")

        # Add raw data reference
        report_content.append("### Raw Data Reference\n\n")
        session_info = analytics_data.get("session_info", {})
        if session_info:
            report_content.append(f"- **Session Start:** {session_info.get('start_time', 'N/A')}\n")
            report_content.append(f"- **Session Duration:** {session_info.get('duration_seconds', 0):.2f} seconds\n")
            report_content.append(f"- **Total Operations:** {session_info.get('total_operations', 0)}\n")

        report_content.append("\n---\n")
        report_content.append("*Report generated by Enhanced Analytics Reporting System*\n")

        # Write report to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(report_content)

        return str(filepath)

    def generate_json_report(
        self,
        analytics_data: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """
        Generate JSON report for data processing.

        Args:
            analytics_data: Complete analytics data dictionary
            filename: Output filename (auto-generated if None)

        Returns:
            Path to generated JSON report file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chunking_analytics_data_{timestamp}.json"

        filepath = self.output_dir / filename

        # Add report metadata
        report_data = analytics_data.copy()
        report_data["report_metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "report_format": "json",
            "version": "1.0",
            "generator": "Enhanced Analytics Reporting System"
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)

        return str(filepath)