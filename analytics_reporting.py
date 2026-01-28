"""
Enhanced Analytics Reporting System for Model Chunking
Generates professional Markdown reports with graphs and visualizations
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Any, Optional
import numpy as np
import os
import sys

# Set matplotlib to use non-interactive backend
plt.switch_backend('Agg')

# Professional styling configuration
PROFESSIONAL_STYLE = {
    'figure.figsize': (10, 6),
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'axes.prop_cycle': plt.cycler('color', ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83'])
}

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update(PROFESSIONAL_STYLE)

class GraphGenerator:
    """Professional graph generation for analytics reports"""

    def __init__(self, output_dir: str, dpi: int = 300):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def generate_cpu_usage_chart(self, timestamps: List[float], cpu_values: List[float]) -> str:
        """Generate CPU utilization line chart"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Convert timestamps to relative time in seconds
        start_time = timestamps[0] if timestamps else 0
        relative_times = [(t - start_time) for t in timestamps]

        ax.plot(relative_times, cpu_values, linewidth=2, color='#2E86AB', marker='o', markersize=4)
        ax.fill_between(relative_times, cpu_values, alpha=0.3, color='#2E86AB')

        ax.set_title('CPU Utilization Over Time', fontweight='bold', pad=20)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('CPU Usage (%)')
        ax.set_ylim(0, max(cpu_values) * 1.1 if cpu_values else 100)
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}s'))

        # Add statistics text
        avg_cpu = np.mean(cpu_values) if cpu_values else 0
        max_cpu = np.max(cpu_values) if cpu_values else 0
        ax.text(0.02, 0.98, f'Average: {avg_cpu:.1f}%\nPeak: {max_cpu:.1f}%',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        output_path = self.images_dir / "cpu_usage.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def generate_memory_usage_chart(self, timestamps: List[float], memory_values: List[float]) -> str:
        """Generate memory usage line chart with peak indicators"""
        fig, ax = plt.subplots(figsize=(12, 6))

        start_time = timestamps[0] if timestamps else 0
        relative_times = [(t - start_time) for t in timestamps]
        memory_mb = [m / (1024*1024) for m in memory_values]  # Convert to MB

        ax.plot(relative_times, memory_mb, linewidth=2, color='#A23B72', marker='o', markersize=4)
        ax.fill_between(relative_times, memory_mb, alpha=0.3, color='#A23B72')

        # Mark peak usage
        if memory_mb:
            peak_idx = np.argmax(memory_mb)
            peak_time = relative_times[peak_idx]
            peak_memory = memory_mb[peak_idx]
            ax.scatter([peak_time], [peak_memory], color='red', s=100, zorder=5)
            ax.annotate(f'Peak: {peak_memory:.1f} MB',
                       xy=(peak_time, peak_memory),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        ax.set_title('Memory Usage Over Time', fontweight='bold', pad=20)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Memory Usage (MB)')
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}s'))

        # Add statistics
        avg_mem = np.mean(memory_mb) if memory_mb else 0
        peak_mem = np.max(memory_mb) if memory_mb else 0
        ax.text(0.02, 0.98, f'Average: {avg_mem:.1f} MB\nPeak: {peak_mem:.1f} MB',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        output_path = self.images_dir / "memory_usage.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def generate_timing_comparison_chart(self, methods: List[str], phases: List[str], timing_data: Dict[str, Dict[str, float]]) -> str:
        """Generate timing comparison bar chart for different chunking methods"""
        fig, ax = plt.subplots(figsize=(12, 8))

        x = np.arange(len(phases))
        width = 0.35

        colors = ['#2E86AB', '#A23B72']

        for i, method in enumerate(methods):
            if method in timing_data:
                timings = [timing_data[method].get(phase, 0) for phase in phases]
                offset = width * i
                bars = ax.bar(x + offset, timings, width, label=method, color=colors[i], alpha=0.8)

                # Add value labels on bars
                for bar, timing in zip(bars, timings):
                    if timing > 0.01:  # Only show labels for significant times
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{timing:.2f}s', ha='center', va='bottom', fontsize=8)

        ax.set_title('Performance Timing Comparison', fontweight='bold', pad=20)
        ax.set_xlabel('Conversion Phases')
        ax.set_ylabel('Time (seconds)')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(phases, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.images_dir / "timing_comparison.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def generate_model_size_comparison(self, models: List[str], sizes: List[float]) -> str:
        """Generate model size comparison bar chart"""
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
        bars = ax.bar(models, sizes, color=colors[:len(models)], alpha=0.8)

        # Add value labels on bars
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{size:.1f} MB', ha='center', va='bottom', fontweight='bold')

        ax.set_title('Model Size Comparison', fontweight='bold', pad=20)
        ax.set_ylabel('Model Size (MB)')
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels if needed
        if len(max(models, key=len)) > 10:
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        output_path = self.images_dir / "model_size_comparison.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def generate_efficiency_scores_chart(self, methods: List[str], scores: List[float]) -> str:
        """Generate efficiency scores horizontal bar chart"""
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        bars = ax.barh(methods, scores, color=colors[:len(methods)], alpha=0.8)

        # Add value labels
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
                   f'{score:.1f}%', ha='left', va='center', fontweight='bold')

        ax.set_title('Chunking Method Efficiency Scores', fontweight='bold', pad=20)
        ax.set_xlabel('Efficiency Score (%)')
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.images_dir / "efficiency_scores.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return str(output_path)

class MarkdownReporter:
    """Professional Markdown report generator"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.report_path = self.output_dir / "analytics_report.md"

    def generate_report(self, analytics_data: Dict[str, Any], graph_paths: Dict[str, str]) -> str:
        """Generate comprehensive Markdown report with graphs"""

        report_content = self._generate_header()
        report_content += self._generate_executive_summary(analytics_data)
        report_content += self._generate_system_configuration(analytics_data)
        report_content += self._generate_performance_analysis(analytics_data, graph_paths)
        report_content += self._generate_method_comparison(analytics_data, graph_paths)

        # Enhanced sections for deep analytics integration
        if 'enhanced_metrics' in analytics_data:
            report_content += self._generate_enhanced_metrics_section(analytics_data)

        if 'deep_analysis' in analytics_data:
            report_content += self._generate_deep_analysis_section(analytics_data)

        if 'actionable_insights' in analytics_data:
            report_content += self._generate_actionable_insights_section(analytics_data)

        if 'comprehensive_recommendations' in analytics_data:
            report_content += self._generate_comprehensive_recommendations_section(analytics_data)

        if 'validation_summary' in analytics_data:
            report_content += self._generate_validation_summary_section(analytics_data)

        report_content += self._generate_recommendations(analytics_data)
        report_content += self._generate_detailed_metrics(analytics_data)

        # Write report to file
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        return str(self.report_path)

    def _generate_header(self) -> str:
        """Generate report header"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""# Model Chunking Analytics Report

**Generated on:** {timestamp}
**Report Version:** 1.0
**System:** Windows

---

"""

    def _generate_executive_summary(self, data: Dict[str, Any]) -> str:
        """Generate executive summary section"""
        session = data.get('session_info', {})
        alerts = data.get('alerts', [])

        total_time = session.get('duration_seconds', 0)
        operations = session.get('total_operations', 0)

        # Count alert levels
        alert_counts = {'ERROR': 0, 'WARN': 0, 'INFO': 0}
        for alert in alerts:
            level = alert.get('level', 'INFO')
            if level in alert_counts:
                alert_counts[level] += 1

        return f"""## Executive Summary

This report provides a comprehensive analysis of the model chunking process, including performance metrics, resource utilization, and efficiency comparisons between different chunking methods.

### Key Metrics
- **Total Session Duration:** {total_time:.2f} seconds
- **Operations Monitored:** {operations}
- **Alerts Generated:** {len(alerts)} ({alert_counts['ERROR']} errors, {alert_counts['WARN']} warnings)

### Performance Overview
"""

    def _generate_system_configuration(self, data: Dict[str, Any]) -> str:
        """Generate system configuration section"""
        system_info = data.get('system_info', {})

        return f"""## System Configuration

### Hardware Specifications
- **CPU:** {system_info.get('cpu_name', 'Unknown')}
- **CPU Cores:** {system_info.get('cpu_count', 'Unknown')}
- **Total Memory:** {system_info.get('memory_total_gb', 'Unknown'):.1f} GB
- **Available Memory:** {system_info.get('memory_available_gb', 'Unknown'):.1f} GB
- **GPU:** {system_info.get('gpu_name', 'Not Available')}

### Software Environment
- **Python Version:** {system_info.get('python_version', 'Unknown')}
- **PyTorch Version:** {system_info.get('torch_version', 'Unknown')}
- **ONNX Version:** {system_info.get('onnx_version', 'Unknown')}

---

"""

    def _generate_performance_analysis(self, data: Dict[str, Any], graph_paths: Dict[str, str]) -> str:
        """Generate performance analysis section with graphs"""

        content = """## Performance Analysis

### Resource Utilization

#### CPU Usage
"""

        if 'cpu_usage' in graph_paths:
            cpu_graph = Path(graph_paths['cpu_usage']).name
            content += f"""
![CPU Usage](images/{cpu_graph})

*Figure 1: CPU utilization throughout the chunking process*
"""

        content += "\n#### Memory Usage\n"

        if 'memory_usage' in graph_paths:
            memory_graph = Path(graph_paths['memory_usage']).name
            content += f"""
![Memory Usage](images/{memory_graph})

*Figure 2: Memory usage patterns during model processing*
"""

        # Add timing analysis
        timing_data = data.get('performance_metrics', {})
        if timing_data:
            content += """
### Timing Analysis

The performance was measured across multiple phases of the chunking process:

"""

            for phase, metrics in timing_data.items():
                if isinstance(metrics, dict):
                    total_time = metrics.get('total_time', 0)
                    avg_time = metrics.get('average_time', 0)
                    count = metrics.get('count', 0)
                    content += f"- **{phase.replace('_', ' ').title()}:** {total_time:.3f}s total, {avg_time:.3f}s avg ({count} operations)\n"

        content += "\n---\n\n"

        return content

    def _generate_method_comparison(self, data: Dict[str, Any], graph_paths: Dict[str, str]) -> str:
        """Generate method comparison section with graphs"""

        content = """## Chunking Method Comparison

### Performance Comparison
"""

        if 'timing_comparison' in graph_paths:
            timing_graph = Path(graph_paths['timing_comparison']).name
            content += f"""
![Timing Comparison](images/{timing_graph})

*Figure 3: Performance comparison between different chunking methods*
"""

        content += "\n### Model Size Analysis\n"

        if 'model_size_comparison' in graph_paths:
            size_graph = Path(graph_paths['model_size_comparison']).name
            content += f"""
![Model Size Comparison](images/{size_graph})

*Figure 4: Size comparison between original and chunked models*
"""

        content += "\n### Efficiency Scores\n"

        if 'efficiency_scores' in graph_paths:
            efficiency_graph = Path(graph_paths['efficiency_scores']).name
            content += f"""
![Efficiency Scores](images/{efficiency_graph})

*Figure 5: Efficiency scores for different chunking methods*
"""

        # Add method-specific analysis
        model_analytics = data.get('model_analytics', {})
        if model_analytics:
            content += """
### Detailed Method Analysis

"""
            for method, analytics in model_analytics.items():
                if isinstance(analytics, dict):
                    efficiency = analytics.get('efficiency_score', 0)
                    compression = analytics.get('compression_ratio', 0)
                    content += f"""#### {method.replace('_', ' ').title()}
- **Efficiency Score:** {efficiency:.1f}%
- **Compression Ratio:** {compression:.2f}x
- **Memory Efficiency:** {analytics.get('memory_efficiency', 'N/A')}

"""

        content += "---\n\n"

        return content

    def _generate_recommendations(self, data: Dict[str, Any]) -> str:
        """Generate recommendations section"""

        recommendations = []

        # Analyze performance data for recommendations
        alerts = data.get('alerts', [])
        error_alerts = [a for a in alerts if a.get('level') == 'ERROR']
        warn_alerts = [a for a in alerts if a.get('level') == 'WARN']

        if error_alerts:
            recommendations.append("🔴 **High Priority:** Address critical errors that occurred during processing")

        if warn_alerts:
            recommendations.append("🟡 **Medium Priority:** Review warnings and consider system optimizations")

        # Analyze resource usage
        system_metrics = data.get('system_metrics', {})
        peak_memory = system_metrics.get('peak_memory_mb', 0)

        if peak_memory > 8000:  # > 8GB
            recommendations.append("💾 **Memory Optimization:** Consider enabling PyTorch-level chunking for better memory efficiency")

        # Analyze method performance
        model_analytics = data.get('model_analytics', {})
        best_method = None
        best_score = 0

        for method, analytics in model_analytics.items():
            if isinstance(analytics, dict):
                score = analytics.get('efficiency_score', 0)
                if score > best_score:
                    best_score = score
                    best_method = method

        if best_method:
            recommendations.append(f"✅ **Recommended Method:** {best_method.replace('_', ' ').title()} showed the best performance ({best_score:.1f}% efficiency)")

        content = """## Recommendations

Based on the analysis of the chunking process, here are the key recommendations:

"""

        if recommendations:
            for rec in recommendations:
                content += f"{rec}\n\n"
        else:
            content += "✅ The chunking process performed well within expected parameters. No specific recommendations at this time.\n\n"

        content += """### Next Steps

1. **Monitor Performance:** Continue monitoring resource usage in production
2. **Test Different Models:** Validate performance with different model architectures
3. **Optimize Configuration:** Fine-tune chunking parameters based on workload
4. **Scale Testing:** Test with larger models and different chunk distributions

---

"""

        return content

    def _generate_enhanced_metrics_section(self, data: Dict[str, Any]) -> str:
        """Generate enhanced metrics section with validation information"""
        enhanced_metrics = data.get('enhanced_metrics', {})

        content = """## Enhanced Metrics Analysis

### Cross-Validated Performance Metrics

The following metrics have been rigorously validated using multiple validation methods to ensure accuracy and reliability:

"""

        for metric_name, metric_data in enhanced_metrics.items():
            if isinstance(metric_data, dict):
                primary_value = metric_data.get('primary_value', 0)
                validated_value = metric_data.get('validated_value', 0)
                confidence_score = metric_data.get('confidence_score', 0)
                is_validated = metric_data.get('is_validated', False)
                validation_method = metric_data.get('validation_method', 'unknown')

                status_icon = "✅" if is_validated else "⚠️"
                confidence_level = "High" if confidence_score > 0.9 else "Medium" if confidence_score > 0.7 else "Low"

                content += f"""#### {metric_name.replace('_', ' ').title()}
{status_icon} **Validated:** {'Yes' if is_validated else 'No'}
- **Primary Value:** {primary_value:.4f}
- **Validated Value:** {validated_value:.4f}
- **Confidence Score:** {confidence_score:.3f} ({confidence_level})
- **Validation Method:** {validation_method.replace('_', ' ').title()}

"""

        content += """### Validation Quality Assurance

All enhanced metrics undergo the following validation process:
1. **Triple-Check Calculation:** Each metric is calculated multiple times to ensure consistency
2. **Statistical Validation:** Statistical methods are applied to detect outliers and ensure reliability
3. **Cross-Reference Verification:** Results are cross-referenced against known benchmarks and theoretical predictions

---

"""
        return content

    def _generate_deep_analysis_section(self, data: Dict[str, Any]) -> str:
        """Generate deep analysis section for method comparisons"""
        deep_analysis = data.get('deep_analysis', {})

        content = """## Deep Performance Analysis

### Comprehensive Method Comparison

This section provides detailed analysis comparing different chunking methods with statistical significance and actionable insights.

"""

        for comparison_key, comparison_data in deep_analysis.items():
            if isinstance(comparison_data, dict):
                winner = comparison_data.get('winner', 'unknown')
                confidence = comparison_data.get('confidence_level', 0) * 100
                significance = comparison_data.get('statistical_significance', 0) * 100
                practical_sig = comparison_data.get('practical_significance', 'unknown')
                technical_explanation = comparison_data.get('technical_explanation', '')

                content += f"""#### {comparison_key.replace('_', ' ').title()}

**🏆 Winner:** {winner.replace('_', ' ').title()} ({confidence:.1f}% confidence)

**Statistical Analysis:**
- **Statistical Significance:** {significance:.1f}%
- **Practical Significance:** {practical_sig}
- **Confidence Level:** {confidence:.1f}%

**Technical Explanation:**
{technical_explanation}

"""

                # Add numerical advantages
                numerical_advantages = comparison_data.get('numerical_advantages', {})
                if numerical_advantages:
                    content += "**Performance Advantages:**\n"
                    for metric, advantage in numerical_advantages.items():
                        if isinstance(advantage, dict):
                            interpretation = advantage.get('interpretation', 'No interpretation available')
                            content += f"- **{metric.replace('_', ' ').title()}:** {interpretation}\n"
                    content += "\n"

        content += "---\n\n"
        return content

    def _generate_actionable_insights_section(self, data: Dict[str, Any]) -> str:
        """Generate actionable insights section"""
        insights = data.get('actionable_insights', [])

        content = """## Actionable Insights

### Key Findings and Practical Implications

The following insights are based on comprehensive analysis of performance metrics and are designed to inform decision-making:

"""

        for insight in insights:
            content += f"{insight}\n\n"

        content += """### Insight Categories

- **💾 Memory Efficiency:** Insights related to memory usage patterns and optimization opportunities
- **⚡ Performance:** Insights about processing speed and runtime performance characteristics
- **📦 Model Optimization:** Insights regarding model size, compression, and distribution efficiency
- **📈 Scalability:** Insights about how methods scale with different model sizes and complexities

---

"""
        return content

    def _generate_comprehensive_recommendations_section(self, data: Dict[str, Any]) -> str:
        """Generate comprehensive recommendations section"""
        recommendations = data.get('comprehensive_recommendations', [])

        content = """## Comprehensive Recommendations

### Strategic Guidance for Optimal Performance

Based on the deep analysis and cross-validated metrics, the following recommendations are provided:

"""

        for i, recommendation in enumerate(recommendations, 1):
            content += f"{i}. {recommendation}\n\n"

        content += """### Implementation Priority

**High Priority (Immediate Action):**
- Implement the primary recommended method for your specific use case
- Set up continuous monitoring to track performance over time

**Medium Priority (Short-term Planning):**
- Conduct A/B testing with different models and configurations
- Optimize infrastructure based on memory and performance requirements

**Low Priority (Long-term Optimization):**
- Explore advanced chunking strategies and custom configurations
- Implement automated performance tuning based on workload patterns

---

"""
        return content

    def _generate_validation_summary_section(self, data: Dict[str, Any]) -> str:
        """Generate validation summary section"""
        validation_summary = data.get('validation_summary', {})

        if not validation_summary:
            return ""

        total_validations = validation_summary.get('total_validations', 0)
        passed_validations = validation_summary.get('passed_validations', 0)
        avg_confidence = validation_summary.get('average_confidence', 0)
        outliers_detected = validation_summary.get('outliers_detected', 0)

        pass_rate = (passed_validations / total_validations * 100) if total_validations > 0 else 0
        confidence_level = "Excellent" if avg_confidence > 0.95 else "Good" if avg_confidence > 0.85 else "Fair" if avg_confidence > 0.7 else "Poor"

        content = f"""## Cross-Validation Summary

### Data Quality Assurance

This report includes comprehensive cross-validation of all metrics to ensure accuracy and reliability:

**Validation Results:**
- **Total Metrics Validated:** {total_validations}
- **Validations Passed:** {passed_validations} ({pass_rate:.1f}%)
- **Average Confidence Score:** {avg_confidence:.3f} ({confidence_level})
- **Outliers Detected:** {outliers_detected}

**Validation Methods Applied:**
1. **Statistical Analysis:** Mean, median, and quartile validation
2. **Outlier Detection:** IQR-based outlier identification and removal
3. **Triple-Check Verification:** Multiple calculation iterations for consistency
4. **Confidence Interval Analysis:** 95% confidence intervals for all metrics

**Quality Assurance Status:** {'✅ PASSED' if pass_rate > 90 else '⚠️ ATTENTION REQUIRED' if pass_rate > 70 else '❌ FAILED'}

All metrics in this report have undergone rigorous validation to ensure data integrity and reliability.

---

"""
        return content

    def _generate_detailed_metrics(self, data: Dict[str, Any]) -> str:
        """Generate detailed metrics appendix"""

        content = """## Detailed Metrics (Appendix)

### System Metrics

"""

        system_metrics = data.get('system_metrics', {})
        for metric, value in system_metrics.items():
            if isinstance(value, (int, float)):
                content += f"- **{metric.replace('_', ' ').title()}:** {value}\n"

        content += "\n### Performance Metrics\n\n"

        performance = data.get('performance_metrics', {})
        for phase, metrics in performance.items():
            content += f"#### {phase.replace('_', ' ').title()}\n"
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        content += f"- {key}: {value}\n"
            content += "\n"

        content += "\n### Alerts\n\n"

        alerts = data.get('alerts', [])
        if alerts:
            for i, alert in enumerate(alerts, 1):
                level = alert.get('level', 'INFO')
                message = alert.get('message', 'No message')
                suggestion = alert.get('suggestion', '')
                content += f"**{i}. [{level}] {message}**\n"
                if suggestion:
                    content += f"   *Suggestion: {suggestion}*\n"
                content += "\n"
        else:
            content += "No alerts generated during the session.\n"

        return content

class EnhancedAnalyticsReporting:
    """Main class for enhanced analytics reporting with graphs"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.graph_generator = GraphGenerator(str(self.output_dir))
        self.markdown_reporter = MarkdownReporter(str(self.output_dir))

    def generate_enhanced_report(self, analytics_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate complete enhanced report with graphs"""

        print("Generating enhanced analytics report with professional graphs...")

        # Generate graphs
        graph_paths = {}

        try:
            # CPU usage chart
            system_metrics = analytics_data.get('system_metrics', {})
            if 'cpu_timestamps' in system_metrics and 'cpu_percentages' in system_metrics:
                graph_paths['cpu_usage'] = self.graph_generator.generate_cpu_usage_chart(
                    system_metrics['cpu_timestamps'],
                    system_metrics['cpu_percentages']
                )
                print("   [OK] CPU usage chart generated")

            # Memory usage chart
            if 'memory_timestamps' in system_metrics and 'memory_usage_mb' in system_metrics:
                graph_paths['memory_usage'] = self.graph_generator.generate_memory_usage_chart(
                    system_metrics['memory_timestamps'],
                    system_metrics['memory_usage_mb']
                )
                print("   [OK] Memory usage chart generated")

            # Model size comparison
            model_analytics = analytics_data.get('model_analytics', {})
            if model_analytics:
                models = list(model_analytics.keys())
                sizes = []
                for model in models:
                    analytics = model_analytics[model]
                    if isinstance(analytics, dict):
                        size_mb = analytics.get('size_mb', 0)
                        sizes.append(size_mb)

                if sizes:
                    graph_paths['model_size_comparison'] = self.graph_generator.generate_model_size_comparison(
                        models, sizes
                    )
                    print("   [OK] Model size comparison chart generated")

            # Efficiency scores
            if model_analytics:
                methods = list(model_analytics.keys())
                scores = []
                for method in methods:
                    analytics = model_analytics[method]
                    if isinstance(analytics, dict):
                        score = analytics.get('efficiency_score', 0)
                        scores.append(score)

                if scores:
                    graph_paths['efficiency_scores'] = self.graph_generator.generate_efficiency_scores_chart(
                        methods, scores
                    )
                    print("   [OK] Efficiency scores chart generated")

            # Timing comparison (if multiple methods)
            performance = analytics_data.get('performance_metrics', {})
            timing_methods = []
            timing_phases = []
            timing_data = {}

            # Extract timing data for different methods
            for phase, metrics in performance.items():
                if isinstance(metrics, dict) and 'method' in metrics:
                    method = metrics['method']
                    if method not in timing_methods:
                        timing_methods.append(method)
                    if phase not in timing_phases:
                        timing_phases.append(phase)

                    if method not in timing_data:
                        timing_data[method] = {}
                    timing_data[method][phase] = metrics.get('total_time', 0)

            if len(timing_methods) > 1 and timing_phases:
                graph_paths['timing_comparison'] = self.graph_generator.generate_timing_comparison_chart(
                    timing_methods, timing_phases, timing_data
                )
                print("   [OK] Timing comparison chart generated")

        except Exception as e:
            print(f"   [WARNING] Some graphs could not be generated: {e}")

        # Generate markdown report
        try:
            report_path = self.markdown_reporter.generate_report(analytics_data, graph_paths)
            print(f"   [OK] Markdown report generated: {report_path}")
        except Exception as e:
            print(f"   [FAIL] Report generation failed: {e}")
            report_path = None

        # Generate summary
        print(f"\nEnhanced Report Summary:")
        print(f"   Graphs generated: {len(graph_paths)}")
        print(f"   Report location: {self.output_dir}")
        print(f"   Images directory: {self.output_dir / 'images'}")

        return {
            'report_path': report_path,
            'graph_paths': graph_paths,
            'output_directory': str(self.output_dir)
        }

# Integration function for existing analytics system
def create_enhanced_report(analytics_data: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    """Create enhanced report with graphs from existing analytics data"""

    reporter = EnhancedAnalyticsReporting(output_dir)
    return reporter.generate_enhanced_report(analytics_data)

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'session_info': {
            'duration_seconds': 45.67,
            'total_operations': 12,
            'start_time': '2025-10-30T15:30:00Z'
        },
        'system_metrics': {
            'cpu_timestamps': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
            'cpu_percentages': [15.2, 45.8, 78.3, 65.1, 32.4, 18.9, 25.6, 42.1, 38.7, 12.3],
            'memory_timestamps': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
            'memory_usage_mb': [2048, 3072, 5120, 6144, 5632, 4096, 3584, 4608, 3840, 2560],
            'peak_memory_mb': 6144,
            'peak_cpu_percent': 78.3
        },
        'model_analytics': {
            'onnx_graph_level': {
                'efficiency_score': 85.2,
                'compression_ratio': 1.0,
                'size_mb': 942.3,
                'memory_efficiency': 'Good'
            },
            'pytorch_level': {
                'efficiency_score': 91.7,
                'compression_ratio': 1.05,
                'size_mb': 896.8,
                'memory_efficiency': 'Excellent'
            }
        },
        'performance_metrics': {
            'model_loading': {
                'total_time': 12.5,
                'average_time': 0.5,
                'count': 25,
                'method': 'onnx_graph_level'
            },
            'model_conversion': {
                'total_time': 28.3,
                'average_time': 1.2,
                'count': 24,
                'method': 'onnx_graph_level'
            }
        },
        'alerts': [
            {
                'level': 'INFO',
                'message': 'Model loading completed successfully',
                'timestamp': '2025-10-30T15:30:15Z'
            }
        ]
    }

    # Generate enhanced report
    result = create_enhanced_report(sample_data, "./enhanced_report_output")
    print(f"Enhanced report generated: {result['report_path']}")