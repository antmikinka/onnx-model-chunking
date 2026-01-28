"""
Comprehensive Analytics and Benchmarking System for Model Chunking

This module provides detailed performance metrics, system resource monitoring,
and benefit analysis for both ONNX graph-level and PyTorch-level chunking methods.

Features:
- Real-time system resource monitoring (CPU, memory, disk, GPU)
- Detailed performance timing analysis
- Model-specific analytics and comparisons
- Actionable benefit analysis
- Professional reporting capabilities
"""

import gc
import json
import os
import psutil
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    GPUtil = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    mdates = None

# Import enhanced reporting system
try:
    from analytics.reporting import GraphGenerator, MarkdownReporter
    ENHANCED_REPORTING_AVAILABLE = True
except ImportError:
    ENHANCED_REPORTING_AVAILABLE = False
    GraphGenerator = None
    MarkdownReporter = None


class MetricType(Enum):
    """Types of metrics collected by the analytics system"""
    TIMING = "timing"
    SYSTEM_RESOURCES = "system_resources"
    MODEL_ANALYTICS = "model_analytics"
    BENEFIT_ANALYSIS = "benefit_analysis"
    PERFORMANCE = "performance"


class AlertLevel(Enum):
    """Severity levels for performance alerts"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ResourceSnapshot:
    """Snapshot of system resource usage at a point in time"""
    timestamp: datetime
    cpu_percent: float
    cpu_freq_mhz: float
    memory_mb: float
    memory_percent: float
    disk_usage_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_sent_mb: float
    network_io_recv_mb: float
    gpu_info: List[Dict[str, Any]] = field(default_factory=list)
    thread_count: int = 0
    process_count: int = 0


@dataclass
class TimingMetric:
    """Detailed timing information for a specific operation"""
    name: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    phase: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelMetric:
    """Model-specific metrics and analytics"""
    model_name: str
    original_size_mb: float
    chunk_count: int
    chunk_sizes_mb: List[float]
    parameter_count: int
    memory_footprint_mb: float
    conversion_time_ms: float
    chunking_time_ms: float
    method: str
    compression_ratio: Optional[float] = None
    efficiency_score: Optional[float] = None


@dataclass
class PerformanceAlert:
    """Performance alert with contextual information"""
    level: AlertLevel
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    suggestions: List[str] = field(default_factory=list)


class SystemMonitor:
    """
    Real-time system resource monitoring with configurable sampling rates.

    Provides comprehensive monitoring of CPU, memory, disk, GPU, and network resources
    with efficient data collection and storage.
    """

    def __init__(self, sampling_interval: float = 1.0, max_history: int = 3600):
        """
        Initialize system monitor.

        Args:
            sampling_interval: Time between snapshots in seconds
            max_history: Maximum number of snapshots to keep in memory
        """
        self.sampling_interval = sampling_interval
        self.max_history = max_history
        self.is_monitoring = False
        self.monitor_thread = None
        self.snapshots = deque(maxlen=max_history)
        self.start_time = None
        self._stop_event = threading.Event()

        # System info collection
        self.process = psutil.Process()
        self.initial_disk_io = psutil.disk_io_counters()
        self.initial_network_io = psutil.net_io_counters()

    def start_monitoring(self):
        """Start background monitoring thread"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.start_time = datetime.now()
        self._stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop background monitoring thread"""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        self._stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

    def _monitor_loop(self):
        """Main monitoring loop running in background thread"""
        while not self._stop_event.is_set():
            try:
                snapshot = self._collect_snapshot()
                self.snapshots.append(snapshot)
                time.sleep(self.sampling_interval)
            except Exception as e:
                print(f"Warning: Error collecting system snapshot: {e}")
                time.sleep(self.sampling_interval)

    def _collect_snapshot(self) -> ResourceSnapshot:
        """Collect current system resource snapshot"""
        timestamp = datetime.now()

        # CPU information
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_freq = psutil.cpu_freq()
        cpu_freq_mhz = cpu_freq.current if cpu_freq else 0.0

        # Memory information
        memory = psutil.virtual_memory()
        memory_mb = memory.used / (1024 * 1024)
        memory_percent = memory.percent

        # Disk information
        disk_usage = psutil.disk_usage('/')
        disk_usage_percent = disk_usage.percent

        # I/O information
        disk_io = psutil.disk_io_counters()
        disk_io_read_mb = (disk_io.read_bytes - self.initial_disk_io.read_bytes) / (1024 * 1024)
        disk_io_write_mb = (disk_io.write_bytes - self.initial_disk_io.write_bytes) / (1024 * 1024)

        # Network information
        net_io = psutil.net_io_counters()
        net_io_sent_mb = (net_io.bytes_sent - self.initial_network_io.bytes_sent) / (1024 * 1024)
        net_io_recv_mb = (net_io.bytes_recv - self.initial_network_io.bytes_recv) / (1024 * 1024)

        # GPU information
        gpu_info = []
        if GPUTIL_AVAILABLE and GPUtil:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_info.append({
                        'id': gpu.id,
                        'name': gpu.name,
                        'load_percent': gpu.load * 100,
                        'memory_used_mb': gpu.memoryUsed,
                        'memory_total_mb': gpu.memoryTotal,
                        'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100 if gpu.memoryTotal > 0 else 0,
                        'temperature_c': gpu.temperature
                    })
            except Exception as e:
                pass  # GPU monitoring failed, continue without it

        # Process information
        thread_count = self.process.num_threads()
        process_count = len(psutil.pids())

        return ResourceSnapshot(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            cpu_freq_mhz=cpu_freq_mhz,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            disk_usage_percent=disk_usage_percent,
            disk_io_read_mb=disk_io_read_mb,
            disk_io_write_mb=disk_io_write_mb,
            network_io_sent_mb=net_io_sent_mb,
            network_io_recv_mb=net_io_recv_mb,
            gpu_info=gpu_info,
            thread_count=thread_count,
            process_count=process_count
        )

    def get_current_snapshot(self) -> Optional[ResourceSnapshot]:
        """Get the most recent snapshot"""
        return self.snapshots[-1] if self.snapshots else None

    def get_peak_usage(self) -> Dict[str, float]:
        """Get peak resource usage during monitoring"""
        if not self.snapshots:
            return {}

        peak_cpu = max(s.cpu_percent for s in self.snapshots)
        peak_memory = max(s.memory_mb for s in self.snapshots)
        peak_memory_percent = max(s.memory_percent for s in self.snapshots)
        peak_disk_read = max(s.disk_io_read_mb for s in self.snapshots)
        peak_disk_write = max(s.disk_io_write_mb for s in self.snapshots)

        peak_gpu = {}
        if self.snapshots and self.snapshots[0].gpu_info:
            for gpu_idx in range(len(self.snapshots[0].gpu_info)):
                gpu_loads = [s.gpu_info[gpu_idx]['load_percent']
                           for s in self.snapshots if gpu_idx < len(s.gpu_info)]
                if gpu_loads:
                    peak_gpu[f'gpu_{gpu_idx}_load'] = max(gpu_loads)

        return {
            'peak_cpu_percent': peak_cpu,
            'peak_memory_mb': peak_memory,
            'peak_memory_percent': peak_memory_percent,
            'peak_disk_read_mb': peak_disk_read,
            'peak_disk_write_mb': peak_disk_write,
            **peak_gpu
        }

    def get_average_usage(self) -> Dict[str, float]:
        """Get average resource usage during monitoring"""
        if not self.snapshots:
            return {}

        avg_cpu = sum(s.cpu_percent for s in self.snapshots) / len(self.snapshots)
        avg_memory = sum(s.memory_mb for s in self.snapshots) / len(self.snapshots)
        avg_memory_percent = sum(s.memory_percent for s in self.snapshots) / len(self.snapshots)

        return {
            'avg_cpu_percent': avg_cpu,
            'avg_memory_mb': avg_memory,
            'avg_memory_percent': avg_memory_percent,
            'total_samples': len(self.snapshots)
        }


class PerformanceTimer:
    """
    High-precision performance timing with context manager support.

    Provides detailed timing analysis for different phases of operations
with support for nested timing and metadata collection.
    """

    def __init__(self):
        self.timings: List[TimingMetric] = []
        self.active_timers: Dict[str, datetime] = {}
        self.timer_stack: List[Tuple[str, datetime]] = []

    @contextmanager
    def time_operation(self, name: str, phase: str = "unknown", **metadata):
        """
        Context manager for timing operations.

        Args:
            name: Name of the operation being timed
            phase: Phase of the operation (e.g., "loading", "conversion")
            **metadata: Additional metadata to store with the timing
        """
        start_time = datetime.now()
        timer_id = f"{phase}_{name}_{id(start_time)}"

        self.active_timers[timer_id] = start_time
        self.timer_stack.append((timer_id, start_time))

        try:
            yield start_time
        finally:
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000

            # Create timing metric
            timing = TimingMetric(
                name=name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                phase=phase,
                metadata=metadata
            )

            self.timings.append(timing)

            # Clean up
            self.active_timers.pop(timer_id, None)
            if self.timer_stack and self.timer_stack[-1][0] == timer_id:
                self.timer_stack.pop()

    def start_timer(self, name: str, phase: str = "unknown", **metadata):
        """Start a named timer"""
        start_time = datetime.now()
        timer_id = f"{phase}_{name}_{id(start_time)}"

        self.active_timers[timer_id] = {
            'start_time': start_time,
            'name': name,
            'phase': phase,
            'metadata': metadata
        }
        return timer_id

    def end_timer(self, timer_id: str) -> Optional[float]:
        """End a timer and return duration in milliseconds"""
        if timer_id not in self.active_timers:
            return None

        timer_info = self.active_timers.pop(timer_id)
        end_time = datetime.now()
        start_time = timer_info['start_time']
        duration_ms = (end_time - start_time).total_seconds() * 1000

        timing = TimingMetric(
            name=timer_info['name'],
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            phase=timer_info['phase'],
            metadata=timer_info['metadata']
        )

        self.timings.append(timing)
        return duration_ms

    def get_timing_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all timings"""
        if not self.timings:
            return {}

        summary = defaultdict(lambda: {'count': 0, 'total_ms': 0.0, 'min_ms': float('inf'), 'max_ms': 0.0})

        for timing in self.timings:
            key = f"{timing.phase}_{timing.name}"
            summary[key]['count'] += 1
            summary[key]['total_ms'] += timing.duration_ms
            summary[key]['min_ms'] = min(summary[key]['min_ms'], timing.duration_ms)
            summary[key]['max_ms'] = max(summary[key]['max_ms'], timing.duration_ms)

        # Calculate averages
        result = {}
        for key, stats in summary.items():
            result[key] = {
                'count': stats['count'],
                'total_ms': stats['total_ms'],
                'avg_ms': stats['total_ms'] / stats['count'],
                'min_ms': stats['min_ms'],
                'max_ms': stats['max_ms']
            }

        return result

    def get_phase_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get timing breakdown by phase"""
        if not self.timings:
            return {}

        phases = defaultdict(list)
        for timing in self.timings:
            phases[timing.phase].append(timing.duration_ms)

        breakdown = {}
        for phase, durations in phases.items():
            breakdown[phase] = {
                'total_ms': sum(durations),
                'avg_ms': sum(durations) / len(durations),
                'min_ms': min(durations),
                'max_ms': max(durations),
                'count': len(durations)
            }

        return breakdown


class ModelAnalyzer:
    """
    Model-specific analytics and benefit analysis.

    Provides detailed analysis of model characteristics, chunking efficiency,
    and performance improvements.
    """

    def __init__(self):
        self.model_metrics: List[ModelMetric] = []
        self.comparisons: List[Dict[str, Any]] = []

    def analyze_model(self, model_name: str, original_size_mb: float, chunk_count: int,
                     chunk_sizes_mb: List[float], parameter_count: int,
                     memory_footprint_mb: float, conversion_time_ms: float,
                     chunking_time_ms: float, method: str,
                     **kwargs) -> ModelMetric:
        """Analyze a chunked model and calculate efficiency metrics"""

        # Calculate compression ratio
        total_chunk_size = sum(chunk_sizes_mb)
        compression_ratio = original_size_mb / total_chunk_size if total_chunk_size > 0 else 1.0

        # Calculate efficiency score (0-100)
        # Based on memory efficiency, time efficiency, and size efficiency
        memory_efficiency = min(100, (memory_footprint_mb / original_size_mb) * 100) if original_size_mb > 0 else 0
        time_efficiency = max(0, 100 - (conversion_time_ms / 1000))  # Penalize long conversion times
        size_efficiency = min(100, compression_ratio * 100) if compression_ratio > 0 else 0

        efficiency_score = (memory_efficiency + time_efficiency + size_efficiency) / 3

        metric = ModelMetric(
            model_name=model_name,
            original_size_mb=original_size_mb,
            chunk_count=chunk_count,
            chunk_sizes_mb=chunk_sizes_mb,
            parameter_count=parameter_count,
            memory_footprint_mb=memory_footprint_mb,
            conversion_time_ms=conversion_time_ms,
            chunking_time_ms=chunking_time_ms,
            method=method,
            compression_ratio=compression_ratio,
            efficiency_score=efficiency_score
        )

        self.model_metrics.append(metric)
        return metric

    def compare_methods(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Compare different chunking methods for the same model"""
        metrics = [m for m in self.model_metrics if m.model_name == model_name]

        if len(metrics) < 2:
            return None

        comparison = {
            'model_name': model_name,
            'methods': {},
            'best_method': None,
            'comparisons': {}
        }

        # Analyze each method
        for metric in metrics:
            comparison['methods'][metric.method] = {
                'efficiency_score': metric.efficiency_score,
                'compression_ratio': metric.compression_ratio,
                'total_time_ms': metric.conversion_time_ms + metric.chunking_time_ms,
                'memory_footprint_mb': metric.memory_footprint_mb,
                'chunk_count': metric.chunk_count
            }

        # Find best method by efficiency score
        best_method = max(metrics, key=lambda m: m.efficiency_score)
        comparison['best_method'] = best_method.method

        # Calculate relative improvements
        methods = list(comparison['methods'].keys())
        if len(methods) >= 2:
            method1, method2 = methods[0], methods[1]
            m1_data = comparison['methods'][method1]
            m2_data = comparison['methods'][method2]

            comparison['comparisons'][f'{method1}_vs_{method2}'] = {
                'efficiency_improvement_percent': ((m1_data['efficiency_score'] - m2_data['efficiency_score']) / m2_data['efficiency_score']) * 100 if m2_data['efficiency_score'] > 0 else 0,
                'time_improvement_percent': ((m2_data['total_time_ms'] - m1_data['total_time_ms']) / m2_data['total_time_ms']) * 100 if m2_data['total_time_ms'] > 0 else 0,
                'memory_improvement_percent': ((m2_data['memory_footprint_mb'] - m1_data['memory_footprint_mb']) / m2_data['memory_footprint_mb']) * 100 if m2_data['memory_footprint_mb'] > 0 else 0
            }

        self.comparisons.append(comparison)
        return comparison


class AlertManager:
    """
    Performance alert management with configurable thresholds.

    Monitors metrics in real-time and generates alerts when thresholds are exceeded.
    """

    def __init__(self):
        self.alerts: List[PerformanceAlert] = []
        self.thresholds = {
            'memory_percent': 80.0,
            'cpu_percent': 90.0,
            'disk_usage_percent': 90.0,
            'conversion_time_seconds': 300.0,
            'memory_usage_gb': 8.0
        }

    def set_threshold(self, metric_name: str, threshold_value: float):
        """Set alert threshold for a specific metric"""
        self.thresholds[metric_name] = threshold_value

    def check_thresholds(self, metrics: Dict[str, Any], timestamp: Optional[datetime] = None) -> List[PerformanceAlert]:
        """Check metrics against thresholds and generate alerts"""
        if timestamp is None:
            timestamp = datetime.now()

        new_alerts = []

        # Check memory usage
        if 'memory_percent' in metrics and metrics['memory_percent'] > self.thresholds['memory_percent']:
            alert = PerformanceAlert(
                level=AlertLevel.WARNING,
                message=f"High memory usage: {metrics['memory_percent']:.1f}%",
                metric_name="memory_percent",
                current_value=metrics['memory_percent'],
                threshold_value=self.thresholds['memory_percent'],
                timestamp=timestamp,
                suggestions=[
                    "Consider reducing model size or chunk size",
                    "Close other memory-intensive applications",
                    "Increase system RAM if possible"
                ]
            )
            new_alerts.append(alert)

        # Check CPU usage
        if 'cpu_percent' in metrics and metrics['cpu_percent'] > self.thresholds['cpu_percent']:
            alert = PerformanceAlert(
                level=AlertLevel.WARNING,
                message=f"High CPU usage: {metrics['cpu_percent']:.1f}%",
                metric_name="cpu_percent",
                current_value=metrics['cpu_percent'],
                threshold_value=self.thresholds['cpu_percent'],
                timestamp=timestamp,
                suggestions=[
                    "Check for other CPU-intensive processes",
                    "Consider reducing parallel processing",
                    "Monitor system temperature"
                ]
            )
            new_alerts.append(alert)

        # Check disk usage
        if 'disk_usage_percent' in metrics and metrics['disk_usage_percent'] > self.thresholds['disk_usage_percent']:
            alert = PerformanceAlert(
                level=AlertLevel.ERROR,
                message=f"Low disk space: {metrics['disk_usage_percent']:.1f}% used",
                metric_name="disk_usage_percent",
                current_value=metrics['disk_usage_percent'],
                threshold_value=self.thresholds['disk_usage_percent'],
                timestamp=timestamp,
                suggestions=[
                    "Free up disk space by removing unnecessary files",
                    "Consider using external storage",
                    "Archive old model files"
                ]
            )
            new_alerts.append(alert)

        self.alerts.extend(new_alerts)
        return new_alerts

    def get_alert_summary(self) -> Dict[str, int]:
        """Get summary of alerts by level"""
        summary = {level.value: 0 for level in AlertLevel}
        for alert in self.alerts:
            summary[alert.level.value] += 1
        return summary


class ChunkingAnalytics:
    """
    Main analytics orchestrator for the chunking system.

    Coordinates system monitoring, performance timing, model analysis,
    and alert management to provide comprehensive analytics.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize chunking analytics system.

        Args:
            config: Analytics configuration dictionary
        """
        self.config = config or self._get_default_config()

        # Initialize components
        self.system_monitor = SystemMonitor(
            sampling_interval=self.config.get('sampling_interval', 1.0),
            max_history=self.config.get('max_history', 3600)
        )

        self.performance_timer = PerformanceTimer()
        self.model_analyzer = ModelAnalyzer()
        self.alert_manager = AlertManager()

        # Enhanced reporting components
        self.graph_generator = None
        self.markdown_reporter = None
        if ENHANCED_REPORTING_AVAILABLE and self.config.get('enhanced_reporting', {}).get('enabled', False):
            output_dir = self.config['enhanced_reporting'].get('output_dir', 'analytics_reports')
            self.graph_generator = GraphGenerator(
                output_dir=f"{output_dir}/graphs",
                dpi=self.config['enhanced_reporting'].get('dpi', 300)
            )
            self.markdown_reporter = MarkdownReporter(
                output_dir=output_dir,
                graph_generator=self.graph_generator
            )

        # Analytics state
        self.is_active = False
        self.session_start = None
        self.session_end = None
        self.current_operation = None

        # Configure thresholds
        if 'performance_alerts' in self.config:
            for metric, threshold in self.config['performance_alerts'].items():
                self.alert_manager.set_threshold(metric, threshold)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default analytics configuration"""
        return {
            "enable_monitoring": True,
            "sampling_interval": 1.0,
            "max_history": 3600,
            "system_resources": ["cpu", "memory", "disk", "gpu"],
            "timing_phases": ["loading", "conversion", "chunking", "saving"],
            "detailed_metrics": True,
            "real_time_updates": True,
            "output_format": ["json", "csv", "summary"],
            "performance_alerts": {
                "memory_percent": 80.0,
                "cpu_percent": 90.0,
                "disk_usage_percent": 90.0,
                "conversion_time_seconds": 300.0,
                "memory_usage_gb": 8.0
            },
            "save_snapshots": False,
            "snapshot_interval": 10,
            "enhanced_reporting": {
                "enabled": False,
                "output_dir": "analytics_reports",
                "include_graphs": True,
                "graph_style": "professional",
                "output_images": True,
                "image_format": "png",
                "dpi": 300,
                "executive_summary": True,
                "report_formats": ["markdown", "json"]
            }
        }

    def start_session(self):
        """Start analytics monitoring session"""
        if self.is_active:
            return

        self.is_active = True
        self.session_start = datetime.now()

        if self.config.get('enable_monitoring', True):
            self.system_monitor.start_monitoring()

        if self.config.get('real_time_updates', False):
            print(f"Analytics session started at {self.session_start}")

    def end_session(self):
        """End analytics monitoring session"""
        if not self.is_active:
            return

        self.session_end = datetime.now()
        self.is_active = False

        self.system_monitor.stop_monitoring()

        if self.config.get('real_time_updates', False):
            duration = self.session_end - self.session_start
            print(f"Analytics session ended at {self.session_end} (duration: {duration})")

    @contextmanager
    def monitor_operation(self, operation_name: str, phase: str = "unknown", **metadata):
        """
        Context manager for monitoring an operation with comprehensive metrics.

        Args:
            operation_name: Name of the operation being monitored
            phase: Phase of the operation
            **metadata: Additional metadata
        """
        if not self.is_active:
            self.start_session()

        self.current_operation = operation_name

        # Check for alerts before operation
        if self.config.get('enable_alerts', True):
            current_snapshot = self.system_monitor.get_current_snapshot()
            if current_snapshot:
                metrics = {
                    'memory_percent': current_snapshot.memory_percent,
                    'cpu_percent': current_snapshot.cpu_percent,
                    'disk_usage_percent': current_snapshot.disk_usage_percent
                }
                alerts = self.alert_manager.check_thresholds(metrics, current_snapshot.timestamp)
                for alert in alerts:
                    self._handle_alert(alert)

        # Time the operation
        with self.performance_timer.time_operation(operation_name, phase, **metadata):
            try:
                yield
            except Exception as e:
                # Log error metrics
                with self.performance_timer.time_operation(f"{operation_name}_error", phase, error=str(e)):
                    pass
                raise

        self.current_operation = None

    def _handle_alert(self, alert: PerformanceAlert):
        """Handle performance alert"""
        if self.config.get('real_time_updates', False):
            alert_symbols = {"info": "[INFO]", "warning": "[WARN]", "error": "[ERROR]", "critical": "[CRITICAL]"}
            symbol = alert_symbols.get(alert.level.value, "[ALERT]")
            print(f"{symbol} {alert.level.value.upper()}: {alert.message}")

            if alert.suggestions:
                for suggestion in alert.suggestions:
                    print(f"   [SUGGESTION] {suggestion}")

    def add_model_analysis(self, **kwargs):
        """Add model analysis to the analytics session"""
        return self.model_analyzer.analyze_model(**kwargs)

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        if not self.session_start:
            return {"error": "No active session"}

        session_end = self.session_end or datetime.now()
        session_duration = session_end - self.session_start

        # System resource summary
        peak_usage = self.system_monitor.get_peak_usage()
        avg_usage = self.system_monitor.get_average_usage()

        # Performance timing summary
        timing_summary = self.performance_timer.get_timing_summary()
        phase_breakdown = self.performance_timer.get_phase_breakdown()

        # Model analysis summary
        model_summary = {}
        for metric in self.model_analyzer.model_metrics:
            model_summary[metric.model_name] = {
                'method': metric.method,
                'efficiency_score': metric.efficiency_score,
                'compression_ratio': metric.compression_ratio,
                'chunk_count': metric.chunk_count,
                'original_size_mb': metric.original_size_mb,
                'total_chunk_size_mb': sum(metric.chunk_sizes_mb)
            }

        # Alert summary
        alert_summary = self.alert_manager.get_alert_summary()

        # Calculate overall metrics
        total_operation_time = sum(
            summary['total_ms'] for summary in timing_summary.values()
        ) if timing_summary else 0

        report = {
            "session_info": {
                "start_time": self.session_start.isoformat(),
                "end_time": session_end.isoformat(),
                "duration_seconds": session_duration.total_seconds(),
                "total_operations": len(self.performance_timer.timings)
            },
            "system_resources": {
                "peak_usage": peak_usage,
                "average_usage": avg_usage,
                "samples_collected": len(self.system_monitor.snapshots)
            },
            "performance_timing": {
                "summary": timing_summary,
                "phase_breakdown": phase_breakdown,
                "total_operation_time_ms": total_operation_time
            },
            "model_analysis": {
                "models_analyzed": model_summary,
                "comparisons": self.model_analyzer.comparisons[-1] if self.model_analyzer.comparisons else None
            },
            "alerts": {
                "summary": alert_summary,
                "total_alerts": sum(alert_summary.values()),
                "recent_alerts": [
                    {
                        "level": alert.level.value,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    for alert in self.alert_manager.alerts[-5:]  # Last 5 alerts
                ]
            },
            "recommendations": self._generate_recommendations()
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analytics data"""
        recommendations = []

        # Analyze memory usage
        peak_usage = self.system_monitor.get_peak_usage()
        if peak_usage.get('peak_memory_percent', 0) > 80:
            recommendations.append("Consider using smaller chunk sizes to reduce memory pressure")

        # Analyze timing efficiency
        phase_breakdown = self.performance_timer.get_phase_breakdown()
        if 'conversion' in phase_breakdown:
            conversion_time = phase_breakdown['conversion']['total_ms'] / 1000
            if conversion_time > 300:  # 5 minutes
                recommendations.append("Conversion time is high. Consider optimizing model structure or reducing complexity")

        # Analyze model efficiency
        if self.model_analyzer.model_metrics:
            best_metric = max(self.model_analyzer.model_metrics, key=lambda m: m.efficiency_score)
            if best_metric.efficiency_score < 50:
                recommendations.append(f"Current chunking method shows low efficiency ({best_metric.efficiency_score:.1f}/100). Consider alternative chunking strategies")

        # Add method-specific recommendations
        if self.model_analyzer.comparisons:
            comparison = self.model_analyzer.comparisons[-1]
            if comparison.get('best_method'):
                best_method = comparison['best_method']
                recommendations.append(f"Based on analysis, '{best_method}' shows the best performance for this model type")

        return recommendations

    def save_report(self, output_dir: str, formats: List[str] = None) -> Dict[str, str]:
        """
        Save analytics report in specified formats.

        Args:
            output_dir: Directory to save reports
            formats: List of formats to save (json, csv, html)

        Returns:
            Dictionary mapping format to file path
        """
        if formats is None:
            formats = self.config.get('output_format', ['json'])

        os.makedirs(output_dir, exist_ok=True)
        report = self.get_comprehensive_report()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}

        # JSON format
        if 'json' in formats:
            json_file = os.path.join(output_dir, f"chunking_analytics_{timestamp}.json")
            with open(json_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            saved_files['json'] = json_file

        # CSV format (if pandas is available)
        if 'csv' in formats and PANDAS_AVAILABLE:
            try:
                csv_file = os.path.join(output_dir, f"chunking_analytics_{timestamp}.csv")

                # Flatten data for CSV
                flattened_data = []

                # System resource data
                if 'system_resources' in report and 'samples_collected' in report['system_resources']:
                    peak = report['system_resources']['peak_usage']
                    avg = report['system_resources']['average_usage']
                    flattened_data.append({
                        'category': 'system_resources',
                        'metric': 'peak_memory_mb',
                        'value': peak.get('peak_memory_mb', 0)
                    })
                    flattened_data.append({
                        'category': 'system_resources',
                        'metric': 'avg_memory_mb',
                        'value': avg.get('avg_memory_mb', 0)
                    })

                # Timing data
                if 'performance_timing' in report and 'phase_breakdown' in report['performance_timing']:
                    for phase, data in report['performance_timing']['phase_breakdown'].items():
                        flattened_data.append({
                            'category': 'timing',
                            'metric': f'{phase}_total_ms',
                            'value': data['total_ms']
                        })
                        flattened_data.append({
                            'category': 'timing',
                            'metric': f'{phase}_avg_ms',
                            'value': data['avg_ms']
                        })

                df = pd.DataFrame(flattened_data)
                df.to_csv(csv_file, index=False)
                saved_files['csv'] = csv_file

            except Exception as e:
                print(f"Warning: Could not save CSV report: {e}")

        # Summary text format
        if 'summary' in formats:
            summary_file = os.path.join(output_dir, f"chunking_analytics_summary_{timestamp}.txt")
            with open(summary_file, 'w') as f:
                f.write("=== CHUNKING ANALYTICS SUMMARY ===\n\n")

                # Session info
                if 'session_info' in report:
                    session = report['session_info']
                    f.write(f"Session Duration: {session['duration_seconds']:.2f} seconds\n")
                    f.write(f"Total Operations: {session['total_operations']}\n\n")

                # System resources
                if 'system_resources' in report:
                    f.write("SYSTEM RESOURCES:\n")
                    peak = report['system_resources']['peak_usage']
                    f.write(f"  Peak Memory: {peak.get('peak_memory_mb', 0):.1f} MB\n")
                    f.write(f"  Peak CPU: {peak.get('peak_cpu_percent', 0):.1f}%\n\n")

                # Performance timing
                if 'performance_timing' in report and 'phase_breakdown' in report['performance_timing']:
                    f.write("PERFORMANCE BREAKDOWN:\n")
                    for phase, data in report['performance_timing']['phase_breakdown'].items():
                        f.write(f"  {phase.capitalize()}: {data['total_ms']:.1f} ms total, {data['avg_ms']:.1f} ms avg\n")
                    f.write("\n")

                # Recommendations
                if 'recommendations' in report:
                    f.write("RECOMMENDATIONS:\n")
                    for i, rec in enumerate(report['recommendations'], 1):
                        f.write(f"  {i}. {rec}\n")

            saved_files['summary'] = summary_file

        return saved_files

    def save_enhanced_reports(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Save enhanced professional reports with graphs and markdown formatting.

        Args:
            output_dir: Directory to save reports (uses config default if None)

        Returns:
            Dictionary mapping format to file path
        """
        if not ENHANCED_REPORTING_AVAILABLE:
            raise ImportError("Enhanced reporting requires matplotlib and the analytics.reporting module")

        if not self.markdown_reporter:
            raise RuntimeError("Enhanced reporting not enabled. Check configuration.")

        if output_dir is None:
            output_dir = self.config['enhanced_reporting'].get('output_dir', 'analytics_reports')

        os.makedirs(output_dir, exist_ok=True)

        # Get comprehensive analytics data
        report = self.get_comprehensive_report()

        # Prepare data for visualization
        enhanced_report_data = self._prepare_enhanced_report_data(report)

        saved_files = {}

        # Generate markdown report with graphs
        if self.config['enhanced_reporting'].get('report_formats', ['markdown'])[0] == 'markdown':
            try:
                markdown_path = self.markdown_reporter.generate_report(
                    enhanced_report_data,
                    include_graphs=self.config['enhanced_reporting'].get('include_graphs', True),
                    include_executive_summary=self.config['enhanced_reporting'].get('executive_summary', True)
                )
                saved_files['markdown'] = markdown_path
            except Exception as e:
                print(f"Warning: Could not generate markdown report: {e}")

        # Generate JSON report
        if 'json' in self.config['enhanced_reporting'].get('report_formats', ['markdown']):
            try:
                json_path = self.markdown_reporter.generate_json_report(enhanced_report_data)
                saved_files['json'] = json_path
            except Exception as e:
                print(f"Warning: Could not generate JSON report: {e}")

        return saved_files

    def _prepare_enhanced_report_data(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare analytics data for enhanced reporting with graph generation.

        Args:
            report: Comprehensive analytics report

        Returns:
            Enhanced report data with visualization-ready formats
        """
        enhanced_data = report.copy()

        # Convert system resource data to time-series format for graphs
        if self.system_monitor.snapshots and ENHANCED_REPORTING_AVAILABLE:
            snapshots = list(self.system_monitor.snapshots)

            # Prepare time-series data
            cpu_data = [(s.timestamp, s.cpu_percent) for s in snapshots]
            memory_data = [(s.timestamp, s.memory_mb) for s in snapshots]
            disk_data = [(s.timestamp, s.disk_io_read_mb + s.disk_io_write_mb) for s in snapshots]

            enhanced_data['time_series_data'] = {
                'cpu_usage': cpu_data,
                'memory_usage': memory_data,
                'disk_io': disk_data
            }

            # Generate system resource graphs
            if self.graph_generator and self.config['enhanced_reporting'].get('include_graphs', True):
                try:
                    # System resource dashboard
                    dashboard_path = self.graph_generator.create_system_resource_dashboard(
                        cpu_data=cpu_data,
                        memory_data=memory_data,
                        disk_data=disk_data if any(v > 0 for _, v in disk_data) else None,
                        title="System Resource Utilization During Chunking"
                    )
                    enhanced_data['graphs'] = enhanced_data.get('graphs', {})
                    enhanced_data['graphs']['system_dashboard'] = dashboard_path

                    # Phase timing breakdown graph
                    if 'performance_timing' in report and 'phase_breakdown' in report['performance_timing']:
                        phase_data = report['performance_timing']['phase_breakdown']
                        if phase_data:
                            # Convert to format suitable for bar chart
                            phase_chart_data = {phase: data['total_ms'] for phase, data in phase_data.items()}
                            timing_chart_path = self.graph_generator.create_bar_chart(
                                data=phase_chart_data,
                                title="Phase Timing Breakdown",
                                xlabel="Processing Phase",
                                ylabel="Time (ms)",
                                orientation="vertical"
                            )
                            enhanced_data['graphs']['phase_timing'] = timing_chart_path

                    # Model efficiency comparison graph
                    if 'model_analysis' in report and 'models_analyzed' in report['model_analysis']:
                        models = report['model_analysis']['models_analyzed']
                        if models:
                            # Create efficiency comparison chart
                            efficiency_data = {}
                            for model_name, metrics in models.items():
                                method = metrics.get('method', model_name)
                                efficiency_data[method] = metrics.get('efficiency_score', 0)

                            if efficiency_data:
                                efficiency_chart_path = self.graph_generator.create_bar_chart(
                                    data=efficiency_data,
                                    title="Method Efficiency Comparison",
                                    xlabel="Chunking Method",
                                    ylabel="Efficiency Score",
                                    orientation="horizontal"
                                )
                                enhanced_data['graphs']['efficiency_comparison'] = efficiency_chart_path

                except Exception as e:
                    print(f"Warning: Could not generate graphs: {e}")

        return enhanced_data


# Convenience functions for backward compatibility
def create_analytics_session(config: Optional[Dict[str, Any]] = None) -> ChunkingAnalytics:
    """Create a new analytics session with optional configuration"""
    return ChunkingAnalytics(config)


@contextmanager
def monitor_chunking_performance(config: Optional[Dict[str, Any]] = None):
    """Context manager for monitoring chunking performance"""
    analytics = ChunkingAnalytics(config)
    analytics.start_session()

    try:
        yield analytics
    finally:
        analytics.end_session()