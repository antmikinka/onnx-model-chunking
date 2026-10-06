# Analytics and Reporting System

**Outcome Statement:** After reading this section, you will understand how to monitor performance, collect metrics, and generate professional reports for model chunking operations.

---

## Analytics System Overview

The analytics system provides comprehensive monitoring and reporting for model chunking operations:

| Component | Purpose | Metrics Collected |
|-----------|---------|-------------------|
| **SystemMonitor** | Real-time resource monitoring | CPU, RAM, Disk I/O, GPU |
| **PerformanceTimer** | High-precision timing | Phase durations, operation timing |
| **ModelAnalyzer** | Model-specific metrics | Size ratios, efficiency scores |
| **AlertManager** | Threshold-based alerts | Performance warnings |

---

## Table of Contents

1. [Performance Metrics](#performance-metrics)
2. [Report Generation](#report-generation)
3. [Configuration](#configuration)

---

## Performance Metrics

**Outcome:** Understand what metrics are collected and how to interpret them.

### System Resource Metrics

| Metric | Description | Unit | Alert Threshold |
|--------|-------------|------|-----------------|
| `cpu_percent` | CPU utilization | % | > 90% |
| `memory_percent` | RAM utilization | % | > 80% |
| `memory_usage_gb` | Absolute RAM usage | GB | > 8.0 GB |
| `disk_usage_percent` | Disk utilization | % | > 90% |
| `disk_read_mb_s` | Disk read speed | MB/s | - |
| `disk_write_mb_s` | Disk write speed | MB/s | - |
| `gpu_percent` | GPU utilization | % | - |
| `gpu_memory_gb` | GPU memory usage | GB | - |

### Timing Metrics

| Metric | Description | Typical Value |
|--------|-------------|---------------|
| `model_loading_time` | Time to load source model | 1-10s |
| `architecture_detection_time` | Time to detect architecture | <1s |
| `splitting_time` | Time to create all components | 10-60s |
| `verification_time` | PSNR verification time | 5-30s |
| `total_processing_time` | End-to-end time | 30-120s |

### Model Analytics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `original_size_mb` | Source model size | Baseline |
| `chunked_total_size_mb` | Total size of all chunks | Should ≈ original |
| `largest_chunk_mb` | Size of largest component | Peak memory (sequential) |
| `memory_savings_percent` | Peak memory reduction | Higher = better |
| `psnr_db` | Accuracy verification | > 35 dB = good |
| `efficiency_score` | Overall efficiency (0-100) | Higher = better |

---

## Report Generation

**Outcome:** Understand how to generate and interpret analytics reports.

### Report Formats

| Format | Purpose | File Extension |
|--------|---------|----------------|
| **JSON** | Complete data, programmatic access | `.json` |
| **CSV** | Spreadsheet analysis | `.csv` |
| **HTML** | Visual report with charts | `.html` |
| **Summary** | Human-readable text | `.md` |

### Generating Reports

**Using Context Manager:**
```python
from chunking_analytics import monitor_chunking_performance

config = {
    "enable_monitoring": True,
    "real_time_updates": True,
    "output_format": ["json", "csv", "summary"]
}

with monitor_chunking_performance(config) as analytics:
    # Perform chunking operations here
    # Reports are automatically saved at exit
    pass
```

**Manual Report Generation:**
```python
from chunking_analytics import ChunkingAnalytics

# Create analytics instance
analytics = ChunkingAnalytics()
analytics.start_session()

# Monitor operations
with analytics.monitor_operation("model_loading", "loading"):
    # Load model here
    pass

with analytics.monitor_operation("chunking", "chunking"):
    # Perform chunking here
    pass

# Add model analysis
analytics.add_model_analysis(
    model_name="qwen2.5-0.5b",
    original_size_mb=780.0,
    chunk_count=31,
    chunk_sizes_mb=[519.0, 16.0, 16.0, ...],
    psnr_db=45.2
)

# End session and generate reports
analytics.end_session()
analytics.save_report("./output_directory")
```

### Sample JSON Report

```json
{
  "session_id": "20260321_143052_qwen2.5-0.5b",
  "timestamp": "2026-03-21T14:30:52Z",
  "model_info": {
    "model_name": "qwen2.5-0.5b",
    "original_size_mb": 780.0,
    "chunk_count": 31,
    "architecture": "qwen2"
  },
  "timing_metrics": {
    "model_loading": {"duration_s": 2.3, "phase": "loading"},
    "architecture_detection": {"duration_s": 0.1, "phase": "analysis"},
    "splitting": {"duration_s": 15.7, "phase": "chunking"},
    "verification": {"duration_s": 8.4, "phase": "verification"},
    "total": {"duration_s": 26.5, "phase": "total"}
  },
  "resource_metrics": {
    "peak_memory_gb": 4.2,
    "average_cpu_percent": 65.3,
    "peak_disk_read_mb_s": 450.0
  },
  "quality_metrics": {
    "psnr_db": 45.2,
    "efficiency_score": 87.5,
    "memory_savings_percent": 51.0
  },
  "alerts": []
}
```

### Sample Summary Report

```markdown
# Chunking Analytics Summary

## Session Information
- **Model:** qwen2.5-0.5b
- **Timestamp:** 2026-03-21 14:30:52
- **Architecture:** Qwen2 (28 layers)

## Performance Summary
- **Total Processing Time:** 26.5 seconds
- **Peak Memory Usage:** 4.2 GB
- **Average CPU Usage:** 65.3%

## Chunking Results
- **Original Size:** 780.0 MB
- **Total Chunked Size:** 1036.0 MB
- **Number of Chunks:** 31
- **Largest Chunk:** 519.0 MB (embeddings)

## Quality Verification
- **PSNR:** 45.2 dB (Excellent)
- **Memory Savings:** 51.0% reduction
- **Efficiency Score:** 87.5/100

## Recommendations
- Sequential mode will reduce peak memory to 519 MB
- Consider persistent mode for GPU deployment
- PSNR indicates production-ready accuracy
```

---

## Configuration

**Outcome:** Configure the analytics system for your needs.

### Analytics Configuration Options

```python
analytics_config = {
    # Monitoring settings
    "enable_monitoring": True,              # Enable/disable monitoring
    "sampling_interval": 1.0,               # Monitoring frequency (seconds)
    "max_history": 3600,                    # Maximum samples to keep

    # Resources to monitor
    "system_resources": ["cpu", "memory", "disk", "gpu"],

    # Timing settings
    "timing_phases": ["loading", "conversion", "chunking", "saving"],
    "detailed_metrics": True,               # Enable detailed metrics

    # Output settings
    "real_time_updates": True,              # Show real-time updates
    "output_format": ["json", "csv", "summary"],

    # Alert settings
    "enable_alerts": True,                  # Enable alert system
    "performance_alerts": {                 # Alert thresholds
        "memory_percent": 80.0,
        "cpu_percent": 90.0,
        "disk_usage_percent": 90.0,
        "conversion_time_seconds": 300.0,
        "memory_usage_gb": 8.0
    }
}
```

### Integration with Chunking

```python
config = {
    "chunking": {
        "method": "onnx_graph_level",
        "onnx_graph_level": {
            "strategy": "weight_based",
            "num_chunks": 2
        },
        "common": {
            "output_directory": "./chunked_models",
            "verify_chunks": True
        },
        "analytics": {
            "enable_monitoring": True,
            "detailed_metrics": True,
            "output_format": ["json", "summary"]
        }
    }
}
```

---

*For detailed analytics implementation, see `chunking_analytics.py` in the root directory.*
