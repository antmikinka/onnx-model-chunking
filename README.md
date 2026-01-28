# ONNX Model Chunking & Analytics System

An advanced toolkit designed for intelligent bisection, performance analytics, and quality assurance of large ONNX models. This system enables efficient deployment of massive models (like Qwen2, Llama, and Mistral) on memory-constrained devices through strategic graph-level chunking.

## 🚀 Key Features

- **Intelligent Graph Bisection**: Automatically identifies optimal split points in ONNX graphs based on weight distribution and layer boundaries.
- **Unified Configuration System**: A robust, validation-driven configuration framework with 100% backward compatibility for legacy implementations.
- **Advanced Performance Analytics**: Deep metrics including:
  - **Accuracy Verification**: PSNR (Peak Signal-to-Noise Ratio) calculation between original and chunked models.
  - **Memory Profiling**: Detailed analysis of weight distribution and memory footprint per chunk.
  - **Real-world Benchmarking**: Integration with Qwen2 and other LLMs for practical conversion testing.
- **Automated Quality Assurance**: Built-in QA system that generates comprehensive reports on chunk integrity and model performance.
- **Support for Multi-Method Chunking**: Infrastructure ready for both ONNX graph-level and PyTorch-level splitting strategies.

## 📁 Project Structure

- `builder.py`: Core model building engine and chunking integration.
- `chunking_config.py`: Unified configuration system and validation logic.
- `chunking_analytics.py`: Advanced metrics and analytics suite.
- `quality_assurance_system.py`: Automated QA and validation framework.
- `analytics/`: Modular analytics components.
- `docs/`: (See Markdown files in root for detailed guides)
  - `CHUNKING_DOCUMENTATION.md`: Core chunking logic and usage.
  - `UNIFIED_CONFIG_GUIDE.md`: Comprehensive guide to the configuration system.
  - `ANALYTICS_REPORTING_README.md`: Details on metrics and reporting.

## 🛠️ Getting Started

### Installation

Ensure you have the required dependencies (ONNX, ONNX Runtime, NumPy):

```bash
pip install onnx onnxruntime numpy
```

### Basic Usage (Unified Configuration)

```python
from builder import Model

extra_options = {
    "chunking": {
        "method": "onnx_graph_level",
        "onnx_graph_level": {
            "strategy": "weight_based",
            "distribution": "equal_memory",
            "num_chunks": 2
        },
        "common": {
            "output_directory": "./model_chunks",
            "verify_chunks": True,
            "min_psnr_threshold": 40.0
        }
    }
}

# Initialize and build
model = Model(config, "float16", "float16", "cpu", "./cache", extra_options)
model.make_model("path/to/large_model.onnx")
```

## 📊 Analytics & Reporting

The system generates detailed JSON metadata and human-readable markdown reports after each chunking operation, providing insights into:
- Split point indices and cumulative weight sizes.
- Boundary tensor identification.
- PSNR verification results.
- Node distribution per chunk.

## 🤝 Contributing

Contributions are welcome! Please ensure all new features include corresponding test cases in `tests/` and updated documentation.

---
*Developed for advanced ONNX model optimization and deployment.*
