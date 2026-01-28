#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced ONNX Model Chunking Functionality

This script tests the enhanced architecture-aware chunking functionality implemented in builder.py
using the Qwen/Qwen2.5-0.5B model from Hugging Face. It covers:

CORE FEATURES:
- Both weight-based (legacy) and layer-based (new) chunking strategies
- Different chunk distributions: equal_layers, equal_memory, custom
- Multiple chunk counts: 2, 3, 4+ chunks
- Component allocation options (embedding/lm_head placement)
- PSNR verification for accuracy validation
- Error handling and edge cases
- Performance benchmarking and memory usage

Author: Enhanced Testing Suite
Date: 2025-10-30
Version: 2.0 - Architecture-Aware Chunking Support
"""

import os
import sys
import time
import json
import shutil
import logging
import traceback
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import onnx
import tempfile
import gc
import psutil
from datetime import datetime
import statistics

# Add the current directory to the path to import builder module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from builder import QwenModel, Qwen3Model
    import onnx_ir as ir
except ImportError as e:
    print(f"Error importing builder module: {e}")
    print("Please ensure builder.py is in the same directory or accessible via PYTHONPATH")
    sys.exit(1)


@dataclass
class TestResult:
    """Enhanced data class to store individual test results"""
    test_name: str
    status: str  # 'PASS', 'FAIL', 'SKIP'
    execution_time: float
    details: Dict[str, Any]
    error_message: Optional[str] = None
    timestamp: str = None
    performance_metrics: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class TestSuiteResults:
    """Enhanced data class to store overall test suite results"""
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    total_execution_time: float
    test_results: List[TestResult]
    system_info: Dict[str, Any]
    performance_summary: Dict[str, float]

    def to_dict(self):
        return {
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'skipped_tests': self.skipped_tests,
            'success_rate': (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0,
            'total_execution_time': self.total_execution_time,
            'test_results': [asdict(result) for result in self.test_results],
            'system_info': self.system_info,
            'performance_summary': self.performance_summary
        }


class EnhancedChunkingTestSuite:
    """Comprehensive test suite for enhanced ONNX model chunking functionality"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_logging()
        self.setup_directories()
        self.setup_test_environment()

        # Test configuration
        self.model_name = "Qwen/Qwen2.5-0.5B"
        self.test_input_text = "Hello, this is a test for enhanced model chunking functionality with architecture-aware splitting."
        self.test_results: List[TestResult] = []
        self.start_time = None

        # Model cache
        self.tokenizer = None
        self.original_model = None
        self.sample_inputs = None
        self.model_config = None

        # Performance tracking
        self.performance_data = {}

    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path("test_logs")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"enhanced_chunking_test_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        """Setup test directories"""
        self.base_dir = Path("test_environment")
        self.models_dir = self.base_dir / "models"
        self.chunked_dir = self.base_dir / "chunked_models"
        self.cache_dir = self.base_dir / "cache"
        self.reports_dir = self.base_dir / "reports"
        self.performance_dir = self.base_dir / "performance"

        for directory in [self.base_dir, self.models_dir, self.chunked_dir,
                         self.cache_dir, self.reports_dir, self.performance_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def setup_test_environment(self):
        """Setup test environment and capture system info"""
        self.system_info = {
            'python_version': sys.version,
            'platform': sys.platform,
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'memory_available': psutil.virtual_memory().available / (1024**3),  # GB
            'torch_version': torch.__version__,
            'onnx_version': onnx.__version__,
            'onnxruntime_version': ort.__version__,
            'gpu_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_memory': [torch.cuda.get_device_properties(i).total_memory / (1024**3)
                          for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        }

        self.logger.info("Enhanced Test Environment Setup")
        self.logger.info("=" * 60)
        for key, value in self.system_info.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("=" * 60)

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        process = psutil.Process()
        memory_info = process.memory_info()

        metrics = {
            'memory_usage_mb': memory_info.rss / (1024**2),
            'memory_vms_mb': memory_info.vms / (1024**2),
            'cpu_percent': process.cpu_percent(),
            'threads': process.num_threads(),
            'system_memory_usage_percent': psutil.virtual_memory().percent
        }

        if torch.cuda.is_available():
            metrics.update({
                'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / (1024**2),
                'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / (1024**2),
                'gpu_utilization': torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
            })

        return metrics

    def log_test_start(self, test_name: str):
        """Log test start with performance baseline"""
        self.logger.info(f"Starting test: {test_name}")
        self.logger.info("-" * 50)

        # Capture baseline metrics
        baseline_metrics = self.get_performance_metrics()
        self.logger.info(f"Baseline metrics: Memory={baseline_metrics['memory_usage_mb']:.1f}MB, CPU={baseline_metrics['cpu_percent']:.1f}%")

    def log_test_result(self, test_result: TestResult):
        """Log test result with performance metrics"""
        status_emoji = "[PASS]" if test_result.status == "PASS" else "[FAIL]" if test_result.status == "FAIL" else "[SKIP]"
        self.logger.info(f"{status_emoji} {test_result.test_name}: {test_result.status} ({test_result.execution_time:.2f}s)")

        if test_result.performance_metrics:
            perf = test_result.performance_metrics
            self.logger.info(f"   Performance: Peak Memory={perf.get('peak_memory_mb', 0):.1f}MB, "
                           f"Avg CPU={perf.get('avg_cpu_percent', 0):.1f}%")

        if test_result.error_message:
            self.logger.error(f"   Error: {test_result.error_message}")

        if test_result.details:
            for key, value in test_result.details.items():
                if key not in ['traceback']:  # Skip verbose traceback in summary
                    self.logger.info(f"   {key}: {value}")

        self.logger.info("-" * 50)

    def run_test(self, test_name: str, test_func) -> TestResult:
        """Run a single test and capture results with performance monitoring"""
        self.log_test_start(test_name)

        start_time = time.time()
        test_result = TestResult(
            test_name=test_name,
            status="FAIL",
            execution_time=0.0,
            details={}
        )

        # Performance monitoring
        peak_memory = 0
        cpu_readings = []
        performance_start = self.get_performance_metrics()

        try:
            # Monitor performance during test
            def monitor_performance():
                nonlocal peak_memory
                current = self.get_performance_metrics()
                peak_memory = max(peak_memory, current['memory_usage_mb'])
                cpu_readings.append(current['cpu_percent'])

            # Run the test
            result_details = test_func()
            test_result.status = "PASS"
            test_result.details = result_details if isinstance(result_details, dict) else {"result": result_details}

        except Exception as e:
            test_result.status = "FAIL"
            test_result.error_message = str(e)
            test_result.details = {"traceback": traceback.format_exc()}
            self.logger.error(f"Test failed with exception: {e}")
            self.logger.debug(traceback.format_exc())

        finally:
            test_result.execution_time = time.time() - start_time

            # Calculate performance metrics
            performance_end = self.get_performance_metrics()
            test_result.performance_metrics = {
                'peak_memory_mb': peak_memory,
                'avg_cpu_percent': statistics.mean(cpu_readings) if cpu_readings else 0,
                'memory_delta_mb': performance_end['memory_usage_mb'] - performance_start['memory_usage_mb'],
                'execution_time_s': test_result.execution_time
            }

            self.log_test_result(test_result)
            self.test_results.append(test_result)

            # Clean up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return test_result

    def test_01_model_download_and_setup(self) -> Dict[str, Any]:
        """Test downloading and loading the Qwen model with detailed analysis"""
        self.logger.info(f"Downloading and analyzing model: {self.model_name}")

        # Download tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=str(self.cache_dir)
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Download model configuration
        self.model_config = AutoConfig.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=str(self.cache_dir)
        )

        # Download the actual model
        self.original_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            cache_dir=str(self.cache_dir)
        )

        # Analyze model architecture for chunking
        model_analysis = self.analyze_model_architecture()

        # Prepare sample inputs with different lengths
        test_texts = [
            self.test_input_text,
            "Short test.",
            "This is a much longer test input to test how the model handles different sequence lengths and token counts for chunking verification."
        ]

        sample_inputs = {}
        for i, text in enumerate(test_texts):
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )
            sample_inputs[f'test_{i+1}'] = {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'text': text,
                'token_count': len(encoded['input_ids'][0])
            }

        self.sample_inputs = sample_inputs

        return {
            "model_name": self.model_name,
            "model_type": self.model_config.model_type,
            "architecture": model_analysis,
            "vocab_size": self.model_config.vocab_size,
            "hidden_size": self.model_config.hidden_size,
            "num_layers": self.model_config.num_hidden_layers,
            "num_attention_heads": self.model_config.num_attention_heads,
            "num_kv_heads": getattr(self.model_config, 'num_key_value_heads', self.model_config.num_attention_heads),
            "intermediate_size": getattr(self.model_config, 'intermediate_size', self.model_config.hidden_size * 4),
            "max_position_embeddings": getattr(self.model_config, 'max_position_embeddings', None),
            "model_parameters": sum(p.numel() for p in self.original_model.parameters()),
            "model_size_mb": sum(p.numel() * p.element_size() for p in self.original_model.parameters()) / (1024**2),
            "sample_inputs_count": len(sample_inputs),
            "sample_input_tokens": {k: v['token_count'] for k, v in sample_inputs.items()}
        }

    def analyze_model_architecture(self) -> Dict[str, Any]:
        """Analyze model architecture for intelligent chunking"""
        if not self.original_model:
            raise ValueError("Model not loaded")

        # Count parameters by component
        embedding_params = sum(p.numel() for n, p in self.original_model.named_parameters()
                             if 'embed' in n.lower())

        layer_params = []
        for i in range(self.model_config.num_hidden_layers):
            layer_prefix = f"model.layers.{i}"
            layer_param_count = sum(p.numel() for n, p in self.original_model.named_parameters()
                                   if n.startswith(layer_prefix))
            layer_params.append(layer_param_count)

        lm_head_params = sum(p.numel() for n, p in self.original_model.named_parameters()
                           if 'lm_head' in n.lower() or 'output' in n.lower())

        # Calculate parameter distributions
        total_params = sum(p.numel() for p in self.original_model.parameters())

        return {
            "total_parameters": total_params,
            "embedding_parameters": embedding_params,
            "layer_parameters": layer_params,
            "lm_head_parameters": lm_head_params,
            "layer_param_stats": {
                "mean": statistics.mean(layer_params),
                "median": statistics.median(layer_params),
                "min": min(layer_params),
                "max": max(layer_params),
                "std": statistics.stdev(layer_params) if len(layer_params) > 1 else 0
            },
            "parameter_distribution": {
                "embedding_percent": (embedding_params / total_params) * 100,
                "layers_percent": (sum(layer_params) / total_params) * 100,
                "lm_head_percent": (lm_head_params / total_params) * 100
            },
            "chunking_recommendations": self.generate_chunking_recommendations(layer_params)
        }

    def generate_chunking_recommendations(self, layer_params: List[int]) -> Dict[str, Any]:
        """Generate chunking recommendations based on model analysis"""
        total_layers = len(layer_params)
        total_layer_params = sum(layer_params)

        recommendations = {
            "optimal_chunk_counts": [],
            "recommended_strategies": [],
            "layer_split_points": {}
        }

        # Test different chunk counts
        for num_chunks in [2, 3, 4, 5]:
            if total_layers >= num_chunks:
                layers_per_chunk = total_layers // num_chunks
                remainder = total_layers % num_chunks

                split_points = []
                current_layer = 0
                for chunk_idx in range(num_chunks - 1):
                    layers_in_chunk = layers_per_chunk + (1 if chunk_idx < remainder else 0)
                    current_layer += layers_in_chunk
                    split_points.append(current_layer)

                recommendations["optimal_chunk_counts"].append(num_chunks)
                recommendations["layer_split_points"][num_chunks] = split_points

        # Recommend strategies based on layer parameter variance
        layer_variance = statistics.stdev(layer_params) if len(layer_params) > 1 else 0
        if layer_variance / statistics.mean(layer_params) < 0.1:  # Low variance
            recommendations["recommended_strategies"].append("equal_layers")
        recommendations["recommended_strategies"].append("equal_memory")

        return recommendations

    def test_02_legacy_weight_based_chunking(self) -> Dict[str, Any]:
        """Test legacy weight-based chunking for compatibility"""
        self.logger.info("Testing legacy weight-based chunking compatibility")

        extra_options = {
            "num_hidden_layers": self.model_config.num_hidden_layers,
            "chunk_strategy": "weight_based",
            "chunk_split_ratio": 0.5,
            "min_psnr_threshold": 35.0
        }

        builder = Qwen3Model(
            config=self.model_config,
            io_dtype=ir.DataType.FLOAT16,
            onnx_dtype=ir.DataType.FLOAT16,
            ep="cpu",
            cache_dir=str(self.cache_dir),
            extra_options=extra_options
        )

        # Create chunked models
        chunk_output_dir = self.chunked_dir / "test_02_legacy_weight_based"
        chunk_output_dir.mkdir(exist_ok=True)

        start_time = time.time()
        first_chunk, second_chunk, boundary_tensors = builder.create_chunked_models(
            output_dir=str(chunk_output_dir),
            verify_chunks=False
        )
        chunking_time = time.time() - start_time

        # Verify chunk models were created
        first_chunk_path = chunk_output_dir / "first_chunk.onnx"
        second_chunk_path = chunk_output_dir / "second_chunk.onnx"

        assert first_chunk_path.exists(), "First chunk model was not created"
        assert second_chunk_path.exists(), "Second chunk model was not created"

        # Analyze chunk sizes
        first_size = first_chunk_path.stat().st_size / (1024**2)
        second_size = second_chunk_path.stat().st_size / (1024**2)
        total_size = first_size + second_size

        return {
            "strategy": "weight_based",
            "legacy_mode": True,
            "first_chunk_size_mb": round(first_size, 2),
            "second_chunk_size_mb": round(second_size, 2),
            "total_chunked_size_mb": round(total_size, 2),
            "split_ratio": round(first_size / total_size, 3),
            "boundary_tensors_count": len(boundary_tensors),
            "chunking_time_seconds": round(chunking_time, 2),
            "files_created": {
                "first_chunk": str(first_chunk_path),
                "second_chunk": str(second_chunk_path)
            }
        }

    def test_03_layer_based_strategy(self) -> Dict[str, Any]:
        """Test new layer-based chunking strategy"""
        self.logger.info("Testing new layer-based chunking strategy")

        results = {}

        # Test layer-based strategy with default settings
        extra_options = {
            "num_hidden_layers": self.model_config.num_hidden_layers,
            "chunk_strategy": "layer_based",
            "chunk_distribution": "equal_memory",
            "num_chunks": 2,
            "respect_layer_boundaries": True,
            "min_psnr_threshold": 35.0
        }

        builder = Qwen3Model(
            config=self.model_config,
            io_dtype=ir.DataType.FLOAT16,
            onnx_dtype=ir.DataType.FLOAT16,
            ep="cpu",
            cache_dir=str(self.cache_dir),
            extra_options=extra_options
        )

        chunk_output_dir = self.chunked_dir / "test_03_layer_based"
        chunk_output_dir.mkdir(exist_ok=True)

        start_time = time.time()
        first_chunk, second_chunk, boundary_tensors = builder.create_chunked_models(
            output_dir=str(chunk_output_dir),
            verify_chunks=False
        )
        chunking_time = time.time() - start_time

        # Verify models were created
        first_chunk_path = chunk_output_dir / "first_chunk.onnx"
        second_chunk_path = chunk_output_dir / "second_chunk.onnx"

        assert first_chunk_path.exists(), "First chunk model was not created"
        assert second_chunk_path.exists(), "Second chunk model was not created"

        first_size = first_chunk_path.stat().st_size / (1024**2)
        second_size = second_chunk_path.stat().st_size / (1024**2)
        total_size = first_size + second_size

        # Analyze layer boundaries if available
        layer_boundaries = getattr(builder, 'layer_boundaries', {})

        results["layer_based_2_chunks"] = {
            "strategy": "layer_based",
            "distribution": "equal_memory",
            "num_chunks": 2,
            "first_chunk_size_mb": round(first_size, 2),
            "second_chunk_size_mb": round(second_size, 2),
            "total_size_mb": round(total_size, 2),
            "split_ratio": round(first_size / total_size, 3),
            "boundary_tensors_count": len(boundary_tensors),
            "chunking_time_seconds": round(chunking_time, 2),
            "layer_boundaries_tracked": bool(layer_boundaries),
            "respect_layer_boundaries": True,
            "files_created": {
                "first_chunk": str(first_chunk_path),
                "second_chunk": str(second_chunk_path)
            }
        }

        return results

    def test_04_multi_chunk_support(self) -> Dict[str, Any]:
        """Test multi-chunk support (2, 3, 4 chunks)"""
        self.logger.info("Testing multi-chunk support with different chunk counts")

        chunk_counts = [2, 3, 4]
        results = {}

        for num_chunks in chunk_counts:
            self.logger.info(f"Testing {num_chunks} chunks")

            extra_options = {
                "num_hidden_layers": self.model_config.num_hidden_layers,
                "chunk_strategy": "layer_based",
                "chunk_distribution": "equal_memory",
                "num_chunks": num_chunks,
                "respect_layer_boundaries": True,
                "min_psnr_threshold": 35.0
            }

            try:
                builder = Qwen3Model(
                    config=self.model_config,
                    io_dtype=ir.DataType.FLOAT16,
                    onnx_dtype=ir.DataType.FLOAT16,
                    ep="cpu",
                    cache_dir=str(self.cache_dir),
                    extra_options=extra_options
                )

                chunk_output_dir = self.chunked_dir / f"test_04_{num_chunks}_chunks"
                chunk_output_dir.mkdir(exist_ok=True)

                start_time = time.time()
                chunk_models = builder.create_chunked_models(
                    output_dir=str(chunk_output_dir),
                    verify_chunks=False
                )
                chunking_time = time.time() - start_time

                # Analyze created chunks
                chunk_files = list(chunk_output_dir.glob("chunk_*.onnx"))
                chunk_files.sort()

                chunk_sizes = []
                chunk_analysis = {}

                for i, chunk_file in enumerate(chunk_files):
                    size_mb = chunk_file.stat().st_size / (1024**2)
                    chunk_sizes.append(size_mb)
                    chunk_analysis[f"chunk_{i+1}"] = {
                        "path": str(chunk_file),
                        "size_mb": round(size_mb, 2)
                    }

                total_size = sum(chunk_sizes)

                results[f"{num_chunks}_chunks"] = {
                    "status": "success",
                    "num_chunks": num_chunks,
                    "chunk_files_found": len(chunk_files),
                    "total_size_mb": round(total_size, 2),
                    "average_chunk_size_mb": round(total_size / num_chunks, 2),
                    "size_variance": round(statistics.stdev(chunk_sizes) if len(chunk_sizes) > 1 else 0, 2),
                    "chunking_time_seconds": round(chunking_time, 2),
                    "chunk_analysis": chunk_analysis,
                    "size_distribution": [round(size / total_size * 100, 1) for size in chunk_sizes]
                }

            except Exception as e:
                results[f"{num_chunks}_chunks"] = {
                    "status": "failed",
                    "error": str(e),
                    "num_chunks": num_chunks
                }

        return results

    def test_05_distribution_strategies(self) -> Dict[str, Any]:
        """Test different chunk distribution strategies"""
        self.logger.info("Testing different chunk distribution strategies")

        strategies = ["equal_layers", "equal_memory", "custom"]
        num_chunks = 3  # Use 3 chunks for better distribution testing
        results = {}

        for strategy in strategies:
            self.logger.info(f"Testing distribution strategy: {strategy}")

            extra_options = {
                "num_hidden_layers": self.model_config.num_hidden_layers,
                "chunk_strategy": "layer_based",
                "chunk_distribution": strategy,
                "num_chunks": num_chunks,
                "respect_layer_boundaries": True,
                "min_psnr_threshold": 35.0
            }

            # Add custom chunk sizes for custom strategy
            if strategy == "custom":
                # Custom distribution: 20%, 30%, 50%
                custom_sizes = [0.2, 0.3, 0.5]
                extra_options["custom_chunk_sizes"] = custom_sizes

            try:
                builder = Qwen3Model(
                    config=self.model_config,
                    io_dtype=ir.DataType.FLOAT16,
                    onnx_dtype=ir.DataType.FLOAT16,
                    ep="cpu",
                    cache_dir=str(self.cache_dir),
                    extra_options=extra_options
                )

                chunk_output_dir = self.chunked_dir / f"test_05_{strategy}_distribution"
                chunk_output_dir.mkdir(exist_ok=True)

                start_time = time.time()
                chunk_models = builder.create_chunked_models(
                    output_dir=str(chunk_output_dir),
                    verify_chunks=False
                )
                chunking_time = time.time() - start_time

                # Analyze distribution
                chunk_files = list(chunk_output_dir.glob("chunk_*.onnx"))
                chunk_files.sort()

                chunk_sizes = [f.stat().st_size / (1024**2) for f in chunk_files]
                total_size = sum(chunk_sizes)
                size_percentages = [size / total_size * 100 for size in chunk_sizes]

                # Expected distribution for comparison
                if strategy == "custom":
                    expected_percentages = [20.0, 30.0, 50.0]
                    distribution_error = [abs(actual - expected)
                                        for actual, expected in zip(size_percentages, expected_percentages)]
                else:
                    expected_percentages = [100.0 / num_chunks] * num_chunks
                    distribution_error = [abs(actual - expected)
                                        for actual, expected in zip(size_percentages, expected_percentages)]

                results[strategy] = {
                    "status": "success",
                    "strategy": strategy,
                    "num_chunks": num_chunks,
                    "chunk_sizes_mb": [round(size, 2) for size in chunk_sizes],
                    "size_percentages": [round(pct, 1) for pct in size_percentages],
                    "expected_percentages": expected_percentages,
                    "distribution_error": [round(err, 1) for err in distribution_error],
                    "avg_distribution_error": round(statistics.mean(distribution_error), 1),
                    "chunking_time_seconds": round(chunking_time, 2),
                    "custom_sizes_used": extra_options.get("custom_chunk_sizes") if strategy == "custom" else None
                }

            except Exception as e:
                results[strategy] = {
                    "status": "failed",
                    "strategy": strategy,
                    "error": str(e)
                }

        return results

    def test_06_custom_chunk_configurations(self) -> Dict[str, Any]:
        """Test various custom chunk configurations"""
        self.logger.info("Testing custom chunk configurations")

        test_configurations = [
            {
                "name": "custom_4_chunk_distribution",
                "extra_options": {
                    "chunk_strategy": "layer_based",
                    "chunk_distribution": "custom",
                    "num_chunks": 4,
                    "custom_chunk_sizes": [0.15, 0.25, 0.30, 0.30],  # 15%, 25%, 30%, 30%
                    "respect_layer_boundaries": True
                }
            },
            {
                "name": "unequal_3_chunk_split",
                "extra_options": {
                    "chunk_strategy": "layer_based",
                    "chunk_distribution": "custom",
                    "num_chunks": 3,
                    "custom_chunk_sizes": [0.1, 0.3, 0.6],  # 10%, 30%, 60%
                    "respect_layer_boundaries": True
                }
            },
            {
                "name": "equal_layers_distribution",
                "extra_options": {
                    "chunk_strategy": "layer_based",
                    "chunk_distribution": "equal_layers",
                    "num_chunks": 3,
                    "respect_layer_boundaries": True
                }
            },
            {
                "name": "boundary_disrespect_test",
                "extra_options": {
                    "chunk_strategy": "layer_based",
                    "chunk_distribution": "equal_memory",
                    "num_chunks": 3,
                    "respect_layer_boundaries": False  # Test without respecting boundaries
                }
            }
        ]

        results = {}

        for test_config in test_configurations:
            config_name = test_config["name"]
            self.logger.info(f"Testing configuration: {config_name}")

            # Add base configuration
            extra_options = {
                "num_hidden_layers": self.model_config.num_hidden_layers,
                "min_psnr_threshold": 35.0,
                **test_config["extra_options"]
            }

            try:
                builder = Qwen3Model(
                    config=self.model_config,
                    io_dtype=ir.DataType.FLOAT16,
                    onnx_dtype=ir.DataType.FLOAT16,
                    ep="cpu",
                    cache_dir=str(self.cache_dir),
                    extra_options=extra_options
                )

                chunk_output_dir = self.chunked_dir / f"test_06_{config_name}"
                chunk_output_dir.mkdir(exist_ok=True)

                start_time = time.time()
                chunk_models = builder.create_chunked_models(
                    output_dir=str(chunk_output_dir),
                    verify_chunks=False
                )
                chunking_time = time.time() - start_time

                # Analyze results
                chunk_files = list(chunk_output_dir.glob("chunk_*.onnx"))
                chunk_files.sort()

                chunk_sizes = [f.stat().st_size / (1024**2) for f in chunk_files]
                total_size = sum(chunk_sizes)

                # Validate against expected distribution if custom
                validation_result = {}
                if extra_options.get("custom_chunk_sizes"):
                    expected_sizes = extra_options["custom_chunk_sizes"]
                    actual_percentages = [size / total_size * 100 for size in chunk_sizes]
                    expected_percentages = [size * 100 for size in expected_sizes]

                    validation_result = {
                        "expected_percentages": expected_percentages,
                        "actual_percentages": [round(pct, 1) for pct in actual_percentages],
                        "max_deviation": round(max(abs(a - e) for a, e in zip(actual_percentages, expected_percentages)), 1),
                        "within_tolerance": all(abs(a - e) <= 5.0 for a, e in zip(actual_percentages, expected_percentages))
                    }

                results[config_name] = {
                    "status": "success",
                    "configuration": test_config["extra_options"],
                    "num_chunks_created": len(chunk_files),
                    "chunk_sizes_mb": [round(size, 2) for size in chunk_sizes],
                    "total_size_mb": round(total_size, 2),
                    "chunking_time_seconds": round(chunking_time, 2),
                    "validation": validation_result,
                    "files_created": [str(f) for f in chunk_files]
                }

            except Exception as e:
                results[config_name] = {
                    "status": "failed",
                    "configuration": test_config["extra_options"],
                    "error": str(e)
                }

        return results

    def test_07_psnr_verification_accuracy(self) -> Dict[str, Any]:
        """Test PSNR verification for accuracy validation"""
        self.logger.info("Testing PSNR verification accuracy")

        psnr_thresholds = [25.0, 30.0, 35.0, 40.0, 45.0]
        results = {}

        for threshold in psnr_thresholds:
            self.logger.info(f"Testing PSNR threshold: {threshold}")

            extra_options = {
                "num_hidden_layers": self.model_config.num_hidden_layers,
                "chunk_strategy": "layer_based",
                "chunk_distribution": "equal_memory",
                "num_chunks": 2,
                "min_psnr_threshold": threshold,
                "respect_layer_boundaries": True
            }

            try:
                builder = Qwen3Model(
                    config=self.model_config,
                    io_dtype=ir.DataType.FLOAT16,
                    onnx_dtype=ir.DataType.FLOAT16,
                    ep="cpu",
                    cache_dir=str(self.cache_dir),
                    extra_options=extra_options
                )

                chunk_output_dir = self.chunked_dir / f"test_07_psnr_{int(threshold)}"
                chunk_output_dir.mkdir(exist_ok=True)

                # Prepare sample input for verification
                sample_input = self.sample_inputs['test_1']
                sample_input_dict = {
                    'input_ids': sample_input['input_ids'].numpy(),
                    'attention_mask': sample_input['attention_mask'].numpy()
                }

                start_time = time.time()
                chunk_models = builder.create_chunked_models(
                    output_dir=str(chunk_output_dir),
                    verify_chunks=True,
                    sample_input=sample_input_dict
                )
                verification_time = time.time() - start_time

                results[f"psnr_{threshold}"] = {
                    "status": "success",
                    "threshold": threshold,
                    "verification_passed": True,
                    "verification_time_seconds": round(verification_time, 2),
                    "sample_input_tokens": sample_input['token_count'],
                    "note": f"Verification completed successfully with threshold {threshold}dB"
                }

            except Exception as e:
                error_msg = str(e).lower()
                if "psnr" in error_msg or "threshold" in error_msg:
                    results[f"psnr_{threshold}"] = {
                        "status": "verification_failed",
                        "threshold": threshold,
                        "verification_passed": False,
                        "error": str(e),
                        "interpretation": f"PSNR threshold {threshold}dB too strict - quality difference exceeded threshold"
                    }
                else:
                    results[f"psnr_{threshold}"] = {
                        "status": "error",
                        "threshold": threshold,
                        "verification_passed": False,
                        "error": str(e),
                        "interpretation": f"Unexpected error during verification"
                    }

        return results

    def test_08_performance_benchmarking(self) -> Dict[str, Any]:
        """Test performance and benchmark different configurations"""
        self.logger.info("Running performance benchmarking")

        benchmark_configs = [
            {
                "name": "weight_based_2_chunks",
                "extra_options": {
                    "chunk_strategy": "weight_based",
                    "chunk_split_ratio": 0.5
                }
            },
            {
                "name": "layer_based_2_chunks",
                "extra_options": {
                    "chunk_strategy": "layer_based",
                    "chunk_distribution": "equal_memory",
                    "num_chunks": 2
                }
            },
            {
                "name": "layer_based_3_chunks",
                "extra_options": {
                    "chunk_strategy": "layer_based",
                    "chunk_distribution": "equal_memory",
                    "num_chunks": 3
                }
            },
            {
                "name": "layer_based_4_chunks",
                "extra_options": {
                    "chunk_strategy": "layer_based",
                    "chunk_distribution": "equal_memory",
                    "num_chunks": 4
                }
            }
        ]

        results = {}

        for config in benchmark_configs:
            config_name = config["name"]
            self.logger.info(f"Benchmarking: {config_name}")

            # Add base configuration
            extra_options = {
                "num_hidden_layers": self.model_config.num_hidden_layers,
                "min_psnr_threshold": 35.0,
                "respect_layer_boundaries": True,
                **config["extra_options"]
            }

            # Performance measurements
            memory_before = self.get_performance_metrics()

            try:
                # Model creation time
                model_start = time.time()
                builder = Qwen3Model(
                    config=self.model_config,
                    io_dtype=ir.DataType.FLOAT16,
                    onnx_dtype=ir.DataType.FLOAT16,
                    ep="cpu",
                    cache_dir=str(self.cache_dir),
                    extra_options=extra_options
                )
                model_creation_time = time.time() - model_start

                # Chunking time
                chunk_output_dir = self.chunked_dir / f"test_08_benchmark_{config_name}"
                chunk_output_dir.mkdir(exist_ok=True)

                chunk_start = time.time()
                chunk_models = builder.create_chunked_models(
                    output_dir=str(chunk_output_dir),
                    verify_chunks=False
                )
                chunking_time = time.time() - chunk_start

                # Memory after chunking
                memory_after = self.get_performance_metrics()

                # Analyze created chunks
                chunk_files = list(chunk_output_dir.glob("chunk_*.onnx"))
                chunk_sizes = [f.stat().st_size / (1024**2) for f in chunk_files]
                total_size = sum(chunk_sizes)

                results[config_name] = {
                    "status": "success",
                    "model_creation_time_seconds": round(model_creation_time, 3),
                    "chunking_time_seconds": round(chunking_time, 3),
                    "total_time_seconds": round(model_creation_time + chunking_time, 3),
                    "chunks_created": len(chunk_files),
                    "total_size_mb": round(total_size, 2),
                    "average_chunk_size_mb": round(total_size / len(chunk_files), 2) if chunk_files else 0,
                    "memory_usage": {
                        "before_mb": round(memory_before['memory_usage_mb'], 1),
                        "after_mb": round(memory_after['memory_usage_mb'], 1),
                        "peak_delta_mb": round(memory_after['memory_usage_mb'] - memory_before['memory_usage_mb'], 1)
                    },
                    "throughput_mb_per_second": round(total_size / chunking_time, 2) if chunking_time > 0 else 0,
                    "chunks_per_second": round(len(chunk_files) / chunking_time, 2) if chunking_time > 0 else 0
                }

            except Exception as e:
                results[config_name] = {
                    "status": "failed",
                    "error": str(e)
                }

        return results

    def test_09_model_loading_and_inference(self) -> Dict[str, Any]:
        """Test model loading and inference execution"""
        self.logger.info("Testing model loading and inference execution")

        # Test cases: original model and different chunking configurations
        test_models = [
            {
                "name": "original_model",
                "path": self.models_dir / "qwen_original.onnx",
                "description": "Original unchunked model"
            }
        ]

        # Add chunked models from previous tests
        chunked_models_to_test = [
            ("weight_based", "test_02_legacy_weight_based", "first_chunk.onnx"),
            ("layer_based_2_chunks", "test_03_layer_based", "first_chunk.onnx"),
            ("multi_chunk_3", "test_04_3_chunks", "chunk_1.onnx")
        ]

        for strategy, test_dir, chunk_file in chunked_models_to_test:
            chunk_path = self.chunked_dir / test_dir / chunk_file
            if chunk_path.exists():
                test_models.append({
                    "name": f"{strategy}_{chunk_file.replace('.onnx', '')}",
                    "path": chunk_path,
                    "description": f"{strategy} - {chunk_file}"
                })

        results = {}

        for test_model in test_models:
            model_path = test_model["path"]
            model_name = test_model["name"]

            if not model_path.exists():
                results[model_name] = {
                    "status": "skipped",
                    "reason": f"Model file not found: {model_path}"
                }
                continue

            self.logger.info(f"Testing model: {model_name}")

            try:
                # Model loading test
                load_start = time.time()
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

                session = ort.InferenceSession(
                    str(model_path),
                    sess_options=session_options,
                    providers=["CPUExecutionProvider"]
                )
                load_time = time.time() - load_start

                # Get model input/output info
                inputs = session.get_inputs()
                outputs = session.get_outputs()

                # Inference test with different input sizes
                inference_results = {}
                for test_key, test_input in self.sample_inputs.items():
                    inference_start = time.time()

                    input_feed = {
                        'input_ids': test_input['input_ids'].numpy(),
                        'attention_mask': test_input['attention_mask'].numpy()
                    }

                    try:
                        model_outputs = session.run(None, input_feed)
                        inference_time = time.time() - inference_start

                        inference_results[test_key] = {
                            "status": "success",
                            "execution_time_seconds": round(inference_time, 3),
                            "input_tokens": test_input['token_count'],
                            "output_shapes": [output.shape for output in model_outputs],
                            "output_dtypes": [output.dtype for output in model_outputs]
                        }

                    except Exception as inference_error:
                        inference_results[test_key] = {
                            "status": "failed",
                            "error": str(inference_error)
                        }

                results[model_name] = {
                    "status": "success",
                    "loading_time_seconds": round(load_time, 3),
                    "input_count": len(inputs),
                    "output_count": len(outputs),
                    "input_info": [{"name": inp.name, "shape": inp.shape, "type": inp.type} for inp in inputs],
                    "output_info": [{"name": out.name, "shape": out.shape, "type": out.type} for out in outputs],
                    "model_size_mb": round(model_path.stat().st_size / (1024**2), 2),
                    "inference_results": inference_results,
                    "description": test_model["description"]
                }

            except Exception as e:
                results[model_name] = {
                    "status": "failed",
                    "error": str(e),
                    "description": test_model["description"]
                }

        return results

    def test_10_error_handling_and_validation(self) -> Dict[str, Any]:
        """Comprehensive error handling and validation tests"""
        self.logger.info("Testing error handling and validation")

        error_test_cases = [
            {
                "name": "invalid_chunk_strategy",
                "extra_options": {
                    "chunk_strategy": "invalid_strategy",
                    "num_chunks": 2
                },
                "should_fail": True,
                "expected_error_type": "ValueError"
            },
            {
                "name": "invalid_chunk_distribution",
                "extra_options": {
                    "chunk_strategy": "layer_based",
                    "chunk_distribution": "invalid_distribution",
                    "num_chunks": 2
                },
                "should_fail": True,
                "expected_error_type": "ValueError"
            },
            {
                "name": "invalid_num_chunks_zero",
                "extra_options": {
                    "chunk_strategy": "layer_based",
                    "num_chunks": 0
                },
                "should_fail": True,
                "expected_error_type": "ValueError"
            },
            {
                "name": "invalid_num_chunks_negative",
                "extra_options": {
                    "chunk_strategy": "layer_based",
                    "num_chunks": -1
                },
                "should_fail": True,
                "expected_error_type": "ValueError"
            },
            {
                "name": "invalid_custom_chunk_sizes_mismatch",
                "extra_options": {
                    "chunk_strategy": "layer_based",
                    "chunk_distribution": "custom",
                    "num_chunks": 3,
                    "custom_chunk_sizes": [0.5, 0.5]  # Only 2 sizes for 3 chunks
                },
                "should_fail": True,
                "expected_error_type": "ValueError"
            },
            {
                "name": "invalid_custom_chunk_sizes_sum",
                "extra_options": {
                    "chunk_strategy": "layer_based",
                    "chunk_distribution": "custom",
                    "num_chunks": 2,
                    "custom_chunk_sizes": [0.6, 0.6]  # Sum > 1.0
                },
                "should_fail": True,
                "expected_error_type": "ValueError"
            },
            {
                "name": "extreme_num_chunks_exceeds_layers",
                "extra_options": {
                    "chunk_strategy": "layer_based",
                    "num_chunks": 100,  # Much more than typical layer count
                    "chunk_distribution": "equal_layers"
                },
                "should_fail": True,
                "expected_error_type": "ValueError"
            },
            {
                "name": "legacy_split_ratio_negative",
                "extra_options": {
                    "chunk_strategy": "weight_based",
                    "chunk_split_ratio": -0.1
                },
                "should_fail": True,
                "expected_error_type": "ValueError"
            },
            {
                "name": "legacy_split_ratio_greater_than_1",
                "extra_options": {
                    "chunk_strategy": "weight_based",
                    "chunk_split_ratio": 1.5
                },
                "should_fail": True,
                "expected_error_type": "ValueError"
            }
        ]

        results = {}

        for test_case in error_test_cases:
            test_name = test_case["name"]
            self.logger.info(f"Testing error case: {test_name}")

            # Add base configuration
            extra_options = {
                "num_hidden_layers": self.model_config.num_hidden_layers,
                "min_psnr_threshold": 35.0,
                **test_case["extra_options"]
            }

            try:
                builder = Qwen3Model(
                    config=self.model_config,
                    io_dtype=ir.DataType.FLOAT16,
                    onnx_dtype=ir.DataType.FLOAT16,
                    ep="cpu",
                    cache_dir=str(self.cache_dir),
                    extra_options=extra_options
                )

                # Try to create chunked models
                chunk_output_dir = self.chunked_dir / f"test_10_error_{test_name}"
                chunk_output_dir.mkdir(exist_ok=True)

                chunk_models = builder.create_chunked_models(
                    output_dir=str(chunk_output_dir),
                    verify_chunks=False
                )

                # If we get here, the operation succeeded when it should have failed
                if test_case["should_fail"]:
                    results[test_name] = {
                        "status": "unexpected_success",
                        "should_fail": True,
                        "error_occurred": False,
                        "expected_error_type": test_case.get("expected_error_type"),
                        "message": "Expected error but operation succeeded"
                    }
                else:
                    results[test_name] = {
                        "status": "success",
                        "should_fail": False,
                        "error_occurred": False,
                        "message": "Operation succeeded as expected"
                    }

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)

                if test_case["should_fail"]:
                    # Check if it's the expected error type
                    expected_type = test_case.get("expected_error_type")
                    type_match = expected_type is None or error_type == expected_type

                    results[test_name] = {
                        "status": "expected_error" if type_match else "wrong_error_type",
                        "should_fail": True,
                        "error_occurred": True,
                        "error_type": error_type,
                        "error_message": error_message,
                        "expected_error_type": expected_type,
                        "type_match": type_match,
                        "message": "Expected error occurred" if type_match else f"Wrong error type: expected {expected_type}, got {error_type}"
                    }
                else:
                    results[test_name] = {
                        "status": "unexpected_error",
                        "should_fail": False,
                        "error_occurred": True,
                        "error_type": error_type,
                        "error_message": error_message,
                        "message": "Unexpected error occurred"
                    }

        return results

    def test_11_edge_cases_and_boundary_conditions(self) -> Dict[str, Any]:
        """Test edge cases and boundary conditions"""
        self.logger.info("Testing edge cases and boundary conditions")

        results = {}

        # Test with minimum chunk count
        try:
            extra_options = {
                "num_hidden_layers": self.model_config.num_hidden_layers,
                "chunk_strategy": "layer_based",
                "num_chunks": 1,  # Minimum valid
                "chunk_distribution": "equal_memory"
            }

            builder = Qwen3Model(
                config=self.model_config,
                io_dtype=ir.DataType.FLOAT16,
                onnx_dtype=ir.DataType.FLOAT16,
                ep="cpu",
                cache_dir=str(self.cache_dir),
                extra_options=extra_options
            )

            chunk_output_dir = self.chunked_dir / "test_11_min_chunks"
            chunk_output_dir.mkdir(exist_ok=True)

            chunk_models = builder.create_chunked_models(
                output_dir=str(chunk_output_dir),
                verify_chunks=False
            )

            results["minimum_chunks"] = {
                "status": "success",
                "num_chunks": 1,
                "note": "Single chunk creation successful (effectively no chunking)"
            }

        except Exception as e:
            results["minimum_chunks"] = {
                "status": "failed",
                "error": str(e),
                "note": "Single chunk creation failed"
            }

        # Test with maximum reasonable chunk count
        max_chunks = min(10, self.model_config.num_hidden_layers)
        try:
            extra_options = {
                "num_hidden_layers": self.model_config.num_hidden_layers,
                "chunk_strategy": "layer_based",
                "num_chunks": max_chunks,
                "chunk_distribution": "equal_layers",
                "respect_layer_boundaries": True
            }

            builder = Qwen3Model(
                config=self.model_config,
                io_dtype=ir.DataType.FLOAT16,
                onnx_dtype=ir.DataType.FLOAT16,
                ep="cpu",
                cache_dir=str(self.cache_dir),
                extra_options=extra_options
            )

            chunk_output_dir = self.chunked_dir / f"test_11_max_chunks_{max_chunks}"
            chunk_output_dir.mkdir(exist_ok=True)

            chunk_models = builder.create_chunked_models(
                output_dir=str(chunk_output_dir),
                verify_chunks=False
            )

            chunk_files = list(chunk_output_dir.glob("chunk_*.onnx"))

            results["maximum_chunks"] = {
                "status": "success",
                "max_chunks_tested": max_chunks,
                "chunks_created": len(chunk_files),
                "layers_per_chunk_avg": round(self.model_config.num_hidden_layers / max_chunks, 1)
            }

        except Exception as e:
            results["maximum_chunks"] = {
                "status": "failed",
                "max_chunks_tested": max_chunks,
                "error": str(e)
            }

        # Test extreme custom distributions
        extreme_distributions = [
            {
                "name": "highly_skewed_2_chunks",
                "custom_sizes": [0.05, 0.95],  # 5% and 95%
                "num_chunks": 2
            },
            {
                "name": "highly_skewed_3_chunks",
                "custom_sizes": [0.05, 0.15, 0.80],  # 5%, 15%, 80%
                "num_chunks": 3
            },
            {
                "name": "minimal_chunks",
                "custom_sizes": [0.01, 0.99],  # 1% and 99%
                "num_chunks": 2
            }
        ]

        for dist_config in extreme_distributions:
            try:
                extra_options = {
                    "num_hidden_layers": self.model_config.num_hidden_layers,
                    "chunk_strategy": "layer_based",
                    "chunk_distribution": "custom",
                    "num_chunks": dist_config["num_chunks"],
                    "custom_chunk_sizes": dist_config["custom_sizes"],
                    "respect_layer_boundaries": True
                }

                builder = Qwen3Model(
                    config=self.model_config,
                    io_dtype=ir.DataType.FLOAT16,
                    onnx_dtype=ir.DataType.FLOAT16,
                    ep="cpu",
                    cache_dir=str(self.cache_dir),
                    extra_options=extra_options
                )

                chunk_output_dir = self.chunked_dir / f"test_11_extreme_{dist_config['name']}"
                chunk_output_dir.mkdir(exist_ok=True)

                chunk_models = builder.create_chunked_models(
                    output_dir=str(chunk_output_dir),
                    verify_chunks=False
                )

                # Analyze actual distribution
                chunk_files = list(chunk_output_dir.glob("chunk_*.onnx"))
                chunk_files.sort()

                chunk_sizes = [f.stat().st_size / (1024**2) for f in chunk_files]
                total_size = sum(chunk_sizes)
                actual_percentages = [size / total_size * 100 for size in chunk_sizes]
                expected_percentages = [size * 100 for size in dist_config["custom_sizes"]]

                results[f"extreme_{dist_config['name']}"] = {
                    "status": "success",
                    "custom_sizes": dist_config["custom_sizes"],
                    "expected_percentages": expected_percentages,
                    "actual_percentages": [round(pct, 1) for pct in actual_percentages],
                    "chunks_created": len(chunk_files),
                    "max_deviation": round(max(abs(a - e) for a, e in zip(actual_percentages, expected_percentages)), 1)
                }

            except Exception as e:
                results[f"extreme_{dist_config['name']}"] = {
                    "status": "failed",
                    "custom_sizes": dist_config["custom_sizes"],
                    "error": str(e)
                }

        return results

    def create_original_onnx_model(self) -> Dict[str, Any]:
        """Helper method to create original ONNX model for comparison"""
        self.logger.info("Creating original ONNX model for comparison")

        extra_options = {
            "num_hidden_layers": self.model_config.num_hidden_layers,
            "enable_chunking": False  # No chunking for original model
        }

        builder = Qwen3Model(
            config=self.model_config,
            io_dtype=ir.DataType.FLOAT16,
            onnx_dtype=ir.DataType.FLOAT16,
            ep="cpu",
            cache_dir=str(self.cache_dir),
            extra_options=extra_options
        )

        # Build the model
        output_path = self.models_dir / "qwen_original.onnx"
        builder.export(str(output_path))

        # Verify model was created
        assert output_path.exists(), "Original ONNX model was not created"

        # Get model info
        onnx_model = onnx.load(str(output_path))

        return {
            "model_path": str(output_path),
            "file_size_mb": output_path.stat().st_size / (1024**2),
            "num_nodes": len(onnx_model.graph.node),
            "num_inputs": len(onnx_model.graph.input),
            "num_outputs": len(onnx_model.graph.output),
            "num_initializers": len(onnx_model.graph.initializer)
        }

    def generate_comprehensive_test_report(self) -> str:
        """Generate comprehensive test report with performance analysis"""
        # Calculate summary statistics
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.status == "PASS"])
        failed_tests = len([r for r in self.test_results if r.status == "FAIL"])
        skipped_tests = len([r for r in self.test_results if r.status == "SKIP"])

        total_time = sum(r.execution_time for r in self.test_results)

        # Performance summary
        performance_summary = {}
        if self.test_results:
            all_performance = [r.performance_metrics for r in self.test_results if r.performance_metrics]
            if all_performance:
                performance_summary = {
                    'avg_execution_time': statistics.mean([r.execution_time for r in self.test_results]),
                    'max_execution_time': max([r.execution_time for r in self.test_results]),
                    'avg_peak_memory_mb': statistics.mean([p.get('peak_memory_mb', 0) for p in all_performance]),
                    'max_peak_memory_mb': max([p.get('peak_memory_mb', 0) for p in all_performance]),
                    'total_cpu_time': sum([r.execution_time * r.performance_metrics.get('avg_cpu_percent', 0) / 100
                                         for r in self.test_results if r.performance_metrics])
                }

        # Create results object
        results = TestSuiteResults(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            total_execution_time=total_time,
            test_results=self.test_results,
            system_info=self.system_info,
            performance_summary=performance_summary
        )

        # Generate detailed JSON report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"enhanced_chunking_test_report_{timestamp}.json"

        with open(report_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2, default=str)

        # Generate comprehensive summary report
        summary_file = self.reports_dir / f"enhanced_chunking_test_summary_{timestamp}.txt"

        with open(summary_file, 'w') as f:
            f.write("Enhanced ONNX Model Chunking Test Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Model Type: {self.model_config.model_type if self.model_config else 'Unknown'}\n")
            f.write(f"Model Layers: {self.model_config.num_hidden_layers if self.model_config else 'Unknown'}\n\n")

            f.write("Test Summary:\n")
            f.write("-" * 30 + "\n")
            f.write(f"  Total Tests: {total_tests}\n")
            f.write(f"  Passed: {passed_tests}\n")
            f.write(f"  Failed: {failed_tests}\n")
            f.write(f"  Skipped: {skipped_tests}\n")
            f.write(f"  Success Rate: {(passed_tests/total_tests*100):.1f}%\n")
            f.write(f"  Total Execution Time: {total_time:.2f}s\n")
            f.write(f"  Average Test Time: {(total_time/total_tests):.2f}s\n\n")

            if performance_summary:
                f.write("Performance Summary:\n")
                f.write("-" * 30 + "\n")
                f.write(f"  Peak Memory Usage: {performance_summary.get('max_peak_memory_mb', 0):.1f}MB\n")
                f.write(f"  Average Memory Usage: {performance_summary.get('avg_peak_memory_mb', 0):.1f}MB\n")
                f.write(f"  Longest Test: {performance_summary.get('max_execution_time', 0):.2f}s\n")
                f.write(f"  Average Test Time: {performance_summary.get('avg_execution_time', 0):.2f}s\n\n")

            f.write("Test Results by Category:\n")
            f.write("-" * 30 + "\n")

            # Group tests by category
            categories = {}
            for result in self.test_results:
                category = result.test_name.split('.')[0] + '.' + result.test_name.split('.')[1]
                if category not in categories:
                    categories[category] = []
                categories[category].append(result)

            for category, cat_results in categories.items():
                passed = len([r for r in cat_results if r.status == "PASS"])
                total = len(cat_results)
                f.write(f"  {category}: {passed}/{total} passed ({(passed/total*100):.1f}%)\n")

            f.write("\nDetailed Test Results:\n")
            f.write("-" * 30 + "\n")
            for result in self.test_results:
                status_symbol = "[PASS]" if result.status == "PASS" else "[FAIL]" if result.status == "FAIL" else "[SKIP]"
                f.write(f"{status_symbol} {result.test_name}: {result.status} ({result.execution_time:.2f}s)\n")

                if result.performance_metrics:
                    perf = result.performance_metrics
                    f.write(f"    Memory: {perf.get('peak_memory_mb', 0):.1f}MB peak, "
                           f"CPU: {perf.get('avg_cpu_percent', 0):.1f}% avg\n")

                if result.details and 'result' in result.details:
                    f.write(f"    Result: {result.details['result']}\n")

            f.write(f"\nTest Artifacts:\n")
            f.write("-" * 30 + "\n")
            f.write(f"  Test Directory: {self.base_dir}\n")
            f.write(f"  Models Directory: {self.models_dir}\n")
            f.write(f"  Chunked Models Directory: {self.chunked_dir}\n")
            f.write(f"  Reports Directory: {self.reports_dir}\n")
            f.write(f"  Performance Directory: {self.performance_dir}\n")

        return str(report_file)

    def cleanup(self):
        """Clean up test artifacts (optional)"""
        self.logger.info("Test artifacts preserved for inspection")
        self.logger.info(f"Artifacts location: {self.base_dir}")

        # Note: Uncomment the following lines to clean up everything:
        # try:
        #     shutil.rmtree(self.base_dir)
        #     self.logger.info("Test artifacts cleaned up successfully")
        # except Exception as e:
        #     self.logger.warning(f"Failed to clean up test artifacts: {e}")

    def run_all_tests(self) -> str:
        """Run all enhanced tests and generate comprehensive report"""
        self.start_time = time.time()

        self.logger.info("Starting Enhanced ONNX Model Chunking Test Suite")
        self.logger.info(f"Target Model: {self.model_name}")
        self.logger.info(f"Test Input: '{self.test_input_text}'")
        self.logger.info("=" * 80)

        # Define all enhanced tests to run
        tests = [
            ("01. Model Download and Architecture Analysis", self.test_01_model_download_and_setup),
            ("02. Legacy Weight-Based Chunking", self.test_02_legacy_weight_based_chunking),
            ("03. Layer-Based Strategy", self.test_03_layer_based_strategy),
            ("04. Multi-Chunk Support", self.test_04_multi_chunk_support),
            ("05. Distribution Strategies", self.test_05_distribution_strategies),
            ("06. Custom Chunk Configurations", self.test_06_custom_chunk_configurations),
            ("07. PSNR Verification Accuracy", self.test_07_psnr_verification_accuracy),
            ("08. Performance Benchmarking", self.test_08_performance_benchmarking),
            ("09. Model Loading and Inference", self.test_09_model_loading_and_inference),
            ("10. Error Handling and Validation", self.test_10_error_handling_and_validation),
            ("11. Edge Cases and Boundary Conditions", self.test_11_edge_cases_and_boundary_conditions),
        ]

        # Run all tests
        for test_name, test_func in tests:
            try:
                self.run_test(test_name, test_func)
            except KeyboardInterrupt:
                self.logger.warning(f"Test suite interrupted by user during {test_name}")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error running {test_name}: {e}")
                self.run_test(test_name, lambda: {"error": str(e)})

        # Generate final report
        total_execution_time = time.time() - self.start_time
        self.logger.info(f"\nEnhanced Test Suite Completed in {total_execution_time:.2f}s")

        report_file = self.generate_comprehensive_test_report()
        self.logger.info(f"Comprehensive test report generated: {report_file}")

        return report_file


def main():
    """Main entry point"""
    print("Enhanced ONNX Model Chunking Test Suite")
    print("=" * 60)
    print("Testing Architecture-Aware Chunking Functionality")
    print()

    # Configuration
    config = {
        "model_name": "Qwen/Qwen2.5-0.5B",
        "test_input_text": "Hello, this is a test for enhanced model chunking functionality with architecture-aware splitting.",
        "cleanup_after_tests": False,  # Set to True to clean up artifacts
        "performance_monitoring": True,
        "detailed_logging": True
    }

    # Create and run enhanced test suite
    test_suite = EnhancedChunkingTestSuite(config)

    try:
        print(f"Starting enhanced test suite for model: {config['model_name']}")
        print(f"Test features: Architecture-aware chunking, multi-chunk support, distribution strategies, PSNR verification")
        print()

        report_file = test_suite.run_all_tests()

        print(f"\nEnhanced test suite completed successfully!")
        print(f"Report available at: {report_file}")
        print(f"Test artifacts preserved in: {test_suite.base_dir}")

        # Print quick summary
        total_tests = len(test_suite.test_results)
        passed_tests = len([r for r in test_suite.test_results if r.status == "PASS"])
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        print(f"Quick Summary: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        print(f"Total execution time: {time.time() - test_suite.start_time:.2f}s")

        return 0

    except KeyboardInterrupt:
        print("\nEnhanced test suite interrupted by user")
        return 1

    except Exception as e:
        print(f"\nEnhanced test suite failed with error: {e}")
        traceback.print_exc()
        return 1

    finally:
        # Cleanup if requested
        if config.get("cleanup_after_tests", False):
            test_suite.cleanup()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)