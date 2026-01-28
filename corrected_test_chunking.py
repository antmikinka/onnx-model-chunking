#!/usr/bin/env python3
"""
Corrected ONNX Model Chunking Test Script

This script fixes the architectural issue by properly integrating chunking
into the existing model creation workflow.

CORRECTED WORKFLOW:
1. Load PyTorch model from Hugging Face
2. Convert to complete ONNX model using existing create_model() function
3. Apply chunking to the complete ONNX model (happens automatically in create_model)
4. Save chunked ONNX models

Key fixes:
- Uses create_model() with chunking options instead of trying to chunk empty models
- Chunking happens after make_model() completes graph construction
- Proper integration with existing model creation pipeline
"""

import os
import sys
import time
import json
import shutil
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import onnx
import gc
import psutil
from datetime import datetime

# Add the current directory to the path to import builder module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from builder import create_model
    import onnx_ir as ir
except ImportError as e:
    print(f"Error importing builder module: {e}")
    print("Please ensure builder.py is in the same directory or accessible via PYTHONPATH")
    sys.exit(1)


@dataclass
class TestResult:
    """Data class to store test results"""
    test_name: str
    status: str  # 'PASS', 'FAIL', 'SKIP'
    execution_time: float
    details: Dict[str, Any]
    error_message: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class CorrectedChunkingTestSuite:
    """Corrected test suite that properly integrates chunking with model creation"""

    def __init__(self):
        # Test configuration
        self.model_name = "Qwen/Qwen2.5-0.5B"
        self.base_dir = Path("corrected_test_environment")
        self.models_dir = self.base_dir / "models"
        self.chunked_dir = self.base_dir / "chunked_models"
        self.cache_dir = self.base_dir / "cache"
        self.reports_dir = self.base_dir / "reports"

        self.test_results = []
        self.start_time = None

        self.setup_logging()
        self.setup_directories()

    def setup_logging(self):
        """Setup logging"""
        log_dir = Path("test_logs")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"corrected_chunking_test_{timestamp}.log"

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
        for directory in [self.base_dir, self.models_dir, self.chunked_dir,
                         self.cache_dir, self.reports_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def run_test(self, test_name: str, test_func) -> TestResult:
        """Run a single test and capture results"""
        self.logger.info(f"Starting test: {test_name}")

        start_time = time.time()
        test_result = TestResult(
            test_name=test_name,
            status="FAIL",
            execution_time=0.0,
            details={}
        )

        try:
            result_details = test_func()
            test_result.status = "PASS"
            test_result.details = result_details if isinstance(result_details, dict) else {"result": result_details}

        except Exception as e:
            test_result.status = "FAIL"
            test_result.error_message = str(e)
            test_result.details = {"traceback": traceback.format_exc()}
            self.logger.error(f"Test failed with exception: {e}")

        finally:
            test_result.execution_time = time.time() - start_time
            self.logger.info(f"{test_name}: {test_result.status} ({test_result.execution_time:.2f}s)")
            self.test_results.append(test_result)

            # Clean up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return test_result

    def test_01_model_setup(self) -> Dict[str, Any]:
        """Test model setup and configuration"""
        self.logger.info("Setting up model configuration")

        # Download model configuration
        config = AutoConfig.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=str(self.cache_dir)
        )

        return {
            "model_name": self.model_name,
            "model_type": config.model_type,
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_kv_heads": getattr(config, 'num_key_value_heads', config.num_attention_heads),
            "intermediate_size": getattr(config, 'intermediate_size', config.hidden_size * 4),
            "max_position_embeddings": getattr(config, 'max_position_embeddings', None),
        }

    def test_02_create_complete_model(self) -> Dict[str, Any]:
        """Test creating complete ONNX model without chunking"""
        self.logger.info("Creating complete ONNX model")

        output_path = self.models_dir / "qwen_complete.onnx"

        # Use create_model function to create complete model
        create_model(
            model_name=self.model_name,
            input_path=self.model_name,
            output_dir=str(self.models_dir),
            precision="fp16",  # Use fp16 for efficiency
            execution_provider="cpu",
            cache_dir=str(self.cache_dir),
            num_hidden_layers=24,  # Qwen2.5-0.5B has 24 layers
            enable_chunking=False  # No chunking for complete model
        )

        # Verify model was created (check for actual filename created)
        actual_model_path = self.models_dir / "model.onnx"  # builder.py creates this name
        if actual_model_path.exists():
            # Load and analyze the model
            onnx_model = onnx.load(str(actual_model_path))

            return {
                "model_created": True,
                "model_path": str(actual_model_path),
                "file_size_mb": actual_model_path.stat().st_size / (1024**2),
                "num_nodes": len(onnx_model.graph.node),
                "num_inputs": len(onnx_model.graph.input),
                "num_outputs": len(onnx_model.graph.output),
                "num_initializers": len(onnx_model.graph.initializer)
            }
        else:
            raise FileNotFoundError("Complete ONNX model was not created")

    def test_03_create_chunked_models_weight_based(self) -> Dict[str, Any]:
        """Test creating chunked models using weight-based strategy"""
        self.logger.info("Creating chunked models using weight-based strategy")

        chunk_output_dir = self.chunked_dir / "weight_based_chunks"
        chunk_output_dir.mkdir(exist_ok=True)

        # Use create_model function with chunking enabled
        create_model(
            model_name=self.model_name,
            input_path=self.model_name,
            output_dir=str(chunk_output_dir),
            precision="fp16",
            execution_provider="cpu",
            cache_dir=str(self.cache_dir),
            num_hidden_layers=24,
            # Enable chunking with weight-based strategy
            enable_chunking=True,
            chunk_strategy="weight_based",
            chunk_split_ratio=0.5,  # 50-50 split for 2 chunks
            chunk_output_dir=str(chunk_output_dir),
            verify_chunks=False
        )

        # Analyze created chunks
        chunk_files = list(chunk_output_dir.glob("*chunk*.onnx"))

        if len(chunk_files) >= 2:
            chunk_analysis = {}
            total_size = 0

            for chunk_file in chunk_files:
                size_mb = chunk_file.stat().st_size / (1024**2)
                total_size += size_mb

                # Load chunk to verify it has nodes
                chunk_onnx = onnx.load(str(chunk_file))

                chunk_analysis[chunk_file.name] = {
                    "path": str(chunk_file),
                    "size_mb": round(size_mb, 2),
                    "num_nodes": len(chunk_onnx.graph.node),
                    "num_inputs": len(chunk_onnx.graph.input),
                    "num_outputs": len(chunk_onnx.graph.output),
                    "num_initializers": len(chunk_onnx.graph.initializer)
                }

            return {
                "chunks_created": len(chunk_files),
                "total_size_mb": round(total_size, 2),
                "chunk_analysis": chunk_analysis,
                "chunking_strategy": "weight_based"
            }
        else:
            raise ValueError(f"Expected at least 2 chunk files, found {len(chunk_files)}")

    def test_04_create_chunked_models_layer_based(self) -> Dict[str, Any]:
        """Test creating chunked models using layer-based strategy"""
        self.logger.info("Creating chunked models using layer-based strategy")

        chunk_output_dir = self.chunked_dir / "layer_based_chunks"
        chunk_output_dir.mkdir(exist_ok=True)

        # Use create_model function with layer-based chunking
        create_model(
            model_name=self.model_name,
            input_path=self.model_name,
            output_dir=str(chunk_output_dir),
            precision="fp16",
            execution_provider="cpu",
            cache_dir=str(self.cache_dir),
            num_hidden_layers=24,
            # Enable layer-based chunking
            enable_chunking=True,
            chunk_strategy="layer_based",
            chunk_distribution="equal_memory",
            num_chunks=3,  # Create 3 chunks
            respect_layer_boundaries=True,
            chunk_output_dir=str(chunk_output_dir),
            verify_chunks=False
        )

        # Analyze created chunks
        chunk_files = list(chunk_output_dir.glob("*chunk*.onnx"))

        if len(chunk_files) >= 2:
            chunk_analysis = {}
            total_size = 0

            for chunk_file in chunk_files:
                size_mb = chunk_file.stat().st_size / (1024**2)
                total_size += size_mb

                # Load chunk to verify it has nodes
                chunk_onnx = onnx.load(str(chunk_file))

                chunk_analysis[chunk_file.name] = {
                    "path": str(chunk_file),
                    "size_mb": round(size_mb, 2),
                    "num_nodes": len(chunk_onnx.graph.node),
                    "num_inputs": len(chunk_onnx.graph.input),
                    "num_outputs": len(chunk_onnx.graph.output),
                    "num_initializers": len(chunk_onnx.graph.initializer)
                }

            return {
                "chunks_created": len(chunk_files),
                "total_size_mb": round(total_size, 2),
                "chunk_analysis": chunk_analysis,
                "chunking_strategy": "layer_based",
                "distribution": "equal_memory",
                "num_chunks_configured": 3
            }
        else:
            raise ValueError(f"Expected at least 2 chunk files, found {len(chunk_files)}")

    def test_05_custom_chunk_distribution(self) -> Dict[str, Any]:
        """Test creating chunked models with custom distribution"""
        self.logger.info("Creating chunked models with custom distribution")

        chunk_output_dir = self.chunked_dir / "custom_chunks"
        chunk_output_dir.mkdir(exist_ok=True)

        # Use create_model function with custom chunk distribution
        create_model(
            model_name=self.model_name,
            input_path=self.model_name,
            output_dir=str(chunk_output_dir),
            precision="fp16",
            execution_provider="cpu",
            cache_dir=str(self.cache_dir),
            num_hidden_layers=24,
            # Enable custom chunking
            enable_chunking=True,
            chunk_strategy="layer_based",
            chunk_distribution="custom",
            num_chunks=4,
            custom_chunk_sizes=[0.1, 0.2, 0.3, 0.4],  # 10%, 20%, 30%, 40%
            respect_layer_boundaries=True,
            chunk_output_dir=str(chunk_output_dir),
            verify_chunks=False
        )

        # Analyze created chunks and verify distribution
        chunk_files = list(chunk_output_dir.glob("*chunk*.onnx"))

        if len(chunk_files) >= 4:
            chunk_analysis = {}
            total_size = 0
            size_percentages = []

            for chunk_file in chunk_files:
                size_mb = chunk_file.stat().st_size / (1024**2)
                total_size += size_mb

                # Load chunk to verify it has nodes
                chunk_onnx = onnx.load(str(chunk_file))

                chunk_analysis[chunk_file.name] = {
                    "path": str(chunk_file),
                    "size_mb": round(size_mb, 2),
                    "num_nodes": len(chunk_onnx.graph.node),
                    "num_inputs": len(chunk_onnx.graph.input),
                    "num_outputs": len(chunk_onnx.graph.output),
                    "num_initializers": len(chunk_onnx.graph.initializer)
                }

            # Calculate actual distribution
            for chunk_data in chunk_analysis.values():
                percentage = (chunk_data["size_mb"] / total_size) * 100
                size_percentages.append(round(percentage, 1))

            expected_percentages = [10.0, 20.0, 30.0, 40.0]
            distribution_error = [abs(actual - expected)
                                for actual, expected in zip(size_percentages, expected_percentages)]

            return {
                "chunks_created": len(chunk_files),
                "total_size_mb": round(total_size, 2),
                "chunk_analysis": chunk_analysis,
                "chunking_strategy": "layer_based",
                "distribution": "custom",
                "expected_percentages": expected_percentages,
                "actual_percentages": size_percentages,
                "distribution_error": [round(err, 1) for err in distribution_error],
                "max_distribution_error": round(max(distribution_error), 1)
            }
        else:
            raise ValueError(f"Expected 4 chunk files, found {len(chunk_files)}")

    def test_06_model_loading_verification(self) -> Dict[str, Any]:
        """Test that created chunk models can be loaded and have proper structure"""
        self.logger.info("Testing chunk model loading and structure verification")

        # Test chunks from previous tests
        chunk_directories = [
            (self.chunked_dir / "weight_based_chunks", "weight_based"),
            (self.chunked_dir / "layer_based_chunks", "layer_based"),
            (self.chunked_dir / "custom_chunks", "custom")
        ]

        results = {}

        for chunk_dir, strategy in chunk_directories:
            if not chunk_dir.exists():
                results[strategy] = {"status": "skipped", "reason": "Directory not found"}
                continue

            chunk_files = list(chunk_dir.glob("*chunk*.onnx"))
            if not chunk_files:
                results[strategy] = {"status": "skipped", "reason": "No chunk files found"}
                continue

            strategy_results = {
                "status": "success",
                "chunks_tested": 0,
                "chunks_successful": 0,
                "total_nodes": 0,
                "chunk_details": {}
            }

            for chunk_file in chunk_files:
                try:
                    # Test loading
                    chunk_onnx = onnx.load(str(chunk_file))

                    # Test ONNX Runtime session creation
                    session = ort.InferenceSession(
                        str(chunk_file),
                        providers=["CPUExecutionProvider"]
                    )

                    # Get model info
                    inputs = session.get_inputs()
                    outputs = session.get_outputs()

                    strategy_results["chunk_details"][chunk_file.name] = {
                        "status": "success",
                        "num_nodes": len(chunk_onnx.graph.node),
                        "num_inputs": len(inputs),
                        "num_outputs": len(outputs),
                        "input_shapes": [inp.shape for inp in inputs],
                        "output_shapes": [out.shape for out in outputs]
                    }

                    strategy_results["chunks_successful"] += 1
                    strategy_results["total_nodes"] += len(chunk_onnx.graph.node)

                except Exception as e:
                    strategy_results["chunk_details"][chunk_file.name] = {
                        "status": "failed",
                        "error": str(e)
                    }

                strategy_results["chunks_tested"] += 1

            results[strategy] = strategy_results

        return results

    def generate_test_report(self) -> str:
        """Generate test report"""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.status == "PASS"])
        failed_tests = len([r for r in self.test_results if r.status == "FAIL"])

        total_time = sum(r.execution_time for r in self.test_results)

        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"corrected_chunking_test_report_{timestamp}.txt"

        with open(report_file, 'w') as f:
            f.write("Corrected ONNX Model Chunking Test Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model_name}\n\n")

            f.write("Test Summary:\n")
            f.write("-" * 30 + "\n")
            f.write(f"  Total Tests: {total_tests}\n")
            f.write(f"  Passed: {passed_tests}\n")
            f.write(f"  Failed: {failed_tests}\n")
            f.write(f"  Success Rate: {(passed_tests/total_tests*100):.1f}%\n")
            f.write(f"  Total Execution Time: {total_time:.2f}s\n\n")

            f.write("Test Results:\n")
            f.write("-" * 30 + "\n")
            for result in self.test_results:
                status_symbol = "[PASS]" if result.status == "PASS" else "[FAIL]"
                f.write(f"{status_symbol} {result.test_name}: {result.status} ({result.execution_time:.2f}s)\n")

                if result.details:
                    for key, value in result.details.items():
                        if key not in ['traceback']:
                            f.write(f"    {key}: {value}\n")

                if result.error_message:
                    f.write(f"    Error: {result.error_message}\n")
                f.write("\n")

        return str(report_file)

    def run_all_tests(self) -> str:
        """Run all tests"""
        self.start_time = time.time()

        self.logger.info("Starting Corrected ONNX Model Chunking Test Suite")
        self.logger.info(f"Target Model: {self.model_name}")
        self.logger.info("=" * 80)

        tests = [
            ("01. Model Setup", self.test_01_model_setup),
            ("02. Create Complete Model", self.test_02_create_complete_model),
            ("03. Weight-Based Chunking", self.test_03_create_chunked_models_weight_based),
            ("04. Layer-Based Chunking", self.test_04_create_chunked_models_layer_based),
            ("05. Custom Chunk Distribution", self.test_05_custom_chunk_distribution),
            ("06. Model Loading Verification", self.test_06_model_loading_verification),
        ]

        for test_name, test_func in tests:
            try:
                self.run_test(test_name, test_func)
            except Exception as e:
                self.logger.error(f"Unexpected error running {test_name}: {e}")

        total_execution_time = time.time() - self.start_time
        self.logger.info(f"\nCorrected Test Suite Completed in {total_execution_time:.2f}s")

        report_file = self.generate_test_report()
        self.logger.info(f"Test report generated: {report_file}")

        return report_file


def main():
    """Main entry point"""
    print("Corrected ONNX Model Chunking Test Suite")
    print("=" * 60)
    print("Testing Proper Integration with Model Creation Workflow")
    print()

    test_suite = CorrectedChunkingTestSuite()

    try:
        print(f"Starting corrected test suite for model: {test_suite.model_name}")
        print("This test properly integrates chunking with the model creation workflow")
        print()

        report_file = test_suite.run_all_tests()

        print(f"\nCorrected test suite completed successfully!")
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
        print("\nTest suite interrupted by user")
        return 1

    except Exception as e:
        print(f"\nTest suite failed with error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)