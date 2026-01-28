"""
Unified Configuration System for Dual Chunking Approach

This module provides a comprehensive configuration system that supports both
PyTorch-level and ONNX graph-level chunking methods with backward compatibility.
"""

import json
import os
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, field, fields
from enum import Enum


class ChunkingMethod(Enum):
    """Supported chunking methods"""
    PYTORCH_LEVEL = "pytorch_level"
    ONNX_GRAPH_LEVEL = "onnx_graph_level"


class ChunkStrategy(Enum):
    """Strategies for chunk distribution"""
    BY_LAYERS = "by_layers"
    BY_WEIGHTS = "by_weights"
    LAYER_BASED = "layer_based"
    WEIGHT_BASED = "weight_based"


class DistributionType(Enum):
    """Distribution types for chunks"""
    EQUAL_LAYERS = "equal_layers"
    EQUAL_MEMORY = "equal_memory"
    EQUAL_WEIGHTS = "equal_weights"
    CUSTOM = "custom"


@dataclass
class CommonChunkingConfig:
    """Common configuration options applicable to all chunking methods"""
    output_directory: str = "./chunked_models"
    save_original_model: bool = True
    verify_chunks: bool = False
    min_psnr_threshold: float = 35.0
    save_metadata: bool = True
    verbose: bool = True


@dataclass
class AnalyticsConfig:
    """Configuration for comprehensive analytics and benchmarking system"""
    enable_monitoring: bool = True
    sampling_interval: float = 1.0
    max_history: int = 3600
    system_resources: List[str] = field(default_factory=lambda: ["cpu", "memory", "disk", "gpu"])
    timing_phases: List[str] = field(default_factory=lambda: ["loading", "conversion", "chunking", "saving"])
    detailed_metrics: bool = True
    real_time_updates: bool = True
    output_format: List[str] = field(default_factory=lambda: ["json", "summary"])
    enable_alerts: bool = True
    save_snapshots: bool = False
    snapshot_interval: int = 10
    performance_alerts: Dict[str, float] = field(default_factory=lambda: {
        "memory_percent": 80.0,
        "cpu_percent": 90.0,
        "disk_usage_percent": 90.0,
        "conversion_time_seconds": 300.0,
        "memory_usage_gb": 8.0
    })


@dataclass
class PyTorchLevelConfig:
    """Configuration for PyTorch-level chunking method"""
    chunk_strategy: str = ChunkStrategy.BY_LAYERS.value
    layer_distribution: Optional[List[int]] = None
    include_embedding_in_first: bool = True
    include_lm_head_in_last: bool = True
    preserve_intermediate_outputs: bool = True
    chunk_overlap_ratio: float = 0.0
    balance_memory_usage: bool = True


@dataclass
class ONNXGraphLevelConfig:
    """Configuration for ONNX graph-level chunking method"""
    strategy: str = ChunkStrategy.WEIGHT_BASED.value
    distribution: str = DistributionType.EQUAL_MEMORY.value
    num_chunks: int = 2
    respect_layer_boundaries: bool = True
    custom_chunk_sizes: Optional[List[float]] = None
    chunk_split_ratio: float = 0.5  # Legacy support for 2-chunk splitting


@dataclass
class UnifiedChunkingConfig:
    """Unified configuration for dual chunking approach"""
    method: str = ChunkingMethod.ONNX_GRAPH_LEVEL.value
    pytorch_level: PyTorchLevelConfig = field(default_factory=PyTorchLevelConfig)
    onnx_graph_level: ONNXGraphLevelConfig = field(default_factory=ONNXGraphLevelConfig)
    common: CommonChunkingConfig = field(default_factory=CommonChunkingConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)


class ConfigurationProcessor:
    """
    Processes and validates unified chunking configuration.

    Handles both new unified format and legacy format for backward compatibility.
    """

    def __init__(self):
        self.config = None
        self.legacy_mode = False

    def process_configuration(self, extra_options: Dict[str, Any]) -> UnifiedChunkingConfig:
        """
        Process extra_options and return unified configuration.

        Args:
            extra_options: Dictionary containing configuration options

        Returns:
            UnifiedChunkingConfig: Processed configuration object

        Raises:
            ValueError: If configuration is invalid
        """
        # Check if this is legacy format
        if self._is_legacy_format(extra_options):
            return self._process_legacy_configuration(extra_options)
        else:
            return self._process_unified_configuration(extra_options)

    def _is_legacy_format(self, extra_options: Dict[str, Any]) -> bool:
        """Check if configuration is in legacy format"""
        legacy_indicators = [
            "enable_chunking" in extra_options,
            "chunk_strategy" in extra_options and "chunking" not in extra_options,
            "num_chunks" in extra_options and "chunking" not in extra_options
        ]
        return any(legacy_indicators)

    def _process_legacy_configuration(self, extra_options: Dict[str, Any]) -> UnifiedChunkingConfig:
        """Process legacy configuration format"""
        self.legacy_mode = True

        # Create unified config from legacy options
        config = UnifiedChunkingConfig()

        # Legacy format always uses ONNX graph-level chunking
        config.method = ChunkingMethod.ONNX_GRAPH_LEVEL.value

        # Map legacy options to ONNX graph-level config
        onnx_config = config.onnx_graph_level

        if "chunk_strategy" in extra_options:
            strategy = extra_options["chunk_strategy"]
            if strategy in ["layer_based", "weight_based"]:
                onnx_config.strategy = strategy
            else:
                raise ValueError(f"Legacy chunk_strategy '{strategy}' is not supported. "
                               f"Supported: layer_based, weight_based")

        if "chunk_distribution" in extra_options:
            distribution = extra_options["chunk_distribution"]
            if distribution in ["equal_layers", "equal_memory", "custom"]:
                onnx_config.distribution = distribution
            else:
                raise ValueError(f"Legacy chunk_distribution '{distribution}' is not supported. "
                               f"Supported: equal_layers, equal_memory, custom")

        if "num_chunks" in extra_options:
            num_chunks = extra_options["num_chunks"]
            if num_chunks >= 2:
                onnx_config.num_chunks = num_chunks
            else:
                raise ValueError(f"Legacy num_chunks must be >= 2, got {num_chunks}")

        if "chunk_split_ratio" in extra_options:
            ratio = extra_options["chunk_split_ratio"]
            if 0.0 < ratio < 1.0:
                onnx_config.chunk_split_ratio = ratio
            else:
                raise ValueError(f"Legacy chunk_split_ratio must be between 0 and 1, got {ratio}")

        if "custom_chunk_sizes" in extra_options:
            onnx_config.custom_chunk_sizes = extra_options["custom_chunk_sizes"]

        if "respect_layer_boundaries" in extra_options:
            onnx_config.respect_layer_boundaries = extra_options["respect_layer_boundaries"]

        # Map common options
        common_config = config.common

        if "chunk_output_dir" in extra_options:
            common_config.output_directory = extra_options["chunk_output_dir"]

        if "verify_chunks" in extra_options:
            common_config.verify_chunks = extra_options["verify_chunks"]

        if "min_psnr_threshold" in extra_options:
            common_config.min_psnr_threshold = extra_options["min_psnr_threshold"]

        # Enable chunking if legacy enable_chunking is present
        if extra_options.get("enable_chunking", False):
            # Legacy mode is enabled, no additional processing needed
            pass

        self.config = config
        self._validate_configuration()
        return config

    def _process_unified_configuration(self, extra_options: Dict[str, Any]) -> UnifiedChunkingConfig:
        """Process new unified configuration format"""
        self.legacy_mode = False

        chunking_config = extra_options.get("chunking", {})

        if not chunking_config:
            # No chunking configuration specified
            config = UnifiedChunkingConfig()
            config.method = ChunkingMethod.ONNX_GRAPH_LEVEL.value
            self.config = config
            return config

        # Create base configuration
        config = UnifiedChunkingConfig()

        # Process chunking method
        if "method" in chunking_config:
            method = chunking_config["method"]
            if method in [ChunkingMethod.PYTORCH_LEVEL.value, ChunkingMethod.ONNX_GRAPH_LEVEL.value]:
                config.method = method
            else:
                raise ValueError(f"Unknown chunking method: {method}. "
                               f"Supported: {ChunkingMethod.PYTORCH_LEVEL.value}, {ChunkingMethod.ONNX_GRAPH_LEVEL.value}")

        # Process method-specific configurations
        if "pytorch_level" in chunking_config:
            config.pytorch_level = self._process_pytorch_level_config(chunking_config["pytorch_level"])

        if "onnx_graph_level" in chunking_config:
            config.onnx_graph_level = self._process_onnx_graph_level_config(chunking_config["onnx_graph_level"])

        # Process common configuration
        if "common" in chunking_config:
            config.common = self._process_common_config(chunking_config["common"])

        # Process analytics configuration
        if "analytics" in chunking_config:
            config.analytics = self._process_analytics_config(chunking_config["analytics"])

        self.config = config
        self._validate_configuration()
        return config

    def _process_pytorch_level_config(self, pytorch_config: Dict[str, Any]) -> PyTorchLevelConfig:
        """Process PyTorch-level chunking configuration"""
        config = PyTorchLevelConfig()

        # Validate and set chunk strategy
        if "chunk_strategy" in pytorch_config:
            strategy = pytorch_config["chunk_strategy"]
            if strategy in [ChunkStrategy.BY_LAYERS.value, ChunkStrategy.BY_WEIGHTS.value]:
                config.chunk_strategy = strategy
            else:
                raise ValueError(f"Unknown PyTorch chunk strategy: {strategy}. "
                               f"Supported: {ChunkStrategy.BY_LAYERS.value}, {ChunkStrategy.BY_WEIGHTS.value}")

        # Validate and set layer distribution
        if "layer_distribution" in pytorch_config:
            distribution = pytorch_config["layer_distribution"]
            if distribution is not None:
                if not isinstance(distribution, list) or not all(isinstance(x, int) for x in distribution):
                    raise ValueError("layer_distribution must be a list of integers")
                if sum(distribution) <= 0:
                    raise ValueError("layer_distribution must sum to a positive number")
                config.layer_distribution = distribution

        # Set other PyTorch-level options
        for key, value in pytorch_config.items():
            if hasattr(config, key) and key not in ["chunk_strategy", "layer_distribution"]:
                setattr(config, key, value)

        return config

    def _process_onnx_graph_level_config(self, onnx_config: Dict[str, Any]) -> ONNXGraphLevelConfig:
        """Process ONNX graph-level chunking configuration"""
        config = ONNXGraphLevelConfig()

        # Validate and set strategy
        if "strategy" in onnx_config:
            strategy = onnx_config["strategy"]
            if strategy in [ChunkStrategy.LAYER_BASED.value, ChunkStrategy.WEIGHT_BASED.value]:
                config.strategy = strategy
            else:
                raise ValueError(f"Unknown ONNX graph strategy: {strategy}. "
                               f"Supported: {ChunkStrategy.LAYER_BASED.value}, {ChunkStrategy.WEIGHT_BASED.value}")

        # Validate and set distribution
        if "distribution" in onnx_config:
            distribution = onnx_config["distribution"]
            if distribution in [dt.value for dt in DistributionType]:
                config.distribution = distribution
            else:
                raise ValueError(f"Unknown distribution type: {distribution}. "
                               f"Supported: {[dt.value for dt in DistributionType]}")

        # Validate and set num_chunks
        if "num_chunks" in onnx_config:
            num_chunks = onnx_config["num_chunks"]
            if isinstance(num_chunks, int) and num_chunks >= 2:
                config.num_chunks = num_chunks
            else:
                raise ValueError(f"num_chunks must be an integer >= 2, got {num_chunks}")

        # Validate and set custom chunk sizes
        if "custom_chunk_sizes" in onnx_config:
            custom_sizes = onnx_config["custom_chunk_sizes"]
            if custom_sizes is not None:
                if not isinstance(custom_sizes, list) or len(custom_sizes) != config.num_chunks:
                    raise ValueError(f"custom_chunk_sizes must be a list of length {config.num_chunks}")
                if not all(isinstance(x, (int, float)) for x in custom_sizes):
                    raise ValueError("custom_chunk_sizes must contain only numbers")
                if not all(x > 0 for x in custom_sizes):
                    raise ValueError("All custom_chunk_sizes must be positive")
                config.custom_chunk_sizes = custom_sizes

        # Set other ONNX graph-level options
        for key, value in onnx_config.items():
            if hasattr(config, key) and key not in ["strategy", "distribution", "num_chunks", "custom_chunk_sizes"]:
                setattr(config, key, value)

        return config

    def _process_common_config(self, common_config: Dict[str, Any]) -> CommonChunkingConfig:
        """Process common configuration options"""
        config = CommonChunkingConfig()

        # Validate and set output directory
        if "output_directory" in common_config:
            output_dir = common_config["output_directory"]
            if isinstance(output_dir, str):
                config.output_directory = output_dir
            else:
                raise ValueError("output_directory must be a string")

        # Validate and set PSNR threshold
        if "min_psnr_threshold" in common_config:
            threshold = common_config["min_psnr_threshold"]
            if isinstance(threshold, (int, float)) and threshold > 0:
                config.min_psnr_threshold = float(threshold)
            else:
                raise ValueError("min_psnr_threshold must be a positive number")

        # Set other common options
        for key, value in common_config.items():
            if hasattr(config, key) and key not in ["output_directory", "min_psnr_threshold"]:
                if isinstance(value, type(getattr(config, key))):
                    setattr(config, key, value)
                else:
                    raise ValueError(f"{key} must be of type {type(getattr(config, key)).__name__}")

        return config

    def _process_analytics_config(self, analytics_config: Dict[str, Any]) -> AnalyticsConfig:
        """Process analytics configuration"""
        config = AnalyticsConfig()

        # Validate and set sampling interval
        if "sampling_interval" in analytics_config:
            interval = analytics_config["sampling_interval"]
            if isinstance(interval, (int, float)) and interval > 0:
                config.sampling_interval = float(interval)
            else:
                raise ValueError("sampling_interval must be a positive number")

        # Validate and set max history
        if "max_history" in analytics_config:
            history = analytics_config["max_history"]
            if isinstance(history, int) and history > 0:
                config.max_history = history
            else:
                raise ValueError("max_history must be a positive integer")

        # Validate and set system resources
        if "system_resources" in analytics_config:
            resources = analytics_config["system_resources"]
            valid_resources = ["cpu", "memory", "disk", "gpu", "network"]
            if isinstance(resources, list) and all(r in valid_resources for r in resources):
                config.system_resources = resources
            else:
                raise ValueError(f"system_resources must be a list of valid resources: {valid_resources}")

        # Validate and set output format
        if "output_format" in analytics_config:
            formats = analytics_config["output_format"]
            valid_formats = ["json", "csv", "html", "summary"]
            if isinstance(formats, list) and all(f in valid_formats for f in formats):
                config.output_format = formats
            else:
                raise ValueError(f"output_format must be a list of valid formats: {valid_formats}")

        # Validate and set performance alerts
        if "performance_alerts" in analytics_config:
            alerts = analytics_config["performance_alerts"]
            if isinstance(alerts, dict):
                for key, value in alerts.items():
                    if isinstance(value, (int, float)) and value >= 0:
                        config.performance_alerts[key] = float(value)
                    else:
                        raise ValueError(f"performance_alerts.{key} must be a non-negative number")
            else:
                raise ValueError("performance_alerts must be a dictionary")

        # Set other analytics options
        for key, value in analytics_config.items():
            if hasattr(config, key) and key not in [
                "sampling_interval", "max_history", "system_resources",
                "output_format", "performance_alerts"
            ]:
                if isinstance(value, type(getattr(config, key))):
                    setattr(config, key, value)
                else:
                    raise ValueError(f"{key} must be of type {type(getattr(config, key)).__name__}")

        return config

    def _validate_configuration(self) -> None:
        """Validate the processed configuration"""
        if self.config is None:
            raise ValueError("No configuration to validate")

        # Method-specific validation
        if self.config.method == ChunkingMethod.PYTORCH_LEVEL.value:
            self._validate_pytorch_level_config()
        elif self.config.method == ChunkingMethod.ONNX_GRAPH_LEVEL.value:
            self._validate_onnx_graph_level_config()

        # Common validation
        self._validate_common_config()

        # Analytics validation
        self._validate_analytics_config()

    def _validate_analytics_config(self) -> None:
        """Validate analytics configuration"""
        analytics_config = self.config.analytics

        # Validate sampling interval
        if not isinstance(analytics_config.sampling_interval, (int, float)) or analytics_config.sampling_interval <= 0:
            raise ValueError("analytics.sampling_interval must be a positive number")

        # Validate max history
        if not isinstance(analytics_config.max_history, int) or analytics_config.max_history <= 0:
            raise ValueError("analytics.max_history must be a positive integer")

        # Validate system resources
        valid_resources = ["cpu", "memory", "disk", "gpu", "network"]
        if not isinstance(analytics_config.system_resources, list) or not all(r in valid_resources for r in analytics_config.system_resources):
            raise ValueError(f"analytics.system_resources must be a list of valid resources: {valid_resources}")

        # Validate output format
        valid_formats = ["json", "csv", "html", "summary"]
        if not isinstance(analytics_config.output_format, list) or not all(f in valid_formats for f in analytics_config.output_format):
            raise ValueError(f"analytics.output_format must be a list of valid formats: {valid_formats}")

        # Validate performance alerts
        if not isinstance(analytics_config.performance_alerts, dict):
            raise ValueError("analytics.performance_alerts must be a dictionary")

        for key, value in analytics_config.performance_alerts.items():
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"analytics.performance_alerts.{key} must be a non-negative number")

    def _validate_pytorch_level_config(self) -> None:
        """Validate PyTorch-level configuration"""
        pytorch_config = self.config.pytorch_level

        # If using by_layers strategy, layer_distribution is required
        if pytorch_config.chunk_strategy == ChunkStrategy.BY_LAYERS.value:
            if pytorch_config.layer_distribution is None:
                raise ValueError("layer_distribution is required when using 'by_layers' strategy")

        # Validate chunk overlap ratio
        if not (0.0 <= pytorch_config.chunk_overlap_ratio < 1.0):
            raise ValueError("chunk_overlap_ratio must be between 0.0 and 1.0 (exclusive of 1.0)")

    def _validate_onnx_graph_level_config(self) -> None:
        """Validate ONNX graph-level configuration"""
        onnx_config = self.config.onnx_graph_level

        # Validate custom chunk sizes
        if onnx_config.custom_chunk_sizes is not None:
            if len(onnx_config.custom_chunk_sizes) != onnx_config.num_chunks:
                raise ValueError(f"custom_chunk_sizes length ({len(onnx_config.custom_chunk_sizes)}) "
                               f"must equal num_chunks ({onnx_config.num_chunks})")

        # Validate chunk split ratio
        if not (0.0 < onnx_config.chunk_split_ratio < 1.0):
            raise ValueError("chunk_split_ratio must be between 0.0 and 1.0 (exclusive)")

    def _validate_common_config(self) -> None:
        """Validate common configuration"""
        common_config = self.config.common

        # Validate output directory
        if not isinstance(common_config.output_directory, str) or len(common_config.output_directory.strip()) == 0:
            raise ValueError("output_directory must be a non-empty string")

        # Validate PSNR threshold
        if not isinstance(common_config.min_psnr_threshold, (int, float)) or common_config.min_psnr_threshold <= 0:
            raise ValueError("min_psnr_threshold must be a positive number")

    def is_legacy_mode(self) -> bool:
        """Check if configuration was processed in legacy mode"""
        return self.legacy_mode

    def get_legacy_style_config(self) -> Dict[str, Any]:
        """
        Convert unified configuration back to legacy format for backward compatibility.

        Returns:
            Dict[str, Any]: Legacy-style configuration
        """
        if self.config is None:
            raise ValueError("No configuration available")

        if self.config.method == ChunkingMethod.ONNX_GRAPH_LEVEL.value:
            onnx_config = self.config.onnx_graph_level
            common_config = self.config.common

            legacy_config = {
                "enable_chunking": True,
                "chunk_strategy": onnx_config.strategy,
                "chunk_distribution": onnx_config.distribution,
                "num_chunks": onnx_config.num_chunks,
                "respect_layer_boundaries": onnx_config.respect_layer_boundaries,
                "chunk_output_dir": common_config.output_directory,
                "verify_chunks": common_config.verify_chunks,
                "min_psnr_threshold": common_config.min_psnr_threshold
            }

            # Add optional legacy parameters
            if onnx_config.custom_chunk_sizes is not None:
                legacy_config["custom_chunk_sizes"] = onnx_config.custom_chunk_sizes

            if onnx_config.num_chunks == 2:
                legacy_config["chunk_split_ratio"] = onnx_config.chunk_split_ratio

            return legacy_config
        else:
            # PyTorch-level chunking doesn't have direct legacy equivalent
            raise ValueError("Cannot convert PyTorch-level configuration to legacy format")

    def save_configuration(self, file_path: str) -> None:
        """Save current configuration to JSON file"""
        if self.config is None:
            raise ValueError("No configuration to save")

        config_dict = {
            "chunking": {
                "method": self.config.method,
                "pytorch_level": self.config.pytorch_level.__dict__,
                "onnx_graph_level": self.config.onnx_graph_level.__dict__,
                "common": self.config.common.__dict__,
                "analytics": self.config.analytics.__dict__
            },
            "metadata": {
                "legacy_mode": self.legacy_mode,
                "version": "1.1.0",
                "analytics_enabled": True
            }
        }

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def get_method_specific_config(self) -> Tuple[str, Any]:
        """
        Get configuration specific to the selected method.

        Returns:
            Tuple[str, Any]: (method_name, method_specific_config)
        """
        if self.config is None:
            raise ValueError("No configuration available")

        if self.config.method == ChunkingMethod.PYTORCH_LEVEL.value:
            return ChunkingMethod.PYTORCH_LEVEL.value, self.config.pytorch_level
        elif self.config.method == ChunkingMethod.ONNX_GRAPH_LEVEL.value:
            return ChunkingMethod.ONNX_GRAPH_LEVEL.value, self.config.onnx_graph_level
        else:
            raise ValueError(f"Unknown method: {self.config.method}")


def create_sample_configurations() -> Dict[str, Dict[str, Any]]:
    """
    Create sample configurations for different chunking scenarios.

    Returns:
        Dict[str, Dict[str, Any]]: Sample configurations
    """
    return {
        "pytorch_level_equal_layers": {
            "chunking": {
                "method": "pytorch_level",
                "pytorch_level": {
                    "chunk_strategy": "by_layers",
                    "layer_distribution": [8, 8, 8],
                    "include_embedding_in_first": True,
                    "include_lm_head_in_last": True,
                    "balance_memory_usage": True
                },
                "common": {
                    "output_directory": "./chunked_models",
                    "verify_chunks": True,
                    "min_psnr_threshold": 35.0
                }
            }
        },

        "onnx_graph_level_weight_based": {
            "chunking": {
                "method": "onnx_graph_level",
                "onnx_graph_level": {
                    "strategy": "weight_based",
                    "distribution": "equal_memory",
                    "num_chunks": 3,
                    "respect_layer_boundaries": True
                },
                "common": {
                    "output_directory": "./chunked_models",
                    "verify_chunks": True,
                    "min_psnr_threshold": 40.0
                },
                "analytics": {
                    "enable_monitoring": True,
                    "detailed_metrics": True,
                    "real_time_updates": True,
                    "output_format": ["json", "summary"],
                    "performance_alerts": {
                        "memory_percent": 85.0,
                        "cpu_percent": 95.0,
                        "conversion_time_seconds": 600.0
                    }
                }
            }
        },

        "analytics_detailed": {
            "chunking": {
                "method": "onnx_graph_level",
                "onnx_graph_level": {
                    "strategy": "weight_based",
                    "num_chunks": 2
                },
                "common": {
                    "output_directory": "./chunked_models",
                    "verify_chunks": True,
                    "verbose": True
                },
                "analytics": {
                    "enable_monitoring": True,
                    "sampling_interval": 0.5,
                    "max_history": 7200,
                    "system_resources": ["cpu", "memory", "disk", "gpu"],
                    "timing_phases": ["loading", "conversion", "chunking", "saving"],
                    "detailed_metrics": True,
                    "real_time_updates": True,
                    "output_format": ["json", "csv", "summary"],
                    "enable_alerts": True,
                    "save_snapshots": True,
                    "snapshot_interval": 5,
                    "performance_alerts": {
                        "memory_percent": 80.0,
                        "cpu_percent": 90.0,
                        "disk_usage_percent": 85.0,
                        "conversion_time_seconds": 300.0,
                        "memory_usage_gb": 16.0
                    }
                }
            }
        },

        "minimal_analytics": {
            "chunking": {
                "method": "onnx_graph_level",
                "onnx_graph_level": {
                    "strategy": "layer_based",
                    "num_chunks": 2
                },
                "common": {
                    "output_directory": "./chunked_models",
                    "verify_chunks": False
                },
                "analytics": {
                    "enable_monitoring": False,
                    "detailed_metrics": False,
                    "real_time_updates": False,
                    "output_format": ["summary"],
                    "enable_alerts": False
                }
            }
        },

        "legacy_format_example": {
            "enable_chunking": True,
            "chunk_strategy": "layer_based",
            "chunk_distribution": "equal_layers",
            "num_chunks": 3,
            "respect_layer_boundaries": True,
            "chunk_output_dir": "./chunked_models",
            "verify_chunks": True,
            "min_psnr_threshold": 35.0
        }
    }


def print_configuration_examples():
    """Print example configurations for user reference"""
    examples = create_sample_configurations()

    print("=== Unified Chunking Configuration Examples ===\n")

    for name, config in examples.items():
        print(f"--- {name.upper().replace('_', ' ')} ---")
        print(json.dumps(config, indent=2))
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Example usage and validation
    print_configuration_examples()

    # Test configuration processing
    processor = ConfigurationProcessor()

    # Test legacy format
    legacy_config = {
        "enable_chunking": True,
        "chunk_strategy": "layer_based",
        "num_chunks": 3,
        "verify_chunks": True
    }

    print("Testing legacy configuration processing...")
    try:
        config = processor.process_configuration(legacy_config)
        print(f"✓ Legacy config processed successfully. Method: {config.method}")
        print(f"✓ Legacy mode: {processor.is_legacy_mode()}")
    except Exception as e:
        print(f"✗ Error processing legacy config: {e}")

    # Test unified format
    unified_config = {
        "chunking": {
            "method": "onnx_graph_level",
            "onnx_graph_level": {
                "strategy": "weight_based",
                "num_chunks": 3
            },
            "common": {
                "verify_chunks": True
            }
        }
    }

    print("\nTesting unified configuration processing...")
    try:
        config = processor.process_configuration(unified_config)
        print(f"✓ Unified config processed successfully. Method: {config.method}")
        print(f"✓ Legacy mode: {processor.is_legacy_mode()}")
    except Exception as e:
        print(f"✗ Error processing unified config: {e}")