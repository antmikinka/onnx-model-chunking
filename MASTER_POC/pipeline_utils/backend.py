#!/usr/bin/env python3
"""
backend.py - Multi-Backend Inference Abstraction Layer

This module provides a unified interface for different inference backends,
enabling the pipeline to work with ONNX Runtime, llama.cpp, or PyTorch.

Usage:
    from pipeline_utils.backend import get_backend, BackendType, DeviceType
    
    backend = get_backend(BackendType.ONNX_RUNTIME)
    backend.load(model_path, device=DeviceType.GPU)
    output = backend.generate("Hello", max_tokens=50)
"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
import os


# =============================================================================
# ENUMS
# =============================================================================

class BackendType(Enum):
    """Supported inference backends."""
    ONNX_RUNTIME = "onnx_runtime"   # Current implementation (ONNX models)
    LLAMA_CPP = "llama_cpp"         # llama.cpp via llama-cpp-python (GGUF)
    TORCH = "torch"                 # Direct PyTorch (SafeTensors)
    VLLM = "vllm"                   # High-throughput server mode


class DeviceType(Enum):
    """Target device for inference."""
    CPU = "cpu"
    GPU = "gpu"       # CUDA/ROCm/MPS
    NPU = "npu"       # VitisAI/OpenVINO/CoreML
    AUTO = "auto"     # Best available


class ModelFormat(Enum):
    """Model file formats."""
    ONNX = "onnx"
    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    PYTORCH_BIN = "pytorch_bin"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BackendConfig:
    """Configuration for inference backends."""
    device: DeviceType = DeviceType.AUTO
    
    # GPU offloading (for hybrid execution)
    n_gpu_layers: int = -1  # -1 = all, 0 = none, N = first N layers
    
    # Context and generation
    n_ctx: int = 4096
    n_batch: int = 512
    
    # Threading
    n_threads: int = 0  # 0 = auto
    
    # Quantization (for llama.cpp)
    quantization: Optional[str] = None  # Q4_K_M, Q8_0, etc.
    
    # ONNX-specific
    onnx_providers: List[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    onnx_optimization_level: str = "DISABLE"
    
    # Memory
    low_memory: bool = False


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================

class InferenceBackend(ABC):
    """
    Abstract base class for inference backends.
    
    All backends must implement load(), generate(), and cleanup().
    This enables swapping between ONNX, llama.cpp, PyTorch, etc.
    """
    
    def __init__(self, config: Optional[BackendConfig] = None):
        self.config = config or BackendConfig()
        self.model = None
        self.is_loaded = False
        self.model_path: Optional[str] = None
    
    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Return the backend type enum."""
        pass
    
    @property
    @abstractmethod
    def supported_formats(self) -> List[ModelFormat]:
        """Return list of supported model formats."""
        pass
    
    @abstractmethod
    def load(self, model_path: str) -> None:
        """
        Load model from path.
        
        Args:
            model_path: Path to model file or directory
        """
        pass
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        pass
    
    def cleanup(self) -> None:
        """Release resources. Override if needed."""
        self.model = None
        self.is_loaded = False
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(loaded={self.is_loaded}, device={self.config.device.value})"


# =============================================================================
# ONNX RUNTIME BACKEND (Wraps existing SessionManager logic)
# =============================================================================

class OnnxRuntimeBackend(InferenceBackend):
    """
    ONNX Runtime backend for split or whole ONNX models.
    
    This integrates with the existing SessionManager infrastructure.
    """
    
    @property
    def backend_type(self) -> BackendType:
        return BackendType.ONNX_RUNTIME
    
    @property
    def supported_formats(self) -> List[ModelFormat]:
        return [ModelFormat.ONNX]
    
    def load(self, model_path: str) -> None:
        """Load ONNX model(s) from path."""
        import onnxruntime as ort
        
        self.model_path = model_path
        
        # Determine if split model (directory) or single model (file)
        if os.path.isdir(model_path):
            # Use SessionManager for split models
            from .session_manager import SessionManager, SessionConfig
            
            config = SessionConfig(
                providers=self.config.onnx_providers,
                optimization_level=self.config.onnx_optimization_level,
            )
            self.model = SessionManager(model_path, config)
            self.model.load_all()
        else:
            # Single ONNX file
            opts = ort.SessionOptions()
            self.model = ort.InferenceSession(
                model_path,
                sess_options=opts,
                providers=self.config.onnx_providers
            )
        
        self.is_loaded = True
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text using ONNX model.
        
        Note: For split models, this delegates to SequentialModelPipeline.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # For split models, we need the full pipeline
        # This is a simplified interface - actual generation requires tokenizer etc.
        raise NotImplementedError(
            "Direct generate() not implemented for ONNX. "
            "Use SequentialModelPipeline for split model inference."
        )
    
    def get_memory_usage(self) -> Dict[str, Any]:
        if hasattr(self.model, 'get_memory_usage'):
            return self.model.get_memory_usage()
        return {"unknown": True}


# =============================================================================
# LLAMA.CPP BACKEND
# =============================================================================

class LlamaCppBackend(InferenceBackend):
    """
    llama.cpp backend via llama-cpp-python bindings.
    
    Supports GGUF format models with GPU offloading.
    """
    
    @property
    def backend_type(self) -> BackendType:
        return BackendType.LLAMA_CPP
    
    @property
    def supported_formats(self) -> List[ModelFormat]:
        return [ModelFormat.GGUF]
    
    def load(self, model_path: str) -> None:
        """Load GGUF model."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for LlamaCppBackend. "
                "Install with: pip install llama-cpp-python"
            )
        
        self.model_path = model_path
        
        # Determine GPU layers based on device config
        n_gpu_layers = 0
        if self.config.device == DeviceType.GPU:
            n_gpu_layers = self.config.n_gpu_layers if self.config.n_gpu_layers >= 0 else 999
        elif self.config.device == DeviceType.AUTO:
            n_gpu_layers = self.config.n_gpu_layers if self.config.n_gpu_layers >= 0 else -1
        
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=self.config.n_ctx,
            n_batch=self.config.n_batch,
            n_threads=self.config.n_threads if self.config.n_threads > 0 else None,
            verbose=False,
        )
        
        self.is_loaded = True
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text using llama.cpp."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=False,
        )
        
        return output["choices"][0]["text"]
    
    def get_memory_usage(self) -> Dict[str, Any]:
        # llama.cpp doesn't expose direct memory stats
        return {
            "backend": "llama_cpp",
            "n_gpu_layers": self.config.n_gpu_layers,
            "n_ctx": self.config.n_ctx,
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_backend(
    backend_type: BackendType,
    config: Optional[BackendConfig] = None
) -> InferenceBackend:
    """
    Factory function to get appropriate backend.
    
    Args:
        backend_type: Type of backend to create
        config: Optional configuration
        
    Returns:
        Configured InferenceBackend instance
    """
    backends = {
        BackendType.ONNX_RUNTIME: OnnxRuntimeBackend,
        BackendType.LLAMA_CPP: LlamaCppBackend,
    }
    
    if backend_type not in backends:
        raise ValueError(f"Unsupported backend: {backend_type}. Available: {list(backends.keys())}")
    
    return backends[backend_type](config)


def detect_model_format(model_path: str) -> ModelFormat:
    """
    Detect model format from path.
    
    Args:
        model_path: Path to model file or directory
        
    Returns:
        Detected ModelFormat
    """
    if os.path.isdir(model_path):
        # Check for split ONNX
        if os.path.exists(os.path.join(model_path, "model_info.json")):
            return ModelFormat.ONNX
        # Check for SafeTensors
        if any(f.endswith(".safetensors") for f in os.listdir(model_path)):
            return ModelFormat.SAFETENSORS
    else:
        ext = os.path.splitext(model_path)[1].lower()
        if ext == ".onnx":
            return ModelFormat.ONNX
        elif ext == ".gguf":
            return ModelFormat.GGUF
        elif ext == ".safetensors":
            return ModelFormat.SAFETENSORS
        elif ext == ".bin":
            return ModelFormat.PYTORCH_BIN
    
    raise ValueError(f"Cannot detect format for: {model_path}")


def get_backend_for_model(
    model_path: str,
    config: Optional[BackendConfig] = None
) -> InferenceBackend:
    """
    Automatically select and configure backend based on model format.
    
    Args:
        model_path: Path to model
        config: Optional configuration
        
    Returns:
        Appropriate backend for the model format
    """
    fmt = detect_model_format(model_path)
    
    format_to_backend = {
        ModelFormat.ONNX: BackendType.ONNX_RUNTIME,
        ModelFormat.GGUF: BackendType.LLAMA_CPP,
        # SafeTensors/PyTorch would need torch backend (not implemented yet)
    }
    
    backend_type = format_to_backend.get(fmt)
    if not backend_type:
        raise ValueError(f"No backend available for format: {fmt}")
    
    return get_backend(backend_type, config)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "BackendType",
    "DeviceType",
    "ModelFormat",
    # Classes
    "BackendConfig",
    "InferenceBackend",
    "OnnxRuntimeBackend",
    "LlamaCppBackend",
    # Functions
    "get_backend",
    "detect_model_format",
    "get_backend_for_model",
]
