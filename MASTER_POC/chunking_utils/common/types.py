"""
Shared data types for model chunking operations.

This module defines the core data types used across both ONNX and SafeTensors
splitting operations, providing a unified interface for configuration and results.
"""

from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


class ModelFormat(str, Enum):
    """
    Canonical model file formats for chunking and inference.

    ONNX, SafeTensors, GGUF, and PyTorch ``.bin`` checkpoints share this enum.
    """
    ONNX = "onnx"
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    PYTORCH_BIN = "pytorch_bin"


class TensorDtype(str, Enum):
    """
    Supported tensor data types with memory implications.
    
    Memory per parameter:
    - FLOAT32: 4 bytes
    - FLOAT16: 2 bytes
    - BFLOAT16: 2 bytes
    - INT8: 1 byte
    - INT4: 0.5 bytes
    - AUTO: Preserve original dtype
    """
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT8 = "int8"
    INT4 = "int4"
    AUTO = "auto"


class MemoryMode(str, Enum):
    """
    Memory loading strategies for model splitting.
    
    - EAGER: Load entire model into RAM (current behavior, fastest but high memory)
    - STREAMING: Process one tensor at a time (lowest memory, recommended for large models)
    - LAZY: Memory-map files for on-demand loading
    - AUTO: Use HuggingFace's device_map="auto" for optimal device placement
    """
    EAGER = "eager"
    STREAMING = "streaming"
    LAZY = "lazy"
    AUTO = "auto"


class ComponentType(str, Enum):
    """
    Types of model components that can be extracted during splitting.
    """
    EMBEDDING = "embedding"
    LAYER = "layer"
    NORM = "norm"
    LM_HEAD = "lm_head"
    OTHER = "other"


class DeviceType(str, Enum):
    """
    Inference device for a split model.

    IPU is the AMD iGPU target. AUTO picks NPU, then iGPU, then GPU, then CPU.
    """
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    IPU = "ipu"
    AUTO = "auto"


class ExecutionPolicy(str, Enum):
    """
    How a split model may be executed.

    AUTO behaves as one monolithic graph. A sequential NPU layer split stays
    rejected. Phase split and MoE routing are the NPU exceptions.
    """
    AUTO = "auto"
    MONOLITHIC_ONLY = "monolithic_only"
    SPLIT_CPU_MEMORY_OPTIMIZED = "split_cpu_memory_optimized"
    SPLIT_GPU_PERSISTENT = "split_gpu_persistent"
    HYBRID_NPU_IGPU_PHASE_SPLIT = "hybrid_npu_igpu_phase_split"
    HYBRID_NPU_MOE_ROUTING = "hybrid_npu_moe_routing"


# Ordered ONNX Runtime fallback for each device. AUTO is resolved later.
# NPU prefers VitisAI, then RyzenAI. IPU is the AMD iGPU order.
PROVIDER_FALLBACK = {
    DeviceType.CPU: ["CPUExecutionProvider"],
    DeviceType.GPU: [
        "ROCmExecutionProvider",
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ],
    DeviceType.NPU: [
        "VitisAIExecutionProvider",
        "RyzenAIExecutionProvider",
        "CPUExecutionProvider",
    ],
    DeviceType.IPU: [
        "DmlExecutionProvider",
        "ROCmExecutionProvider",
        "MIGraphXExecutionProvider",
        "CPUExecutionProvider",
    ],
    DeviceType.AUTO: None,
}


def _coerce_device_type(device: Any) -> DeviceType:
    """Accept a DeviceType or its value, such as npu or ipu."""
    if isinstance(device, DeviceType):
        return device
    text = str(device).split(".")[-1].strip().lower()
    try:
        return DeviceType(text)
    except ValueError as exc:
        known = ", ".join(item.value for item in DeviceType)
        raise ValueError(
            f"Unknown target_device {device!r}. Expected one of: {known}"
        ) from exc


def _coerce_execution_policy(policy: Any) -> ExecutionPolicy:
    """Accept an ExecutionPolicy or its value."""
    if isinstance(policy, ExecutionPolicy):
        return policy
    text = str(policy).split(".")[-1].strip().lower()
    try:
        return ExecutionPolicy(text)
    except ValueError as exc:
        known = ", ".join(item.value for item in ExecutionPolicy)
        raise ValueError(
            f"Unknown execution_policy {policy!r}. Expected one of: {known}"
        ) from exc


def _auto_provider_fallback() -> List[str]:
    """NPU, then AMD iGPU, then discrete GPU, with CPU last."""
    selected: List[str] = []
    for kind in (DeviceType.NPU, DeviceType.IPU, DeviceType.GPU):
        for name in PROVIDER_FALLBACK[kind]:
            if name == "CPUExecutionProvider" or name in selected:
                continue
            selected.append(name)
    selected.append("CPUExecutionProvider")
    return selected


@dataclass
class SplitConfig:
    """
    Configuration for model splitting operations.
    
    This is a unified configuration that works for both ONNX and SafeTensors
    model splitting. Some fields may only be relevant for specific formats.
    
    Attributes:
        model: Path to the model file/directory or HuggingFace model ID.
        output: Base directory for split output files.
        split_only: If True, skip inference validation after splitting.
        
        # Memory efficiency options
        dtype: Target tensor dtype (AUTO preserves original).
        memory_mode: Loading strategy (STREAMING for low-memory environments).
        device: Torch or runtime device string, such as cpu or cuda:0.
        providers: ONNX Runtime provider fallback, first choice first.
        target_device: CPU, GPU, NPU, IPU, or AUTO.
        execution_policy: Monolithic, split, or hybrid NPU placement.
        low_cpu_mem_usage: Use HuggingFace's low-memory loading.
        
        # HuggingFace integration
        trust_remote_code: Allow custom code from model repos.
        use_accelerate: Enable accelerate library for device_map.
        
        # Inference testing
        context: Maximum context length for tokenization.
        batch_size: Batch size for inference testing.
        prompt: Test prompt for inference validation.
        max_new_tokens: Maximum new tokens to generate.
        verbose: Enable verbose output during splitting.
    """
    model: str
    output: str
    split_only: bool = False
    
    # Memory efficiency options
    dtype: TensorDtype = TensorDtype.AUTO
    memory_mode: MemoryMode = MemoryMode.STREAMING  # Default to memory-efficient
    device: str = "cpu"
    providers: List[str] = field(default_factory=list)
    target_device: DeviceType = DeviceType.AUTO
    execution_policy: ExecutionPolicy = ExecutionPolicy.AUTO
    low_cpu_mem_usage: bool = True
    
    # HuggingFace integration
    trust_remote_code: bool = True
    use_accelerate: bool = False
    
    # Inference testing
    context: int = 1024
    batch_size: int = 1
    prompt: str = "Test prompt."
    max_new_tokens: int = 50
    verbose: bool = False

    def __post_init__(self) -> None:
        self.target_device = _coerce_device_type(self.target_device)
        self.execution_policy = _coerce_execution_policy(self.execution_policy)
        if self.providers is None:
            self.providers = []
        if isinstance(self.providers, (str, bytes)) or not isinstance(self.providers, list):
            raise ValueError(
                f"providers must be a list of provider names, got {self.providers!r}"
            )
        cleaned: List[str] = []
        for item in self.providers:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"provider name must be a non-empty string, got {item!r}"
                )
            cleaned.append(item.strip())
        self.providers = cleaned

    def resolved_providers(self) -> List[str]:
        """
        Provider fallback order.

        An explicit ``providers`` list is kept. An empty list uses the order
        for ``target_device``. AUTO tries the NPU, then the AMD iGPU, then
        a discrete GPU, and CPU last.
        """
        if self.providers:
            return list(self.providers)
        mapped = PROVIDER_FALLBACK[self.target_device]
        if mapped is None:
            return _auto_provider_fallback()
        return list(mapped)


from .architecture import ArchitectureType  # Re-export for convenience

@dataclass
class SplitResult:
    """
    Result of a model splitting operation.
    
    This is a unified result type that works for both ONNX and SafeTensors
    splitting operations.
    
    Attributes:
        output_dir: Directory where split files were saved.
        num_layers: Number of transformer layers in the model.
        component_files: Mapping of component names to file paths.
        model_info: Additional model metadata and information.
        format: The format that was split (safetensors or onnx).
    """
    output_dir: str
    num_layers: int
    component_files: Dict[str, str]
    model_info: Dict[str, Any]
    format: ModelFormat = ModelFormat.SAFETENSORS


@dataclass
class ComponentInfo:
    """
    Information about a semantic model component.
    
    This is a base class that can be extended by format-specific implementations.
    Used to track metadata about extracted model components during splitting.
    
    Attributes:
        component_type: Type of component (embedding, layer, norm, lm_head, other).
        layer_id: For layer components, the layer index (0-indexed).
        file_path: Path to the saved component file.
        num_tensors: Number of tensors/weights in this component.
        size_bytes: Approximate size in bytes.
    """
    component_type: ComponentType
    layer_id: Optional[int] = None
    file_path: Optional[str] = None
    num_tensors: int = 0
    size_bytes: int = 0
    
    @property
    def name(self) -> str:
        """Generate a canonical name for this component."""
        if self.component_type == ComponentType.LAYER:
            return f"layer_{self.layer_id}"
        return self.component_type.value


@dataclass
class MergeResult:
    """
    Result of a model merging operation.
    
    Attributes:
        success: Whether the merge completed successfully.
        output_path: Path to the merged model file (if applicable).
        num_components: Number of components that were merged.
        total_tensors: Total number of tensors in the merged model.
        error: Error message if merge failed.
    """
    success: bool
    output_path: Optional[str] = None
    num_components: int = 0
    total_tensors: int = 0
    error: Optional[str] = None
