"""
ONNX model chunking utilities.

This module provides utilities for splitting and merging ONNX models
into per-component files (embeddings, layers, norm, lm_head).

Usage:
    from chunking_utils.onnx import split_onnx, merge_onnx
    
    # Split an ONNX model
    result = split_onnx(model="./model.onnx", output="./output")
    
    # Merge back
    merged_model = merge_onnx("./output/model")
"""

from .splitter import (
    split_onnx,
    extract_components,
    create_component_onnx,
    collect_boundary_tensors,
    build_boundary_contract,
    save_components,
    write_execution_contract,
    resolve_tensor_type,
    extract_quantization_info,
    extract_kv_cache_info,
    extract_execution_policy,
    detect_architecture_from_onnx,
)
from .validator import (
    compute_psnr,
    verify_split_correctness,
    validate_boundary_contracts,
    validate_component_io,
)
from .weight_store import (
    WeightStrategy,
    coerce_weight_strategy,
    plan_weight_storage,
    store_component_weights,
)
from .merger import merge_onnx
from .loader import load_onnx_model, resolve_model_path
from .types import ComponentInfo, OnnxSplitResult
from .inference import run_inference
from .golden_attention_subgraph import (
    build_golden_attention_model,
    build_projected_attention_model,
    golden_attention_facts,
)
from .quark_optimize import (
    QuarkOptimizeConfig,
    QuarkOptimizeResult,
    QuarkUnavailable,
    optimize_for_npu,
    pipeline_order,
    prepare_model_for_split,
)
from .static_shapes import (
    apply_static_contract,
    decompose_gqa_to_npu_primitives,
    right_aligned_attention_bias,
    static_attention_bias,
)

__all__ = [
    # Main operations
    "split_onnx",
    "merge_onnx",
    # Loader
    "load_onnx_model",
    "resolve_model_path",
    # Component extraction
    "extract_components",
    "create_component_onnx",
    "collect_boundary_tensors",
    "build_boundary_contract",
    "save_components",
    "write_execution_contract",
    "resolve_tensor_type",
    "extract_quantization_info",
    "extract_kv_cache_info",
    "extract_execution_policy",
    "compute_psnr",
    "verify_split_correctness",
    "validate_boundary_contracts",
    "validate_component_io",
    "WeightStrategy",
    "coerce_weight_strategy",
    "plan_weight_storage",
    "store_component_weights",
    # Architecture detection
    "detect_architecture_from_onnx",
    # Types
    "ComponentInfo",
    "OnnxSplitResult",
    # Inference
    "run_inference",
    "build_golden_attention_model",
    "build_projected_attention_model",
    "golden_attention_facts",
    "QuarkOptimizeConfig",
    "QuarkOptimizeResult",
    "QuarkUnavailable",
    "optimize_for_npu",
    "pipeline_order",
    "prepare_model_for_split",
    "apply_static_contract",
    "decompose_gqa_to_npu_primitives",
    "right_aligned_attention_bias",
    "static_attention_bias",
]
