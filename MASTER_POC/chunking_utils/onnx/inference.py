"""
ONNX model inference utilities.

This module provides utilities for running inference on split/merged ONNX models
using ONNX Runtime.
"""

import os
from typing import List, Optional

import onnx

from ..common.types import SplitConfig
from .merger import merge_onnx

# Optional ONNX Runtime import
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


def run_inference(
    output_dir: str,
    prompt: str = "Test prompt.",
    config: Optional[SplitConfig] = None,
) -> List[str]:
    """
    Run inference on a split/merged ONNX model.
    
    Note: Full text generation requires tokenizer integration.
    This function validates that the merged model can be loaded
    and creates an inference session.
    
    Args:
        output_dir: Directory containing split model.
        prompt: Test prompt (for future tokenizer integration).
        config: Optional SplitConfig with inference parameters.
    
    Returns:
        List of validation messages.
    
    Example:
        >>> results = run_inference("./output/model")
        >>> print(results[0])
    """
    if not HAS_ORT:
        print("Warning: onnxruntime not available, skipping inference")
        return ["onnxruntime not installed"]
    
    print("\nRunning inference...")
    
    # Merge components
    merged_model = merge_onnx(output_dir)
    
    # Save merged model temporarily
    merged_path = os.path.join(output_dir, "merged_model.onnx")
    onnx.save(merged_model, merged_path)
    
    # Create inference session
    session = ort.InferenceSession(merged_path)
    
    # Get input info
    input_info = session.get_inputs()
    print(f"  Model inputs: {[i.name for i in input_info]}")
    
    # Get output info
    output_info = session.get_outputs()
    print(f"  Model outputs: {[o.name for o in output_info]}")
    
    print("  Model loaded successfully for inference")
    print("  (Full inference requires tokenizer integration)")
    
    return ["Model validated for inference"]


def create_inference_session(
    output_dir: str,
    merge_if_needed: bool = True,
) -> "ort.InferenceSession":
    """
    Create an ONNX Runtime inference session from split components.
    
    Args:
        output_dir: Directory containing split model or merged model.
        merge_if_needed: If True, merge components if merged_model.onnx doesn't exist.
    
    Returns:
        ONNX Runtime InferenceSession.
    
    Raises:
        ImportError: If onnxruntime is not installed.
        FileNotFoundError: If model files not found.
    """
    if not HAS_ORT:
        raise ImportError("onnxruntime is required for inference. Install with: pip install onnxruntime")
    
    merged_path = os.path.join(output_dir, "merged_model.onnx")
    
    if not os.path.exists(merged_path):
        if merge_if_needed:
            print("Merged model not found, merging components...")
            merged_model = merge_onnx(output_dir)
            onnx.save(merged_model, merged_path)
        else:
            raise FileNotFoundError(f"merged_model.onnx not found in {output_dir}")
    
    return ort.InferenceSession(merged_path)


def get_session_info(session: "ort.InferenceSession") -> dict:
    """
    Get information about an ONNX Runtime inference session.
    
    Args:
        session: ONNX Runtime InferenceSession.
    
    Returns:
        Dictionary with session information.
    """
    inputs = []
    for inp in session.get_inputs():
        inputs.append({
            "name": inp.name,
            "shape": inp.shape,
            "type": inp.type,
        })
    
    outputs = []
    for out in session.get_outputs():
        outputs.append({
            "name": out.name,
            "shape": out.shape,
            "type": out.type,
        })
    
    return {
        "inputs": inputs,
        "outputs": outputs,
        "providers": session.get_providers(),
    }


def validate_split_model(output_dir: str) -> bool:
    """
    Validate that a split ONNX model can be merged and loaded.
    
    Args:
        output_dir: Directory containing the split model files.
    
    Returns:
        True if model validates successfully, False otherwise.
    """
    if not HAS_ORT:
        print("Warning: onnxruntime not available, performing basic validation only")
        try:
            merged_model = merge_onnx(output_dir)
            print(f"Model merge successful: {len(merged_model.graph.node)} nodes")
            return True
        except Exception as e:
            print(f"Model validation failed: {e}")
            return False
    
    try:
        # Try to create inference session
        session = create_inference_session(output_dir)
        info = get_session_info(session)
        print(f"Model validation successful")
        print(f"  Inputs: {len(info['inputs'])}")
        print(f"  Outputs: {len(info['outputs'])}")
        print(f"  Providers: {info['providers']}")
        return True
    except Exception as e:
        print(f"Model validation failed: {e}")
        return False
