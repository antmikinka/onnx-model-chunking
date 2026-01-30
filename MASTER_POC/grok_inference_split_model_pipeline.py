#!/usr/bin/env python3
"""
grok_inference_split_model_pipeline.py - Sequential Layer-by-Layer Inference

This script implements a memory-efficient inference pipeline that loads only
ONE ONNX model component (e.g., a single transformer layer) at a time.
After running the forward pass for that component, it is immediately unloaded.

Workflow:
1. Tokenize input
2. Run embeddings.onnx -> Unload
3. For each layer in order:
    Load layer_X.onnx -> Run -> Unload
4. Run norm.onnx -> Unload
5. Run lm_head.onnx -> Unload
6. Select next token

This approach allows running large models on machines with very limited RAM/VRAM.
"""

import os
import json
import argparse
import gc
import time
from typing import Dict, List, Any, Optional
import numpy as np

import onnxruntime as ort
from transformers import AutoTokenizer


class SequentialModelPipeline:
    """
    Inference pipeline that loads and unloads ONNX components sequentially.
    """
    
    def __init__(self, model_dir: str, tokenizer_path: Optional[str] = None):
        """
        Initialize the pipeline.
        
        Args:
            model_dir: Directory containing the split ONNX components and model_info.json
            tokenizer_path: Optional path to the tokenizer (defaults to the original model directory)
        """
        self.model_dir = model_dir
        self.info_path = os.path.join(model_dir, "model_info.json")
        
        if not os.path.exists(self.info_path):
            raise FileNotFoundError(f"model_info.json not found in {model_dir}")
            
        with open(self.info_path, 'r') as f:
            self.model_info = json.load(f)
            
        # Resolve tokenizer path
        if tokenizer_path:
            t_path = tokenizer_path
        else:
            # Get original model path and extract directory (tokenizer is usually in same dir)
            orig_model = self.model_info.get("original_model", "")
            if orig_model.endswith(".onnx"):
                t_path = os.path.dirname(orig_model)
            else:
                t_path = orig_model
                
        print(f"Loading tokenizer from: {t_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(t_path, trust_remote_code=True)
        
        # State management
        self.tensor_registry: Dict[str, np.ndarray] = {}
        self.num_layers = self.model_info["num_layers"]
        
        # Metadata defaults
        self.num_kv_heads = 8
        self.head_dim = 128
        
        # Try to load genai_config for precise metadata
        genai_config_path = os.path.join(t_path, "genai_config.json")
        if os.path.exists(genai_config_path):
            try:
                with open(genai_config_path, 'r') as f:
                    config = json.load(f)
                    decoder = config.get("model", {}).get("decoder", {})
                    self.num_kv_heads = decoder.get("num_key_value_heads", self.num_kv_heads)
                    self.head_dim = decoder.get("head_size", self.head_dim)
                    print(f"  [Config] Detected {self.num_kv_heads} KV heads, {self.head_dim} head dim from genai_config.json")
            except Exception as e:
                print(f"  [Warning] Could not parse genai_config.json: {e}")
        
    def _run_component(self, component_name: str) -> None:
        """
        Load, run, and unload a specific ONNX component.
        
        Args:
            component_name: Key in model_info['components']
        """
        comp_info = self.model_info["components"].get(component_name)
        if not comp_info:
            print(f"  Warning: Component {component_name} not found in model_info, skipping.")
            return
            
        filename = comp_info["file"]
        filepath = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"  Warning: Component file {filename} not found, skipping.")
            return

        # Load session first to get expected input types
        print(f"  [Pipeline] Running {component_name} ({filename})...")
        start_time = time.time()
        
        # Create session options to handle dynamic shapes
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess_options.add_session_config_entry("session.disable_node_shape_check", "1")
        
        session = ort.InferenceSession(filepath, sess_options=sess_options, providers=['VitisAIExecutionProvider'])
        
        # Build expected input type map
        expected_types = {}
        for inp in session.get_inputs():
            expected_types[inp.name] = inp.type
            
        # Prepare inputs from registry with dtype casting
        input_names = comp_info.get("input_names", [])
        output_names = comp_info.get("output_names", [])
        
        # ONNX type string to numpy dtype
        def onnx_type_to_numpy(onnx_type: str):
            if "float16" in onnx_type or "bfloat16" in onnx_type:
                return np.float16
            elif "float" in onnx_type:
                return np.float32
            elif "int64" in onnx_type:
                return np.int64
            elif "int32" in onnx_type:
                return np.int32
            elif "bool" in onnx_type:
                return np.bool_
            return None  # Keep original type
        
        session_inputs = {}
        for name in input_names:
            if name in self.tensor_registry:
                val = self.tensor_registry[name]
                # Cast to expected type if needed
                if name in expected_types:
                    target_dtype = onnx_type_to_numpy(expected_types[name])
                    if target_dtype and val.dtype != target_dtype:
                        print(f"    [Casting] {name}: {val.dtype} -> {target_dtype}")
                        val = val.astype(target_dtype)
                session_inputs[name] = val
            else:
                # print(f"    [Warning] Missing input: {name}")
                pass
        
        # Only run if we have inputs (some components like 'other' may need all inputs)
        if not session_inputs and input_names:
            print(f"  Warning: No matching inputs for {component_name}, may fail.")
                
        # Run inference
        outputs = session.run(output_names if output_names else None, session_inputs)
        elapsed = time.time() - start_time
        
        # Update registry with outputs
        actual_output_names = output_names if output_names else [o.name for o in session.get_outputs()]
        for name, val in zip(actual_output_names, outputs):
            self.tensor_registry[name] = val
            
        # Unload immediately
        del session
        gc.collect()
        
        print(f"  [Pipeline] {component_name} finished in {elapsed:.3f}s")
        
    def generate(self, prompt: str, max_new_tokens: int = 20) -> str:
        """
        Generate text using the sequential pipeline.
        
        Args:
            prompt: Text prompt
            max_new_tokens: Number of tokens to generate
            
        Returns:
            Generated text
        """
        print(f"\nStarting sequential generation for '{prompt}'")
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="np")
        for k, v in inputs.items():
            self.tensor_registry[k] = v
            
        generated_token_ids = []
        
        # Initialize empty KV cache for all layers
        # Shape: [batch, num_heads, 0, head_dim] - empty sequence length initially
        # Using float16 to match the model's precision
        print(f"Initializing empty KV cache for {self.num_layers} layers...")
        for i in range(self.num_layers):
            # Empty cache with 0 past tokens
            # Shape inferred from model/config
            self.tensor_registry[f"past_key_values.{i}.key"] = np.zeros((1, self.num_kv_heads, 0, self.head_dim), dtype=np.float16)
            self.tensor_registry[f"past_key_values.{i}.value"] = np.zeros((1, self.num_kv_heads, 0, self.head_dim), dtype=np.float16)
        
        # Generation loop
        for token_idx in range(max_new_tokens):
            print(f"\n--- Generating Token {token_idx+1}/{max_new_tokens} ---")
            
            # 0. Initialize position_ids if not present
            seq_len = self.tensor_registry["input_ids"].shape[1]
            if "position_ids" not in self.tensor_registry:
                self.tensor_registry["position_ids"] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
            else:
                # Update for new sequence length
                self.tensor_registry["position_ids"] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
            
            # 0.5 Run preprocessing/other component first (produces constants, position formatting, attention masks)
            if "other" in self.model_info["components"]:
                self._run_component("other")
            
            # 1. Embeddings
            self._run_component("embedding")
            
            # 2. Sequential Layers (with KV cache propagation)
            for i in range(self.num_layers):
                layer_key = f"layer_{i}"
                self._run_component(layer_key)
                
                # Map present.N.* outputs to past_key_values.N.* for next token
                if f"present.{i}.key" in self.tensor_registry:
                    self.tensor_registry[f"past_key_values.{i}.key"] = self.tensor_registry[f"present.{i}.key"]
                if f"present.{i}.value" in self.tensor_registry:
                    self.tensor_registry[f"past_key_values.{i}.value"] = self.tensor_registry[f"present.{i}.value"]
                
            # 3. Final Norm
            if "norm" in self.model_info["components"]:
                self._run_component("norm")
                
            # 4. LM Head
            self._run_component("lm_head")
            
            # 5. Extract Logits and Select Token
            # The lm_head output is typically named 'logits' or matching output_names
            logits_name = self.model_info["components"]["lm_head"]["output_names"][0]
            logits = self.tensor_registry[logits_name]
            
            # Greedy sampling: take last token's logits
            next_token_id = np.argmax(logits[:, -1, :], axis=-1)[0]
            generated_token_ids.append(next_token_id)
            
            # Update input_ids for next iteration (simple concatenation for now)
            # Note: This is an O(N^2) or O(N) approach without KV cache.
            # For a true memory-efficient pipeline, each iteration runs the full stack.
            new_input_ids = np.concatenate([self.tensor_registry["input_ids"], [[next_token_id]]], axis=1)
            self.tensor_registry["input_ids"] = new_input_ids
            
            # Update attention mask if present
            if "attention_mask" in self.tensor_registry:
                new_mask = np.concatenate([self.tensor_registry["attention_mask"], [[1]]], axis=1)
                self.tensor_registry["attention_mask"] = new_mask
                
            print(f"Generated token: {self.tokenizer.decode([next_token_id])}")
            
            # Check for EOS
            if next_token_id == self.tokenizer.eos_token_id:
                break
                
        full_text = self.tokenizer.decode(self.tensor_registry["input_ids"][0])
        return full_text


def parse_args():
    parser = argparse.ArgumentParser(description="Sequential Inference Pipeline for Split ONNX Models")
    parser.add_argument("--model-dir", required=True, help="Directory containing split ONNX components")
    parser.add_argument("--prompt", default="The meaning of life is", help="Prompt for generation")
    parser.add_argument("--tokens", type=int, default=10, help="Max new tokens to generate")
    parser.add_argument("--tokenizer", help="Optional tokenizer path if different from original model")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Sequential ONNX Model Pipeline")
    print("One-component-at-a-time memory efficient inference")
    print("=" * 60)
    
    pipeline = SequentialModelPipeline(args.model_dir, args.tokenizer)
    result_text = pipeline.generate(args.prompt, max_new_tokens=args.tokens)
    
    print("\n" + "=" * 60)
    print("Final Generated Text:")
    print("-" * 60)
    print(result_text)
    print("=" * 60)


if __name__ == "__main__":
    main()
