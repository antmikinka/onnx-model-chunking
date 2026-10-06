#!/usr/bin/env python3
"""
grok_inference_split_model_pipeline.py - Sequential Layer-by-Layer Inference

This script implements a memory-efficient inference pipeline that loads
ONNX model components either:
- SEQUENTIALLY (default): One at a time, unloading after use (RAM-efficient)
- PERSISTENTLY (--persist): All at once, keeping resident (NPU-optimized)

Workflow:
1. Tokenize input
2. PREFILL: Run full prompt through pipeline (embed -> layers -> norm -> lm_head)
3. DECODE: For each new token:
   - Input only the new token (not full sequence)
   - Use KV cache for past context
   - Run through pipeline -> select next token
4. Return generated text

Uses modular utilities from pipeline_utils/:
- dtypes.py: Comprehensive ONNX data type handling
- kv_cache.py: KV cache management for incremental decoding
- config_loader.py: Model configuration loading
- session_manager.py: ONNX session lifecycle and persistence

Version: 2.1.0 - Added session persistence mode for NPU optimization
"""

import os
import json
import argparse
import gc
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np

import onnxruntime as ort
from transformers import AutoTokenizer

# Import modular utilities
from pipeline_utils import (
    KVCacheManager,
    get_numpy_dtype,
    safe_cast,
    load_model_config,
    ModelConfig,
    SessionManager,
    SessionConfig,
    get_available_providers,
    get_provider_for_device,
    PROVIDER_CATEGORIES
)


class SequentialModelPipeline:
    """
    Inference pipeline that loads and unloads ONNX components sequentially.

    Uses proper prefill/decode separation for efficient KV-cached inference.
    Supports both sequential (RAM-efficient) and persistent (NPU-optimized) modes.

    Features:
    - Per-component provider routing (hybrid dispatch)
    - Phase-based provider selection (prefill vs decode)
    - Session persistence for NPU optimization

    Example usage with per-component routing:
    ```python
    pipeline = SequentialModelPipeline(
        model_path="split_output",
        component_providers={
            "embedding": ["DmlExecutionProvider"],      # iGPU
            "layer_*":   ["VitisAIExecutionProvider"],  # NPU
            "norm":      ["CPUExecutionProvider"],      # CPU
            "lm_head":   ["DmlExecutionProvider"],      # iGPU
        }
    )
    ```

    Example usage with phase-based routing:
    ```python
    pipeline = SequentialModelPipeline(
        model_path="split_output",
        phase_providers={
            "prefill": ["VitisAIExecutionProvider"],  # NPU for compute-intensive
            "decode": ["DmlExecutionProvider"],       # iGPU for memory-bandwidth
        }
    )
    ```
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        providers: Optional[List[str]] = None,
        component_providers: Optional[Dict[str, List[str]]] = None,
        phase_providers: Optional[Dict[str, List[str]]] = None,
        verbose: bool = True,
        persist_sessions: bool = False
    ):
        """
        Initialize the pipeline.

        Args:
            model_path: Directory containing split components OR path to monolithic .onnx file
            tokenizer_path: Optional path to the tokenizer
            providers: List of execution providers (default for all components/phases)
            component_providers: Optional dict mapping component patterns to providers.
                Keys can be exact names ("embedding") or patterns ("layer_*").
                Pattern matching: "layer_*" matches "layer_0", "layer_1", etc.
            phase_providers: Optional dict mapping phase names to providers.
                Keys: "prefill" (process full prompt) or "decode" (generate tokens)
            verbose: Enable verbose output
            persist_sessions: If True, keep sessions loaded (implied for Monolith)
        """
        self.model_path = model_path
        self.is_monolith = os.path.isfile(model_path)
        self.model_dir = os.path.dirname(model_path) if self.is_monolith else model_path

        self.providers = providers or ['CPUExecutionProvider']
        self.component_providers = component_providers or {}
        self.phase_providers = phase_providers or {}
        self.verbose = verbose
        self.persist_sessions = persist_sessions

        # Phase tracking for phase-based dispatch
        # "prefill" = processing full prompt, "decode" = generating tokens one at a time
        self.phase: str = "prefill"
        # Load model info
        self.info_path = os.path.join(self.model_dir, "model_info.json")
        if self.is_monolith:
            # Synthetic info for monolith
            self.model_info = {"num_layers": 32} # Default pivot, will refine from config
            if os.path.exists(self.info_path):
                with open(self.info_path, 'r') as f:
                    self.model_info.update(json.load(f))
        else:
            if not os.path.exists(self.info_path):
                raise FileNotFoundError(f"model_info.json not found in {self.model_dir}")
            with open(self.info_path, 'r') as f:
                self.model_info = json.load(f)

        # Resolve tokenizer logic...
        if tokenizer_path:
            t_path = tokenizer_path
        else:
            # Try to find tokenizer in model dir or use original path from info
            t_path = self.model_dir
            if "original_model" in self.model_info:
                orig = self.model_info["original_model"]
                if os.path.exists(orig) or (os.path.isdir(os.path.dirname(orig)) and os.path.exists(os.path.dirname(orig))):
                     t_path = os.path.dirname(orig) if orig.endswith('.onnx') else orig

        self._log(f"Loading tokenizer from: {t_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(t_path, trust_remote_code=True)
        except Exception as e:
            self._log(f"Warning: Could not load tokenizer from {t_path}, trying fallback behavior. Error: {e}")
            # Fallback for testing without tokenizer
            self.tokenizer = None

        self.tensor_registry: Dict[str, np.ndarray] = {}
        self.num_layers = self.model_info.get("num_layers", 32)
        
        # Load model configuration using modular loader
        try:
            self.config = load_model_config(self.model_dir, t_path)
            self._log(f"  [Config] Loaded from {self.config.source_file}")
            quant_note = ""
            method = getattr(self.config, "quantization_method", None)
            if isinstance(method, str) and method:
                quant_note = (
                    f", quant {method}"
                    f" weights {getattr(self.config, 'weight_dtype', None)}"
                    f" activations {getattr(self.config, 'activation_dtype', None)}"
                    f" group {getattr(self.config, 'quantization_group_size', None)}"
                    f" format {getattr(self.config, 'quantization_format', None)}"
                )
            arch_note = ""
            variant = getattr(self.config, "attention_variant", "dense")
            if isinstance(variant, str) and variant and variant != "dense":
                arch_note = f", attention {variant}"
            if getattr(self.config, "is_moe", False) is True:
                arch_note += (
                    f", moe experts {getattr(self.config, 'num_experts', 0)}"
                    f" per_token {getattr(self.config, 'num_experts_per_token', 0)}"
                )
            kv_note = ""
            kv_dtype_name = getattr(self.config, "kv_cache_dtype", None)
            if isinstance(kv_dtype_name, str) and kv_dtype_name:
                kv_note += f", kv {kv_dtype_name}"
            kv_layout = getattr(self.config, "kv_cache_layout", "BNSD")
            if isinstance(kv_layout, str) and kv_layout and kv_layout != "BNSD":
                kv_note += f", kv layout {kv_layout}"
            if getattr(self.config, "use_sliding_window", False) is True:
                kv_note += ", sliding kv"
            self._log(
                f"  [Config] {self.config.num_layers} layers, {self.config.num_kv_heads} KV heads, "
                f"{self.config.head_dim} head dim, max_seq_len {self.config.max_seq_len}"
                f"{quant_note}{arch_note}{kv_note}"
            )
        except FileNotFoundError:
            self._log("  [Warning] No config found, using defaults (may be incorrect)")
            self.config = ModelConfig(
                num_layers=self.num_layers,
                num_kv_heads=8,
                head_dim=128,
                source_file="defaults"
            )
        
        # Initialize KV cache manager. A positive kv_cache_max_len, or else
        # max_seq_len, preallocates the static buffer once. 0 leaves a CPU
        # cache to grow per token. BNSD is the layout this manager allocates.
        raw_limit = getattr(self.config, "max_seq_len", 0)
        try:
            max_seq_len = int(raw_limit)
        except (TypeError, ValueError):
            max_seq_len = 0
        raw_kv_limit = getattr(self.config, "kv_cache_max_len", None)
        if (
            isinstance(raw_kv_limit, int)
            and not isinstance(raw_kv_limit, bool)
            and raw_kv_limit > 0
        ):
            max_seq_len = raw_kv_limit
        kv_kwargs = {}
        if max_seq_len > 0:
            kv_kwargs["max_seq_len"] = max_seq_len
        kv_scheme = getattr(self.config, "kv_cache_dtype", None)
        if isinstance(kv_scheme, str) and kv_scheme in {
            "fp16", "bf16", "int8", "fp8", "int4", "fp4",
        }:
            kv_kwargs["kv_quantization"] = kv_scheme
        page = getattr(self.config, "kv_cache_page_size", None)
        if isinstance(page, int) and not isinstance(page, bool) and page > 0:
            kv_kwargs["block_size"] = page
        if getattr(self.config, "use_sliding_window", False) is True:
            window = None
            extras = getattr(self.config, "architecture_extras", None)
            if isinstance(extras, dict):
                raw_window = extras.get("sliding_window")
                if (
                    isinstance(raw_window, int)
                    and not isinstance(raw_window, bool)
                    and raw_window > 0
                ):
                    window = raw_window
            if window is None and max_seq_len > 0:
                window = max_seq_len
            if isinstance(window, int) and window > 0:
                kv_kwargs["policy"] = "sliding_window"
                kv_kwargs["window_size"] = window
        self.kv_cache = KVCacheManager(
            num_layers=self.config.num_layers,
            num_kv_heads=self.config.num_kv_heads,
            head_dim=self.config.head_dim,
            dtype=self.config.dtype,
            batch_size=1,
            **kv_kwargs,
        )
        
        # Session management
        self.session_manager: Optional[SessionManager] = None
        
        if persist_sessions:
            self._init_persistent_sessions()
    
    def _init_persistent_sessions(self) -> None:
        """Initialize and load all sessions for persistent mode (NPU-optimized)."""
        self._log("\n[NPU Mode] Initializing persistent sessions...")
        
        session_config = SessionConfig(
            providers=self.providers,
            optimization_level="DISABLE",  # For dynamic shapes
            disable_shape_check=True
        )
        
        # In monolith mode, model_dir is just the dir, but we need to register the monolith session
        if self.is_monolith:
             # Manually register the monolith session in the manager
             # This requires SessionManager to handle explicit file paths different from its model_dir assumption
             # For now, we'll let SessionManager handle it via a new mechanism or just rely on load_component_session
             pass 
        
        self.session_manager = SessionManager(
            model_dir=self.model_dir,
            config=session_config,
            model_info=self.model_info
        )
        
        # Load all sessions at startup
        self.session_manager.load_all()
        
        mem = self.session_manager.get_memory_usage()
        self._log(f"[NPU Mode] All sessions loaded: {mem['total_mb']:.1f} MB in memory")
        self._log(f"[NPU Mode] Sessions will remain resident for optimal NPU performance\n")
        
    def _log(self, msg: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(msg)

    def _get_component_providers(self, component_name: str) -> List[str]:
        """
        Get execution providers for a specific component.

        Uses pattern matching to find the best provider configuration:
        1. Exact match: "embedding" matches "embedding"
        2. Wildcard match: "layer_*" matches "layer_0", "layer_1", etc.
        3. Fallback: Returns self.providers if no pattern matches

        Args:
            component_name: Name of the component (e.g., "embedding", "layer_0")

        Returns:
            List of provider names for the component

        Example:
            >>> pipeline.component_providers = {
            ...     "embedding": ["DmlExecutionProvider"],
            ...     "layer_*": ["VitisAIExecutionProvider"]
            ... }
            >>> pipeline._get_component_providers("embedding")
            ["DmlExecutionProvider"]
            >>> pipeline._get_component_providers("layer_5")
            ["VitisAIExecutionProvider"]
            >>> pipeline._get_component_providers("norm")
            ["CPUExecutionProvider"]  # fallback
        """
        # First try exact match
        if component_name in self.component_providers:
            return self.component_providers[component_name]

        # Then try wildcard patterns
        for pattern, providers in self.component_providers.items():
            if pattern.endswith('*'):
                prefix = pattern[:-1]
                if component_name.startswith(prefix):
                    return providers

        # A phase list covers components that are not pinned to one device.
        if self.phase_providers:
            return self.phase_providers.get(self.phase, self.providers)

        # Fallback to default providers
        return self.providers

    def _get_current_providers(self) -> List[str]:
        """
        Get execution providers for the current phase.

        Phase-based dispatch allows different providers for prefill vs decode:
        - "prefill": Compute-intensive, benefits from NPU
        - "decode": Memory-bandwidth-intensive, may benefit from iGPU

        Returns:
            List of provider names for current phase

        Example:
            >>> pipeline.phase_providers = {
            ...     "prefill": ["VitisAIExecutionProvider"],
            ...     "decode": ["DmlExecutionProvider"]
            ... }
            >>> pipeline.phase = "prefill"
            >>> pipeline._get_current_providers()
            ["VitisAIExecutionProvider"]
        """
        return self.phase_providers.get(self.phase, self.providers)

    def _set_phase(self, phase: str) -> None:
        """
        Set the current inference phase.

        Args:
            phase: Either "prefill" (processing full prompt) or "decode" (generating tokens)

        Raises:
            ValueError: If phase is not "prefill" or "decode"
        """
        if phase not in ("prefill", "decode"):
            raise ValueError(f"Invalid phase: {phase}. Must be 'prefill' or 'decode'")
        self.phase = phase
        self._log(f"  [Phase] Switched to '{phase}' mode")
        
    def load_component_session(self, component_name: str) -> Optional[ort.InferenceSession]:
        """
        Load the ONNX session for a specific component.

        In persistent mode, returns pre-loaded session.
        In sequential mode, loads from disk with component-specific providers.

        Returns None if component/file missing.

        Args:
            component_name: Name of the component to load (e.g., "embedding", "layer_0")

        Returns:
            Loaded InferenceSession or None if failed
        """
        # Persistent mode: use SessionManager
        if self.session_manager is not None:
            session = self.session_manager.get_session(component_name)
            if session:
                self._log(f"  [Pipeline] Using cached {component_name}")
            return session

        # Sequential mode: load from disk with component-specific providers
        if self.is_monolith and component_name == "monolith":
            filepath = self.model_path
            filename = os.path.basename(filepath)
        else:
            comp_info = self.model_info["components"].get(component_name)
            if not comp_info:
                self._log(f"  Warning: Component {component_name} not found in model_info, skipping.")
                return None
            filename = comp_info["file"]
            filepath = os.path.join(self.model_dir, filename)

        if not os.path.exists(filepath):
            self._log(f"  Warning: Component file {filename} not found, skipping.")
            return None

        self._log(f"  [Pipeline] Loading {component_name} ({filename})...")

        # Get component-specific providers (P1.1: Per-component Provider Routing)
        component_prov = self._get_component_providers(component_name)
        self._log(f"  [Pipeline] Using providers: {component_prov}")

        # Create session options to handle dynamic shapes
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess_options.add_session_config_entry("session.disable_node_shape_check", "1")

        try:
            session = ort.InferenceSession(
                filepath,
                sess_options=sess_options,
                providers=component_prov  # Use component-specific providers
            )
            return session
        except Exception as e:
            self._log(f"  [Error] Failed to load session {filename} with providers {component_prov}: {e}")
            # Try fallback to default providers
            if component_prov != self.providers:
                self._log(f"  [Fallback] Trying with default providers: {self.providers}")
                try:
                    session = ort.InferenceSession(
                        filepath,
                        sess_options=sess_options,
                        providers=self.providers
                    )
                    return session
                except Exception as e2:
                    self._log(f"  [Error] Fallback also failed: {e2}")
            return None

    def _detect_kv_format(self) -> Tuple[str, str]:
        """
        Dynamically detect the naming format for KV cache outputs from layer 0.
        
        Returns:
            Tuple of (key_format, value_format) strings, e.g. ("present.{layer}.key", "present.{layer}.value")
        """
        if self.is_monolith:
            # Monolithic models usually follow standard naming or can be inferred differently
            # For now, default to present.{layer}.key style or try to sniff session outputs
            # We can try to sniff the session directly if available
            session = self.load_component_session("monolith")
            if session:
                output_names = [o.name for o in session.get_outputs()]
                # Use same heuristic
                key_name = None
                val_name = None
                for name in output_names:
                    if "key" in name.lower() and "past" not in name.lower():
                        key_name = name
                    if "value" in name.lower() and "past" not in name.lower():
                        val_name = name
                
                if key_name and val_name:
                    k_fmt = key_name.replace("0", "{layer}")
                    v_fmt = val_name.replace("0", "{layer}")
                    
                    # If replacement didn't change anything (no '0' in name), we might need a different strategy
                    # But often monoliths use present_i_key style
                    # Heuristic: try to find the number in the string
                    import re
                    if k_fmt == key_name:
                        k_fmt = re.sub(r'\d+', '{layer}', key_name, count=1)
                    if v_fmt == val_name:
                        v_fmt = re.sub(r'\d+', '{layer}', val_name, count=1)
                        
                    self._log(f"  [Pipeline] Detected KV format (Monolith): {k_fmt}, {v_fmt}")
                    return k_fmt, v_fmt

            return "present.{layer}.key", "present.{layer}.value"
            
        if "layer_0" not in self.model_info["components"]:
            return "present.{layer}.key", "present.{layer}.value"
            
        outputs = self.model_info["components"]["layer_0"]["output_names"]
        
        # Heuristic: look for outputs containing "present" or "key"/"value"
        # Common patterns:
        # - present.0.key, present.0.value
        # - present_key_values_0 (packed) - NOT SUPPORTED yet by this logic
        # - key_states, value_states
        
        key_name = None
        val_name = None
        
        for name in outputs:
            if "key" in name.lower() and "past" not in name.lower():
                key_name = name
            if "value" in name.lower() and "past" not in name.lower():
                val_name = name
                
        if key_name and val_name:
            # Try to infer pattern by replacing '0' with '{layer}'
            # This is simple string replacement, robust enough for standard ONNX exports
            k_fmt = key_name.replace("0", "{layer}")
            v_fmt = val_name.replace("0", "{layer}")
            self._log(f"  [Pipeline] Detected KV format: {k_fmt}, {v_fmt}")
            return k_fmt, v_fmt
            
        self._log("  [Warning] Could not auto-detect KV format, using defaults")
        return "present.{layer}.key", "present.{layer}.value"

    def run_component_session(self, component_name: str, session: ort.InferenceSession) -> None:
        """
        Run inference on a loaded session using inputs from the registry.
        Updates registry with outputs. Uses safe_cast for type conversion.
        """
        if session is None:
            return

        if self.is_monolith and component_name == "monolith":
             # Synthesize info for monolith
             input_names = [i.name for i in session.get_inputs()]
             output_names = [o.name for o in session.get_outputs()]
             comp_info = {"input_names": input_names, "output_names": output_names}
        else:
            comp_info = self.model_info["components"].get(component_name)
            
        start_time = time.time()
        
        # Build expected input type map
        expected_types = {}
        for inp in session.get_inputs():
            expected_types[inp.name] = inp.type
            
        # Prepare inputs from registry with dtype casting
        input_names = comp_info.get("input_names", [])
        output_names = comp_info.get("output_names", [])
        
        session_inputs = {}
        missing_inputs = []
        
        for name in input_names:
            if name in self.tensor_registry:
                val = self.tensor_registry[name]
                # Cast to expected type using safe_cast
                if name in expected_types:
                    try:
                        target_dtype = get_numpy_dtype(expected_types[name])
                        val = safe_cast(val, target_dtype, warn_on_precision_loss=False)
                    except ValueError:
                        pass  # Keep original type if unknown
                session_inputs[name] = val
            else:
                missing_inputs.append(name)
        
        # Explicit validation: ERROR on missing inputs
        if missing_inputs:
            raise RuntimeError(f"Missing inputs for {component_name}: {missing_inputs}")
        
        if not session_inputs and comp_info.get("input_names"):
             raise RuntimeError(f"No inputs found for {component_name}")
                
        # Run inference
        output_names = comp_info.get("output_names", [o.name for o in session.get_outputs()])
        outputs = session.run(output_names if output_names else None, session_inputs)
        elapsed = time.time() - start_time
        
        # Update registry with outputs
        actual_output_names = output_names if output_names else [o.name for o in session.get_outputs()]
        for name, val in zip(actual_output_names, outputs):
            self.tensor_registry[name] = val
            if self.verbose:
                self._log(f"    Output: {name} (shape: {val.shape if hasattr(val, 'shape') else 'N/A'})")
            
        self._log(f"  [Pipeline] {component_name} ran in {elapsed:.3f}s")

        # Only unload in sequential mode
        if self.session_manager is None:
            del session
            gc.collect()

    def _run_component(self, component_name: str) -> None:
        """
        In persistent mode, session is not unloaded.
        In sequential mode, session is unloaded after use.
        """
        session = self.load_component_session(component_name)
        if session:
            self.run_component_session(component_name, session)
            
            # Only unload in sequential mode
            if self.session_manager is None:
                del session
                gc.collect()
    
    def _run_layers_with_kv_update(self) -> None:
        """
        Run all transformer layers and update KV cache from present outputs.
        """
        # Detect format once (lazy init could be better but this is fine)
        present_key_fmt, present_val_fmt = self._detect_kv_format()
        
        if self.is_monolith:
             # Run single monolith component
             self._run_component("monolith")
             
             # The monolith presumably outputs all present states
             # We just need to make sure the registry has them
             pass
        else:
            for i in range(self.num_layers):
                layer_key = f"layer_{i}"
                self._run_component(layer_key)
                
                # Update KV cache from present outputs using KVCacheManager's helper
            # This is cleaner and less error-prone than manual update
            self.kv_cache.extract_from_registry(
                self.tensor_registry,
                present_key_format=present_key_fmt,
                present_value_format=present_val_fmt
            )
        
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        stream_callback: Optional[Callable[[str, int], None]] = None,
    ) -> str:
        """
        Generate text using the sequential pipeline with proper prefill/decode.
        
        Args:
            prompt: Text prompt
            max_new_tokens: Number of tokens to generate
            stream_callback: Called with each new token's text and its
                0-based index as soon as that token is chosen.
            
        Returns:
            Generated text
        """
        self._log(f"\nStarting sequential generation for '{prompt}'")
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="np")
        prompt_ids = inputs["input_ids"]
        prompt_len = prompt_ids.shape[1]
        
        # Reset KV cache for new generation session
        self.kv_cache.reset()
        
        generated_token_ids = []
        last_token_id = None
        
        self._log(f"Prompt length: {prompt_len} tokens")
        self._log(f"Using KV cache: {self.kv_cache}")
        
        # Generation loop
        for token_idx in range(max_new_tokens):
            self._log(f"\n--- Token {token_idx+1}/{max_new_tokens} ---")

            # ================================================================
            # PREFILL (first iteration) - Process full prompt
            # ================================================================
            if token_idx == 0:
                # P1.3: Set phase to prefill for compute-intensive processing
                self._set_phase("prefill")

                current_input_ids = prompt_ids
                current_position_ids = np.arange(prompt_len, dtype=np.int64).reshape(1, -1)
                current_attention_mask = np.ones((1, prompt_len), dtype=np.int64)
                self._log(f"  [Prefill] Processing full prompt ({prompt_len} tokens)")

            # ================================================================
            # DECODE (subsequent iterations) - Process only new token
            # ================================================================
            else:
                # P1.3: Set phase to decode for memory-bandwidth-intensive processing
                self._set_phase("decode")

                # Only pass the NEW token, not the full sequence
                current_input_ids = np.array([[last_token_id]], dtype=np.int64)

                # Position is the current sequence position (past tokens + 1)
                past_len = self.kv_cache.get_sequence_length()
                current_position_ids = np.array([[past_len]], dtype=np.int64)

                # Attention mask covers all tokens (past + current)
                current_attention_mask = np.ones((1, past_len + 1), dtype=np.int64)
                self._log(f"  [Decode] New token, position {past_len}, total seq {past_len + 1}")
            
            # Set tensors in registry
            self.tensor_registry["input_ids"] = current_input_ids
            self.tensor_registry["position_ids"] = current_position_ids
            self.tensor_registry["attention_mask"] = current_attention_mask
            
            # Inject KV cache into registry (past_key_values.*.key/value)
            self.kv_cache.inject_into_registry(self.tensor_registry)
            
            # Run preprocessing/other component if exists
            if not self.is_monolith and "other" in self.model_info["components"]:
                self._run_component("other")
            
            # Run embeddings
            if not self.is_monolith:
                self._run_component("embedding")
            
            # Run all transformer layers with KV cache update
            self._run_layers_with_kv_update()
            
            # Run final norm if exists
            if not self.is_monolith and "norm" in self.model_info["components"]:
                self._run_component("norm")
                
            # Run LM head
            if not self.is_monolith:
                self._run_component("lm_head")
            
            # Extract logits and select next token
            if self.is_monolith:
                 # In monolith, logits is usually the first output or named 'logits'
                 if "logits" in self.tensor_registry:
                     logits_name = "logits"
                 else:
                     # Fallback: find any output with 'logits' in name or take the first non-KV output
                     logits_name = "logits" # Default assumption
                     for k in self.tensor_registry:
                         if "logits" in k:
                             logits_name = k
                             break
            else:
                logits_name = self.model_info["components"]["lm_head"]["output_names"][0]
            
            # Final safety check
            if logits_name not in self.tensor_registry:
                # If we still can't find it, just take the last registered tensor that looks like logits
                # This is a desperate fallback for robustness
                for k, v in self.tensor_registry.items():
                    if len(v.shape) == 3: # (batch, seq, vocab)
                        logits_name = k
                        break
            
            logits = self.tensor_registry[logits_name]
            last_token_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])
            generated_token_ids.append(last_token_id)
            
            token_text = self.tokenizer.decode([last_token_id])
            if stream_callback is not None:
                stream_callback(token_text, token_idx)
            self._log(f"  Generated: '{token_text}' (id={last_token_id})")
            
            # Check for EOS
            if last_token_id == self.tokenizer.eos_token_id:
                self._log("  [EOS] End of sequence token reached")
                break
        
        # Decode full output: original prompt + generated tokens
        full_ids = np.concatenate([prompt_ids, np.array([generated_token_ids], dtype=np.int64)], axis=1)
        full_text = self.tokenizer.decode(full_ids[0])
        
        return full_text
    
    def cleanup(self) -> None:
        """Cleanup resources, unload sessions."""
        if self.session_manager is not None:
            count = self.session_manager.unload_all()
            self._log(f"[Cleanup] Unloaded {count} sessions")


def parse_args():
    parser = argparse.ArgumentParser(description="Sequential Inference Pipeline for Split ONNX Models")
    parser.add_argument("--model", dest="model_path", required=True, help="Path to split directory OR monolithic .onnx file")
    # Backwards compatibility
    parser.add_argument("--model-dir", dest="model_dir_compat", help="Deprecated: use --model")
    parser.add_argument("--prompt", default="The meaning of life is", help="Prompt for generation")
    parser.add_argument("--tokens", type=int, default=10, help="Number of tokens to generate")
    parser.add_argument("--tokenizer", type=str, help="Path to tokenizer directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--quiet", action="store_true", help="Only print generated text")
    
    # NPU optimization flags
    parser.add_argument("--persist", action="store_true", 
                        help="Enable session persistence (NPU-optimized: load all sessions once)")
    parser.add_argument("--providers", help="Comma-separated list of execution providers (overrides --device)")
    parser.add_argument("--device", choices=["auto", "npu", "gpu", "cpu"], default="auto", 
                       help="Target device category (auto-selects best provider)")
    parser.add_argument("--list-providers", action="store_true",
                        help="List available execution providers and exit")
    
    args = parser.parse_args()
    if args.model_dir_compat and not args.model_path:
        args.model_path = args.model_dir_compat
    return args


def main():
    args = parse_args()
    
    # Handle --list-providers
    if args.list_providers:
        print("Available ONNX Runtime Execution Providers:")
        for provider in get_available_providers():
            print(f"  - {provider}")
        return
    
    # Device selection logic
    if args.device:
        providers = get_provider_for_device(args.device)
    elif args.providers:
        providers = [p.strip() for p in args.providers.split(",")]
    else:
        # Default to auto detection (prioritizes NPU -> GPU -> CPU)
        providers = get_provider_for_device("auto")
        
    print("=" * 70)
    print("  SEQUENTIAL ONNX MODEL PIPELINE v3.0 - UNIVERSAL")
    print("=" * 70)
    
    print(f"  Target Device: {args.device.upper() if args.device else 'AUTO'}")
    print(f"  Active Providers: {providers}")
    
    # Hardware status reporting
    available_providers = get_available_providers()
    
    npu_active = any(p in available_providers and p in PROVIDER_CATEGORIES['npu'] for p in providers)
    gpu_active = any(p in available_providers and p in PROVIDER_CATEGORIES['gpu'] for p in providers)
    
    status_msg = "HARDWARE STATUS: "
    if npu_active:
        status_msg += "NPU ACCELERATION ACTIVE 🚀"
    elif gpu_active:
        status_msg += "GPU ACCELERATION ACTIVE ⚡"
    else:
        status_msg += "CPU ONLY (SAFE MODE) 🐌"
    
    print(f"  {status_msg}")
    print(f"  Mode: {'PERSISTENT (High Performance)' if args.persist else 'SEQUENTIAL (Memory Efficient)'}")
    print("-" * 70)
    
    pipeline = SequentialModelPipeline(
        args.model_path, 
        args.tokenizer,
        providers=providers,
        verbose=not args.quiet,
        persist_sessions=args.persist
    )
    
    result_text = pipeline.generate(args.prompt, max_new_tokens=args.tokens)
    
    print("\n" + "=" * 60)
    print("Final Generated Text:")
    print("-" * 60)
    print(result_text)
    print("=" * 60)
    
    # Cleanup
    pipeline.cleanup()


if __name__ == "__main__":
    main()
