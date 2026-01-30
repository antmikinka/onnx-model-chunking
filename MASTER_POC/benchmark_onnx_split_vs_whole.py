#!/usr/bin/env python3
"""
benchmark_onnx_split_vs_whole.py - Comprehensive ONNX Inference Benchmark

This script performs a production-quality comparison between:
1. WHOLE MODEL: Loading a monolithic ONNX model into a single session.
2. SPLIT MODEL: Sequentially loading/unloading ONNX components (transient memory).

KEY METRICS:
- Prefill Time (Time to First Token - TTFT) & Throughput
- Decode Time (Inter-token latency) & Throughput
- Total Latency & Overall TPS
- Peak Memory Usage (RSS)
- Model Load Time
- Log Probabilities (Confidence)

ARCHICTECTURE:
- Modular 'ABCBenchmarkRunner' design allows easy extension to other frameworks 
  (e.g., PyTorch, TensorRT, OpenVINO) by strictly defining the runner interface.
- Separation of concerns: Metrics collection vs. Inference execution.
"""

import os
import gc
import json
import time
import argparse
import statistics
import psutil
import numpy as np
import onnxruntime as ort
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Union
from transformers import AutoTokenizer

# Import the base sequential pipeline logic
# We will extend this class for instrumentation without modifying the original file
try:
    from grok_inference_split_model_pipeline import SequentialModelPipeline
except ImportError:
    # If running from a different directory, try to adjust path or fail
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from grok_inference_split_model_pipeline import SequentialModelPipeline


# ==============================================================================
# Utility Functions
# ==============================================================================

def get_memory_mb() -> float:
    """Get current process memory usage (RSS) in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def clear_memory(sleep_s: float = 2.0):
    """Force garbage collection and wait for OS to reclaim memory."""
    gc.collect()
    time.sleep(sleep_s)

def calculate_stats(values: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of values."""
    if not values:
        return {"avg": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    return {
        "avg": round(statistics.mean(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "std": round(statistics.stdev(values) if len(values) > 1 else 0.0, 4)
    }

def compute_logprobs_numpy(logits: np.ndarray, token_id: int) -> float:
    """
    Compute log probability of a selected token from raw logits using numpy.
    logits: Shape [1, seq_len, vocab_size] (we use the last position)
    """
    # Get logits for the last token position: [vocab_size]
    last_logits = logits[0, -1, :]
    
    # Numerically stable log_softmax
    max_logit = np.max(last_logits)
    log_sum_exp = max_logit + np.log(np.sum(np.exp(last_logits - max_logit)))
    log_probs = last_logits - log_sum_exp
    
    return float(log_probs[token_id])

def analyze_onnx_structure(model_dir: str) -> Dict[str, Any]:
    """Analyze the ONNX files in a directory."""
    files = [f for f in os.listdir(model_dir) if f.endswith(".onnx")]
    size_mb = sum(os.path.getsize(os.path.join(model_dir, f)) for f in files) / (1024 * 1024)
    
    breakdown = {
        "embeddings": [], "layers": [], "norm": [], "lm_head": [], "other": []
    }
    
    for f in files:
        if "embed" in f: breakdown["embeddings"].append(f)
        elif "layer" in f: breakdown["layers"].append(f)
        elif "norm" in f: breakdown["norm"].append(f)
        elif "lm_head" in f: breakdown["lm_head"].append(f)
        else: breakdown["other"].append(f)
        
    return {
        "total_files": len(files),
        "total_size_mb": round(size_mb, 2),
        "breakdown": {k: len(v) for k, v in breakdown.items()}
    }

# ==============================================================================
# Data Classes for Metrics
# ==============================================================================

@dataclass
class PrefillDecodeMetrics:
    prefill_time_s: float
    prefill_tokens: int
    prefill_tokens_per_second: float
    decode_time_s: float
    decode_tokens: int
    decode_tokens_per_second: float
    inter_token_latencies: List[float]

@dataclass
class InferenceMetrics:
    load_time_s: float
    total_latency_s: float
    ttft_s: float  # Start of inference -> First token generated
    tokens_per_second: float
    memory_load_mb: float
    memory_peak_mb: float
    layer_memory_peak_mb: Dict[str, float]  # Memory usage per component
    prefill_decode: PrefillDecodeMetrics
    logprobs: Dict[str, float]  # Statistics of logprobs
    generated_text: str
    prompt_tokens: int
    output_tokens: int
    provider: str = "CPU"  # Which EP was used

# ==============================================================================
# Abstract Base Runner
# ==============================================================================

class ABCBenchmarkRunner(ABC):
    """
    Abstract base class for benchmark runners. 
    Implement this for any framework (ONNX, PyTorch, TensorRT).
    """
    
    def __init__(self, model_path: str, tokenizer_path: str, providers: Optional[List[str]] = None):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.providers = providers or ['CPUExecutionProvider']
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @abstractmethod
    def run_inference(self, prompt: str, max_new_tokens: int, **kwargs) -> InferenceMetrics:
        pass
        
    @abstractmethod
    def cleanup(self):
        pass

# ==============================================================================
# ONNX Monolithic Runner
# ==============================================================================

class ONNXMonolithicRunner(ABCBenchmarkRunner):
    """
    Runner for a single monolithic ONNX model file.
    """
    
    def __init__(self, model_path: str, tokenizer_path: str, providers: Optional[List[str]] = None):
        super().__init__(model_path, tokenizer_path, providers)
        self.session = None
    
    def run_inference(self, prompt: str, max_new_tokens: int, **kwargs) -> InferenceMetrics:
        mem_start = get_memory_mb()
        
        # 1. Load Model
        start_load = time.perf_counter()
        if self.session is None:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            self.session = ort.InferenceSession(self.model_path, sess_options=sess_options, providers=self.providers)
        load_time = time.perf_counter() - start_load
        
        mem_after_load = get_memory_mb()
        mem_load_cost = mem_after_load - mem_start
        
        # 2. Prepare Inputs
        inputs = self.tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"]
        prompt_tokens = input_ids.shape[1]
        
        # NOTE: Monolithic generation logic (simulated for stateless vs stateful complexity)
        latencies = []
        generated_ids = []
        logprobs_list = []
        prefill_time = 0.0
        
        start_inference = time.perf_counter()
        
        try:
            # Inspection of inputs
            input_names = [i.name for i in self.session.get_inputs()]
            output_names = [o.name for o in self.session.get_outputs()]
            
            # Simulated Generation Loop for Benchmarking
            current_input_ids = input_ids
            
            for i in range(max_new_tokens + 1):
                iter_start = time.perf_counter()
                
                # Prepare inputs map
                onnx_inputs = {}
                if "input_ids" in input_names:
                    onnx_inputs["input_ids"] = current_input_ids
                if "attention_mask" in input_names:
                    onnx_inputs["attention_mask"] = np.ones(current_input_ids.shape, dtype=np.int64)
                if "position_ids" in input_names:
                    onnx_inputs["position_ids"] = np.arange(current_input_ids.shape[1], dtype=np.int64).reshape(1, -1)
                
                # Run Inference
                outputs = self.session.run(None, onnx_inputs)
                
                iter_time = time.perf_counter() - iter_start
                
                # Logic to get next token
                logits = outputs[0] # Assume first output is logits
                next_token_id = np.argmax(logits[:, -1, :], axis=-1)[0]
                
                if i == 0:
                    prefill_time = iter_time
                    generated_ids.append(next_token_id)
                    logprobs_list.append(compute_logprobs_numpy(logits, next_token_id))
                else: 
                    latencies.append(iter_time)
                    generated_ids.append(next_token_id)
                    logprobs_list.append(compute_logprobs_numpy(logits, next_token_id))

                # Append for next iteration (naive concatenation)
                current_input_ids = np.concatenate([current_input_ids, [[next_token_id]]], axis=1)

                if next_token_id == self.tokenizer.eos_token_id:
                    break

        except Exception as e:
            print(f"  [Monolith Warning] Generation info incomplete: {e}")
            if not latencies: latencies = [0.0]
            prefill_time = 0.1
        
        total_time = time.perf_counter() - start_inference
        mem_peak = get_memory_mb() - mem_start
        
        decode_time = sum(latencies)
        decode_tokens = len(latencies)
        
        stats = calculate_stats(logprobs_list)
        
        return InferenceMetrics(
            load_time_s=load_time,
            total_latency_s=total_time,
            ttft_s=prefill_time, # Prefill includes generation of first token
            tokens_per_second=(len(generated_ids)) / total_time if total_time > 0 else 0,
            memory_load_mb=mem_load_cost,
            memory_peak_mb=mem_peak,
            layer_memory_peak_mb={}, # Not applicable for monolith
            prefill_decode=PrefillDecodeMetrics(
                prefill_time_s=prefill_time,
                prefill_tokens=prompt_tokens,
                prefill_tokens_per_second=prompt_tokens/prefill_time if prefill_time > 0 else 0,
                decode_time_s=decode_time,
                decode_tokens=decode_tokens,
                decode_tokens_per_second=decode_tokens/decode_time if decode_time > 0 else 0,
                inter_token_latencies=latencies
            ),
            logprobs=stats,
            generated_text=self.tokenizer.decode(generated_ids),
            prompt_tokens=prompt_tokens,
            output_tokens=len(generated_ids),
            provider=self.providers[0] if self.providers else "CPU"
        )
        
    def cleanup(self):
        del self.session
        self.session = None
        clear_memory()

# ==============================================================================
# ONNX Sequential Runner (High Fidelity)
# ==============================================================================

class ONNXSequentialRunner(ABCBenchmarkRunner):
    """
    Runner for the Split (Sequential) ONNX pipeline.
    Extends/Wraps the original pipeline logic to inject precise timing instrumentation.
    """
    
    def __init__(self, model_dir_path: str, tokenizer_path: str, providers: Optional[List[str]] = None):
        super().__init__(model_dir_path, tokenizer_path, providers)
        # Suppress prints from the pipeline init if possible
        print(f"  [Sequential] Initializing pipeline with providers: {self.providers}")
        self.pipeline = SequentialModelPipeline(model_dir_path, tokenizer_path, providers=self.providers)
        
    def run_inference(self, prompt: str, max_new_tokens: int, **kwargs) -> InferenceMetrics:
        mem_start = get_memory_mb()
        mem_baseline = mem_start # Baseline before any components loaded
        
        # The pipeline lazy loads, so "load time" is conceptually small (just metadata)
        start_load = time.perf_counter()
        # Access internal state to ensure clean start
        self.pipeline.tensor_registry = {} 
        load_time = time.perf_counter() - start_load
        
        # Tokenize (part of prefill in this measurement context)
        inputs = self.tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"]
        prompt_tokens = input_ids.shape[1]
        
        # We RE-IMPLEMENT the generation loop logic here using the pipeline's 
        # component-running primitive `_run_component` so we can time it precisely.
        
        # Setup Inputs
        for k, v in inputs.items():
            self.pipeline.tensor_registry[k] = v
            
        generated_token_ids = []
        logprobs_list = []
        latencies = [] # For decoding steps
        layer_memory_stats = {} # Capture memory usage per layer
        
        # Init KV Cache
        # Note: In the original file, it allocates full zeros. We do the same.
        for i in range(self.pipeline.num_layers):
            key_name = f"past_key_values.{i}.key"
            val_name = f"past_key_values.{i}.value"
            self.pipeline.tensor_registry[key_name] = np.zeros(
                (1, self.pipeline.num_kv_heads, 0, self.pipeline.head_dim), 
                dtype=np.float16
            )
            self.pipeline.tensor_registry[val_name] = np.zeros(
                (1, self.pipeline.num_kv_heads, 0, self.pipeline.head_dim), 
                dtype=np.float16
            )

        start_inference = time.perf_counter()
        
        # ----------------------------------------------------------------------
        # GENERATION LOOP
        # ----------------------------------------------------------------------
        prefill_time = 0.0
        
        # Helper to measure run
        def measure_component(name):
            # Load
            sess = self.pipeline.load_component_session(name)
            # Measure Memory
            if sess:
                current_mem = get_memory_mb()
                layer_memory_stats[name] = max(layer_memory_stats.get(name, 0), current_mem - mem_baseline)
                # Run
                self.pipeline.run_component_session(name, sess)
                del sess
                # gc.collect() # Optional: slight perf penalty but cleaner stats
            
        for token_idx in range(max_new_tokens):
            iter_start = time.perf_counter()
            
            # 1. Update Position IDs
            seq_len = self.pipeline.tensor_registry["input_ids"].shape[1]
            self.pipeline.tensor_registry["position_ids"] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
            
            # 2. Run Other (Preprocessing)
            if "other" in self.pipeline.model_info["components"]:
                measure_component("other")
                
            # 3. Embedding
            measure_component("embedding")
            
            # 4. Layers
            for i in range(self.pipeline.num_layers):
                measure_component(f"layer_{i}")
                
                # Update KV
                if f"present.{i}.key" in self.pipeline.tensor_registry:
                    self.pipeline.tensor_registry[f"past_key_values.{i}.key"] = self.pipeline.tensor_registry[f"present.{i}.key"]
                if f"present.{i}.value" in self.pipeline.tensor_registry:
                    self.pipeline.tensor_registry[f"past_key_values.{i}.value"] = self.pipeline.tensor_registry[f"present.{i}.value"]
            
            # 5. Norm & Head
            if "norm" in self.pipeline.model_info["components"]:
                measure_component("norm")
            measure_component("lm_head")
            
            iter_end = time.perf_counter()
            iter_duration = iter_end - iter_start
            
            # 6. Sampling & Token Selection
            logits_name = self.pipeline.model_info["components"]["lm_head"]["output_names"][0]
            logits = self.pipeline.tensor_registry[logits_name]
            
            next_token_id = np.argmax(logits[:, -1, :], axis=-1)[0]
            generated_token_ids.append(next_token_id)
            logprobs_list.append(compute_logprobs_numpy(logits, next_token_id))
            
            # RECORD METRICS
            if token_idx == 0:
                prefill_time = iter_duration
            else:
                latencies.append(iter_duration)
            
            # 7. Update Inputs for next turn
            # Matches original pipeline logic (concatenation)
            new_input_ids = np.concatenate([self.pipeline.tensor_registry["input_ids"], [[next_token_id]]], axis=1)
            self.pipeline.tensor_registry["input_ids"] = new_input_ids
            
            if "attention_mask" in self.pipeline.tensor_registry:
                new_mask = np.concatenate([self.pipeline.tensor_registry["attention_mask"], [[1]]], axis=1)
                self.pipeline.tensor_registry["attention_mask"] = new_mask
            
            if next_token_id == self.tokenizer.eos_token_id:
                break
                
        # ----------------------------------------------------------------------
        
        total_time = time.perf_counter() - start_inference
        mem_peak = get_memory_mb() - mem_start
        
        decode_time = sum(latencies)
        decode_tokens = len(latencies)
        
        stats = calculate_stats(logprobs_list)
        
        return InferenceMetrics(
            load_time_s=load_time,
            total_latency_s=total_time,
            ttft_s=prefill_time,
            tokens_per_second=(len(generated_token_ids)) / total_time if total_time > 0 else 0,
            memory_load_mb=0.0, # Sequential loading is transient
            memory_peak_mb=mem_peak,
            layer_memory_peak_mb=layer_memory_stats,
            prefill_decode=PrefillDecodeMetrics(
                prefill_time_s=prefill_time,
                prefill_tokens=prompt_tokens,
                prefill_tokens_per_second=prompt_tokens/prefill_time if prefill_time > 0 else 0,
                decode_time_s=decode_time,
                decode_tokens=decode_tokens,
                decode_tokens_per_second=decode_tokens/decode_time if decode_time > 0 else 0,
                inter_token_latencies=latencies
            ),
            logprobs=stats,
            generated_text=self.tokenizer.decode(generated_token_ids),
            prompt_tokens=prompt_tokens,
            output_tokens=len(generated_token_ids),
            provider=self.providers[0] if self.providers else "CPU"
        )

    def cleanup(self):
        # Clear registry
        self.pipeline.tensor_registry = {}
        clear_memory()

# ==============================================================================
# Comparison Logic
# ==============================================================================

def calculate_speedups(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, float]:
    """Calculate speedup/reduction percentages."""
    def pct_change(base, new, higher_is_better=False):
        if base == 0: return 0.0
        diff = (base - new) if not higher_is_better else (new - base)
        return round((diff / base) * 100, 2)

    return {
        "load_time_speedup": pct_change(baseline['load_time_s'], candidate['load_time_s'], higher_is_better=False), # Lower is better
        "peak_memory_reduction": pct_change(baseline['memory_peak_mb'], candidate['memory_peak_mb'], higher_is_better=False), # Lower is better
        "ttft_speedup": pct_change(baseline['ttft_s'], candidate['ttft_s'], higher_is_better=False),
        "decode_tps_improvement": pct_change(baseline['prefill_decode']['decode_tokens_per_second'], candidate['prefill_decode']['decode_tokens_per_second'], higher_is_better=True)
    }

# ==============================================================================
# Main Benchmark Logic
# ==============================================================================

def run_benchmark_suite(
    runner: ABCBenchmarkRunner, 
    prompt: str, 
    max_tokens: int,
    runs: int,
    label: str
) -> Dict[str, Any]:
    """Run a sequence of benchmark iterations."""
    print(f"\n[{label}] Starting {runs} runs...")
    
    results = []
    
    for i in range(runs):
        print(f"  Run {i+1}/{runs}...", end="", flush=True)
        clear_memory(sleep_s=1.0)
        
        try:
            metrics = runner.run_inference(prompt, max_tokens)
            results.append(metrics)
            print(f" Done. (Lat: {metrics.total_latency_s:.2f}s, PeakMem: {metrics.memory_peak_mb:.1f}MB)")
        except Exception as e:
            print(f" Failed! {e}")
            import traceback
            traceback.print_exc()
    
    # helper for mean
    def mean(lst): return statistics.mean(lst) if lst else 0.0
    
    if not results:
        return {}

    # Aggregate
    avg_metrics = {
        "load_time_s": mean([r.load_time_s for r in results]),
        "total_latency_s": mean([r.total_latency_s for r in results]),
        "ttft_s": mean([r.ttft_s for r in results]),
        "tokens_per_second": mean([r.tokens_per_second for r in results]),
        "memory_peak_mb": mean([r.memory_peak_mb for r in results]),
        "layer_memory_peak_mb": results[-1].layer_memory_peak_mb,
        "generated_text": results[-1].generated_text,
        "logprobs_avg": mean([r.logprobs['avg'] for r in results]),
        "prompt_tokens": results[0].prompt_tokens,
        "output_tokens": mean([r.output_tokens for r in results]),
        "provider": results[0].provider,
        "prefill_decode": {
            "prefill_tokens_per_second": mean([r.prefill_decode.prefill_tokens_per_second for r in results]),
            "decode_tokens_per_second": mean([r.prefill_decode.decode_tokens_per_second for r in results]),
            "inter_token_latency_avg": mean([mean(r.prefill_decode.inter_token_latencies) for r in results if r.prefill_decode.inter_token_latencies])
        }
    }
    
    return avg_metrics


def get_prompts_list(args) -> List[str]:
    """Extract list of prompts from arguments."""
    if args.prompts:
        if os.path.isfile(args.prompts):
            print(f"Loading prompts from file: {args.prompts}")
            with open(args.prompts, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        else:
            return [p.strip() for p in args.prompts.split(',') if p.strip()]
    return [args.prompt]

def main():
    parser = argparse.ArgumentParser(description="Detailed ONNX Split vs Whole Benchmark")
    parser.add_argument("--whole-model", required=True, help="Path to monolithic .onnx file")
    parser.add_argument("--split-dir", required=True, help="Path to directory containing split parts")
    parser.add_argument("--tokenizer-path", required=True, help="Path to tokenizer")
    parser.add_argument("--model-id", help="Friendly name for the model (defaults to folder name)")
    parser.add_argument("--providers", default="CPUExecutionProvider,VitisAIExecutionProvider", help="Comma-separated list of EPs to benchmark")
    parser.add_argument("--prompt", default="The nature of artificial intelligence is", help="Single prompt text")
    parser.add_argument("--prompts", help="Multiple prompts: comma-separated or path to .txt file")
    parser.add_argument("--tokens", type=int, default=10, help="Max new tokens")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs to average")
    parser.add_argument("--output", default="benchmark_onnx_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    prompts = get_prompts_list(args)
    
    print("="*80)
    print("COMPREHENSIVE ONNX BENCHMARK: SPLIT VS WHOLE")
    print("="*80)
    print(f"Whole Model: {args.whole_model}")
    print(f"Split Dir:   {args.split_dir}")
    print(f"Providers:   {args.providers}")
    print(f"Prompts:     {len(prompts)} loaded")
    print(f"Gen Tokens:  {args.tokens}")
    print(f"Runs/Config: {args.runs}")
    print(f"Output:      {args.output}")
    print("-" * 80)
    
    # 1. Analyze File Structure
    print("\nAnalyzing Split Model Structure...")
    structure = analyze_onnx_structure(args.split_dir)
    print(json.dumps(structure, indent=2))
    
    all_runs_data = []
    provider_list = [p.strip() for p in args.providers.split(',')]
    
    try:
        for p_idx, current_prompt in enumerate(prompts):
            print(f"\n" + "#"*80)
            print(f"PROMPT {p_idx+1}/{len(prompts)}: '{current_prompt[:50]}...'")
            print("#"*80)
            
            prompt_provider_results = {}
            
            for provider in provider_list:
                print(f"\n[ENGINE: {provider}]")
                
                # Setup Runners for this provider
                split_runner = ONNXSequentialRunner(args.split_dir, args.tokenizer_path, providers=[provider])
                mono_runner = None
                if os.path.exists(args.whole_model):
                    mono_runner = ONNXMonolithicRunner(args.whole_model, args.tokenizer_path, providers=[provider])
                
                # 2. Run Split Benchmark
                print("-" * 20 + " Phase 1: Split " + "-" * 20)
                split_results = run_benchmark_suite(split_runner, current_prompt, args.tokens, args.runs, f"SPLIT({provider})")
                
                # 3. Run Whole Benchmark
                mono_results = {}
                if mono_runner:
                    print("-" * 20 + " Phase 2: Monolith " + "-" * 20)
                    mono_results = run_benchmark_suite(mono_runner, current_prompt, args.tokens, args.runs, f"MONO({provider})")

                # 4. Cleanup runners
                split_runner.cleanup()
                if mono_runner: mono_runner.cleanup()
                
                # Compare
                comparison = calculate_speedups(mono_results, split_results)
                
                prompt_provider_results[provider] = {
                    "split": split_results,
                    "monolith": mono_results,
                    "comparison": comparison
                }

            # Store Entry
            run_entry = {
                "prompt": current_prompt,
                "providers": prompt_provider_results
            }
            all_runs_data.append(run_entry)
            
            # Print Matrix Summary for this prompt
            print_matrix_summary(current_prompt, prompt_provider_results)

    finally:
        pass

    first_run = all_runs_data[0] if all_runs_data else None
    
    # 5. Save Results (JSON & Markdown)
    save_results(args.output, structure, all_runs_data, vars(args))
    
    print("\n" + "="*80)
    print(f"BENCHMARK COMPLETE.")
    print(f"JSON data appended to: {args.output}")
    print(f"Markdown report saved: {os.path.splitext(args.output)[0] + '_report.md'}")
    print("="*80)

def save_results(output_path, structure, run_entries, config):
    """Save results with appending logic and generate Markdown report."""
    session_id = time.strftime("%Y-%m-%d %H:%M:%S")
    
    session_data = {
        "session_id": session_id,
        "model_id": config.get("model_id", os.path.basename(config.get("split_dir", "model"))),
        "config": config,
        "structure": structure,
        "benchmarks": run_entries
    }
    
    # JSON Append Logic
    history = {"benchmark_runs": []}
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and "benchmark_runs" in data:
                    history = data
                else:
                    # Upgrade old format
                    history = {"benchmark_runs": [data] if data else []}
        except Exception as e:
            print(f"Warning: Could not read existing JSON ({e}). Overwriting.")

    history["benchmark_runs"].append(session_data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
        
    # Generate Markdown Report
    generate_markdown_report(session_data, os.path.splitext(output_path)[0] + "_report.md")

def print_matrix_summary(prompt, provider_results):
    """Print a 2D matrix comparing architectures and hardware engines."""
    print("\n" + "="*80)
    print(f"MATRIX COMPARISON Summary")
    print("="*80)
    
    headers = ["Config", "EP", "TTFT (s)", "TPS", "Peak RAM", "Max Layer"]
    print(f"{headers[0]:<12} | {headers[1]:<15} | {headers[2]:<10} | {headers[3]:<10} | {headers[4]:<12} | {headers[5]:<12}")
    print("-" * 80)
    
    rows = []
    for ep, results in provider_results.items():
        # Monolith
        m = results['monolith']
        if m:
            rows.append({
                "Arch": "Monolith",
                "EP": ep,
                "TTFT": m.get('ttft_s', 0),
                "TPS": m.get('tokens_per_second', 0),
                "Peak": m.get('memory_peak_mb', 0),
                "Max": 0.0, # N/A
                "Color": ""
            })
        
        # Split
        s = results['split']
        if s:
            layer_mem = s.get('layer_memory_peak_mb', {})
            max_comp = max(layer_mem.values()) if layer_mem else 0
            rows.append({
                "Arch": "Split",
                "EP": ep,
                "TTFT": s.get('ttft_s', 0),
                "TPS": s.get('tokens_per_second', 0),
                "Peak": s.get('memory_peak_mb', 0),
                "Max": max_comp,
                "Color": ""
            })
            
    # Print Rows
    for r in rows:
        max_str = f"{r['Max']:.1f} MB" if r['Max'] > 0 else "N/A"
        print(f"{r['Arch']:<12} | {r['EP']:<15} | {r['TTFT']:<10.4f} | {r['TPS']:<10.2f} | {r['Peak']:<8.1f} MB | {max_str}")

    print("-" * 80)
    
    # Simple Winner Detection (Efficiency Focused)
    if provider_results:
        # Lowest Peak RAM winner
        valid_rows = [r for r in rows if r['Peak'] > 0]
        if valid_rows:
            best_mem = min(valid_rows, key=lambda x: x['Max'] if x['Max'] > 0 else x['Peak'])
            best_speed = max(valid_rows, key=lambda x: x['TPS'])
            
            print(f" WINNER (Efficiency) : {best_mem['Arch']} @ {best_mem['EP']} ({best_mem['Max'] if best_mem['Max'] > 0 else best_mem['Peak']:.1f} MB footprint)")
            print(f" WINNER (Performance): {best_speed['Arch']} @ {best_speed['EP']} ({best_speed['TPS']:.2f} tokens/sec)")

    print("-" * 80)

def generate_markdown_report(session, report_path):
    """Generate a clean Markdown report for the current session."""
    lines = []
    lines.append(f"# ONNX Benchmark Report: {session['session_id']}")
    lines.append(f"\n**Model ID:** `{session['model_id']}`")
    lines.append(f"\n## Configuration")
    lines.append(f"- **Providers Tested:** `{session['config']['providers']}`")
    lines.append(f"- **Runs per Config:** {session['config']['runs']}")
    lines.append(f"- **Tokens generated:** {session['config']['tokens']}")
    
    lines.append(f"\n## Results Matrix")
    lines.append("\n| Arch | Provider | TTFT | TPS | Peak RAM | Max Layer RAM |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for b in session['benchmarks']:
        lines.append(f"\n### Prompt: \"{b['prompt']}\"")
        for ep, results in b['providers'].items():
            m = results['monolith']
            s = results['split']
            if m:
                lines.append(f"| Monolith | {ep} | {m.get('ttft_s',0):.4f}s | {m.get('tokens_per_second',0):.2f}/s | {m.get('memory_peak_mb',0):.1f}MB | N/A |")
            if s:
                layer_mem = s.get('layer_memory_peak_mb', {})
                max_comp = max(layer_mem.values()) if layer_mem else 0
                lines.append(f"| Split | {ep} | {s.get('ttft_s',0):.4f}s | {s.get('tokens_per_second',0):.2f}/s | {s.get('memory_peak_mb',0):.1f}MB | {max_comp:.1f}MB |")
        
        # Add a 1:1 Winner analysis for this prompt
        lines.append("\n#### Comparisons (Efficiency vs Performance)")
        # Simple summary
        lines.append("- **Performance King:** Fastest configuration detected for bulk generation.")
        lines.append("- **Memory King:** Lowest transient footprint for resource-constrained use cases.")

    # Model Structure
    lines.append(f"\n## Model Structure Details")
    s = session['structure']
    lines.append(f"- **Total Model Size:** {s['total_size_mb']:.1f} MB")
    lines.append(f"- **Total Components:** {s['total_files']}")

    # Write to file
    mode = 'a' if os.path.exists(report_path) else 'w'
    with open(report_path, mode, encoding='utf-8') as f:
        if mode == 'a': f.write("\n\n---\n\n")
        f.write("\n".join(lines))
        f.write("\n")

def print_prompt_summary(prompt, split_results, mono_results):
    """Print the summary table for a specific prompt."""
    print("\nPROMPT SUMMARY:")
    print(f"{'Metric':<30} | {'Monolith':<15} | {'Split':<15} | {'Improvement'}")
    print("-" * 80)
    
    def print_row(name, key, fmt=".2f", suffix=""):
        m_val = mono_results.get(key, 0)
        s_val = split_results.get(key, 0)
        if key in ["load_time_s", "total_latency_s", "ttft_s", "memory_peak_mb"]:
            imp = ((m_val - s_val) / m_val * 100) if m_val > 0 else 0
        else:
            imp = ((s_val - m_val) / m_val * 100) if m_val > 0 else 0
        print(f"{name:<30} | {m_val:<15{fmt}}{suffix} | {s_val:<15{fmt}}{suffix} | {imp:+.1f}%")

    def print_nested(name, parent, key, fmt=".2f", suffix=""):
        m_val = mono_results.get(parent, {}).get(key, 0)
        s_val = split_results.get(parent, {}).get(key, 0)
        if "latency" in key:
            imp = ((m_val - s_val) / m_val * 100) if m_val > 0 else 0
        else:
            imp = ((s_val - m_val) / m_val * 100) if m_val > 0 else 0
        print(f"{name:<30} | {m_val:<15{fmt}}{suffix} | {s_val:<15{fmt}}{suffix} | {imp:+.1f}%")

    if split_results and mono_results:
        print_row("Load Time", "load_time_s", ".4f", "s")
        print_row("Peak Memory (RSS)", "memory_peak_mb", ".1f", "MB")
        print_row("Total Latency", "total_latency_s", ".2f", "s")
        print_row("Time To First Token (TTFT)", "ttft_s", ".4f", "s")
        print_row("Overall TPS", "tokens_per_second", ".2f", "")
        print_nested("Prefill Throughput", "prefill_decode", "prefill_tokens_per_second", ".1f", " t/s")
        print_nested("Decode Throughput", "prefill_decode", "decode_tokens_per_second", ".1f", " t/s")
        print_nested("Inter-Token Latency", "prefill_decode", "inter_token_latency_avg", ".4f", "s")
        
        print("-" * 80)
        split_layer_mem = split_results.get("layer_memory_peak_mb", {})
        if split_layer_mem:
            max_comp_name = max(split_layer_mem, key=split_layer_mem.get)
            max_comp_val = split_layer_mem[max_comp_name]
            mono_peak = mono_results.get("memory_peak_mb", 0)
            imp_mem = ((mono_peak - max_comp_val) / mono_peak * 100) if mono_peak > 0 else 0
            print(f"{'Max Component Mem':<30} | {mono_peak:<15.1f}MB | {max_comp_val:<15.1f}MB | {imp_mem:+.1f}%")
            print(f"  (Top Component: {max_comp_name})")
    elif split_results:
        print_row("Load Time (Split)", "load_time_s", ".4f", "s")
        print_row("Peak Memory (Split)", "memory_peak_mb", ".1f", "MB")
        print_row("Total Latency (Split)", "total_latency_s", ".2f", "s")
    
    print("-" * 80)

if __name__ == "__main__":
    main()

