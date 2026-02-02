#!/usr/bin/env python3
# Modular PyTorch Model Splitter and CPU Inference
# Outputs one safetensors file per layer + embeddings/norm/lm_head
# Supports both HuggingFace Hub models and local model paths

import os
import re
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import snapshot_download


# ==============================================================================
# Utility Functions
# ==============================================================================

def is_local_path(model_id: str) -> bool:
    """Check if the model_id is a local filesystem path."""
    return os.path.isdir(model_id) or os.path.isabs(model_id)


def sanitize_model_name(model_id: str) -> str:
    """Convert a model ID or path to a filesystem-safe folder name.
    
    For local paths, uses the folder basename to keep output names clean.
    For Hub IDs, replaces slash with double dash.
    """
    # For local paths, use just the folder basename
    if is_local_path(model_id):
        # Normalize path and get the base folder name
        normalized = os.path.normpath(model_id)
        safe_name = os.path.basename(normalized)
    else:
        # For Hub IDs like "meta-llama/Llama-2-7b"
        safe_name = model_id.replace("/", "--")
    
    # Remove illegal Windows characters
    safe_name = re.sub(r'[<>:"|?*\\]', '', safe_name)
    # Strip leading/trailing whitespace
    return safe_name.strip()


def get_model_prefixes(config):
    """Return weight prefixes for different model architectures."""
    model_type = getattr(config, 'model_type', None)
    if model_type in ["llama", "mistral", "phi", "gemma", "qwen2", "qwen3"]:
        embed_prefix = "model.embed_tokens."
        layer_prefix = "model.layers."
        norm_prefix = "model.norm."
        lm_prefix = "lm_head."
    elif model_type == "gpt2":
        embed_prefix = "transformer.wte."
        layer_prefix = "transformer.h."
        norm_prefix = "transformer.ln_f."
        lm_prefix = "lm_head."
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return embed_prefix, layer_prefix, norm_prefix, lm_prefix


def detect_architecture_from_weights(state_dict: dict) -> str:
    """Detect model architecture by examining weight key patterns.
    
    Returns the detected model_type string or raises ValueError if unknown.
    """
    keys = list(state_dict.keys())
    sample_keys = keys[:50]  # Check first 50 keys
    
    # Check for common architecture patterns
    if any("model.layers." in k for k in sample_keys):
        if any("self_attn.q_norm" in k or "self_attn.k_norm" in k for k in keys):
            return "qwen3"  # Qwen3 has q_norm/k_norm in attention
        return "llama"  # Generic LLaMA-style architecture
    elif any("transformer.h." in k for k in sample_keys):
        return "gpt2"
    
    raise ValueError(
        "Could not detect model architecture from weight keys. "
        "Please ensure your model has a valid config.json with 'model_type' field."
    )


def find_complete_config(model_path: str) -> dict:
    """Find the most complete config file in a model directory.
    
    Searches for config.json and any backup config files, returning
    the one with the most complete information.
    """
    config_candidates = [
        "config.json",
        "not-sure-if-this-part-of-project-config.json",  # Common backup name
    ]
    
    best_config = None
    best_score = 0
    required_fields = ["model_type", "num_hidden_layers", "hidden_size"]
    
    for filename in os.listdir(model_path):
        if filename.endswith(".json") and "config" in filename.lower():
            try:
                filepath = os.path.join(model_path, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                
                # Score based on required fields present
                score = sum(1 for field in required_fields if field in config_data)
                if score > best_score:
                    best_score = score
                    best_config = config_data
            except (json.JSONDecodeError, IOError):
                continue
    
    return best_config


def ensure_complete_config(model_path: str) -> None:
    """Ensure the model's config.json has all required fields.
    
    If config.json is incomplete, tries to find and merge data from
    other config files in the directory.
    """
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return
    
    with open(config_path, "r", encoding="utf-8") as f:
        current_config = json.load(f)
    
    required_fields = ["model_type", "num_hidden_layers", "hidden_size", "architectures"]
    missing_fields = [f for f in required_fields if f not in current_config]
    
    if not missing_fields:
        return  # Config is complete
    
    print(f"Config is missing fields: {missing_fields}")
    print("Searching for complete config...")
    
    complete_config = find_complete_config(model_path)
    if complete_config:
        # Merge missing fields from complete config
        for field in missing_fields:
            if field in complete_config:
                current_config[field] = complete_config[field]
                print(f"  Merged '{field}' = {complete_config[field]}")
        
        # Write updated config
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(current_config, f, indent=2)
        print(f"Updated config.json with complete info")
    else:
        print("Warning: Could not find complete config. Will try architecture auto-detection.")


# ==============================================================================
# Model Loading
# ==============================================================================

def download_model(model_id: str) -> str:
    """Ensure model is available locally, downloading from Hub if needed."""
    if os.path.exists(model_id):
        return model_id
    print(f"Path {model_id} not found. Downloading from Hub...")
    return snapshot_download(model_id)


def load_state_dict(model_path: str) -> dict:
    """Load state dict from safetensors or pytorch_model.bin."""
    model_path = download_model(model_path)
    state_dict = {}

    # Try sharded safetensors
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        import json
        with open(index_path, "r") as f:
            index = json.load(f)["weight_map"]
        files = set(index.values())
        for file in files:
            with safe_open(os.path.join(model_path, file), framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        return state_dict

    # Try single safetensors
    safetensors_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(safetensors_path):
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        return state_dict

    # Fallback to pytorch_model.bin
    pytorch_bin = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(pytorch_bin):
        return torch.load(pytorch_bin, map_location="cpu")

    raise FileNotFoundError(
        f"Could not find model weights in {model_path} "
        "(checked safetensors and pytorch_model.bin)"
    )


# ==============================================================================
# Model Splitting (Per-Layer Output)
# ==============================================================================

def split_model(params) -> str:
    """
    Split a HuggingFace model into per-layer safetensors files.
    Returns the output directory path.
    
    Supports both Hub model IDs and local filesystem paths.
    """
    model_path = params.model
    
    # For local paths, ensure config is complete before loading
    if is_local_path(model_path) and os.path.exists(model_path):
        print(f"Detected local model path: {model_path}")
        ensure_complete_config(model_path)
    
    # Load config, tokenizer, and weights
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Could not load config normally: {e}")
        print("Attempting to load weights first for architecture detection...")
        state_dict = load_state_dict(model_path)
        detected_type = detect_architecture_from_weights(state_dict)
        print(f"Detected architecture: {detected_type}")
        # Try loading config again with detected type hint
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # Load tokenizer with fallback
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        print("Splitting will continue without tokenizer save.")
        tokenizer = None
    
    # Load state dict if not already loaded
    state_dict = load_state_dict(model_path)

    # Create correlative output folder
    model_folder_name = sanitize_model_name(params.model)
    output_dir = os.path.join(params.output, model_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save config and tokenizer
    config.save_pretrained(output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

    # Get prefixes for this architecture
    embed_prefix, layer_prefix, norm_prefix, lm_prefix = get_model_prefixes(config)
    num_layers = config.num_hidden_layers

    print(f"Splitting model into {num_layers} layer files + embeddings/norm/lm_head...")

    # Save embeddings
    embed_keys = {k: v for k, v in state_dict.items() if k.startswith(embed_prefix)}
    if embed_keys:
        save_file(embed_keys, os.path.join(output_dir, "embeddings.safetensors"))
        print(f"  Saved embeddings.safetensors ({len(embed_keys)} tensors)")

    # Save each layer individually
    for i in range(num_layers):
        layer_keys = {
            k: v for k, v in state_dict.items()
            if k.startswith(f"{layer_prefix}{i}.")
        }
        if layer_keys:
            filename = f"layer_{i}_of_{num_layers}.safetensors"
            save_file(layer_keys, os.path.join(output_dir, filename))
            print(f"  Saved {filename} ({len(layer_keys)} tensors)")

    # Save norm
    norm_keys = {k: v for k, v in state_dict.items() if k.startswith(norm_prefix)}
    if norm_keys:
        save_file(norm_keys, os.path.join(output_dir, "norm.safetensors"))
        print(f"  Saved norm.safetensors ({len(norm_keys)} tensors)")

    # Save LM head
    lm_keys = {k: v for k, v in state_dict.items() if k.startswith(lm_prefix)}
    if lm_keys:
        save_file(lm_keys, os.path.join(output_dir, "lm_head.safetensors"))
        print(f"  Saved lm_head.safetensors ({len(lm_keys)} tensors)")

    print(f"Model split complete. Output: {output_dir}")
    return output_dir


# ==============================================================================
# Model Merging (for Inference)
# ==============================================================================

def merge_state_dict(output_dir: str) -> dict:
    """Merge all layer/embedding/norm/lm_head safetensors into a single state dict."""
    state_dict = {}
    exclude_files = {"kv_cache.safetensors", "logits.safetensors"}

    for file in sorted(os.listdir(output_dir)):
        if file.endswith(".safetensors") and file not in exclude_files:
            path = os.path.join(output_dir, file)
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

    return state_dict


# ==============================================================================
# Inference
# ==============================================================================

def infer(output_dir: str, params) -> list:
    """Run inference on the split model."""
    config = AutoConfig.from_pretrained(output_dir)
    full_state = merge_state_dict(output_dir)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
    model.load_state_dict(full_state)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(output_dir)

    # Batched prompts
    prompts = [params.prompt] * params.batch_size
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        max_length=params.context,
        truncation=True,
        padding=True
    )

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=params.max_new_tokens,
            use_cache=True,
            do_sample=False
        )

    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Save KV cache and logits
    with torch.no_grad():
        out = model(**inputs, use_cache=True)

    past_kv = out.past_key_values
    kv_dict = {}
    for i, (k, v) in enumerate(past_kv):
        kv_dict[f"layer_{i}.key"] = k
        kv_dict[f"layer_{i}.value"] = v
    save_file(kv_dict, os.path.join(output_dir, "kv_cache.safetensors"))

    logits_dict = {"logits": out.logits}
    save_file(logits_dict, os.path.join(output_dir, "logits.safetensors"))

    return texts


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Split HF model into per-layer safetensors and run CPU inference"
    )
    parser.add_argument(
        "--model", required=True, 
        help="HuggingFace model ID or local path (e.g., 'meta-llama/Llama-2-7b' or 'C:\\path\\to\\model')"
    )
    parser.add_argument("--output", required=True, help="Base directory for split output")
    parser.add_argument("--context", type=int, default=1024, help="Max context length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--prompt", default="Test prompt.", help="Prompt for inference")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Max new tokens")
    parser.add_argument(
        "--split-only", action="store_true",
        help="Only split the model, skip inference (useful for large models)"
    )
    return parser.parse_args()


def main():
    params = parse_args()
    print("=" * 60)
    print("Model Splitter - Local and Hub Model Support")
    print("=" * 60)
    print(f"Model: {params.model}")
    print(f"Output: {params.output}")
    print(f"Split Only: {params.split_only}")
    print("=" * 60)
    
    output_dir = split_model(params)
    
    if params.split_only:
        print("\n--split-only specified, skipping inference.")
        print(f"Split complete! Files saved to: {output_dir}")
    else:
        print("\nRunning inference on split model...")
        results = infer(output_dir, params)
        print("Generated texts:", results)


if __name__ == "__main__":
    main()