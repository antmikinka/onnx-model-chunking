"""
Configuration file utilities for SafeTensors models.

This module provides utilities for handling model configuration files,
including finding complete configs and ensuring required fields are present.
"""

import os
import json
from typing import Dict, List, Optional, Any


def find_complete_config(model_path: str) -> Optional[Dict[str, Any]]:
    """
    Find the most complete config file in a model directory.
    
    Searches for config.json and any backup config files, returning
    the one with the most complete information.
    
    Args:
        model_path: Path to the model directory.
    
    Returns:
        Dictionary with config data, or None if no config found.
    """
    required_fields = ["model_type", "num_hidden_layers", "hidden_size"]
    
    best_config: Optional[Dict[str, Any]] = None
    best_score = 0
    
    try:
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
    except OSError:
        pass
    
    return best_config


def ensure_complete_config(model_path: str) -> bool:
    """
    Ensure the model's config.json has all required fields.
    
    If config.json is incomplete, tries to find and merge data from
    other config files in the directory.
    
    Args:
        model_path: Path to the model directory.
    
    Returns:
        True if config is now complete, False otherwise.
    """
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return False
    
    with open(config_path, "r", encoding="utf-8") as f:
        current_config = json.load(f)
    
    required_fields = ["model_type", "num_hidden_layers", "hidden_size", "architectures"]
    missing_fields = [f for f in required_fields if f not in current_config]
    
    if not missing_fields:
        return True  # Config is complete
    
    print(f"Config is missing fields: {missing_fields}")
    print("Searching for complete config...")
    
    complete_config = find_complete_config(model_path)
    if complete_config:
        # Merge missing fields from complete config
        updated = False
        for field in missing_fields:
            if field in complete_config:
                current_config[field] = complete_config[field]
                print(f"  Merged '{field}' = {complete_config[field]}")
                updated = True
        
        if updated:
            # Write updated config
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(current_config, f, indent=2)
            print(f"Updated config.json with complete info")
            return True
    
    print("Warning: Could not find complete config. Will try architecture auto-detection.")
    return False


def get_config_field(model_path: str, field: str, default: Any = None) -> Any:
    """
    Get a specific field from the model's config.json.
    
    Args:
        model_path: Path to the model directory.
        field: Name of the config field to retrieve.
        default: Default value if field not found.
    
    Returns:
        The field value, or default if not found.
    """
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return default
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get(field, default)
    except (json.JSONDecodeError, IOError):
        return default


def get_num_layers(model_path: str) -> Optional[int]:
    """
    Get the number of hidden layers from the model config.
    
    Args:
        model_path: Path to the model directory.
    
    Returns:
        Number of hidden layers, or None if not found.
    """
    return get_config_field(model_path, "num_hidden_layers")


def get_model_type(model_path: str) -> Optional[str]:
    """
    Get the model type from the model config.
    
    Args:
        model_path: Path to the model directory.
    
    Returns:
        Model type string (e.g., "llama", "qwen2"), or None if not found.
    """
    return get_config_field(model_path, "model_type")
