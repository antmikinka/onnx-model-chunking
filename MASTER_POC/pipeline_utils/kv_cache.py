#!/usr/bin/env python3
"""
kv_cache.py - KV Cache Management Module

This module provides a KVCacheManager class for managing key-value cache state
during incremental transformer decoding.

Usage:
    from pipeline_utils.kv_cache import KVCacheManager
    
    cache = KVCacheManager(num_layers=24, num_kv_heads=8, head_dim=128, dtype=np.float16)
    cache.inject_into_registry(tensor_registry)
    # After layer run:
    cache.update_from_present(layer_idx, present_key, present_value)
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np


class KVCacheManager:
    """
    Manages KV (Key-Value) cache state for incremental transformer decoding.
    
    In autoregressive generation with KV caching:
    - First iteration (prefill): Process full prompt, populate cache
    - Subsequent iterations (decode): Process only new token, use cached keys/values
    
    Attributes:
        num_layers: Number of transformer layers
        num_kv_heads: Number of key-value heads per layer
        head_dim: Dimension of each head
        dtype: Data type for cache tensors
        batch_size: Batch size (default 1)
    """
    
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: Any = np.float16,
        batch_size: int = 1
    ):
        """
        Initialize the KV cache manager.
        
        Args:
            num_layers: Number of transformer layers
            num_kv_heads: Number of key-value attention heads
            head_dim: Dimension of each attention head
            dtype: Numpy dtype for cache tensors (default: float16)
            batch_size: Batch size for inference (default: 1)
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.batch_size = batch_size
        
        # Internal cache storage: layer_idx -> {"key": array, "value": array}
        self._cache: Dict[int, Dict[str, np.ndarray]] = {}
        
        # Initialize with empty cache
        self.reset()
    
    def reset(self) -> None:
        """
        Reset the cache to empty state.
        
        Creates empty cache tensors with shape [batch, heads, 0, head_dim]
        representing 0 past tokens. Call this before each new generation session.
        """
        self._cache.clear()
        for i in range(self.num_layers):
            self._cache[i] = {
                "key": np.zeros(
                    (self.batch_size, self.num_kv_heads, 0, self.head_dim),
                    dtype=self.dtype
                ),
                "value": np.zeros(
                    (self.batch_size, self.num_kv_heads, 0, self.head_dim),
                    dtype=self.dtype
                )
            }
    
    def get_sequence_length(self) -> int:
        """
        Get the current cached sequence length.
        
        Returns:
            Number of tokens currently in the cache (0 if empty)
        """
        if 0 in self._cache and self._cache[0]["key"].shape[2] > 0:
            return self._cache[0]["key"].shape[2]
        return 0
    
    def get_past_tensors(self, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the past key and value tensors for a specific layer.
        
        Args:
            layer_idx: Index of the transformer layer (0-indexed)
            
        Returns:
            Tuple of (past_key, past_value) arrays
            
        Raises:
            KeyError: If layer_idx is out of range
        """
        if layer_idx not in self._cache:
            raise KeyError(f"Layer {layer_idx} not in cache (have {self.num_layers} layers)")
        return self._cache[layer_idx]["key"], self._cache[layer_idx]["value"]
    
    def update_from_present(
        self,
        layer_idx: int,
        present_key: np.ndarray,
        present_value: np.ndarray,
        validate_shape: bool = True
    ) -> None:
        """
        Update cache for a layer with new present key/value tensors.
        
        After running a layer forward pass, the layer outputs 'present' tensors
        that contain the updated cache. This method stores them for the next iteration.
        
        Args:
            layer_idx: Index of the transformer layer
            present_key: The present.{layer_idx}.key output from the layer
            present_value: The present.{layer_idx}.value output from the layer
            validate_shape: If True, validate tensor shapes (default True)
        """
        if layer_idx not in self._cache:
            raise KeyError(f"Layer {layer_idx} not in cache (have {self.num_layers} layers)")
        
        if validate_shape:
            # Validate present_key and present_value have matching shapes
            if present_key.shape != present_value.shape:
                raise ValueError(
                    f"Layer {layer_idx}: present_key shape {present_key.shape} != "
                    f"present_value shape {present_value.shape}"
                )
            
            # Validate expected dimensions: [batch, heads, seq, head_dim]
            if len(present_key.shape) != 4:
                raise ValueError(
                    f"Layer {layer_idx}: Expected 4D tensor [batch, heads, seq, head_dim], "
                    f"got shape {present_key.shape}"
                )
            
            # Validate heads and head_dim match expected
            if present_key.shape[1] != self.num_kv_heads:
                raise ValueError(
                    f"Layer {layer_idx}: Expected {self.num_kv_heads} KV heads, "
                    f"got {present_key.shape[1]}"
                )
            if present_key.shape[3] != self.head_dim:
                raise ValueError(
                    f"Layer {layer_idx}: Expected head_dim {self.head_dim}, "
                    f"got {present_key.shape[3]}"
                )
        
        self._cache[layer_idx]["key"] = present_key
        self._cache[layer_idx]["value"] = present_value
    
    def inject_into_registry(
        self,
        registry: Dict[str, np.ndarray],
        key_format: str = "past_key_values.{layer}.key",
        value_format: str = "past_key_values.{layer}.value"
    ) -> None:
        """
        Inject all cache tensors into a tensor registry.
        
        This is used to prepare inputs for ONNX model layers that expect
        past_key_values inputs.
        
        Args:
            registry: Dictionary to inject cache tensors into
            key_format: Format string for key tensor names (use {layer} placeholder)
            value_format: Format string for value tensor names (use {layer} placeholder)
        """
        for i in range(self.num_layers):
            key_name = key_format.format(layer=i)
            value_name = value_format.format(layer=i)
            registry[key_name] = self._cache[i]["key"]
            registry[value_name] = self._cache[i]["value"]
    
    def extract_from_registry(
        self,
        registry: Dict[str, np.ndarray],
        present_key_format: str = "present.{layer}.key",
        present_value_format: str = "present.{layer}.value"
    ) -> int:
        """
        Extract present tensors from registry and update cache.
        
        This is a convenience method to update all layers' cache from
        a tensor registry after running the full model.
        
        Args:
            registry: Dictionary containing present.* tensors
            present_key_format: Format string for present key names
            present_value_format: Format string for present value names
            
        Returns:
            Number of layers successfully updated
        """
        updated = 0
        for i in range(self.num_layers):
            key_name = present_key_format.format(layer=i)
            value_name = present_value_format.format(layer=i)
            
            if key_name in registry and value_name in registry:
                self.update_from_present(i, registry[key_name], registry[value_name])
                updated += 1
        
        return updated
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage statistics for the cache.
        
        Returns:
            Dictionary with memory stats (total_bytes, per_layer_bytes, etc.)
        """
        total_bytes = 0
        per_layer = []
        
        for i in range(self.num_layers):
            key_bytes = self._cache[i]["key"].nbytes
            value_bytes = self._cache[i]["value"].nbytes
            layer_bytes = key_bytes + value_bytes
            per_layer.append(layer_bytes)
            total_bytes += layer_bytes
        
        seq_len = self.get_sequence_length()
        
        return {
            "total_bytes": total_bytes,
            "total_mb": total_bytes / (1024 * 1024),
            "per_layer_bytes": per_layer,
            "sequence_length": seq_len,
            "num_layers": self.num_layers,
            "dtype": str(self.dtype),
        }
    
    def __repr__(self) -> str:
        seq_len = self.get_sequence_length()
        mem = self.get_memory_usage()
        return (
            f"KVCacheManager(layers={self.num_layers}, heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, seq_len={seq_len}, memory={mem['total_mb']:.2f}MB)"
        )


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

__all__ = ["KVCacheManager"]
