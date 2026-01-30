"""
Local ONNX Model Splitter

A modular, scalable system for splitting pre-existing ONNX models from local disk
into coherent chunks using weight-based or layer-based chunking strategies.

This module adapts the proven chunking algorithms from builder.py to work with
local ONNX model files rather than models built from HuggingFace.

Usage:
    from local_model_splitter import LocalONNXModelSplitter
    
    splitter = LocalONNXModelSplitter("path/to/model")
    splitter.split_model(output_dir="./chunks", num_chunks=2, strategy="weight_based")
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

# Try to import onnx_ir - the modern ONNX IR library used by builder.py
try:
    import onnx_ir as ir
    HAS_ONNX_IR = True
except ImportError:
    HAS_ONNX_IR = False

# Fall back to standard onnx if onnx_ir is not available
import onnx
from onnx import numpy_helper

# Import configuration system (reuse existing)
try:
    from chunking_config import (
        ConfigurationProcessor,
        UnifiedChunkingConfig,
        ChunkingMethod,
        ChunkStrategy,
        DistributionType
    )
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

# Import analytics system (optional)
try:
    from chunking_analytics import (
        ChunkingAnalytics,
        monitor_chunking_performance
    )
    HAS_ANALYTICS = True
except ImportError:
    HAS_ANALYTICS = False


@dataclass
class ChunkInfo:
    """Information about a single chunk"""
    chunk_idx: int
    start_node_idx: int
    end_node_idx: int
    node_count: int
    memory_mb: float
    boundary_inputs: List[str]
    boundary_outputs: List[str]


@dataclass
class SplitResult:
    """Result of a model split operation"""
    chunk_models: List[Any]  # List of onnx.ModelProto or ir.Model
    chunk_info: List[ChunkInfo]
    total_memory_mb: float
    split_points: List[int]
    metadata: Dict[str, Any]


class LocalONNXModelSplitter:
    """
    Modular ONNX model splitter for local models.
    
    Supports:
    - Weight-based splitting (equal memory distribution)
    - Layer-based splitting (respects layer boundaries)
    - Custom chunk size distribution
    - Both onnx_ir and standard onnx formats
    
    Attributes:
        model_path: Path to the ONNX model file or directory
        model: Loaded ONNX model
        graph: The model's graph
        config: Configuration for chunking
        analytics: Optional analytics system
    """
    
    def __init__(
        self, 
        model_path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
        use_onnx_ir: bool = True
    ):
        """
        Initialize the local model splitter.
        
        Args:
            model_path: Path to ONNX model file or directory containing model.onnx
            config: Optional configuration dictionary (uses chunking_config format)
            use_onnx_ir: Whether to use onnx_ir for graph manipulation (recommended)
        """
        self.model_path = Path(model_path)
        self.use_onnx_ir = use_onnx_ir and HAS_ONNX_IR
        self.model = None
        self.graph = None
        self.model_proto = None  # Keep original proto for reference
        
        # Analytics
        self.analytics = None
        if HAS_ANALYTICS:
            try:
                self.analytics = ChunkingAnalytics()
            except Exception:
                pass
        
        # Configuration
        self.config = config or {}
        self.num_chunks = 2
        self.chunk_strategy = "weight_based"
        self.chunk_distribution = "equal_memory"
        self.chunk_split_ratio = 0.5
        self.custom_chunk_sizes = None
        self.respect_layer_boundaries = True
        
        # Process configuration if available
        if HAS_CONFIG and config:
            self._process_configuration(config)
        elif config:
            self._process_simple_config(config)
        
        # Model analysis results
        self._total_memory_mb = None
        self._node_count = None
        self._initializer_count = None
        self._layer_boundaries = {}
    
    def _process_configuration(self, config: Dict[str, Any]) -> None:
        """Process configuration using the unified configuration system"""
        try:
            processor = ConfigurationProcessor()
            unified_config = processor.process_configuration(config)
            
            if unified_config.method == ChunkingMethod.ONNX_GRAPH_LEVEL.value:
                onnx_config = unified_config.onnx_graph_level
                self.num_chunks = onnx_config.num_chunks
                self.chunk_strategy = onnx_config.strategy
                self.chunk_distribution = onnx_config.distribution
                self.chunk_split_ratio = onnx_config.chunk_split_ratio
                self.custom_chunk_sizes = onnx_config.custom_chunk_sizes
                self.respect_layer_boundaries = onnx_config.respect_layer_boundaries
                
        except Exception as e:
            print(f"Warning: Failed to process unified configuration: {e}")
            self._process_simple_config(config)
    
    def _process_simple_config(self, config: Dict[str, Any]) -> None:
        """Process simple configuration dictionary"""
        self.num_chunks = config.get("num_chunks", 2)
        self.chunk_strategy = config.get("chunk_strategy", "weight_based")
        self.chunk_distribution = config.get("chunk_distribution", "equal_memory")
        self.chunk_split_ratio = config.get("chunk_split_ratio", 0.5)
        self.custom_chunk_sizes = config.get("custom_chunk_sizes", None)
        self.respect_layer_boundaries = config.get("respect_layer_boundaries", True)
    
    def _resolve_model_path(self) -> Path:
        """Resolve the actual path to the ONNX model file"""
        if self.model_path.is_file():
            return self.model_path
        elif self.model_path.is_dir():
            # Look for model.onnx in the directory
            model_file = self.model_path / "model.onnx"
            if model_file.exists():
                return model_file
            # Try other common names
            for name in ["model.onnx", "model_optimized.onnx", "decoder_model.onnx"]:
                candidate = self.model_path / name
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(f"No ONNX model found in directory: {self.model_path}")
        else:
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
    
    def load_model(self) -> None:
        """
        Load the ONNX model from disk.
        
        Supports:
        - Standard .onnx files
        - Models with external data files (.onnx.data)
        """
        model_file = self._resolve_model_path()
        print(f"Loading ONNX model from: {model_file}")
        
        try:
            # Load the model proto
            self.model_proto = onnx.load(str(model_file), load_external_data=True)
            print(f"  Model loaded successfully")
            print(f"  IR version: {self.model_proto.ir_version}")
            print(f"  Producer: {self.model_proto.producer_name}")
            
            # Check for external data
            data_file = Path(str(model_file) + ".data")
            if data_file.exists():
                print(f"  External data file detected: {data_file.name} ({data_file.stat().st_size / (1024*1024):.1f} MB)")
            
            if self.use_onnx_ir:
                # Convert to onnx_ir for advanced manipulation
                self.model = ir.from_proto(self.model_proto)
                self.graph = self.model.graph
                print(f"  Using onnx_ir for graph manipulation")
            else:
                # Use standard onnx
                self.model = self.model_proto
                self.graph = self.model_proto.graph
                print(f"  Using standard onnx for graph manipulation")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def analyze_model(self) -> Dict[str, Any]:
        """
        Analyze the loaded model structure.
        
        Returns:
            Dictionary containing model analysis data
        """
        if self.model is None:
            self.load_model()
        
        print("\nAnalyzing model structure...")
        
        # Get node count
        if self.use_onnx_ir:
            self._node_count = len(list(self.graph))
            self._initializer_count = len(list(self.graph.initializers))
        else:
            self._node_count = len(self.graph.node)
            self._initializer_count = len(self.graph.initializer)
        
        # Calculate total memory
        self._total_memory_mb = self._calculate_total_memory()
        
        # Detect layer boundaries (look for common patterns)
        self._detect_layer_boundaries()
        
        analysis = {
            "node_count": self._node_count,
            "initializer_count": self._initializer_count,
            "total_memory_mb": self._total_memory_mb,
            "layer_count": len(self._layer_boundaries.get("layers", [])),
            "input_count": len(self.graph.input) if not self.use_onnx_ir else len(list(self.graph.inputs)),
            "output_count": len(self.graph.output) if not self.use_onnx_ir else len(list(self.graph.outputs)),
        }
        
        print(f"  Nodes: {analysis['node_count']}")
        print(f"  Initializers (weights): {analysis['initializer_count']}")
        print(f"  Total memory: {analysis['total_memory_mb']:.2f} MB")
        print(f"  Detected layers: {analysis['layer_count']}")
        print(f"  Inputs: {analysis['input_count']}")
        print(f"  Outputs: {analysis['output_count']}")
        
        return analysis
    
    def _calculate_total_memory(self) -> float:
        """
        Calculate total memory usage of all weights/initializers.
        
        Adapted from builder.py _calculate_total_memory (lines 4211-4219)
        """
        total_size_mb = 0.0
        
        if self.use_onnx_ir:
            for initializer in self.graph.initializers:
                try:
                    if hasattr(initializer, 'const_value') and initializer.const_value is not None:
                        tensor = initializer.const_value
                        if hasattr(tensor, 'numpy'):
                            np_array = tensor.numpy()
                            size_mb = np_array.nbytes / (1024 * 1024)
                            total_size_mb += size_mb
                        elif hasattr(tensor, 'to_numpy'):
                            np_array = tensor.to_numpy()
                            size_mb = np_array.nbytes / (1024 * 1024)
                            total_size_mb += size_mb
                except Exception:
                    continue
        else:
            for initializer in self.graph.initializer:
                try:
                    np_array = numpy_helper.to_array(initializer)
                    size_mb = np_array.nbytes / (1024 * 1024)
                    total_size_mb += size_mb
                except Exception:
                    continue
        
        return total_size_mb
    
    def _calculate_chunk_memory(self, start_idx: int, end_idx: int) -> float:
        """
        Calculate memory usage of nodes from start_idx to end_idx.
        
        Adapted from builder.py _calculate_chunk_memory (lines 4221-4238)
        """
        chunk_memory = 0.0
        processed_initializers = set()
        
        if self.use_onnx_ir:
            nodes = list(self.graph)
            initializers = {init.name: init for init in self.graph.initializers if hasattr(init, 'name')}
            
            for i in range(start_idx, min(end_idx + 1, len(nodes))):
                node = nodes[i]
                for input_val in node.inputs:
                    input_name = input_val.name if hasattr(input_val, 'name') else str(input_val)
                    if input_name in initializers and input_name not in processed_initializers:
                        try:
                            init = initializers[input_name]
                            if hasattr(init, 'const_value') and init.const_value is not None:
                                tensor = init.const_value
                                if hasattr(tensor, 'numpy'):
                                    np_array = tensor.numpy()
                                elif hasattr(tensor, 'to_numpy'):
                                    np_array = tensor.to_numpy()
                                else:
                                    continue
                                size_mb = np_array.nbytes / (1024 * 1024)
                                chunk_memory += size_mb
                                processed_initializers.add(input_name)
                        except Exception:
                            continue
        else:
            nodes = list(self.graph.node)
            initializer_dict = {init.name: init for init in self.graph.initializer}
            
            for i in range(start_idx, min(end_idx + 1, len(nodes))):
                node = nodes[i]
                for input_name in node.input:
                    if input_name in initializer_dict and input_name not in processed_initializers:
                        try:
                            init = initializer_dict[input_name]
                            np_array = numpy_helper.to_array(init)
                            size_mb = np_array.nbytes / (1024 * 1024)
                            chunk_memory += size_mb
                            processed_initializers.add(input_name)
                        except Exception:
                            continue
        
        return chunk_memory
    
    def _detect_layer_boundaries(self) -> None:
        """
        Detect layer boundaries in the model by looking for common patterns.
        
        Looks for patterns like:
        - LayerNorm/RMSNorm operations
        - Sequential numbered layer names (e.g., layers.0, layers.1)
        """
        self._layer_boundaries = {
            "embedding": {"start_idx": None, "end_idx": None},
            "layers": [],
            "lm_head": {"start_idx": None, "end_idx": None},
        }
        
        current_layer = -1
        
        if self.use_onnx_ir:
            nodes = list(self.graph)
            for i, node in enumerate(nodes):
                node_name = node.name if hasattr(node, 'name') else ""
                
                # Detect layer patterns in node names
                if "layers." in node_name or "layer." in node_name:
                    import re
                    match = re.search(r'layers?\.(\d+)', node_name)
                    if match:
                        layer_num = int(match.group(1))
                        if layer_num != current_layer:
                            if current_layer >= 0 and self._layer_boundaries["layers"]:
                                self._layer_boundaries["layers"][-1]["end_idx"] = i - 1
                            current_layer = layer_num
                            self._layer_boundaries["layers"].append({
                                "layer_id": layer_num,
                                "start_idx": i,
                                "end_idx": None
                            })
                
                # Detect embedding layer
                if "embed" in node_name.lower() and self._layer_boundaries["embedding"]["start_idx"] is None:
                    self._layer_boundaries["embedding"]["start_idx"] = i
                
                # Detect lm_head
                if "lm_head" in node_name.lower():
                    if self._layer_boundaries["lm_head"]["start_idx"] is None:
                        self._layer_boundaries["lm_head"]["start_idx"] = i
                    self._layer_boundaries["lm_head"]["end_idx"] = i
            
            # Close the last layer
            if self._layer_boundaries["layers"]:
                self._layer_boundaries["layers"][-1]["end_idx"] = len(nodes) - 1
        else:
            # Standard onnx pattern detection
            for i, node in enumerate(self.graph.node):
                node_name = node.name
                
                if "layers." in node_name or "layer." in node_name:
                    import re
                    match = re.search(r'layers?\.(\d+)', node_name)
                    if match:
                        layer_num = int(match.group(1))
                        if layer_num != current_layer:
                            if current_layer >= 0 and self._layer_boundaries["layers"]:
                                self._layer_boundaries["layers"][-1]["end_idx"] = i - 1
                            current_layer = layer_num
                            self._layer_boundaries["layers"].append({
                                "layer_id": layer_num,
                                "start_idx": i,
                                "end_idx": None
                            })
            
            if self._layer_boundaries["layers"]:
                self._layer_boundaries["layers"][-1]["end_idx"] = len(self.graph.node) - 1
    
    def _get_weight_based_split_points(self) -> Tuple[List[int], List[float], float]:
        """
        Calculate split points based on weight memory distribution.
        
        Adapted from builder.py _get_weight_based_split_location (lines 4142-4180)
        and _get_weight_based_split_points_multi (lines 4182-4208)
        
        Returns:
            Tuple of (split_points, split_sizes_mb, total_memory_mb)
        """
        if self._total_memory_mb is None:
            self._total_memory_mb = self._calculate_total_memory()
        
        total_memory = self._total_memory_mb
        target_memory_per_chunk = total_memory / self.num_chunks
        
        split_points = []
        split_sizes = []
        current_memory = 0.0
        
        if self.use_onnx_ir:
            nodes = list(self.graph)
            initializers = {init.name: init for init in self.graph.initializers if hasattr(init, 'name')}
        else:
            nodes = list(self.graph.node)
            initializers = {init.name: init for init in self.graph.initializer}
        
        start_idx = 0
        processed_initializers = set()
        
        for chunk_idx in range(1, self.num_chunks):
            target_memory = chunk_idx * target_memory_per_chunk
            best_split_idx = None
            min_diff = float('inf')
            cumulative_memory = current_memory
            
            for i in range(start_idx, len(nodes)):
                node = nodes[i]
                
                # Get input names based on format
                if self.use_onnx_ir:
                    inputs = [inp.name if hasattr(inp, 'name') else str(inp) for inp in node.inputs]
                else:
                    inputs = list(node.input)
                
                # Add memory of weights consumed by this node
                for input_name in inputs:
                    if input_name in initializers and input_name not in processed_initializers:
                        try:
                            init = initializers[input_name]
                            if self.use_onnx_ir:
                                if hasattr(init, 'const_value') and init.const_value is not None:
                                    tensor = init.const_value
                                    if hasattr(tensor, 'numpy'):
                                        np_array = tensor.numpy()
                                    elif hasattr(tensor, 'to_numpy'):
                                        np_array = tensor.to_numpy()
                                    else:
                                        continue
                                    size_mb = np_array.nbytes / (1024 * 1024)
                                else:
                                    continue
                            else:
                                np_array = numpy_helper.to_array(init)
                                size_mb = np_array.nbytes / (1024 * 1024)
                            
                            cumulative_memory += size_mb
                            # Note: Don't add to processed here as we need to track per-chunk
                        except Exception:
                            continue
                
                # Check if we're close to target
                diff = abs(cumulative_memory - target_memory)
                if diff < min_diff:
                    min_diff = diff
                    best_split_idx = i
                
                # Stop if we've exceeded target by significant margin
                if cumulative_memory > target_memory * 1.1:
                    break
            
            if best_split_idx is not None:
                split_points.append(best_split_idx)
                chunk_memory = self._calculate_chunk_memory(start_idx, best_split_idx)
                split_sizes.append(chunk_memory)
                current_memory += chunk_memory
                start_idx = best_split_idx + 1
        
        # Final chunk size
        final_size = self._calculate_chunk_memory(start_idx, len(nodes) - 1)
        split_sizes.append(final_size)
        
        return split_points, split_sizes, total_memory
    
    def _get_chunk_boundary_tensors(self, start_idx: int, end_idx: int) -> Tuple[List[str], List[str]]:
        """
        Get input and output boundary tensors for a chunk.
        
        Adapted from builder.py _get_chunk_boundary_tensors (lines 4290-4306)
        
        Returns:
            Tuple of (input_tensors, output_tensors) - names of boundary tensors
        """
        boundary_inputs = set()
        boundary_outputs = set()
        
        if self.use_onnx_ir:
            nodes = list(self.graph)
            chunk_nodes = set(nodes[start_idx:end_idx + 1])
            chunk_output_names = set()
            
            # Collect all outputs produced by nodes in this chunk
            for node in chunk_nodes:
                for output in node.outputs:
                    out_name = output.name if hasattr(output, 'name') else str(output)
                    chunk_output_names.add(out_name)
            
            # Find inputs that come from outside this chunk
            for node in chunk_nodes:
                for input_val in node.inputs:
                    input_name = input_val.name if hasattr(input_val, 'name') else str(input_val)
                    if input_name and input_name not in chunk_output_names:
                        # Check if it's a graph input or comes from a previous chunk
                        is_initializer = False
                        for init in self.graph.initializers:
                            if hasattr(init, 'name') and init.name == input_name:
                                is_initializer = True
                                break
                        if not is_initializer:
                            boundary_inputs.add(input_name)
            
            # Find outputs that are used by nodes outside this chunk
            for node in chunk_nodes:
                for output in node.outputs:
                    out_name = output.name if hasattr(output, 'name') else str(output)
                    # Check if any node outside chunk uses this output
                    for other_node in nodes:
                        if other_node not in chunk_nodes:
                            for inp in other_node.inputs:
                                inp_name = inp.name if hasattr(inp, 'name') else str(inp)
                                if inp_name == out_name:
                                    boundary_outputs.add(out_name)
                                    break
        else:
            nodes = list(self.graph.node)
            chunk_nodes = nodes[start_idx:end_idx + 1]
            chunk_output_names = set()
            
            for node in chunk_nodes:
                for output in node.output:
                    chunk_output_names.add(output)
            
            initializer_names = {init.name for init in self.graph.initializer}
            graph_input_names = {inp.name for inp in self.graph.input}
            
            for node in chunk_nodes:
                for input_name in node.input:
                    if input_name and input_name not in chunk_output_names:
                        if input_name not in initializer_names:
                            boundary_inputs.add(input_name)
            
            for node in chunk_nodes:
                for output in node.output:
                    for other_node in nodes[end_idx + 1:]:
                        if output in other_node.input:
                            boundary_outputs.add(output)
                            break
        
        return list(boundary_inputs), list(boundary_outputs)
    
    def _create_chunk_model(
        self, 
        chunk_idx: int,
        start_idx: int, 
        end_idx: int
    ) -> onnx.ModelProto:
        """
        Create a single chunk model from the specified node range.
        
        Adapted from builder.py _make_chunk_model (lines 4308-4337)
        """
        print(f"  Creating chunk {chunk_idx}: nodes {start_idx} to {end_idx}")
        
        if self.use_onnx_ir:
            nodes = list(self.graph)
            chunk_nodes = nodes[start_idx:end_idx + 1]
            
            # Get boundary tensors
            boundary_inputs, boundary_outputs = self._get_chunk_boundary_tensors(start_idx, end_idx)
            
            # For first chunk, include original graph inputs
            if start_idx == 0:
                for graph_input in self.graph.inputs:
                    inp_name = graph_input.name if hasattr(graph_input, 'name') else str(graph_input)
                    if inp_name not in boundary_inputs:
                        boundary_inputs.insert(0, inp_name)
            
            # For last chunk, include original graph outputs
            if end_idx >= len(nodes) - 1:
                for graph_output in self.graph.outputs:
                    out_name = graph_output.name if hasattr(graph_output, 'name') else str(graph_output)
                    if out_name not in boundary_outputs:
                        boundary_outputs.append(out_name)
            
            # Create new graph with copied nodes
            # Use onnx directly for the final model creation
            chunk_proto = self._build_chunk_proto_onnx_ir(
                chunk_idx, chunk_nodes, boundary_inputs, boundary_outputs
            )
            
        else:
            chunk_nodes = list(self.graph.node)[start_idx:end_idx + 1]
            boundary_inputs, boundary_outputs = self._get_chunk_boundary_tensors(start_idx, end_idx)
            
            # For first chunk, include original graph inputs
            if start_idx == 0:
                for graph_input in self.graph.input:
                    if graph_input.name not in boundary_inputs:
                        boundary_inputs.insert(0, graph_input.name)
            
            # For last chunk, include original graph outputs  
            if end_idx >= len(self.graph.node) - 1:
                for graph_output in self.graph.output:
                    if graph_output.name not in boundary_outputs:
                        boundary_outputs.append(graph_output.name)
            
            chunk_proto = self._build_chunk_proto_standard(
                chunk_idx, chunk_nodes, boundary_inputs, boundary_outputs
            )
        
        return chunk_proto
    
    def _build_chunk_proto_onnx_ir(
        self,
        chunk_idx: int,
        chunk_nodes: List,
        boundary_inputs: List[str],
        boundary_outputs: List[str]
    ) -> onnx.ModelProto:
        """Build chunk ModelProto when using onnx_ir"""
        
        # Collect initializers used by chunk nodes
        used_initializers = set()
        for node in chunk_nodes:
            for inp in node.inputs:
                inp_name = inp.name if hasattr(inp, 'name') else str(inp)
                used_initializers.add(inp_name)
        
        # Get relevant initializers from original model
        chunk_initializers = []
        for init in self.graph.initializers:
            init_name = init.name if hasattr(init, 'name') else ""
            if init_name in used_initializers:
                # Convert to onnx TensorProto
                if hasattr(init, 'const_value') and init.const_value is not None:
                    try:
                        tensor = init.const_value
                        if hasattr(tensor, 'numpy'):
                            np_array = tensor.numpy()
                        elif hasattr(tensor, 'to_numpy'):
                            np_array = tensor.to_numpy()
                        else:
                            continue
                        tensor_proto = numpy_helper.from_array(np_array, name=init_name)
                        chunk_initializers.append(tensor_proto)
                    except Exception:
                        pass
        
        # Create input value infos
        chunk_inputs = []
        for inp_name in boundary_inputs:
            # Try to find type info from original graph inputs
            found_info = None
            for graph_input in self.graph.inputs:
                gi_name = graph_input.name if hasattr(graph_input, 'name') else str(graph_input)
                if gi_name == inp_name:
                    found_info = graph_input
                    break
            
            if found_info is not None:
                # Create value info from original
                vi = onnx.helper.make_tensor_value_info(
                    inp_name,
                    onnx.TensorProto.FLOAT,  # Default type
                    None  # Unknown shape
                )
            else:
                vi = onnx.helper.make_tensor_value_info(
                    inp_name,
                    onnx.TensorProto.FLOAT,
                    None
                )
            chunk_inputs.append(vi)
        
        # Create output value infos
        chunk_outputs = []
        for out_name in boundary_outputs:
            vi = onnx.helper.make_tensor_value_info(
                out_name,
                onnx.TensorProto.FLOAT,
                None
            )
            chunk_outputs.append(vi)
        
        # Convert nodes to onnx NodeProto format
        chunk_node_protos = []
        for node in chunk_nodes:
            # Get inputs and outputs as strings
            inputs = [inp.name if hasattr(inp, 'name') else str(inp) for inp in node.inputs]
            outputs = [out.name if hasattr(out, 'name') else str(out) for out in node.outputs]
            
            # Build node proto
            node_proto = onnx.helper.make_node(
                node.op_type,
                inputs=inputs,
                outputs=outputs,
                name=node.name if hasattr(node, 'name') else f"chunk_{chunk_idx}_node"
            )
            
            # Copy attributes
            if hasattr(node, 'attributes'):
                for attr_name, attr_value in node.attributes.items():
                    try:
                        if isinstance(attr_value, (int, np.integer)):
                            attr = onnx.helper.make_attribute(attr_name, int(attr_value))
                        elif isinstance(attr_value, (float, np.floating)):
                            attr = onnx.helper.make_attribute(attr_name, float(attr_value))
                        elif isinstance(attr_value, str):
                            attr = onnx.helper.make_attribute(attr_name, attr_value)
                        elif isinstance(attr_value, (list, tuple)):
                            if len(attr_value) > 0:
                                if isinstance(attr_value[0], (int, np.integer)):
                                    attr = onnx.helper.make_attribute(attr_name, [int(v) for v in attr_value])
                                elif isinstance(attr_value[0], (float, np.floating)):
                                    attr = onnx.helper.make_attribute(attr_name, [float(v) for v in attr_value])
                                else:
                                    continue
                            else:
                                continue
                        else:
                            continue
                        node_proto.attribute.append(attr)
                    except Exception:
                        pass
            
            chunk_node_protos.append(node_proto)
        
        # Create graph
        chunk_graph = onnx.helper.make_graph(
            chunk_node_protos,
            name=f"chunk_{chunk_idx}_graph",
            inputs=chunk_inputs,
            outputs=chunk_outputs,
            initializer=chunk_initializers
        )
        
        # Create model
        chunk_model = onnx.helper.make_model(
            chunk_graph,
            producer_name=f"local_model_splitter_chunk_{chunk_idx}",
            opset_imports=list(self.model_proto.opset_import)
        )
        
        return chunk_model
    
    def _build_chunk_proto_standard(
        self,
        chunk_idx: int,
        chunk_nodes: List,
        boundary_inputs: List[str],
        boundary_outputs: List[str]
    ) -> onnx.ModelProto:
        """Build chunk ModelProto using standard onnx format"""
        
        # Collect initializers used by chunk nodes
        used_initializers = set()
        for node in chunk_nodes:
            for inp in node.input:
                used_initializers.add(inp)
        
        # Get relevant initializers
        chunk_initializers = []
        for init in self.graph.initializer:
            if init.name in used_initializers:
                chunk_initializers.append(init)
        
        # Create input value infos
        chunk_inputs = []
        for inp_name in boundary_inputs:
            # Try to find type info from original
            found_info = None
            for graph_input in self.graph.input:
                if graph_input.name == inp_name:
                    found_info = graph_input
                    break
            
            if found_info is not None:
                chunk_inputs.append(found_info)
            else:
                vi = onnx.helper.make_tensor_value_info(
                    inp_name,
                    onnx.TensorProto.FLOAT,
                    None
                )
                chunk_inputs.append(vi)
        
        # Create output value infos
        chunk_outputs = []
        for out_name in boundary_outputs:
            found_info = None
            for graph_output in self.graph.output:
                if graph_output.name == out_name:
                    found_info = graph_output
                    break
            
            if found_info is not None:
                chunk_outputs.append(found_info)
            else:
                vi = onnx.helper.make_tensor_value_info(
                    out_name,
                    onnx.TensorProto.FLOAT,
                    None
                )
                chunk_outputs.append(vi)
        
        # Create graph
        chunk_graph = onnx.helper.make_graph(
            chunk_nodes,
            name=f"chunk_{chunk_idx}_graph",
            inputs=chunk_inputs,
            outputs=chunk_outputs,
            initializer=chunk_initializers
        )
        
        # Create model
        chunk_model = onnx.helper.make_model(
            chunk_graph,
            producer_name=f"local_model_splitter_chunk_{chunk_idx}",
            opset_imports=list(self.model_proto.opset_import)
        )
        
        return chunk_model
    
    def split_model(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        num_chunks: Optional[int] = None,
        strategy: Optional[str] = None,
        verify_chunks: bool = False
    ) -> SplitResult:
        """
        Main entry point for splitting the model.
        
        Args:
            output_dir: Directory to save chunk models (optional)
            num_chunks: Number of chunks to create (overrides config)
            strategy: Splitting strategy (overrides config)
            verify_chunks: Whether to verify chunk correctness
            
        Returns:
            SplitResult containing chunk models and metadata
        """
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Analyze model
        analysis = self.analyze_model()
        
        # Apply overrides
        if num_chunks is not None:
            self.num_chunks = num_chunks
        if strategy is not None:
            self.chunk_strategy = strategy
        
        print(f"\nSplitting model into {self.num_chunks} chunks using {self.chunk_strategy} strategy...")
        
        # Calculate split points
        if self.chunk_strategy == "weight_based":
            split_points, split_sizes, total_memory = self._get_weight_based_split_points()
        else:
            # Default to weight-based for now
            split_points, split_sizes, total_memory = self._get_weight_based_split_points()
        
        print(f"\nCalculated split points:")
        for i, (point, size) in enumerate(zip(split_points, split_sizes[:-1])):
            print(f"  Split {i+1}: after node {point}, chunk size: {size:.2f} MB")
        print(f"  Final chunk size: {split_sizes[-1]:.2f} MB")
        print(f"  Total memory: {total_memory:.2f} MB")
        
        # Create chunk models
        chunk_models = []
        chunk_info = []
        
        if self.use_onnx_ir:
            nodes = list(self.graph)
        else:
            nodes = list(self.graph.node)
        
        start_idx = 0
        for i, split_point in enumerate(split_points):
            end_idx = split_point
            chunk_model = self._create_chunk_model(i, start_idx, end_idx)
            chunk_models.append(chunk_model)
            
            boundary_inputs, boundary_outputs = self._get_chunk_boundary_tensors(start_idx, end_idx)
            info = ChunkInfo(
                chunk_idx=i,
                start_node_idx=start_idx,
                end_node_idx=end_idx,
                node_count=end_idx - start_idx + 1,
                memory_mb=split_sizes[i],
                boundary_inputs=boundary_inputs,
                boundary_outputs=boundary_outputs
            )
            chunk_info.append(info)
            
            start_idx = split_point + 1
        
        # Create final chunk
        end_idx = len(nodes) - 1
        chunk_model = self._create_chunk_model(len(split_points), start_idx, end_idx)
        chunk_models.append(chunk_model)
        
        boundary_inputs, boundary_outputs = self._get_chunk_boundary_tensors(start_idx, end_idx)
        info = ChunkInfo(
            chunk_idx=len(split_points),
            start_node_idx=start_idx,
            end_node_idx=end_idx,
            node_count=end_idx - start_idx + 1,
            memory_mb=split_sizes[-1],
            boundary_inputs=boundary_inputs,
            boundary_outputs=boundary_outputs
        )
        chunk_info.append(info)
        
        print(f"\nCreated {len(chunk_models)} chunk models")
        
        # Create result
        result = SplitResult(
            chunk_models=chunk_models,
            chunk_info=chunk_info,
            total_memory_mb=total_memory,
            split_points=split_points,
            metadata={
                "strategy": self.chunk_strategy,
                "distribution": self.chunk_distribution,
                "num_chunks": self.num_chunks,
                "original_node_count": len(nodes),
                "original_model_path": str(self.model_path),
                "split_sizes_mb": split_sizes,
            }
        )
        
        # Save if output directory provided
        if output_dir is not None:
            self.save_chunks(result, output_dir)
        
        return result
    
    def save_chunks(
        self,
        result: SplitResult,
        output_dir: Union[str, Path]
    ) -> List[str]:
        """
        Save chunk models to disk.
        
        Args:
            result: SplitResult from split_model
            output_dir: Directory to save chunks
            
        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving chunks to: {output_dir}")
        
        saved_paths = []
        for i, chunk_model in enumerate(result.chunk_models):
            chunk_path = output_dir / f"chunk_{i}.onnx"
            chunk_data_path = output_dir / f"chunk_{i}.onnx.data"
            
            print(f"  Saving chunk {i} to {chunk_path}")
            
            try:
                # Save with external data for large models
                onnx.save(
                    chunk_model,
                    str(chunk_path),
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=f"chunk_{i}.onnx.data",
                    size_threshold=1024
                )
            except Exception as e:
                # Fallback: save without external data
                print(f"    Warning: Saving without external data ({e})")
                onnx.save(chunk_model, str(chunk_path))
            
            saved_paths.append(str(chunk_path))
        
        # Save metadata
        metadata_path = output_dir / "chunking_metadata.json"
        metadata = {
            **result.metadata,
            "chunk_paths": saved_paths,
            "chunk_info": [
                {
                    "chunk_idx": info.chunk_idx,
                    "start_node_idx": info.start_node_idx,
                    "end_node_idx": info.end_node_idx,
                    "node_count": info.node_count,
                    "memory_mb": info.memory_mb,
                    "boundary_inputs": info.boundary_inputs,
                    "boundary_outputs": info.boundary_outputs
                }
                for info in result.chunk_info
            ]
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved metadata to {metadata_path}")
        
        return saved_paths

    # ==========================================================================
    # Layer-Based Splitting (adapted from grok_hf_split.py)
    # ==========================================================================
    
    def get_model_prefixes(self, model_type: Optional[str] = None) -> Tuple[str, str, str, str]:
        """
        Return weight/node name prefixes for different model architectures.
        
        Adapted from grok_hf_split.py get_model_prefixes (lines 47-62)
        
        Returns:
            Tuple of (embed_prefix, layer_prefix, norm_prefix, lm_prefix)
        """
        # Try to detect model type from node names if not provided
        if model_type is None:
            model_type = self._detect_model_type_from_nodes()
        
        if model_type in ["llama", "mistral", "phi", "gemma", "qwen2", "qwen3", "qwen"]:
            embed_prefix = "model/embed"
            layer_prefix = "model/layers."
            norm_prefix = "model/norm"
            lm_prefix = "lm_head"
        elif model_type == "gpt2":
            embed_prefix = "transformer/wte"
            layer_prefix = "transformer/h."
            norm_prefix = "transformer/ln_f"
            lm_prefix = "lm_head"
        else:
            # Default to LLaMA-style prefixes (most common)
            embed_prefix = "model/embed"
            layer_prefix = "model/layers."
            norm_prefix = "model/norm"
            lm_prefix = "lm_head"
        
        return embed_prefix, layer_prefix, norm_prefix, lm_prefix
    
    def _detect_model_type_from_nodes(self) -> str:
        """
        Detect model architecture from node name patterns.
        
        Adapted from grok_hf_split.py detect_architecture_from_weights (lines 65-84)
        """
        if self.model is None:
            self.load_model()
        
        sample_names = []
        if self.use_onnx_ir:
            nodes = list(self.graph)[:50]
            for node in nodes:
                if hasattr(node, 'name') and node.name:
                    sample_names.append(node.name)
        else:
            for node in list(self.graph.node)[:50]:
                sample_names.append(node.name)
        
        # Check for common architecture patterns
        if any("/model/layers." in n or "model.layers." in n for n in sample_names):
            # Check for Qwen3 specific patterns
            if any("q_norm" in n or "k_norm" in n for n in sample_names):
                return "qwen3"
            return "llama"  # Generic LLaMA-style
        elif any("transformer/h." in n or "transformer.h." in n for n in sample_names):
            return "gpt2"
        
        return "llama"  # Default to LLaMA-style
    
    def _get_layer_components(self) -> Dict[str, Dict[str, Any]]:
        """
        Extract layer components from the ONNX graph.
        
        Returns a dictionary with:
        - 'embedding': nodes for embedding layer
        - 'layers': dict of layer_id -> nodes for each layer
        - 'norm': nodes for final normalization
        - 'lm_head': nodes for language model head
        """
        embed_prefix, layer_prefix, norm_prefix, lm_prefix = self.get_model_prefixes()
        
        components = {
            "embedding": {"node_indices": [], "nodes": []},
            "layers": {},
            "norm": {"node_indices": [], "nodes": []},
            "lm_head": {"node_indices": [], "nodes": []},
            "other": {"node_indices": [], "nodes": []},  # Nodes not matching any pattern
        }
        
        if self.use_onnx_ir:
            nodes = list(self.graph)
        else:
            nodes = list(self.graph.node)
        
        import re
        
        for i, node in enumerate(nodes):
            if self.use_onnx_ir:
                node_name = node.name if hasattr(node, 'name') else ""
            else:
                node_name = node.name
            
            node_name_lower = node_name.lower()
            
            # Check for embedding
            if embed_prefix.lower() in node_name_lower or "embed_tokens" in node_name_lower:
                components["embedding"]["node_indices"].append(i)
                components["embedding"]["nodes"].append(node)
                continue
            
            # Check for layers (pattern: layers.N or layers.N.)
            layer_match = re.search(r'layers[./](\d+)', node_name_lower)
            if layer_match:
                layer_id = int(layer_match.group(1))
                if layer_id not in components["layers"]:
                    components["layers"][layer_id] = {"node_indices": [], "nodes": []}
                components["layers"][layer_id]["node_indices"].append(i)
                components["layers"][layer_id]["nodes"].append(node)
                continue
            
            # Check for norm (final layer norm)
            if norm_prefix.lower() in node_name_lower and "layers" not in node_name_lower:
                components["norm"]["node_indices"].append(i)
                components["norm"]["nodes"].append(node)
                continue
            
            # Check for lm_head
            if lm_prefix.lower() in node_name_lower:
                components["lm_head"]["node_indices"].append(i)
                components["lm_head"]["nodes"].append(node)
                continue
            
            # Everything else
            components["other"]["node_indices"].append(i)
            components["other"]["nodes"].append(node)
        
        return components
    
    def _get_layer_based_split_points(self, layers_per_chunk: int = 1) -> Tuple[List[int], List[float], float]:
        """
        Calculate split points based on layer boundaries.
        
        Args:
            layers_per_chunk: Number of layers to include per chunk (1 = one layer per file)
            
        Returns:
            Tuple of (split_points, split_sizes_mb, total_memory_mb)
        """
        components = self._get_layer_components()
        
        if self._total_memory_mb is None:
            self._total_memory_mb = self._calculate_total_memory()
        
        split_points = []
        split_sizes = []
        
        layer_ids = sorted(components["layers"].keys())
        num_layers = len(layer_ids)
        
        if num_layers == 0:
            print("Warning: No layers detected, falling back to weight-based splitting")
            return self._get_weight_based_split_points()
        
        print(f"  Detected {num_layers} transformer layers")
        
        # Calculate split points at layer boundaries
        # Grouping: [embedding + layers 0 to N-1], [layers N to 2N-1], ..., [layers M to end + norm + lm_head]
        
        current_chunk_layers = []
        chunk_start_idx = 0
        
        # Include embedding in first chunk
        if components["embedding"]["node_indices"]:
            chunk_start_idx = min(components["embedding"]["node_indices"])
        else:
            # Start from first layer
            if components["layers"] and layer_ids:
                first_layer = components["layers"][layer_ids[0]]
                if first_layer["node_indices"]:
                    chunk_start_idx = min(first_layer["node_indices"])
        
        for i, layer_id in enumerate(layer_ids):
            current_chunk_layers.append(layer_id)
            
            # Check if we should create a split point
            if len(current_chunk_layers) >= layers_per_chunk and (i + 1) < num_layers:
                # Split after this layer
                layer_data = components["layers"][layer_id]
                if layer_data["node_indices"]:
                    split_idx = max(layer_data["node_indices"])
                    split_points.append(split_idx)
                    
                    # Calculate memory for this chunk
                    chunk_memory = self._calculate_chunk_memory(chunk_start_idx, split_idx)
                    split_sizes.append(chunk_memory)
                    
                    # Reset for next chunk
                    chunk_start_idx = split_idx + 1
                    current_chunk_layers = []
        
        # Calculate final chunk size (remaining layers + norm + lm_head)
        if self.use_onnx_ir:
            total_nodes = len(list(self.graph))
        else:
            total_nodes = len(list(self.graph.node))
        
        final_size = self._calculate_chunk_memory(chunk_start_idx, total_nodes - 1)
        split_sizes.append(final_size)
        
        return split_points, split_sizes, self._total_memory_mb
    
    def split_by_layers(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        layers_per_chunk: int = 1,
        verify_chunks: bool = False
    ) -> SplitResult:
        """
        Split model by layer boundaries.
        
        This method creates ONNX chunks that respect transformer layer boundaries,
        similar to grok_hf_split.py's per-layer safetensors output.
        
        Args:
            output_dir: Directory to save chunk models (optional)
            layers_per_chunk: Number of layers per chunk (1 = one layer per file)
            verify_chunks: Whether to verify chunk correctness
            
        Returns:
            SplitResult containing chunk models and metadata
        """
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Analyze model
        analysis = self.analyze_model()
        
        # Get layer components
        components = self._get_layer_components()
        layer_ids = sorted(components["layers"].keys())
        num_layers = len(layer_ids)
        
        if num_layers == 0:
            print("Warning: No layers detected in model, falling back to weight-based splitting")
            return self.split_model(output_dir=output_dir)
        
        # Calculate number of chunks based on layers_per_chunk
        num_chunks = (num_layers + layers_per_chunk - 1) // layers_per_chunk
        
        # Add chunks for embedding (if separate) and lm_head (if separate)
        has_embedding = len(components["embedding"]["node_indices"]) > 0
        has_lm_head = len(components["lm_head"]["node_indices"]) > 0
        
        print(f"\nSplitting model by layers:")
        print(f"  Total layers: {num_layers}")
        print(f"  Layers per chunk: {layers_per_chunk}")
        print(f"  Expected chunks: ~{num_chunks}")
        print(f"  Has separate embedding: {has_embedding}")
        print(f"  Has separate lm_head: {has_lm_head}")
        
        # Calculate split points
        split_points, split_sizes, total_memory = self._get_layer_based_split_points(layers_per_chunk)
        
        print(f"\nCalculated {len(split_points)} split points:")
        for i, (point, size) in enumerate(zip(split_points, split_sizes[:-1])):
            print(f"  Split {i+1}: after node {point}, chunk size: {size:.2f} MB")
        print(f"  Final chunk size: {split_sizes[-1]:.2f} MB")
        print(f"  Total memory: {total_memory:.2f} MB")
        
        # Create chunk models
        chunk_models = []
        chunk_info = []
        
        if self.use_onnx_ir:
            nodes = list(self.graph)
        else:
            nodes = list(self.graph.node)
        
        start_idx = 0
        chunk_layer_ranges = []
        
        for i, split_point in enumerate(split_points):
            end_idx = split_point
            
            # Determine which layers are in this chunk
            layers_in_chunk = []
            for layer_id in layer_ids:
                layer_data = components["layers"][layer_id]
                if layer_data["node_indices"]:
                    min_idx = min(layer_data["node_indices"])
                    max_idx = max(layer_data["node_indices"])
                    if min_idx >= start_idx and max_idx <= end_idx:
                        layers_in_chunk.append(layer_id)
            
            chunk_layer_ranges.append(layers_in_chunk)
            
            chunk_model = self._create_chunk_model(i, start_idx, end_idx)
            chunk_models.append(chunk_model)
            
            boundary_inputs, boundary_outputs = self._get_chunk_boundary_tensors(start_idx, end_idx)
            info = ChunkInfo(
                chunk_idx=i,
                start_node_idx=start_idx,
                end_node_idx=end_idx,
                node_count=end_idx - start_idx + 1,
                memory_mb=split_sizes[i],
                boundary_inputs=boundary_inputs,
                boundary_outputs=boundary_outputs
            )
            chunk_info.append(info)
            
            start_idx = split_point + 1
        
        # Create final chunk
        end_idx = len(nodes) - 1
        
        # Determine layers in final chunk
        layers_in_chunk = []
        for layer_id in layer_ids:
            layer_data = components["layers"][layer_id]
            if layer_data["node_indices"]:
                min_idx = min(layer_data["node_indices"])
                max_idx = max(layer_data["node_indices"])
                if min_idx >= start_idx and max_idx <= end_idx:
                    layers_in_chunk.append(layer_id)
        chunk_layer_ranges.append(layers_in_chunk)
        
        chunk_model = self._create_chunk_model(len(split_points), start_idx, end_idx)
        chunk_models.append(chunk_model)
        
        boundary_inputs, boundary_outputs = self._get_chunk_boundary_tensors(start_idx, end_idx)
        info = ChunkInfo(
            chunk_idx=len(split_points),
            start_node_idx=start_idx,
            end_node_idx=end_idx,
            node_count=end_idx - start_idx + 1,
            memory_mb=split_sizes[-1],
            boundary_inputs=boundary_inputs,
            boundary_outputs=boundary_outputs
        )
        chunk_info.append(info)
        
        print(f"\nCreated {len(chunk_models)} layer-based chunk models:")
        for i, layers in enumerate(chunk_layer_ranges):
            if layers:
                print(f"  Chunk {i}: layers {min(layers)}-{max(layers)} ({len(layers)} layers)")
            else:
                print(f"  Chunk {i}: embedding/other components")
        
        # Create result
        result = SplitResult(
            chunk_models=chunk_models,
            chunk_info=chunk_info,
            total_memory_mb=total_memory,
            split_points=split_points,
            metadata={
                "strategy": "layer_based",
                "distribution": f"{layers_per_chunk}_layers_per_chunk",
                "num_chunks": len(chunk_models),
                "num_layers": num_layers,
                "layers_per_chunk": layers_per_chunk,
                "chunk_layer_ranges": [list(lr) for lr in chunk_layer_ranges],
                "original_node_count": len(nodes),
                "original_model_path": str(self.model_path),
                "split_sizes_mb": split_sizes,
            }
        )
        
        # Save if output directory provided
        if output_dir is not None:
            self.save_layer_chunks(result, output_dir, chunk_layer_ranges)
        
        return result
    
    def save_layer_chunks(
        self,
        result: SplitResult,
        output_dir: Union[str, Path],
        chunk_layer_ranges: List[List[int]]
    ) -> List[str]:
        """
        Save layer-based chunk models with descriptive names.
        
        Names follow the pattern from grok_hf_split.py:
        - chunk_embedding.onnx (if first chunk is embedding)
        - chunk_layers_0_7.onnx (layers 0-7)
        - chunk_layers_8_15.onnx (layers 8-15)
        - chunk_layers_16_27_lm_head.onnx (final layers + lm_head)
        
        Args:
            result: SplitResult from split_by_layers
            output_dir: Directory to save chunks
            chunk_layer_ranges: List of layer IDs per chunk
            
        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving layer-based chunks to: {output_dir}")
        
        saved_paths = []
        num_chunks = len(result.chunk_models)
        
        for i, chunk_model in enumerate(result.chunk_models):
            layers = chunk_layer_ranges[i] if i < len(chunk_layer_ranges) else []
            
            # Generate descriptive filename
            if i == 0 and not layers:
                # First chunk without layers = embedding
                chunk_name = "chunk_embedding"
            elif i == num_chunks - 1:
                # Last chunk includes lm_head
                if layers:
                    chunk_name = f"chunk_layers_{min(layers)}_{max(layers)}_lm_head"
                else:
                    chunk_name = "chunk_lm_head"
            else:
                if layers:
                    chunk_name = f"chunk_layers_{min(layers)}_{max(layers)}"
                else:
                    chunk_name = f"chunk_{i}"
            
            chunk_path = output_dir / f"{chunk_name}.onnx"
            chunk_data_path = output_dir / f"{chunk_name}.onnx.data"
            
            print(f"  Saving {chunk_name} to {chunk_path}")
            
            try:
                # Save with external data for large models
                onnx.save(
                    chunk_model,
                    str(chunk_path),
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=f"{chunk_name}.onnx.data",
                    size_threshold=1024
                )
            except Exception as e:
                # Fallback: save without external data
                print(f"    Warning: Saving without external data ({e})")
                onnx.save(chunk_model, str(chunk_path))
            
            saved_paths.append(str(chunk_path))
        
        # Save metadata
        metadata_path = output_dir / "chunking_metadata.json"
        metadata = {
            **result.metadata,
            "chunk_paths": saved_paths,
            "chunk_info": [
                {
                    "chunk_idx": info.chunk_idx,
                    "start_node_idx": info.start_node_idx,
                    "end_node_idx": info.end_node_idx,
                    "node_count": info.node_count,
                    "memory_mb": info.memory_mb,
                    "layers": chunk_layer_ranges[info.chunk_idx] if info.chunk_idx < len(chunk_layer_ranges) else [],
                    "boundary_inputs": info.boundary_inputs,
                    "boundary_outputs": info.boundary_outputs
                }
                for info in result.chunk_info
            ]
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved metadata to {metadata_path}")
        
        return saved_paths


def split_local_model(
    model_path: str,
    output_dir: str,
    num_chunks: int = 2,
    strategy: str = "weight_based"
) -> SplitResult:
    """
    Convenience function to split a local ONNX model.
    
    Args:
        model_path: Path to ONNX model file or directory
        output_dir: Directory to save chunk models
        num_chunks: Number of chunks to create
        strategy: Splitting strategy ("weight_based" or "layer_based")
        
    Returns:
        SplitResult containing chunk models and metadata
    """
    splitter = LocalONNXModelSplitter(model_path)
    return splitter.split_model(
        output_dir=output_dir,
        num_chunks=num_chunks,
        strategy=strategy
    )


def split_local_model_by_layers(
    model_path: str,
    output_dir: str,
    layers_per_chunk: int = 1
) -> SplitResult:
    """
    Convenience function to split a local ONNX model by layer boundaries.
    
    This function creates ONNX chunks similar to grok_hf_split.py's per-layer
    safetensors output, but for ONNX format.
    
    Args:
        model_path: Path to ONNX model file or directory
        output_dir: Directory to save chunk models
        layers_per_chunk: Number of layers per chunk (1 = one layer per file)
        
    Returns:
        SplitResult containing chunk models and metadata
    """
    splitter = LocalONNXModelSplitter(model_path)
    return splitter.split_by_layers(
        output_dir=output_dir,
        layers_per_chunk=layers_per_chunk
    )


if __name__ == "__main__":
    # Example usage - this will be invoked by split_local_model.py CLI
    print("LocalONNXModelSplitter module loaded successfully")
    print("Use split_local_model.py for command-line usage")

