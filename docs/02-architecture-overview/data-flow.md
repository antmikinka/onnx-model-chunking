# Data Flow Documentation

**Outcome Statement:** After reading this section, you will understand exactly how data flows through the system during both splitting and inference operations.

---

## Part 1: Splitting Data Flow

### Phase 1: Model Loading

**Outcome:** Understand how source models are loaded and prepared for splitting.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SOURCE MODEL LOADING FLOW                            │
└─────────────────────────────────────────────────────────────────────────┘

Step 1: Identify Source Type
│
├── HuggingFace Hub ID (e.g., "Qwen/Qwen2.5-0.5B")
│   └── Download using transformers.AutoModel
│
├── Local Safetensors Directory
│   └── Load using safetensors.numpy.load_file()
│
└── Local ONNX File
    └── Load using onnx.load_model()

Step 2: Load Model Components
│
├── Weights (.safetensors or .bin)
│   └── Load into memory as numpy arrays
│
├── Graph Structure (ONNX only)
│   └── Parse ONNX protobuf to onnx.ModelProto
│
└── Configuration (config.json)
    └── Parse to extract: hidden_size, num_layers, vocab_size, etc.

Step 3: Create Internal Representation
│
└── Model Object
    ├── onnx_model: onnx.ModelProto
    ├── weights: Dict[str, numpy.ndarray]
    ├── config: Dict[str, Any]
    └── metadata: ModelMetadata
```

**Data Structures:**
```python
@dataclass
class ModelSource:
    source_type: str  # "hf_hub" | "local_safetensors" | "local_onnx"
    path: str
    config: Dict[str, Any]

@dataclass
class LoadedModel:
    model: onnx.ModelProto
    weights: Dict[str, np.ndarray]
    config: Dict[str, Any]
    metadata: ModelMetadata
```

---

### Phase 2: Architecture Detection

**Outcome:** Understand how the system identifies model architecture.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  ARCHITECTURE DETECTION FLOW                            │
└─────────────────────────────────────────────────────────────────────────┘

Input: onnx.ModelProto
│
▼
Scan Node Names for Patterns
│
├── Search for "model/embed" or "transformer/wte"
│   └── Found → Architecture has separate embedding layer
│
├── Search for "model/layers." or "transformer/h."
│   └── Pattern match → Extract layer count (e.g., layers.0 to layers.27 = 28 layers)
│
├── Search for "model/norm" or "transformer/ln_f"
│   └── Found → Architecture has final normalization
│
└── Search for "lm_head" or "head"
    └── Found → Architecture has separate LM head

▼
Classify Architecture
│
├── LlamaFamily: "model.layers.", "model.embed_tokens", "model.norm"
├── QwenFamily: "model.layers.", "model.embed_tokens", "lm_head"
├── GPT2Family: "transformer.h.", "transformer.wte", "transformer.ln_f"
└── Unknown → Fall back to weight-based splitting

▼
Output: ArchitectureInfo
│
└── ArchitectureInfo
    ├── arch_type: ArchitectureType
    ├── num_layers: int
    ├── embedding_name: str
    ├── layer_prefix: str
    ├── norm_name: str
    └── lm_head_name: str
```

**Pattern Matching Table:**

| Architecture | Embedding Pattern | Layer Pattern | Norm Pattern | LM Head |
|--------------|-------------------|---------------|--------------|---------|
| Llama/Mistral | `model.embed_tokens` | `model.layers.\d+` | `model.norm` | `lm_head` |
| Qwen/Qwen2 | `model.embed_tokens` | `model.layers.\d+` | `model.norm` | `lm_head` |
| GPT-2 | `transformer.wte` | `transformer.h.\d+` | `transformer.ln_f` | N/A (uses embeddings) |
| Phi | `model.embed_tokens` | `model.layers.\d+` | `model.final_layernorm` | `lm_head` |

---

### Phase 3: Component Extraction

**Outcome:** Understand how semantic components are identified and extracted.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   COMPONENT EXTRACTION FLOW                             │
└─────────────────────────────────────────────────────────────────────────┘

Input: onnx.ModelProto + ArchitectureInfo
│
▼
Step 1: Identify Nodes per Component
│
├── Embedding Component
│   └── Collect all nodes dominated by embedding input
│
├── Layer Components (for each layer_idx)
│   └── Collect nodes matching layer prefix + index
│
├── Normalization Component
│   └── Collect nodes dominated by final layer output
│
└── LM Head Component
    └── Collect nodes producing logits output

▼
Step 2: Identify Boundary Tensors
│
For each component:
│
├── Input Tensors
│   ├── External inputs (from model input list)
│   └── Cross-component dependencies (from previous component's outputs)
│
├── Output Tensors
│   ├── External outputs (from model output list)
│   └── Cross-component dependencies (needed by next component)
│
└── Pass-Through Tensors
    └── Tensors that cross component boundary unchanged (e.g., attention_mask)

▼
Step 3: Create ComponentInfo
│
└── ComponentInfo
    ├── component_type: str  # "embedding", "layer_0", "norm", "lm_head"
    ├── layer_id: Optional[int]
    ├── node_indices: List[int]
    ├── node_names: List[str]
    ├── initializer_names: Set[str]
    ├── input_names: List[str]
    └── output_names: List[str]
```

**Boundary Tensor Example (Qwen2):**

```
Embedding → Layer 0 Boundary:
├── Inputs to Layer 0:
│   ├── hidden_states: [batch, seq, hidden_dim] (from Embedding)
│   ├── attention_mask: [batch, seq] (pass-through from input)
│   └── position_ids: [batch, seq] (pass-through from input)
│
└── Outputs from Layer 0:
    ├── hidden_states: [batch, seq, hidden_dim] (to Layer 1)
    ├── present_key: [batch, kv_heads, seq, head_dim] (to KV cache)
    └── present_value: [batch, kv_heads, seq, head_dim] (to KV cache)
```

---

### Phase 4: Subgraph Creation

**Outcome:** Understand how standalone ONNX components are created.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SUBGRAPH CREATION FLOW                               │
└─────────────────────────────────────────────────────────────────────────┘

For each ComponentInfo:
│
▼
Step 1: Create New Graph
│
├── Create empty onnx.GraphProto
│
├── Copy nodes from ComponentInfo.node_indices
│   └── Include all operators, attributes, domains
│
└── Set graph name = component_type

▼
Step 2: Set Initializers (Weights)
│
├── For each initializer_name in ComponentInfo:
│   ├── Copy tensor from original model.initializers
│   └── Add to new graph.initializers
│
└── Handle External Data
    └── Preserve external_data_path if model uses separate weight files

▼
Step 3: Set Inputs
│
├── Create ValueInfoProto for each input_name
│   ├── Set name, type (tensor_type), dtype, shape
│   └── Handle dynamic axes (batch, sequence_length)
│
└── Add to graph.input

▼
Step 4: Set Outputs
│
├── Create ValueInfoProto for each output_name
│   ├── Set name, type, dtype, shape
│   └── Ensure explicit typing (no ORT auto-cast)
│
└── Add to graph.output

▼
Step 5: Create Component Model
│
├── Create new onnx.ModelProto
│   ├── Set ir_version
│   ├── Set opset_import (inherit from original)
│   ├── Set graph = new graph
│   └── Copy metadata (producer_name, domain)
│
└── Validate Model
    ├── onnx.checker.check_model()
    └── Verify all inputs/outputs are properly typed

▼
Step 6: Save to Disk
│
└── onnx.save(component_model, output_dir / f"{component_type}.onnx")
```

**Critical Implementation Detail: Explicit Type Annotations**

```python
# Each boundary tensor must have explicit type annotation
# This prevents ONNX Runtime auto-cast errors

def create_boundary_tensor(name, dtype, shape):
    return onnx.ValueInfoProto(
        name=name,
        type=onnx.TensorTypeProto(
            elem_type=dtype,  # e.g., TensorProto.FLOAT
            shape=onnx.TensorShapeProto(
                dim=[onnx.TensorShapeProto.Dimension(dim_value=s) for s in shape]
            )
        )
    )
```

---

## Part 2: Inference Data Flow

### Prefill Phase Data Flow

**Outcome:** Trace data flow during prompt processing.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PREFILL PHASE DATA FLOW                              │
│                  (Process Full Prompt in One Pass)                      │
└─────────────────────────────────────────────────────────────────────────┘

Input: "The future of AI is"
│
▼
┌────────────────────────────────────────────────────────────────────────┐
│ Step 1: Tokenization                                                   │
│                                                                        │
│ tokenizer.encode("The future of AI is")                               │
│ → input_ids: [1, 5823, 29871, 3975, 3186]  # shape: [1, 5]           │
│ attention_mask: [1, 1, 1, 1, 1]                                        │
│ position_ids: [0, 1, 2, 3, 4]                                          │
└────────────────────────────────────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────────────────────┐
│ Step 2: Embedding Lookup                                               │
│                                                                        │
│ Load embeddings.onnx → Run → Unload                                   │
│                                                                        │
│ Inputs:                                                                │
│   input_ids: int64[1, 5]                                              │
│                                                                        │
│ Outputs:                                                               │
│   hidden_states: float32[1, 5, 896]  # 896 = hidden_dim              │
│                                                                        │
│ Store in registry:                                                     │
│   registry["hidden_states"] = output["hidden_states"]                 │
│   registry["attention_mask"] = attention_mask                         │
│   registry["position_ids"] = position_ids                             │
└────────────────────────────────────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────────────────────┐
│ Step 3: Layer-by-Layer Processing (Layers 0-27)                        │
│                                                                        │
│ For layer_idx in range(28):                                           │
│   │                                                                    │
│   Load layer_{layer_idx}.onnx                                         │
│   │                                                                    │
│   Inputs:                                                              │
│     hidden_states: float32[1, 5, 896]  # from registry                │
│     attention_mask: int64[1, 5]        # from registry                │
│     position_ids: int64[1, 5]          # from registry                │
│     past_key: float32[...]           # from KV cache (empty on first) │
│     past_value: float32[...]           # from KV cache (empty on first)│
│   │                                                                    │
│   Run layer                                                           │
│   │                                                                    │
│   Outputs:                                                             │
│     hidden_states: float32[1, 5, 896]  # updated hidden states        │
│     present_key: float32[1, 8, 5, 128]  # new KV cache               │
│     present_value: float32[1, 8, 5, 128]  # new KV cache             │
│   │                                                                    │
│   Update registry:                                                     │
│     registry["hidden_states"] = new_hidden_states                     │
│   │                                                                    │
│   Update KV cache:                                                     │
│     kv_cache.update(layer_idx, present_key, present_value)            │
│   │                                                                    │
│   Unload layer                                                        │
│                                                                        │
│ After all layers: hidden_states contains encoded representation       │
└────────────────────────────────────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────────────────────┐
│ Step 4: Final Normalization                                            │
│                                                                        │
│ Load norm.onnx → Run → Unload                                         │
│                                                                        │
│ Inputs:                                                                │
│   hidden_states: float32[1, 5, 896]  # from last layer                │
│                                                                        │
│ Outputs:                                                               │
│   normalized_hidden: float32[1, 5, 896]  # RMSNorm applied            │
│                                                                        │
│ Store in registry:                                                     │
│   registry["normalized"] = output["normalized_hidden"]                │
└────────────────────────────────────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────────────────────┐
│ Step 5: LM Head (Logits Generation)                                    │
│                                                                        │
│ Load lm_head.onnx → Run → Unload                                      │
│                                                                        │
│ Inputs:                                                                │
│   normalized: float32[1, 5, 896]  # from norm                         │
│                                                                        │
│ Outputs:                                                               │
│   logits: float32[1, 5, 151936]  # 151936 = vocab_size               │
│                                                                        │
│ Select last position:                                                  │
│   next_token_logits = logits[0, -1, :]  # shape: [151936]            │
└────────────────────────────────────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────────────────────┐
│ Step 6: Token Selection                                                │
│                                                                        │
│ next_token_id = argmax(next_token_logits)                             │
│                                                                        │
│ Generated token: 28493 → "bright"                                     │
│                                                                        │
│ Store for decode phase:                                                │
│   generated_tokens = [28493]                                          │
└────────────────────────────────────────────────────────────────────────┘
```

---

### Decode Phase Data Flow

**Outcome:** Trace data flow during token generation.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DECODE PHASE DATA FLOW                               │
│                (Generate Tokens One-by-One)                             │
└─────────────────────────────────────────────────────────────────────────┘

For token_idx in range(max_new_tokens):
│
▼
┌────────────────────────────────────────────────────────────────────────┐
│ Iteration N: Generate Next Token                                       │
│                                                                        │
│ Current state:                                                         │
│   generated_tokens: [28493, 3186, ...]  # tokens so far              │
│   kv_cache: populated for all layers                                  │
│                                                                        │
│ Step 1: Prepare Single-Token Input                                     │
│   │                                                                    │
│   input_ids: int64[1, 1]  # Just the last generated token             │
│   position_ids: int64[1, 1]  # position = total_tokens_so_far         │
│   attention_mask: int64[1, total_length]  # All 1s                    │
│   │                                                                    │
│ Step 2: Run Components with KV Cache                                   │
│   │                                                                    │
│   embeddings: input_ids → hidden_states[1, 1, 896]                    │
│   │                                                                    │
│   For each layer:                                                     │
│     │                                                                  │
│     Inputs:                                                            │
│       hidden_states: [1, 1, 896]  # single token                      │
│       past_key: [1, 8, total_seq, 128]  # cached KV                   │
│       past_value: [1, 8, total_seq, 128]  # cached KV                 │
│     │                                                                  │
│     Outputs:                                                           │
│       hidden_states: [1, 1, 896]                                       │
│       present_key: [1, 8, 1, 128]  # new KV only for current token   │
│       present_value: [1, 8, 1, 128]                                    │
│     │                                                                  │
│     Update KV cache: append new keys/values                           │
│   │                                                                    │
│   norm: hidden_states → normalized                                    │
│   lm_head: normalized → logits[1, 1, vocab_size]                      │
│   │                                                                    │
│ Step 3: Sample Next Token                                              │
│   │                                                                    │
│   Apply temperature scaling (if temperature != 1.0)                   │
│   Sample from logits (argmax or nucleus sampling)                     │
│   next_token_id = sampled_token                                        │
│   │                                                                    │
│ Step 4: Check Stopping Conditions                                      │
│   │                                                                    │
│   if next_token_id == EOS_TOKEN: break                                │
│   if len(generated_tokens) >= max_new_tokens: break                   │
│   │                                                                    │
│ Step 5: Append Token                                                   │
│   │                                                                    │
│   generated_tokens.append(next_token_id)                              │
│   total_seq_length += 1                                                │
└────────────────────────────────────────────────────────────────────────┘
```

**Key Optimization: KV Cache Reuse**

```
Without KV Cache (recompute every token):
├── Token 1: Process positions [0,1,2,3,4] → Layer 0 processes 5 tokens
├── Token 2: Process positions [0,1,2,3,4,5] → Layer 0 processes 6 tokens
├── Token 3: Process positions [0,1,2,3,4,5,6] → Layer 0 processes 7 tokens
└── O(n²) complexity

With KV Cache (reuse previous computation):
├── Token 1: Process positions [0,1,2,3,4] → Cache keys/values
├── Token 2: Process position [5] only → Use cached KV
├── Token 3: Process position [6] only → Use cached KV
└── O(n) complexity
```

---

## Tensor Registry Data Flow

**Outcome:** Understand how the Tensor Registry enables cross-component data passing.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TENSOR REGISTRY LIFECYCLE                            │
└─────────────────────────────────────────────────────────────────────────┘

Token Generation Loop:
│
├── registry = {}  # Fresh registry for each token
│
├── # Embedding phase
│   session = load_session("embeddings")
│   outputs = session.run({"input_ids": current_ids})
│   registry["hidden_states"] = outputs["hidden_states"]
│   registry["attention_mask"] = attention_mask
│   registry["position_ids"] = position_ids
│   unload_session("embeddings")
│
├── # Layer processing
│   for layer_idx in range(num_layers):
│       session = load_session(f"layer_{layer_idx}")
│       inputs = {
│           "hidden_states": registry["hidden_states"],
│           "attention_mask": registry["attention_mask"],
│           "position_ids": registry["position_ids"],
│           "past_key": kv_cache.get_key(layer_idx),
│           "past_value": kv_cache.get_value(layer_idx),
│       }
│       outputs = session.run(inputs)
│       registry["hidden_states"] = outputs["hidden_states"]
│       kv_cache.update(layer_idx, outputs["present_key"], outputs["present_value"])
│       unload_session(f"layer_{layer_idx}")
│
├── # Normalization phase
│   session = load_session("norm")
│   outputs = session.run({"hidden_states": registry["hidden_states"]})
│   registry["normalized"] = outputs["normalized_hidden"]
│   unload_session("norm")
│
├── # LM Head phase
│   session = load_session("lm_head")
│   outputs = session.run({"normalized": registry["normalized"]})
│   logits = outputs["logits"]
│   unload_session("lm_head")
│
└── # Registry discarded after token complete
    (KV cache persists across tokens)
```

---

*For execution mode details, see [Inference Pipelines](../04-inference-pipelines/README.md).*
