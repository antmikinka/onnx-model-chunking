"""
Architecture definitions and prefix mappings for model chunking.

This module defines supported model architectures and their corresponding
weight/node name prefixes for both SafeTensors and ONNX formats.

To add support for a new architecture:
1. Add it to ArchitectureType enum
2. Add it to LLAMA_STYLE_ARCHITECTURES when it uses the shared dense LLaMA prefixes
3. Otherwise add its own record to SAFETENSORS_PREFIXES and ONNX_PREFIXES
4. Add an ArchitectureDescriptor to ARCHITECTURE_DESCRIPTORS
"""

from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple
from dataclasses import dataclass, field, replace


class ArchitectureType(str, Enum):
    """
    Supported model architectures.
    
    The string value corresponds to the 'model_type' field in config.json.
    """
    LLAMA = "llama"
    MISTRAL = "mistral"
    PHI = "phi"
    GEMMA = "gemma"
    QWEN = "qwen"
    QWEN2 = "qwen2"
    QWEN3 = "qwen3"
    QWEN3_MOE = "qwen3_moe"
    QWEN3_HYBRID = "qwen3_hybrid"  # GDN + QSA + MoE
    GPT2 = "gpt2"
    UNKNOWN = "unknown"


# Dense architectures that share one LLaMA prefix record.
# Qwen3 MoE and Qwen3.8 keep the same four strings on their own records.
LLAMA_STYLE_ARCHITECTURES = {
    ArchitectureType.LLAMA,
    ArchitectureType.MISTRAL,
    ArchitectureType.PHI,
    ArchitectureType.GEMMA,
    ArchitectureType.QWEN,
    ArchitectureType.QWEN2,
    ArchitectureType.QWEN3,
}


@dataclass(frozen=True)
class ArchitecturePrefixes:
    """
    Weight/node name prefixes for a model architecture.

    Dense models use the four core prefixes. Router, expert, multi-token
    prediction, vision, cross-attention, and draft prefixes stay unset until
    an architecture names that component.

    Attributes:
        embed_prefix: Prefix for embedding layer weights/nodes
        layer_prefix: Prefix for transformer layer weights/nodes (followed by layer index)
        norm_prefix: Prefix for final normalization layer weights/nodes
        lm_head_prefix: Prefix for language model head weights/nodes
        router_prefix: Prefix for an MoE router, when the model has one
        expert_prefix: Prefix for MoE expert weights, when the model has them
        mtp_head_prefix: Prefix for a multi-token prediction head
        vision_prefix: Prefix for a multimodal vision tower
        cross_attn_prefix: Prefix for encoder-decoder cross-attention
        speculative_prefix: Prefix for a draft model
    """
    embed_prefix: str
    layer_prefix: str
    norm_prefix: str
    lm_head_prefix: str
    router_prefix: Optional[str] = None
    expert_prefix: Optional[str] = None
    mtp_head_prefix: Optional[str] = None
    vision_prefix: Optional[str] = None
    cross_attn_prefix: Optional[str] = None
    speculative_prefix: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("embed_prefix", "layer_prefix", "norm_prefix", "lm_head_prefix"):
            _require_prefix(getattr(self, name), name)
        for name in (
            "router_prefix",
            "expert_prefix",
            "mtp_head_prefix",
            "vision_prefix",
            "cross_attn_prefix",
            "speculative_prefix",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_prefix(value, name)


def _require_prefix(value: object, name: str) -> None:
    """Reject a missing or blank prefix string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


# One SafeTensors record for every dense LLaMA-style architecture.
_LLAMA_STYLE = ArchitecturePrefixes(
    embed_prefix="model.embed_tokens.",
    layer_prefix="model.layers.",
    norm_prefix="model.norm.",
    lm_head_prefix="lm_head.",
)

# One ONNX record for every dense LLaMA-style architecture.
_LLAMA_STYLE_ONNX = ArchitecturePrefixes(
    embed_prefix="/model/embed",
    layer_prefix="/model/layers.",
    norm_prefix="/model/norm",
    lm_head_prefix="/lm_head",
)

# Qwen3 MoE keeps the dense prefixes and names the router and experts.
_QWEN3_MOE_STYLE = replace(
    _LLAMA_STYLE,
    router_prefix="mlp.gate.",
    expert_prefix="mlp.experts.",
)
_QWEN3_MOE_ONNX = replace(
    _LLAMA_STYLE_ONNX,
    router_prefix="/mlp/gate",
    expert_prefix="/mlp/experts",
)

# Qwen3.8 adds the multi-token prediction head beside the MoE prefixes.
_QWEN3_HYBRID_STYLE = replace(
    _LLAMA_STYLE,
    router_prefix="mlp.gate.",
    expert_prefix="mlp.experts.",
    mtp_head_prefix="model.mtp_head.",
)
_QWEN3_HYBRID_ONNX = replace(
    _LLAMA_STYLE_ONNX,
    router_prefix="/mlp/gate",
    expert_prefix="/mlp/experts",
    mtp_head_prefix="/mtp_head",
)


# ==============================================================================
# SafeTensors Prefix Mappings (PyTorch weight key format)
# ==============================================================================

SAFETENSORS_PREFIXES: Dict[ArchitectureType, ArchitecturePrefixes] = {
    **{arch: _LLAMA_STYLE for arch in LLAMA_STYLE_ARCHITECTURES},
    ArchitectureType.QWEN3_MOE: _QWEN3_MOE_STYLE,
    ArchitectureType.QWEN3_HYBRID: _QWEN3_HYBRID_STYLE,
    ArchitectureType.GPT2: ArchitecturePrefixes(
        embed_prefix="transformer.wte.",
        layer_prefix="transformer.h.",
        norm_prefix="transformer.ln_f.",
        lm_head_prefix="lm_head.",
    ),
    ArchitectureType.UNKNOWN: _LLAMA_STYLE,
}


# ==============================================================================
# ONNX Prefix Mappings (ONNX node name format)
# ==============================================================================

ONNX_PREFIXES: Dict[ArchitectureType, ArchitecturePrefixes] = {
    **{arch: _LLAMA_STYLE_ONNX for arch in LLAMA_STYLE_ARCHITECTURES},
    ArchitectureType.QWEN3_MOE: _QWEN3_MOE_ONNX,
    ArchitectureType.QWEN3_HYBRID: _QWEN3_HYBRID_ONNX,
    ArchitectureType.GPT2: ArchitecturePrefixes(
        embed_prefix="/transformer/wte",
        layer_prefix="/transformer/h.",
        norm_prefix="/transformer/ln_f",
        lm_head_prefix="/lm_head",
    ),
    ArchitectureType.UNKNOWN: _LLAMA_STYLE_ONNX,
}


# Quantized ONNX ops keep the same layer prefix. The node name ends in the op.
QUANTIZED_OP_PATTERNS = {
    "MatMulNBits": "int4_weight",
    "QLinearMatMul": "int8_quantized",
    "AWQMatMul": "awq_int4",
}


def quantized_op_scheme(name: str) -> Optional[str]:
    """
    Return the weight scheme for a quantized ONNX op type or node path.

    ``/model/layers.0/self_attn/q_proj/MatMulNBits`` is int4_weight. A plain
    MatMul is not a quantized op. An empty name is rejected.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("node name must be a non-empty string")
    segment = name.strip().rsplit("/", 1)[-1]
    if "." in segment:
        tail = segment.rsplit(".", 1)[-1]
    else:
        tail = segment
    folded = tail.lower()
    for op, scheme in sorted(QUANTIZED_OP_PATTERNS.items(), key=lambda item: len(item[0]), reverse=True):
        op_folded = op.lower()
        if folded == op_folded or folded.startswith(op_folded + "_"):
            return scheme
    return None


def get_prefixes(
    arch: ArchitectureType,
    format: str  # "safetensors" or "onnx"
) -> ArchitecturePrefixes:
    """
    Get the weight/node name prefixes for a given architecture and format.
    
    Args:
        arch: The model architecture type.
        format: Either "safetensors" or "onnx".
    
    Returns:
        ArchitecturePrefixes for the specified architecture and format.
    
    Raises:
        ValueError: If format is not "safetensors" or "onnx".
    
    Example:
        >>> prefixes = get_prefixes(ArchitectureType.LLAMA, "safetensors")
        >>> prefixes.layer_prefix
        'model.layers.'
    """
    format_lower = format.lower()
    
    if format_lower == "safetensors":
        prefix_map = SAFETENSORS_PREFIXES
    elif format_lower == "onnx":
        prefix_map = ONNX_PREFIXES
    else:
        raise ValueError(f"Unknown format: {format}. Must be 'safetensors' or 'onnx'.")
    
    # Return prefixes for architecture, falling back to UNKNOWN if not found
    return prefix_map.get(arch, prefix_map[ArchitectureType.UNKNOWN])


def get_prefixes_tuple(
    arch: ArchitectureType,
    format: str
) -> Tuple[str, str, str, str]:
    """
    Get prefixes as a tuple (for backward compatibility with existing code).
    
    Args:
        arch: The model architecture type.
        format: Either "safetensors" or "onnx".
    
    Returns:
        Tuple of (embed_prefix, layer_prefix, norm_prefix, lm_head_prefix).

        Optional router, expert, multi-token, vision, cross-attention, and
        draft prefixes stay on the ArchitecturePrefixes record.
    """
    p = get_prefixes(arch, format)
    return (p.embed_prefix, p.layer_prefix, p.norm_prefix, p.lm_head_prefix)


def _is_qwen3_hybrid_name(model_type: str) -> bool:
    """True for Qwen3.8 and other hybrid Qwen3 names."""
    if "hybrid" in model_type:
        return True
    compact = model_type.replace("_", "").replace("-", "").replace(" ", "")
    return "qwen3.8" in model_type or "qwen38" in compact


def classify_qwen_family(names: Iterable[str]) -> Optional[ArchitectureType]:
    """
    Distinguish dense Qwen3, Qwen3 MoE, and hybrid Qwen3.8.

    Gated DeltaNet, sparse attention, a multi-token prediction head, or
    n-gram embeddings select the hybrid. Expert weights beside a Qwen q_norm
    select MoE. q_norm alone stays dense Qwen3. Any other name list returns
    None so the caller can keep its llama or GPT-2 result.
    """
    hybrid = False
    moe = False
    dense = False
    for name in names:
        text = name.lower()
        if (
            "linear_attn" in text
            or "gated_deltanet" in text
            or "gated_delta" in text
            or "sparse_attn" in text
            or "sparse_attention" in text
            or "mtp_head" in text
            or "multi_token" in text
            or "ngram_embed" in text
        ):
            hybrid = True
        if (
            "mlp.experts" in text
            or ".experts." in text
            or "shared_expert" in text
        ):
            moe = True
        if "q_norm" in text or "k_norm" in text:
            dense = True
    if hybrid:
        return ArchitectureType.QWEN3_HYBRID
    if moe and dense:
        return ArchitectureType.QWEN3_MOE
    if dense:
        return ArchitectureType.QWEN3
    return None


def architecture_from_string(model_type: str) -> ArchitectureType:
    """
    Convert a model_type string to ArchitectureType enum.
    
    Args:
        model_type: The model_type string from config.json.
    
    Returns:
        Corresponding ArchitectureType, or UNKNOWN if not recognized.
    """
    model_type_lower = model_type.lower() if model_type else ""
    
    for arch in ArchitectureType:
        if arch.value == model_type_lower:
            return arch
    
    # Check for partial matches
    if "llama" in model_type_lower:
        return ArchitectureType.LLAMA
    if "qwen" in model_type_lower:
        # Only check for explicit qwen3/qwen2 - NOT just "3" or "2" which causes false positives
        if "qwen3" in model_type_lower:
            if _is_qwen3_hybrid_name(model_type_lower):
                return ArchitectureType.QWEN3_HYBRID
            if "moe" in model_type_lower:
                return ArchitectureType.QWEN3_MOE
            return ArchitectureType.QWEN3
        if "qwen2" in model_type_lower:
            return ArchitectureType.QWEN2
        return ArchitectureType.QWEN
    
    return ArchitectureType.UNKNOWN


def _copy_attention_variants(value: object) -> List[str]:
    """Copy attention names. A string is not a list of names."""
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise ValueError("attention_variants must be a list of strings")
    copied: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("attention_variants must be a list of strings")
        text = item.strip().lower()
        if text not in copied:
            copied.append(text)
    return copied


@dataclass(frozen=True)
class ArchitectureDescriptor:
    """
    Family metadata for one ArchitectureType.

    supports_gqa and supports_mqa say the family can use that attention
    grouping. is_moe with zero expert counts means the checkpoint supplies
    the count. max_seq_len 0 means the same for context length. More than
    one attention variant is hybrid attention.
    """
    arch_type: ArchitectureType
    supports_gqa: bool = False
    supports_mqa: bool = False
    is_moe: bool = False
    num_experts: int = 0
    experts_per_token: int = 0
    has_sliding_window: bool = False
    has_tied_embeddings: bool = False
    attention_variants: List[str] = field(default_factory=list)
    max_seq_len: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.arch_type, ArchitectureType):
            if isinstance(self.arch_type, str):
                text = self.arch_type.strip().lower()
                try:
                    object.__setattr__(self, "arch_type", ArchitectureType(text))
                except ValueError as exc:
                    known = ", ".join(item.value for item in ArchitectureType)
                    raise ValueError(
                        f"Unknown arch_type {self.arch_type!r}. Expected one of: {known}"
                    ) from exc
            else:
                raise ValueError(
                    f"arch_type must be an ArchitectureType, got {self.arch_type!r}"
                )
        for name in (
            "supports_gqa",
            "supports_mqa",
            "is_moe",
            "has_sliding_window",
            "has_tied_embeddings",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a bool")
        for name in ("num_experts", "experts_per_token", "max_seq_len"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative int")
        if not self.is_moe and (self.num_experts or self.experts_per_token):
            raise ValueError("expert counts require is_moe")
        if self.experts_per_token and self.num_experts == 0:
            raise ValueError("experts_per_token requires num_experts")
        if self.num_experts and self.experts_per_token > self.num_experts:
            raise ValueError("experts_per_token exceeds num_experts")
        object.__setattr__(
            self, "attention_variants", _copy_attention_variants(self.attention_variants)
        )

    @property
    def has_hybrid_attention(self) -> bool:
        """True when the family names more than one attention variant."""
        return len(self.attention_variants) > 1


def _describe(arch: ArchitectureType, **kwargs: object) -> ArchitectureDescriptor:
    return ArchitectureDescriptor(arch_type=arch, **kwargs)


# Family traits. Expert counts stay 0 when checkpoints disagree.
# Qwen3.8 is the hybrid member: 512 experts, 10 chosen per token.
ARCHITECTURE_DESCRIPTORS: Dict[ArchitectureType, ArchitectureDescriptor] = {
    ArchitectureType.LLAMA: _describe(ArchitectureType.LLAMA, supports_gqa=True),
    ArchitectureType.MISTRAL: _describe(
        ArchitectureType.MISTRAL,
        supports_gqa=True,
        has_sliding_window=True,
        attention_variants=["sliding_window"],
    ),
    ArchitectureType.PHI: _describe(ArchitectureType.PHI, supports_gqa=True),
    ArchitectureType.GEMMA: _describe(
        ArchitectureType.GEMMA,
        supports_gqa=True,
        has_tied_embeddings=True,
    ),
    ArchitectureType.QWEN: _describe(ArchitectureType.QWEN),
    ArchitectureType.QWEN2: _describe(ArchitectureType.QWEN2, supports_gqa=True),
    ArchitectureType.QWEN3: _describe(ArchitectureType.QWEN3, supports_gqa=True),
    ArchitectureType.QWEN3_MOE: _describe(
        ArchitectureType.QWEN3_MOE,
        supports_gqa=True,
        is_moe=True,
    ),
    ArchitectureType.QWEN3_HYBRID: _describe(
        ArchitectureType.QWEN3_HYBRID,
        supports_gqa=True,
        is_moe=True,
        num_experts=512,
        experts_per_token=10,
        attention_variants=["full", "gated_deltanet", "sparse_attention"],
    ),
    ArchitectureType.GPT2: _describe(
        ArchitectureType.GPT2,
        has_tied_embeddings=True,
    ),
    ArchitectureType.UNKNOWN: _describe(ArchitectureType.UNKNOWN),
}


def get_architecture_descriptor(arch: object) -> ArchitectureDescriptor:
    """
    Return the family descriptor for an ArchitectureType or a model_type string.
    """
    if isinstance(arch, ArchitectureType):
        key = arch
    elif isinstance(arch, str):
        key = architecture_from_string(arch)
    else:
        raise ValueError(f"arch must be an ArchitectureType or model_type string, got {arch!r}")
    return ARCHITECTURE_DESCRIPTORS.get(key, ARCHITECTURE_DESCRIPTORS[ArchitectureType.UNKNOWN])
