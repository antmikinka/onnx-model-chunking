"""
Config-aware architecture detection for ONNX graphs.

``detect_architecture`` reads node names, op types, and initializers from
the whole graph. It also reads ``config.json``, a GenAI config, and
``model_info``. The result is an ``ArchitectureDescriptor`` with MoE,
attention variants, an MTP head, n-gram embeddings, and the quantization
format. RyzenAI compiled names such as ``gqo_``, ``ssmlp_``, and
``MatMulNBits_k_`` still count as a layer graph when the original
``/model/layers.`` prefix was removed.
"""

import json
import re
from typing import Any, Dict, Iterable, List

from .architecture import (
    ArchitectureDescriptor,
    ArchitectureType,
    architecture_from_string,
    get_architecture_descriptor,
    quantized_op_scheme,
)

# Same compiled-op shapes the ONNX splitter uses for layer ids.
_RYZEN_LAYER = re.compile(
    r"^(?:matmulnbits_[a-z]_\d+_\d+|gqo_\d+_\d+_\d+|ssmlp_\d+_\d+_\d+)$",
    re.IGNORECASE,
)

_NESTED_CONFIG_KEYS = {
    "model",
    "decoder",
    "text_config",
    "config",
    "quantization_config",
    "quantization",
    "quant_config",
    "architecture_extras",
}

_QUANT_PRIORITY = ("matmulnbits", "awq", "gptq", "qlinear")

_SCHEME_FORMAT = {
    "int4_weight": "matmulnbits",
    "awq_int4": "awq",
    "int8_quantized": "qlinear",
}


def detect_architecture(
    graph,
    config_json=None,
    genai_config=None,
    model_info=None,
) -> ArchitectureDescriptor:
    """
    Detect an architecture from an ONNX graph and optional configs.

    ``config_json``, ``genai_config``, and ``model_info`` are objects or
    JSON strings. A later source overrides an earlier ``model_type``.
    Gated DeltaNet is recorded as ``gdn`` and sparse attention as ``qsa``.
    An MTP head, n-gram embeddings, and expert routing select a Qwen3.8
    hybrid when those signals are present. A graph with no stronger match
    stays LLaMA. No graph and no config stays unknown.

    Args:
        graph: ONNX graph, ONNX model, or None.
        config_json: Hugging Face ``config.json`` object.
        genai_config: GenAI config object.
        model_info: Split ``model_info.json`` object.

    Returns:
        ArchitectureDescriptor for the detected model.
    """
    configs = (
        _as_object(config_json, "config_json"),
        _as_object(genai_config, "genai_config"),
        _as_object(model_info, "model_info"),
    )
    names = _graph_names(graph)
    lowered = [name.lower() for name in names]
    layer_types = _string_list(configs, ("layer_types", "attention_variants"))
    layer_types.extend(_string_values(configs, ("attention_type", "attention_variant")))

    has_gdn = _any_text(lowered, _is_gdn) or _any_text(layer_types, _is_gdn)
    has_qsa = _any_text(lowered, _is_qsa) or _any_text(layer_types, _is_qsa)
    has_full = _any_text(layer_types, _is_full_attention)
    has_mtp = (
        _any_text(lowered, _is_mtp)
        or _any_true(configs, ("has_mtp_head", "use_mtp"))
        or _positive_int(configs, ("num_nextn_predict_layers",)) > 0
    )
    has_ngram = _any_text(lowered, _is_ngram) or _any_true(
        configs, ("has_ngram_embedding", "use_ngram_embedding")
    )
    has_moe = (
        _any_text(lowered, _is_moe_name)
        or _any_true(configs, ("is_moe",))
        or _positive_int(configs, ("num_experts", "num_local_experts")) > 0
    )
    has_qwen = any("q_norm" in text or "k_norm" in text for text in lowered)
    has_layers = any(
        "/model/layers." in text or "model.layers." in text for text in lowered
    )
    has_gpt2 = any(
        "/transformer/h." in text or "transformer.h." in text for text in lowered
    )
    has_ryzen = any(_is_ryzen_name(text) for text in lowered)

    model_type = _last_string(configs, ("model_type", "architecture", "arch_type"))
    named = (
        architecture_from_string(model_type)
        if model_type
        else ArchitectureType.UNKNOWN
    )
    hybrid = has_gdn or has_qsa or has_mtp or has_ngram or named is ArchitectureType.QWEN3_HYBRID
    if hybrid:
        arch = ArchitectureType.QWEN3_HYBRID
    elif named is not ArchitectureType.UNKNOWN:
        arch = named
        if has_moe and arch in {
            ArchitectureType.LLAMA,
            ArchitectureType.QWEN,
            ArchitectureType.QWEN2,
            ArchitectureType.QWEN3,
        }:
            arch = ArchitectureType.QWEN3_MOE
        elif has_qwen and arch in {
            ArchitectureType.LLAMA,
            ArchitectureType.QWEN,
            ArchitectureType.QWEN2,
        }:
            arch = ArchitectureType.QWEN3
    elif has_moe and (has_qwen or has_layers or has_ryzen):
        arch = ArchitectureType.QWEN3_MOE
    elif has_qwen:
        arch = ArchitectureType.QWEN3
    elif has_gpt2 and not has_layers:
        arch = ArchitectureType.GPT2
    elif graph is not None or has_layers or has_ryzen:
        arch = ArchitectureType.LLAMA
    else:
        arch = ArchitectureType.UNKNOWN

    family = get_architecture_descriptor(arch)
    num_experts = _positive_int(configs, ("num_experts", "num_local_experts"))
    experts_per_token = _positive_int(
        configs, ("num_experts_per_token", "num_experts_per_tok")
    )
    is_moe = bool(family.is_moe or has_moe or num_experts)
    if experts_per_token and num_experts == 0:
        experts_per_token = 0

    variants: List[str] = []
    if has_gdn:
        variants.append("gdn")
    if has_qsa:
        variants.append("qsa")
    if has_full and variants:
        variants.append("full")

    quant = _quant_from_names(names)
    config_quant = _last_string(
        configs,
        ("quantization_format", "quantization_method", "quant_method", "quant_type"),
    )
    if config_quant:
        quant = config_quant

    max_seq_len = _positive_int(
        configs, ("max_seq_len", "max_position_embeddings", "max_sequence_length")
    )

    return ArchitectureDescriptor(
        arch_type=arch,
        supports_gqa=family.supports_gqa,
        supports_mqa=family.supports_mqa,
        is_moe=is_moe,
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        has_sliding_window=family.has_sliding_window,
        has_tied_embeddings=family.has_tied_embeddings,
        attention_variants=variants,
        max_seq_len=max_seq_len,
        has_mtp_head=has_mtp,
        has_ngram_embedding=has_ngram,
        quantization_format=quant,
    )


def _as_object(value: object, label: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{label} must be an object")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} is not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{label} is not an object")
        return parsed
    raise ValueError(f"{label} must be an object")


def _graph_names(graph: object) -> List[str]:
    if graph is None:
        return []
    if not hasattr(graph, "node") and hasattr(graph, "graph"):
        graph = getattr(graph, "graph")
    nodes = getattr(graph, "node", None)
    if nodes is None:
        raise ValueError("graph must be an ONNX graph")
    names: List[str] = []
    for node in nodes:
        for attr in ("name", "op_type"):
            value = getattr(node, attr, None)
            if isinstance(value, str) and value.strip():
                names.append(value.strip())
    initializers = getattr(graph, "initializer", ()) or ()
    for init in initializers:
        value = getattr(init, "name", None)
        if isinstance(value, str) and value.strip():
            names.append(value.strip())
    return names


def _iter_dicts(payload: Dict[str, Any], depth: int = 0) -> Iterable[Dict[str, Any]]:
    yield payload
    if depth >= 4:
        return
    for key, value in payload.items():
        if key in _NESTED_CONFIG_KEYS and isinstance(value, dict):
            yield from _iter_dicts(value, depth + 1)


def _last_string(payloads: Iterable[Dict[str, Any]], keys: Iterable[str]) -> str:
    found = ""
    key_set = tuple(keys)
    for payload in payloads:
        chosen = ""
        for blob in _iter_dicts(payload):
            for key in key_set:
                value = blob.get(key)
                if isinstance(value, str) and value.strip():
                    chosen = value.strip()
                    break
            if chosen:
                break
        if chosen:
            found = chosen
    return found


def _positive_int(payloads: Iterable[Dict[str, Any]], keys: Iterable[str]) -> int:
    found = 0
    key_set = tuple(keys)
    for payload in payloads:
        chosen = 0
        for blob in _iter_dicts(payload):
            for key in key_set:
                value = blob.get(key)
                if type(value) is int and value > 0:
                    chosen = value
                    break
            if chosen:
                break
        if chosen:
            found = chosen
    return found


def _any_true(payloads: Iterable[Dict[str, Any]], keys: Iterable[str]) -> bool:
    key_set = tuple(keys)
    for payload in payloads:
        for blob in _iter_dicts(payload):
            for key in key_set:
                if blob.get(key) is True:
                    return True
    return False


def _string_list(payloads: Iterable[Dict[str, Any]], keys: Iterable[str]) -> List[str]:
    found: List[str] = []
    key_set = tuple(keys)
    for payload in payloads:
        for blob in _iter_dicts(payload):
            for key in key_set:
                value = blob.get(key)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and item.strip():
                            found.append(item.strip().lower())
    return found


def _string_values(payloads: Iterable[Dict[str, Any]], keys: Iterable[str]) -> List[str]:
    found: List[str] = []
    key_set = tuple(keys)
    for payload in payloads:
        for blob in _iter_dicts(payload):
            for key in key_set:
                value = blob.get(key)
                if isinstance(value, str) and value.strip():
                    found.append(value.strip().lower())
    return found


def _any_text(values: Iterable[str], predicate) -> bool:
    return any(predicate(value) for value in values)


def _is_gdn(text: str) -> bool:
    return any(
        token in text
        for token in (
            "linear_attention",
            "linear_attn",
            "gated_deltanet",
            "gated_delta",
            "gdn",
        )
    )


def _is_qsa(text: str) -> bool:
    return any(token in text for token in ("sparse_attn", "sparse_attention", "qsa"))


def _is_full_attention(text: str) -> bool:
    return "full_attention" in text or text == "full"


def _is_mtp(text: str) -> bool:
    return any(token in text for token in ("mtp_head", "multi_token", "/mtp", ".mtp"))


def _is_ngram(text: str) -> bool:
    return "ngram" in text


def _is_moe_name(text: str) -> bool:
    if any(
        token in text
        for token in (
            "mlp.experts",
            ".experts.",
            "/experts/",
            "shared_expert",
        )
    ):
        return True
    if "gate_proj" in text:
        return False
    return any(
        token in text
        for token in ("mlp.gate.", "/mlp/gate", ".router", "/router")
    )


def _is_ryzen_name(text: str) -> bool:
    tail = text.rsplit("/", 1)[-1]
    return _RYZEN_LAYER.match(tail) is not None


def _quant_from_names(names: Iterable[str]) -> str:
    found = ""
    rank = {name: index for index, name in enumerate(_QUANT_PRIORITY)}
    for name in names:
        try:
            scheme = quantized_op_scheme(name)
        except ValueError:
            scheme = None
        if scheme is None:
            continue
        quant = _SCHEME_FORMAT.get(scheme, "")
        if not quant:
            continue
        if not found or rank.get(quant, 99) < rank.get(found, 99):
            found = quant
    return found
