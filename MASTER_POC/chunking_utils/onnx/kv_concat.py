"""
Static KV concatenation for spatial NPUs.

Stephen Panaro's ``models/gpt2.py`` (``more-ane-transformers``) tries to
overwrite one row of the K cache and rejects every form that becomes a
scatter:

- ``torch.index_copy`` is not an ANE op
- ``scatter`` does not run on the ANE
- ``cat([old[:, :index], new, old[:, index+1:]])`` needs a runtime index
- ``k[:, index] = new_k`` lowers to ScatterND

The update that runs is ``k = torch.cat([old_k, new_k], dim=1)``. The host
then installs that output as the next ``kv_cache`` input.

XDNA has the same two constraints, plus a third that Panaro's later notes
already hit on ANE: the compiled program has a fixed shape. A concat whose
sequence axis grows cannot be an XCLBIN output. This module describes the
two static shapes that stay legal.

``host_prefix``
    The graph emits only the new tokens, shape ``[batch, heads, query_len,
    head_dim]``. It does not slice the past with ``position_ids``. The host
    writes ``concat(past[:valid], new)`` into the shadow buffer of shape
    ``max_seq_len`` and swaps pointers. This is the masked dense cache.

``fixed_window``
    The graph itself emits the concatenated tensor. ``Slice`` uses one
    axis, the sequence axis: ``starts=[query_len]``, ``ends=[max_seq_len]``,
    ``axes=[seq_axis]``. Those are initializers, never a runtime offset.
    ``present = Concat(past[:, :, Q:, :], new)`` has the same length as
    ``past``. This roll is valid only when the buffer is already full.
    A short prefix uses ``host_prefix``. The host only swaps the present
    pointer onto the next past input.

The attention graph that ran as one NPU kernel is
``golden_attention_subgraph``. It uses this fixed-window slice and adds
the two matmuls. This module does not build that graph.
"""

from typing import Dict, Sequence


FORBIDDEN_KV_MUTATION_OPS = ("Scatter", "ScatterND", "ScatterElements")

_LAYOUTS = ("host_prefix", "fixed_window")


def assert_no_kv_scatter(op_types: Sequence[str]) -> None:
    """Reject scatter-style cache updates before they reach a spatial NPU."""
    found = [op for op in op_types if op in FORBIDDEN_KV_MUTATION_OPS]
    if found:
        names = ", ".join(found)
        raise ValueError(
            f"Spatial NPU KV update cannot contain {names}. "
            "Concatenate the new tokens and swap cache pointers instead."
        )


def static_kv_concat_recipe(
    max_seq_len: int,
    query_len: int,
    seq_axis: int = 2,
    layout: str = "host_prefix",
) -> Dict[str, object]:
    """
    Describe the KV update a split ONNX component may emit.

    ``max_seq_len`` and ``query_len`` are compile-time constants. The
    returned op list never includes Scatter, ScatterND, or a Slice whose
    begin depends on the running position.
    """
    limit = int(max_seq_len)
    query = int(query_len)
    axis = int(seq_axis)
    if limit < 1:
        raise ValueError(f"max_seq_len must be positive, got {limit}")
    if query < 1 or query > limit:
        raise ValueError(
            f"query_len must be in 1..{limit}, got {query}"
        )
    if axis < 0 or axis > 3:
        raise ValueError(f"seq_axis must be in 0..3, got {axis}")
    chosen = str(layout).strip().lower().replace("-", "_")
    if chosen not in _LAYOUTS:
        raise ValueError(
            f"Unknown KV concat layout '{layout}'. Expected one of: "
            + ", ".join(_LAYOUTS)
        )

    if chosen == "fixed_window":
        kept = limit - query
        nodes = [
            {
                "op_type": "Slice",
                "inputs": ["past", "starts", "ends", "axes"],
                "output": "kept_past",
                "starts": [query],
                "ends": [limit],
                "axes": [axis],
                "kept": kept,
                "seq_axis": axis,
                "note": (
                    "Drop the oldest query_len tokens on the sequence axis "
                    "only. starts, ends, and axes are initializers."
                ),
            },
            {
                "op_type": "Concat",
                "inputs": ["kept_past", "new"],
                "output": "present",
                "axis": axis,
                "note": "present length stays max_seq_len.",
            },
        ]
        output_seq = limit
        host = (
            "Use this only when the cache is full. Bind past to the active "
            "OrtValue and present to the shadow. commit_present_swap installs "
            "present as the next past input."
        )
    else:
        nodes = [
            {
                "op_type": "Identity",
                "inputs": ["new"],
                "output": "new",
                "seq_len": query,
                "note": (
                    "The graph emits the new tokens only. A runtime slice of "
                    "the valid prefix is not an ONNX node."
                ),
            }
        ]
        output_seq = query
        host = (
            "Write concat(past[:valid], new) into the shadow buffer of "
            "shape max_seq_len, then swap it with the active pointer."
        )

    op_types = [node["op_type"] for node in nodes]
    assert_no_kv_scatter(op_types)
    return {
        "layout": chosen,
        "max_seq_len": limit,
        "query_len": query,
        "seq_axis": axis,
        "output_seq": output_seq,
        "present_shape_is_static": True,
        "forbidden_ops": list(FORBIDDEN_KV_MUTATION_OPS),
        "op_types": op_types,
        "nodes": nodes,
        "host": host,
    }
