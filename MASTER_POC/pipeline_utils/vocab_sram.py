"""
Vocabulary rows that fit in NPU tile SRAM.

The embedding table and the LM head are the two tensors whose leading
dimension is the vocabulary. A GPT-2 table is about 77 MB in float16,
and this repo's tile SRAM figure is 16-32 MB. A chunk is half of that
tile so the activation can reside with the weight.

14000 is the Apple Neural Engine row count. At hidden 768 and float16
that chunk is about 20.5 MB, which does not fit a 16 MB tile. The row
count here comes from the byte budget. ``split_rows`` forces a width,
including 14000, when the tile size is known to be larger.

Embedding chunks are exclusive, so the graph adds masked gathers.
Those gathers keep the short tail from ``vocab_chunk_rows``.

LM-head MatMuls use ``uniform_lm_head_rows``. A short tail is padded
with zeros to the same row count as the other tiles, the MatMuls are
concatenated, and a slice drops the padded columns. Neither graph
selects a chunk with an If.

A 149-tile chain of byte-identical one-hot weights plus a 640-zero
tail raised VAIML-BE-TPGEN 9032. The same shapes with a different
constant on each tile, including that zero tail, returned a finite
tensor: VAIML 153, NPU 1, ELF 5328344 bytes, no 9032. That probe was
one nonzero row per tile, not a dense vocabulary table. The kept
columns matched the unpadded matmul at gap 0.0 on CPU.
"""

from typing import List, Optional, Sequence


DEFAULT_TILE_SRAM_MB = 16
DEFAULT_HEADROOM = 0.5
_ROW_ALIGN = 64


def vocab_chunk_rows(
    vocab_size: int,
    hidden_size: int,
    dtype_bytes: int,
    *,
    tile_sram_mb: float = DEFAULT_TILE_SRAM_MB,
    headroom: float = DEFAULT_HEADROOM,
    split_rows: Optional[int] = None,
) -> List[int]:
    """
    Return the row count of each vocabulary chunk.

    One entry means the whole table fits and the graph should stay a
    single Gather or MatMul. ``split_rows`` below 1 disables the split.
    """
    vocab = int(vocab_size)
    hidden = int(hidden_size)
    width = int(dtype_bytes)
    if vocab < 1 or hidden < 1 or width < 1:
        raise ValueError("vocab_size, hidden_size, and dtype_bytes must be positive")
    if split_rows is not None:
        rows = int(split_rows)
        if rows < 1 or rows >= vocab:
            return [vocab]
    else:
        if float(tile_sram_mb) <= 0:
            raise ValueError(f"tile_sram_mb must be positive, got {tile_sram_mb}")
        if not 0 < float(headroom) <= 1:
            raise ValueError(f"headroom must be in (0, 1], got {headroom}")
        budget = int(float(tile_sram_mb) * 1024 * 1024 * float(headroom))
        raw = budget // (hidden * width)
        if raw >= _ROW_ALIGN:
            rows = (raw // _ROW_ALIGN) * _ROW_ALIGN
        else:
            rows = max(1, raw)
        if rows >= vocab:
            return [vocab]
    sizes: List[int] = []
    left = vocab
    while left > 0:
        take = min(rows, left)
        sizes.append(take)
        left -= take
    return sizes


def apply_split_embedding(table, token_ids, sizes: Sequence[int]):
    """Masked gather per chunk, then add. Matches a single embedding lookup."""
    import numpy as np

    ids = np.asarray(token_ids)
    hidden = table.shape[1]
    out = np.zeros(ids.shape + (hidden,), dtype=table.dtype)
    start = 0
    for rows in sizes:
        chunk = table[start:start + rows]
        shifted = np.clip(ids.astype(np.int64) - start, 0, rows - 1)
        gathered = chunk[shifted]
        mask = ((ids >= start) & (ids < start + rows)).astype(table.dtype)
        out = out + gathered * np.expand_dims(mask, -1)
        start += rows
    return out


def apply_split_lm_head(weight, hidden_states, sizes: Sequence[int]):
    """Matmul each vocabulary slice and concatenate on the vocab axis."""
    import numpy as np

    parts = []
    start = 0
    states = np.asarray(hidden_states)
    for rows in sizes:
        chunk = weight[start:start + rows]
        parts.append(states @ np.swapaxes(chunk, -1, -2))
        start += rows
    return np.concatenate(parts, axis=-1)


def uniform_lm_head_rows(vocab_size: int, chunk_rows: int) -> List[int]:
    """Equal LM-head tiles. The short tail is padded, then sliced off.

    ``vocab_chunk_rows`` still returns the short tail for embeddings.
    A chunk below 1, or a chunk that already covers the table, returns
    one entry of ``vocab_size``.
    """
    vocab = int(vocab_size)
    chunk = int(chunk_rows)
    if vocab < 1:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    if chunk < 1 or chunk >= vocab:
        return [vocab]
    count = (vocab + chunk - 1) // chunk
    return [chunk] * count


def apply_uniform_lm_head(weight, hidden_states, chunk_rows: int, *, last_token: bool = False):
    """Pad vocabulary rows, matmul equal slices, and drop the padded columns.

    ``weight`` is ``[vocab, hidden]``. ``last_token`` keeps the last
    sequence position before the matmul. The weight layout stored in
    ONNX is the transpose of each slice, ``[hidden, rows]``.
    """
    import numpy as np

    table = np.asarray(weight)
    states = np.asarray(hidden_states)
    if last_token:
        if states.ndim < 2:
            raise ValueError("last_token needs a sequence axis")
        states = states[..., -1:, :]
    vocab = int(table.shape[0])
    sizes = uniform_lm_head_rows(vocab, int(chunk_rows))
    padded = int(sum(sizes))
    if padded == vocab and len(sizes) == 1:
        return states @ np.swapaxes(table, -1, -2)
    pad_rows = padded - vocab
    if pad_rows:
        pad = [(0, pad_rows)] + [(0, 0)] * (table.ndim - 1)
        table = np.pad(table, pad, mode="constant")
    full = apply_split_lm_head(table, states, sizes)
    return full[..., :vocab]
