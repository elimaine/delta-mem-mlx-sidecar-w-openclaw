from __future__ import annotations

from typing import Any


def delta_affine_scan(
    state: Any,
    memory_q_seq: Any,
    memory_k_seq: Any,
    memory_v_seq: Any,
    keep_seq: Any,
    erase_seq: Any,
    write_seq: Any,
    token_mask: Any | None = None,
) -> tuple[Any, Any]:
    """MLX implementation of the δ-mem online affine state update.

    Inputs follow the single-partition upstream shape convention:
    `state=[batch, rank, rank]`, sequence tensors `[batch, seq, rank]`, and
    gate tensors `[batch, seq, rank]`.
    """

    try:
        import mlx.core as mx  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("MLX is required for delta_affine_scan") from exc

    current_state = state
    reads = []
    seq_len = int(memory_q_seq.shape[1])
    for token_idx in range(seq_len):
        q_t = memory_q_seq[:, token_idx, :]
        k_t = memory_k_seq[:, token_idx, :]
        v_t = memory_v_seq[:, token_idx, :]
        keep_t = mx.expand_dims(keep_seq[:, token_idx, :], -1)
        erase_t = mx.expand_dims(erase_seq[:, token_idx, :], -1)
        write_t = mx.expand_dims(write_seq[:, token_idx, :], -1)

        read_t = mx.matmul(current_state, mx.expand_dims(q_t, -1)).squeeze(-1)
        if token_mask is not None:
            valid = mx.expand_dims(token_mask[:, token_idx], -1).astype(read_t.dtype)
            read_t = read_t * valid

        pred_t = mx.matmul(current_state, mx.expand_dims(k_t, -1)).squeeze(-1)
        write_outer = mx.expand_dims(v_t, -1) * mx.expand_dims(k_t, 1)
        pred_outer = mx.expand_dims(pred_t, -1) * mx.expand_dims(k_t, 1)
        next_state = keep_t * current_state - erase_t * pred_outer + write_t * write_outer

        if token_mask is not None:
            valid_state = mx.expand_dims(
                mx.expand_dims(token_mask[:, token_idx], -1),
                -1,
            ).astype(next_state.dtype)
            current_state = next_state * valid_state + current_state * (1.0 - valid_state)
        else:
            current_state = next_state

        reads.append(read_t)

    return current_state, mx.stack(reads, axis=1)
