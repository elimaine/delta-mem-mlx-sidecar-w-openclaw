import pytest


mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

from delta_mem_sidecar.mlx_delta_math import delta_affine_scan


def test_delta_affine_scan_matches_torch_reference() -> None:
    torch.manual_seed(7)
    batch, seq_len, rank = 2, 5, 3
    state = torch.randn(batch, rank, rank)
    q = torch.randn(batch, seq_len, rank)
    k = torch.randn(batch, seq_len, rank)
    v = torch.randn(batch, seq_len, rank)
    keep = torch.sigmoid(torch.randn(batch, seq_len, rank))
    erase = torch.sigmoid(torch.randn(batch, seq_len, rank))
    write = torch.sigmoid(torch.randn(batch, seq_len, rank))
    token_mask = torch.tensor([[True, True, False, True, True], [True, False, True, True, False]])

    expected_state, expected_reads = _torch_delta_affine_scan(
        state,
        q,
        k,
        v,
        keep,
        erase,
        write,
        token_mask,
    )
    actual_state, actual_reads = delta_affine_scan(
        mx.array(state.numpy()),
        mx.array(q.numpy()),
        mx.array(k.numpy()),
        mx.array(v.numpy()),
        mx.array(keep.numpy()),
        mx.array(erase.numpy()),
        mx.array(write.numpy()),
        token_mask=mx.array(token_mask.numpy()),
    )
    mx.eval(actual_state, actual_reads)

    torch.testing.assert_close(
        torch.tensor(actual_state.tolist()),
        expected_state,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        torch.tensor(actual_reads.tolist()),
        expected_reads,
        atol=1e-5,
        rtol=1e-5,
    )


def _torch_delta_affine_scan(
    state,
    memory_q_seq,
    memory_k_seq,
    memory_v_seq,
    keep_seq,
    erase_seq,
    write_seq,
    token_mask,
):
    batch_size, seq_len, _ = memory_q_seq.shape
    current_state = state
    reads = []
    for token_idx in range(seq_len):
        q_t = memory_q_seq[:, token_idx, :]
        k_t = memory_k_seq[:, token_idx, :]
        v_t = memory_v_seq[:, token_idx, :]
        keep_t = keep_seq[:, token_idx, :].unsqueeze(-1)
        erase_t = erase_seq[:, token_idx, :].unsqueeze(-1)
        write_t = write_seq[:, token_idx, :].unsqueeze(-1)

        read_t = torch.einsum("bij,bj->bi", current_state, q_t)
        valid = token_mask[:, token_idx].view(batch_size, 1)
        read_t = read_t * valid.to(dtype=read_t.dtype)

        pred_t = torch.einsum("bij,bj->bi", current_state, k_t)
        write_outer = v_t.unsqueeze(-1) * k_t.unsqueeze(1)
        pred_outer = pred_t.unsqueeze(-1) * k_t.unsqueeze(1)
        next_state = keep_t * current_state - erase_t * pred_outer + write_t * write_outer
        valid_state = token_mask[:, token_idx].view(batch_size, 1, 1).to(dtype=next_state.dtype)
        current_state = next_state * valid_state + current_state * (1.0 - valid_state)
        reads.append(read_t)
    return current_state, torch.stack(reads, dim=1)
