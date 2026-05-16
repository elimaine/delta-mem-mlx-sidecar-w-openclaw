import pytest


mlx = pytest.importorskip("mlx.core")

from delta_mem_sidecar.mlx_delta_math import delta_affine_scan


def test_delta_affine_scan_matches_hand_computed_single_token() -> None:
    state = mlx.array([[[1.0, 0.0], [0.0, 1.0]]])
    q = mlx.array([[[2.0, 3.0]]])
    k = mlx.array([[[0.5, 0.25]]])
    v = mlx.array([[[4.0, 8.0]]])
    keep = mlx.array([[[1.0, 1.0]]])
    erase = mlx.array([[[0.5, 0.5]]])
    write = mlx.array([[[0.25, 0.25]]])

    final_state, reads = delta_affine_scan(state, q, k, v, keep, erase, write)
    mlx.eval(final_state, reads)

    assert reads.tolist() == [[[2.0, 3.0]]]
    assert final_state.tolist() == [[[1.375, 0.1875], [0.9375, 1.46875]]]


def test_delta_affine_scan_token_mask_preserves_state() -> None:
    state = mlx.array([[[1.0, 0.0], [0.0, 1.0]]])
    q = mlx.array([[[2.0, 3.0]]])
    k = mlx.array([[[0.5, 0.25]]])
    v = mlx.array([[[4.0, 8.0]]])
    keep = mlx.array([[[1.0, 1.0]]])
    erase = mlx.array([[[0.5, 0.5]]])
    write = mlx.array([[[0.25, 0.25]]])
    token_mask = mlx.array([[False]])

    final_state, reads = delta_affine_scan(
        state,
        q,
        k,
        v,
        keep,
        erase,
        write,
        token_mask=token_mask,
    )
    mlx.eval(final_state, reads)

    assert reads.tolist() == [[[0.0, 0.0]]]
    assert final_state.tolist() == state.tolist()
