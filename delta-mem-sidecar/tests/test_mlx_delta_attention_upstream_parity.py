from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Attention, ModelArgs

from delta_mem_sidecar.mlx_delta_attention import (
    MlxDeltaAttention,
    MlxDeltaAttentionConfig,
    MlxDeltaAttentionWeights,
)
from delta_mem_sidecar.mlx_delta_math import delta_affine_scan


UPSTREAM_PATH = Path(os.environ.get("DELTA_MEM_UPSTREAM_PATH", "/private/tmp/delta-Mem-upstream"))
if not (UPSTREAM_PATH / "deltamem" / "core" / "delta_impl.py").exists():
    pytest.skip("upstream delta-Mem checkout is not available", allow_module_level=True)
sys.path.insert(0, str(UPSTREAM_PATH))

from deltamem.core.delta_impl import DeltaMemAttention, HFDeltaMemConfig  # noqa: E402
from transformers import Qwen3Config  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention  # noqa: E402


def test_mlx_delta_attention_components_match_upstream_torch() -> None:
    torch.manual_seed(17)
    batch, seq_len, hidden_size, rank = 2, 4, 8, 2
    hidden_states = torch.randn(batch, seq_len, hidden_size)
    weights = _torch_weights(hidden_size=hidden_size, rank=rank)

    upstream = _upstream_delta_attention(hidden_size=hidden_size, rank=rank)
    _copy_upstream_weights(upstream, weights)

    mlx_attention = MlxDeltaAttention(
        _mlx_base_attention(hidden_size=hidden_size),
        _mlx_weights(weights),
        MlxDeltaAttentionConfig(
            rank=rank,
            alpha=4.0,
            active_delta_heads=frozenset({"q", "o"}),
            normalize_qk=True,
            couple_lambda=True,
            state_update_mode="standard",
        ),
    )

    expected_q, expected_k, expected_v, expected_beta, expected_lambda = (
        upstream._memory_sequence_projections(hidden_states)
    )
    actual_q, actual_k, actual_v, actual_beta, actual_lambda = (
        mlx_attention._memory_sequence_projections(mx.array(hidden_states.numpy()))
    )
    mx.eval(actual_q, actual_k, actual_v, actual_beta, actual_lambda)

    _assert_close(actual_q, expected_q)
    _assert_close(actual_k, expected_k)
    _assert_close(actual_v, expected_v)
    _assert_close(actual_beta, expected_beta.squeeze(-1))
    _assert_close(actual_lambda, expected_lambda.squeeze(-1))

    upstream_state = torch.randn(batch, rank, rank)
    expected_state, expected_reads = upstream._memory_affine_scan(
        upstream_state,
        expected_q,
        expected_k,
        expected_v,
        expected_beta,
        expected_lambda,
    )
    keep, erase, write = mlx_attention._memory_update_coefficients(actual_beta, actual_lambda)
    actual_state, actual_reads = delta_affine_scan(
        mx.array(upstream_state.numpy()),
        actual_q,
        actual_k,
        actual_v,
        keep,
        erase,
        write,
    )
    mx.eval(actual_state, actual_reads)

    _assert_close(actual_state, expected_state)
    _assert_close(actual_reads, expected_reads)

    expected_delta_q, _, _ = upstream._compute_delta_qkv_from_reads(expected_reads)
    expected_delta_o = upstream._project_delta_head(expected_reads, upstream.delta_o_proj, "o")
    actual_delta_q = mlx_attention._project_delta(actual_reads, mlx_attention.weights.delta_q_proj, "q")
    actual_delta_o = mlx_attention._project_delta(actual_reads, mlx_attention.weights.delta_o_proj, "o")
    mx.eval(actual_delta_q, actual_delta_o)

    _assert_close(actual_delta_q, expected_delta_q)
    _assert_close(actual_delta_o, expected_delta_o)


def _upstream_delta_attention(*, hidden_size: int, rank: int) -> DeltaMemAttention:
    config = Qwen3Config(
        hidden_size=hidden_size,
        head_dim=hidden_size // 2,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=16,
        num_hidden_layers=1,
        vocab_size=32,
        max_position_embeddings=16,
    )
    base = Qwen3Attention(config, layer_idx=0)
    delta_config = HFDeltaMemConfig(
        rank=rank,
        alpha=4.0,
        delta_heads=("q", "o"),
        target_layers=(0,),
        normalize_qk=True,
        couple_lambda=True,
        state_update_mode="standard",
        rankwise_gates=True,
    )
    return DeltaMemAttention(base, delta_config)


def _mlx_base_attention(*, hidden_size: int) -> Attention:
    return Attention(
        ModelArgs(
            model_type="qwen2",
            hidden_size=hidden_size,
            num_hidden_layers=1,
            intermediate_size=16,
            num_attention_heads=2,
            rms_norm_eps=1e-6,
            vocab_size=32,
            num_key_value_heads=1,
            max_position_embeddings=16,
        )
    )


def _torch_weights(*, hidden_size: int, rank: int) -> dict[str, torch.Tensor]:
    return {
        "memory_q_proj": torch.randn(rank, hidden_size) * 0.2,
        "memory_k_proj": torch.randn(rank, hidden_size) * 0.2,
        "memory_v_proj": torch.randn(rank, hidden_size) * 0.2,
        "delta_q_proj": torch.randn(hidden_size, rank) * 0.2,
        "delta_o_proj": torch.randn(hidden_size, rank) * 0.2,
        "beta_proj": torch.randn(rank, hidden_size) * 0.2,
        "beta_bias": torch.randn(rank) * 0.2,
    }


def _copy_upstream_weights(attention: DeltaMemAttention, weights: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, value in weights.items():
            getattr(attention, name).copy_(value)
        attention.delta_k_proj.zero_()
        attention.delta_v_proj.zero_()


def _mlx_weights(weights: dict[str, torch.Tensor]) -> MlxDeltaAttentionWeights:
    return MlxDeltaAttentionWeights(
        memory_q_proj=mx.array(weights["memory_q_proj"].numpy()),
        memory_k_proj=mx.array(weights["memory_k_proj"].numpy()),
        memory_v_proj=mx.array(weights["memory_v_proj"].numpy()),
        delta_q_proj=mx.array(weights["delta_q_proj"].numpy()),
        delta_o_proj=mx.array(weights["delta_o_proj"].numpy()),
        beta_proj=mx.array(weights["beta_proj"].numpy()),
        beta_bias=mx.array(weights["beta_bias"].numpy()),
    )


def _assert_close(actual, expected: torch.Tensor) -> None:
    torch.testing.assert_close(
        torch.tensor(actual.tolist()),
        expected.detach(),
        atol=1e-5,
        rtol=1e-5,
    )
