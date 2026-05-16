import pytest


mx = pytest.importorskip("mlx.core")

from mlx_lm.models.qwen2 import Attention, ModelArgs

from delta_mem_sidecar.mlx_delta_attention import (
    MlxDeltaAttention,
    MlxDeltaAttentionConfig,
    MlxDeltaAttentionWeights,
    get_mlx_delta_state,
    reset_mlx_delta_states,
    wrap_qwen_attention_layers,
)


def test_zero_delta_attention_matches_base_attention() -> None:
    base = _tiny_attention()
    weights = _weights(hidden_size=8, rank=2)
    config = MlxDeltaAttentionConfig(rank=2, active_delta_heads=frozenset({"q", "o"}))
    wrapped = MlxDeltaAttention(base, weights, config)
    x = mx.ones((1, 3, 8))

    base_output = base(x)
    wrapped_output = wrapped(x)
    mx.eval(base_output, wrapped_output)

    assert mx.max(mx.abs(base_output - wrapped_output)).item() < 1e-6
    assert wrapped.delta_state is not None


def test_delta_attention_can_change_output_and_reset_state() -> None:
    base = _tiny_attention()
    weights = _weights(
        hidden_size=8,
        rank=2,
        memory_fill=0.01,
        delta_q_fill=0.01,
        delta_o_fill=0.02,
    )
    config = MlxDeltaAttentionConfig(rank=2, active_delta_heads=frozenset({"q", "o"}))
    wrapped = MlxDeltaAttention(base, weights, config)
    x = mx.ones((1, 3, 8))

    base_output = base(x)
    wrapped_output = wrapped(x)
    mx.eval(base_output, wrapped_output)

    assert mx.max(mx.abs(base_output - wrapped_output)).item() > 0
    wrapped.reset_state()
    assert wrapped.delta_state is None


def test_wrap_qwen_attention_layers_replaces_selected_layers() -> None:
    model = type("Model", (), {})()
    model.model = type("Inner", (), {})()
    model.model.layers = [type("Layer", (), {})(), type("Layer", (), {})()]
    model.model.layers[0].self_attn = _tiny_attention()
    model.model.layers[1].self_attn = _tiny_attention()
    weights = _weights(hidden_size=8, rank=2)

    wrapped = wrap_qwen_attention_layers(
        model,
        weights_by_layer={1: weights},
        config=MlxDeltaAttentionConfig(rank=2),
    )

    assert wrapped == [1]
    assert isinstance(model.model.layers[1].self_attn, MlxDeltaAttention)
    assert not isinstance(model.model.layers[0].self_attn, MlxDeltaAttention)
    assert get_mlx_delta_state(model) == {}
    reset_mlx_delta_states(model)


def test_memory_update_coefficients_follow_upstream_modes() -> None:
    wrapped = MlxDeltaAttention(
        _tiny_attention(),
        _weights(hidden_size=8, rank=2),
        MlxDeltaAttentionConfig(rank=2, state_update_mode="standard"),
    )
    beta = mx.array([[[0.25, 0.5]]])
    lam = mx.array([[[0.75, 0.5]]])

    keep, erase, write = wrapped._memory_update_coefficients(beta, lam)

    assert keep.tolist() == lam.tolist()
    assert erase.tolist() == beta.tolist()
    assert write.tolist() == beta.tolist()

    wrapped = MlxDeltaAttention(
        _tiny_attention(),
        _weights(hidden_size=8, rank=2),
        MlxDeltaAttentionConfig(rank=2, state_update_mode="lambda_outside"),
    )

    keep, erase, write = wrapped._memory_update_coefficients(beta, lam)

    assert keep.tolist() == lam.tolist()
    assert erase.tolist() == [[[0.1875, 0.25]]]
    assert write.tolist() == beta.tolist()


def _tiny_attention() -> Attention:
    return Attention(
        ModelArgs(
            model_type="qwen2",
            hidden_size=8,
            num_hidden_layers=1,
            intermediate_size=16,
            num_attention_heads=2,
            rms_norm_eps=1e-6,
            vocab_size=16,
            num_key_value_heads=1,
            max_position_embeddings=16,
        )
    )


def _weights(
    *,
    hidden_size: int,
    rank: int,
    memory_fill: float = 0.0,
    delta_q_fill: float = 0.0,
    delta_o_fill: float = 0.0,
) -> MlxDeltaAttentionWeights:
    return MlxDeltaAttentionWeights(
        memory_q_proj=mx.full((rank, hidden_size), memory_fill),
        memory_k_proj=mx.full((rank, hidden_size), memory_fill),
        memory_v_proj=mx.full((rank, hidden_size), memory_fill),
        delta_q_proj=mx.full((hidden_size, rank), delta_q_fill),
        delta_o_proj=mx.full((hidden_size, rank), delta_o_fill),
        beta_proj=mx.zeros((rank, hidden_size)),
        beta_bias=mx.zeros((rank,)),
    )
