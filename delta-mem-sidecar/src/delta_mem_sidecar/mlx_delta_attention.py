from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from delta_mem_sidecar.mlx_delta_math import delta_affine_scan


@dataclass(frozen=True)
class MlxDeltaAttentionWeights:
    memory_q_proj: Any
    memory_k_proj: Any
    memory_v_proj: Any
    delta_q_proj: Any
    delta_o_proj: Any
    beta_proj: Any
    beta_bias: Any
    lambda_proj: Any | None = None
    lambda_bias: Any | None = None


@dataclass(frozen=True)
class MlxDeltaAttentionConfig:
    rank: int
    alpha: float = 1.0
    active_delta_heads: frozenset[str] = frozenset({"q", "o"})
    normalize_qk: bool = True
    couple_lambda: bool = True
    state_update_mode: str = "standard"

    @property
    def delta_scaling(self) -> float:
        return self.alpha / self.rank


class MlxDeltaAttention:
    """MLX wrapper that applies δ-mem Q/O corrections around Qwen attention."""

    def __init__(
        self,
        base_attention: Any,
        weights: MlxDeltaAttentionWeights,
        config: MlxDeltaAttentionConfig,
    ) -> None:
        self.base = base_attention
        self.weights = weights
        self.delta_config = config
        self.n_heads = base_attention.n_heads
        self.n_kv_heads = base_attention.n_kv_heads
        self.scale = base_attention.scale
        self.q_proj = base_attention.q_proj
        self.k_proj = base_attention.k_proj
        self.v_proj = base_attention.v_proj
        self.o_proj = base_attention.o_proj
        self.rope = base_attention.rope
        self.q_norm = getattr(base_attention, "q_norm", None)
        self.k_norm = getattr(base_attention, "k_norm", None)
        self.delta_state: Any | None = None
        self.write_enabled = True
        self.last_reads: Any | None = None

    def reset_state(self) -> None:
        self.delta_state = None
        self.last_reads = None

    def state_snapshot(self) -> Any | None:
        return self.delta_state

    def load_state_snapshot(self, state: Any | None) -> None:
        self.delta_state = state

    def set_write_enabled(self, enabled: bool) -> None:
        self.write_enabled = enabled

    def __call__(
        self,
        x: Any,
        mask: Any | None = None,
        cache: Any | None = None,
    ) -> Any:
        import mlx.core as mx
        from mlx_lm.models.qwen2 import scaled_dot_product_attention

        batch_size, seq_len, _ = x.shape
        memory_q, memory_k, memory_v, beta, lam = self._memory_sequence_projections(x)
        state = self._ensure_state(batch_size, memory_q.dtype)
        if self.write_enabled:
            keep, erase, write = self._memory_update_coefficients(beta, lam)
            state, reads = delta_affine_scan(
                state,
                memory_q,
                memory_k,
                memory_v,
                keep,
                erase,
                write,
            )
            self.delta_state = state
        else:
            reads = mx.matmul(state, mx.expand_dims(memory_q, -1)).squeeze(-1)
        self.last_reads = reads

        queries = self.q_proj(x)
        delta_q = self._project_delta(reads, self.weights.delta_q_proj, "q")
        if delta_q is not None:
            queries = queries + delta_q.astype(queries.dtype)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(batch_size, seq_len, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(batch_size, seq_len, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_len, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if self.q_norm is not None:
            queries = self.q_norm(queries)
        if self.k_norm is not None:
            keys = self.k_norm(keys)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        base_output = self.o_proj(output)
        delta_o = self._project_delta(reads, self.weights.delta_o_proj, "o")
        if delta_o is not None:
            base_output = base_output + delta_o.astype(base_output.dtype)
        return base_output

    def _ensure_state(self, batch_size: int, dtype: Any) -> Any:
        import mlx.core as mx

        rank = self.delta_config.rank
        if self.delta_state is None or tuple(self.delta_state.shape) != (batch_size, rank, rank):
            self.delta_state = mx.zeros((batch_size, rank, rank), dtype=dtype)
        return self.delta_state

    def _memory_sequence_projections(self, x: Any) -> tuple[Any, Any, Any, Any, Any]:
        import mlx.core as mx

        memory_q = _linear(x, self.weights.memory_q_proj)
        memory_k = _linear(x, self.weights.memory_k_proj)
        memory_v = _linear(x, self.weights.memory_v_proj)
        if self.delta_config.normalize_qk:
            memory_q = _normalize_memory_projection(memory_q)
            memory_k = _normalize_memory_projection(memory_k)

        beta = mx.sigmoid(_linear(x, self.weights.beta_proj) + self.weights.beta_bias)
        if self.delta_config.state_update_mode == "no_lambda":
            lam = mx.ones_like(beta)
        elif self.delta_config.couple_lambda:
            lam = 1.0 - beta
        else:
            if self.weights.lambda_proj is None or self.weights.lambda_bias is None:
                raise ValueError("lambda weights are required when couple_lambda=False")
            lam = mx.sigmoid(_linear(x, self.weights.lambda_proj) + self.weights.lambda_bias)
        return memory_q, memory_k, memory_v, beta, lam

    def _project_delta(self, reads: Any, weight: Any, head_name: str) -> Any | None:
        if head_name not in self.delta_config.active_delta_heads:
            return None
        return _linear(reads, weight) * self.delta_config.delta_scaling

    def _memory_update_coefficients(self, beta: Any, lam: Any) -> tuple[Any, Any, Any]:
        import mlx.core as mx

        if self.delta_config.state_update_mode == "standard":
            return lam, beta, beta
        if self.delta_config.state_update_mode == "lambda_outside":
            return lam, lam * beta, beta
        if self.delta_config.state_update_mode == "no_lambda":
            return mx.ones_like(beta), beta, beta
        raise ValueError(f"Unsupported state_update_mode: {self.delta_config.state_update_mode}")


def wrap_qwen_attention_layers(
    model: Any,
    *,
    weights_by_layer: dict[int, MlxDeltaAttentionWeights],
    config: MlxDeltaAttentionConfig,
) -> list[int]:
    wrapped: list[int] = []
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers")
    for layer_index, weights in weights_by_layer.items():
        layer = layers[layer_index]
        layer.self_attn = MlxDeltaAttention(layer.self_attn, weights, config)
        wrapped.append(layer_index)
    return wrapped


def iter_mlx_delta_attention_modules(model: Any):
    inner = getattr(model, "model", model)
    for layer_index, layer in enumerate(getattr(inner, "layers", [])):
        attention = getattr(layer, "self_attn", None)
        if isinstance(attention, MlxDeltaAttention):
            yield layer_index, attention


def reset_mlx_delta_states(model: Any) -> None:
    for _, attention in iter_mlx_delta_attention_modules(model):
        attention.reset_state()


def get_mlx_delta_state(model: Any) -> dict[int, Any]:
    return {
        layer_index: attention.state_snapshot()
        for layer_index, attention in iter_mlx_delta_attention_modules(model)
        if attention.state_snapshot() is not None
    }


def load_mlx_delta_state(model: Any, state: dict[int, Any]) -> None:
    for layer_index, attention in iter_mlx_delta_attention_modules(model):
        attention.load_state_snapshot(state.get(layer_index))


def _linear(x: Any, weight: Any) -> Any:
    return x @ weight.T


def _normalize_memory_projection(x: Any) -> Any:
    import mlx.core as mx

    x = mx.tanh(x)
    norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
    return x / mx.maximum(norm, mx.array(1e-6, dtype=x.dtype))
