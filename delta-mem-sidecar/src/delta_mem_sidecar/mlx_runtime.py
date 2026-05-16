from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from delta_mem_sidecar.runtime import (
    ChatMessage,
    GenerationResult,
    RuntimeState,
    _rough_token_count,
)


class MlxRuntimeState(RuntimeState):
    def __init__(self) -> None:
        super().__init__()
        self.delta_state: dict[int, Any] = {}


class MlxBackboneRuntime:
    """Apple Silicon runtime backed by `mlx-lm`.

    This is the efficient local backbone path for Apple Silicon. It does not yet
    apply δ-mem adapter weights; the MLX δ-mem attention path is tracked
    separately because upstream adapters are not standard MLX-LM adapters.
    """

    def __init__(
        self,
        *,
        model_path: str,
        model_id: str | None = None,
        adapter_dir: str | None = None,
        max_tokens: int = 512,
    ) -> None:
        self.model_path = model_path
        self.model_id = model_id or model_path
        self.adapter_dir = adapter_dir
        self.max_tokens = max_tokens
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._delta_enabled = False

    def fresh_state(self) -> MlxRuntimeState:
        return MlxRuntimeState()

    def save_state(self, state: RuntimeState, state_dir: str | Path) -> None:
        if not isinstance(state, MlxRuntimeState):
            raise TypeError("MlxBackboneRuntime requires MlxRuntimeState")
        path = Path(state_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "runtime_state.json").write_text(
            json.dumps({"updates": state.updates, "history": state.history}),
            encoding="utf-8",
        )
        delta_state_path = path / "delta_state.npz"
        if state.delta_state:
            import mlx.core as mx

            mx.savez(
                str(delta_state_path),
                **{f"layer_{layer_index}": value for layer_index, value in state.delta_state.items()},
            )
        elif delta_state_path.exists():
            delta_state_path.unlink()

    def load_state(self, state_dir: str | Path) -> MlxRuntimeState:
        path = Path(state_dir)
        state = self.fresh_state()
        metadata_path = path / "runtime_state.json"
        if metadata_path.exists():
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            state.updates = int(data.get("updates", 0))
            state.history = [str(item) for item in data.get("history", [])]
        delta_state_path = path / "delta_state.npz"
        if delta_state_path.exists():
            import mlx.core as mx

            loaded = mx.load(str(delta_state_path))
            state.delta_state = {
                int(key.removeprefix("layer_")): value
                for key, value in loaded.items()
                if key.startswith("layer_")
            }
        return state

    def generate(
        self,
        *,
        messages: list[ChatMessage],
        state: RuntimeState,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        if not isinstance(state, MlxRuntimeState):
            raise TypeError("MlxBackboneRuntime requires MlxRuntimeState")
        model, tokenizer = self._load()
        if self._delta_enabled:
            from delta_mem_sidecar.mlx_delta_attention import (
                load_mlx_delta_state,
                reset_mlx_delta_states,
            )

            if state.delta_state:
                load_mlx_delta_state(model, state.delta_state)
            else:
                reset_mlx_delta_states(model)
        prompt = _render_prompt(tokenizer, messages)
        kwargs: dict[str, Any] = {
            "max_tokens": max_tokens or self.max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        content = _call_mlx_generate(model, tokenizer, prompt, kwargs)
        if self._delta_enabled:
            from delta_mem_sidecar.mlx_delta_attention import get_mlx_delta_state

            state.delta_state = get_mlx_delta_state(model)
        state.updates += 1
        state.history.append(_last_user_text(messages))
        return GenerationResult(
            content=content,
            prompt_tokens=sum(_rough_token_count(message.content) for message in messages),
            completion_tokens=_rough_token_count(content),
        )

    def _load(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            try:
                from mlx_lm import load  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError(
                    "MLX runtime is not installed. Install the `mlx` optional extra "
                    "or run `pip install mlx-lm` before setting DELTA_MEM_RUNTIME=mlx."
                ) from exc
            self._model, self._tokenizer = load(self.model_path)
            if self.adapter_dir:
                from delta_mem_sidecar.mlx_delta_adapter import load_mlx_delta_adapter
                from delta_mem_sidecar.mlx_delta_attention import wrap_qwen_attention_layers

                adapter = load_mlx_delta_adapter(self.adapter_dir)
                _validate_adapter_compatible(self._model, adapter.weights_by_layer)
                wrap_qwen_attention_layers(
                    self._model,
                    weights_by_layer=adapter.weights_by_layer,
                    config=adapter.config,
                )
                self._delta_enabled = True
        return self._model, self._tokenizer


def _render_prompt(tokenizer: Any, messages: list[ChatMessage]) -> str:
    rendered_messages = [
        {"role": message.role, "content": message.content}
        for message in messages
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(
            rendered_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if isinstance(rendered, str):
            return rendered
    return "\n".join(f"{message.role}: {message.content}" for message in messages) + "\nassistant:"


def _call_mlx_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    requested_kwargs: dict[str, Any],
) -> str:
    try:
        from mlx_lm import generate  # type: ignore[import-not-found]
        from mlx_lm.sample_utils import make_sampler  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "MLX runtime is not installed. Install the `mlx` optional extra "
            "or run `pip install mlx-lm` before setting DELTA_MEM_RUNTIME=mlx."
        ) from exc

    generation_kwargs = dict(requested_kwargs)
    temperature = generation_kwargs.pop("temperature", None)
    if temperature is not None:
        generation_kwargs["sampler"] = make_sampler(temp=temperature)

    signature = inspect.signature(generate)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs = generation_kwargs if accepts_kwargs else {
        key: value
        for key, value in generation_kwargs.items()
        if key in signature.parameters
    }
    return str(generate(model, tokenizer, prompt=prompt, verbose=False, **kwargs))


def _last_user_text(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def _validate_adapter_compatible(model: Any, weights_by_layer: dict[int, Any]) -> None:
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers")
    for layer_index, weights in weights_by_layer.items():
        if layer_index >= len(layers):
            raise RuntimeError(
                f"δ-mem adapter targets layer {layer_index}, but model has only {len(layers)} layers"
            )
        attention = layers[layer_index].self_attn
        hidden_size = getattr(attention.q_proj, "input_dims", None)
        q_out = getattr(attention.q_proj, "output_dims", None)
        o_out = getattr(attention.o_proj, "output_dims", None)
        if hidden_size is not None and weights.memory_q_proj.shape[-1] != hidden_size:
            raise RuntimeError(
                f"δ-mem adapter layer {layer_index} hidden size mismatch: "
                f"adapter has {weights.memory_q_proj.shape[-1]}, model has {hidden_size}"
            )
        if q_out is not None and weights.delta_q_proj.shape[0] != q_out:
            raise RuntimeError(
                f"δ-mem adapter layer {layer_index} q output mismatch: "
                f"adapter has {weights.delta_q_proj.shape[0]}, model has {q_out}"
            )
        if o_out is not None and weights.delta_o_proj.shape[0] != o_out:
            raise RuntimeError(
                f"δ-mem adapter layer {layer_index} o output mismatch: "
                f"adapter has {weights.delta_o_proj.shape[0]}, model has {o_out}"
            )
