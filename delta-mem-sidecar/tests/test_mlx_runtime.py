import sys
from types import ModuleType

import pytest

from delta_mem_sidecar.mlx_delta_attention import MlxDeltaAttentionWeights
from delta_mem_sidecar.mlx_runtime import MlxBackboneRuntime, _validate_adapter_compatible
from delta_mem_sidecar.runtime import ChatMessage


def test_mlx_runtime_missing_dependency_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    runtime = MlxBackboneRuntime(model_path="mlx-community/test-model")

    with pytest.raises(RuntimeError, match="MLX runtime is not installed"):
        runtime.generate(
            messages=[ChatMessage(role="user", content="hello")],
            state=runtime.fresh_state(),
        )


def test_mlx_runtime_generates_with_lazy_loaded_mlx_lm(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("mlx_lm")
    sample_utils = ModuleType("mlx_lm.sample_utils")
    calls: dict[str, object] = {}

    class Tokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            calls["messages"] = messages
            calls["tokenize"] = tokenize
            calls["add_generation_prompt"] = add_generation_prompt
            return "rendered prompt"

    def load(model_path: str):
        calls["model_path"] = model_path
        return "model", Tokenizer()

    def make_sampler(temp: float):
        return f"sampler:{temp}"

    def generate(model, tokenizer, *, prompt: str, verbose: bool, max_tokens: int, sampler):
        calls["generate"] = {
            "model": model,
            "prompt": prompt,
            "verbose": verbose,
            "max_tokens": max_tokens,
            "sampler": sampler,
        }
        return "mlx response"

    module.load = load
    module.generate = generate
    sample_utils.make_sampler = make_sampler
    monkeypatch.setitem(sys.modules, "mlx_lm", module)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)

    runtime = MlxBackboneRuntime(model_path="mlx-community/test-model")
    state = runtime.fresh_state()
    result = runtime.generate(
        messages=[ChatMessage(role="user", content="hello")],
        state=state,
        max_tokens=7,
        temperature=0.2,
    )

    assert result.content == "mlx response"
    assert state.updates == 1
    assert calls["model_path"] == "mlx-community/test-model"
    assert calls["messages"] == [{"role": "user", "content": "hello"}]
    assert calls["add_generation_prompt"] is True
    assert calls["generate"] == {
        "model": "model",
        "prompt": "rendered prompt",
        "verbose": False,
        "max_tokens": 7,
        "sampler": "sampler:0.2",
    }


def test_mlx_runtime_passes_generation_kwargs_when_generate_accepts_var_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("mlx_lm")
    sample_utils = ModuleType("mlx_lm.sample_utils")
    calls: dict[str, object] = {}

    class Tokenizer:
        pass

    def load(model_path: str):
        return "model", Tokenizer()

    def generate(model, tokenizer, *, prompt: str, verbose: bool, **kwargs):
        calls["kwargs"] = kwargs
        return "ok"

    def make_sampler(temp: float):
        return f"sampler:{temp}"

    module.load = load
    module.generate = generate
    sample_utils.make_sampler = make_sampler
    monkeypatch.setitem(sys.modules, "mlx_lm", module)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)

    runtime = MlxBackboneRuntime(model_path="mlx-community/test-model")
    runtime.generate(
        messages=[ChatMessage(role="user", content="hello")],
        state=runtime.fresh_state(),
        max_tokens=3,
        temperature=0.0,
    )

    assert calls["kwargs"] == {"max_tokens": 3, "sampler": "sampler:0.0"}


def test_validate_adapter_compatible_rejects_hidden_size_mismatch() -> None:
    model = type("Model", (), {})()
    model.model = type("Inner", (), {})()
    layer = type("Layer", (), {})()
    attention = type("Attention", (), {})()
    attention.q_proj = type("Linear", (), {"input_dims": 8, "output_dims": 8})()
    attention.o_proj = type("Linear", (), {"output_dims": 8})()
    layer.self_attn = attention
    model.model.layers = [layer]
    weights = MlxDeltaAttentionWeights(
        memory_q_proj=type("Shape", (), {"shape": (2, 9)})(),
        memory_k_proj=type("Shape", (), {"shape": (2, 9)})(),
        memory_v_proj=type("Shape", (), {"shape": (2, 9)})(),
        delta_q_proj=type("Shape", (), {"shape": (8, 2)})(),
        delta_o_proj=type("Shape", (), {"shape": (8, 2)})(),
        beta_proj=type("Shape", (), {"shape": (2, 9)})(),
        beta_bias=type("Shape", (), {"shape": (2,)})(),
    )

    with pytest.raises(RuntimeError, match="hidden size mismatch"):
        _validate_adapter_compatible(model, {0: weights})
