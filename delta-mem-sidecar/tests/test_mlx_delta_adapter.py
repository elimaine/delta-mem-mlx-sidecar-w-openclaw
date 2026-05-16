import json
import sys
from types import ModuleType

import pytest


mx = pytest.importorskip("mlx.core")

from delta_mem_sidecar.mlx_delta_adapter import (
    attention_config_from_raw,
    convert_torch_adapter_to_mlx_npz,
    load_mlx_delta_adapter,
)


def test_attention_config_from_raw_maps_upstream_fields() -> None:
    config = attention_config_from_raw(
        {
            "rank": 8,
            "alpha": 16,
            "delta_heads": ["q", "o"],
            "normalize_qk": True,
            "couple_lambda": False,
            "state_update_mode": "lambda_outside",
        }
    )

    assert config.rank == 8
    assert config.alpha == 16
    assert config.active_delta_heads == frozenset({"q", "o"})
    assert config.normalize_qk is True
    assert config.couple_lambda is False
    assert config.state_update_mode == "lambda_outside"


def test_load_mlx_delta_adapter_converts_torch_state_dict(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "delta_mem_config.json").write_text(
        json.dumps({"rank": 2, "alpha": 4, "delta_heads": ["q", "o"]})
    )
    (tmp_path / "delta_mem_adapter.pt").write_bytes(b"placeholder")
    torch = ModuleType("torch")

    def load(path, map_location=None, weights_only=False):
        assert path == tmp_path / "delta_mem_adapter.pt"
        assert map_location == "cpu"
        assert weights_only is True
        return {
            "model.layers.3.self_attn.memory_q_proj": [[1.0, 2.0]],
            "model.layers.3.self_attn.memory_k_proj": [[3.0, 4.0]],
            "model.layers.3.self_attn.memory_v_proj": [[5.0, 6.0]],
            "model.layers.3.self_attn.delta_q_proj": [[7.0], [8.0]],
            "model.layers.3.self_attn.delta_o_proj": [[9.0], [10.0]],
            "model.layers.3.self_attn.beta_proj": [[0.0, 0.0]],
            "model.layers.3.self_attn.beta_bias": [0.0],
            "unrelated.weight": [[99.0]],
        }

    torch.load = load
    monkeypatch.setitem(sys.modules, "torch", torch)

    adapter = load_mlx_delta_adapter(tmp_path)

    assert adapter.config.rank == 2
    assert adapter.config.alpha == 4
    assert sorted(adapter.weights_by_layer) == [3]
    weights = adapter.weights_by_layer[3]
    assert weights.memory_q_proj.tolist() == [[1.0, 2.0]]
    assert weights.delta_o_proj.tolist() == [[9.0], [10.0]]


def test_load_mlx_delta_adapter_prefers_mlx_npz_without_torch(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "delta_mem_config.json").write_text(
        json.dumps({"rank": 2, "alpha": 4, "delta_heads": ["q", "o"]})
    )
    mx.savez(
        str(tmp_path / "delta_mem_adapter_mlx.npz"),
        **{
            "model.layers.1.self_attn.memory_q_proj": mx.array([[1.0, 2.0]]),
            "model.layers.1.self_attn.memory_k_proj": mx.array([[3.0, 4.0]]),
            "model.layers.1.self_attn.memory_v_proj": mx.array([[5.0, 6.0]]),
            "model.layers.1.self_attn.delta_q_proj": mx.array([[7.0], [8.0]]),
            "model.layers.1.self_attn.delta_o_proj": mx.array([[9.0], [10.0]]),
            "model.layers.1.self_attn.beta_proj": mx.array([[0.0, 0.0]]),
            "model.layers.1.self_attn.beta_bias": mx.array([0.0]),
        },
    )
    monkeypatch.setitem(sys.modules, "torch", None)

    adapter = load_mlx_delta_adapter(tmp_path)

    assert sorted(adapter.weights_by_layer) == [1]
    assert adapter.weights_by_layer[1].memory_k_proj.tolist() == [[3.0, 4.0]]


def test_convert_torch_adapter_to_mlx_npz_writes_runtime_artifact(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "delta_mem_adapter.pt").write_bytes(b"placeholder")
    torch = ModuleType("torch")
    torch.load = lambda *args, **kwargs: {
        "model.layers.2.self_attn.memory_q_proj": [[1.0, 2.0]],
        "model.layers.2.self_attn.memory_k_proj": [[3.0, 4.0]],
        "unrelated.weight": [[99.0]],
    }
    monkeypatch.setitem(sys.modules, "torch", torch)

    output_path = convert_torch_adapter_to_mlx_npz(tmp_path)

    loaded = dict(mx.load(str(output_path)))
    assert sorted(loaded) == [
        "model.layers.2.self_attn.memory_k_proj",
        "model.layers.2.self_attn.memory_q_proj",
    ]


def test_load_mlx_delta_adapter_requires_complete_layer_weights(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "delta_mem_config.json").write_text(json.dumps({"rank": 2}))
    (tmp_path / "delta_mem_adapter.pt").write_bytes(b"placeholder")
    torch = ModuleType("torch")
    torch.load = lambda *args, **kwargs: {
        "model.layers.0.self_attn.memory_q_proj": [[1.0, 2.0]],
    }
    monkeypatch.setitem(sys.modules, "torch", torch)

    with pytest.raises(ValueError, match="missing"):
        load_mlx_delta_adapter(tmp_path)
