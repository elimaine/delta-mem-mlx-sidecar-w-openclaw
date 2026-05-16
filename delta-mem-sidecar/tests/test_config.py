import pytest

from delta_mem_sidecar.config import create_runtime_from_env
from delta_mem_sidecar.mlx_runtime import MlxBackboneRuntime
from delta_mem_sidecar.official_runtime import OfficialDeltaRuntime
from delta_mem_sidecar.runtime import FakeDeltaRuntime


def test_create_runtime_defaults_to_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DELTA_MEM_RUNTIME", raising=False)

    runtime = create_runtime_from_env()

    assert isinstance(runtime, FakeDeltaRuntime)


def test_official_runtime_requires_model_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DELTA_MEM_RUNTIME", "official")
    monkeypatch.delenv("DELTA_MEM_MODEL_PATH", raising=False)
    monkeypatch.setenv("DELTA_MEM_ADAPTER_DIR", "/models/delta-adapter")

    with pytest.raises(RuntimeError, match="DELTA_MEM_MODEL_PATH"):
        create_runtime_from_env()


def test_official_runtime_requires_adapter_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DELTA_MEM_RUNTIME", "official")
    monkeypatch.setenv("DELTA_MEM_MODEL_PATH", "/models/qwen")
    monkeypatch.delenv("DELTA_MEM_ADAPTER_DIR", raising=False)

    with pytest.raises(RuntimeError, match="DELTA_MEM_ADAPTER_DIR"):
        create_runtime_from_env()


def test_official_runtime_is_configured_without_loading_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DELTA_MEM_RUNTIME", "official")
    monkeypatch.setenv("DELTA_MEM_MODEL_PATH", "/models/qwen")
    monkeypatch.setenv("DELTA_MEM_ADAPTER_DIR", "/models/delta-adapter")
    monkeypatch.setenv("DELTA_MEM_DEVICE", "cuda:1")
    monkeypatch.setenv("DELTA_MEM_DTYPE", "float16")
    monkeypatch.setenv("DELTA_MEM_ATTN_IMPLEMENTATION", "flash_attention_2")
    monkeypatch.setenv("DELTA_MEM_MAX_NEW_TOKENS", "128")

    runtime = create_runtime_from_env()

    assert isinstance(runtime, OfficialDeltaRuntime)
    assert runtime.model_path == "/models/qwen"
    assert runtime.adapter_dir == "/models/delta-adapter"
    assert runtime.device == "cuda:1"
    assert runtime.dtype == "float16"
    assert runtime.attn_implementation == "flash_attention_2"
    assert runtime.max_new_tokens == 128


def test_mlx_runtime_requires_model_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DELTA_MEM_RUNTIME", "mlx")
    monkeypatch.delenv("DELTA_MEM_MODEL_PATH", raising=False)

    with pytest.raises(RuntimeError, match="DELTA_MEM_MODEL_PATH"):
        create_runtime_from_env()


def test_mlx_runtime_is_configured_without_loading_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DELTA_MEM_RUNTIME", "mlx")
    monkeypatch.setenv("DELTA_MEM_MODEL_PATH", "mlx-community/Qwen3-4B-Instruct-2507-4bit")
    monkeypatch.setenv("DELTA_MEM_ADAPTER_DIR", "/models/delta-mem-adapter")
    monkeypatch.setenv("DELTA_MEM_MODEL_ID", "delta-mem-mlx-backbone")
    monkeypatch.setenv("DELTA_MEM_MAX_NEW_TOKENS", "64")

    runtime = create_runtime_from_env()

    assert isinstance(runtime, MlxBackboneRuntime)
    assert runtime.model_path == "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    assert runtime.adapter_dir == "/models/delta-mem-adapter"
    assert runtime.model_id == "delta-mem-mlx-backbone"
    assert runtime.max_tokens == 64
