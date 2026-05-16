from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from delta_mem_sidecar.mlx_delta_attention import (
    MlxDeltaAttentionConfig,
    MlxDeltaAttentionWeights,
)


_ADAPTER_KEY_RE = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\.self_attn\."
    r"(?P<name>memory_q_proj|memory_k_proj|memory_v_proj|delta_q_proj|"
    r"delta_o_proj|beta_proj|beta_bias|lambda_proj|lambda_bias)$"
)


@dataclass(frozen=True)
class MlxDeltaAdapter:
    config: MlxDeltaAttentionConfig
    weights_by_layer: dict[int, MlxDeltaAttentionWeights]
    raw_config: dict[str, Any]


def load_mlx_delta_adapter(adapter_dir: str | Path) -> MlxDeltaAdapter:
    adapter_path = Path(adapter_dir)
    raw_config = json.loads((adapter_path / "delta_mem_config.json").read_text())
    config = attention_config_from_raw(raw_config)
    state_dict = _load_adapter_state_dict(adapter_path)
    grouped: dict[int, dict[str, Any]] = {}
    for key, tensor in state_dict.items():
        match = _ADAPTER_KEY_RE.search(key)
        if match is None:
            continue
        grouped.setdefault(int(match.group("layer")), {})[match.group("name")] = _to_mlx_array(tensor)

    weights_by_layer = {
        layer: _weights_from_group(layer, group)
        for layer, group in sorted(grouped.items())
    }
    if not weights_by_layer:
        raise ValueError(f"no supported δ-mem attention weights found in {adapter_path}")
    return MlxDeltaAdapter(
        config=config,
        weights_by_layer=weights_by_layer,
        raw_config=raw_config,
    )


def convert_torch_adapter_to_mlx_npz(
    adapter_dir: str | Path,
    *,
    output_path: str | Path | None = None,
) -> Path:
    adapter_path = Path(adapter_dir)
    target_path = Path(output_path) if output_path is not None else adapter_path / "delta_mem_adapter_mlx.npz"
    state_dict = _load_torch_state_dict(adapter_path / "delta_mem_adapter.pt")
    arrays = {
        key: _to_mlx_array(tensor)
        for key, tensor in state_dict.items()
        if _ADAPTER_KEY_RE.search(key) is not None
    }
    if not arrays:
        raise ValueError(f"no supported δ-mem attention weights found in {adapter_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    _save_mlx_npz(target_path, arrays)
    return target_path


def attention_config_from_raw(raw_config: dict[str, Any]) -> MlxDeltaAttentionConfig:
    return MlxDeltaAttentionConfig(
        rank=int(raw_config["rank"]),
        alpha=float(raw_config.get("alpha", raw_config["rank"])),
        active_delta_heads=frozenset(raw_config.get("delta_heads", ("q", "o"))),
        normalize_qk=bool(raw_config.get("normalize_qk", True)),
        couple_lambda=bool(raw_config.get("couple_lambda", True)),
        state_update_mode=str(raw_config.get("state_update_mode", "standard")),
    )


def _load_adapter_state_dict(adapter_path: Path) -> dict[str, Any]:
    mlx_path = adapter_path / "delta_mem_adapter_mlx.npz"
    if mlx_path.exists():
        return _load_mlx_npz(mlx_path)
    return _load_torch_state_dict(adapter_path / "delta_mem_adapter.pt")


def _load_torch_state_dict(path: Path) -> dict[str, Any]:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Loading `delta_mem_adapter.pt` currently requires PyTorch for the "
            "one-time Torch checkpoint read. Install torch in the conversion "
            "environment, then convert the tensors to MLX arrays."
        ) from exc
    return torch.load(path, map_location="cpu", weights_only=True)


def _load_mlx_npz(path: Path) -> dict[str, Any]:
    import mlx.core as mx

    return dict(mx.load(str(path)))


def _save_mlx_npz(path: Path, arrays: dict[str, Any]) -> None:
    import mlx.core as mx

    mx.savez(str(path), **arrays)


def _weights_from_group(layer: int, group: dict[str, Any]) -> MlxDeltaAttentionWeights:
    required = {
        "memory_q_proj",
        "memory_k_proj",
        "memory_v_proj",
        "delta_q_proj",
        "delta_o_proj",
        "beta_proj",
        "beta_bias",
    }
    missing = sorted(required - set(group))
    if missing:
        raise ValueError(f"layer {layer} is missing δ-mem weights: {', '.join(missing)}")
    return MlxDeltaAttentionWeights(
        memory_q_proj=group["memory_q_proj"],
        memory_k_proj=group["memory_k_proj"],
        memory_v_proj=group["memory_v_proj"],
        delta_q_proj=group["delta_q_proj"],
        delta_o_proj=group["delta_o_proj"],
        beta_proj=group["beta_proj"],
        beta_bias=group["beta_bias"],
        lambda_proj=group.get("lambda_proj"),
        lambda_bias=group.get("lambda_bias"),
    )


def _to_mlx_array(tensor: Any) -> Any:
    import mlx.core as mx

    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "float"):
        tensor = tensor.float()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()
    return mx.array(tensor)
