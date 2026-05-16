from __future__ import annotations

import os
from dataclasses import dataclass

from delta_mem_sidecar.mlx_runtime import MlxBackboneRuntime
from delta_mem_sidecar.official_runtime import OfficialDeltaRuntime
from delta_mem_sidecar.runtime import DeltaRuntime, FakeDeltaRuntime


@dataclass(frozen=True)
class RuntimeSettings:
    runtime: str = "fake"
    model_path: str | None = None
    adapter_dir: str | None = None
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    attn_implementation: str | None = None
    max_new_tokens: int = 2048
    model_id: str | None = None

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        return cls(
            runtime=os.getenv("DELTA_MEM_RUNTIME", "fake").strip().lower(),
            model_path=os.getenv("DELTA_MEM_MODEL_PATH"),
            adapter_dir=os.getenv("DELTA_MEM_ADAPTER_DIR"),
            device=os.getenv("DELTA_MEM_DEVICE", "cuda:0"),
            dtype=os.getenv("DELTA_MEM_DTYPE", "bfloat16"),
            attn_implementation=os.getenv("DELTA_MEM_ATTN_IMPLEMENTATION") or None,
            max_new_tokens=int(os.getenv("DELTA_MEM_MAX_NEW_TOKENS", "2048")),
            model_id=os.getenv("DELTA_MEM_MODEL_ID") or None,
        )


def create_runtime_from_env() -> DeltaRuntime:
    settings = RuntimeSettings.from_env()
    if settings.runtime == "fake":
        return FakeDeltaRuntime()
    if settings.runtime == "official":
        if not settings.model_path:
            raise RuntimeError("DELTA_MEM_MODEL_PATH is required when DELTA_MEM_RUNTIME=official")
        if not settings.adapter_dir:
            raise RuntimeError("DELTA_MEM_ADAPTER_DIR is required when DELTA_MEM_RUNTIME=official")
        return OfficialDeltaRuntime(
            model_path=settings.model_path,
            adapter_dir=settings.adapter_dir,
            device=settings.device,
            dtype=settings.dtype,
            attn_implementation=settings.attn_implementation,
            max_new_tokens=settings.max_new_tokens,
        )
    if settings.runtime == "mlx":
        if not settings.model_path:
            raise RuntimeError("DELTA_MEM_MODEL_PATH is required when DELTA_MEM_RUNTIME=mlx")
        return MlxBackboneRuntime(
            model_path=settings.model_path,
            model_id=settings.model_id,
            adapter_dir=settings.adapter_dir,
            max_tokens=settings.max_new_tokens,
        )
    raise RuntimeError(f"unsupported DELTA_MEM_RUNTIME: {settings.runtime}")
