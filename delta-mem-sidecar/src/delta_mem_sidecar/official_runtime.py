from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any

from delta_mem_sidecar.runtime import (
    ChatMessage,
    GenerationResult,
    RuntimeState,
    _rough_token_count,
)


@dataclass
class OfficialRuntimeState(RuntimeState):
    """Per-session state snapshot for the upstream δ-mem chat runtime."""

    snapshot: Any | None = None


class OfficialDeltaRuntime:
    """Adapter for the upstream `declare-lab/delta-Mem` runtime.

    Upstream δ-mem mutates online state on the loaded model object. This adapter
    serializes generation and swaps each logical session snapshot onto the model
    before generating so the sidecar can keep the existing per-session API.
    """

    def __init__(
        self,
        *,
        model_path: str,
        adapter_dir: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attn_implementation: str | None = None,
        max_new_tokens: int = 2048,
    ) -> None:
        self.model_id = "delta-mem-qwen3-4b-instruct"
        self.model_path = model_path
        self.adapter_dir = adapter_dir
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.max_new_tokens = max_new_tokens
        self._lock = RLock()
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._session_cls: Any | None = None

    def fresh_state(self) -> OfficialRuntimeState:
        return OfficialRuntimeState()

    def generate(
        self,
        *,
        messages: list[ChatMessage],
        state: RuntimeState,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        if not isinstance(state, OfficialRuntimeState):
            raise TypeError("OfficialDeltaRuntime requires OfficialRuntimeState")

        user_text = _last_user_text(messages)
        if not user_text:
            raise ValueError("official δ-mem runtime requires at least one user message")

        with self._lock:
            model, tokenizer, session_cls = self._load()
            session = session_cls(model=model, tokenizer=tokenizer, device=self.device)
            if state.snapshot is None:
                session.reset()
            else:
                session.load_snapshot(state.snapshot)

            generation = session.generate_reply(
                user_text,
                max_new_tokens=max_tokens or self.max_new_tokens,
                do_sample=(temperature is not None and temperature > 0),
                temperature=temperature or 1.0,
            )
            state.snapshot = session.snapshot()
            state.updates += 1

        content = str(generation.get("assistant_display") or generation.get("assistant") or "")
        return GenerationResult(
            content=content,
            prompt_tokens=sum(_rough_token_count(message.content) for message in messages),
            completion_tokens=_rough_token_count(content),
        )

    def _load(self) -> tuple[Any, Any, Any]:
        if self._model is None or self._tokenizer is None or self._session_cls is None:
            try:
                from deltamem.runtime.session import (  # type: ignore[import-not-found]
                    DeltaMemChatSession,
                    load_delta_mem_chat_model,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Official δ-mem runtime is not installed. Install the upstream "
                    "`declare-lab/delta-Mem` package on PYTHONPATH with its Torch/"
                    "Transformers dependencies before setting DELTA_MEM_RUNTIME=official."
                ) from exc

            self._model, self._tokenizer = load_delta_mem_chat_model(
                model_path=self.model_path,
                adapter_dir=self.adapter_dir,
                device=self.device,
                dtype=self.dtype,
                attn_implementation=self.attn_implementation,
            )
            self._session_cls = DeltaMemChatSession
        return self._model, self._tokenizer, self._session_cls


def _last_user_text(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""
