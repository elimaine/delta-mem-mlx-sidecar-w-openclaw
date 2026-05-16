from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class RuntimeState:
    """Opaque holder for future delta-Mem tensor state."""

    updates: int = 0
    history: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class GenerationResult:
    content: str
    prompt_tokens: int
    completion_tokens: int


class DeltaRuntime(Protocol):
    model_id: str

    def fresh_state(self) -> RuntimeState:
        """Create an empty per-session state."""

    def generate(
        self,
        *,
        messages: list[ChatMessage],
        state: RuntimeState,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        """Generate a response and mutate the provided state."""


class FakeDeltaRuntime:
    """Deterministic runtime for API and state-isolation tests."""

    def __init__(self, model_id: str = "delta-mem-fake") -> None:
        self.model_id = model_id

    def fresh_state(self) -> RuntimeState:
        return RuntimeState()

    def save_state(self, state: RuntimeState, state_dir: str | Path) -> None:
        path = Path(state_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "runtime_state.json").write_text(
            json.dumps({"updates": state.updates, "history": state.history}),
            encoding="utf-8",
        )

    def load_state(self, state_dir: str | Path) -> RuntimeState:
        path = Path(state_dir) / "runtime_state.json"
        if not path.exists():
            return self.fresh_state()
        data = json.loads(path.read_text(encoding="utf-8"))
        return RuntimeState(
            updates=int(data.get("updates", 0)),
            history=[str(item) for item in data.get("history", [])],
        )

    def generate(
        self,
        *,
        messages: list[ChatMessage],
        state: RuntimeState,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        user_text = _last_user_text(messages)
        state.updates += 1
        state.history.append(user_text)

        content = f"fake[{state.updates}]: {user_text}"
        return GenerationResult(
            content=content,
            prompt_tokens=sum(_rough_token_count(message.content) for message in messages),
            completion_tokens=_rough_token_count(content),
        )


def _last_user_text(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def _rough_token_count(text: str) -> int:
    return max(1, len(text.split())) if text else 0
