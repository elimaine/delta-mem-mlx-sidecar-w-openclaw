from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatCompletionMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = ""


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatCompletionMessage]
    attention_state: Any | None = None
    attentionState: Any | None = None
    delta_attention_state: Any | None = None
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0)


class StateMetadataResponse(BaseModel):
    object: Literal["delta.state"] = "delta.state"
    state_key_hash: str
    created_at: str
    updated_at: str
    updates: int


JsonDict = dict[str, Any]
