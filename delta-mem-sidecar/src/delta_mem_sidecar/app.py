from __future__ import annotations

import json
import os
import time
import uuid
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import StreamingResponse

from delta_mem_sidecar.config import create_runtime_from_env
from delta_mem_sidecar.runtime import ChatMessage, DeltaRuntime
from delta_mem_sidecar.schemas import ChatCompletionRequest, JsonDict, StateMetadataResponse
from delta_mem_sidecar.state import InMemoryStateStore, StateMetadata

SESSION_HEADER = "X-OpenClaw-Session-Key"
STATE_HASH_HEADER = "X-Delta-State-Key-Hash"
DEFAULT_SESSION_PREAMBLE = (
    "Your LLM is running through delta-mem-mlx. This is an experimental "
    "MLX-native delta-memory adapter that keeps per-session neural state keyed "
    "by X-OpenClaw-Session-Key. It may improve continuity and recall across "
    "turns, but recall can be incomplete or wrong; prefer explicit recent "
    "context when accuracy matters."
)


def create_app(runtime: DeltaRuntime | None = None) -> FastAPI:
    runtime = runtime or create_runtime_from_env()
    store = InMemoryStateStore(runtime, persistence_dir=os.getenv("DELTA_MEM_STATE_DIR") or None)

    app = FastAPI(title="delta-mem-sidecar", version="0.1.0")
    app.state.runtime = runtime
    app.state.state_store = store

    @app.get("/health")
    def health() -> JsonDict:
        return {
            "status": "ok",
            "runtime": runtime.__class__.__name__,
            "model": runtime.model_id,
        }

    @app.get("/v1/models")
    def models() -> JsonDict:
        return {
            "object": "list",
            "data": [
                {
                    "id": runtime.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(
        request: ChatCompletionRequest,
        response: Response,
        state_key: Annotated[str, Depends(require_state_key)],
    ) -> JsonDict:
        if request.model != runtime.model_id:
            raise HTTPException(status_code=404, detail=f"unknown model: {request.model}")

        state = store.get_or_create(state_key)
        messages = [
            ChatMessage(role=message.role, content=coerce_message_content(message.content))
            for message in request.messages
        ]
        messages = prepend_session_preamble(messages)
        result = runtime.generate(
            messages=messages,
            state=state,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        metadata = store.mark_updated(state_key)
        response.headers[STATE_HASH_HEADER] = metadata.state_key_hash

        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        if request.stream:
            return stream_response(
                completion_id=completion_id,
                created=created,
                model=runtime.model_id,
                content=result.content,
                state_key_hash=metadata.state_key_hash,
            )

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": runtime.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
        }

    @app.post("/delta/session/reset", response_model=StateMetadataResponse)
    def reset_current_session(
        state_key: Annotated[str, Depends(require_state_key)],
    ) -> StateMetadataResponse:
        return metadata_response(store.reset(state_key))

    @app.post("/delta/state/{state_key}/reset", response_model=StateMetadataResponse)
    def reset_state(state_key: str) -> StateMetadataResponse:
        return metadata_response(store.reset(state_key))

    @app.get("/delta/state/{state_key}/metadata", response_model=StateMetadataResponse)
    def state_metadata(state_key: str) -> StateMetadataResponse:
        metadata = store.metadata(state_key)
        if metadata is None:
            raise HTTPException(status_code=404, detail="unknown state key")
        return metadata_response(metadata)

    return app


def stream_response(
    *,
    completion_id: str,
    created: int,
    model: str,
    content: str,
    state_key_hash: str,
) -> StreamingResponse:
    def events():
        first = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(first, separators=(',', ':'))}\n\n"
        if content:
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"
        done = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(done, separators=(',', ':'))}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            STATE_HASH_HEADER: state_key_hash,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def require_state_key(
    x_openclaw_session_key: Annotated[str | None, Header()] = None,
) -> str:
    if not x_openclaw_session_key or not x_openclaw_session_key.strip():
        raise HTTPException(
            status_code=400,
            detail=f"{SESSION_HEADER} header is required for state isolation",
        )
    return x_openclaw_session_key.strip()


def coerce_message_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif item.get("type") == "input_text" and isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return "\n".join(part for part in parts if part)
    return str(content)


def prepend_session_preamble(messages: list[ChatMessage]) -> list[ChatMessage]:
    preamble = session_preamble()
    if not preamble:
        return messages
    if messages and messages[0].role in {"system", "developer"} and preamble in messages[0].content:
        return messages
    return [ChatMessage(role="system", content=preamble), *messages]


def session_preamble() -> str:
    import os

    value = os.getenv("DELTA_MEM_SESSION_PREAMBLE")
    if value is None:
        return DEFAULT_SESSION_PREAMBLE
    return value.strip()


def metadata_response(metadata: StateMetadata) -> StateMetadataResponse:
    return StateMetadataResponse(
        state_key_hash=metadata.state_key_hash,
        created_at=metadata.created_at.isoformat(),
        updated_at=metadata.updated_at.isoformat(),
        updates=metadata.updates,
    )


app = create_app()
