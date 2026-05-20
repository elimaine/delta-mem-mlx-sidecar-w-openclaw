from fastapi.testclient import TestClient

from delta_mem_sidecar.app import STATE_HASH_HEADER, create_app
from delta_mem_sidecar.runtime import ChatMessage, FakeDeltaRuntime, GenerationResult, RuntimeState


def test_health_and_models() -> None:
    client = TestClient(create_app(FakeDeltaRuntime(model_id="delta-test")))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["model"] == "delta-test"

    models = client.get("/v1/models")
    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "delta-test"


def test_chat_requires_delta_mem_session_key() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/chat/completions",
        json={"model": "delta-mem-fake", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 400
    assert "X-Delta-Mem-Session-Key" in response.json()["detail"]


def test_chat_accepts_delta_mem_session_key() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Delta-Mem-Session-Key": "session-a"},
        json={"model": "delta-mem-fake", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "fake[1]: hi"


def test_state_isolated_by_session_key() -> None:
    client = TestClient(create_app())

    a1 = _chat(client, "session-a", "first")
    a2 = _chat(client, "session-a", "second")
    b1 = _chat(client, "session-b", "first")

    assert a1["choices"][0]["message"]["content"] == "fake[1]: first"
    assert a2["choices"][0]["message"]["content"] == "fake[2]: second"
    assert b1["choices"][0]["message"]["content"] == "fake[1]: first"


def test_reset_endpoint_resets_one_session() -> None:
    client = TestClient(create_app())

    assert _chat(client, "session-a", "first")["choices"][0]["message"]["content"] == "fake[1]: first"
    assert _chat(client, "session-a", "second")["choices"][0]["message"]["content"] == "fake[2]: second"
    assert _chat(client, "session-b", "first")["choices"][0]["message"]["content"] == "fake[1]: first"

    reset = client.post("/delta/state/session-a/reset")
    assert reset.status_code == 200
    assert reset.json()["updates"] == 0
    assert "state_key_hash" in reset.json()
    assert "session-a" not in reset.text

    assert _chat(client, "session-a", "after reset")["choices"][0]["message"]["content"] == "fake[1]: after reset"
    assert _chat(client, "session-b", "still warm")["choices"][0]["message"]["content"] == "fake[2]: still warm"


def test_metadata_endpoint_does_not_echo_raw_state_key() -> None:
    client = TestClient(create_app())

    _chat(client, "raw-channel-id", "hello")
    metadata = client.get("/delta/state/raw-channel-id/metadata")

    assert metadata.status_code == 200
    assert metadata.json()["updates"] == 1
    assert "state_key_hash" in metadata.json()
    assert "raw-channel-id" not in metadata.text


def test_streaming_returns_openai_compatible_sse_chunks() -> None:
    client = TestClient(create_app())

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"X-Delta-Mem-Session-Key": "session-a"},
        json={
            "model": "delta-mem-fake",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers[STATE_HASH_HEADER]
    assert response.headers["X-Delta-Attention-State-Count"] == "0"
    assert response.headers["X-Delta-Attention-State-Source"] == "none"
    assert '"role":"assistant"' in body
    assert '"content":"fake[1]: hi"' in body
    assert body.rstrip().endswith("data: [DONE]")


def test_chat_accepts_rich_content_parts() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Delta-Mem-Session-Key": "session-a"},
        json={
            "model": "delta-mem-fake",
            "messages": [
                {"role": "developer", "content": [{"type": "text", "text": "instruction"}]},
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "fake[1]: hello"


def test_chat_prepends_delta_mem_session_preamble(monkeypatch) -> None:
    runtime = CaptureRuntime()
    client = TestClient(create_app(runtime))

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Delta-Mem-Session-Key": "session-a"},
        json={
            "model": "delta-capture",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert runtime.last_messages is not None
    assert runtime.last_messages[0].role == "system"
    assert "delta-mem-mlx" in runtime.last_messages[0].content
    assert runtime.last_messages[1] == ChatMessage(role="user", content="hello")


def test_chat_can_disable_session_preamble(monkeypatch) -> None:
    monkeypatch.setenv("DELTA_MEM_SESSION_PREAMBLE", "")
    runtime = CaptureRuntime()
    client = TestClient(create_app(runtime))

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Delta-Mem-Session-Key": "session-a"},
        json={
            "model": "delta-capture",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert runtime.last_messages == [ChatMessage(role="user", content="hello")]


def test_attention_state_preloads_state_before_user_turn() -> None:
    runtime = CaptureRuntime()
    client = TestClient(create_app(runtime))

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Delta-Mem-Session-Key": "session-a"},
        json={
            "model": "delta-capture",
            "attention_state": [{"text": "retrieved memory fact"}],
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["X-Delta-Attention-State-Count"] == "1"
    assert len(runtime.calls) == 2
    assert runtime.calls[0][1].content == "retrieved memory fact"
    assert runtime.calls[0][1].role == "user"
    assert runtime.calls[1][-1] == ChatMessage(role="user", content="hello")


def test_attention_state_header_accepts_json_snippets() -> None:
    runtime = CaptureRuntime()
    client = TestClient(create_app(runtime))

    response = client.post(
        "/v1/chat/completions",
        headers={
            "X-Delta-Mem-Session-Key": "session-a",
            "X-Delta-Attention-State": '[{"snippet":"header memory"}]',
        },
        json={
            "model": "delta-capture",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["X-Delta-Attention-State-Count"] == "1"
    assert len(runtime.calls) == 2
    assert runtime.calls[0][1].content == "header memory"


def test_state_persistence_round_trips_fake_runtime(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DELTA_MEM_STATE_DIR", str(tmp_path))
    app = create_app()
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        headers={"X-Delta-Mem-Session-Key": "persisted-session"},
        json={
            "model": "delta-mem-fake",
            "messages": [{"role": "user", "content": "first"}],
        },
    )

    assert first.status_code == 200
    state_hash = first.headers[STATE_HASH_HEADER]
    assert (tmp_path / state_hash / "runtime_state.json").exists()

    second_app = create_app()
    second_client = TestClient(second_app)
    metadata = second_client.get("/delta/state/persisted-session/metadata")

    assert metadata.status_code == 200
    assert metadata.json()["updates"] == 1

    second = second_client.post(
        "/v1/chat/completions",
        headers={"X-Delta-Mem-Session-Key": "persisted-session"},
        json={
            "model": "delta-mem-fake",
            "messages": [{"role": "user", "content": "second"}],
        },
    )

    assert second.status_code == 200
    assert second.json()["choices"][0]["message"]["content"].startswith("fake[2]:")


def _chat(client: TestClient, session_key: str, content: str) -> dict:
    response = client.post(
        "/v1/chat/completions",
        headers={"X-Delta-Mem-Session-Key": session_key},
        json={
            "model": "delta-mem-fake",
            "messages": [{"role": "user", "content": content}],
        },
    )
    assert response.status_code == 200
    assert response.headers[STATE_HASH_HEADER]
    return response.json()


class CaptureRuntime:
    model_id = "delta-capture"

    def __init__(self) -> None:
        self.last_messages: list[ChatMessage] | None = None
        self.calls: list[list[ChatMessage]] = []

    def fresh_state(self) -> RuntimeState:
        return RuntimeState()

    def generate(
        self,
        *,
        messages: list[ChatMessage],
        state: RuntimeState,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        self.last_messages = messages
        self.calls.append(messages)
        return GenerationResult(content="captured", prompt_tokens=1, completion_tokens=1)
