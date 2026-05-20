from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.transcript_toolbelt import (  # noqa: E402
    build_probes,
    build_sanitized_session,
    extract_events,
    sanitize_text,
    summarize_all,
    write_delta_svg,
)


def test_sanitize_text_redacts_common_sensitive_shapes() -> None:
    text = sanitize_text("email me@example.com token sk-abc123456789012345678901234 path /Users/example/secret")

    assert "<redacted-email>" in text
    assert "<redacted-token>" in text
    assert "<path>" in text


def test_extract_events_accepts_claude_style_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "session.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "message": {"role": "user", "content": "Use local gateway."}}),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {"role": "assistant", "content": [{"type": "text", "text": "Route via MLX."}]},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    events = extract_events(source, max_events=10, max_event_chars=100)

    assert events == [
        {"role": "user", "content": "Use local gateway."},
        {"role": "assistant", "content": "Route via MLX."},
    ]


def test_extract_events_accepts_nested_message_records(tmp_path: Path) -> None:
    source = tmp_path / "session.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"type": "session", "id": "s1"}),
                json.dumps({"type": "message", "message": {"role": "user", "content": "Run the benchmark."}}),
                json.dumps({"type": "message", "message": {"role": "assistant", "content": "Benchmark complete."}}),
            ]
        ),
        encoding="utf-8",
    )

    events = extract_events(source, max_events=10, max_event_chars=100)

    assert events == [
        {"role": "user", "content": "Run the benchmark."},
        {"role": "assistant", "content": "Benchmark complete."},
    ]


def test_extract_events_accepts_payload_wrapped_response_items(tmp_path: Path) -> None:
    source = tmp_path / "session.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"type": "session_meta", "payload": {"id": "s1"}}),
                json.dumps({"type": "response_item", "payload": {"role": "user", "content": "Need session replay."}}),
                json.dumps({"type": "response_item", "payload": {"role": "assistant", "content": "Replay ready."}}),
            ]
        ),
        encoding="utf-8",
    )

    events = extract_events(source, max_events=10, max_event_chars=100)

    assert events == [
        {"role": "user", "content": "Need session replay."},
        {"role": "assistant", "content": "Replay ready."},
    ]


def test_build_probes_has_eight_deterministic_items(tmp_path: Path) -> None:
    source = tmp_path / "session.jsonl"
    source.write_text("{}", encoding="utf-8")
    events = [
        {"role": "user", "content": "local gateway memory benchmark."},
        {"role": "assistant", "content": "MLX adapter session replay."},
        {"role": "user", "content": "retrieval graph context."},
        {"role": "assistant", "content": "Delta memory scoring."},
    ]

    session = build_sanitized_session(source, events)
    probes = build_probes(session)

    assert len(probes) == 8
    assert probes[0]["expected"] == [session["session_id"]]
    assert probes[2]["expected"] == [str(session["turn_count"])]


def test_summarize_all_reports_paired_delta_and_win_rate() -> None:
    rows = [
        {"session_id": "s1", "condition": "base_no_history", "score": {"score": 0.0}, "passed": False, "elapsed_ms": 10},
        {"session_id": "s1", "condition": "replayed_history", "score": {"score": 1.0}, "passed": True, "elapsed_ms": 20},
    ]
    sessions = [{"session_id": "s1", "base_score": 0.0, "replay_score": 1.0, "delta": 1.0}]

    summary = summarize_all(rows, sessions)

    assert summary["score_delta_mean"] == 1.0
    assert summary["win_rate"] == 1.0
    assert summary["latency_ratio_replay_over_base"] == 2.0


def test_write_delta_svg_creates_svg(tmp_path: Path) -> None:
    output = tmp_path / "deltas.svg"

    write_delta_svg(output, [{"base_score": 0.25, "replay_score": 0.75, "delta": 0.5}])

    assert output.read_text(encoding="utf-8").startswith("<svg")
