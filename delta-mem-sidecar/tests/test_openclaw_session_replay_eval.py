from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.openclaw_session_replay_eval import (  # noqa: E402
    chunk_history,
    load_history_events,
    load_probes,
    score_output,
    summarize,
)


def test_load_history_events_accepts_openclaw_style_object(tmp_path: Path) -> None:
    history_file = tmp_path / "history.json"
    history_file.write_text(
        json.dumps(
            {
                "events": [
                    {"type": "agent", "message": "Use delta-mem-mlx for this session."},
                    {"speaker": "human", "content": [{"text": "Spawn pike tests."}]},
                    {"role": "tool", "body": ""},
                ]
            }
        ),
        encoding="utf-8",
    )

    events = load_history_events(history_file)

    assert events == [
        {"role": "assistant", "content": "Use delta-mem-mlx for this session."},
        {"role": "user", "content": "Spawn pike tests."},
    ]


def test_load_history_events_accepts_jsonl_records(tmp_path: Path) -> None:
    history_file = tmp_path / "history.jsonl"
    history_file.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "Remember the gateway changed transport."}),
                json.dumps({"author_role": "assistant", "response": "Use chat completions."}),
            ]
        ),
        encoding="utf-8",
    )

    events = load_history_events(history_file)

    assert events[0]["role"] == "user"
    assert events[1]["content"] == "Use chat completions."


def test_chunk_history_respects_character_budget() -> None:
    events = [
        {"role": "user", "content": "alpha"},
        {"role": "assistant", "content": "beta beta beta"},
        {"role": "user", "content": "gamma"},
    ]

    chunks = chunk_history(events, chunk_chars=35)

    assert len(chunks) == 3
    assert chunks[0].startswith("1. user: alpha")
    assert chunks[1].startswith("2. assistant:")


def test_load_probes_combines_file_and_inline_entries(tmp_path: Path) -> None:
    probes_file = tmp_path / "probes.json"
    probes_file.write_text(
        json.dumps({"probes": [{"name": "model", "question": "Which model?", "expected": "delta-mem-mlx"}]}),
        encoding="utf-8",
    )

    probes = load_probes(probes_file, ["transport||Which API?||chat completions,session key"])

    assert [probe.name for probe in probes] == ["model", "transport"]
    assert probes[0].expected == ["delta-mem-mlx"]
    assert probes[1].expected == ["chat completions", "session key"]


def test_score_output_uses_overlap_metrics() -> None:
    score = score_output("The agent should use delta mem mlx with attention shaping.", ["delta-mem-mlx"])

    assert score["score"] > 0
    assert score["key_term_recall"] == 1.0


def test_summarize_reports_passes_and_latency_means() -> None:
    summary = summarize(
        [
            {"score": {"score": 1.0}, "passed": True, "elapsed_ms": 10.0},
            {"score": {"score": 0.25}, "passed": False, "elapsed_ms": 30.0},
        ],
        [{"elapsed_ms": 5.0}, {"elapsed_ms": 15.0}],
    )

    assert summary["probes"] == 2
    assert summary["passed"] == 1
    assert summary["score_mean"] == 0.625
    assert summary["probe_latency_ms_mean"] == 20.0
    assert summary["replay_latency_ms_mean"] == 10.0
