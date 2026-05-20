from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.deterministic_context_list import (  # noqa: E402
    Candidate,
    context_query,
    rank_candidates,
    render_context_events,
    tokens,
)
from benchmarks.session_replay_eval import Probe  # noqa: E402


def test_tokens_are_lowercase_and_skip_common_words() -> None:
    assert tokens("What should session use with MLX?") == {"session", "mlx"}


def test_context_query_includes_probe_expected_terms_and_target() -> None:
    query = context_query(
        [{"role": "user", "content": "Remove Ollama references."}],
        [Probe("adapter", "Where is the adapter?", ["Hugging Face"])],
    )

    assert "Hugging Face" in query
    assert "Remove Ollama" in query


def test_rank_candidates_is_deterministic_with_tie_breaks() -> None:
    candidates = [
        Candidate("b.jsonl", 0, "assistant", "session optional integration."),
        Candidate("a.jsonl", 1, "assistant", "session optional plugin."),
        Candidate("a.jsonl", 0, "assistant", "Unrelated note."),
    ]

    ranked = rank_candidates(candidates, query_terms={"session", "optional"})

    assert ranked[0]["candidate"].source == "a.jsonl"
    assert ranked[0]["candidate"].index == 1
    assert ranked[1]["candidate"].source == "b.jsonl"
    assert ranked[2]["score"] < ranked[0]["score"]


def test_render_context_events_creates_single_system_event() -> None:
    selected = [
        {
            "candidate": Candidate("memory.jsonl", 0, "assistant", "session should be optional."),
            "score": 1.0,
            "matched_terms": ["session", "optional"],
        }
    ]

    events = render_context_events(selected, max_item_chars=120)

    assert len(events) == 1
    assert events[0]["role"] == "system"
    assert "session should be optional" in events[0]["content"]
