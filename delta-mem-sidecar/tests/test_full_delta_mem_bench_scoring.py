from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.full_delta_mem_bench import (  # noqa: E402
    compare_profiles,
    key_term_recall,
    score_answer,
    token_f1,
)


def test_key_term_recall_scores_partial_and_complete_matches() -> None:
    assert key_term_recall("Use Solarized Dark in the terminal.", ["solarized", "dark"]) == 1.0
    assert key_term_recall("Use Solarized Light in the terminal.", ["solarized", "dark"]) == 0.5


def test_token_f1_rewards_reference_overlap() -> None:
    score = token_f1("The answer is Delta Orion.", "Delta Orion")

    assert 0 < score < 1


def test_score_answer_uses_best_available_metric() -> None:
    score = score_answer(
        "The current project codename is Delta Orion.",
        references=["Delta Orion"],
        key_terms=["delta", "orion"],
    )

    assert score["score"] == 1.0
    assert score["key_term_recall"] == 1.0


def test_compare_profiles_reports_delta_gain_and_latency_overhead() -> None:
    comparison = compare_profiles(
        [
            {
                "profile": "plain",
                "summary": {
                    "score_mean": 0.25,
                    "latency_ms_mean": 100.0,
                    "scores_by_condition": {"no_context_recovery": 0.0},
                    "scores_by_benchmark": {"memory_agent_style": 0.0},
                },
            },
            {
                "profile": "delta",
                "summary": {
                    "score_mean": 0.5,
                    "latency_ms_mean": 125.0,
                    "scores_by_condition": {"no_context_recovery": 0.5},
                    "scores_by_benchmark": {"memory_agent_style": 0.5},
                },
            },
        ]
    )

    assert comparison["score_delta_points"] == 0.25
    assert comparison["score_ratio_delta_over_plain"] == 2.0
    assert comparison["latency_overhead_ratio_delta_over_plain"] == 1.25
    assert comparison["scores_by_condition_delta_minus_plain"]["no_context_recovery"] == 0.5
