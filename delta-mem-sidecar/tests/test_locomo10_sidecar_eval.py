from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.locomo10_sidecar_eval import (  # noqa: E402
    build_cases,
    locomo_session_texts,
    score_prediction,
)


def test_locomo_session_texts_formats_numbered_sessions() -> None:
    row = {
        "conversation": {
            "speaker_a": "A",
            "speaker_b": "B",
            "session_2_date_time": "Tuesday",
            "session_2": [{"speaker": "B", "dia_id": "D2:1", "text": "second"}],
            "session_1_date_time": "Monday",
            "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "first"}],
        }
    }

    sessions = list(locomo_session_texts(row))

    assert sessions[0][0] == "session_1"
    assert "session_1 (Monday)" in sessions[0][1]
    assert "D1:1 A: first" in sessions[0][1]
    assert sessions[1][0] == "session_2"


def test_build_cases_normalizes_answers_to_strings() -> None:
    cases = build_cases(
        {
            "sample_id": "conv-test",
            "qa": [
                {"question": "When?", "answer": 2022, "category": 2},
            ],
        },
        limit=1,
    )

    assert cases[0].sample_id == "conv-test"
    assert cases[0].answer == "2022"
    assert cases[0].category == 2


def test_score_prediction_accepts_substring_match() -> None:
    score = score_prediction("Caroline is single.", "Single")

    assert score["score"] == 1.0
    assert score["substring_match"] == 1.0
