from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.full_delta_mem_bench import key_term_recall, token_f1


@dataclass(frozen=True)
class LocomoCase:
    sample_id: str
    question_index: int
    question: str
    answer: str
    category: int | None


def main() -> int:
    parser = argparse.ArgumentParser(description="Small LoCoMo-10 sidecar eval using the official sample JSON.")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--model", required=True)
    parser.add_argument("--sample-limit", type=int, default=1)
    parser.add_argument("--question-limit", type=int, default=10)
    parser.add_argument("--session-limit", type=int, default=0, help="0 means all non-empty sessions.")
    parser.add_argument(
        "--query-context",
        choices=("none", "written"),
        default="none",
        help="When set to written, include the written LoCoMo sessions in each query prompt.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = json.loads(args.data_file.read_text(encoding="utf-8"))
    records = []
    started = time.perf_counter()
    for sample_index, row in enumerate(rows[: args.sample_limit]):
        session_key = f"locomo10:{row.get('sample_id', sample_index)}"
        write_records = write_locomo_history(
            row,
            base_url=args.base_url,
            model=args.model,
            session_key=session_key,
            session_limit=args.session_limit,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
        )
        query_context = build_query_context(row, session_limit=args.session_limit) if args.query_context == "written" else None
        for case in build_cases(row, limit=args.question_limit):
            query_started = time.perf_counter()
            response = post_chat(
                base_url=args.base_url,
                model=args.model,
                messages=[
                    {
                        "role": "user",
                        "content": build_query_prompt(case.question, query_context=query_context),
                    }
                ],
                session_key=session_key,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            prediction = response["choices"][0]["message"]["content"]
            records.append(
                {
                    "sample_id": case.sample_id,
                    "question_index": case.question_index,
                    "category": case.category,
                    "question": case.question,
                    "answer": case.answer,
                    "prediction": prediction,
                    "score": score_prediction(prediction, case.answer),
                    "elapsed_ms": round((time.perf_counter() - query_started) * 1000, 3),
                    "write_turns": len(write_records),
                    "query_context": args.query_context,
                }
            )
    summary = summarize(records, elapsed_ms=round((time.perf_counter() - started) * 1000, 3))
    result = {
        "kind": "locomo10_sidecar_eval",
        "data_file": str(args.data_file),
        "base_url": args.base_url,
        "model": args.model,
        "sample_limit": args.sample_limit,
        "question_limit": args.question_limit,
        "session_limit": args.session_limit,
        "query_context": args.query_context,
        "summary": summary,
        "records": records,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def write_locomo_history(
    row: dict[str, Any],
    *,
    base_url: str,
    model: str,
    session_key: str,
    session_limit: int,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> list[dict[str, Any]]:
    records = []
    for session_name, session_text in locomo_session_texts(row):
        if session_limit and len(records) >= session_limit:
            break
        started = time.perf_counter()
        response = post_chat(
            base_url=base_url,
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Store this LoCoMo conversation session for later question answering. "
                        "Acknowledge briefly.\n\n"
                        f"{session_text}"
                    ),
                }
            ],
            session_key=session_key,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        records.append(
            {
                "session": session_name,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "ack": response["choices"][0]["message"]["content"],
            }
        )
    return records


def selected_locomo_session_texts(row: dict[str, Any], *, session_limit: int) -> list[tuple[str, str]]:
    selected = []
    for session_name, session_text in locomo_session_texts(row):
        if session_limit and len(selected) >= session_limit:
            break
        selected.append((session_name, session_text))
    return selected


def build_query_context(row: dict[str, Any], *, session_limit: int) -> str:
    sessions = [session_text for _, session_text in selected_locomo_session_texts(row, session_limit=session_limit)]
    return "\n\n".join(sessions)


def build_query_prompt(question: str, *, query_context: str | None) -> str:
    if query_context:
        return (
            "Answer the question using this conversation context. "
            "Give only the answer when possible.\n\n"
            f"{query_context}\n\n"
            f"Question: {question}"
        )
    return (
        "Answer the question using the prior conversation memory. "
        "Give only the answer when possible.\n"
        f"Question: {question}"
    )


def locomo_session_texts(row: dict[str, Any]):
    conversation = row.get("conversation", {})
    for key in sorted(conversation, key=session_sort_key):
        if not re.fullmatch(r"session_\d+", key):
            continue
        turns = conversation.get(key)
        if not turns:
            continue
        date = conversation.get(f"{key}_date_time")
        lines = [f"{key} ({date})" if date else key]
        for turn in turns:
            speaker = turn.get("speaker", "speaker")
            dia_id = turn.get("dia_id", "")
            text = turn.get("text", "")
            lines.append(f"{dia_id} {speaker}: {text}".strip())
        yield key, "\n".join(lines)


def session_sort_key(value: str) -> tuple[int, str]:
    match = re.fullmatch(r"session_(\d+)(?:_date_time)?", value)
    return (int(match.group(1)) if match else 9999, value)


def build_cases(row: dict[str, Any], *, limit: int) -> list[LocomoCase]:
    sample_id = str(row.get("sample_id", "sample"))
    cases = []
    for index, qa in enumerate(row.get("qa", [])[:limit]):
        cases.append(
            LocomoCase(
                sample_id=sample_id,
                question_index=index,
                question=str(qa["question"]),
                answer=str(qa["answer"]),
                category=int(qa["category"]) if qa.get("category") is not None else None,
            )
        )
    return cases


def score_prediction(prediction: str, answer: str) -> dict[str, float]:
    terms = answer_terms(answer)
    exact = float(normalize(prediction) == normalize(answer))
    substring = float(normalize(answer) in normalize(prediction))
    term_score = key_term_recall(prediction, terms) if terms else 0.0
    f1 = token_f1(prediction, answer)
    return {
        "exact_match": round(exact, 4),
        "substring_match": round(substring, 4),
        "key_term_recall": round(term_score, 4),
        "token_f1": round(f1, 4),
        "score": round(max(exact, substring, term_score, f1), 4),
    }


def answer_terms(answer: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", answer.lower())
    return [token for token in tokens if len(token) > 2 or token.isdigit()]


def normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-zA-Z0-9]+", value.lower()))


def summarize(records: list[dict[str, Any]], *, elapsed_ms: float) -> dict[str, Any]:
    if not records:
        return {"records": 0, "score_mean": 0.0, "elapsed_ms": elapsed_ms}
    score_mean = sum(record["score"]["score"] for record in records) / len(records)
    by_category: dict[str, list[float]] = {}
    for record in records:
        by_category.setdefault(str(record["category"]), []).append(record["score"]["score"])
    return {
        "records": len(records),
        "score_mean": round(score_mean, 4),
        "elapsed_ms": elapsed_ms,
        "scores_by_category": {
            key: round(sum(values) / len(values), 4)
            for key, values in sorted(by_category.items())
        },
    }


def post_chat(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    session_key: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-OpenClaw-Session-Key": session_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


if __name__ == "__main__":
    sys.exit(main())
