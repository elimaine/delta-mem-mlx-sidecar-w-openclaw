from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.openclaw_session_replay_eval import load_history_events, load_probes


@dataclass(frozen=True)
class Candidate:
    source: str
    index: int
    role: str
    content: str


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic memory-preload list for OpenClaw replay.")
    parser.add_argument("--memory-file", type=Path, action="append", required=True)
    parser.add_argument("--target-file", type=Path, required=True)
    parser.add_argument("--probes-file", type=Path)
    parser.add_argument("--probe", action="append")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-item-chars", type=int, default=360)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    probes = load_probes(args.probes_file, args.probe)
    query = context_query(load_history_events(args.target_file), probes)
    candidates = load_candidates(args.memory_file)
    ranked = rank_candidates(candidates, query_terms=tokens(query))
    selected = ranked[: args.top_k]
    events = render_context_events(selected, max_item_chars=args.max_item_chars)
    events.extend(load_history_events(args.target_file))

    result = {
        "kind": "deterministic_memory_preload_list",
        "memory_files": [str(path) for path in args.memory_file],
        "target_file": str(args.target_file),
        "top_k": args.top_k,
        "query_terms": sorted(tokens(query)),
        "selected": [
            {
                "rank": rank + 1,
                "source": item["candidate"].source,
                "index": item["candidate"].index,
                "role": item["candidate"].role,
                "score": item["score"],
                "content": item["candidate"].content,
                "matched_terms": item["matched_terms"],
            }
            for rank, item in enumerate(selected)
        ],
        "events": events,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def context_query(events: list[dict[str, str]], probes: list[Any]) -> str:
    probe_text = " ".join(
        " ".join([probe.name, probe.question, " ".join(probe.expected)])
        for probe in probes
    )
    target_text = " ".join(event["content"] for event in events)
    return f"{probe_text} {target_text}"


def load_candidates(paths: list[Path]) -> list[Candidate]:
    candidates = []
    for path in paths:
        for index, event in enumerate(load_history_events(path)):
            candidates.append(
                Candidate(
                    source=str(path),
                    index=index,
                    role=event["role"],
                    content=event["content"],
                )
            )
    return candidates


def rank_candidates(candidates: list[Candidate], *, query_terms: set[str]) -> list[dict[str, Any]]:
    ranked = []
    for candidate in candidates:
        candidate_terms = tokens(candidate.content)
        matched_terms = sorted(candidate_terms & query_terms)
        score = candidate_score(candidate_terms, query_terms, candidate.content)
        ranked.append(
            {
                "candidate": candidate,
                "score": score,
                "matched_terms": matched_terms,
            }
        )
    return sorted(
        ranked,
        key=lambda item: (
            -item["score"],
            item["candidate"].source,
            item["candidate"].index,
            item["candidate"].role,
            item["candidate"].content,
        ),
    )


def candidate_score(candidate_terms: set[str], query_terms: set[str], content: str) -> float:
    if not candidate_terms or not query_terms:
        return 0.0
    overlap = len(candidate_terms & query_terms)
    coverage = overlap / len(query_terms)
    density = overlap / len(candidate_terms)
    exact_phrase_bonus = sum(1 for term in query_terms if re.search(rf"\b{re.escape(term)}\b", content.lower())) * 0.001
    return round((2.0 * coverage) + density + exact_phrase_bonus, 6)


def render_context_events(selected: list[dict[str, Any]], *, max_item_chars: int) -> list[dict[str, str]]:
    if not selected:
        return []
    lines = [
        "Deterministic memory preload selected by lexical overlap against the target session and probes.",
        "Use these as relevant older memory items; prefer newer target-session turns when they conflict.",
    ]
    for rank, item in enumerate(selected, start=1):
        candidate = item["candidate"]
        content = truncate(candidate.content, max_item_chars)
        lines.append(f"{rank}. [{candidate.role}] {content}")
    return [{"role": "system", "content": "\n".join(lines)}]


def truncate(value: str, max_chars: int) -> str:
    value = " ".join(value.split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def tokens(value: str) -> set[str]:
    stopwords = {
        "about",
        "after",
        "again",
        "also",
        "and",
        "are",
        "because",
        "but",
        "can",
        "did",
        "for",
        "from",
        "have",
        "how",
        "into",
        "not",
        "our",
        "out",
        "should",
        "that",
        "the",
        "this",
        "through",
        "use",
        "was",
        "what",
        "when",
        "where",
        "with",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", value.lower())
        if (len(token) > 2 or token.isdigit()) and token not in stopwords
    }


if __name__ == "__main__":
    raise SystemExit(main())
