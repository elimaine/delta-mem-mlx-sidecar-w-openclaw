from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.full_delta_mem_bench import key_term_recall, token_f1


@dataclass(frozen=True)
class Probe:
    name: str
    question: str
    expected: list[str]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay real OpenClaw session history into the sidecar and run behavior probes."
    )
    parser.add_argument("--history-file", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--model", required=True)
    parser.add_argument("--session-key", required=True)
    parser.add_argument("--probes-file", type=Path, help="JSON/JSONL probe file with name/question/expected.")
    parser.add_argument(
        "--probe",
        action="append",
        help="Inline probe as name||question||expected one,expected two. May be repeated.",
    )
    parser.add_argument("--max-history-events", type=int, default=80)
    parser.add_argument("--chunk-chars", type=int, default=3000)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    events = load_history_events(args.history_file)
    chunks = chunk_history(events[: args.max_history_events], chunk_chars=args.chunk_chars)
    probes = load_probes(args.probes_file, args.probe)
    if not probes:
        raise SystemExit("at least one --probe or --probes-file entry is required")

    replay_records = replay_history(
        chunks,
        base_url=args.base_url,
        model=args.model,
        session_key=args.session_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
    )
    probe_records = run_probes(
        probes,
        base_url=args.base_url,
        model=args.model,
        session_key=args.session_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
    )
    result = {
        "kind": "openclaw_session_replay_eval",
        "history_file": str(args.history_file),
        "base_url": args.base_url,
        "model": args.model,
        "session_key": args.session_key,
        "history_events": len(events),
        "history_events_replayed": min(len(events), args.max_history_events),
        "history_chunks": len(chunks),
        "summary": summarize(probe_records, replay_records),
        "replay": replay_records,
        "probes": probe_records,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def load_history_events(path: Path) -> list[dict[str, str]]:
    data = load_json_or_jsonl(path)
    if isinstance(data, dict):
        for key in ("messages", "turns", "history", "events", "items"):
            value = data.get(key)
            if isinstance(value, list):
                data = value
                break
    if not isinstance(data, list):
        raise ValueError("history file must be a JSON list, JSONL records, or object with messages/turns/history/events")
    events = []
    for item in data:
        event = normalize_history_event(item)
        if event is not None:
            events.append(event)
    if not events:
        raise ValueError("no usable history events found")
    return events


def load_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records


def normalize_history_event(item: Any) -> dict[str, str] | None:
    if isinstance(item, str):
        content = item
        role = "user"
    elif isinstance(item, dict):
        content_value = first_present(item, "content", "text", "message", "prompt", "response", "body")
        if isinstance(content_value, list):
            content = "\n".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content_value)
        elif content_value is None:
            return None
        else:
            content = str(content_value)
        role = str(first_present(item, "role", "type", "speaker", "author_role") or "user").lower()
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            role = "assistant" if role in {"agent", "model", "bot"} else "user"
    else:
        return None
    content = content.strip()
    if not content:
        return None
    return {"role": role, "content": content}


def first_present(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item:
            return item[key]
    return None


def chunk_history(events: list[dict[str, str]], *, chunk_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for index, event in enumerate(events):
        line = f"{index + 1}. {event['role']}: {event['content']}"
        if current and current_len + len(line) + 1 > chunk_chars:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def load_probes(path: Path | None, inline: list[str] | None) -> list[Probe]:
    probes: list[Probe] = []
    if path is not None:
        data = load_json_or_jsonl(path)
        if isinstance(data, dict):
            data = data.get("probes", [])
        if not isinstance(data, list):
            raise ValueError("probe file must be a list, JSONL records, or object with probes")
        for item in data:
            if not isinstance(item, dict):
                continue
            expected = item.get("expected", item.get("expected_substrings", []))
            if isinstance(expected, str):
                expected = [expected]
            probes.append(Probe(str(item.get("name", item.get("question", "probe"))), str(item["question"]), [str(x) for x in expected]))
    for value in inline or []:
        parts = value.split("||")
        if len(parts) != 3:
            raise ValueError("--probe must be name||question||expected one,expected two")
        probes.append(Probe(parts[0], parts[1], [part.strip() for part in parts[2].split(",") if part.strip()]))
    return probes


def replay_history(
    chunks: list[str],
    *,
    base_url: str,
    model: str,
    session_key: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> list[dict[str, Any]]:
    records = []
    for index, chunk in enumerate(chunks):
        started = time.perf_counter()
        response = post_chat(
            base_url=base_url,
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Store this real OpenClaw session-history chunk for later behavioral continuity probes. "
                        "Acknowledge briefly and do not summarize in detail.\n\n"
                        f"{chunk}"
                    ),
                }
            ],
            session_key=session_key,
            max_tokens=min(max_tokens, 64),
            temperature=temperature,
            timeout=timeout,
        )
        records.append(
            {
                "chunk_index": index,
                "chars": len(chunk),
                "elapsed_ms": elapsed_ms(started),
                "ack": response["choices"][0]["message"]["content"],
            }
        )
    return records


def run_probes(
    probes: list[Probe],
    *,
    base_url: str,
    model: str,
    session_key: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> list[dict[str, Any]]:
    records = []
    for probe in probes:
        started = time.perf_counter()
        response = post_chat(
            base_url=base_url,
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Answer using the replayed OpenClaw session history and current behavior context. "
                        "If the history does not support an answer, say so.\n"
                        f"Question: {probe.question}"
                    ),
                }
            ],
            session_key=session_key,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        output = response["choices"][0]["message"]["content"]
        score = score_output(output, probe.expected)
        records.append(
            {
                "name": probe.name,
                "question": probe.question,
                "expected": probe.expected,
                "output": output,
                "score": score,
                "passed": score["score"] >= 0.5,
                "elapsed_ms": elapsed_ms(started),
            }
        )
    return records


def score_output(output: str, expected: list[str]) -> dict[str, float]:
    if not expected:
        return {"score": 0.0, "exact_or_substring": 0.0, "key_term_recall": 0.0, "token_f1": 0.0}
    substring = max(float(normalize(item) in normalize(output)) for item in expected)
    term_score = max(key_term_recall(output, terms(item)) for item in expected)
    f1 = max(token_f1(output, item) for item in expected)
    return {
        "score": round(max(substring, term_score, f1), 4),
        "exact_or_substring": round(substring, 4),
        "key_term_recall": round(term_score, 4),
        "token_f1": round(f1, 4),
    }


def summarize(probe_records: list[dict[str, Any]], replay_records: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [record["score"]["score"] for record in probe_records]
    probe_latencies = [record["elapsed_ms"] for record in probe_records]
    replay_latencies = [record["elapsed_ms"] for record in replay_records]
    return {
        "probes": len(probe_records),
        "passed": sum(1 for record in probe_records if record["passed"]),
        "score_mean": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "probe_latency_ms_mean": round(sum(probe_latencies) / len(probe_latencies), 3) if probe_latencies else 0.0,
        "replay_chunks": len(replay_records),
        "replay_latency_ms_mean": round(sum(replay_latencies) / len(replay_latencies), 3) if replay_latencies else 0.0,
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


def terms(value: str) -> list[str]:
    return [token for token in re.findall(r"[a-zA-Z0-9]+", value.lower()) if len(token) > 2 or token.isdigit()]


def normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-zA-Z0-9]+", value.lower()))


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


if __name__ == "__main__":
    raise SystemExit(main())
