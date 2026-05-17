from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.openclaw_session_replay_eval import (  # noqa: E402
    PASS_THRESHOLD,
    Probe,
    chunk_history,
    post_chat,
    replay_history,
    score_output,
)

STOPWORDS = {
    "about",
    "after",
    "agent",
    "also",
    "and",
    "are",
    "assistant",
    "before",
    "being",
    "can",
    "codex",
    "content",
    "could",
    "from",
    "have",
    "here",
    "into",
    "just",
    "like",
    "message",
    "need",
    "not",
    "openclaw",
    "please",
    "should",
    "that",
    "the",
    "this",
    "tool",
    "user",
    "with",
    "would",
    "your",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Toolbelt for sanitized OpenClaw transcript replay benchmarks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_lima = subparsers.add_parser("export-lima", help="Copy session JSONL files from a Lima OpenClaw VM.")
    export_lima.add_argument("--instance", default="clawfactory")
    export_lima.add_argument("--remote-root", default="/srv/clawfactory/bot_repos/sandy/state/agents")
    export_lima.add_argument("--remote-path-pattern", default="*/sessions/*.jsonl")
    export_lima.add_argument("--output-dir", type=Path, required=True)
    export_lima.add_argument("--limit", type=int, default=16)
    export_lima.add_argument("--max-bytes", type=int, default=3_000_000)

    sanitize = subparsers.add_parser("sanitize", help="Sanitize local JSONL transcript files.")
    sanitize.add_argument("sources", nargs="+", type=Path)
    sanitize.add_argument("--output-dir", type=Path, required=True)
    sanitize.add_argument("--limit", type=int, default=16)
    sanitize.add_argument("--max-bytes", type=int, default=3_000_000)
    sanitize.add_argument("--max-events", type=int, default=80)
    sanitize.add_argument("--max-event-chars", type=int, default=700)

    probe = subparsers.add_parser("probes", help="Generate deterministic probe files from sanitized sessions.")
    probe.add_argument("--sessions-file", type=Path, required=True)
    probe.add_argument("--output-dir", type=Path, required=True)
    probe.add_argument("--probes-per-session", type=int, default=8)

    run = subparsers.add_parser("run", help="Run no-history base and replay conditions single-threaded.")
    run.add_argument("--sessions-file", type=Path, required=True)
    run.add_argument("--probes-file", type=Path, required=True)
    run.add_argument("--base-url", default="http://127.0.0.1:8765")
    run.add_argument("--model", default="delta-mem-qwen3-4b-mlx")
    run.add_argument("--session-prefix", default="openclaw-16")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--chunk-chars", type=int, default=2600)
    run.add_argument("--max-tokens", type=int, default=40)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--timeout", type=float, default=120.0)

    report = subparsers.add_parser("report", help="Summarize result JSONL and emit SVG graphs.")
    report.add_argument("--results-jsonl", type=Path, required=True)
    report.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "export-lima":
        return export_lima_sessions(args)
    if args.command == "sanitize":
        return sanitize_command(args)
    if args.command == "probes":
        return probes_command(args)
    if args.command == "run":
        return run_command(args)
    if args.command == "report":
        return report_command(args)
    raise AssertionError(args.command)


def export_lima_sessions(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    listing = subprocess.check_output(
        [
            "limactl",
            "shell",
            args.instance,
            "--",
            "bash",
            "-lc",
            (
                "sudo find "
                + sh_quote(args.remote_root)
                + " -path "
                + sh_quote("*/" + args.remote_path_pattern)
                + " -type f"
                + " -size -"
                + str(args.max_bytes)
                + "c | sort | head -"
                + str(args.limit)
            ),
        ],
        text=True,
    )
    paths = [line.strip() for line in listing.splitlines() if line.strip()]
    manifest = []
    for index, remote_path in enumerate(paths, start=1):
        local_name = f"lima-session-{index:02d}-{Path(remote_path).name}"
        local_path = args.output_dir / local_name
        with local_path.open("wb") as handle:
            subprocess.run(
                ["limactl", "shell", args.instance, "--", "sudo", "cat", remote_path],
                check=True,
                stdout=handle,
            )
        manifest.append({"remote_path": remote_path, "local_path": str(local_path), "bytes": local_path.stat().st_size})
    (args.output_dir / "export-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"exported": len(manifest), "output_dir": str(args.output_dir)}, indent=2))
    return 0


def sanitize_command(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(path for path in args.sources if path.is_file() and path.stat().st_size <= args.max_bytes)
    sessions = []
    skipped = []
    for path in candidates:
        if len(sessions) >= args.limit:
            break
        events = extract_events(path, max_events=args.max_events, max_event_chars=args.max_event_chars)
        if len(events) < 2:
            skipped.append({"path": str(path), "reason": "too_few_events"})
            continue
        session = build_sanitized_session(path, events)
        session_path = args.output_dir / f"{session['session_id']}.jsonl"
        write_events_jsonl(session_path, session["events"])
        session["path"] = str(session_path)
        sessions.append(session)
    write_jsonl(args.output_dir / "sessions.jsonl", sessions)
    (args.output_dir / "manifest.json").write_text(
        json.dumps({"sessions": sessions, "skipped": skipped}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"sessions": len(sessions), "skipped": len(skipped), "output_dir": str(args.output_dir)}, indent=2))
    return 0


def probes_command(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sessions = read_jsonl(args.sessions_file)
    all_probes = []
    for session in sessions:
        probes = build_probes(session)[: args.probes_per_session]
        probe_file = args.output_dir / f"{session['session_id']}-probes.json"
        probe_file.write_text(json.dumps({"probes": probes}, indent=2) + "\n", encoding="utf-8")
        all_probes.append({"session_id": session["session_id"], "probes": probes, "probe_file": str(probe_file)})
    write_jsonl(args.output_dir / "probes.jsonl", all_probes)
    print(json.dumps({"sessions": len(all_probes), "output_dir": str(args.output_dir)}, indent=2))
    return 0


def run_command(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sessions = {session["session_id"]: session for session in read_jsonl(args.sessions_file)}
    probe_sets = read_jsonl(args.probes_file)
    rows = []
    session_summaries = []
    for item in probe_sets:
        session = sessions[item["session_id"]]
        probes = [Probe(probe["name"], probe["question"], list(probe["expected"])) for probe in item["probes"]]
        base_key = f"{args.session_prefix}:{session['session_id']}:base:{int(time.time())}"
        replay_key = f"{args.session_prefix}:{session['session_id']}:replay:{int(time.time())}"
        base_rows = run_probe_condition(
            condition="base_no_history",
            probes=probes,
            base_url=args.base_url,
            model=args.model,
            session_key=base_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
        )
        chunks = chunk_history(session["events"], chunk_chars=args.chunk_chars)
        replay_records = replay_history(
            chunks,
            base_url=args.base_url,
            model=args.model,
            session_key=replay_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
        )
        replay_rows = run_probe_condition(
            condition="replayed_history",
            probes=probes,
            base_url=args.base_url,
            model=args.model,
            session_key=replay_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
        )
        for row in [*base_rows, *replay_rows]:
            row.update(
                {
                    "session_id": session["session_id"],
                    "topic_terms": session["topic_terms"],
                    "source_hash": session["source_hash"],
                }
            )
            rows.append(row)
        session_summaries.append(summarize_session_result(session, base_rows, replay_rows, replay_records))
        write_jsonl(args.output_dir / "results.jsonl", rows)
        (args.output_dir / "session-summary.json").write_text(
            json.dumps(session_summaries, indent=2) + "\n",
            encoding="utf-8",
        )
    summary = summarize_all(rows, session_summaries)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def report_command(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.results_jsonl)
    summary = summarize_all(rows, summarize_sessions_from_rows(rows))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    bars = args.output_dir / "score-pass-bars.svg"
    deltas = args.output_dir / "paired-session-deltas.svg"
    write_bar_svg(bars, summary)
    write_delta_svg(deltas, summary["sessions"])
    markdown = render_markdown_report(summary, bars, deltas)
    (args.output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"summary": str(args.output_dir / "summary.json"), "graphs": [str(bars), str(deltas)]}, indent=2))
    return 0


def extract_events(path: Path, *, max_events: int, max_event_chars: int) -> list[dict[str, str]]:
    events = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if len(events) >= max_events:
            break
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        role, text = extract_role_text(item)
        text = sanitize_text(text)
        if role and text:
            events.append({"role": role, "content": text[:max_event_chars]})
    return events


def extract_role_text(item: Any) -> tuple[str | None, str]:
    if not isinstance(item, dict):
        return None, ""
    if isinstance(item.get("payload"), dict):
        payload_role, payload_text = extract_role_text(item["payload"])
        if payload_role and payload_text:
            return payload_role, payload_text
    role = str(item.get("role") or item.get("type") or "").lower()
    content = item.get("content")
    if isinstance(item.get("message"), dict):
        message = item["message"]
        role = str(message.get("role") or role).lower()
        content = message.get("content", content)
    if role not in {"user", "assistant", "system", "developer", "tool"}:
        if role in {"agent", "bot"}:
            role = "assistant"
        else:
            return None, ""
    if content is None and "text" in item:
        content = item["text"]
    return role, extract_text(content)


def extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") in {"text", "input_text", "output_text"} and part.get("text"):
                    parts.append(str(part["text"]))
                elif part.get("type") == "tool_result" and isinstance(part.get("content"), str):
                    parts.append(str(part["content"]))
        return "\n".join(parts)
    if isinstance(value, dict):
        return extract_text(value.get("text") or value.get("content") or "")
    return ""


def sanitize_text(text: str) -> str:
    text = re.sub(r"-----BEGIN [^-]+-----.*?-----END [^-]+-----", "<redacted-key>", text, flags=re.S)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "<redacted-email>", text)
    text = re.sub(r"https?://\S+", "<url>", text)
    text = re.sub(r"/Users/[^\s\"']+", "<path>", text)
    text = re.sub(r"/srv/[^\s\"']+", "<path>", text)
    text = re.sub(r"\b(?:sk|ghp|github_pat|xox[baprs])-?[A-Za-z0-9_]{20,}\b", "<redacted-token>", text)
    text = re.sub(r"\b\d{16,20}\b", "<snowflake>", text)
    text = re.sub(r"\b[0-9a-f]{32,64}\b", "<hex-id>", text, flags=re.I)
    text = re.sub(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", "<uuid>", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def build_sanitized_session(path: Path, events: list[dict[str, str]]) -> dict[str, Any]:
    source_hash = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
    session_id = f"oc16-{source_hash}"
    terms = top_terms(events, limit=6)
    length_class = "short" if len(events) < 15 else "medium" if len(events) < 45 else "long"
    header = {
        "role": "system",
        "content": (
            f"Sanitized OpenClaw benchmark transcript marker: {session_id}. "
            f"Source class: openclaw-session. Sanitized turn count: {len(events)}. "
            f"Length class: {length_class}. Topic terms: {', '.join(terms)}."
        ),
    }
    return {
        "session_id": session_id,
        "source_hash": source_hash,
        "source_bytes": path.stat().st_size,
        "source_name": path.name,
        "turn_count": len(events),
        "length_class": length_class,
        "topic_terms": terms,
        "events": [header, *events],
    }


def top_terms(events: list[dict[str, str]], *, limit: int) -> list[str]:
    counts: Counter[str] = Counter()
    for event in events:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", event["content"].lower()):
            if token not in STOPWORDS and not token.startswith("redacted"):
                counts[token] += 1
    terms = [term for term, _ in counts.most_common(limit)]
    fallback = ["session", "memory", "gateway", "benchmark", "model", "transcript"]
    return (terms + fallback)[:limit]


def build_probes(session: dict[str, Any]) -> list[dict[str, Any]]:
    terms = session["topic_terms"]
    return [
        {
            "name": "marker",
            "question": "What is the sanitized benchmark transcript marker for this replayed session?",
            "expected": [session["session_id"]],
        },
        {
            "name": "source_class",
            "question": "What source class was stated in the replayed sanitized transcript header?",
            "expected": ["openclaw-session"],
        },
        {
            "name": "turn_count",
            "question": "What sanitized turn count was stated in the replayed transcript header?",
            "expected": [str(session["turn_count"])],
        },
        {
            "name": "length_class",
            "question": "What length class was stated for the replayed transcript?",
            "expected": [session["length_class"]],
        },
        {
            "name": "topic_primary",
            "question": "Name one primary topic term from the replayed transcript header.",
            "expected": [terms[0]],
        },
        {
            "name": "topic_pair",
            "question": "Name two topic terms from the replayed transcript header.",
            "expected": terms[:2],
        },
        {
            "name": "topic_later",
            "question": "What later topic term was listed in the replayed transcript header?",
            "expected": [terms[min(3, len(terms) - 1)]],
        },
        {
            "name": "sanitized_kind",
            "question": "Was this benchmark input raw private data or a sanitized transcript?",
            "expected": ["sanitized"],
        },
    ]


def run_probe_condition(
    *,
    condition: str,
    probes: list[Probe],
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
                        "Answer using only replayed session history if available. "
                        "If the answer was not provided, say it is not available.\n"
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
                "condition": condition,
                "probe": probe.name,
                "question": probe.question,
                "expected": probe.expected,
                "output": output,
                "score": score,
                "passed": score["score"] >= PASS_THRESHOLD,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        )
    return records


def summarize_session_result(
    session: dict[str, Any],
    base_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    replay_records: list[dict[str, Any]],
) -> dict[str, Any]:
    base = mean_score(base_rows)
    replay = mean_score(replay_rows)
    return {
        "session_id": session["session_id"],
        "turn_count": session["turn_count"],
        "topic_terms": session["topic_terms"],
        "base_score": base,
        "replay_score": replay,
        "delta": round(replay - base, 4),
        "base_passed": sum(1 for row in base_rows if row["passed"]),
        "replay_passed": sum(1 for row in replay_rows if row["passed"]),
        "replay_chunks": len(replay_records),
        "replay_latency_ms": round(sum(row["elapsed_ms"] for row in replay_records), 3),
    }


def summarize_sessions_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    session_ids = sorted({row["session_id"] for row in rows})
    summaries = []
    for session_id in session_ids:
        base = [row for row in rows if row["session_id"] == session_id and row["condition"] == "base_no_history"]
        replay = [row for row in rows if row["session_id"] == session_id and row["condition"] == "replayed_history"]
        summaries.append(
            {
                "session_id": session_id,
                "base_score": mean_score(base),
                "replay_score": mean_score(replay),
                "delta": round(mean_score(replay) - mean_score(base), 4),
                "base_passed": sum(1 for row in base if row["passed"]),
                "replay_passed": sum(1 for row in replay if row["passed"]),
            }
        )
    return summaries


def summarize_all(rows: list[dict[str, Any]], session_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    base_rows = [row for row in rows if row["condition"] == "base_no_history"]
    replay_rows = [row for row in rows if row["condition"] == "replayed_history"]
    deltas = [item["delta"] for item in session_summaries]
    ci_low, ci_high = mean_ci(deltas)
    return {
        "transcripts": len(session_summaries),
        "probes_per_transcript": int(len(base_rows) / len(session_summaries)) if session_summaries else 0,
        "base_score_mean": mean_score(base_rows),
        "replay_score_mean": mean_score(replay_rows),
        "score_delta_mean": round(mean_score(replay_rows) - mean_score(base_rows), 4),
        "delta_95ci": [ci_low, ci_high],
        "base_pass_rate": pass_rate(base_rows),
        "replay_pass_rate": pass_rate(replay_rows),
        "win_rate": round(sum(1 for delta in deltas if delta > 0) / len(deltas), 4) if deltas else 0.0,
        "base_latency_ms_mean": mean_latency(base_rows),
        "replay_latency_ms_mean": mean_latency(replay_rows),
        "latency_ratio_replay_over_base": safe_ratio(mean_latency(replay_rows), mean_latency(base_rows)),
        "sessions": session_summaries,
    }


def mean_score(rows: list[dict[str, Any]]) -> float:
    return round(sum(row["score"]["score"] for row in rows) / len(rows), 4) if rows else 0.0


def pass_rate(rows: list[dict[str, Any]]) -> float:
    return round(sum(1 for row in rows if row["passed"]) / len(rows), 4) if rows else 0.0


def mean_latency(rows: list[dict[str, Any]]) -> float:
    return round(sum(row["elapsed_ms"] for row in rows) / len(rows), 3) if rows else 0.0


def safe_ratio(a: float, b: float) -> float:
    return round(a / b, 4) if b else 0.0


def mean_ci(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = statistics.fmean(values)
    if len(values) < 2:
        return round(mean, 4), round(mean, 4)
    stderr = statistics.stdev(values) / math.sqrt(len(values))
    return round(mean - 1.96 * stderr, 4), round(mean + 1.96 * stderr, 4)


def write_bar_svg(path: Path, summary: dict[str, Any]) -> None:
    bars = [
        ("Base score", summary["base_score_mean"], "#78716c"),
        ("Replay score", summary["replay_score_mean"], "#2563eb"),
        ("Base pass", summary["base_pass_rate"], "#a8a29e"),
        ("Replay pass", summary["replay_pass_rate"], "#16a34a"),
    ]
    width, height = 760, 420
    max_bar = 260
    parts = [svg_header(width, height), '<text x="24" y="34" class="title">OpenClaw 16 Transcript Replay Summary</text>']
    for index, (label, value, color) in enumerate(bars):
        y = 80 + index * 72
        w = max(2, value * max_bar)
        parts.append(f'<text x="24" y="{y + 20}" class="label">{escape_xml(label)}</text>')
        parts.append(f'<rect x="180" y="{y}" width="{w:.1f}" height="32" fill="{color}" rx="4"/>')
        parts.append(f'<text x="{190 + w:.1f}" y="{y + 22}" class="value">{value:.3f}</text>')
    parts.append(svg_footer())
    path.write_text("\n".join(parts), encoding="utf-8")


def write_delta_svg(path: Path, sessions: list[dict[str, Any]]) -> None:
    width, height = 900, 420
    plot_x, plot_y, plot_w, plot_h = 70, 60, 780, 280
    parts = [svg_header(width, height), '<text x="24" y="34" class="title">Paired Per-Transcript Score Deltas</text>']
    parts.append(f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="#444"/>')
    if sessions:
        step = plot_w / max(1, len(sessions) - 1)
        for index, session in enumerate(sessions):
            x = plot_x + index * step
            base_y = plot_y + plot_h - session["base_score"] * plot_h
            replay_y = plot_y + plot_h - session["replay_score"] * plot_h
            color = "#16a34a" if session["delta"] > 0 else "#dc2626" if session["delta"] < 0 else "#78716c"
            parts.append(f'<line x1="{x:.1f}" y1="{base_y:.1f}" x2="{x:.1f}" y2="{replay_y:.1f}" stroke="{color}" stroke-width="2"/>')
            parts.append(f'<circle cx="{x:.1f}" cy="{base_y:.1f}" r="4" fill="#78716c"/>')
            parts.append(f'<circle cx="{x:.1f}" cy="{replay_y:.1f}" r="4" fill="{color}"/>')
    parts.append('<text x="70" y="380" class="label">gray = no-history base, color = replayed history</text>')
    parts.append(svg_footer())
    path.write_text("\n".join(parts), encoding="utf-8")


def render_markdown_report(summary: dict[str, Any], bars: Path, deltas: Path) -> str:
    return textwrap.dedent(
        f"""\
        # OpenClaw 16 Transcript Replay Benchmark

        Sanitized transcripts: `{summary['transcripts']}`
        Probes per transcript: `{summary['probes_per_transcript']}`

        | Metric | No-history base | Replayed history |
        | --- | ---: | ---: |
        | Mean score | `{summary['base_score_mean']:.4f}` | `{summary['replay_score_mean']:.4f}` |
        | Pass rate | `{summary['base_pass_rate']:.4f}` | `{summary['replay_pass_rate']:.4f}` |
        | Mean probe latency | `{summary['base_latency_ms_mean']:.1f} ms` | `{summary['replay_latency_ms_mean']:.1f} ms` |

        Mean paired score delta: `{summary['score_delta_mean']:.4f}` with rough 95% CI `{summary['delta_95ci'][0]:.4f}` to `{summary['delta_95ci'][1]:.4f}`.
        Win rate: `{summary['win_rate']:.4f}`.
        Latency ratio replay/base: `{summary['latency_ratio_replay_over_base']:.2f}x`.

        Graphs:

        - `{bars}`
        - `{deltas}`
        """
    )


def svg_header(width: int, height: int) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #1c1917; }}
.title {{ font-size: 22px; font-weight: 700; }}
.label {{ font-size: 14px; }}
.value {{ font-size: 14px; font-weight: 600; }}
</style>
<rect width="100%" height="100%" fill="#fafaf9"/>"""


def svg_footer() -> str:
    return "</svg>"


def escape_xml(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def write_events_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    write_jsonl(path, records)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
