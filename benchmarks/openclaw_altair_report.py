from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Altair charts for OpenClaw transcript replay results.")
    parser.add_argument("--results-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--sessions-jsonl", type=Path, required=True)
    parser.add_argument("--benchmark-note", type=Path, required=True)
    parser.add_argument(
        "--legacy-results-dir",
        type=Path,
        help="Directory containing older paired *-plain.json/*-delta.json benchmark outputs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        import altair as alt
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Install report dependencies first: python3 -m pip install --user altair pandas vl-convert-python") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.results_jsonl)
    sessions = read_jsonl(args.sessions_jsonl)
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    local_summary = parse_benchmark_note(args.benchmark_note)

    results_df = pd.DataFrame(flatten_result(row) for row in rows)
    sessions_df = pd.DataFrame(sessions)
    session_df = build_session_frame(pd, results_df, sessions_df)
    probe_df = build_probe_frame(pd, results_df)
    paper_df = build_paper_local_frame(pd, summary, local_summary)
    timeline_df = build_timeline_frame(pd, summary, local_summary)
    legacy_results_dir = args.legacy_results_dir or args.summary_json.parents[2]
    cost_df = build_cost_frame(pd, summary, legacy_results_dir)
    gallery_df = build_error_gallery_frame(pd, results_df)

    alt.data_transformers.disable_max_rows()
    theme = {
        "config": {
            "background": "#fbfaf8",
            "title": {"font": "Inter, -apple-system, BlinkMacSystemFont, sans-serif", "fontSize": 18, "anchor": "start"},
            "axis": {
                "labelFont": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
                "titleFont": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
                "gridColor": "#e7e5e4",
                "domainColor": "#a8a29e",
            },
            "legend": {
                "labelFont": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
                "titleFont": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            },
            "view": {"stroke": "transparent"},
        }
    }
    alt.themes.register("delta_mem_report", lambda: theme)
    alt.themes.enable("delta_mem_report")

    charts = {
        "01-paired-slope": paired_slope_chart(alt, session_df),
        "02-delta-strip": delta_strip_chart(alt, session_df, summary),
        "03-probe-heatmap": probe_heatmap(alt, results_df),
        "04-pass-contribution": pass_contribution(alt, probe_df),
        "05-length-vs-delta": length_vs_delta(alt, session_df),
        "06-latency-vs-accuracy": latency_vs_accuracy(alt, session_df),
        "07-summary-card": summary_card(alt, pd, summary),
        "08-small-multiples": small_multiples(alt, session_df),
        "09-paper-local-lift-ladder": paper_local_lift_ladder(alt, paper_df),
        "10-normalized-score-recovery": normalized_score_recovery(alt, paper_df),
        "11-probe-type-recovery": probe_type_recovery(alt, probe_df),
        "12-wins-ties-losses": wins_ties_losses(alt, pd, session_df),
        "13-top-five-gains": top_five_gains(alt, session_df),
        "14-error-gallery": error_gallery(alt, gallery_df),
        "15-local-result-timeline": local_result_timeline(alt, timeline_df),
        "16-cost-of-memory-signal": cost_of_memory_signal(alt, cost_df),
        "17-research-vs-non-openclaw": research_vs_non_openclaw(alt, paper_df),
    }

    manifest = []
    for name, chart in charts.items():
        html_path = args.output_dir / f"{name}.html"
        png_path = args.output_dir / f"{name}.png"
        chart.save(html_path)
        png_status = "not_written"
        try:
            chart.save(png_path, scale_factor=2)
            png_status = "written"
        except Exception as exc:  # Altair export depends on vl-convert.
            png_status = f"failed: {exc}"
        manifest.append({"name": name, "html": str(html_path), "png": str(png_path), "png_status": png_status})

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "index.md").write_text(render_index(manifest, summary), encoding="utf-8")
    print(json.dumps({"charts": len(manifest), "output_dir": str(args.output_dir)}, indent=2))
    return 0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def flatten_result(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": row["session_id"],
        "condition": row["condition"],
        "probe": row["probe"],
        "question": row["question"],
        "expected": ", ".join(row["expected"]),
        "output": row["output"],
        "score": row["score"]["score"],
        "required_evidence": row["score"]["required_evidence"],
        "passed": bool(row["passed"]),
        "elapsed_ms": row["elapsed_ms"],
        "topic_terms": ", ".join(row.get("topic_terms", [])),
    }


def build_session_frame(pd: Any, results_df: Any, sessions_df: Any) -> Any:
    grouped = (
        results_df.groupby(["session_id", "condition"], as_index=False)
        .agg(score=("score", "mean"), pass_count=("passed", "sum"), latency_ms=("elapsed_ms", "mean"))
    )
    pivot = grouped.pivot(index="session_id", columns="condition", values=["score", "pass_count", "latency_ms"])
    pivot.columns = ["_".join(col).strip("_") for col in pivot.columns.to_flat_index()]
    frame = pivot.reset_index()
    frame["delta"] = frame.get("score_replayed_history", 0) - frame.get("score_base_no_history", 0)
    frame["pass_delta"] = frame.get("pass_count_replayed_history", 0) - frame.get("pass_count_base_no_history", 0)
    frame["latency_ratio"] = frame.get("latency_ms_replayed_history", 0) / frame.get("latency_ms_base_no_history", 1)
    frame["outcome"] = frame["delta"].map(lambda value: "win" if value > 0 else "loss" if value < 0 else "tie")
    slim_sessions = sessions_df[["session_id", "turn_count", "length_class", "topic_terms"]].copy()
    slim_sessions["topic_label"] = slim_sessions["topic_terms"].map(lambda terms: ", ".join(terms[:3]))
    frame = frame.merge(slim_sessions, on="session_id", how="left")
    frame["short_id"] = frame["session_id"].str.replace("oc16-", "", regex=False).str[:6]
    return frame.sort_values(["delta", "session_id"], ascending=[False, True])


def build_probe_frame(pd: Any, results_df: Any) -> Any:
    grouped = (
        results_df.groupby(["probe", "condition"], as_index=False)
        .agg(score=("score", "mean"), pass_rate=("passed", "mean"), latency_ms=("elapsed_ms", "mean"))
    )
    pivot = grouped.pivot(index="probe", columns="condition", values=["score", "pass_rate", "latency_ms"])
    pivot.columns = ["_".join(col).strip("_") for col in pivot.columns.to_flat_index()]
    frame = pivot.reset_index()
    frame["score_delta"] = frame.get("score_replayed_history", 0) - frame.get("score_base_no_history", 0)
    frame["pass_delta"] = frame.get("pass_rate_replayed_history", 0) - frame.get("pass_rate_base_no_history", 0)
    return frame.sort_values("score_delta", ascending=False)


def parse_benchmark_note(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    values = {
        "paper_average": 1.10,
        "paper_memoryagentbench": 1.31,
        "paper_locomo": 1.20,
        "fixed_locomo": 1.07,
        "sanitized_raw": 1.17,
        "hybrid": 1.10,
        "qmd_search": 1.30,
        "qmd_vsearch": 1.30,
        "qmd_deterministic_500": 1.30,
        "qmd_synthesized_100": 1.30,
    }
    return values


def build_paper_local_frame(pd: Any, summary: dict[str, Any], local: dict[str, float]) -> Any:
    rows = [
        ("Paper average", "paper", local["paper_average"], None, "paper reported"),
        ("Paper MemoryAgentBench", "paper", local["paper_memoryagentbench"], None, "paper reported"),
        ("Paper LoCoMo", "paper", local["paper_locomo"], None, "paper reported"),
        ("Local synthetic paper-style", "non-OpenClaw", 1.00, 0.5129, "flat local probes"),
        ("Local LoCoMo no-context", "non-OpenClaw", 3.67, 0.1833, "bad baseline, deprecated"),
        ("Local fixed LoCoMo", "non-OpenClaw", local["fixed_locomo"], 0.5000, "corrected local sample"),
        ("OpenClaw raw replay", "OpenClaw older", local["sanitized_raw"], 0.6667, "older lenient scorer"),
        ("OpenClaw QMD search", "OpenClaw older", local["qmd_search"], 0.7292, "older lenient scorer"),
        ("OpenClaw QMD vsearch", "OpenClaw older", local["qmd_vsearch"], 0.7292, "older lenient scorer"),
        ("OpenClaw-16 replay", "OpenClaw strict", None, summary["replay_score_mean"], "strict scorer"),
    ]
    return pd.DataFrame(rows, columns=["label", "source", "lift", "normalized_score", "note"])


def build_timeline_frame(pd: Any, summary: dict[str, Any], local: dict[str, float]) -> Any:
    rows = [
        (1, "Paper-style synthetic", 1.00, 0.0, "flat"),
        (2, "Fixed LoCoMo-10", local["fixed_locomo"], 0.5000, "small positive"),
        (3, "Raw OpenClaw replay", local["sanitized_raw"], 0.6667, "older scorer"),
        (4, "Hybrid memory + replay", local["hybrid"], 0.6208, "older scorer"),
        (5, "QMD search", local["qmd_search"], 0.7292, "older scorer"),
        (6, "QMD vsearch", local["qmd_vsearch"], 0.7292, "older scorer"),
        (7, "OpenClaw-16 strict", None, summary["replay_score_mean"], "current strict"),
    ]
    return pd.DataFrame(rows, columns=["order", "label", "lift", "normalized_score", "note"])


def build_cost_frame(pd: Any, summary: dict[str, Any], legacy_results_dir: Path) -> Any:
    rows = []
    rows.extend(load_research_claim_rows(legacy_results_dir))
    rows.extend(load_paired_cost_rows(legacy_results_dir))
    rows.append(
        {
            "label": "OpenClaw strict base",
            "latency_ratio": summary["latency_ratio_replay_over_base"],
            "score_delta": summary["score_delta_mean"],
            "kind": "OpenClaw strict scorer",
            "bench": "OpenClaw n16",
            "run_n": f"n{int(summary.get('transcripts', len(summary.get('sessions', [])) or 16))}",
            "source": "benchmarks/results/openclaw-16/report/summary.json",
        }
    )
    frame = pd.DataFrame(rows)
    frame["point_id"] = [f"a{index}" for index in range(1, len(frame) + 1)]
    frame["bench"] = frame["bench"].fillna(frame["label"].map(bench_family))
    frame["run_n"] = frame["run_n"].fillna(frame["label"].map(run_count_label))
    frame["latency_label"] = frame.apply(
        lambda row: "n/a" if row["kind"] == "research claimed" else f"{row['latency_ratio']:.2f}x",
        axis=1,
    )
    frame["quality_label"] = frame["score_delta"].map(lambda value: f"{value:+.3f}")
    frame["guide"] = frame["point_id"] + "  " + frame["label"] + " · " + frame["bench"]
    frame["guide_order"] = range(1, len(frame) + 1)
    return frame


def load_research_claim_rows(results_dir: Path) -> list[dict[str, Any]]:
    path = results_dir / "research-claimed-results.jsonl"
    if not path.exists():
        path = REPO_ROOT / "benchmarks/fixtures/research-claimed-results.jsonl"
    if not path.exists():
        return []
    rows = []
    for record in read_jsonl(path):
        rows.append(
            {
                "label": str(record["label"]),
                "latency_ratio": float(record["latency_ratio"]),
                "score_delta": float(record["response_delta"]),
                "kind": str(record.get("kind", "research claimed")),
                "bench": str(record.get("bench", "research")),
                "run_n": str(record.get("run_n", "paper")),
                "source": str(path),
            }
        )
    return rows


def load_paired_cost_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    delta_paths = sorted(results_dir.glob("*-delta*.json"))
    for delta_path in delta_paths:
        plain_path = Path(str(delta_path).replace("-delta", "-plain"))
        if not plain_path.exists():
            continue
        delta = json.loads(delta_path.read_text(encoding="utf-8"))
        plain = json.loads(plain_path.read_text(encoding="utf-8"))
        label = label_from_result_path(delta_path)
        if "smoke" in label.lower():
            continue
        if label in {"QMD deterministic 200", "QMD deterministic 400", "QMD synthesized 200", "QMD synthesized 400"}:
            continue
        plain_score = paired_score(plain)
        delta_score = paired_score(delta)
        if plain_score is None or delta_score is None:
            continue
        plain_latency = paired_latency(plain)
        delta_latency = paired_latency(delta)
        latency_ratio = round(delta_latency / plain_latency, 2) if plain_latency else 0.0
        score_delta = round(delta_score - plain_score, 4)
        if "-strict-delta" in delta_path.stem and is_openclaw_label(label):
            kind = "OpenClaw current strict run"
        elif is_openclaw_label(label):
            kind = "OpenClaw strict-rescored"
        else:
            kind = "non-OpenClaw"
        rows.append(
            {
                "label": label,
                "latency_ratio": latency_ratio,
                "score_delta": score_delta,
                "kind": kind,
                "bench": bench_family(label),
                "run_n": run_count_from_result(label, delta),
                "source": str(delta_path),
            }
        )
    return rows


def paired_score(result: dict[str, Any]) -> float | None:
    if result.get("kind") == "memoryagentbench_sidecar_eval":
        metrics = ((result.get("summary") or {}).get("averaged_metrics") or {})
        value = metrics.get("exact_match")
        return float(value) if value is not None else None
    probes = result.get("probes")
    if isinstance(probes, list) and probes:
        from benchmarks.openclaw_session_replay_eval import score_output

        return round(sum(score_output(row.get("output", ""), row.get("expected", []))["score"] for row in probes) / len(probes), 4)
    summary = result.get("summary") or {}
    value = summary.get("score_mean")
    return float(value) if value is not None else None


def paired_latency(result: dict[str, Any]) -> float:
    if result.get("kind") == "memoryagentbench_sidecar_eval":
        metrics = ((result.get("summary") or {}).get("averaged_metrics") or {})
        return float(metrics.get("query_time_len") or result.get("summary", {}).get("elapsed_ms") or 0)
    summary = result.get("summary") or {}
    return float(summary.get("probe_latency_ms_mean") or summary.get("elapsed_ms") or 0)


def is_openclaw_label(label: str) -> bool:
    return label.startswith(("OpenClaw", "QMD", "Ygraph", "Memorg", "Hybrid", "Related", "Wiki", "Deterministic"))


def bench_family(label: str) -> str:
    if label.startswith("Paper claimed"):
        return "research"
    if label.startswith("MemoryAgentBench"):
        return "MemoryAgentBench"
    if "LoCoMo" in label:
        return "LoCoMo"
    if label.startswith("OpenClaw n16"):
        return "OpenClaw n16"
    if is_openclaw_label(label):
        return "OpenClaw replay"
    return "local"


def run_count_label(label: str) -> str:
    if label.startswith("Paper claimed"):
        return "paper"
    if label.startswith("MemoryAgentBench"):
        return "n100"
    if "LoCoMo-10" in label:
        return "n10"
    if label.startswith("OpenClaw n16"):
        return "n16"
    if is_openclaw_label(label):
        return "n1"
    return "n1"


def run_count_from_result(label: str, result: dict[str, Any]) -> str:
    summary = result.get("summary") or {}
    if result.get("kind") == "memoryagentbench_sidecar_eval":
        queries = summary.get("queries")
        return f"n{int(queries)}" if queries is not None else run_count_label(label)
    if "records" in summary:
        return f"n{int(summary['records'])}"
    if "probes" in summary:
        replay_chunks = summary.get("replay_chunks")
        if replay_chunks is not None:
            return f"n{int(replay_chunks)}"
        return f"n{int(summary['probes'])}"
    return run_count_label(label)


def label_from_result_path(path: Path) -> str:
    stem = path.stem
    stem = stem.replace("-strict", "")
    stem = stem.replace("-delta-context", "-context")
    stem = stem.replace("-delta", "")
    labels = {
        "locomo10-sidecar-context": "Fixed LoCoMo-10",
        "locomo10-sidecar": "LoCoMo-10 no-context",
        "memoryagentbench-factconsolidation-sh-6k": "MemoryAgentBench CR full",
        "memoryagentbench-factconsolidation-sh-6k-smoke": "MemoryAgentBench CR smoke",
        "openclaw-deterministic-context": "Deterministic lexical preload",
        "openclaw-" + "forever" + "green": "Wiki preload",
        "openclaw-hybrid": "Hybrid memory + half replay",
        "openclaw-memorg": "Memorg only",
        "openclaw-qmd-search": "QMD search",
        "openclaw-qmd-vsearch": "QMD vsearch",
        "openclaw-related-preload": "Related sessions preload",
        "openclaw-ygraph-keywords": "Ygraph keyword thoughts",
    }
    if stem in labels:
        return labels[stem]
    prefix = "openclaw-qmd-deterministic-"
    if stem.startswith(prefix) and stem.endswith("tok"):
        return f"QMD deterministic {stem.removeprefix(prefix).removesuffix('tok')}"
    prefix = "openclaw-qmd-synthesized-"
    if stem.startswith(prefix) and stem.endswith("tok"):
        return f"QMD synthesized {stem.removeprefix(prefix).removesuffix('tok')}"
    return stem.replace("-", " ").title()


def build_error_gallery_frame(pd: Any, results_df: Any) -> Any:
    failures = results_df[~results_df["passed"]].copy()
    failures["output_short"] = failures["output"].str.slice(0, 90)
    failures["expected_short"] = failures["expected"].str.slice(0, 60)
    return failures.sort_values(["condition", "score"]).head(18)


def paired_slope_chart(alt: Any, session_df: Any) -> Any:
    long_df = session_df.melt(
        id_vars=["session_id", "short_id", "delta", "topic_label"],
        value_vars=["score_base_no_history", "score_replayed_history"],
        var_name="condition",
        value_name="score",
    )
    long_df["condition_label"] = long_df["condition"].map(
        {"score_base_no_history": "No-history base", "score_replayed_history": "Replayed history"}
    )
    return (
        alt.Chart(long_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("condition_label:N", title=None, sort=["No-history base", "Replayed history"]),
            y=alt.Y("score:Q", title="Mean score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("delta:Q", title="Delta", scale=alt.Scale(scheme="bluegreen")),
            detail="session_id:N",
            tooltip=["session_id", "topic_label", "delta", "score"],
        )
        .properties(width=560, height=360, title="1. Paired Slope: Base vs Replayed History")
    )


def delta_strip_chart(alt: Any, session_df: Any, summary: dict[str, Any]) -> Any:
    plot_df = session_df.copy()
    plot_df["jitter"] = [0.35 + ((index % 7) * 0.05) for index in range(len(plot_df))]
    points = alt.Chart(plot_df).mark_circle(size=90, opacity=0.8).encode(
        x=alt.X("delta:Q", title="Replay score minus base score"),
        y=alt.Y("jitter:Q", title=None, axis=None),
        color=alt.Color("outcome:N", scale=alt.Scale(domain=["win", "tie", "loss"], range=["#16a34a", "#78716c", "#dc2626"])),
        tooltip=["session_id", "delta", "topic_label"],
    )
    mean_rule = alt.Chart({"values": [{"x": summary["score_delta_mean"], "y0": 0.2, "y1": 0.8}]}).mark_rule(
        color="#1d4ed8", strokeWidth=3
    ).encode(x="x:Q", y="y0:Q", y2="y1:Q")
    ci = alt.Chart(
        {"values": [{"lo": summary["delta_95ci"][0], "hi": summary["delta_95ci"][1], "y0": 0.45, "y1": 0.55}]}
    ).mark_rect(color="#60a5fa", opacity=0.45).encode(x="lo:Q", x2="hi:Q", y="y0:Q", y2="y1:Q")
    return (ci + points + mean_rule).properties(width=620, height=160, title="2. Delta Strip Plot with Mean and 95% CI")


def probe_heatmap(alt: Any, results_df: Any) -> Any:
    replay = results_df[results_df["condition"] == "replayed_history"].copy()
    replay["short_id"] = replay["session_id"].str.replace("oc16-", "", regex=False).str[:6]
    return alt.Chart(replay).mark_rect().encode(
        x=alt.X("probe:N", title="Probe"),
        y=alt.Y("short_id:N", title="Transcript"),
        color=alt.Color("score:Q", title="Score", scale=alt.Scale(scheme="blues", domain=[0, 1])),
        tooltip=["session_id", "probe", "score", "passed", "expected"],
    ).properties(width=700, height=420, title="3. Probe Recall Heatmap")


def pass_contribution(alt: Any, probe_df: Any) -> Any:
    long_df = probe_df.melt(
        id_vars=["probe"],
        value_vars=["pass_rate_base_no_history", "pass_rate_replayed_history"],
        var_name="condition",
        value_name="pass_rate",
    )
    long_df["condition_label"] = long_df["condition"].map(
        {"pass_rate_base_no_history": "Base", "pass_rate_replayed_history": "Replay"}
    )
    return alt.Chart(long_df).mark_bar().encode(
        x=alt.X("probe:N", sort="-y", title="Probe"),
        y=alt.Y("pass_rate:Q", title="Pass rate", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("condition_label:N", title="Condition"),
        xOffset="condition_label:N",
        tooltip=["probe", "condition_label", "pass_rate"],
    ).properties(width=720, height=340, title="4. Pass Contribution by Probe")


def length_vs_delta(alt: Any, session_df: Any) -> Any:
    return alt.Chart(session_df).mark_circle(size=120, opacity=0.8).encode(
        x=alt.X("turn_count:Q", title="Sanitized turn count"),
        y=alt.Y("delta:Q", title="Score delta"),
        color=alt.Color("outcome:N", title="Outcome"),
        size=alt.Size("pass_count_replayed_history:Q", title="Replay passes"),
        tooltip=["session_id", "turn_count", "delta", "topic_label"],
    ).properties(width=620, height=360, title="5. Session Length vs Delta")


def latency_vs_accuracy(alt: Any, session_df: Any) -> Any:
    return alt.Chart(session_df).mark_circle(size=120, opacity=0.8).encode(
        x=alt.X("latency_ms_replayed_history:Q", title="Replay probe latency mean (ms)"),
        y=alt.Y("score_replayed_history:Q", title="Replay score", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("delta:Q", title="Delta", scale=alt.Scale(scheme="tealblues")),
        size=alt.Size("turn_count:Q", title="Turns"),
        tooltip=["session_id", "score_replayed_history", "latency_ms_replayed_history", "turn_count", "topic_label"],
    ).properties(width=620, height=360, title="6. Latency vs Accuracy")


def summary_card(alt: Any, pd: Any, summary: dict[str, Any]) -> Any:
    rows = [
        ("Base score", f"{summary['base_score_mean']:.4f}", 1),
        ("Replay score", f"{summary['replay_score_mean']:.4f}", 2),
        ("Delta", f"{summary['score_delta_mean']:+.4f}", 3),
        ("95% CI", f"{summary['delta_95ci'][0]:+.4f} to {summary['delta_95ci'][1]:+.4f}", 4),
        ("Win rate", f"{summary['win_rate']:.4f}", 5),
        ("Latency ratio", f"{summary['latency_ratio_replay_over_base']:.2f}x", 6),
    ]
    df = pd.DataFrame(rows, columns=["metric", "value", "order"])
    metric = alt.Chart(df).mark_text(align="right", fontSize=18, fontWeight="bold", color="#57534e").encode(
        y=alt.Y("order:O", axis=None),
        text="metric",
    )
    value = alt.Chart(df).mark_text(align="left", fontSize=22, fontWeight="bold", color="#1d4ed8").encode(
        y=alt.Y("order:O", axis=None),
        text="value",
    )
    return (metric.encode(x=alt.value(210)) + value.encode(x=alt.value(235))).properties(
        width=520, height=280, title="7. OpenClaw-16 Summary Card"
    )


def small_multiples(alt: Any, session_df: Any) -> Any:
    long_df = session_df.melt(
        id_vars=["session_id", "short_id", "topic_label"],
        value_vars=["score_base_no_history", "score_replayed_history"],
        var_name="condition",
        value_name="score",
    )
    long_df["condition_label"] = long_df["condition"].map(
        {"score_base_no_history": "Base", "score_replayed_history": "Replay"}
    )
    return alt.Chart(long_df).mark_bar().encode(
        x=alt.X("condition_label:N", title=None),
        y=alt.Y("score:Q", title=None, scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("condition_label:N", legend=None),
        tooltip=["session_id", "score", "topic_label"],
    ).properties(width=120, height=90).facet(
        column=alt.Column("short_id:N", title="Transcript"),
        columns=4,
    ).properties(title="8. Small Multiples by Transcript")


def paper_local_lift_ladder(alt: Any, paper_df: Any) -> Any:
    lift = paper_df[paper_df["lift"].notna()].copy()
    return alt.Chart(lift).mark_bar().encode(
        x=alt.X("lift:Q", title="Normalized lift over base (base = 1.00x)"),
        y=alt.Y("label:N", sort="-x", title=None),
        color=alt.Color("source:N", title="Source"),
        tooltip=["label", "lift", "note"],
    ).properties(width=640, height=360, title="9. Normalized Paper vs Local Lift Ladder")


def normalized_score_recovery(alt: Any, paper_df: Any) -> Any:
    score = paper_df[paper_df["normalized_score"].notna()].copy()
    return alt.Chart(score).mark_bar().encode(
        x=alt.X("normalized_score:Q", title="Score on 0..1 scorer ceiling", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("label:N", sort="-x", title=None),
        color=alt.Color("note:N", title="Caveat"),
        tooltip=["label", "normalized_score", "note"],
    ).properties(width=640, height=320, title="10. Normalized Score Recovery")


def probe_type_recovery(alt: Any, probe_df: Any) -> Any:
    return alt.Chart(probe_df).mark_bar().encode(
        x=alt.X("score_delta:Q", title="Replay minus base score"),
        y=alt.Y("probe:N", sort="-x", title="Probe"),
        color=alt.Color("score_delta:Q", legend=None, scale=alt.Scale(scheme="bluegreen")),
        tooltip=["probe", "score_delta", "score_replayed_history", "pass_rate_replayed_history"],
    ).properties(width=620, height=340, title="11. Probe-Type Recovery Ranking")


def wins_ties_losses(alt: Any, pd: Any, session_df: Any) -> Any:
    counts = session_df.groupby("outcome", as_index=False).agg(count=("session_id", "count"))
    return alt.Chart(counts).mark_arc(innerRadius=70).encode(
        theta="count:Q",
        color=alt.Color("outcome:N", scale=alt.Scale(domain=["win", "tie", "loss"], range=["#16a34a", "#78716c", "#dc2626"])),
        tooltip=["outcome", "count"],
    ).properties(width=360, height=320, title="12. Replay Wins, Ties, Losses")


def top_five_gains(alt: Any, session_df: Any) -> Any:
    top = session_df.sort_values("delta", ascending=False).head(5)
    return alt.Chart(top).mark_bar().encode(
        x=alt.X("delta:Q", title="Score delta", scale=alt.Scale(domain=[0, max(0.15, float(top["delta"].max()))])),
        y=alt.Y("short_id:N", sort="-x", title="Transcript"),
        color=alt.Color("turn_count:Q", title="Turns", scale=alt.Scale(scheme="viridis")),
        tooltip=["session_id", "delta", "turn_count", "topic_label"],
    ).properties(width=620, height=300, title="13. Top-5 Transcript Gains")


def error_gallery(alt: Any, gallery_df: Any) -> Any:
    return alt.Chart(gallery_df).mark_text(align="left", baseline="middle", fontSize=11).encode(
        y=alt.Y("probe:N", title="Probe"),
        x=alt.X("condition:N", title="Condition"),
        text="output_short:N",
        color=alt.Color("score:Q", title="Score", scale=alt.Scale(scheme="reds", reverse=True)),
        tooltip=["session_id", "probe", "condition", "expected_short", "output_short"],
    ).properties(width=760, height=380, title="14. False-Recovery / Error Gallery")


def local_result_timeline(alt: Any, timeline_df: Any) -> Any:
    return alt.Chart(timeline_df).mark_line(point=True).encode(
        x=alt.X("order:O", title="Run order"),
        y=alt.Y("normalized_score:Q", title="Normalized score", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("note:N", title="Caveat"),
        tooltip=["label", "normalized_score", "lift", "note"],
    ).properties(width=680, height=340, title="15. Local Result Timeline")


def cost_of_memory_signal(alt: Any, cost_df: Any) -> Any:
    plot_df = cost_df.copy()
    plot_df["_cluster"] = plot_df["latency_ratio"].round(2).astype(str) + "|" + plot_df["score_delta"].round(4).astype(str)
    plot_df["_cluster_index"] = plot_df.groupby("_cluster").cumcount()
    plot_df["_cluster_size"] = plot_df.groupby("_cluster")["_cluster"].transform("count")
    plot_df["score_delta_plot"] = plot_df.apply(
        lambda row: row["score_delta"] + ((row["_cluster_index"] - ((row["_cluster_size"] - 1) / 2)) * 0.006),
        axis=1,
    )
    x_min = 0.9 if float(plot_df["latency_ratio"].min()) >= 0.9 else float(plot_df["latency_ratio"].min())
    x_max = min(2.0, round(float(plot_df["latency_ratio"].max()) + 0.08, 2))
    y_min = round(float(plot_df["score_delta_plot"].min()) - 0.025, 2)
    y_max = round(float(plot_df["score_delta_plot"].max()) + 0.025, 2)
    x_tick_values = [round(x_min + (index * 0.1), 1) for index in range(int(((x_max - x_min) / 0.1) + 1.5))]
    split_at = (len(plot_df) + 1) // 2
    guide_left = plot_df[plot_df["guide_order"] <= split_at]
    guide_right = plot_df[plot_df["guide_order"] > split_at].copy()
    guide_right["guide_order"] = guide_right["guide_order"] - split_at
    kind_domain = ["research claimed", "non-OpenClaw", "OpenClaw strict-rescored", "OpenClaw current strict run", "OpenClaw strict scorer"]
    kind_range = ["#7c3aed", "#78716c", "#2563eb", "#16a34a", "#dc2626"]
    kind_color = alt.Color("kind:N", title="Result kind", scale=alt.Scale(domain=kind_domain, range=kind_range))
    guide_color = alt.Color("kind:N", legend=None, scale=alt.Scale(domain=kind_domain, range=kind_range))
    base = alt.Chart(plot_df).encode(
        x=alt.X(
            "latency_ratio:Q",
            title="Latency ratio",
            scale=alt.Scale(domain=[x_min, x_max]),
            axis=alt.Axis(values=x_tick_values, labelExpr="format(datum.value, '.1f') + 'x'"),
        ),
        y=alt.Y(
            "score_delta_plot:Q",
            title="Response delta",
            scale=alt.Scale(domain=[y_min, y_max]),
            axis=alt.Axis(labelExpr="format(datum.value, '.2f')"),
        ),
    )
    points = base.mark_point(size=150, filled=True, opacity=0.9, stroke="#292524", strokeWidth=0.6).encode(
        color=kind_color,
        shape=alt.Shape("kind:N", title="Result kind", scale=alt.Scale(domain=kind_domain, range=["cross", "circle", "square", "triangle-up", "diamond"])),
        tooltip=["point_id", "label", "bench", "latency_ratio", "score_delta", "kind"],
    )
    labels = base.mark_text(dx=11, dy=-9, fontSize=11, fontWeight="bold", color="#292524").encode(
        text="point_id:N",
        tooltip=["point_id", "label", "bench", "latency_ratio", "score_delta", "kind"],
    )
    legend = color_guide(alt, kind_domain, kind_range)
    guide_header = alt.hconcat(guide_header_panel(alt), guide_header_panel(alt), spacing=4)
    guide_left_chart = guide_panel(alt, guide_left, guide_color)
    guide_right_chart = guide_panel(alt, guide_right, guide_color, "")
    guide = alt.vconcat(legend, guide_header, alt.hconcat(guide_left_chart, guide_right_chart, spacing=4), spacing=2)
    scatter = (points + labels).properties(width=620, height=420, title=f"16. Cost of Response Delta (n={len(plot_df)})")
    return alt.hconcat(scatter, guide).resolve_scale(color="shared")


def color_guide(alt: Any, kind_domain: list[str], kind_range: list[str]) -> Any:
    rows = [
        {"x": 12, "label_x": 24, "kind": "research claimed", "label": "paper claim"},
        {"x": 148, "label_x": 160, "kind": "non-OpenClaw", "label": "research/local"},
        {"x": 300, "label_x": 312, "kind": "OpenClaw strict-rescored", "label": "lenient"},
        {"x": 405, "label_x": 417, "kind": "OpenClaw current strict run", "label": "strict"},
        {"x": 510, "label_x": 522, "kind": "OpenClaw strict scorer", "label": "OpenClaw strict base"},
    ]
    color = alt.Color("kind:N", legend=None, scale=alt.Scale(domain=kind_domain, range=kind_range))
    points = alt.Chart({"values": rows}).mark_point(size=100, filled=True, stroke="#292524", strokeWidth=0.5).encode(
        x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0, 620])),
        y=alt.value(18),
        color=color,
        shape=alt.Shape("kind:N", legend=None, scale=alt.Scale(domain=kind_domain, range=["cross", "circle", "square", "triangle-up", "diamond"])),
    )
    labels = alt.Chart({"values": rows}).mark_text(align="left", baseline="middle", fontSize=10, color="#57534e").encode(
        x=alt.X("label_x:Q", axis=None, scale=alt.Scale(domain=[0, 620])),
        y=alt.value(18),
        text="label:N",
    )
    return (points + labels).properties(width=620, height=34, title="Guide")


def guide_header_panel(alt: Any) -> Any:
    header_rows = [
        {"x": 28, "text": "id", "align": "right"},
        {"x": 64, "text": "n", "align": "right"},
        {"x": 108, "text": "lat", "align": "right"},
        {"x": 158, "text": "delta", "align": "right"},
        {"x": 168, "text": "label", "align": "left"},
    ]
    header_right = alt.Chart({"values": [row for row in header_rows if row["align"] == "right"]}).mark_text(
        align="right",
        baseline="middle",
        fontSize=9,
        fontWeight="bold",
        color="#57534e",
        font="Menlo, Monaco, Consolas, monospace",
    ).encode(
        x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0, 370])),
        y=alt.value(8),
        text="text:N",
    )
    header_left = alt.Chart({"values": [row for row in header_rows if row["align"] == "left"]}).mark_text(
        align="left",
        baseline="middle",
        fontSize=9,
        fontWeight="bold",
        color="#57534e",
        font="Menlo, Monaco, Consolas, monospace",
    ).encode(
        x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0, 370])),
        y=alt.value(8),
        text="text:N",
    )
    return (header_right + header_left).properties(width=370, height=18)


def guide_panel(alt: Any, frame: Any, color: Any, title: str = "") -> Any:
    y = alt.Y("guide_order:O", axis=None, sort="ascending")
    ids = alt.Chart(frame).mark_text(
        align="right",
        baseline="middle",
        fontSize=10,
        fontWeight="bold",
        font="Menlo, Monaco, Consolas, monospace",
    ).encode(
        x=alt.value(28),
        y=y,
        text="point_id:N",
        color=color,
    )
    run_n = alt.Chart(frame).mark_text(
        align="right",
        baseline="middle",
        fontSize=9,
        font="Menlo, Monaco, Consolas, monospace",
        color="#57534e",
    ).encode(
        x=alt.value(64),
        y=y,
        text="run_n:N",
    )
    latency = alt.Chart(frame).mark_text(
        align="right",
        baseline="middle",
        fontSize=9,
        font="Menlo, Monaco, Consolas, monospace",
        color="#57534e",
    ).encode(
        x=alt.value(108),
        y=y,
        text="latency_label:N",
    )
    quality = alt.Chart(frame).mark_text(
        align="right",
        baseline="middle",
        fontSize=9,
        font="Menlo, Monaco, Consolas, monospace",
        color="#57534e",
    ).encode(
        x=alt.value(158),
        y=y,
        text="quality_label:N",
    )
    labels = alt.Chart(frame).mark_text(align="left", baseline="middle", fontSize=10).encode(
        x=alt.value(168),
        y=y,
        text="label:N",
        color=color,
    )
    return (ids + run_n + latency + quality + labels).properties(width=370, height=400, title=title)


def research_vs_non_openclaw(alt: Any, paper_df: Any) -> Any:
    rows = paper_df[paper_df["source"].isin(["paper", "non-OpenClaw"])].copy()
    return alt.Chart(rows).mark_circle(size=180, opacity=0.85).encode(
        x=alt.X("lift:Q", title="Accuracy lift over base", scale=alt.Scale(domain=[0.8, 3.8])),
        y=alt.Y("normalized_score:Q", title="Observed normalized score", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("source:N", title="Run family"),
        shape=alt.Shape("note:N", title="Caveat"),
        tooltip=["label", "source", "lift", "normalized_score", "note"],
    ).properties(width=700, height=420, title="17. Research vs Non-OpenClaw Local Tests")


def render_index(manifest: list[dict[str, str]], summary: dict[str, Any]) -> str:
    lines = [
        "# OpenClaw Altair Chart Pack",
        "",
        f"- Base score mean: `{summary['base_score_mean']:.4f}`",
        f"- Replay score mean: `{summary['replay_score_mean']:.4f}`",
        f"- Score delta mean: `{summary['score_delta_mean']:+.4f}`",
        f"- Win rate: `{summary['win_rate']:.4f}`",
        f"- Latency ratio: `{summary['latency_ratio_replay_over_base']:.2f}x`",
        "",
        "## Charts",
        "",
    ]
    for item in manifest:
        lines.append(f"- `{item['name']}`: HTML `{item['html']}`, PNG `{item['png']}` ({item['png_status']})")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
