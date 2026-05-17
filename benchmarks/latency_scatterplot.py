from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openclaw_altair_report import build_cost_frame, cost_of_memory_signal


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the latency-ratio vs response-delta scatterplot for paired "
            "plain and delta benchmark result JSON files."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks/results",
        help="Directory containing paired *-plain.json and *-delta.json benchmark outputs.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=REPO_ROOT / "benchmarks/results/openclaw-16/report/summary.json",
        help="OpenClaw-16 report summary JSON to add as the current strict replay point.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "benchmarks/results/openclaw-16/altair-report/16-cost-of-memory-signal.png",
        help="PNG output path.",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        help="Optional HTML output path. Defaults to the PNG path with .html suffix.",
    )
    args = parser.parse_args()

    try:
        import altair as alt
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "Install report dependencies first: "
            "python3 -m pip install --user altair pandas vl-convert-python"
        ) from exc

    summary = read_summary(args.summary_json)
    cost_df = build_cost_frame(pd, summary, args.results_dir)
    chart = cost_of_memory_signal(alt, cost_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    html_output = args.html_output or args.output.with_suffix(".html")
    html_output.parent.mkdir(parents=True, exist_ok=True)

    chart.save(html_output)
    chart.save(args.output, scale_factor=2)

    print(
        json.dumps(
            {
                "rows": len(cost_df),
                "output": str(args.output),
                "html_output": str(html_output),
                "instructions": [
                    "Use paired result files to compare plain vs delta runs.",
                    "Strict-rescore saved OpenClaw probe outputs before plotting.",
                    "Exclude QMD deterministic and synthesized 200-token and 400-token preload rows.",
                    "Use short point labels such as a1 and a two-column guide.",
                    "Include the benchmark family in the guide for each point.",
                    "Keep the latency-ratio axis tight and render ticks with an x suffix.",
                ],
            },
            indent=2,
        )
    )
    return 0


def read_summary(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "latency_ratio_replay_over_base": 0,
        "score_delta_mean": 0,
    }


if __name__ == "__main__":
    raise SystemExit(main())
