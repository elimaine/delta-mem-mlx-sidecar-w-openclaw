from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROMPTS = [
    "Reply in one short sentence: what endpoint is being tested?",
    "Summarize this in one sentence: the sidecar exposes an OpenAI-compatible API.",
    "Write a tiny Python function that subtracts two numbers.",
]


@dataclass(frozen=True)
class Measurement:
    prompt_index: int
    status_code: int
    output_chars: int
    elapsed_ms: float
    output: str


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark sidecar /v1/chat/completions latency.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--model", default="qwen2.5-0.5b-mlx-test")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--prompt", action="append", help="Prompt to benchmark. May be repeated.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    prompts = args.prompt or DEFAULT_PROMPTS
    result = run_benchmark(
        base_url=args.base_url,
        model=args.model,
        prompts=prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        warmup=args.warmup,
        repeats=args.repeats,
        timeout=args.timeout,
    )

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0


def run_benchmark(
    *,
    base_url: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    warmup: int,
    repeats: int,
    timeout: float,
) -> dict[str, Any]:
    for index in range(warmup):
        post_chat(
            base_url=base_url,
            model=model,
            prompt=prompts[index % len(prompts)],
            max_tokens=max_tokens,
            temperature=temperature,
            session_key=f"bench:warmup:{index}",
            timeout=timeout,
        )

    measurements: list[Measurement] = []
    for repeat_index in range(repeats):
        for prompt_index, prompt in enumerate(prompts):
            started = time.perf_counter()
            status_code, output = post_chat(
                base_url=base_url,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                session_key=f"bench:{repeat_index}:{prompt_index}",
                timeout=timeout,
            )
            measurements.append(
                Measurement(
                    prompt_index=prompt_index,
                    status_code=status_code,
                    output_chars=len(output),
                    elapsed_ms=elapsed_ms(started),
                    output=output.strip(),
                )
            )

    latencies = [measurement.elapsed_ms for measurement in measurements]
    output_chars = [measurement.output_chars for measurement in measurements]
    return {
        "kind": "sidecar_chat_completions",
        "base_url": base_url,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "warmup": warmup,
        "repeats": repeats,
        "summary": summarize(latencies, output_chars),
        "measurements": [measurement.__dict__ for measurement in measurements],
    }


def post_chat(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    session_key: str,
    timeout: float,
) -> tuple[int, str]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-OpenClaw-Session-Key": session_key,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
        output = body["choices"][0]["message"]["content"]
        return response.status, output


def summarize(latencies: list[float], output_chars: list[int]) -> dict[str, float]:
    total_chars = sum(output_chars)
    total_seconds = sum(latencies) / 1000.0
    return {
        "count": len(latencies),
        "latency_ms_min": round(min(latencies), 3),
        "latency_ms_mean": round(statistics.fmean(latencies), 3),
        "latency_ms_p50": round(statistics.median(latencies), 3),
        "latency_ms_max": round(max(latencies), 3),
        "output_chars_per_second": round(total_chars / total_seconds, 3) if total_seconds else 0.0,
    }


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


if __name__ == "__main__":
    raise SystemExit(main())
