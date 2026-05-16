from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROMPTS = [
    "Reply in one short sentence: what is this runtime testing?",
    "Summarize this in one sentence: Apple Silicon uses unified memory with MLX.",
    "Write a tiny Python function that adds two numbers.",
]


@dataclass(frozen=True)
class Measurement:
    prompt_index: int
    prompt_chars: int
    output_chars: int
    elapsed_ms: float
    output: str


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark direct MLX-LM generation.")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompt", action="append", help="Prompt to benchmark. May be repeated.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    prompts = args.prompt or DEFAULT_PROMPTS
    result = run_benchmark(
        model_id=args.model,
        prompts=prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0


def run_benchmark(
    *,
    model_id: str,
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    load_started = time.perf_counter()
    model, tokenizer = load(model_id)
    load_ms = elapsed_ms(load_started)
    sampler = make_sampler(temp=temperature)

    for index in range(warmup):
        prompt = render_prompt(tokenizer, prompts[index % len(prompts)])
        generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler, verbose=False)

    measurements: list[Measurement] = []
    for repeat_index in range(repeats):
        for prompt_index, prompt_text in enumerate(prompts):
            prompt = render_prompt(tokenizer, prompt_text)
            started = time.perf_counter()
            output = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            )
            measurements.append(
                Measurement(
                    prompt_index=prompt_index,
                    prompt_chars=len(prompt),
                    output_chars=len(output),
                    elapsed_ms=elapsed_ms(started),
                    output=output.strip(),
                )
            )

    latencies = [measurement.elapsed_ms for measurement in measurements]
    output_chars = [measurement.output_chars for measurement in measurements]
    return {
        "kind": "direct_mlx_generation",
        "model": model_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "warmup": warmup,
        "repeats": repeats,
        "load_ms": round(load_ms, 3),
        "summary": summarize(latencies, output_chars),
        "measurements": [measurement.__dict__ for measurement in measurements],
    }


def render_prompt(tokenizer: object, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if isinstance(rendered, str):
            return rendered
    return f"user: {prompt}\nassistant:"


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
