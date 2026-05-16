from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ADAPTER_DIR = os.getenv("DELTA_MEM_ADAPTER_DIR", "")


@dataclass(frozen=True)
class BenchCase:
    name: str
    benchmark: str
    session_key: str
    write_turns: list[str]
    query: str
    references: list[str]
    key_terms: list[str]
    condition: str
    carry_history_on_query: bool = False


CASES = [
    BenchCase(
        name="direct_instruction",
        benchmark="general_control",
        session_key="fullbench:direct",
        write_turns=[],
        query="Answer with exactly one word: ready",
        references=["ready"],
        key_terms=["ready"],
        condition="with_context",
        carry_history_on_query=True,
    ),
    BenchCase(
        name="no_context_preference_recovery",
        benchmark="locomo_style",
        session_key="fullbench:preference",
        write_turns=[
            "Remember that my preferred terminal theme is Solarized Dark.",
            "Talk briefly about shell prompts without mentioning themes.",
        ],
        query="What terminal theme did I ask you to remember?",
        references=["Solarized Dark"],
        key_terms=["solarized", "dark"],
        condition="no_context_recovery",
    ),
    BenchCase(
        name="no_context_correction_recovery",
        benchmark="memory_agent_style",
        session_key="fullbench:correction",
        write_turns=[
            "Remember that the project codename is Pike.",
            "Correction: the project codename is Delta Pike.",
        ],
        query="What is the current project codename?",
        references=["Delta Pike"],
        key_terms=["delta", "pike"],
        condition="no_context_recovery",
    ),
    BenchCase(
        name="no_context_multi_hop_task_state",
        benchmark="memory_agent_style",
        session_key="fullbench:task-state",
        write_turns=[
            "Project status update: todo 31 is complete, todo 32 is blocked on GPU access, and todo 33 is next.",
            "Unrelated deployment note: the sidecar listens on port 8765.",
        ],
        query="Which todo is next?",
        references=["todo 33", "33"],
        key_terms=["33"],
        condition="no_context_recovery",
    ),
    BenchCase(
        name="with_context_hotpot_style_bridge",
        benchmark="hotpotqa_style",
        session_key="fullbench:bridge",
        write_turns=[
            "Fact A: the adapter artifact we need is called delta_mem_adapter.pt.",
            "Fact B: the adapter artifact must be paired with delta_mem_config.json.",
        ],
        query="Which two files are required for the adapter artifact package?",
        references=["delta_mem_adapter.pt and delta_mem_config.json"],
        key_terms=["delta_mem_adapter.pt", "delta_mem_config.json"],
        condition="with_context",
        carry_history_on_query=True,
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark full Qwen3 MLX vs Qwen3+delta-Mem sidecar setups.")
    parser.add_argument("--python", default="delta-mem-sidecar/.venv/bin/python")
    parser.add_argument("--uvicorn", default="delta-mem-sidecar/.venv/bin/uvicorn")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-path", default="mlx-community/Qwen3-4B-Instruct-2507-4bit")
    parser.add_argument(
        "--adapter-dir",
        default=DEFAULT_ADAPTER_DIR,
        help="Path to a converted delta-Mem adapter directory. Defaults to DELTA_MEM_ADAPTER_DIR.",
    )
    parser.add_argument("--plain-model-id", default="qwen3-4b-mlx")
    parser.add_argument("--delta-model-id", default="delta-mem-qwen3-4b-mlx")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--warmup", type=int, default=1, help="Unmeasured warmup requests per profile.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()
    if not args.adapter_dir:
        raise SystemExit(
            "--adapter-dir is required for the delta profile. Set DELTA_MEM_ADAPTER_DIR "
            "or pass --adapter-dir /path/to/delta-mem-qwen3-4b-instruct-mlx-adapter."
        )

    uvicorn = str(Path(args.uvicorn).expanduser().resolve())
    base_url = f"http://{args.host}:{args.port}"
    results = []
    results.append(
        run_profile(
            profile="plain",
            uvicorn=uvicorn,
            host=args.host,
            port=args.port,
            base_url=base_url,
            model_path=args.model_path,
            adapter_dir=None,
            model_id=args.plain_model_id,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            warmup=args.warmup,
            startup_timeout=args.startup_timeout,
            request_timeout=args.request_timeout,
        )
    )
    results.append(
        run_profile(
            profile="delta",
            uvicorn=uvicorn,
            host=args.host,
            port=args.port,
            base_url=base_url,
            model_path=args.model_path,
            adapter_dir=args.adapter_dir,
            model_id=args.delta_model_id,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            warmup=args.warmup,
            startup_timeout=args.startup_timeout,
            request_timeout=args.request_timeout,
        )
    )

    result = {
        "kind": "full_delta_mem_benchmark",
        "model_path": args.model_path,
        "adapter_dir": args.adapter_dir,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "warmup": args.warmup,
        "cases": [
            {
                "name": case.name,
                "benchmark": case.benchmark,
                "condition": case.condition,
                "write_turn_count": len(case.write_turns),
                "carry_history_on_query": case.carry_history_on_query,
                "references": case.references,
                "key_terms": case.key_terms,
            }
            for case in CASES
        ],
        "profiles": results,
    }
    result["comparison"] = compare_profiles(results)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0


def run_profile(
    *,
    profile: str,
    uvicorn: str,
    host: str,
    port: int,
    base_url: str,
    model_path: str,
    adapter_dir: str | None,
    model_id: str,
    max_tokens: int,
    temperature: float,
    warmup: int,
    startup_timeout: float,
    request_timeout: float,
) -> dict[str, Any]:
    env = {
        "DELTA_MEM_RUNTIME": "mlx",
        "DELTA_MEM_MODEL_PATH": model_path,
        "DELTA_MEM_MODEL_ID": model_id,
        "DELTA_MEM_MAX_NEW_TOKENS": str(max_tokens),
    }
    if adapter_dir is not None:
        env["DELTA_MEM_ADAPTER_DIR"] = adapter_dir

    started = time.perf_counter()
    process = subprocess.Popen(
        [
            uvicorn,
            "delta_mem_sidecar.app:create_app",
            "--factory",
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd="delta-mem-sidecar",
        env={**os.environ, **env},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        wait_for_health(base_url, timeout=startup_timeout)
        startup_ms = elapsed_ms(started)
        for warmup_index in range(warmup):
            post_chat(
                base_url=base_url,
                model_id=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": "Warm up the model. Reply with exactly: ready",
                    }
                ],
                session_key=f"fullbench:warmup:{profile}:{warmup_index}",
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=request_timeout,
            )
        measurements = [
            run_case(
                case,
                base_url=base_url,
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=request_timeout,
            )
            for case in CASES
        ]
        latencies = [
            turn["elapsed_ms"]
            for measurement in measurements
            for turn in measurement["turns"]
        ]
        scores = [measurement["score"] for measurement in measurements]
        return {
            "profile": profile,
            "model_id": model_id,
            "startup_ms": round(startup_ms, 3),
            "warmup_requests": warmup,
            "summary": {
                "turn_count": len(latencies),
                "latency_ms_min": round(min(latencies), 3),
                "latency_ms_mean": round(statistics.fmean(latencies), 3),
                "latency_ms_p50": round(statistics.median(latencies), 3),
                "latency_ms_max": round(max(latencies), 3),
                "score_mean": round(statistics.fmean(scores), 4),
                "cases_passed": sum(1 for measurement in measurements if measurement["passed"]),
                "cases_total": len(measurements),
                "scores_by_benchmark": summarize_scores_by(measurements, "benchmark"),
                "scores_by_condition": summarize_scores_by(measurements, "condition"),
            },
            "cases": measurements,
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def run_case(
    case: BenchCase,
    *,
    base_url: str,
    model_id: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    turns = []
    for turn_index, prompt in enumerate(case.write_turns):
        started = time.perf_counter()
        response = post_chat(
            base_url=base_url,
            model_id=model_id,
            messages=[{"role": "user", "content": prompt}],
            session_key=case.session_key,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        elapsed = elapsed_ms(started)
        output = response["choices"][0]["message"]["content"]
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": output})
        turns.append(
            {
                "turn_index": turn_index,
                "phase": "write",
                "elapsed_ms": round(elapsed, 3),
                "output": output,
            }
        )
    query_messages = (
        [*messages, {"role": "user", "content": case.query}]
        if case.carry_history_on_query
        else [{"role": "user", "content": case.query}]
    )
    started = time.perf_counter()
    response = post_chat(
        base_url=base_url,
        model_id=model_id,
        messages=query_messages,
        session_key=case.session_key,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )
    elapsed = elapsed_ms(started)
    final_output = response["choices"][0]["message"]["content"]
    turns.append(
        {
            "turn_index": len(case.write_turns),
            "phase": "query",
            "elapsed_ms": round(elapsed, 3),
            "output": final_output,
        }
    )
    metadata = get_metadata(base_url, case.session_key, timeout=timeout)
    score_detail = score_answer(final_output, references=case.references, key_terms=case.key_terms)
    passed = score_detail["score"] >= 0.5
    return {
        "name": case.name,
        "benchmark": case.benchmark,
        "condition": case.condition,
        "passed": passed,
        "carry_history_on_query": case.carry_history_on_query,
        "references": case.references,
        "key_terms": case.key_terms,
        "score": score_detail["score"],
        "score_detail": score_detail,
        "final_output": final_output,
        "metadata_updates": metadata["updates"],
        "turns": turns,
    }


def score_answer(answer: str, *, references: list[str], key_terms: list[str]) -> dict[str, Any]:
    key_term_score = key_term_recall(answer, key_terms)
    f1 = max((token_f1(answer, reference) for reference in references), default=0.0)
    exact = max((exact_match(answer, reference) for reference in references), default=0.0)
    score = max(key_term_score, f1, exact)
    return {
        "score": round(score, 4),
        "key_term_recall": round(key_term_score, 4),
        "token_f1": round(f1, 4),
        "exact_match": round(exact, 4),
    }


def key_term_recall(answer: str, key_terms: list[str]) -> float:
    if not key_terms:
        return 0.0
    normalized = normalize_text(answer)
    hits = sum(1 for term in key_terms if normalize_text(term) in normalized)
    return hits / len(key_terms)


def token_f1(answer: str, reference: str) -> float:
    answer_tokens = normalize_text(answer).split()
    reference_tokens = normalize_text(reference).split()
    if not answer_tokens or not reference_tokens:
        return 0.0
    common = 0
    remaining = list(reference_tokens)
    for token in answer_tokens:
        if token in remaining:
            common += 1
            remaining.remove(token)
    if common == 0:
        return 0.0
    precision = common / len(answer_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(answer: str, reference: str) -> float:
    return 1.0 if normalize_text(answer) == normalize_text(reference) else 0.0


def normalize_text(text: str) -> str:
    lowered = text.lower().replace("_", " ")
    return " ".join(
        "".join(character for character in token if character.isalnum() or character in ".-")
        for token in lowered.split()
    ).strip()


def summarize_scores_by(measurements: list[dict[str, Any]], field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for measurement in measurements:
        grouped.setdefault(str(measurement[field]), []).append(float(measurement["score"]))
    return {key: round(statistics.fmean(values), 4) for key, values in sorted(grouped.items())}


def compare_profiles(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {profile["profile"]: profile for profile in profiles}
    plain = by_name.get("plain")
    delta = by_name.get("delta")
    if plain is None or delta is None:
        return {}
    plain_score = plain["summary"]["score_mean"]
    delta_score = delta["summary"]["score_mean"]
    plain_latency = plain["summary"]["latency_ms_mean"]
    delta_latency = delta["summary"]["latency_ms_mean"]
    return {
        "method": "paper_style_backbone_vs_delta",
        "score_delta_points": round(delta_score - plain_score, 4),
        "score_ratio_delta_over_plain": safe_ratio(delta_score, plain_score),
        "latency_overhead_ratio_delta_over_plain": safe_ratio(delta_latency, plain_latency),
        "scores_by_condition_delta_minus_plain": compare_nested_scores(
            plain["summary"]["scores_by_condition"],
            delta["summary"]["scores_by_condition"],
        ),
        "scores_by_benchmark_delta_minus_plain": compare_nested_scores(
            plain["summary"]["scores_by_benchmark"],
            delta["summary"]["scores_by_benchmark"],
        ),
        "interpretation": (
            "Positive score deltas indicate the delta-Mem profile recovered or used "
            "session state better than the frozen backbone under the same prompts. "
            "This mirrors the paper's backbone-vs-delta and no-context recovery framing; "
            "it is not a perfect-recall contract."
        ),
    }


def compare_nested_scores(plain: dict[str, float], delta: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(plain) | set(delta))
    return {key: round(delta.get(key, 0.0) - plain.get(key, 0.0), 4) for key in keys}


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def wait_for_health(base_url: str, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=1.0):
                return
        except Exception as exc:  # noqa: BLE001 - keep polling until timeout.
            last_error = exc
            time.sleep(0.2)
    raise RuntimeError(f"sidecar did not become healthy: {last_error}")


def post_chat(
    *,
    base_url: str,
    model_id: str,
    messages: list[dict[str, str]],
    session_key: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model_id,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    return request_json(
        f"{base_url}/v1/chat/completions",
        payload=payload,
        headers={"X-OpenClaw-Session-Key": session_key},
        timeout=timeout,
    )


def get_metadata(base_url: str, session_key: str, *, timeout: float) -> dict[str, Any]:
    encoded = urllib.parse.quote(session_key, safe="")
    request = urllib.request.Request(f"{base_url}/delta/state/{encoded}/metadata")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def request_json(
    url: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


if __name__ == "__main__":
    raise SystemExit(main())
