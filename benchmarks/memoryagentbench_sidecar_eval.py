from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAB_ROOT = REPO_ROOT.parent / "vendor" / "MemoryAgentBench"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a small MemoryAgentBench subset through the delta-mem sidecar session API."
    )
    parser.add_argument("--memoryagentbench-root", type=Path, default=DEFAULT_MAB_ROOT)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--model", default="delta-mem-qwen3-4b-mlx")
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=DEFAULT_MAB_ROOT / "configs/data_conf/Conflict_Resolution/Factconsolidation_sh_6k.yaml",
    )
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "benchmarks/results/memoryagentbench-sidecar-delta.json")
    parser.add_argument("--max-contexts", type=int, default=1)
    parser.add_argument("--max-queries", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, help="Override MemoryAgentBench chunk_size.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--session-prefix", default="memoryagentbench")
    args = parser.parse_args()

    mab_root = args.memoryagentbench_root.expanduser().resolve()
    if not mab_root.exists():
        raise SystemExit(f"MemoryAgentBench repo not found: {mab_root}")
    sys.path.insert(0, str(mab_root))

    ensure_nltk_tokenizers()

    from conversation_creator import ConversationCreator  # type: ignore[import-not-found]
    from utils.eval_other_utils import metrics_summarization  # type: ignore[import-not-found]

    dataset_config = load_yaml(args.dataset_config)
    if args.chunk_size:
        dataset_config["chunk_size"] = args.chunk_size
    dataset_config["max_test_samples"] = min(int(dataset_config.get("max_test_samples", 1)), args.max_contexts)
    dataset_config.setdefault("debug", False)

    agent_config = {
        "agent_name": "Agentic_memory_delta_mem_sidecar",
        "model": args.model,
        "temperature": args.temperature,
        "input_length_limit": 300000,
        "buffer_length": 15000,
        "output_dir": str(args.output.parent),
        "retrieve_num": 0,
    }

    creator = ConversationCreator(agent_config, dataset_config)
    all_context_chunks = creator.get_chunks()[: args.max_contexts]
    all_query_answer_pairs = creator.get_query_and_answers()[: args.max_contexts]

    rows: list[dict[str, Any]] = []
    metrics: dict[str, list[Any]] = defaultdict(list)
    query_count = 0
    started = time.perf_counter()
    for context_index, (chunks, query_answer_pairs) in enumerate(zip(all_context_chunks, all_query_answer_pairs)):
        if query_count >= args.max_queries:
            break
        session_key = f"{args.session_prefix}:{dataset_config['sub_dataset']}:{context_index}:{int(time.time())}"
        replay_records = replay_chunks(
            chunks,
            base_url=args.base_url,
            model=args.model,
            session_key=session_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
        )
        for local_query_index, (query, answer, qa_pair_id) in enumerate(query_answer_pairs):
            if query_count >= args.max_queries:
                break
            response = post_chat(
                base_url=args.base_url,
                model=args.model,
                messages=[
                    {
                        "role": "user",
                        "content": query,
                    }
                ],
                session_key=session_key,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            output = response["choices"][0]["message"]["content"]
            usage = response.get("usage") or {}
            record = {
                "output": output,
                "input_len": int(usage.get("prompt_tokens") or rough_tokens(query)),
                "output_len": int(usage.get("completion_tokens") or rough_tokens(output)),
                "memory_construction_time": sum(item["elapsed_ms"] for item in replay_records) / 1000.0,
                "query_time_len": response["elapsed_ms"] / 1000.0,
                "context_id": context_index,
                "local_query_id": local_query_index,
                "session_key_hash": response.get("state_key_hash"),
            }
            metrics, rows = metrics_summarization(
                record,
                query,
                answer,
                dataset_config,
                metrics,
                rows,
                query_id=query_count,
                qa_pair_id=qa_pair_id,
            )
            rows[-1]["replay_chunks"] = len(replay_records)
            rows[-1]["session_key"] = session_key
            query_count += 1

    summary = summarize(metrics, rows, started)
    result = {
        "kind": "memoryagentbench_sidecar_eval",
        "benchmark": "MemoryAgentBench",
        "memoryagentbench_root": str(mab_root),
        "dataset_config": dataset_config,
        "agent_config": agent_config,
        "base_url": args.base_url,
        "model": args.model,
        "summary": summary,
        "data": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))
    return 0


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"config is not a mapping: {path}")
    return data


def ensure_nltk_tokenizers() -> None:
    try:
        import nltk
    except ImportError:
        return
    for resource, package in (("tokenizers/punkt", "punkt"), ("tokenizers/punkt_tab/english", "punkt_tab")):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)


def replay_chunks(
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
        response = post_chat(
            base_url=base_url,
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Memorize this MemoryAgentBench context chunk for later questions. "
                        "Acknowledge briefly and do not answer any questions yet.\n\n"
                        f"{chunk}"
                    ),
                }
            ],
            session_key=session_key,
            max_tokens=min(max_tokens, 32),
            temperature=temperature,
            timeout=timeout,
        )
        records.append(
            {
                "chunk_index": index,
                "chars": len(chunk),
                "elapsed_ms": response["elapsed_ms"],
                "ack": response["choices"][0]["message"]["content"],
            }
        )
    return records


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
            "X-Delta-Mem-Session-Key": session_key,
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            payload["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
            payload["state_key_hash"] = response.headers.get("X-Delta-State-Key-Hash")
            return payload
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def summarize(metrics: dict[str, list[Any]], rows: list[dict[str, Any]], started: float) -> dict[str, Any]:
    averaged = {}
    for key, values in metrics.items():
        numeric = [float(value) for value in values if isinstance(value, (int, float, bool))]
        if numeric:
            averaged[key] = round(sum(numeric) / len(numeric), 4)
    return {
        "queries": len(rows),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "averaged_metrics": averaged,
    }


def rough_tokens(value: str) -> int:
    return max(1, len(value.split()))


if __name__ == "__main__":
    raise SystemExit(main())
