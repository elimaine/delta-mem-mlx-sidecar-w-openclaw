from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from delta_mem_sidecar.mlx_delta_attention import iter_mlx_delta_attention_modules
from delta_mem_sidecar.mlx_runtime import MlxBackboneRuntime
from delta_mem_sidecar.runtime import ChatMessage


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose whether the MLX δ-mem adapter path is active.")
    parser.add_argument("--model-path", default="mlx-community/Qwen3-4B-Instruct-2507-4bit")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    plain = MlxBackboneRuntime(
        model_path=args.model_path,
        model_id="plain-diagnostic",
        max_tokens=args.max_tokens,
    )
    delta = MlxBackboneRuntime(
        model_path=args.model_path,
        model_id="delta-diagnostic",
        adapter_dir=args.adapter_dir,
        max_tokens=args.max_tokens,
    )
    plain._load()
    delta._load()

    write_prompt = "Remember that the verification token is aurora-17."
    query_prompt = "What is the verification token?"

    plain_state = plain.fresh_state()
    delta_state = delta.fresh_state()
    plain_write = plain.generate(
        messages=[ChatMessage(role="user", content=write_prompt)],
        state=plain_state,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    delta_write = delta.generate(
        messages=[ChatMessage(role="user", content=write_prompt)],
        state=delta_state,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    after_write = collect_delta_metrics(delta)
    plain_query = plain.generate(
        messages=[ChatMessage(role="user", content=query_prompt)],
        state=plain_state,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    delta_query = delta.generate(
        messages=[ChatMessage(role="user", content=query_prompt)],
        state=delta_state,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    after_query = collect_delta_metrics(delta)

    result = {
        "kind": "mlx_delta_diagnostics",
        "model_path": args.model_path,
        "adapter_dir": args.adapter_dir,
        "wrapped_layers": [layer for layer, _ in iter_mlx_delta_attention_modules(delta._model)],
        "delta_enabled": bool(delta._delta_enabled),
        "write": {
            "prompt": write_prompt,
            "plain_output": plain_write.content,
            "delta_output": delta_write.content,
            "outputs_equal": plain_write.content == delta_write.content,
        },
        "query": {
            "prompt": query_prompt,
            "plain_output": plain_query.content,
            "delta_output": delta_query.content,
            "outputs_equal": plain_query.content == delta_query.content,
        },
        "after_write": after_write,
        "after_query": after_query,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not result["delta_enabled"] or not result["wrapped_layers"]:
        return 2
    if after_write["state_layers"] == 0 or after_write["state_norm_sum"] <= 0:
        return 3
    return 0


def collect_delta_metrics(runtime: MlxBackboneRuntime) -> dict[str, Any]:
    state_norms: dict[str, float] = {}
    read_norms: dict[str, float] = {}
    weight_norms: dict[str, float] = {}
    for layer_index, attention in iter_mlx_delta_attention_modules(runtime._model):
        state = attention.state_snapshot()
        reads = attention.last_reads
        state_norms[str(layer_index)] = array_norm(state) if state is not None else 0.0
        read_norms[str(layer_index)] = array_norm(reads) if reads is not None else 0.0
        weight_norms[str(layer_index)] = array_norm(attention.weights.delta_o_proj) + array_norm(
            attention.weights.delta_q_proj
        )
    return {
        "state_layers": sum(1 for value in state_norms.values() if value > 0),
        "state_norm_sum": round(sum(state_norms.values()), 6),
        "read_norm_sum": round(sum(read_norms.values()), 6),
        "weight_norm_sum": round(sum(weight_norms.values()), 6),
        "state_norms": state_norms,
        "read_norms": read_norms,
    }


def array_norm(value: Any) -> float:
    import mlx.core as mx

    norm = mx.sqrt(mx.sum(value * value))
    mx.eval(norm)
    result = float(norm.item())
    if math.isnan(result) or math.isinf(result):
        return 0.0
    return result


if __name__ == "__main__":
    raise SystemExit(main())
