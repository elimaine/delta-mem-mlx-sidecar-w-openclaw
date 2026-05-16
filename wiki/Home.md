# Delta-Mem MLX Sidecar With OpenClaw

This project runs a local MLX model behind an OpenAI-compatible sidecar. It can run directly through HTTP or be routed through OpenClaw. The optional δ-mem path adds a compact online memory state coupled to attention, so a stable session key can shape future responses without replaying all prior context.

Why it is good: the δ-mem paper reports that a tiny online state improved average score to 1.10x over the frozen backbone, with larger gains on memory-heavy tasks: 1.31x on MemoryAgentBench and 1.20x on LoCoMo. The public adapter is released for Qwen3-4B, and this repo keeps the default run small with a toy MLX model while leaving the adapter path available.

References:

- Paper: https://arxiv.org/abs/2605.12357
- Hugging Face paper page: https://huggingface.co/papers/2605.12357
- Released adapter: https://huggingface.co/declare-lab/delta-mem_qwen3_4b-instruct

## Expected From User

- Run on Apple Silicon for the MLX path.
- Install Python 3.11+.
- Install sidecar dependencies with `pip install -e ".[mlx,test]"`.
- Start the sidecar on port `8765`.
- Configure OpenClaw only if agent routing is desired.
- Preserve a stable `X-OpenClaw-Session-Key` per agent/session.
- Treat exact hidden-fact recall as non-guaranteed; the target signal is measurable attention-shaped continuity.

## Default Run

The export-friendly default uses:

```text
DELTA_MEM_RUNTIME=mlx
DELTA_MEM_MODEL_PATH=mlx-community/Qwen2.5-0.5B-Instruct-4bit
DELTA_MEM_MODEL_ID=qwen2.5-0.5b-mlx-test
```

This validates OpenClaw routing, local MLX inference, request shape, state-key handling, and benchmark plumbing without requiring a large model.

## OpenClaw Setup

OpenClaw is optional. Skip this section if you only want the local sidecar API.

Use `openclaw.example.json5` as the provider snippet. The important transport setting is:

```text
transportProtocol: "openai-chat-completions"
```

For OpenClaw inside Lima, point the provider at:

```text
http://host.lima.internal:8765/v1
```

For host-only testing:

```text
http://127.0.0.1:8765/v1
```

## Optional δ-mem Adapter Run

The released adapter is compatible with Qwen3-4B, not the toy model:

```text
mlx-community/Qwen3-4B-Instruct-2507-4bit
ofthetrees/delta-mem-qwen3-4b-instruct-mlx-adapter
```

Download the model and converted MLX adapter, then set `DELTA_MEM_ADAPTER_DIR` when starting the sidecar. The upstream Torch adapter remains available if you need to regenerate the MLX artifact.

## Hugging Face Adapter Artifacts

Upstream adapter:

- https://huggingface.co/declare-lab/delta-mem_qwen3_4b-instruct

MLX-converted adapter:

- https://huggingface.co/ofthetrees/delta-mem-qwen3-4b-instruct-mlx-adapter

## Verification

Use the benchmarks to compare plain backbone behavior against δ-mem behavior:

- `benchmarks/openclaw_memory_eval.py`
- `benchmarks/full_delta_mem_bench.py`

The paper-style harness reports memory-heavy probes, no-context recovery, score deltas, ratios, and latency overhead.
