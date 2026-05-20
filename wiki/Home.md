# Delta-Mem MLX Sidecar

This project runs local MLX models behind an OpenAI-compatible sidecar. The
optional delta-mem path adds a compact online memory state coupled to attention,
so a stable session key can shape future responses without replaying all prior
context.

The paper reports that a tiny online state improved average score to `1.10x`
over the frozen backbone, with larger gains on memory-heavy tasks: `1.31x` on
MemoryAgentBench and `1.20x` on LoCoMo. This repo keeps the default run small
with a toy MLX model while leaving the Qwen3 adapter path available.

## Default Run

```text
DELTA_MEM_RUNTIME=mlx
DELTA_MEM_MODEL_PATH=mlx-community/Qwen2.5-0.5B-Instruct-4bit
DELTA_MEM_MODEL_ID=qwen2.5-0.5b-mlx-test
```

Requests that should share memory state must provide:

```text
X-Delta-Mem-Session-Key: <stable logical session key>
```

## Optional Delta-Mem Adapter Run

The released adapter is compatible with Qwen3-4B, not the toy model:

```text
mlx-community/Qwen3-4B-Instruct-2507-4bit
ofthetrees/delta-mem-qwen3-4b-instruct-mlx-adapter
```

Download the model and converted MLX adapter, then set
`DELTA_MEM_ADAPTER_DIR` when starting the sidecar. The upstream Torch adapter
remains available if you need to regenerate the MLX artifact.

## Verification

Use the benchmarks to compare plain backbone behavior against delta-mem behavior:

- `benchmarks/session_memory_eval.py`
- `benchmarks/session_replay_eval.py`
- `benchmarks/full_delta_mem_bench.py`
- `wiki/Benchmark-Findings.md`
