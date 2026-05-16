# Benchmark Harness

`openclaw_memory_eval.py` runs a small set of OpenClaw-style memory regression
cases against an OpenAI-compatible `/v1/chat/completions` endpoint.

It is intentionally dependency-free and uses Python's standard library.

## Run Against Local Sidecar

Start the sidecar:

```sh
cd delta-mem-sidecar
uvicorn delta_mem_sidecar.app:create_app --factory --host 127.0.0.1 --port 8765
```

Run the eval:

```sh
python3 benchmarks/openclaw_memory_eval.py \
  --base-url http://127.0.0.1:8765 \
  --model delta-mem-fake
```

The fake model is not expected to pass semantic memory tests. It is useful for
checking request shape, session-key plumbing, and result capture. Real pass/fail
signals require a stronger local model runtime behind the sidecar.

## MLX Performance Appraisal

Use `mlx_generation_bench.py` to measure direct `mlx-lm` generation without HTTP
overhead:

```sh
cd delta-mem-sidecar
. .venv/bin/activate
cd ..
python benchmarks/mlx_generation_bench.py \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --max-tokens 32 \
  --warmup 1 \
  --repeats 3 \
  --output benchmarks/results/mlx-direct-qwen2.5-0.5b.json
```

Use `sidecar_latency_bench.py` to measure the OpenAI-compatible sidecar path:

```sh
cd delta-mem-sidecar
DELTA_MEM_RUNTIME=mlx \
DELTA_MEM_MODEL_PATH=mlx-community/Qwen2.5-0.5B-Instruct-4bit \
DELTA_MEM_MODEL_ID=qwen2.5-0.5b-mlx-test \
DELTA_MEM_MAX_NEW_TOKENS=32 \
.venv/bin/uvicorn delta_mem_sidecar.app:create_app --factory --host 127.0.0.1 --port 8765
```

In another shell:

```sh
python benchmarks/sidecar_latency_bench.py \
  --base-url http://127.0.0.1:8765 \
  --model qwen2.5-0.5b-mlx-test \
  --max-tokens 32 \
  --warmup 1 \
  --repeats 3 \
  --output benchmarks/results/mlx-sidecar-qwen2.5-0.5b.json
```

Both scripts emit JSON with per-call measurements and a summary containing min,
mean, p50, max latency, and output characters per second. Character throughput is
only a rough proxy; use it for local before/after comparisons, not model-to-model
claims.

## Full Qwen3 + delta-Mem Paper-Style Appraisal

Use `full_delta_mem_bench.py` to benchmark the sidecar with the compatible
Qwen3-4B MLX backbone both with and without the released delta-Mem adapter:

```sh
python benchmarks/full_delta_mem_bench.py \
  --max-tokens 32 \
  --warmup 1 \
  --output benchmarks/results/full-delta-mem-qwen3-4b-smoke.json
```

The script starts and stops the sidecar once for each profile:

- `plain`: `DELTA_MEM_RUNTIME=mlx` with `mlx-community/Qwen3-4B-Instruct-2507-4bit`
- `delta`: the same backbone plus the converted `delta_mem_adapter_mlx.npz`
  artifact from `declare-lab/delta-mem_qwen3_4b-instruct`

The harness now follows the paper's verification shape more closely:

- compare frozen backbone vs backbone plus δ-mem;
- include memory-heavy categories inspired by MemoryAgentBench and LoCoMo;
- include no-context recovery probes where the write turns update session state
  but the final query does not replay prior chat history;
- report token/key-term scores instead of only substring pass/fail;
- report score deltas, score ratios, and latency overhead for the δ-mem profile.

These are still compact local probes, not a reproduction of the full paper
benchmark suite. A useful result is a positive δ-mem delta over the plain
backbone, especially in `no_context_recovery`. Exact hidden-fact recall is not
the contract; measurable attention-shaped recovery is the signal.
