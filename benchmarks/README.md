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
export DELTA_MEM_ADAPTER_DIR=/path/to/delta-mem-qwen3-4b-instruct-mlx-adapter
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

## MLX δ-mem Diagnostics

Use `mlx_delta_diagnostics.py` to check whether the MLX adapter path is actually
active:

```sh
python benchmarks/mlx_delta_diagnostics.py \
  --adapter-dir /path/to/delta-mem-qwen3-4b-instruct-mlx-adapter \
  --output benchmarks/results/mlx-delta-diagnostics.json
```

The diagnostic reports wrapped attention layers, δ-state norms, read norms,
weight norms, and a controlled plain-vs-δ output comparison. A useful first
check is nonzero wrapped layers plus nonzero state/read norms after a write.

## LoCoMo-10 Sidecar Eval

Use `locomo10_sidecar_eval.py` with the official δ-mem repo's
`data/locomo10.json` sample to run a small LoCoMo-style sidecar probe:

```sh
python benchmarks/locomo10_sidecar_eval.py \
  --data-file /path/to/delta-Mem/data/locomo10.json \
  --base-url http://127.0.0.1:8765 \
  --model delta-mem-qwen3-4b-mlx \
  --sample-limit 1 \
  --question-limit 10 \
  --session-limit 4 \
  --output benchmarks/results/locomo10-sidecar-delta.json
```

This is closer to the paper's LoCoMo setup than the synthetic memory probes, but
it is still a small local sample and not a reproduction of the full official
benchmark suite.

## Real OpenClaw Session Replay Eval

Use `openclaw_session_replay_eval.py` when you want to test behavior against a
real OpenClaw agent session instead of synthetic benchmark cases. Export and
sanitize the session history first, then replay it into the sidecar under a
stable session key and run behavior probes against the replayed memory:

```sh
python benchmarks/openclaw_session_replay_eval.py \
  --history-file /path/to/sanitized-openclaw-history.jsonl \
  --base-url http://127.0.0.1:8765 \
  --model delta-mem-qwen3-4b-mlx \
  --session-key openclaw:replay:example \
  --probe 'model_choice||What model stack should this agent be using?||delta-mem-mlx' \
  --probe 'transport||What transport should the integration use?||chat completions,X-OpenClaw-Session-Key' \
  --output benchmarks/results/openclaw-session-replay-delta.json
```

The history file may be a JSON list, JSONL records, or a JSON object containing
`messages`, `turns`, `history`, `events`, or `items`. Probe files may be JSON,
JSONL, or an object containing `probes`, with each probe shaped like:

```json
{"name": "transport", "question": "What transport should the integration use?", "expected": ["chat completions"]}
```

For a useful comparison, run the same history and probes twice: once against the
plain backbone model and once against the δ-mem sidecar profile. Compare
`summary.score_mean`, `summary.passed`, and probe latency. This harness does not
pull private OpenClaw gateway history directly; keeping export and sanitization
outside the benchmark makes the public repo repeatable and safe to share.

## Sanitized Transcript Replay Toolbelt

Use `openclaw_transcript_toolbelt.py` for larger local replay batches without
hammering model servers. It keeps calls single-threaded, writes rich JSON/JSONL
artifacts, and emits SVG graph summaries.

Example for a local Lima OpenClaw instance:

```sh
python benchmarks/openclaw_transcript_toolbelt.py export-lima \
  --instance clawfactory \
  --output-dir benchmarks/results/openclaw-16/raw \
  --limit 16

python benchmarks/openclaw_transcript_toolbelt.py sanitize \
  benchmarks/results/openclaw-16/raw/*.jsonl \
  --output-dir benchmarks/results/openclaw-16/sanitized \
  --limit 16

python benchmarks/openclaw_transcript_toolbelt.py probes \
  --sessions-file benchmarks/results/openclaw-16/sanitized/sessions.jsonl \
  --output-dir benchmarks/results/openclaw-16/probes

python benchmarks/openclaw_transcript_toolbelt.py run \
  --sessions-file benchmarks/results/openclaw-16/sanitized/sessions.jsonl \
  --probes-file benchmarks/results/openclaw-16/probes/probes.jsonl \
  --base-url http://127.0.0.1:8765 \
  --model delta-mem-qwen3-4b-mlx \
  --output-dir benchmarks/results/openclaw-16/run

python benchmarks/openclaw_transcript_toolbelt.py report \
  --results-jsonl benchmarks/results/openclaw-16/run/results.jsonl \
  --output-dir benchmarks/results/openclaw-16/report
```

The base condition is a no-history session using the same local model. The
replay condition first stores the sanitized transcript under a stable session
key and then asks the same eight deterministic probes. This measures whether
replayed session memory improves recall of sanitized transcript facts; it is a
confidence benchmark, not a substitute for task-specific human scoring.
