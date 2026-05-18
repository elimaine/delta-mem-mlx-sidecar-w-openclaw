# delta-mem-mlx-sidecar-w-openclaw

**Migration note:** this repo may lag behind the most up-to-date implementation now lives in the Openclaw native plugin repo: https://github.com/elimaine/openclaw-delta-mem-mlx-plugin which can be installed sidecar only. Keeping v0 here to preserve links and for benchmark tests.

OpenAI-compatible sidecar for running local MLX models directly or behind OpenClaw, with an optional δ-mem adapter path for session-shaped memory.

δ-mem augments a frozen transformer with a compact online memory state that reads and writes through attention. The paper reports an average 1.10x gain over the frozen backbone, 1.31x on MemoryAgentBench, and 1.20x on LoCoMo using a small online state rather than full fine-tuning or explicit context extension.

References:

- Paper: https://arxiv.org/abs/2605.12357
- Hugging Face paper page: https://huggingface.co/papers/2605.12357
- Released adapter: https://huggingface.co/declare-lab/delta-mem_qwen3_4b-instruct

## Expected From User

- Apple Silicon Mac for the default MLX path.
- Python 3.11+.
- Hugging Face access for the model downloads.
- OpenClaw gateway only if you want agent routing through this sidecar.
- A stable `X-OpenClaw-Session-Key` per logical agent/session when using memory state.

## Quick Start

The default local run uses a small toy MLX model so setup is cheap and export-friendly:

```sh
cd delta-mem-sidecar
python3.11 -m venv .venv
. .venv/bin/activate
pip install -e ".[mlx,test]"

DELTA_MEM_RUNTIME=mlx \
DELTA_MEM_MODEL_PATH=mlx-community/Qwen2.5-0.5B-Instruct-4bit \
DELTA_MEM_MODEL_ID=qwen2.5-0.5b-mlx-test \
DELTA_MEM_MAX_NEW_TOKENS=256 \
uvicorn delta_mem_sidecar.app:create_app --factory --host 127.0.0.1 --port 8765
```

Health check:

```sh
curl -fsS http://127.0.0.1:8765/health
```

Chat-completions smoke:

```sh
curl -fsS http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-OpenClaw-Session-Key: local:smoke' \
  -d '{
    "model": "qwen2.5-0.5b-mlx-test",
    "messages": [{"role": "user", "content": "Reply with exactly READY"}],
    "max_tokens": 16,
    "temperature": 0
  }'
```

## OpenClaw

OpenClaw is optional. The sidecar runs as a normal local HTTP service without it.

Use `openclaw.example.json5` as the provider shape. The important parts are:

- `baseUrl` points at this sidecar's `/v1` root.
- `transportProtocol` is `openai-chat-completions`.
- `X-OpenClaw-Session-Key` is present and stable for a logical session.

The default example assumes OpenClaw and the sidecar run on the same machine, so the provider URL is `http://127.0.0.1:8765/v1`.

Most users can skip an OpenClaw plugin and configure this as a model provider only. If you need a custom embedded harness, see `wiki/OpenClaw-Plugin.md` for the optional plugin reference.

## Released δ-mem Adapter

The released public adapter targets `Qwen/Qwen3-4B-Instruct-2507`, not the small toy model. To run it locally through MLX, download the compatible MLX backbone and the converted MLX adapter, then start the sidecar with `DELTA_MEM_ADAPTER_DIR`.

```sh
hf download mlx-community/Qwen3-4B-Instruct-2507-4bit
hf download ofthetrees/delta-mem-qwen3-4b-instruct-mlx-adapter
```

See `delta-mem-sidecar/README.md` for the exact optional adapter run command. If you need to rebuild the MLX artifact from the upstream Torch adapter, the converter is included in the sidecar package.

## Hugging Face Adapter Artifacts

Upstream Torch adapter:

- https://huggingface.co/declare-lab/delta-mem_qwen3_4b-instruct

Published MLX-converted adapter artifact:

- https://huggingface.co/ofthetrees/delta-mem-qwen3-4b-instruct-mlx-adapter

The converted MLX artifact should be published as a small adapter repo, not as merged model weights.

## Verification

The benchmark harness follows the paper's verification shape at a small local scale:

- frozen backbone vs δ-mem profile
- memory-heavy probes
- no-context recovery probes
- score deltas and latency overhead

```sh
python benchmarks/openclaw_memory_eval.py \
  --base-url http://127.0.0.1:8765 \
  --model qwen2.5-0.5b-mlx-test
```

For the released Qwen3 adapter path, use:

```sh
export DELTA_MEM_ADAPTER_DIR=/path/to/delta-mem-qwen3-4b-instruct-mlx-adapter
python benchmarks/full_delta_mem_bench.py \
  --max-tokens 32 \
  --warmup 1
```

Current local benchmark findings, including OpenClaw retrieved-memory preload
results, QMD search/vsearch comparisons, ygraph notes, and preload-size totals are in
`wiki/Benchmark-Findings.md`.

## Repository Layout

- `delta-mem-sidecar/`: FastAPI sidecar and runtime implementations.
- `benchmarks/`: OpenClaw and paper-style verification probes.
- `wiki/`: concise project wiki for users and operators.
- `openclaw.example.json5`: OpenClaw provider snippet.
