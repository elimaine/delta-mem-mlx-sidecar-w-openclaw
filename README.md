# delta-mem-mlx-sidecar-w-openclaw

OpenAI-compatible sidecar for running local MLX models behind OpenClaw, with an optional δ-mem adapter path for session-shaped memory.

δ-mem augments a frozen transformer with a compact online memory state that reads and writes through attention. The paper reports an average 1.10x gain over the frozen backbone, 1.31x on MemoryAgentBench, and 1.20x on LoCoMo using a small online state rather than full fine-tuning or explicit context extension.

References:

- Paper: https://arxiv.org/abs/2605.12357
- Hugging Face paper page: https://huggingface.co/papers/2605.12357
- Released adapter: https://huggingface.co/declare-lab/delta-mem_qwen3_4b-instruct

## Expected From User

- Apple Silicon Mac for the default MLX path.
- Python 3.11+.
- Hugging Face access for the model downloads.
- OpenClaw gateway if you want agent routing through this sidecar.
- A stable `X-OpenClaw-Session-Key` per logical agent/session.

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

Use `openclaw.example.json5` as the provider shape. The important parts are:

- `baseUrl` points at this sidecar's `/v1` root.
- `transportProtocol` is `openai-chat-completions`.
- `X-OpenClaw-Session-Key` is present and stable for a logical session.

When OpenClaw runs inside Lima, `host.lima.internal:8765` is the expected route back to the macOS sidecar. For host-only testing, use `127.0.0.1:8765`.

The `openclaw-plugin/` directory contains the live gateway's embedded harness plugin export. It is included as a reference/plugin starting point for OpenClaw runtimes that need a custom harness instead of only a model provider.

## Released δ-mem Adapter

The released public adapter targets `Qwen/Qwen3-4B-Instruct-2507`, not the small toy model. To run it locally through MLX, download the compatible MLX backbone and adapter, convert the adapter once, then start the sidecar with `DELTA_MEM_ADAPTER_DIR`.

```sh
hf download mlx-community/Qwen3-4B-Instruct-2507-4bit
hf download declare-lab/delta-mem_qwen3_4b-instruct
```

See `delta-mem-sidecar/README.md` for the exact optional adapter run command.

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
python benchmarks/full_delta_mem_bench.py \
  --max-tokens 32 \
  --warmup 1
```

## Repository Layout

- `delta-mem-sidecar/`: FastAPI sidecar and runtime implementations.
- `benchmarks/`: OpenClaw and paper-style verification probes.
- `openclaw-plugin/`: live gateway embedded harness plugin export.
- `wiki/`: concise project wiki for users and operators.
- `openclaw.example.json5`: OpenClaw provider snippet.
