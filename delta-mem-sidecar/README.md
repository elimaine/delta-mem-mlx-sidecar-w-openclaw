# delta-mem-sidecar

Minimal FastAPI sidecar for delta-Mem runtime experiments.

The default runtime is fake and deterministic. It exists to validate the API,
session-key plumbing, state isolation, and reset behavior on machines without
the CUDA-first official runtime. Set `DELTA_MEM_RUNTIME=official` to load the
real upstream δ-mem adapter.

On Apple Silicon, use `DELTA_MEM_RUNTIME=mlx` for the efficient MLX-LM backbone
path. The MLX runtime is currently a backbone runtime; the δ-mem adapter math is
being ported separately because upstream δ-mem adapters are not standard MLX-LM
adapters.

## API

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /delta/session/reset`
- `GET /delta/state/{state_key}/metadata`
- `POST /delta/state/{state_key}/reset`

`POST /v1/chat/completions` supports OpenAI-compatible non-streaming and
streaming responses and requires a stable logical session key:

```text
X-Delta-Mem-Session-Key: <stable logical session key>
```

The sidecar hashes the logical key before internal storage so raw channel,
thread, or account identifiers do not become storage identifiers.

## Attention State

The sidecar accepts retrieved memory snippets without adding them to the user
prompt. These snippets are used to warm the per-session delta state before the
real assistant response:

- request body: `attention_state`, `attentionState`, or `delta_attention_state`
- request header: `X-Delta-Attention-State` containing text or JSON snippets

Responses include `X-Delta-Attention-State-Count` and
`X-Delta-Attention-State-Source` so the caller can verify whether explicit
memory snippets were preloaded.

By default, the sidecar prepends a short system notice to each request:

```text
Your LLM is running through delta-mem-mlx. This is an experimental MLX-native delta-memory adapter that keeps per-session neural state keyed by X-Delta-Mem-Session-Key. It may improve continuity and recall across turns, but recall can be incomplete or wrong; prefer explicit recent context when accuracy matters.
```

Set `DELTA_MEM_SESSION_PREAMBLE` to replace that text. Set it to an empty value
to disable the preamble.

## Run

```sh
python3.11 -m venv .venv
. .venv/bin/activate
pip install -e ".[test]"
uvicorn delta_mem_sidecar.app:create_app --factory --host 127.0.0.1 --port 8765
```

## Run With The Official δ-mem Adapter

The official adapter path currently targets NVIDIA/CUDA. Install the upstream
`declare-lab/delta-Mem` repo and its requirements first, then put that repo on
`PYTHONPATH` when starting this sidecar.

```sh
git clone https://github.com/declare-lab/delta-Mem.git /path/to/delta-Mem
cd /path/to/delta-Mem
bash scripts/setup_uv_env.sh
huggingface-cli download declare-lab/delta-mem_qwen3_4b-instruct \
  --local-dir /path/to/delta-mem_qwen3_4b-instruct
```

Start the sidecar from this package with the real runtime selected:

```sh
git clone https://github.com/elimaine/delta-mem-mlx-sidecar.git
cd delta-mem-mlx/delta-mem-sidecar
. /path/to/delta-Mem/.venv/bin/activate
pip install -e .
PYTHONPATH=/path/to/delta-Mem \
DELTA_MEM_RUNTIME=official \
DELTA_MEM_MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 \
DELTA_MEM_ADAPTER_DIR=/path/to/delta-mem_qwen3_4b-instruct \
DELTA_MEM_DEVICE=cuda:0 \
DELTA_MEM_DTYPE=bfloat16 \
uvicorn delta_mem_sidecar.app:create_app --factory --host 127.0.0.1 --port 8765
```

Optional runtime settings:

- `DELTA_MEM_ATTN_IMPLEMENTATION`, for example `flash_attention_2`
- `DELTA_MEM_MAX_NEW_TOKENS`, default `2048`
- `DELTA_MEM_STATE_DIR`, optional directory for hashed per-session state
  persistence

The sidecar uses upstream `DeltaMemChatSession.generate_reply`, so each turn
ingests the latest user message and materializes the generated assistant message
into δ-state. Because upstream δ-state mutates on the loaded model object,
requests are serialized while the sidecar swaps the active session snapshot.

## Test

```sh
pytest
```

## Runtime Seam

Implement `DeltaRuntime` in `delta_mem_sidecar.runtime` to connect the official
delta-Mem runtime. The sidecar assumes each session has an opaque mutable
`RuntimeState`; a real adapter can store tensor handles or serialized blobs
behind that seam.

`delta_mem_sidecar.official_runtime.OfficialDeltaRuntime` is the real adapter
implementation. It lazy-loads the upstream model only when the first generation
request arrives, so import-time health checks and fake-runtime tests do not need
Torch, Transformers, or model assets.

## Run With MLX On Apple Silicon

Install the sidecar with the MLX optional extra:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
pip install -e ".[mlx,test]"
```

Start with an MLX-compatible or MLX-quantized model:

```sh
hf download mlx-community/Qwen2.5-0.5B-Instruct-4bit

DELTA_MEM_RUNTIME=mlx \
DELTA_MEM_MODEL_PATH=mlx-community/Qwen2.5-0.5B-Instruct-4bit \
DELTA_MEM_MODEL_ID=qwen2.5-0.5b-mlx-test \
DELTA_MEM_MAX_NEW_TOKENS=512 \
uvicorn delta_mem_sidecar.app:create_app --factory --host 127.0.0.1 --port 8765
```

This path uses `mlx_lm.load` and `mlx_lm.generate`, which are optimized for
Apple Silicon unified memory. It is the target execution base for the native
δ-mem port, but it does not yet load `declare-lab/delta-mem_qwen3_4b-instruct`.

Validated smoke-test model:

- `mlx-community/Qwen2.5-0.5B-Instruct-4bit`

## Run With The Released δ-mem Adapter On MLX

The released δ-mem adapter targets Qwen3-4B, not the small Qwen2.5 0.5B smoke
model. Download the compatible MLX backbone and released adapter:

```sh
hf download mlx-community/Qwen3-4B-Instruct-2507-4bit
hf download declare-lab/delta-mem_qwen3_4b-instruct
```

Convert the released Torch checkpoint once into an MLX-native runtime artifact:

```sh
python -m delta_mem_sidecar.convert_adapter \
  /Users/elimaine/.cache/huggingface/hub/models--declare-lab--delta-mem_qwen3_4b-instruct/snapshots/c46dc31155608e412d44bf56638d5a6f856f2e7e
```

This writes `delta_mem_adapter_mlx.npz` next to the downloaded adapter files.
After that, the sidecar runtime loads the MLX `.npz` directly and does not need
PyTorch to start.

Start the sidecar with both model and adapter:

```sh
DELTA_MEM_RUNTIME=mlx \
DELTA_MEM_MODEL_PATH=mlx-community/Qwen3-4B-Instruct-2507-4bit \
DELTA_MEM_ADAPTER_DIR=/Users/elimaine/.cache/huggingface/hub/models--declare-lab--delta-mem_qwen3_4b-instruct/snapshots/c46dc31155608e412d44bf56638d5a6f856f2e7e \
DELTA_MEM_MODEL_ID=delta-mem-qwen3-4b-mlx \
DELTA_MEM_MAX_NEW_TOKENS=256 \
uvicorn delta_mem_sidecar.app:create_app --factory --host 127.0.0.1 --port 8765
```

This path wraps all 36 Qwen3 attention layers with MLX δ-mem Q/O corrections
and stores per-session δ-state snapshots behind `X-Delta-Mem-Session-Key`.

Set `DELTA_MEM_STATE_DIR` to persist hashed per-session runtime state and MLX
delta tensors across sidecar restarts. The sidecar persists delta state only,
not KV caches or full chat snapshots; see `docs/delta-state-persistence.md`.

## Benchmark Findings

Current local benchmark findings, including session replay, retrieved-memory
preloads, and context-size totals are summarized in
`../docs/benchmark-findings.md`. The repo is:

```text
https://github.com/elimaine/delta-mem-mlx-sidecar
```
