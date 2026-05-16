# delta-mem-sidecar

FastAPI sidecar exposing local model runtimes through an OpenAI-compatible `/v1/chat/completions` API, either directly or behind OpenClaw.

## Default Toy MLX Runtime

```sh
python3.11 -m venv .venv
. .venv/bin/activate
pip install -e ".[mlx,test]"

DELTA_MEM_RUNTIME=mlx \
DELTA_MEM_MODEL_PATH=mlx-community/Qwen2.5-0.5B-Instruct-4bit \
DELTA_MEM_MODEL_ID=qwen2.5-0.5b-mlx-test \
DELTA_MEM_MAX_NEW_TOKENS=256 \
uvicorn delta_mem_sidecar.app:create_app --factory --host 127.0.0.1 --port 8765
```

## API

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /delta/session/reset`
- `GET /delta/state/{state_key}/metadata`
- `POST /delta/state/{state_key}/reset`

Chat requests require:

```text
X-OpenClaw-Session-Key: <stable logical session key>
```

## Optional Released δ-mem Adapter

The public adapter targets Qwen3-4B:

- `mlx-community/Qwen3-4B-Instruct-2507-4bit`
- `ofthetrees/delta-mem-qwen3-4b-instruct-mlx-adapter`

Download the MLX backbone and converted adapter:

```sh
hf download mlx-community/Qwen3-4B-Instruct-2507-4bit
hf download ofthetrees/delta-mem-qwen3-4b-instruct-mlx-adapter
```

If you need to rebuild the MLX artifact from the upstream Torch adapter:

```sh
hf download declare-lab/delta-mem_qwen3_4b-instruct
python -m delta_mem_sidecar.convert_adapter /path/to/delta-mem_qwen3_4b-instruct
```

Run with the adapter:

```sh
DELTA_MEM_RUNTIME=mlx \
DELTA_MEM_MODEL_PATH=mlx-community/Qwen3-4B-Instruct-2507-4bit \
DELTA_MEM_ADAPTER_DIR=/path/to/delta-mem-qwen3-4b-instruct-mlx-adapter \
DELTA_MEM_MODEL_ID=delta-mem-qwen3-4b-mlx \
DELTA_MEM_MAX_NEW_TOKENS=256 \
uvicorn delta_mem_sidecar.app:create_app --factory --host 127.0.0.1 --port 8765
```

## Tests

```sh
pytest
```
