# Agent Notes

This repository packages a local MLX sidecar for delta-mem research.

Use the toy model path by default:

```text
DELTA_MEM_RUNTIME=mlx
DELTA_MEM_MODEL_PATH=mlx-community/Qwen2.5-0.5B-Instruct-4bit
DELTA_MEM_MODEL_ID=qwen2.5-0.5b-mlx-test
```

Stateful calls route through OpenAI-compatible chat completions and require:

```text
X-Delta-Mem-Session-Key: <stable logical session key>
```

Do not add private planning notes, local status logs, machine caches, or downloaded model weights to this repo.

Do not vendor external agent plugins by default.
