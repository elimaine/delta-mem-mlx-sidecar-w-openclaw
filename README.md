# delta-mem-mlx-sidecar

Research workspace for running and evaluating delta-mem on Apple Silicon through
a local MLX/OpenAI-compatible sidecar.

The sidecar lives in `delta-mem-sidecar/`. It supports fake, official upstream,
MLX backbone, and MLX delta-mem runtime tracks. Session state is isolated by the
generic header:

```text
X-Delta-Mem-Session-Key: <stable logical session key>
```

## Quick Start

```sh
cd delta-mem-sidecar
python3.11 -m venv .venv
. .venv/bin/activate
pip install -e ".[mlx,test]"
uvicorn delta_mem_sidecar.app:create_app --factory --host 127.0.0.1 --port 8765
```

## Benchmarks

- `benchmarks/session_memory_eval.py`: compact session-memory probes.
- `benchmarks/session_replay_eval.py`: replay one sanitized transcript/history
  file and run behavior probes.
- `benchmarks/transcript_toolbelt.py`: export, sanitize, probe, run, and report
  batches of JSONL transcripts.
- `benchmarks/full_delta_mem_bench.py`: compare Qwen3-4B MLX backbone with and
  without the released delta-mem adapter.

Benchmark notes are in `wiki/Benchmark-Findings.md`.
