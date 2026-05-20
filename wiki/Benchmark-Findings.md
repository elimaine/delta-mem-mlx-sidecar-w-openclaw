# Benchmark Findings

These are small local benchmark runs for the MLX sidecar and optional delta-mem
adapter path. They are useful for implementation direction, not paper-grade
claims.

Repo: https://github.com/elimaine/delta-mem-mlx-sidecar

## Summary

- The delta-mem paper reports meaningful gains using Qwen3-4B-Instruct: `1.10x`
  average over the frozen backbone, `1.31x` on MemoryAgentBench, and `1.20x`
  on LoCoMo.
- Local synthetic paper-style probes were flat: `0.5129` plain vs `0.5129`
  delta-mem.
- The corrected local LoCoMo session-context comparison is `0.4667` plain vs
  `0.5000` delta-mem, or `1.07x`.
- Sanitized session replay showed directional signal in smaller/local fixture
  runs, but strict broad transcript replay was weak on exact recall:
  `+0.0391` absolute score delta from a `0.0000` no-history base.
- Retrieved-memory preload experiments produced the strongest local signals when
  snippets were short, direct, and fact-dense.
- Preload volume alone is not predictive. Relevance and direct factual wording
  seem more important than raw preload size.

## Retrieved-Memory Preload Totals

| Variant | Preload events | Est. preload tokens | Plain | delta-mem | Ratio | Delta | Probe latency ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| memory organization only | `12` | `~386` | `0.5625` | `0.5292` | `0.94x` | `-0.0333` | `1.46x` |
| hybrid memory + half replay | `18` | `~550` | `0.5625` | `0.6208` | `1.10x` | `+0.0583` | `1.49x` |
| related sessions preload + target replay | `19` | `~538` | `0.5625` | `0.5958` | `1.06x` | `+0.0333` | `1.53x` |
| wiki + target replay | `15` | `~678` | `0.5625` | `0.5542` | `0.99x` | `-0.0083` | `1.44x` |
| deterministic lexical context list + target replay | `19` | `~893` | `0.5625` | `0.5542` | `0.99x` | `-0.0083` | `1.44x` |
| graph keyword thoughts + target replay | `8` | `~1069` | `0.5625` | `0.6042` | `1.07x` | `+0.0417` | `1.54x` |
| retrieved search snippets + target replay | `8` | `~605` | `0.5625` | `0.7292` | `1.30x` | `+0.1667` | `1.63x` |
| retrieved vector snippets + target replay | `8` | `~608` | `0.5625` | `0.7292` | `1.30x` | `+0.1667` | `1.48x` |

## Interpretation

The current evidence does not show a simple sweet spot based only on preload
size. Very small preload underperformed, mid-sized retrieved preloads helped,
and larger rich-language preload packs often regressed toward zero gain.
Retrieval quality and direct fact density appear more important than total
preload volume.

## Reproduction Notes

- Use `benchmarks/session_memory_eval.py` for compact memory-regression probes.
- Use `benchmarks/session_replay_eval.py` for a single sanitized history file.
- Use `benchmarks/transcript_toolbelt.py` for export/sanitize/probe/run/report
  workflows over batches of JSONL transcripts.
- Result JSON and JSONL artifacts are ignored under `benchmarks/results/`.
