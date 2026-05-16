# Benchmark Findings

These are small local benchmark runs for the MLX sidecar and optional δ-mem adapter path. They are useful for implementation direction, not paper-grade claims.

Public repo: https://github.com/elimaine/delta-mem-mlx-sidecar-w-openclaw

## Summary

- The δ-mem paper reports meaningful gains using Qwen3-4B-Instruct: `1.10x` average over the frozen backbone, `1.31x` on MemoryAgentBench, and `1.20x` on LoCoMo.
- Our local synthetic paper-style probes were flat: `0.5129` plain vs `0.5129` δ-mem.
- The original local LoCoMo `3.67x` number was inflated by a bad plain baseline with no conversation context. The corrected session-context comparison is `0.4667` plain vs `0.5000` δ-mem, or `1.07x`.
- The strongest OpenClaw-shaped replay result before retrieval tests was raw sanitized replay: `0.5701` plain vs `0.6667` δ-mem, or `1.17x`.
- QMD search and QMD vsearch produced the strongest current injected-context result: both reached `0.7292` δ-mem vs `0.5625` plain, or `1.30x`.
- Context volume alone is not predictive. The best QMD packs were only about `~605` tokens, while larger ygraph and deterministic packs were weaker. Relevance and direct factual wording seem more important than raw context size.
- δ-mem is slower in these local tests. OpenClaw probe latency overhead ranged from about `1.30x` to `1.63x`.

## Terms

QMD is a local markdown retrieval tool. In these tests, `qmd search` means BM25/full-text retrieval and `qmd vsearch` means embedding/vector similarity retrieval. A fresh named QMD index was built over sanitized OpenClaw memory fixtures so the test did not mutate the live OpenClaw/Sandy index.

Ygraph thoughts are graph-derived thought atoms from the local OpenClaw/ygraph workspace. They are candidate memory statements with provenance-style metadata. In this benchmark they were selected deterministically by keyword overlap, then injected as context.

Cognee is a graph/RAG-style memory system referenced by the local memory-stack notes. It was not on the active hot path for these benchmark runs; the ygraph-selected thoughts mostly referenced Cognee as background process-memory context.

## OpenClaw Context Totals

| Variant | Context events | Context chars | Est. tokens | System chars | Plain | δ-mem | Ratio | Δ | Pass | Probe latency ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| memorg only | `12` | `1542` | `~386` | `117` | `0.5625` | `0.5292` | `0.94x` | `-0.0333` | `5/8 -> 5/8` | `1.46x` |
| hybrid memory + half replay | `18` | `2201` | `~550` | `117` | `0.5625` | `0.6208` | `1.10x` | `+0.0583` | `5/8 -> 6/8` | `1.49x` |
| related sessions preload + target replay | `19` | `2154` | `~538` | `322` | `0.5625` | `0.5958` | `1.06x` | `+0.0333` | `5/8 -> 6/8` | `1.53x` |
| Forevergreen wiki + target replay | `15` | `2713` | `~678` | `1826` | `0.5625` | `0.5542` | `0.99x` | `-0.0083` | `5/8 -> 6/8` | `1.44x` |
| deterministic lexical context list + target replay | `19` | `3571` | `~893` | `1487` | `0.5625` | `0.5542` | `0.99x` | `-0.0083` | `5/8 -> 6/8` | `1.44x` |
| ygraph keyword thoughts + target replay | `8` | `4276` | `~1069` | `3550` | `0.5625` | `0.6042` | `1.07x` | `+0.0417` | `5/8 -> 6/8` | `1.54x` |
| QMD search snippets + target replay | `8` | `2419` | `~605` | `1693` | `0.5625` | `0.7292` | `1.30x` | `+0.1667` | `5/8 -> 7/8` | `1.63x` |
| QMD vsearch snippets + target replay | `8` | `2434` | `~608` | `1708` | `0.5625` | `0.7292` | `1.30x` | `+0.1667` | `5/8 -> 7/8` | `1.48x` |

## Interpretation

The current evidence does not show a simple sweet spot based only on context size. Very small memorg context underperformed, mid-sized hybrid/QMD contexts helped, and larger rich-language context packs often regressed toward zero gain. The QMD results suggest that retrieval quality and direct fact density matter more than total context.

The richer-language hypothesis is still plausible: wiki/ygraph process prose is useful to a human, but benchmark probes ask for direct recall of facts such as adapter location, exact LoCoMo numbers, and OpenClaw replay scores. A better next test is to normalize retrieved context into short fact triples and compare that against the same context left as rich prose.

## Reproduction Notes

- QMD search index: fresh named index `delta-mem-openclaw-fresh`.
- Corpus: sanitized OpenClaw markdown fixtures derived from ignored local benchmark results.
- QMD vector embedding ran on CPU inside Lima because Vulkan was missing.
- Result JSON and JSONL artifacts are ignored under `benchmarks/results/`.
