# Benchmark Findings

These are small local benchmark runs for the MLX sidecar and optional δ-mem adapter path. They are useful for implementation direction, not paper-grade claims.

Public repo: https://github.com/elimaine/delta-mem-mlx-sidecar-w-openclaw

## Shareable Summary

The base model consistently performed better in the strongest local comparisons with the δ-mem TSW adapter attached. The most interesting runs ranged from about `1.07x` to `1.30x` score lift, at the cost of about `1.26x` to `1.69x` probe-latency slowdown.

That alone is reason to be excited about this.

Preloading memory into the weights has proven difficult to pin down. Possibly because of the small model size. I am currently exploring this; see the benchmark details below.

The important caveat is that context volume by itself was not predictive. Compact, relevant QMD context worked better than larger, richer wiki/ygraph context. That suggests the current bottleneck may be retrieval quality, fact density, and wording shape rather than simply adding more memory.

## Most Interesting Results

| Test | Plain/base score | δ-mem score | Lift | Pass-rate change | Slowdown | Why it matters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Fixed LoCoMo-10 session-context | `0.4667` | `0.5000` | `1.07x` | n/a | `1.95x` total runtime | Corrected the bad no-context baseline; shows a small but plausible memory gain. |
| Sanitized OpenClaw raw replay | `0.5701` | `0.6667` | `1.17x` | `6/8 -> 7/8` | `1.30x` probe latency | Best early practical OpenClaw-shaped result. |
| Hybrid memory + half replay | `0.5625` | `0.6208` | `1.10x` | `5/8 -> 6/8` | `1.49x` probe latency | Shows structured memory plus partial replay can help. |
| QMD search snippets + target replay | `0.5625` | `0.7292` | `1.30x` | `5/8 -> 7/8` | `1.63x` probe latency | Strongest injected-context result from BM25/full-text retrieval. |
| QMD vsearch snippets + target replay | `0.5625` | `0.7292` | `1.30x` | `5/8 -> 7/8` | `1.48x` probe latency | Vector retrieval matched QMD search quality with slightly lower latency ratio. |
| Reduced QMD deterministic context, ~490 injected tokens | `0.5625` | `0.7292` | `1.30x` | `5/8 -> 7/8` | `1.66x` probe latency | Smaller, ranked QMD sections preserved the right facts. |
| Reduced QMD synthesized context, ~95 injected tokens | `0.5625` | `0.7292` | `1.30x` | `5/8 -> 7/8` | `1.69x` probe latency | Very small fact-style context tied the best score, suggesting fact density matters more than volume. |

## Current Read

The results are encouraging because the adapter can improve behavior on memory-shaped tasks, but the mechanism is not yet cleanly controlled. Memory preloading can help, hurt, or do nothing depending on what gets injected. The strongest signal so far is not "more context"; it is "the right facts, in the right form, at the right point in the session."

The δ-mem paper reports meaningful gains using Qwen3-4B-Instruct: `1.10x` average over the frozen backbone, `1.31x` on MemoryAgentBench, and `1.20x` on LoCoMo. Our local synthetic paper-style probes were flat, but corrected LoCoMo and OpenClaw-shaped replay both showed positive signal.

## Retrieval Terms

QMD is a local markdown retrieval tool. In these tests, `qmd search` means BM25/full-text retrieval and `qmd vsearch` means embedding/vector similarity retrieval. The retrieval tests used an isolated QMD index over sanitized OpenClaw memory fixtures so they did not mutate an operator's live index.

Ygraph thoughts are graph-derived thought atoms from the local OpenClaw/ygraph workspace. They are candidate memory statements with provenance-style metadata. In this benchmark they were selected deterministically by keyword overlap, then injected as context.

Cognee is a graph/RAG-style memory system referenced by the local memory-stack notes. It was not on the active hot path for these benchmark runs; the ygraph-selected thoughts mostly referenced Cognee as background process-memory context.

## Context Injection Results

This table compares OpenClaw-shaped memory/context variants using the same 8-probe evaluation set.

| Variant | Context events | Est. tokens | Plain | δ-mem | Ratio | Δ | Pass | Probe latency ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| memorg only | `12` | `~386` | `0.5625` | `0.5292` | `0.94x` | `-0.0333` | `5/8 -> 5/8` | `1.46x` |
| hybrid memory + half replay | `18` | `~550` | `0.5625` | `0.6208` | `1.10x` | `+0.0583` | `5/8 -> 6/8` | `1.49x` |
| related sessions preload + target replay | `19` | `~538` | `0.5625` | `0.5958` | `1.06x` | `+0.0333` | `5/8 -> 6/8` | `1.53x` |
| curated wiki + target replay | `15` | `~678` | `0.5625` | `0.5542` | `0.99x` | `-0.0083` | `5/8 -> 6/8` | `1.44x` |
| deterministic lexical context list + target replay | `19` | `~893` | `0.5625` | `0.5542` | `0.99x` | `-0.0083` | `5/8 -> 6/8` | `1.44x` |
| ygraph keyword thoughts + target replay | `8` | `~1069` | `0.5625` | `0.6042` | `1.07x` | `+0.0417` | `5/8 -> 6/8` | `1.54x` |
| QMD search snippets + target replay | `8` | `~605` | `0.5625` | `0.7292` | `1.30x` | `+0.1667` | `5/8 -> 7/8` | `1.63x` |
| QMD vsearch snippets + target replay | `8` | `~608` | `0.5625` | `0.7292` | `1.30x` | `+0.1667` | `5/8 -> 7/8` | `1.48x` |

## Reduced QMD Context Sweep

This sweep kept the target replay constant and varied only the injected QMD context budget. "Deterministic" means full QMD-hit document sections ranked by lexical overlap with probe terms. "Synthesized" means direct fact-style context synthesized from the same retrieved sanitized fixtures.

| Method | Injected budget | Actual injected tokens | Total tokens | Plain | δ-mem | Ratio | Δ | Pass | Latency ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| deterministic | `500` | `~490` | `~688` | `0.5625` | `0.7292` | `1.30x` | `+0.1667` | `5/8 -> 7/8` | `1.66x` |
| deterministic | `400` | `~398` | `~596` | `0.5625` | `0.5542` | `0.99x` | `-0.0083` | `5/8 -> 6/8` | `1.57x` |
| deterministic | `300` | `~296` | `~493` | `0.5625` | `0.6042` | `1.07x` | `+0.0417` | `5/8 -> 6/8` | `1.49x` |
| deterministic | `200` | `~185` | `~382` | `0.5625` | `0.6042` | `1.07x` | `+0.0417` | `5/8 -> 6/8` | `1.26x` |
| deterministic | `100` | `~98` | `~295` | `0.5625` | `0.6042` | `1.07x` | `+0.0417` | `5/8 -> 6/8` | `1.49x` |
| synthesized | `500` | `~440` | `~638` | `0.5625` | `0.5542` | `0.99x` | `-0.0083` | `5/8 -> 6/8` | `1.48x` |
| synthesized | `400` | `~390` | `~588` | `0.5625` | `0.5542` | `0.99x` | `-0.0083` | `5/8 -> 6/8` | `1.53x` |
| synthesized | `300` | `~299` | `~496` | `0.5625` | `0.5292` | `0.94x` | `-0.0333` | `5/8 -> 5/8` | `1.61x` |
| synthesized | `200` | `~194` | `~392` | `0.5625` | `0.5542` | `0.99x` | `-0.0083` | `5/8 -> 6/8` | `1.67x` |
| synthesized | `100` | `~95` | `~293` | `0.5625` | `0.7292` | `1.30x` | `+0.1667` | `5/8 -> 7/8` | `1.69x` |

Reducing context can help, but not smoothly by token count. The deterministic 500-token pack and synthesized 100-token pack tied for best result (`1.30x`). Intermediate budgets often regressed toward zero or below baseline. This strongly suggests the important variable is whether the reduced context preserves the right high-priority facts in a usable order, not raw context volume.

## Reproduction Outline

- Build an isolated QMD index for the sanitized memory corpus.
- Run `qmd search` and `qmd vsearch` against the same corpus and query.
- Convert retrieved snippets into injected context fixtures with the same target replay.
- Compare plain backbone and δ-mem sidecar runs with the same probes, temperature, and max-token settings.
- Record context event count, estimated token count, score, pass rate, and latency for each run.
