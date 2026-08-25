# Eval baselines

Recorded before any classifier work, so the iteration-2 decision rests on
measurement rather than on the version of the system that happened to be
running when the idea came up.

Interim: these live in a file until #1 lands and the manifest records them
automatically. A number in a document is still better than a number in a chat
log, but it is not the goal.

Corpus: `~/src`, 1,004 files, 10,702 chunks. 16 eval cases, RRF fusion,
`limit=10`, rerank `candidates=30`.

## Results

| # | Embedder | Reranker | recall@10 | MRR@10 | misses |
|---|---|---|---|---|---|
| 1 | `fastembed:BAAI/bge-small-en-v1.5` @384 | local cross-encoder | 0.781 | 0.582 | 4 |
| 2 | `voyageai:voyage-code-4` @2048 | local cross-encoder | 0.844 | 0.669 | 3 |
| 3 | `voyageai:voyage-code-4` @2048 | `voyageai:rerank-2.5-lite` | **0.875** | **0.760** | 3 |
| 4 | `voyageai:voyage-code-4` @1024 | `voyageai:rerank-2.5-lite` | **0.875** | **0.760** | 3 |

One variable changes per row, so each delta is attributable.

## What the numbers say

**A code-trained embedder is worth more than the reranker.** Run 1 → 2 changed
only the embedding model and moved recall +0.063 and MRR +0.087. Run 2 → 3
changed only the reranker and moved recall +0.031, MRR +0.091. Both matter;
the embedder matters more for *finding* the document and the reranker for
*ranking it first*, which is exactly the division of labour the design assumed.

**2048 dimensions buy nothing over 1024 here.** Runs 3 and 4 are identical to
three decimal places on every case. This is the question the iteration-1 plan
flagged and deliberately deferred to measurement, and the answer on this corpus
is that the wider vector is pure storage cost.

The plan predicted this, for the stated reason: with a reranker in the pipeline
the dense branch only has to land the right chunk somewhere in the candidate
set, and extra dimensions mainly sharpen fine-grained ranking precision -- the
work the reranker is already doing.

**Recommendation: index at 2048 and serve from the 1024 reprojection.** The
asymmetry still holds -- 2048 → 1024 is free, 1024 → 2048 is a full re-embed --
so indexing wide preserves the option at no ongoing cost.

## Timings, same corpus

| Operation | Local `bge-small` (CPU ONNX) | `voyage-code-4` (API) |
|---|---|---|
| Full index, 10,702 chunks | 2h 40m | **2m 35s** |
| Incremental, nothing changed | 21s | 21s |
| Reproject 2048 → 1024 | n/a | 1m 17s, **no API calls** |
| Full eval, 16 cases | 60s | ~90s |

The API is ~62× faster on a full index. The incremental number is unchanged
because it never reaches the embedder -- that is the manifest doing its job.

Cost: 3.15M tokens at $0.12/M, against a 200M-token free tier. Effectively
zero. Note our own cost tracking reported `$0.0000` because genai-prices does
not yet price `voyage-code-4`; the run recorded tokens correctly.

## The three remaining misses

The same three survive every configuration, which is what makes them
interesting rather than noise.

1. **"how do we avoid re-embedding a file that was only reformatted"** —
   run 3 does return `state/manifest.py` first, but the case expects
   `chunking/file_reader.py` too and only gets one of two. Partial credit; the
   dataset may simply be over-specified.

2. **"what conventions must I follow when structuring a new module"** —
   returns *other projects'* prompt YAML instead of `workspace-indexer/CLAUDE.md`.
   Never fixed by any embedder or reranker. **This is the classifier's case.**
   The right answer is a normative document; the wrong answers are other
   repositories' agent instructions that merely discuss conventions.

3. **"how should this project handle logging and observability"** — finds
   `docs/iteration-1-plan.md` at rank 1 but misses `CLAUDE.md`, so it scores
   0.5. Same shape as case 2: the normative document loses to the discursive
   one.

Two of the three are the exact discrimination `doc_type` is meant to provide,
and no amount of better embedding fixed them. That is the evidence for #5, and
it is stronger than it was before these runs.

## Reproducing

```bash
# .env
EMBEDDING_MODEL=voyageai:voyage-code-4
EMBEDDING_DIMENSIONS=2048
VOYAGE_API_KEY=...

workspace-indexer index
workspace-indexer eval                       # run 3
workspace-indexer reproject --dimensions 1024
workspace-indexer eval --dimensions 1024     # run 4
```
