# workspace-indexer

A hybrid (semantic + keyword) index over a multi-repo workspace, built so an LLM
can find code and documentation by meaning instead of by grep.

Points at directories from a config file, chunks them with a strategy chosen per
file type (tree-sitter for code, heading-aware for markdown), embeds the chunks,
and stores them in Qdrant with both dense and BM25 sparse vectors for
reciprocal-rank-fusion search plus optional reranking. Reindexing is
incremental: an unchanged file costs one `stat()` and no API call.

See `docs/iteration-1-plan.md` for the design and the reasoning behind it.

## Getting started

```bash
poetry install
cp config/workspace.example.yaml config/workspace.yaml   # what to index
cp .env.example .env                                     # how to index
```

Everything runs locally with no API key if you want it to:

```bash
# .env
EMBEDDING_MODEL=fastembed:BAAI/bge-small-en-v1.5
EMBEDDING_DIMENSIONS=384
```

...or point it at a paid provider by changing one string:

```bash
EMBEDDING_MODEL=voyageai:voyage-code-4
EMBEDDING_DIMENSIONS=2048
VOYAGE_API_KEY=...
```

## Commands

```bash
workspace-indexer index --dry-run     # chunk plan + token estimate, no API calls
workspace-indexer index               # incremental; unchanged files cost a stat()
workspace-indexer search "how does incremental reindexing decide to skip a file"
workspace-indexer status              # what is indexed, in which spaces, what runs cost
workspace-indexer explain path/to/file.py   # the chunks one file produces
workspace-indexer reproject -d 1024   # narrower collection, no re-embedding
workspace-indexer eval                # recall@k and MRR@k against config/eval.yaml
```

`--dry-run` exists so chunking can be tuned without paying to iterate.

## Development

```bash
poetry run pytest -q                     # full suite
poetry run pytest -m "not integration"   # fast, fully offline
poetry run ruff check src/ tests/
poetry run pyright                       # strict; must be 0 errors
```
