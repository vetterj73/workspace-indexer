# workspace-indexer

A hybrid (semantic + keyword) index over a multi-repo workspace, built so an LLM
can find code and documentation by meaning instead of by grep.

Points at a set of directories from a config file, chunks them with a strategy
chosen per file type (tree-sitter for code, heading-aware for markdown), embeds
the chunks, and stores them in Qdrant with dense and BM25 sparse vectors for
reciprocal-rank-fusion search plus optional reranking.

See `config/workspace.example.yaml` and `.env.example` to get started.

```bash
poetry install
cp .env.example .env            # add VOYAGE_API_KEY
cp config/workspace.example.yaml config/workspace.yaml
poetry run workspace-indexer index --dry-run
poetry run workspace-indexer index
poetry run workspace-indexer search "how does incremental reindexing decide to skip a file"
```
