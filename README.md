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
poetry install                     # add --extras pdf to index PDFs
cp config/workspace.example.yaml config/workspace.yaml   # what to index
cp .env.example .env                                     # how to index
```

`poetry install` also installs the dev group (ruff, pyright, pytest — about
66 MB). To only *run* the indexer, `poetry install --only main` skips it.

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
workspace-indexer watch               # reindex as files change
workspace-indexer serve               # MCP server over stdio, for an agent
```

`--dry-run` exists so chunking can be tuned without paying to iterate.

`eval` can score an MCP tool rather than the raw search path, which is the only
way to measure what a document-type filter is actually worth:

```bash
workspace-indexer eval --group guidance --tool search          # baseline
workspace-indexer eval --group guidance --tool find_guidance   # with the filter
```

## Using it from Claude Code

`serve` speaks MCP over stdio: the client starts the process and talks to it
down a pipe, so there is no port and nothing left running. Register it once —
in `.mcp.json` at a project root, or with `claude mcp add`:

```json
{
  "mcpServers": {
    "workspace-indexer": {
      "command": "/absolute/path/to/workspace-indexer/.venv/bin/workspace-indexer",
      "args": ["serve", "--config", "/absolute/path/to/workspace-indexer/config/workspace.yaml"],
      "env": {
        "QDRANT_MODE": "server",
        "QDRANT_URL": "http://127.0.0.1:6333",
        "EMBEDDING_MODEL": "voyageai:voyage-code-4",
        "EMBEDDING_DIMENSIONS": "1024",
        "VOYAGE_API_KEY": "..."
      }
    }
  }
}
```

Everything absolute, and the environment set explicitly, for one reason: **the
client launches this from its own working directory, not from the repository.**
A relative config path is not found, `.env` is not read, and the defaults then
resolve to an embedded Qdrant at a `data/qdrant` that does not exist. That used
to start up perfectly and serve an empty index — every query answering "nothing
found", which an agent believes. `serve` now checks the collection at startup
and refuses to run rather than serve nothing:

```
the collection for voyageai_voyage-code-4_2048 is empty (embedded qdrant at
/some/other/cwd/data/qdrant), so every query would return nothing.
```

Four tools, and the split between the first two is the point — an agent already
knows its intent, and declaring it is far more reliable than inferring it from
the query text:

| tool | what it searches |
|---|---|
| `search_code` | implementation and docs; tests and generated files excluded |
| `find_guidance` | specs, design documents and guides only |
| `get_file_context` | every indexed chunk of one file, in order |
| `list_document_types` | what kinds of document this workspace holds, with counts |

The same taxonomy is also served as a resource at `workspace-indexer://taxonomy`.
Both surfaces exist because clients differ in how reliably a model sees a
resource that the user has not attached, whereas a tool is always in context.

A count of zero in `list_document_types` is a real answer, not a gap: if
`normative` is 0, this workspace has no written standards and an agent should
read the implementation rather than keep hunting for specs.

**Note:** the server holds the index open for as long as the session lasts, so
Qdrant must be in server mode (`QDRANT_MODE=server`). Embedded Qdrant takes an
exclusive lock on its storage directory, and indexing from another terminal
would fail for as long as the agent is connected.

## Reference

**[docs/reference.md](docs/reference.md)** is the lookup page: every command
and flag, every `workspace.yaml` key and `.env` variable with its default, and
every MCP tool with its parameters. A test asserts it lists everything that
exists, so it cannot quietly fall behind the code.

## Changing it

**[docs/testing.md](docs/testing.md)** is the page to read before touching the
MCP surface, the storage seam or logging. It covers what the guard suites are
for -- each one exists because of a real failure that was silent and unrelated
to the code being changed at the time -- and the operational facts that are
easy to lose: how structlog's processor list interacts with `capture_logs`, and
the four things about Atlas that will otherwise cost you an afternoon.

The short version: `poetry run pytest -q -m "not integration"` is what CI runs,
and **before merging anything that changes those areas, run `poetry run pytest
-q` with no marker filter.** Integration tests do not run in CI, which is
exactly how one of them went stale for two pull requests.

## Running it somewhere else

This README is a developer quickstart. **[docs/deployment.md](docs/deployment.md)**
covers operating it on a box that is not the dev machine: what has to be
co-located with what, choosing a storage mode, installing Qdrant on Linux and
Windows, a worked split deployment, and what re-indexing actually costs.

The short version of the part that surprises people: the indexer is the only
component pinned to the source files. Qdrant can live anywhere, and the MCP
server needs the source only for staleness checks — which is a setting.

## Development

```bash
poetry run pytest -q                     # full suite
poetry run pytest -m "not integration"   # fast, fully offline
poetry run ruff check src/ tests/
poetry run pyright                       # strict; must be 0 errors
```
