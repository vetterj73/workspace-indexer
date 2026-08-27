# Reference

Every command, every configuration option, every MCP tool. `README.md` is the
quickstart and `docs/deployment.md` is how to run it somewhere else; this is
the page you look things up in.

Defaults here are the ones in code, not the ones in the example config —
`config/workspace.example.yaml` sets some values explicitly for illustration.
Where a default deserves an explanation, it has one.

---

## 1. Commands

All commands accept `--config PATH` (default `config/workspace.yaml`). An MCP
client or a systemd unit starts from its own working directory, so pass an
absolute path there.

### `index`

Walk the configured roots and index what changed.

| flag | default | |
|---|---|---|
| `--root LABEL` | all roots | Index only this root. **Scopes orphan pruning too**, so other roots are neither walked nor touched. Removing a root from `workspace.yaml` is the opposite: it *is* pruned. |
| `--dry-run` | off | Chunk plan and token estimate, no API calls and nothing stored. How to tune chunking without paying to iterate. |
| `--force` | off | Ignore the mtime and hash shortcuts and re-embed everything. Needed after changing anything that alters *what is embedded* without altering chunk identity — `chunking.embed_doc_type` is the example. |

### `search`

| flag | default | |
|---|---|---|
| `--limit N` | `search.default_limit` | |
| `--unit NAME` | all | Restrict to one top-level directory of a root. |
| `--lang NAME` | all | `python`, `csharp`, `tsx`… |
| `--kind KIND` | all | `code`, `markdown`, `pdf`, `text`, `image`, `opaque`. |
| `--path PREFIX` | all | Restrict to a directory. |
| `--fusion MODE` | `search.fusion` | `rrf`, `dense_only`, `sparse_only`. A debugging tool: when a query returns junk, the first question is which branch produced it. |
| `--rerank / --no-rerank` | config | Override reranking for one call. |
| `--full` | off | Print whole chunks instead of a preview. |

### `status`

What is indexed, in which spaces, what recent runs cost, and import-graph
coverage per language. Reports `inconsistent` when the manifest and the store
disagree about how many chunks exist — usually a store rebuilt without the
manifest, which `index --force` fixes.

### `explain PATH`

Dump the chunks one file produces, with symbols and line ranges. The
chunk-quality debugging tool; no API calls.

### `reproject --dimensions N`

Derive a narrower collection by Matryoshka truncation, with no re-embedding.
The first N components of a Voyage vector are a valid N-dimensional embedding,
so going 2048 → 1024 is free. Going the other way is a full re-embed, which is
why indexing wide and narrowing later is the cheaper order.

### `eval`

Score retrieval against a dataset of query/expected-file pairs.

| flag | default | |
|---|---|---|
| `--dataset PATH` | `eval.dataset` | |
| `--limit N` | `10` | The k in recall@k and MRR@k. |
| `--tool NAME` | `search` | `search`, `search_code` or `find_guidance`. Scores an **MCP tool** rather than the raw search path — the only way to measure what a document-type filter is worth, since the filter is the hypothesis. |
| `--group NAME` | `all` | `all`, `retrieval` or `guidance`. |
| `--fusion MODE` | config | |
| `--rerank / --no-rerank` | config | Is the reranker earning its latency? |
| `--dimensions N` | active space | Evaluate a reprojected collection. |
| `--save / --no-save` | save | Record the run under `evals/`. |
| `--compare / --no-compare` | compare | Diff against the last **comparable** run. Two runs are comparable only when config hash, fusion, reranker, tool, case group and case count all match — see §2.4. |

### `serve`

Run the MCP server over stdio. See §4.

### `watch`

Watch the roots and reindex as files change. A trigger, not a second indexing
path: every change goes through the same `index --root` the CLI performs.

---

## 2. `workspace.yaml`

*What* to index. Safe to commit; its `roots` are paths, so prefer `~`-relative
ones or keep a per-host file.

### 2.1 `workspace`

| key | default | |
|---|---|---|
| `name` | **required** | Names the Qdrant collection: `{name}__{embedding-space}`. Two workspaces with different names get separate collections, which is how you keep a test corpus from shifting your baselines. |
| `roots` | **required** | List of `{path, label?, recurse_into_children?}`. `label` defaults to the directory name and keys the manifest and the payload filter, so a collision would silently merge two roots. |

### 2.2 `index`

| key | default | |
|---|---|---|
| `respect_gitignore` | `true` | Honours `.gitignore` per directory. **Only works inside a git repo** — a plain folder has none. |
| `follow_symlinks` | `false` | |
| `max_file_bytes` | `1048576` | |
| `exclude` | `[]` | gitignore syntax. Excluded directories are *pruned*, not filtered, so `**/node_modules/**` costs one `stat()` rather than a walk. Exclusion is retroactive: adding a pattern removes chunks already indexed. |
| `secret_allow` | `[]` | Files where the content secret-scan is expected to false-positive — token fixtures, checksum lists. Scoped by glob, so allowing one file does not disable the check elsewhere. |

Some exclusions are not configurable at all: `logs/`, `data/`, `.git/`,
`*.sqlite3`, `evals/` and `docs/eval-baselines.md`. A log file inside a watched
root is an infinite reindex loop, and an eval artefact quotes every query
verbatim so indexing one corrupts the measurement. Those are correctness
rules, not preferences.

### 2.3 `chunking`

| key | default | |
|---|---|---|
| `code.max_tokens` | `512` | Applies to header + source, not source alone. |
| `code.min_tokens` | `24` | Below this, adjacent definitions merge. |
| `code.include_context_header` | `true` | Prefix repo, path, language and symbol before embedding. A method lifted from its class is meaningless without it. |
| `markdown.max_tokens` | `512` | |
| `markdown.split_on_heading_depth` | `3` | |
| `text.max_tokens` | `512` | |
| `text.overlap_paragraphs` | `1` | |
| `opaque.mode` | `metadata_only` | Binary and image files: recorded, not embedded. |
| `embed_doc_type` | `false` | Also write `# type: normative` into the embedded text. **Off on measured evidence**: it dropped recall@10 from 0.875 to 0.812, and worst on the guidance cases it was meant to help. `doc_type` has nine values across thousands of chunks, so the line carries almost no discriminating signal. Changing it needs `index --force` — the header is excluded from `content_sha`, so a normal run finds identical chunk ids and re-embeds nothing. |
| `overrides` | `{}` | Pin an extension to a chunker: `{".mdx": "markdown"}`. The extension point for a file type whose detected language routes it somewhere unhelpful. |

### 2.4 `search`

| key | default | |
|---|---|---|
| `fusion` | `rrf` | `rrf` fuses dense and sparse on *rank*, which is right because cosine and BM25 scores are not on a comparable scale. |
| `prefetch_limit` | `50` | Candidates per branch before fusion. Cheaper than more dimensions if recall is short. |
| `default_limit` | `10` | |
| `check_staleness` | `true` | Verify each hit against the file on disk. Needs read access to the source: **set false where the MCP server runs next to Qdrant rather than next to the code**, or every hit comes back flagged and the flag stops meaning anything. |
| `rerank.enabled` | `true` | Off resolves to a `NoopReranker` — a different object, not a branch in the search path. |
| `rerank.model` | `voyageai:rerank-2.5-lite` | |
| `rerank.candidates` | `50` | Fused hits fed to the reranker. |
| `rerank.top_n` | `10` | Returned after reranking. |
| `rerank.rerank_text` | `embed_text` | Rerank the context-headed text, not bare source. |
| `rerank.instruction` | none | Prepended to the query — the API has no instruction parameter. |
| `rerank.on_error` | `degrade` | `degrade` falls back to fusion order and logs; `fail` raises, which the eval harness wants so a silent degradation cannot corrupt a measurement. |

### 2.5 `watch`

| key | default | |
|---|---|---|
| `mode` | `auto` | `auto` decides per root from `/proc/mounts`. inotify on a 9p, CIFS or NFS mount **succeeds and then never fires**, so those are polled. Unknown filesystems are polled too: guessing "native" wrong costs a watcher that silently never fires. |
| `debounce_ms` | `1500` | One editor save is several events; a formatter run is hundreds. |
| `poll_interval_ms` | `5000` | For roots that cannot use inotify. |
| `reload_config` | `true` | Reload this file on change. Settings and ignore rules go live; **a newly added root still needs a restart**. |

### 2.6 `eval` and `logging`

| key | default | |
|---|---|---|
| `eval.dataset` | `config/eval.yaml` | Never indexed — it quotes every query. |
| `eval.metrics` | `[recall@10, mrr@10]` | |
| `logging.level` | `INFO` | Console level. `.env`'s `LOG_LEVEL` overrides it. |
| `logging.console` | `pretty` | `pretty`, `json` or `off`. Writes to **stderr**, which is what keeps stdio MCP usable. |
| `logging.file.path` | `logs/workspace-indexer.jsonl` | Always DEBUG regardless of console level — you cannot retroactively raise a log level after the failure you needed to see. |
| `logging.file.max_bytes` | `20971520` | Size-based rotation, because indexer output is bursty. |
| `logging.file.backup_count` | `10` | |
| `logging.logfire.enabled` | `false` | |
| `logging.logfire.send_to_cloud` | `false` | **pydantic-ai instrumentation captures call inputs, and for an embedding call the input is your source code.** Never a default. |

---

## 3. `.env`

*How* to index. Never committed. Real environment variables win over the file,
which is what makes the `.mcp.json` `env` block work.

| variable | default | |
|---|---|---|
| `EMBEDDING_MODEL` | `voyageai:voyage-code-4` | `provider:model`. The whole provider abstraction. |
| `EMBEDDING_DIMENSIONS` | `2048` | Part of the collection name, so changing it addresses a different collection rather than migrating one. |
| `EMBEDDING_QUANTIZATION` | `float32` | Accepted, currently unused. |
| `EMBEDDING_BATCH_SIZE` | `64` | |
| `EMBEDDING_MAX_CONCURRENCY` | `4` | |
| `EMBEDDING_MAX_BATCH_TOKENS` | `100000` | Providers cap tokens per request, not just document count. |
| `EMBEDDING_PRICE_PER_MTOK` | none | Used only when the provider reports no price. `genai-prices` has no entry for `voyage-code-4`, so without this a run records as *unpriced* rather than as free. A configured rate goes stale silently, so anything priced this way displays as `~$0.1234`. |
| `EMBEDDING_FREE_TIER_TOKENS` | none | `status` shows drawdown against it. A floor on usage, never an authority: the allowance is per account and spent by anything using the key. |
| `VOYAGE_API_KEY` | none | Exported to `os.environ` at startup, because provider SDKs read it from there. |
| `SPARSE_MODEL` | `Qdrant/bm25` | Local, free, no API call. |
| `RERANK_ENABLED` | `true` | |
| `RERANK_MODEL` | `voyageai:rerank-2.5-lite` | Missing key resolves to a no-op reranker and logs once — a search never fails because an optional enhancement is unconfigured. |
| `VECTOR_STORE` | `qdrant` | |
| `QDRANT_MODE` | `embedded` | `embedded` is single-process and **ignores payload indexes**, so `doc_type` filtering scans. `server` is required for the MCP server and the watcher. |
| `QDRANT_PATH` | `data/qdrant` | Embedded mode only. Relative — a process started elsewhere resolves it elsewhere. |
| `QDRANT_URL` | `http://localhost:6333` | |
| `QDRANT_API_KEY` | none | **Qdrant has no authentication by default, and the payload contains your source text.** Set this whenever it is not bound to loopback. |
| `QDRANT_ON_DISK_PAYLOAD` | `true` | |
| `STATE_DB` | `data/manifest.sqlite3` | Give a second workspace its own, or both share one and the divergence check misfires. |
| `LOG_LEVEL` | from yaml | |
| `LOGFIRE_ENABLED` / `LOGFIRE_SEND_TO_CLOUD` / `LOGFIRE_TOKEN` | none | |

---

## 4. MCP server

`workspace-indexer serve` speaks MCP over stdio: the client starts the process
and talks down a pipe. No port, nothing left running. See
`docs/deployment.md` §4 for the `.mcp.json` recipe and why every path in it
must be absolute.

Requires `QDRANT_MODE=server` — the server holds the index open for the whole
session, which embedded mode's exclusive lock forbids.

### Tools

Separate tools rather than one tool with an intent argument, because the agent
already knows its intent and declaring it is far more reliable than inferring
it from query text.

**`search_code`** — implementation and documentation, by meaning.

| parameter | type | default | |
|---|---|---|---|
| `query` | string | required | |
| `limit` | integer 1–50 | `8` | |
| `repo` | string | none | Restrict to one repository. |
| `language` | string | none | |
| `path_prefix` | string | none | |
| `include_tests` | boolean | `false` | Tests and generated files are excluded by default: a test naming a symbol twenty times otherwise outranks the file defining it. |

**`find_guidance`** — specifications, design documents and guides only.

| parameter | type | default | |
|---|---|---|---|
| `query` | string | required | |
| `limit` | integer 1–50 | `8` | |
| `repo` | string | none | |
| `doc_type` | string | none | Narrow to one type. Accepts aliases (`spec`, `adr`, `architecture`, `readme`…). **An unrecognised value returns an error naming the valid types — never an empty result.** |

Guidance covers `normative`, `design` **and `guide`**. The last is there on
evidence: normative + design alone scored no better than plain search over the
eight guidance cases, because filtering out `guide` lost `CONTRIBUTING.md`
entirely. Adding it took recall from 0.812 to 0.938.

**`get_file_context`** — every indexed chunk of one file, in order.

| parameter | type | default | |
|---|---|---|---|
| `rel_path` | string | required | A path from a search result, or a trailing portion of one. |
| `limit` | integer 1–100 | `20` | |

**`list_document_types`** — the taxonomy, with counts and example paths. No
parameters. A type reported at **count 0** genuinely has none in this
workspace: if `normative` is 0, read the implementation instead of hunting for
specs.

### Resource

`workspace-indexer://taxonomy` serves the same taxonomy as JSON. Both surfaces
exist because clients differ in how reliably a model sees a resource the user
has not attached, whereas a tool is always in context.

### Import graph

Each code file's imports are extracted during indexing and, where possible,
resolved to another indexed file. `status` reports both coverage (which
languages have an extractor) and resolution (how many edges point at a file).

Resolved within a repository only. Python relative and absolute imports,
and JS/TS relative specifiers — including TypeScript's ESM convention where
`import './x.js'` names a file that is actually `x.ts`.

**Not** resolved, deliberately: packages (`react`, `pydantic`), tsconfig path
aliases (`@/lib/utils`), and C# namespaces, which name no path at all. Those
need a build system or a workspace-wide symbol table; until then they resolve
to nothing rather than to something plausible. An unresolved edge is not a
missing dependency.

The reverse edge — *which files import this one* — spans every repository in
the workspace, which is what a per-project language server cannot answer.

### Recorded calls

Every tool call is recorded twice: as an `mcp.tool_call` event in the rolling
log, and as a row in the manifest's `mcp_calls` table. Both carry the tool,
the parameters, the returned paths, `dropped_for_budget`, any note and the
duration.

`status` summarises them and names recent calls worth turning into eval cases.
The harvesting query is the point:

```sql
SELECT tool, query, returned FROM mcp_calls
WHERE returned = 0 OR dropped_for_budget > 0
ORDER BY called_at DESC;
```

Those are queries a real agent asked that the index could not answer, which is
a better source of eval cases than any dataset written from imagination.

**What it cannot tell you.** A call that returns three confidently irrelevant
results looks identical here to one that returns three good ones — only the
count is recorded, not whether the answer was right. Empty and clipped
responses are detectable; *wrong* ones still need a human or an eval.

Recording never fails a call: if the manifest write raises, the log entry has
already succeeded and the tool returns normally.

### Result shape

Every result is `path:start-end` anchored, so the next action is a read with
nothing to work out. Responses are token-budgeted; anything dropped is reported
in `dropped_for_budget` rather than silently cut, and an empty response carries
a note saying what was filtered and what to try next.

### Document types

`normative` · `design` · `guide` · `reference` · `record` · `implementation` ·
`test` · `generated` · `unknown`

Call `list_document_types` for the definitions and live counts — they come from
the index, not from this page.

---

## 5. Things that surprise people

**Adding a corpus under an existing root joins the main index** and shifts
every eval number for reasons unrelated to any code change. Give it its own
`workspace.name`.

**Changing `embed_doc_type` needs `--force`.** It changes what is embedded but
not chunk identity, so a normal run finds identical ids and does nothing.

**A manifest survives a store rebuild.** Point at a fresh Qdrant and the
decision ladder skips every file because the manifest says they are done.
`index` warns; `status` reports it as `inconsistent`; `--force` fixes it.

**`data/` is derived state.** The backup story is "re-index", not "back it up".

**Opaque files produce zero chunks.** A file whose bytes do not decode as text
is recorded but not embedded, so it cannot be found by search. An unknown
*language* is fine — that falls to the text chunker and stays searchable.
