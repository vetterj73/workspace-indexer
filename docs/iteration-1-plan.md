# Workspace Indexer — Iteration 1 Plan

> **Status.** Config, logging, discovery, and the data models are built and on
> `main`. Chunking, embedding, storage, state, rerank, search, and the CLI are
> not yet written. Where implementation taught us something the plan got wrong,
> this document has been corrected rather than left as history — it is the
> design of record, not a diary. Corrections are marked **[revised]**.

## Context

We want a Python tool that keeps a semantic index of a *workspace* — a directory holding a mix of git repos, non-repo folders, and a `.claude/` tree of docs — so that an LLM (primarily Claude Code, via MCP) can find relevant code and documentation by meaning rather than by grep. The workspace root is not itself a repo; each child may or may not be one, and that fact is metadata we want to capture.

The problem this solves: an agent dropped into a large multi-repo workspace has no cheap way to answer "where is auth handled" or "what does our deployment doc say about rollbacks." Grep needs the right literal token; directory listings need the agent to already know the layout. A hybrid vector + keyword index over well-chosen chunks turns that into one query.

Four constraints shape every decision below:

1. **Pluggable at every seam.** Chunking strategy, embedding provider, and vector store must each be swappable via config without touching the others. Content is a mix of code, markdown, PDFs, and images — one chunking strategy cannot serve all of them.
2. **Incremental, never full-rebuild.** A watcher will eventually run continuously over a large tree. Re-embedding unchanged files is the dominant cost risk, so change detection is a first-class component, not an optimization.
3. **Observable from the first commit.** Embedding calls cost money and fail in ways that are invisible without instrumentation (silent truncation, partial batches, 429 backoff, dimension mismatch). Logging is in iteration 1, not retrofitted.
4. **Iterative delivery.** This iteration builds a narrow slice end-to-end (config → chunk → embed → store → hybrid search) to prove the seams. Breadth (more languages, more file types, the watcher, MCP) comes later against interfaces that already exist.

This file plans **iteration 1 only**, with later iterations sketched so the interfaces are shaped correctly now.

---

## Decisions already settled

| Concern | Decision |
|---|---|
| Build order | Thin vertical slice first, then widen |
| Query surfaces | MCP server (primary goal), CLI search, importable library — CLI in this iteration |
| Qdrant | Both embedded and server mode, selected by config |
| Search | Hybrid: dense + BM25 sparse, RRF fusion — **iteration 1**, because the collection schema must be right from the first run |
| Reranking | Voyage `rerank-2.5-lite`, toggleable, degrades gracefully when disabled or unconfigured |
| Embedding model | `voyageai:voyage-code-4` (from config, not hardcoded) |
| Dimensions | Index at **2048**, evaluate, expect to ship **1024** — see "Choosing dimensions" |
| Code chunking | Symbol-level tree-sitter chunks with a synthetic context header |
| Other file types | Per-kind chunker registry; markdown/PDF/image each get their own strategy |
| Logging | `structlog` → console + rolling JSONL file; Logfire optional and off by default |
| Quality measurement | Minimal eval harness in iteration 1 — the knobs above are too numerous to tune by feel |

### Mandates

Three rules bind every file in this repo. They live in `CLAUDE.md` and are
enforced, not trusted:

- **One class per file**, module named after the class in snake_case, grouped
  into packages that re-export from `__init__.py` so call sites stay readable.
  `tests/test_one_class_per_file.py` checks both layout and naming.
- **Tests ship with the code**, in the same unit of work. A check worth running
  in a shell is a check worth committing.
- **pyright strict over `src/` and `tests/` alike**, zero errors. Strict mode on
  tests is what catches an untyped fixture widening a parameter to `Any` and
  thereby hiding real errors in the code under test.

### Library choices, with reasoning

- **`pydantic-ai` for embeddings** — correct call. Its `Embedder` class takes a `provider:model` string and supports VoyageAI, OpenAI, Google, Cohere, Bedrock, and local sentence-transformers. `EMBEDDING_MODEL=voyageai:voyage-code-4` in `.env` is the entire provider abstraction. It also exposes `count_tokens()` / `max_input_tokens()` (needed to size chunks) and Voyage's `input_type` query/document distinction via settings. **Caveat: text-only.** Images need the `voyageai` SDK's separate `multimodal_embed()` endpoint — see "Images" below.
- **`tree-sitter` + `tree-sitter-language-pack`** — `tree-sitter-languages` is unmaintained; the language-pack fork is the live successor. **[revised]** It turns out to provide much more than grammars: `process()` returns symbol-aware `chunks` already carrying `context_path` (the enclosing class/module trail), `symbols_defined`, `chunk_index`/`total_chunks`, and a `has_error_nodes` flag, plus a hierarchical `structure` tree and `detect_language_from_path()`. That deletes the per-language node-type tables the plan budgeted for and lets iteration 1 cover *every* language instead of two. Two operational notes: `chunk_max_size` is in **bytes**, not tokens, and grammars are **downloaded on demand** to a local cache rather than bundled — so first use of a new language needs network, and a download failure must degrade to the text chunker rather than losing the file.
- **`fastembed`** — Qdrant's own library, provides local BM25 sparse encoding with no API call and no cost. This is what makes hybrid search cheap enough to include in iteration 1.
- **`voyageai` SDK** — needed *alongside* pydantic-ai, for two things pydantic-ai's embeddings API doesn't cover: `client.rerank()` and (later) `multimodal_embed()`. Dense text embedding still goes through pydantic-ai so the provider stays swappable; the Voyage SDK is only reached through the `Reranker` protocol, so a Cohere or local cross-encoder reranker drops in without touching the search path.
- **`structlog`** — structured logging over the stdlib. Same event dict renders as pretty console output *and* as JSON lines to file, which is the whole requirement in one library.
- **`watchfiles`** — the library you were told about; Rust-backed, works on Linux/WSL/Windows. Iteration 2.
- **`pathspec`** — parses `.gitignore` syntax so we honor ignore files without shelling out to git per path.
- **`qdrant-client`** — one client class covers both embedded and server mode.
- **`typer`** + **`pydantic-settings`** — CLI and `.env` loading.
- **Python 3.12**, **Poetry** (already installed at `/home/jeremy/.local/bin/poetry`).

### Three Linux details that affect the design

**inotify does not work on Windows-backed mounts.** On WSL and similar setups, `inotify_add_watch()` on a path under `/mnt/c` *succeeds* and then never delivers an event, because the 9P protocol WSL uses to reach the Windows filesystem carries no change notifications. inotify is a service of the kernel's filesystem layer, so it only works on real Linux filesystems (ext4, overlayfs) — not 9P, NFS, or FUSE mounts. We cannot fix this; we detect it. Iteration 2 reads `/proc/mounts` to find the filesystem type backing each root and falls back to polling when it isn't native, with a `watch_mode: auto|native|poll` override.

**inotify watch limits are per-user and finite.** `/proc/sys/fs/inotify/max_user_watches` (often 65536, sometimes 8192) caps how many directories one user can watch. A workspace with `node_modules` present will blow through it. Our ignore rules aren't just about index quality — they're what keeps the watcher functional. Iteration 2 logs the watch count against the limit rather than failing opaquely.

**Never write logs inside a watched root.** The watcher would fire on our own log writes, re-index, write more logs, and spin forever. The log directory is on the *hardcoded* exclude list, not the configurable one — a user shouldn't be able to configure themselves into an infinite loop. Same reasoning applies to `data/qdrant` and the SQLite manifest.

---

## The config contract

`workspace.yaml` — the file the tool watches and reloads:

```yaml
workspace:
  name: labbox
  roots:
    - path: ~/src
      recurse_into_children: true    # treat each child dir as its own unit
    - path: ~/src/workspace-indexer/.claude
      label: claude-config

index:
  respect_gitignore: true
  follow_symlinks: false
  max_file_bytes: 1_048_576
  exclude:                            # gitignore syntax, applied everywhere
    - "**/node_modules/**"
    - "**/.venv/**"
    - "**/__pycache__/**"
    - "**/dist/**"
    - "**/*.lock"
    - "**/.git/**"

chunking:
  code:
    max_tokens: 512
    min_tokens: 24
    include_context_header: true
  markdown:
    max_tokens: 512
    split_on_heading_depth: 3
  opaque:
    mode: metadata_only               # later: caption | multimodal

search:
  fusion: rrf                         # rrf | dense_only | sparse_only
  prefetch_limit: 50                  # candidates per branch before fusion
  default_limit: 10
  rerank:
    enabled: true
    model: rerank-2.5-lite            # or rerank-2.5
    candidates: 50                    # fused hits fed to the reranker
    top_n: 10                         # returned after reranking
    rerank_text: embed_text           # embed_text | source_text
    instruction: >                    # prepended to the query; no dedicated API param
      Prioritize implementation code over tests, fixtures, and generated files.
    on_error: degrade                 # degrade | fail

eval:
  dataset: ./config/eval.yaml         # query -> expected paths
  metrics: [recall@10, mrr@10]

logging:
  level: INFO
  console: pretty                     # pretty | json | off
  file:
    path: ./logs/workspace-indexer.jsonl
    max_bytes: 20_971_520             # 20 MB
    backup_count: 10
  logfire:
    enabled: false                    # see the data-egress note below
    send_to_cloud: false
```

`.env` — secrets and backend selection, kept separate from workspace layout:

```
EMBEDDING_MODEL=voyageai:voyage-code-4
EMBEDDING_DIMENSIONS=2048         # see "Choosing dimensions"
EMBEDDING_QUANTIZATION=float32    # int8 | binary — accepted but unused in iteration 1
VOYAGE_API_KEY=...

SPARSE_MODEL=Qdrant/bm25

RERANK_ENABLED=true
RERANK_MODEL=rerank-2.5-lite      # unset/absent key -> reranking silently skipped

VECTOR_STORE=qdrant
QDRANT_MODE=embedded              # embedded | server
QDRANT_PATH=./data/qdrant
QDRANT_URL=http://localhost:6333
QDRANT_ON_DISK_PAYLOAD=true

STATE_DB=./data/manifest.sqlite3
LOG_LEVEL=INFO
LOGFIRE_ENABLED=false
LOGFIRE_TOKEN=
```

Split rationale: `workspace.yaml` is *what to index* and is safe to commit; `.env` is *how to index* and holds credentials. Only the former needs hot-reloading, which keeps the iteration-2 watcher simple.

---

## Logging & observability

Console + rolling flat file, wired up before the first embedding call is ever made.

### Setup — `obs/logging.py`

One `configure_logging(cfg)` called once from `cli.py`. `structlog` processes an event dict through a shared pipeline, then renders differently per sink:

- **Console** — `structlog.dev.ConsoleRenderer`, colored and human-readable, at the configured level.
- **File** — JSON lines via `structlog.processors.JSONRenderer` into a stdlib `RotatingFileHandler` (`max_bytes` × `backup_count`, so 20 MB × 10 = 200 MB ceiling). Always DEBUG regardless of console level: the console is for watching, the file is for forensics, and you cannot retroactively raise a log level after the failure you needed to see.

Size-based rotation rather than time-based (`TimedRotatingFileHandler`) because indexer output is bursty — a full reindex produces a huge volume in minutes, then near-silence for days. Rotating on size keeps each file usefully sized; rotating daily would give you one enormous file and nine empty ones.

JSONL because it makes the log queryable with tools already on the box — `jq 'select(.event=="embed.batch") | .duration_ms' logs/workspace-indexer.jsonl` answers "how slow were my embedding calls" without any log-parsing code.

### Contextvars: the thing that makes debugging actually work

`structlog.contextvars` binds `run_id` (a UUID per invocation) and, inside the file loop, `root_label` + `rel_path` — so **every** log line emitted while processing a file automatically carries that file's identity, including lines from deep inside the chunker or the embedding retry loop. Without this you get a traceback with no indication of which of 40,000 files caused it. This is the single highest-value logging decision in the plan.

### What gets logged where

| Event | Level | Key fields |
|---|---|---|
| `run.start` / `run.end` | INFO | `run_id`, config hash, roots, embedding space, mode (index/dry-run), totals, wall time |
| `discovery.skip` | DEBUG | `rel_path`, `reason` (gitignored / excluded / too_large / symlink / lockfile / empty / unreadable) |
| `discovery.prune` | DEBUG | **[revised, new]** `path`, `reason` — a directory not descended into. Counted separately from file skips because one pruned directory can stand for thousands of files, so folding the two together makes the file tally meaningless. |
| `file.decision` | DEBUG | `rel_path`, `decision` (skip_mtime / skip_hash / chunk / force), `mtime_ns`, `sha256` |
| `chunk.produced` | DEBUG | `rel_path`, chunker, chunk count, token histogram, parse_fallback flag |
| `chunk.parse_failed` | **WARNING** | `rel_path`, language, error — a silent tree-sitter failure degrading to text chunking is exactly the bug you'd never notice |
| `embed.batch` | INFO | batch size, total tokens, model, `duration_ms`, `attempt`, estimated cost |
| `embed.retry` | WARNING | `attempt`, HTTP status, backoff seconds, error body |
| `embed.truncated` | **WARNING** | `rel_path`, chunk tokens vs `max_input_tokens` — silent truncation is the classic invisible quality bug |
| `store.upsert` / `store.delete` | INFO | collection, count, `duration_ms` |
| `search.query` | INFO | query text, filters, per-branch candidate counts, fused count, `duration_ms` |
| `rerank.call` | INFO | model, candidate count, `total_tokens`, `duration_ms`, rank churn (how far the top-1 moved) |
| `rerank.skipped` | INFO (once per process) | reason: disabled / no API key / no provider — never an error |
| `rerank.degraded` | WARNING | error, falling back to RRF order |
| `error.*` | ERROR | full context + traceback, `rel_path` from contextvars |

**Cost and token accounting is logged per batch and summarized per run**, because "why did this cost $40" needs to be answerable from the log alone. The run summary also lands in a SQLite `runs` table so `workspace-indexer status` can show history without parsing logs.

### Logfire — optional, off by default

Assessment: Logfire is solid engineering and it is the *right* tool for tracing the embedding layer, because it auto-instruments `pydantic-ai` and gives per-call spans for free. But it is a tracing platform, not a logging library — it has no rotating-file concept, and its default sink is Pydantic's hosted service.

The reason it is off by default: **`pydantic-ai` instrumentation captures call inputs, and for an embedding call the input is your source code.** With `send_to_logfire=True`, chunks of private repos leave the machine. That should be a deliberate choice, never a default.

The integration is designed so this is a two-line switch, not a rewrite:

- `logfire.enabled: true`, `send_to_cloud: false` → local-only spans, console output, nothing leaves the box. Genuinely useful for seeing embedding-call timing without committing to the SaaS.
- `send_to_cloud: true` → requires `LOGFIRE_TOKEN`, and the CLI prints an explicit one-time warning that source content will be transmitted.
- Either way, `logfire.integrations.logging.LogfireLoggingHandler` attaches as an *additional* handler. The structlog console and file sinks are unaffected, so nothing depends on Logfire being present.

---

## Core interfaces

These are the contracts every later iteration codes against. Getting them right now is the whole point of the vertical slice.

**`models.py`**

```python
class FileKind(StrEnum):
    CODE = "code"; MARKDOWN = "markdown"; PDF = "pdf"
    IMAGE = "image"; TEXT = "text"; OPAQUE = "opaque"

class ChunkMeta(BaseModel):
    workspace: str
    root_label: str
    abs_path: Path
    rel_path: str            # relative to its root — what a human/LLM reads
    kind: FileKind
    language: str | None     # "python", "typescript", None for prose
    repo: RepoInfo | None    # remote_url, branch, head_sha, is_dirty
    symbol_path: str | None  # "ClassName.method_name" / "## Heading > ### Sub"
    symbol_kind: str | None  # function | method | class | heading | preamble
    start_line: int
    end_line: int
    content_sha: str         # sha256 of source_text

class Chunk(BaseModel):
    meta: ChunkMeta
    source_text: str         # exact bytes from the file — what we display
    embed_text: str          # context header + source_text — what we embed
    chunk_id: str            # deterministic: sha256(rel_path + symbol_path + content_sha)
```

The `source_text` / `embed_text` split matters: we embed an enriched string but return the real code. A method extracted from a class is meaningless in isolation, so `embed_text` gets a header:

```
# repo: labbox/workspace-indexer (main)
# file: src/workspace_indexer/storage/qdrant.py
# class QdrantStore
def upsert(self, chunks: list[Chunk]) -> None:
    ...
```

`chunk_id` is content-addressed and stable: edit one function and only that chunk's id changes, so the delete/upsert set is exactly the changed functions.

**`chunking/base.py`**

```python
class Chunker(Protocol):
    kinds: frozenset[FileKind]
    version: int   # bump to force re-chunk of this kind on next run
    def chunk(self, file: SourceFile, cfg: ChunkConfig) -> Iterator[Chunk]: ...
```

`registry.py` resolves `FileKind` → `Chunker`, with an override map in config so a specific extension can be pinned to a specific chunker. This is the "different strategies for different file types" requirement, and it is the extension point for PDFs, notebooks, and anything else later.

**`embedding/base.py`**

```python
class EmbeddingSpace(BaseModel):
    model: str          # "voyageai:voyage-code-4"
    dimensions: int
    sparse_model: str   # "Qdrant/bm25"
    def slug(self) -> str: ...   # "voyageai_voyage-code-4_2048" — collection name

class EmbeddingBackend(Protocol):
    space: EmbeddingSpace
    def max_input_tokens(self) -> int: ...
    def count_tokens(self, text: str) -> int: ...
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...

class SparseBackend(Protocol):
    def encode_documents(self, texts: list[str]) -> list[SparseVec]: ...
    def encode_query(self, text: str) -> SparseVec: ...

class Reranker(Protocol):
    name: str
    def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]: ...
```

`NoopReranker` implements the same protocol and returns `hits[:top_n]` unchanged. Everything downstream is written against `Reranker`, so "reranking off" is a different object, not a branch in the search path — there is no `if rerank_enabled:` scattered through the code.

`text_pydantic_ai.py` wraps `pydantic_ai.Embedder` — construct from the `.env` model string, pass `dimensions` via `EmbeddingSettings`, delegate token counting. Batching, retry/backoff on 429, a concurrency semaphore, and the `embed.batch` / `embed.retry` / `embed.truncated` log events all live here, not in the pipeline.

**`storage/base.py`**

```python
class VectorStore(Protocol):
    def ensure_collection(self, space: EmbeddingSpace) -> None: ...
    def upsert(self, space, chunks, dense: list[list[float]], sparse: list[SparseVec]) -> None: ...
    def delete_by_ids(self, space, chunk_ids: list[str]) -> None: ...
    def delete_by_path(self, space, root_label: str, rel_path: str) -> None: ...
    def search(self, space, query: QuerySpec, filters: SearchFilters) -> list[SearchHit]: ...
```

`delete_by_path` is what makes file deletion and rename correct without a manifest lookup.

---

## Qdrant collection & payload schema

One collection per embedding space, named `{workspace}__{space.slug()}`. **Both named vectors are declared at creation**, because adding a named vector to a populated Qdrant collection is not a simple migration — get this wrong and hybrid search costs a full re-embed later, which is precisely the outcome we're designing against.

```python
client.create_collection(
    collection_name,
    vectors_config={"dense": VectorParams(size=space.dimensions, distance=Distance.COSINE)},
    sparse_vectors_config={"bm25": SparseVectorParams(modifier=Modifier.IDF)},
    on_disk_payload=True,
)
```

`modifier=Modifier.IDF` is required for BM25 — it tells Qdrant to compute inverse document frequency server-side across the collection. Omit it and sparse scoring silently degrades to raw term frequency, which ranks badly and gives no error. Easy to miss, hard to notice.

`on_disk_payload=True` because we store `source_text` in the payload (see below) and it would otherwise dominate RAM.

### Payload fields

| Field | Type | Purpose |
|---|---|---|
| `workspace` | keyword | multi-workspace separation |
| `root_label` | keyword | which configured root — **indexed**, primary filter |
| `unit` | keyword | **[revised, new]** the top-level subdirectory within the root — a repo *or* a plain folder. **Indexed.** This is the "search only Repo2" filter. A `repo_name` filter alone could not express it, because a workspace root holds non-repo folders that still need to be selectable, and they have no repo name. |
| `rel_path` | keyword | display path and dedup key — **indexed** |
| `file_name`, `ext` | keyword | "search only `.py`" / filename matching |
| `kind` | keyword | code / markdown / pdf / text — **indexed** |
| `language` | keyword | python / typescript / null — **indexed** |
| `is_repo` | bool | non-repo dirs are indexed too |
| `repo_name` | keyword | **indexed**, the "search only Repo2" filter |
| `repo_remote`, `repo_branch`, `repo_head_sha` | keyword | provenance; head_sha answers "was this indexed before or after that refactor" |
| `symbol_path` | text | `ClassName.method` / heading trail — shown in results |
| `symbol_kind`, `symbol_name` | keyword | filter to "classes only"; symbol_name aids exact-name lookup |
| `start_line`, `end_line` | integer | **`file:line` anchoring — the single most valuable field for an LLM consumer** |
| `source_text` | text | the actual chunk content |
| `token_count` | integer | lets the MCP layer budget a response without re-tokenizing |
| `content_sha` | keyword | staleness detection against disk |
| `chunk_index`, `chunk_total` | integer | reassemble a split function; fetch neighbors for context expansion |
| `space_slug`, `chunker_version`, `indexed_at` | keyword / datetime | which model and chunker produced this; drives targeted invalidation |

**Payload indexes** are created explicitly on `root_label`, `rel_path`, `kind`, `language`, `repo_name`, `repo_branch`, `symbol_kind`. Without them Qdrant filters by scanning, and filtered search over a large index gets slow in a way that looks like a vector problem but isn't.

**Storing `source_text` in the payload** costs disk but makes a search result self-contained — no disk read to render a hit. That matters for two reasons: the MCP server may not run on the same machine as the indexed files, and the file on disk may have changed since indexing, so re-reading would show text that doesn't match what actually matched. `content_sha` lets us flag a stale hit rather than silently showing the wrong thing.

### Hybrid query

```python
client.query_points(
    collection_name,
    prefetch=[
        Prefetch(query=dense_vec, using="dense", limit=cfg.prefetch_limit, filter=flt),
        Prefetch(query=sparse_vec, using="bm25",  limit=cfg.prefetch_limit, filter=flt),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=cfg.default_limit,
    with_payload=True,
)
```

Why hybrid is not optional for this use case: dense embeddings are good at "how does auth work" and bad at `QDRANT_ON_DISK_PAYLOAD` — a rare literal identifier has no semantic neighborhood. Code search needs both, and BM25 sparse is the half that makes exact symbol names, error strings, and config keys findable. RRF fuses on *rank*, not score, which is exactly right here because dense cosine and BM25 scores aren't on a comparable scale.

`fusion: dense_only | sparse_only` stays in config as a debugging tool — when a query returns junk, the first question is which branch produced the junk.

---

## Reranking

Full search path: **hybrid retrieve `candidates` (50) → rerank → return `top_n` (10)**.

`rerank/voyage.py` wraps `voyageai.Client.rerank(query, documents, model, top_k, truncation)`. API constraints, none of which bind us at ~512-token chunks: max 1,000 documents per call, query ≤ 8K tokens, query + any single document ≤ 32K, total ≤ 600K tokens per call.

**No `instruction` parameter exists**, despite `rerank-2.5*` being instruction-following models — the instruction is prepended to the query string. So `search.rerank.instruction` is concatenated client-side. Worth using: for a workspace index, "prioritize implementation code over tests, fixtures, and generated files" targets the most common failure mode of code search, where a test file that mentions a symbol twenty times outranks the one file that defines it.

**Rerank on `embed_text`, not `source_text`** (configurable). The reranker benefits from the same context header the embedder gets — a bare `def upsert(...)` body is ambiguous, `# file: storage/qdrant.py / # class QdrantStore` plus the body is not.

**Starting with `-lite` is right, and latency is the reason** — not cost. At 50 candidates × ~500 tokens ≈ 25K tokens per query, `$0.02/M` is ~$0.0005 a search; you could run thousands of queries for a dollar. The real cost is one extra API round trip on *every* search, and the primary consumer is an MCP tool an agent may call dozens of times in a single task. `-lite` is the latency-optimized variant, so it's the correct default; `rerank-2.5` is a one-line config change if the eval harness shows the quality gap justifies the wait.

### Turning it off must be boring

Three independent ways reranking ends up disabled, and none of them may break a search:

1. `rerank.enabled: false` → `NoopReranker` is constructed instead. No code path differs.
2. `enabled: true` but no `VOYAGE_API_KEY` / SDK missing → resolve to `NoopReranker`, log `rerank.skipped` **once per process** at INFO with the reason. A search must never fail because an optional quality enhancement isn't configured.
3. Configured correctly but the API call fails at query time → `on_error: degrade` (the default) logs `rerank.degraded` at WARNING and returns the RRF ordering. Results get worse; nothing errors. `on_error: fail` exists for the eval harness, where a silent degradation would corrupt a measurement.

The `rerank.call` log records **rank churn** — how far the reranker moved the top result. If churn is consistently zero, reranking is paying latency for nothing and the log will say so.

---

## Choosing dimensions

`voyage-code-4` supports 2048, 1024, 512, and 256 via Matryoshka learning, plus float32 / int8 / binary quantization.

**Plan: index at 2048, evaluate, expect to settle on 1024.**

The reasoning for 1024 as the likely production answer:

- **The reranker changes what dense retrieval is for.** With `rerank-2.5-lite` in the pipeline, the dense branch only needs to land the right chunk *somewhere in the top-50 candidate set*; final ordering is the reranker's job. Extra dimensions mainly sharpen fine-grained ranking precision — the exact work we just delegated. Spending 2× storage to improve the least load-bearing metric in the pipeline is the wrong allocation.
- **Cheaper levers exist for recall.** Raising `prefetch_limit` 50 → 100 costs almost nothing and targets candidate recall far more directly than doubling vector width.
- **2× storage, 2× HNSW memory, slower search**, at workspace scale on a VM.
- **Voyage publishes no 1024 → 2048 quality delta.** Their docs describe truncation as "a slight loss of retrieval quality" without numbers. An unquantified gain is usually a small one.

The reasoning for *embedding* at 2048 anyway:

Matryoshka means the first 1024 entries of a 2048-d vector **are** a valid 1024-d embedding. So we can derive the 1024 collection from vectors we already have — no re-embedding, no additional API spend. Hence `workspace-indexer reproject --dimensions 1024`: scroll the 2048 collection with `with_vectors=True`, truncate each dense vector (re-normalizing, so the same code is correct if we ever switch to dot-product distance), re-attach the existing sparse vector and payload, upsert into a new collection. Then run `workspace-indexer eval` against both.

The risk is asymmetric, which decides it: going 2048 → 1024 is free, going 1024 → 2048 is a full re-embed of the whole workspace. Start high, measure, then narrow.

**If storage becomes the pressure, reach for int8 quantization before fewer dimensions** — 4× reduction versus 2×, with less quality cost per byte saved. `EMBEDDING_QUANTIZATION` is a config field for this, unused in iteration 1.

---

## Measuring quality

Every knob above — dimensions, `fusion`, `prefetch_limit`, rerank model, chunk `max_tokens`, whether the context header helps — is a plausible-sounding choice that can only be settled by measurement. A minimal harness in iteration 1 keeps the whole thing from becoming folklore.

`config/eval.yaml`: a hand-written list of ~20–30 realistic queries against this actual workspace, each with the file paths that *should* come back.

```yaml
- query: "how does the file watcher decide between inotify and polling"
  expect: ["src/workspace_indexer/pipeline/watcher.py"]
- query: "where do we set the BM25 IDF modifier"
  expect: ["src/workspace_indexer/storage/qdrant.py"]
```

`workspace-indexer eval [--fusion X] [--rerank on|off] [--collection C]` reports recall@10 and MRR@10, and prints a per-query diff against the previous run so a regression is visible immediately. Writing the queries takes an hour and it is what makes every later tuning decision cheap.

---

## Chunking strategies

| Kind | Strategy |
|---|---|
| **Code** | **[revised]** `tree_sitter_language_pack.process()` with `chunk_max_size` derived from `max_tokens`; it already splits on definition boundaries and packs adjacent small definitions together. We map its output onto our `Chunk`: `context_path` → `symbol_path`, `symbols_defined` → `symbol_name`, and `has_error_nodes` → `parse_degraded` plus a `chunk.parse_failed` WARNING. Chunks below `min_tokens` are dropped (the parser emits a few degenerate ones). No grammar, a download failure, or a parse error → fall back to the text chunker; never drop the file. |
| **Markdown** | Split on heading structure up to `split_on_heading_depth`; `symbol_path` is the heading trail (`## Setup > ### WSL`). Sections over `max_tokens` split on paragraph boundaries, repeating the heading trail in each part's header. Fenced code blocks are never split mid-block. |
| **Text** | Paragraph-greedy packing to `max_tokens` with ~1-paragraph overlap. The universal fallback. |
| **PDF** *(iteration 2)* | `pymupdf` text extraction → treat as markdown-ish, chunk by page and detected heading, record page number in `symbol_path`. A PDF with no text layer (a scan) is recorded as OPAQUE rather than silently indexed as empty; OCR is a later decision. |
| **Image / binary** | `opaque.py` produces **zero vector chunks** but one manifest row, so the file is *known* and `status` can report it. See below. |

### On images — honest assessment

You can genuinely index images, but not with the same model as the code, and that has an architectural consequence worth deciding once.

`voyage-multimodal-3` embeds text and images into a *shared* vector space, so a text query can retrieve a screenshot. But it is a different model on a different endpoint (`voyageai.Client.multimodal_embed()`, which pydantic-ai's text-only `Embedder` does not expose), which means a different vector space from `voyage-code-4`. **Vectors from different models are not comparable** — cosine distance between them is noise. So images cannot live in the same Qdrant collection as code.

The design that handles this cleanly, and which the interfaces above already support:

- One collection per `EmbeddingSpace.slug()`. Code and docs go in the `voyage-code-4` collection; if we later enable image indexing, images go in a `voyage-multimodal-3` collection.
- Search fans out across registered spaces and merges by *rank* — the same RRF logic already used for dense+sparse fusion, reused rather than reinvented.

This is real added complexity, so **iteration 1 does not embed images.** It records them in the manifest with dimensions and mtime, and the `opaque.mode` knob (`metadata_only` → `caption` → `multimodal`) is the declared upgrade path. The middle option — captioning an image with a vision model and embedding the *caption text* — is worth trying first, because it keeps one model, one space, one collection. Revisit once there are actual images in `.claude/` to test against.

---

## Incremental indexing

This is the mechanism behind "we don't need to rewrite the entire app to reindex."

SQLite (`state/manifest.py`), three tables:

```sql
files (root_label, rel_path, abs_path, mtime_ns, size, sha256,
       kind, language, chunker_version, indexed_at,
       PRIMARY KEY (root_label, rel_path))

chunks (chunk_id PRIMARY KEY, root_label, rel_path, space_slug,
        embedded_at, token_count, FOREIGN KEY -> files)

runs (run_id PRIMARY KEY, started_at, finished_at, mode,
      files_seen, files_changed, chunks_upserted, chunks_deleted,
      tokens_embedded, est_cost_usd, errors, config_hash)
```

Per-file decision ladder, cheapest test first:

1. `mtime_ns` and `size` both unchanged → skip, zero reads. This is the common case and it's one `stat()` syscall.
2. Changed → read and hash. Same `sha256` → touch mtime, skip. (Catches the very common "file rewritten identically" — a formatter run, a `git checkout` of the same content.)
3. New hash → chunk. Compare the produced `chunk_id` set against the manifest: upsert only new ids, delete only vanished ids. A one-function edit in a 40-function file re-embeds one chunk.
4. `chunker_version` bumped (we changed a chunking strategy) → force re-chunk that kind, ignoring hashes.
5. Rows in `files` with no corresponding file on disk → `delete_by_path`, then drop the rows.
6. Any file whose chunks lack a row for the *current* `space_slug` → embed into that space. **This is the model-swap path:** change `EMBEDDING_MODEL`, and the next run backfills a new collection without discarding the old one — no wipe, no code change.

SQLite over "just query Qdrant's payloads" because the hot path is millions of cheap local existence checks, and because it survives the vector store being swapped out entirely. The `runs` table is what makes cost regressions visible over time rather than a surprise on the invoice.

---

## Repository layout to create

```
workspace-indexer/
├── pyproject.toml
├── .env.example
├── README.md
├── config/workspace.example.yaml
├── logs/                           # gitignored, hardcoded-excluded from indexing
├── data/                           # gitignored, hardcoded-excluded from indexing
├── src/workspace_indexer/
│   ├── cli.py                      # typer: index / search / status / explain
│   ├── models/                     # [revised] a package, one class per module:
│   │                               #   chunk, chunk_meta, chunk_id, file_kind,
│   │                               #   repo_info, source_file, sparse_vec,
│   │                               #   embedding_space, search_hit,
│   │                               #   search_filters, run_stats, hashing
│   ├── config/                     # [revised] likewise one class per module,
│   │                               #   ~17 of them, plus settings.py (.env)
│   │                               #   and excludes.py (hardcoded patterns)
│   ├── obs/
│   │   ├── logging.py              # structlog: console + RotatingFileHandler
│   │   ├── context.py              # contextvars binding (run_id, rel_path)
│   │   └── logfire_sink.py         # optional, guarded import
│   ├── discovery/                  # [revised] walker, ignore_matcher,
│   │                               #   git_metadata, classify, skip_reason,
│   │                               #   file_candidate
│   ├── chunking/{base,registry,code,markdown,text,opaque}.py
│   ├── embedding/{base,text_pydantic_ai,sparse_fastembed}.py
│   ├── rerank/{base,voyage,noop}.py
│   ├── storage/{base,qdrant}.py
│   ├── state/{manifest,schema.sql}
│   ├── search/{service,fusion,reproject}.py
│   ├── evaluation/harness.py       # not `eval/`: shadows a builtin
│   └── pipeline/indexer.py
└── tests/
    ├── fixtures/workspace/         # 2 fake git repos + .claude/ with md + png
    ├── test_discovery.py
    ├── test_chunking_code.py       # golden chunk assertions
    ├── test_chunking_markdown.py
    ├── test_manifest.py
    ├── test_logging.py
    ├── test_hybrid_search.py
    ├── test_rerank.py              # incl. all three disable paths
    ├── test_reproject.py           # matryoshka truncation correctness
    └── test_pipeline_e2e.py        # local model + embedded qdrant, no API key
```

---

## Iteration 1 deliverable

End-to-end but deliberately narrow:

- **Logging first**, before any other module: structlog console + rolling JSONL, contextvars binding, the event table above, optional Logfire sink defaulting to off.
- Config loading (`workspace.yaml` + `.env`) with validation and clear errors on a bad path.
- Discovery over the configured roots: scandir walk, `.gitignore` + config excludes via `pathspec`, hardcoded excludes for `logs/` and `data/`, size cap, symlink policy, git metadata per root (read via one `git` subprocess per root, not per file).
- **[revised]** Chunker registry live with **code (every language the pack ships)**, **markdown**, **text fallback**, and the **opaque stub**. The original plan scoped this to Python and TypeScript because it assumed hand-written node-type tables per language; `process()` removes that cost entirely. A startup `prefetch()` warms the grammar cache for the languages actually present in the workspace, so the download cost is paid once rather than mid-walk.
- Dense embedding via `pydantic_ai.Embedder` (`voyageai:voyage-code-4` at 2048 dims from `.env`) with batching, concurrency cap, and 429 backoff; sparse via `fastembed` BM25, local and free.
- Qdrant store with both named vectors, all payload indexes, embedded mode default and server mode via `QDRANT_MODE`.
- SQLite manifest with the full decision ladder and the `runs` table.
- Reranking via `rerank-2.5-lite` behind the `Reranker` protocol, with `NoopReranker` and all three graceful-disable paths.
- Minimal eval harness plus `reproject` so the dimension question is answered by measurement.
- CLI:
  - `workspace-indexer index [--root LABEL] [--dry-run] [--force]` — `--dry-run` prints the chunk plan and estimated token cost without calling the API, which is how we tune chunking without spending money.
  - `workspace-indexer search "query" [--repo R] [--lang L] [--kind K] [--fusion rrf|dense|sparse] [--rerank on|off] [--limit N]`
  - `workspace-indexer status` — files/chunks per root, spaces present, recent runs with token spend, skipped-file reasons.
  - `workspace-indexer explain PATH` — dump the chunks a single file produces. The chunk-quality debugging tool.
  - `workspace-indexer reproject --dimensions 1024` — derive a truncated Matryoshka collection with no re-embedding.
  - `workspace-indexer eval [--fusion X] [--rerank on|off] [--collection C]` — recall@10 / MRR@10 against `eval.yaml`.

**Not in iteration 1:** watcher, MCP server, PDF, image embedding, HTTP API, quantization.

### Later iterations

- **2 — Watcher & breadth.** `watchfiles` with `/proc/mounts`-based polling fallback, debounce/coalesce, config hot-reload adding roots at runtime, inotify watch-count reporting, PDF chunker, more languages.
- **3 — MCP server.** `search_code` and `get_file_context` tools. This is the primary consumer and it constrains result shape: hits must be token-budgeted (the `token_count` payload field exists for this), `file:line` anchored, and deduplicated by file so one query doesn't flood a context window. `chunk_index`/`chunk_total` enable context expansion around a hit.
- **4 — Retrieval quality, round two.** Grow `eval.yaml` from a smoke test into a real dataset; settle the 2048-vs-1024 and `rerank-2.5`-vs-`-lite` questions against it; int8 quantization if storage bites; per-language chunk tuning; a local cross-encoder reranker as the offline/no-API-key option.

---

## Verification

Each of these is runnable at the end of the iteration:

1. **No-API-key test suite.** `poetry run pytest`. **[revised]** Two local backends rather than one, because they answer different questions. `pydantic_ai.embeddings.TestEmbeddingModel` is free and instant but returns all-ones vectors, so it can only verify *plumbing* — batching, ids, upsert/delete accounting. For anything asserting that search returns semantically relevant results, use **fastembed's `BAAI/bge-small-en-v1.5`** (~130 MB ONNX, already a dependency for BM25) instead of `sentence-transformers`, which drags in PyTorch at roughly 2 GB for no added value here. Either way: no API key, no network at test time, no cost.
2. **Golden chunk tests.** `tests/fixtures/workspace/` holds two synthetic repos (`git init`-ed in a tmp dir by a fixture, so `RepoInfo` is real) plus a `.claude/` tree with markdown and a PNG. Assertions on exact chunk boundaries, `symbol_path` values, and header contents for a known Python file — these catch chunking regressions, which are otherwise invisible.
3. **Ignore-rule test.** Fixture contains `node_modules/` and `.venv/`; assert zero chunks from either, and assert a `.gitignore`d file in a repo is skipped while the same filename in the non-repo root is indexed. Separately assert `logs/` and `data/` are excluded even if a user adds them to `roots` — the loop-prevention guarantee.
4. **Incremental correctness.** Index the fixture, record chunk count and manifest state. Re-run → assert zero embedding calls (spy the backend). Edit one function body → assert exactly one chunk deleted and one upserted. Delete a file → assert its chunks are gone from the store. Rename → assert old path's chunks removed, new path's added.
5. **Model-swap test.** Index with model A, change the configured model to B, re-run → assert a second collection exists, both populated, and nothing deleted from A's collection.
6. **Hybrid search test.** Seed a fixture with (a) a distinctive rare identifier appearing in exactly one file and (b) a paraphrasable doc section. Assert `--fusion sparse` finds the identifier and `--fusion dense` finds the paraphrase, and that default RRF finds **both** in the top 5. This is the test that proves hybrid earns its complexity — if RRF doesn't beat both single branches here, the fusion config is wrong.
7. **Logging test.** Run an index against the fixture with a forced failure injected (unreadable file, mocked 429, a deliberately oversized chunk). Assert the JSONL file contains `embed.retry`, `embed.truncated`, and an `error.*` line, that every line carries `run_id`, and that failure lines carry `rel_path` from contextvars. Assert rotation works by setting `max_bytes` low and checking backups appear.
8. **Reranker disable paths.** Three separate tests, because this is the requirement most likely to be quietly half-implemented: (a) `enabled: false` returns RRF order and logs nothing at WARNING; (b) `enabled: true` with `VOYAGE_API_KEY` unset returns results, logs `rerank.skipped` exactly once across many searches, and raises nothing; (c) a mocked API 500 with `on_error: degrade` returns RRF order and logs `rerank.degraded`, while `on_error: fail` raises. Plus a positive test with a stubbed reranker that reverses scores, asserting the returned order actually changed — otherwise a no-op bug passes every other test.
9. **Matryoshka truncation.** Unit-test `reproject`: for a known 2048-d vector, assert the 1024-d output equals the normalized first 1024 components. Then index the fixture at 2048, reproject to 1024, and assert both collections have identical point counts and identical payloads, and that a search against each returns overlapping top hits. This proves the free-experiment path works before relying on it.
10. **Real run against this machine.** Point `workspace.yaml` at `~/src` with `voyageai:voyage-code-4` at 2048. Run `workspace-indexer index --dry-run` first to see the token estimate, then a real index, then `workspace-indexer search "how does the file watcher decide between inotify and polling"` and confirm the top hits are the relevant files. `workspace-indexer status` to sanity-check counts against `find | wc -l`, and `jq` the log for per-batch timings.
11. **The dimension experiment — the deliverable that answers your question.** After the real index at 2048: `workspace-indexer reproject --dimensions 1024`, then `workspace-indexer eval` against both collections, and `workspace-indexer eval --rerank off` against both. Four numbers. If 2048 doesn't clearly beat 1024 with reranking on, ship 1024 and reclaim the storage. Record the numbers in the README so the decision isn't re-argued later.
12. **Both Qdrant modes.** Same index+search with `QDRANT_MODE=embedded`, then with a `qdrant/qdrant` container and `QDRANT_MODE=server`; confirm identical top hits.
