-- Manifest schema.
--
-- Local, single-file, and deliberately not the vector store: the hot path is
-- millions of cheap existence checks, and this survives swapping Qdrant out
-- entirely.

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

-- One row per file we have seen. mtime_ns and size drive the fast path; sha256
-- is only ever compared after those say something changed.
CREATE TABLE IF NOT EXISTS files (
    root_label      TEXT    NOT NULL,
    rel_path        TEXT    NOT NULL,
    abs_path        TEXT    NOT NULL,
    mtime_ns        INTEGER NOT NULL,
    size            INTEGER NOT NULL,
    sha256          TEXT    NOT NULL,
    kind            TEXT    NOT NULL,
    language        TEXT,
    chunker         TEXT,
    chunker_version INTEGER NOT NULL DEFAULT 0,
    -- What role this document plays, cached against the content hash above so
    -- unchanged bytes are never reclassified. classifier_version invalidates
    -- precisely the files a rule change should affect.
    doc_type        TEXT    NOT NULL DEFAULT 'unknown',
    doc_confidence  REAL    NOT NULL DEFAULT 0.0,
    doc_reason      TEXT    NOT NULL DEFAULT '',
    classifier_version INTEGER NOT NULL DEFAULT 0,
    indexed_at      TEXT    NOT NULL,
    PRIMARY KEY (root_label, rel_path)
);

-- Which embedding spaces a given file has been embedded into, and how many
-- chunks it produced. chunk_count exists so a file that legitimately produces
-- nothing -- an image, a binary -- is recorded as complete rather than looking
-- forever like a file whose chunks are missing.
CREATE TABLE IF NOT EXISTS file_spaces (
    root_label  TEXT    NOT NULL,
    rel_path    TEXT    NOT NULL,
    space_slug  TEXT    NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    embedded_at TEXT    NOT NULL,
    PRIMARY KEY (root_label, rel_path, space_slug),
    FOREIGN KEY (root_label, rel_path) REFERENCES files (root_label, rel_path)
        ON DELETE CASCADE
);

-- One row per chunk per space. The primary key is the pair, not the id alone:
-- the same chunk exists in two spaces during a model swap, which is the whole
-- point of being able to backfill a new collection without discarding the old.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT    NOT NULL,
    space_slug  TEXT    NOT NULL,
    root_label  TEXT    NOT NULL,
    rel_path    TEXT    NOT NULL,
    content_sha TEXT    NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    embedded_at TEXT    NOT NULL,
    PRIMARY KEY (chunk_id, space_slug),
    FOREIGN KEY (root_label, rel_path) REFERENCES files (root_label, rel_path)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS chunks_by_file
    ON chunks (root_label, rel_path, space_slug);

-- Run history, so "why did this cost $40" is answerable from the manifest
-- rather than by scraping logs, and a cost regression is visible over time.
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    started_at      TEXT    NOT NULL,
    finished_at     TEXT,
    mode            TEXT    NOT NULL,
    files_seen      INTEGER NOT NULL DEFAULT 0,
    files_skipped   INTEGER NOT NULL DEFAULT 0,
    files_changed   INTEGER NOT NULL DEFAULT 0,
    chunks_upserted INTEGER NOT NULL DEFAULT 0,
    chunks_deleted  INTEGER NOT NULL DEFAULT 0,
    tokens_embedded INTEGER NOT NULL DEFAULT 0,
    est_cost_usd    REAL    NOT NULL DEFAULT 0.0,
    errors          INTEGER NOT NULL DEFAULT 0,
    config_hash     TEXT
);
