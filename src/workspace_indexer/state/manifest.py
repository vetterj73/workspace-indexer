"""The SQLite manifest that drives incremental reindexing.

Deliberately synchronous. SQLite here is local single-file I/O measured in
microseconds, and wrapping each of tens of thousands of tiny queries in
asyncio.to_thread would cost more in executor hand-off than the queries
themselves. Writes are batched into transactions instead.

Not stored in Qdrant because the hot path is millions of cheap existence checks
against a local file, and because this has to survive swapping the vector store
out entirely.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

from workspace_indexer.discovery.file_candidate import FileCandidate
from workspace_indexer.models import Chunk, RunStats, SourceFile
from workspace_indexer.obs.logging import get_logger
from workspace_indexer.state.chunk_delta import ChunkDelta
from workspace_indexer.state.file_record import FileRecord
from workspace_indexer.state.index_decision import IndexDecision
from workspace_indexer.state.run_record import RunRecord

log = get_logger("workspace_indexer.state.manifest")

_SCHEMA = Path(__file__).with_name("schema.sql")


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Manifest:
    def __init__(self, path: Path) -> None:
        self._path = path.expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self._path, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        # WAL so a reader (an MCP server) is not blocked by the indexer's
        # writes. Without it SQLite takes a global write lock.
        self._db.executescript(_SCHEMA.read_text(encoding="utf-8"))

    def __enter__(self) -> Manifest:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()

    # ---- the decision ladder -------------------------------------------

    def decide_from_stat(
        self,
        candidate: FileCandidate,
        *,
        space_slug: str,
        chunker_version: int,
        force: bool = False,
    ) -> IndexDecision:
        """Rungs 1, 4 and 6 — everything answerable without opening the file."""
        if force:
            return IndexDecision.FORCED

        row = self._db.execute(
            "SELECT mtime_ns, size, chunker_version FROM files "
            "WHERE root_label = ? AND rel_path = ?",
            (candidate.root_label, candidate.rel_path),
        ).fetchone()
        if row is None:
            return IndexDecision.NEW

        if row["chunker_version"] != chunker_version:
            # We changed a strategy, so content hashes say nothing useful.
            return IndexDecision.RECHUNK_STRATEGY

        if not self._has_space(candidate.root_label, candidate.rel_path, space_slug):
            # The model-swap path: unchanged file, new vector space to fill.
            return IndexDecision.BACKFILL_SPACE

        if row["mtime_ns"] == candidate.mtime_ns and row["size"] == candidate.size:
            return IndexDecision.SKIP_UNCHANGED

        return IndexDecision.REINDEX

    def decide_from_hash(self, source: SourceFile) -> IndexDecision:
        """Rung 2 — the file looked changed, but is it?

        Catches the very common case of identical bytes rewritten: a formatter
        run, a `git checkout` of the same content, a `touch`. Costs one read
        and zero embedding calls.
        """
        row = self._db.execute(
            "SELECT sha256 FROM files WHERE root_label = ? AND rel_path = ?",
            (source.root_label, source.rel_path),
        ).fetchone()
        if row is None:
            return IndexDecision.NEW
        if row["sha256"] == source.sha256:
            return IndexDecision.SKIP_SAME_CONTENT
        return IndexDecision.REINDEX

    def touch(self, source: SourceFile) -> None:
        """Record the new mtime for a file whose content turned out identical,
        so the next run gets the cheap answer from rung 1."""
        self._db.execute(
            "UPDATE files SET mtime_ns = ?, size = ?, indexed_at = ? "
            "WHERE root_label = ? AND rel_path = ?",
            (source.mtime_ns, source.size, _now(), source.root_label, source.rel_path),
        )

    # ---- recording -----------------------------------------------------

    def record_file(self, source: SourceFile, *, chunker: str, chunker_version: int) -> None:
        self._db.execute(
            "INSERT INTO files (root_label, rel_path, abs_path, mtime_ns, size, sha256, "
            "kind, language, chunker, chunker_version, indexed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (root_label, rel_path) DO UPDATE SET "
            "abs_path = excluded.abs_path, mtime_ns = excluded.mtime_ns, "
            "size = excluded.size, sha256 = excluded.sha256, kind = excluded.kind, "
            "language = excluded.language, chunker = excluded.chunker, "
            "chunker_version = excluded.chunker_version, indexed_at = excluded.indexed_at",
            (
                source.root_label,
                source.rel_path,
                str(source.abs_path),
                source.mtime_ns,
                source.size,
                source.sha256,
                source.kind.value,
                source.language,
                chunker,
                chunker_version,
                _now(),
            ),
        )

    def diff_chunks(
        self, root_label: str, rel_path: str, space_slug: str, chunk_ids: Sequence[str]
    ) -> ChunkDelta:
        """Compare produced ids against stored ids for one file and space."""
        stored = {
            row["chunk_id"]
            for row in self._db.execute(
                "SELECT chunk_id FROM chunks "
                "WHERE root_label = ? AND rel_path = ? AND space_slug = ?",
                (root_label, rel_path, space_slug),
            )
        }
        produced = list(dict.fromkeys(chunk_ids))
        produced_set = set(produced)
        return ChunkDelta(
            to_upsert=[cid for cid in produced if cid not in stored],
            to_delete=sorted(stored - produced_set),
            unchanged=[cid for cid in produced if cid in stored],
        )

    def record_chunks(self, chunks: Iterable[Chunk], space_slug: str) -> None:
        rows = [
            (
                chunk.chunk_id,
                space_slug,
                chunk.meta.root_label,
                chunk.meta.rel_path,
                chunk.meta.content_sha,
                chunk.meta.token_estimate,
                _now(),
            )
            for chunk in chunks
        ]
        if not rows:
            return
        self._db.executemany(
            "INSERT INTO chunks (chunk_id, space_slug, root_label, rel_path, content_sha, "
            "token_count, embedded_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (chunk_id, space_slug) DO UPDATE SET "
            "content_sha = excluded.content_sha, token_count = excluded.token_count, "
            "embedded_at = excluded.embedded_at",
            rows,
        )

    def forget_chunks(self, chunk_ids: Sequence[str], space_slug: str) -> None:
        if not chunk_ids:
            return
        self._db.executemany(
            "DELETE FROM chunks WHERE chunk_id = ? AND space_slug = ?",
            [(cid, space_slug) for cid in chunk_ids],
        )

    def record_space(
        self, root_label: str, rel_path: str, space_slug: str, chunk_count: int
    ) -> None:
        """Mark a file complete for a space, including when it produced nothing.

        Without the zero case, every image and binary would look forever like a
        file whose chunks are missing, and re-chunk on every single run.
        """
        self._db.execute(
            "INSERT INTO file_spaces (root_label, rel_path, space_slug, chunk_count, "
            "embedded_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (root_label, rel_path, space_slug) DO UPDATE SET "
            "chunk_count = excluded.chunk_count, embedded_at = excluded.embedded_at",
            (root_label, rel_path, space_slug, chunk_count, _now()),
        )

    # ---- deletion ------------------------------------------------------

    def chunk_ids_for(self, root_label: str, rel_path: str, space_slug: str) -> list[str]:
        return [
            row["chunk_id"]
            for row in self._db.execute(
                "SELECT chunk_id FROM chunks "
                "WHERE root_label = ? AND rel_path = ? AND space_slug = ?",
                (root_label, rel_path, space_slug),
            )
        ]

    def forget_file(self, root_label: str, rel_path: str) -> None:
        """Drops the file and, by cascade, its chunk and space rows."""
        self._db.execute(
            "DELETE FROM files WHERE root_label = ? AND rel_path = ?", (root_label, rel_path)
        )

    def copy_space(self, source_slug: str, target_slug: str) -> int:
        """Record that every chunk in one space now also exists in another.

        Reprojection derives a narrower collection from vectors already paid
        for, deliberately preserving chunk ids. Without copying these rows the
        manifest has no record of the target space at all, so a later index
        cannot tell which chunks are already present and orphan cleanup cannot
        see them — which is how stale content ends up stranded in a live
        collection.

        Returns the number of chunk rows written.
        """
        self._db.execute(
            "INSERT INTO chunks (chunk_id, space_slug, root_label, rel_path, "
            "content_sha, token_count, embedded_at) "
            "SELECT chunk_id, ?, root_label, rel_path, content_sha, token_count, ? "
            "FROM chunks WHERE space_slug = ? "
            "ON CONFLICT (chunk_id, space_slug) DO UPDATE SET "
            "content_sha = excluded.content_sha, token_count = excluded.token_count, "
            "embedded_at = excluded.embedded_at",
            (target_slug, _now(), source_slug),
        )
        self._db.execute(
            "INSERT INTO file_spaces (root_label, rel_path, space_slug, chunk_count, "
            "embedded_at) "
            "SELECT root_label, rel_path, ?, chunk_count, ? "
            "FROM file_spaces WHERE space_slug = ? "
            "ON CONFLICT (root_label, rel_path, space_slug) DO UPDATE SET "
            "chunk_count = excluded.chunk_count, embedded_at = excluded.embedded_at",
            (target_slug, _now(), source_slug),
        )
        copied = self.chunk_count(target_slug)
        log.info("state.space_copied", source=source_slug, target=target_slug, chunks=copied)
        return copied

    def forget_space(self, space_slug: str) -> int:
        """Drop every record of one embedding space.

        Used when a collection is deleted: without this the manifest still
        believes the space exists, `status` reports collections that are gone,
        and the backfill rung thinks those files are already complete.

        The files themselves stay -- they may well be indexed in another space.
        Returns the number of chunk rows removed.
        """
        row = self._db.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE space_slug = ?", (space_slug,)
        ).fetchone()
        removed = int(row["n"])
        self._db.execute("DELETE FROM chunks WHERE space_slug = ?", (space_slug,))
        self._db.execute("DELETE FROM file_spaces WHERE space_slug = ?", (space_slug,))
        log.info("state.space_forgotten", space=space_slug, chunks=removed)
        return removed

    def orphans(self, seen: set[tuple[str, str]], root_label: str | None = None) -> list[
        tuple[str, str]
    ]:
        """Rung 5: rows with no corresponding file on disk.

        Scoped to a root when only one was walked, so indexing a single root
        does not delete every other root's index.
        """
        if root_label is None:
            rows = self._db.execute("SELECT root_label, rel_path FROM files")
        else:
            rows = self._db.execute(
                "SELECT root_label, rel_path FROM files WHERE root_label = ?", (root_label,)
            )
        return [
            (row["root_label"], row["rel_path"])
            for row in rows
            if (row["root_label"], row["rel_path"]) not in seen
        ]

    # ---- reads ---------------------------------------------------------

    def get_file(self, root_label: str, rel_path: str) -> FileRecord | None:
        row = self._db.execute(
            "SELECT * FROM files WHERE root_label = ? AND rel_path = ?", (root_label, rel_path)
        ).fetchone()
        return None if row is None else FileRecord(**dict(row))

    def file_count(self, root_label: str | None = None) -> int:
        if root_label is None:
            row = self._db.execute("SELECT COUNT(*) AS n FROM files").fetchone()
        else:
            row = self._db.execute(
                "SELECT COUNT(*) AS n FROM files WHERE root_label = ?", (root_label,)
            ).fetchone()
        return int(row["n"])

    def chunk_count(self, space_slug: str | None = None) -> int:
        if space_slug is None:
            row = self._db.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        else:
            row = self._db.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE space_slug = ?", (space_slug,)
            ).fetchone()
        return int(row["n"])

    def spaces(self) -> list[str]:
        return [
            row["space_slug"]
            for row in self._db.execute(
                "SELECT DISTINCT space_slug FROM chunks ORDER BY space_slug"
            )
        ]

    def counts_by_root(self) -> dict[str, int]:
        return {
            row["root_label"]: int(row["n"])
            for row in self._db.execute(
                "SELECT root_label, COUNT(*) AS n FROM files GROUP BY root_label"
            )
        }

    def _has_space(self, root_label: str, rel_path: str, space_slug: str) -> bool:
        row = self._db.execute(
            "SELECT 1 FROM file_spaces "
            "WHERE root_label = ? AND rel_path = ? AND space_slug = ?",
            (root_label, rel_path, space_slug),
        ).fetchone()
        return row is not None

    # ---- run history ---------------------------------------------------

    def start_run(self, stats: RunStats) -> None:
        self._db.execute(
            "INSERT INTO runs (run_id, started_at, mode, config_hash) VALUES (?, ?, ?, ?)",
            (stats.run_id, stats.started_at.isoformat(), stats.mode, stats.config_hash),
        )

    def finish_run(self, stats: RunStats) -> None:
        self._db.execute(
            "UPDATE runs SET finished_at = ?, files_seen = ?, files_skipped = ?, "
            "files_changed = ?, chunks_upserted = ?, chunks_deleted = ?, "
            "tokens_embedded = ?, est_cost_usd = ?, errors = ? WHERE run_id = ?",
            (
                (stats.finished_at or datetime.now(UTC)).isoformat(),
                stats.files_seen,
                stats.files_skipped,
                stats.files_changed,
                stats.chunks_upserted,
                stats.chunks_deleted,
                stats.tokens_embedded,
                stats.est_cost_usd,
                stats.errors,
                stats.run_id,
            ),
        )

    def recent_runs(self, limit: int = 10) -> list[RunRecord]:
        return [
            RunRecord(**dict(row))
            for row in self._db.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            )
        ]

    # ---- transactions --------------------------------------------------

    def begin(self) -> None:
        """Batch writes. Per-statement autocommit would fsync tens of
        thousands of times on a full index."""
        self._db.execute("BEGIN")

    def commit(self) -> None:
        self._db.execute("COMMIT")

    def rollback(self) -> None:
        self._db.execute("ROLLBACK")
