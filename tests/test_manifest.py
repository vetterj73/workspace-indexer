"""The manifest and its decision ladder.

This is where the incremental-reindex guarantee lives, so the tests are about
cost as much as correctness: each rung has to be reachable, and the expensive
rungs must not be reached when a cheaper one would answer.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.conftest import make_source
from workspace_indexer.chunking.chunk_factory import build_chunk
from workspace_indexer.discovery.file_candidate import FileCandidate
from workspace_indexer.models import Chunk, FileKind, RunStats, SourceFile
from workspace_indexer.state.index_decision import IndexDecision
from workspace_indexer.state.manifest import Manifest

SPACE = "voyageai_voyage-code-4_2048"
OTHER_SPACE = "voyageai_voyage-code-4_1024"
VERSION = 1


@pytest.fixture
def manifest(tmp_path: Path) -> Iterator[Manifest]:
    with Manifest(tmp_path / "manifest.sqlite3") as m:
        yield m


def _candidate(
    rel_path: str = "src/widget.py", *, mtime_ns: int = 1000, size: int = 200
) -> FileCandidate:
    return FileCandidate(
        root_label="repo_one",
        unit="repo_one",
        abs_path=Path("/tmp") / rel_path,
        rel_path=rel_path,
        kind=FileKind.CODE,
        language="python",
        size=size,
        mtime_ns=mtime_ns,
    )


def _source(
    rel_path: str = "src/widget.py", text: str = "x = 1", *, mtime_ns: int = 1000
) -> SourceFile:
    source = make_source(text, rel_path=rel_path)
    return source.model_copy(update={"mtime_ns": mtime_ns, "size": len(text.encode())})


def _chunk(source: SourceFile, body: str, index: int = 0) -> Chunk:
    return build_chunk(
        source,
        "labbox",
        source_text=body,
        start_line=1,
        end_line=1,
        chunker="code",
        version=VERSION,
        chunk_index=index,
        symbol_path=f"sym{index}",
    )


def _register(manifest: Manifest, source: SourceFile, chunks: list[Chunk]) -> None:
    """Record a file as fully indexed, the way the pipeline will."""
    manifest.record_file(source, chunker="code", chunker_version=VERSION)
    manifest.record_chunks(chunks, SPACE)
    manifest.record_space(source.root_label, source.rel_path, SPACE, len(chunks))


def _decide(
    manifest: Manifest, candidate: FileCandidate, *, force: bool = False
) -> IndexDecision:
    return manifest.decide_from_stat(
        candidate, space_slug=SPACE, chunker_version=VERSION, force=force
    )


# ---- the ladder --------------------------------------------------------


def test_unknown_file_is_new(manifest: Manifest) -> None:
    assert _decide(manifest, _candidate()) is IndexDecision.NEW


def test_unchanged_file_skips_without_a_read(manifest: Manifest) -> None:
    """Rung 1, the common case on every rerun: one stat(), zero reads."""
    source = _source()
    _register(manifest, source, [_chunk(source, "body")])
    decision = _decide(manifest, _candidate(mtime_ns=source.mtime_ns, size=source.size))
    assert decision is IndexDecision.SKIP_UNCHANGED
    assert not decision.needs_read


def test_changed_mtime_requires_a_read(manifest: Manifest) -> None:
    source = _source()
    _register(manifest, source, [_chunk(source, "body")])
    decision = _decide(manifest, _candidate(mtime_ns=9999, size=source.size))
    assert decision is IndexDecision.REINDEX
    assert decision.needs_read


def test_changed_size_alone_requires_a_read(manifest: Manifest) -> None:
    """Some tools preserve mtime while rewriting content, so size is checked
    too rather than trusted away."""
    source = _source()
    _register(manifest, source, [_chunk(source, "body")])
    decision = _decide(manifest, _candidate(mtime_ns=source.mtime_ns, size=source.size + 1))
    assert decision is IndexDecision.REINDEX


def test_identical_content_after_a_touch_costs_no_embedding(manifest: Manifest) -> None:
    """Rung 2: a formatter pass or a `git checkout` of the same bytes."""
    source = _source()
    _register(manifest, source, [_chunk(source, "body")])
    rewritten = _source(mtime_ns=9999)
    decision = manifest.decide_from_hash(rewritten)
    assert decision is IndexDecision.SKIP_SAME_CONTENT
    assert not decision.needs_embedding


def test_different_content_reindexes(manifest: Manifest) -> None:
    source = _source(text="x = 1")
    _register(manifest, source, [_chunk(source, "body")])
    decision = manifest.decide_from_hash(_source(text="x = 2", mtime_ns=9999))
    assert decision is IndexDecision.REINDEX
    assert decision.needs_embedding


def test_hash_decision_on_an_unknown_file_is_new(manifest: Manifest) -> None:
    assert manifest.decide_from_hash(_source("never/seen.py")) is IndexDecision.NEW


def test_touch_makes_the_next_run_take_the_cheap_path(manifest: Manifest) -> None:
    """After rung 2 decides the content is identical, the new mtime has to be
    recorded or every subsequent run pays the read again."""
    source = _source()
    _register(manifest, source, [_chunk(source, "body")])
    rewritten = _source(mtime_ns=9999)
    manifest.touch(rewritten)
    assert _decide(manifest, _candidate(mtime_ns=9999, size=rewritten.size)) is (
        IndexDecision.SKIP_UNCHANGED
    )


def test_chunker_version_bump_forces_a_rechunk(manifest: Manifest) -> None:
    """Rung 4: we changed a strategy, so content hashes say nothing useful."""
    source = _source()
    _register(manifest, source, [_chunk(source, "body")])
    decision = manifest.decide_from_stat(
        _candidate(mtime_ns=source.mtime_ns, size=source.size),
        space_slug=SPACE,
        chunker_version=VERSION + 1,
    )
    assert decision is IndexDecision.RECHUNK_STRATEGY
    assert decision.needs_embedding


def test_new_space_backfills_an_unchanged_file(manifest: Manifest) -> None:
    """Rung 6, the model-swap path: change EMBEDDING_MODEL and the next run
    fills a new collection without discarding the old one."""
    source = _source()
    _register(manifest, source, [_chunk(source, "body")])
    decision = manifest.decide_from_stat(
        _candidate(mtime_ns=source.mtime_ns, size=source.size),
        space_slug=OTHER_SPACE,
        chunker_version=VERSION,
    )
    assert decision is IndexDecision.BACKFILL_SPACE


def test_force_skips_straight_past_every_shortcut(manifest: Manifest) -> None:
    source = _source()
    _register(manifest, source, [_chunk(source, "body")])
    decision = _decide(
        manifest, _candidate(mtime_ns=source.mtime_ns, size=source.size), force=True
    )
    assert decision is IndexDecision.FORCED
    assert decision.needs_embedding


def test_a_file_that_produces_no_chunks_is_not_retried_forever(manifest: Manifest) -> None:
    """The bug this table exists to prevent: an image legitimately produces
    zero chunks, and without recording that it looks identical to a file whose
    chunks are missing — so every run would re-chunk every binary."""
    source = make_source("", kind=FileKind.IMAGE, language=None, rel_path="docs/logo.png")
    manifest.record_file(source, chunker="opaque", chunker_version=VERSION)
    manifest.record_chunks([], SPACE)
    manifest.record_space(source.root_label, source.rel_path, SPACE, 0)

    decision = manifest.decide_from_stat(
        _candidate("docs/logo.png", mtime_ns=source.mtime_ns, size=source.size),
        space_slug=SPACE,
        chunker_version=VERSION,
    )
    assert decision is IndexDecision.SKIP_UNCHANGED


# ---- chunk-level diffing ----------------------------------------------


def test_editing_one_function_touches_one_chunk(manifest: Manifest) -> None:
    """The heart of the cost story: forty functions in a file, one edited, one
    chunk re-embedded."""
    source = _source()
    original = [_chunk(source, f"body {i}", i) for i in range(40)]
    _register(manifest, source, original)

    edited = list(original)
    edited[7] = _chunk(source, "body 7 CHANGED", 7)
    delta = manifest.diff_chunks(
        source.root_label, source.rel_path, SPACE, [c.chunk_id for c in edited]
    )
    assert len(delta.to_upsert) == 1
    assert len(delta.to_delete) == 1
    assert len(delta.unchanged) == 39


def test_unchanged_file_diffs_to_nothing(manifest: Manifest) -> None:
    source = _source()
    chunks = [_chunk(source, f"body {i}", i) for i in range(5)]
    _register(manifest, source, chunks)
    delta = manifest.diff_chunks(
        source.root_label, source.rel_path, SPACE, [c.chunk_id for c in chunks]
    )
    assert delta.is_noop


def test_shrinking_a_file_deletes_the_surplus(manifest: Manifest) -> None:
    source = _source()
    chunks = [_chunk(source, f"body {i}", i) for i in range(5)]
    _register(manifest, source, chunks)
    delta = manifest.diff_chunks(
        source.root_label, source.rel_path, SPACE, [c.chunk_id for c in chunks[:2]]
    )
    assert delta.to_upsert == []
    assert len(delta.to_delete) == 3


def test_diff_is_scoped_to_the_space(manifest: Manifest) -> None:
    """A chunk present in the old space must still count as new in a fresh one,
    or a backfill would write nothing."""
    source = _source()
    chunks = [_chunk(source, "body", 0)]
    _register(manifest, source, chunks)
    delta = manifest.diff_chunks(
        source.root_label, source.rel_path, OTHER_SPACE, [c.chunk_id for c in chunks]
    )
    assert delta.to_upsert == [chunks[0].chunk_id]


def test_duplicate_produced_ids_are_collapsed(manifest: Manifest) -> None:
    """Two identical chunks in one file share an id; upserting it twice would
    double-count the cost."""
    source = _source()
    chunk = _chunk(source, "body", 0)
    delta = manifest.diff_chunks(
        source.root_label, source.rel_path, SPACE, [chunk.chunk_id, chunk.chunk_id]
    )
    assert delta.to_upsert == [chunk.chunk_id]


def test_forget_chunks_is_space_scoped(manifest: Manifest) -> None:
    source = _source()
    chunk = _chunk(source, "body", 0)
    _register(manifest, source, [chunk])
    manifest.record_chunks([chunk], OTHER_SPACE)
    manifest.forget_chunks([chunk.chunk_id], SPACE)
    assert manifest.chunk_count(SPACE) == 0
    assert manifest.chunk_count(OTHER_SPACE) == 1


# ---- deletion and rename ----------------------------------------------


def test_deleted_file_is_reported_as_an_orphan(manifest: Manifest) -> None:
    """Rung 5: a row with no file on disk."""
    source = _source()
    _register(manifest, source, [_chunk(source, "body")])
    assert manifest.orphans(seen=set()) == [("repo_one", "src/widget.py")]


def test_a_file_still_on_disk_is_not_an_orphan(manifest: Manifest) -> None:
    source = _source()
    _register(manifest, source, [_chunk(source, "body")])
    assert manifest.orphans(seen={("repo_one", "src/widget.py")}) == []


def test_orphan_detection_is_scoped_to_the_root_that_was_walked(manifest: Manifest) -> None:
    """Indexing one root must not delete every other root's index."""
    a = _source("a.py")
    b = _source("b.py").model_copy(update={"root_label": "repo_two"})
    _register(manifest, a, [_chunk(a, "body")])
    _register(manifest, b, [_chunk(b, "body")])
    assert manifest.orphans(seen=set(), root_label="repo_one") == [("repo_one", "a.py")]


def test_forgetting_a_file_cascades_to_its_chunks(manifest: Manifest) -> None:
    source = _source()
    _register(manifest, source, [_chunk(source, f"body {i}", i) for i in range(3)])
    assert manifest.chunk_count(SPACE) == 3
    manifest.forget_file(source.root_label, source.rel_path)
    assert manifest.chunk_count(SPACE) == 0
    assert manifest.file_count() == 0


def test_rename_removes_the_old_path_and_adds_the_new(manifest: Manifest) -> None:
    old = _source("src/old.py")
    _register(manifest, old, [_chunk(old, "body")])
    old_ids = manifest.chunk_ids_for(old.root_label, old.rel_path, SPACE)

    new = _source("src/new.py")
    _register(manifest, new, [_chunk(new, "body")])
    manifest.forget_file(old.root_label, old.rel_path)

    assert manifest.chunk_ids_for(old.root_label, old.rel_path, SPACE) == []
    new_ids = manifest.chunk_ids_for(new.root_label, new.rel_path, SPACE)
    assert new_ids and new_ids != old_ids


def test_chunk_ids_for_a_missing_file_is_empty_not_an_error(manifest: Manifest) -> None:
    assert manifest.chunk_ids_for("repo_one", "nope.py", SPACE) == []


# ---- reads and history ------------------------------------------------


def test_record_file_is_idempotent(manifest: Manifest) -> None:
    source = _source()
    for _ in range(3):
        manifest.record_file(source, chunker="code", chunker_version=VERSION)
    assert manifest.file_count() == 1


def test_get_file_round_trips(manifest: Manifest) -> None:
    source = _source()
    manifest.record_file(source, chunker="code", chunker_version=VERSION)
    record = manifest.get_file(source.root_label, source.rel_path)
    assert record is not None
    assert record.sha256 == source.sha256
    assert record.chunker == "code"
    assert record.language == "python"


def test_get_missing_file_is_none(manifest: Manifest) -> None:
    assert manifest.get_file("repo_one", "nope.py") is None


def test_counts_by_root_and_space(manifest: Manifest) -> None:
    a = _source("a.py")
    b = _source("b.py").model_copy(update={"root_label": "repo_two"})
    _register(manifest, a, [_chunk(a, "body")])
    _register(manifest, b, [_chunk(b, "body"), _chunk(b, "other", 1)])
    assert manifest.counts_by_root() == {"repo_one": 1, "repo_two": 1}
    assert manifest.chunk_count(SPACE) == 3
    assert manifest.chunk_count() == 3


def test_spaces_are_listed(manifest: Manifest) -> None:
    """`status` reports which collections exist, including after a model swap."""
    source = _source()
    chunk = _chunk(source, "body")
    _register(manifest, source, [chunk])
    manifest.record_chunks([chunk], OTHER_SPACE)
    assert manifest.spaces() == sorted([SPACE, OTHER_SPACE])


def test_run_history_records_cost(manifest: Manifest) -> None:
    """So "why did this cost $40" is answerable from the manifest rather than
    by scraping logs."""
    stats = RunStats(run_id="run1", started_at=datetime.now(UTC), mode="index",
                     config_hash="abc123")
    manifest.start_run(stats)
    stats.finished_at = datetime.now(UTC)
    stats.files_seen = 100
    stats.chunks_upserted = 42
    stats.tokens_embedded = 5000
    stats.est_cost_usd = 0.12
    manifest.finish_run(stats)

    runs = manifest.recent_runs()
    assert len(runs) == 1
    assert runs[0].chunks_upserted == 42
    assert runs[0].est_cost_usd == pytest.approx(0.12)
    assert runs[0].config_hash == "abc123"
    assert runs[0].finished_at is not None


def test_an_interrupted_run_stays_unfinished(manifest: Manifest) -> None:
    """A crashed run should be visible as such, not silently absent."""
    manifest.start_run(RunStats(run_id="run1", started_at=datetime.now(UTC)))
    assert manifest.recent_runs()[0].unfinished


# ---- durability -------------------------------------------------------


def test_state_survives_reopening(tmp_path: Path) -> None:
    path = tmp_path / "manifest.sqlite3"
    source = _source()
    with Manifest(path) as first:
        _register(first, source, [_chunk(source, "body")])
    with Manifest(path) as second:
        assert second.file_count() == 1
        assert second.chunk_count(SPACE) == 1


def test_wal_mode_is_enabled(tmp_path: Path) -> None:
    """So an MCP server can read while the indexer writes; without WAL SQLite
    takes a global write lock."""
    import sqlite3

    path = tmp_path / "manifest.sqlite3"
    with Manifest(path):
        pass
    db = sqlite3.connect(path)
    mode = db.execute("PRAGMA journal_mode").fetchone()[0]
    db.close()
    assert mode.lower() == "wal"


def test_parent_directory_is_created(tmp_path: Path) -> None:
    """Otherwise the failure lands at the first write, deep into a run."""
    with Manifest(tmp_path / "nested" / "deeper" / "manifest.sqlite3") as m:
        assert m.file_count() == 0


def test_schema_file_ships_with_the_package() -> None:
    """It is read at runtime relative to the module, so a packaging change that
    drops non-Python files would break every fresh install."""
    from workspace_indexer.state import manifest as module

    schema = Path(module.__file__).with_name("schema.sql")
    assert schema.is_file()
    assert "CREATE TABLE IF NOT EXISTS files" in schema.read_text(encoding="utf-8")


def test_transaction_can_be_rolled_back(manifest: Manifest) -> None:
    """Batched writes must not leave a half-indexed file behind on failure."""
    source = _source()
    manifest.begin()
    manifest.record_file(source, chunker="code", chunker_version=VERSION)
    manifest.rollback()
    assert manifest.file_count() == 0


def test_transaction_commits(manifest: Manifest) -> None:
    source = _source()
    manifest.begin()
    manifest.record_file(source, chunker="code", chunker_version=VERSION)
    manifest.commit()
    assert manifest.file_count() == 1


@pytest.mark.integration
def test_schema_sql_is_included_in_a_built_wheel(tmp_path: Path) -> None:
    """schema.sql is read at runtime, so a packaging change that drops
    non-Python files would break every fresh install while leaving the editable
    dev install working. Marked integration because it builds a wheel.
    """
    import subprocess
    import sys
    import zipfile

    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "wheel"
    # sys.executable, not "python": the interpreter running the tests is not
    # guaranteed to be on PATH under that name.
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", str(root), "-w", str(out), "--no-deps", "-q"],
        check=True,
        capture_output=True,
    )
    wheels = list(out.glob("*.whl"))
    assert wheels, "no wheel produced"
    names = zipfile.ZipFile(wheels[0]).namelist()
    assert "workspace_indexer/state/schema.sql" in names
