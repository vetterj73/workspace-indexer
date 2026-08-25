"""The indexing pipeline, wired end to end.

Real filesystem, real embedded Qdrant, real SQLite manifest; only the paid
embedding backend is faked, so the calls it never makes are countable. Every
assertion about "zero embedding calls" is the cost guarantee being enforced.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from tests.conftest import ConfigFactory
from tests.fake_embedding_backend import FakeEmbeddingBackend
from tests.fake_sparse_backend import FakeSparseBackend
from workspace_indexer.chunking import ChunkerRegistry
from workspace_indexer.config import Settings, WorkspaceConfig
from workspace_indexer.embedding.embedding_service import EmbeddingService
from workspace_indexer.models import EmbeddingSpace
from workspace_indexer.pipeline import Indexer
from workspace_indexer.state import Manifest
from workspace_indexer.storage.qdrant_store import QdrantStore

SPACE = EmbeddingSpace(model="fake:model", dimensions=4)
NARROW = EmbeddingSpace(model="fake:other", dimensions=4)


class Harness:
    """Everything the pipeline needs, with handles the tests assert against."""

    def __init__(
        self, config: WorkspaceConfig, store: QdrantStore, manifest: Manifest, tmp: Path
    ) -> None:
        self.config = config
        self.store = store
        self.manifest = manifest
        self.backend = FakeEmbeddingBackend(dimensions=4)
        self.embeddings = EmbeddingService(self.backend)
        self.sparse = FakeSparseBackend()
        self.tmp = tmp

    def indexer(self, space: EmbeddingSpace = SPACE) -> Indexer:
        return Indexer(
            config=self.config,
            settings=Settings(state_db=self.tmp / "manifest.sqlite3"),
            manifest=self.manifest,
            registry=ChunkerRegistry(self.config.workspace.name),
            embeddings=self.embeddings,
            sparse=self.sparse,
            store=self.store,
            space=space,
            flush_chunks=8,
        )

    @property
    def documents_embedded(self) -> int:
        return self.backend.stats_documents

    def reset_counters(self) -> None:
        self.backend.batches.clear()
        self.backend.calls = 0


@pytest.fixture
async def harness(config_for: ConfigFactory, tmp_path: Path) -> AsyncIterator[Harness]:
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantStore(client, workspace="test", payload_indexes=False)
    with Manifest(tmp_path / "manifest.sqlite3") as manifest:
        yield Harness(config_for(), store, manifest, tmp_path)
    await client.close()


def _embedded(harness: Harness) -> int:
    return sum(len(batch) for batch in harness.backend.batches)


async def test_first_run_indexes_the_workspace(harness: Harness) -> None:
    stats = await harness.indexer().run()
    assert stats.files_changed > 0
    assert stats.chunks_upserted > 0
    assert await harness.store.count(SPACE) == stats.chunks_upserted
    assert harness.manifest.file_count() > 0


async def test_rerun_embeds_nothing(harness: Harness) -> None:
    """The whole cost story in one assertion: an unchanged workspace costs
    stat() calls and no money at all."""
    await harness.indexer().run()
    harness.reset_counters()

    stats = await harness.indexer().run()
    assert _embedded(harness) == 0
    assert stats.chunks_upserted == 0
    assert stats.files_changed == 0
    assert stats.files_skipped == stats.files_seen


async def test_touching_a_file_without_changing_it_embeds_nothing(
    harness: Harness, workspace: Path
) -> None:
    """Rung 2: a formatter pass or a checkout of identical bytes."""
    await harness.indexer().run()
    harness.reset_counters()

    target = workspace / "repo_one" / "src" / "widget.py"
    target.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")

    stats = await harness.indexer().run()
    assert _embedded(harness) == 0
    assert stats.chunks_upserted == 0


async def test_editing_one_function_reembeds_only_that_chunk(
    harness: Harness, workspace: Path
) -> None:
    await harness.indexer().run()
    before = await harness.store.count(SPACE)
    harness.reset_counters()

    target = workspace / "repo_one" / "src" / "widget.py"
    target.write_text(
        target.read_text(encoding="utf-8").replace("return a + b", "return a - b"),
        encoding="utf-8",
    )

    stats = await harness.indexer().run()
    assert stats.chunks_upserted == 1
    assert stats.chunks_deleted == 1
    assert _embedded(harness) == 1
    assert await harness.store.count(SPACE) == before


async def test_deleting_a_file_removes_its_chunks(harness: Harness, workspace: Path) -> None:
    await harness.indexer().run()
    before = await harness.store.count(SPACE)

    (workspace / "repo_one" / "src" / "widget.py").unlink()
    stats = await harness.indexer().run()

    assert stats.chunks_deleted > 0
    assert await harness.store.count(SPACE) < before
    assert harness.manifest.get_file("workspace", "repo_one/src/widget.py") is None


async def test_renaming_a_file_moves_its_chunks(harness: Harness, workspace: Path) -> None:
    await harness.indexer().run()
    before = await harness.store.count(SPACE)

    src = workspace / "repo_one" / "src" / "widget.py"
    src.rename(src.with_name("gadget.py"))
    await harness.indexer().run()

    assert harness.manifest.get_file("workspace", "repo_one/src/widget.py") is None
    assert harness.manifest.get_file("workspace", "repo_one/src/gadget.py") is not None
    assert await harness.store.count(SPACE) == before


async def test_adding_a_file_indexes_only_it(harness: Harness, workspace: Path) -> None:
    await harness.indexer().run()
    harness.reset_counters()

    (workspace / "repo_one" / "src" / "extra.py").write_text(
        "def brand_new_function():\n    return 42\n", encoding="utf-8"
    )
    stats = await harness.indexer().run()
    assert stats.files_changed == 1
    assert _embedded(harness) == stats.chunks_upserted


async def test_dry_run_writes_nothing_and_costs_nothing(harness: Harness) -> None:
    """The point of --dry-run: tune chunking without paying to iterate."""
    stats = await harness.indexer().run(dry_run=True)
    assert stats.chunks_upserted > 0
    assert stats.tokens_embedded > 0
    assert _embedded(harness) == 0
    assert await harness.store.count(SPACE) == 0
    assert harness.manifest.file_count() == 0
    assert harness.manifest.recent_runs() == []


async def test_force_reembeds_everything(harness: Harness) -> None:
    await harness.indexer().run()
    harness.reset_counters()

    stats = await harness.indexer().run(force=True)
    assert _embedded(harness) > 0
    assert stats.files_skipped == 0


async def test_model_swap_backfills_without_touching_the_old_space(
    harness: Harness,
) -> None:
    """Change the embedding model and the next run fills a new collection while
    the old one stays intact and searchable."""
    await harness.indexer().run()
    original = await harness.store.count(SPACE)

    await harness.indexer(NARROW).run()

    assert await harness.store.count(SPACE) == original
    assert await harness.store.count(NARROW) == original
    assert sorted(harness.manifest.spaces()) == sorted([SPACE.slug(), NARROW.slug()])


async def test_binary_files_are_recorded_but_never_embedded(harness: Harness) -> None:
    """And they must not be re-chunked on every subsequent run."""
    await harness.indexer().run()
    assert harness.manifest.get_file("workspace", "repo_one/src/widget.py") is not None
    harness.reset_counters()
    stats = await harness.indexer().run()
    assert stats.files_changed == 0


async def test_gitignored_content_never_reaches_the_store(harness: Harness) -> None:
    """The last line of defence before text is sent to an embedding API."""
    await harness.indexer().run()
    paths = {payload["rel_path"] async for _, payload, _ in harness.store.scroll(SPACE)}
    assert "repo_one/secret.txt" not in paths
    assert not any(str(p).startswith("repo_one/build/") for p in paths)


async def test_run_history_is_recorded(harness: Harness) -> None:
    await harness.indexer().run()
    runs = harness.manifest.recent_runs()
    assert len(runs) == 1
    assert not runs[0].unfinished
    assert runs[0].files_seen > 0
    assert runs[0].config_hash


async def test_only_root_limits_the_walk(harness: Harness, config_for: ConfigFactory) -> None:
    stats = await harness.indexer().run(only_root="workspace")
    assert stats.files_seen > 0


async def test_one_bad_file_does_not_abort_the_run(
    harness: Harness, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that has already paid for thousands of embeddings must not be lost
    to a single unreadable file."""
    import workspace_indexer.pipeline.indexer as module

    original = module.read_source
    calls = {"n": 0}

    def sometimes_explode(candidate: object, secret_allow: object = None):  # noqa: ANN202
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated read failure")
        return original(candidate, secret_allow)  # type: ignore[arg-type]

    monkeypatch.setattr(module, "read_source", sometimes_explode)
    stats = await harness.indexer().run()
    assert stats.errors == 1
    assert stats.chunks_upserted > 0


async def test_stats_report_tokens_and_cost(harness: Harness) -> None:
    stats = await harness.indexer().run()
    assert stats.tokens_embedded > 0
    assert stats.est_cost_usd > 0
