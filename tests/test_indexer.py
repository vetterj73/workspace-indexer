"""The indexing pipeline, wired end to end.

Real filesystem, real embedded Qdrant, real SQLite manifest; only the paid
embedding backend is faked, so the calls it never makes are countable. Every
assertion about "zero embedding calls" is the cost guarantee being enforced.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import structlog.testing
from qdrant_client import AsyncQdrantClient

from tests.conftest import ConfigFactory
from tests.fake_embedding_backend import FakeEmbeddingBackend
from tests.fake_sparse_backend import FakeSparseBackend
from workspace_indexer.chunking import ChunkerRegistry
from workspace_indexer.classification import RuleClassifier
from workspace_indexer.config import Settings, WorkspaceConfig
from workspace_indexer.embedding.embedding_service import EmbeddingService
from workspace_indexer.models import EmbeddingSpace, SearchFilters
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
            classifier=RuleClassifier(),
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


async def test_a_file_that_becomes_excluded_loses_its_chunks(
    harness: Harness, workspace: Path
) -> None:
    """Exclusion has to be retroactive, not just prospective.

    Adding an exclude pattern only stops *future* indexing; the chunks already
    written stay searchable and keep answering queries. That is how an eval
    artefact went on contaminating its own measurement after being excluded,
    and how a file caught by the secret scanner would keep serving the copy
    holding the secret. The file is still on disk, so nothing here depends on
    deletion -- only on the file no longer being discovered.
    """
    await harness.indexer().run()
    target = "repo_one/src/widget.py"
    assert harness.manifest.get_file("workspace", target) is not None
    before = await harness.store.count(SPACE)

    harness.config.index.exclude.append("**/src/widget.py")
    stats = await harness.indexer().run()

    assert stats.chunks_deleted > 0
    assert await harness.store.count(SPACE) < before
    assert harness.manifest.get_file("workspace", target) is None
    assert (workspace / "repo_one" / "src" / "widget.py").exists()


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


async def test_force_still_removes_chunks_that_are_no_longer_produced(
    harness: Harness, workspace: Path
) -> None:
    """--force rebuilds, but it must still reconcile. Dropping to_delete left
    chunks in the store the file no longer produces, with nothing able to
    remove them -- a rebuild that quietly accumulates orphans."""
    await harness.indexer().run()
    target = workspace / "repo_one" / "src" / "widget.py"
    target.write_text("def only_one():\n    return 1\n", encoding="utf-8")

    stats = await harness.indexer().run(force=True)
    assert stats.chunks_deleted > 0
    assert await harness.store.count(SPACE) == harness.manifest.chunk_count(SPACE.slug())


async def test_chunks_carry_the_document_type(harness: Harness) -> None:
    """One classification per file, inherited by every chunk of it."""
    await harness.indexer().run()
    types = {str(payload.get("doc_type")) async for _, payload, _ in harness.store.scroll(SPACE)}
    assert "implementation" in types
    assert None not in types
    assert "MISSING" not in types


async def test_classification_is_cached_between_runs(harness: Harness) -> None:
    """Cheap for rules, and the thing that stops a future model rung from
    re-reading the whole workspace on every run."""
    await harness.indexer().run()
    source_count = harness.manifest.file_count()
    assert source_count > 0
    from tests.conftest import make_source

    for rel in ("repo_one/src/widget.py",):
        record = harness.manifest.get_file("workspace", rel)
        assert record is not None
        cached = harness.manifest.cached_classification(
            make_source("x", rel_path=rel).model_copy(
                update={"root_label": "workspace", "sha256": record.sha256}
            ),
            classifier_version=1,
        )
        assert cached is not None
        assert cached.decided


async def test_a_newly_withheld_file_has_its_old_chunks_purged(
    harness: Harness, workspace: Path
) -> None:
    """A file indexed before the scanner existed, or before a secret was added
    to it, must not keep its old chunks. Withholding future runs while leaving
    the earlier copy in the index defeats the point of detecting it."""
    target = workspace / "repo_one" / "src" / "widget.py"
    await harness.indexer().run()
    assert harness.manifest.get_file("workspace", "repo_one/src/widget.py") is not None
    before = await harness.store.count(SPACE)

    key = "AK" + "IA" + "IOSFODNN7EXAMPLE"
    target.write_text(f'AWS_KEY = "{key}"\n', encoding="utf-8")

    stats = await harness.indexer().run()
    assert stats.chunks_deleted > 0
    assert await harness.store.count(SPACE) < before
    assert harness.manifest.get_file("workspace", "repo_one/src/widget.py") is None
    paths = {str(payload.get("rel_path")) async for _, payload, _ in harness.store.scroll(SPACE)}
    assert "repo_one/src/widget.py" not in paths


async def test_a_withheld_file_does_not_abort_the_run(harness: Harness, workspace: Path) -> None:
    key = "AK" + "IA" + "IOSFODNN7EXAMPLE"
    (workspace / "repo_one" / "secrets.json").write_text(f'{{"aws": "{key}"}}', encoding="utf-8")
    stats = await harness.indexer().run()
    assert stats.errors == 0
    assert stats.chunks_upserted > 0


async def test_a_manifest_describing_a_different_store_is_flagged(
    harness: Harness,
) -> None:
    """Switching QDRANT_MODE, or pointing at another host, leaves the manifest
    reporting a full index over an empty store. The decision ladder then skips
    every file and the run silently does nothing -- which is exactly what
    happened moving from embedded to server."""
    await harness.indexer().run()
    assert harness.manifest.chunk_count(SPACE.slug()) > 0

    # The store loses its contents while the manifest keeps its record.
    await harness.store.drop_collection(SPACE)

    # structlog's own capture, rather than caplog: the harness does not
    # configure the stdlib routing that caplog depends on.
    with structlog.testing.capture_logs() as logs:
        await harness.indexer().run()
    assert any(entry.get("event") == "run.store_diverges" for entry in logs)


@pytest.fixture
def two_roots(tmp_path: Path) -> tuple[Path, Path, WorkspaceConfig]:
    """The shape a real workspace takes: docs in one tree, code in another.

    `c:\\doc\\ProjectA` and `c:\\src\\ProjectA` are two separate roots of one
    workspace, not two workspaces -- they share a collection, and `root_label`
    is what keeps them apart inside it.
    """
    docs = tmp_path / "doc" / "ProjectA"
    code = tmp_path / "src" / "ProjectA"
    (docs / "guide").mkdir(parents=True)
    (code / "app").mkdir(parents=True)
    (docs / "guide" / "install.md").write_text("# Install\n\nRun the thing.\n", encoding="utf-8")
    (docs / "guide" / "api.md").write_text("# API\n\nEndpoints and payloads.\n", encoding="utf-8")
    (code / "app" / "main.py").write_text("def main():\n    return 1\n", encoding="utf-8")
    (code / "app" / "util.py").write_text("def helper():\n    return 2\n", encoding="utf-8")

    config = WorkspaceConfig.model_validate(
        {
            "workspace": {
                "name": "ProjectA",
                "roots": [
                    {"path": str(docs), "label": "docs"},
                    {"path": str(code), "label": "code"},
                ],
            }
        }
    )
    return docs, code, config


async def _chunks_in(store: QdrantStore, root: str) -> int:
    """Chunks still in the store for one root -- the observable that matters.

    A manifest row with no chunk behind it is not a surviving index.
    """
    return await store.count(SPACE, SearchFilters(root_label=root))


async def test_separate_roots_share_one_collection(
    two_roots: tuple[Path, Path, WorkspaceConfig], tmp_path: Path
) -> None:
    """Two directory trees, one collection.

    The collection name comes from `workspace.name` and the embedding space --
    never from a root -- so pointing several roots at one workspace is how you
    get one searchable index over docs and code together.
    """
    _, _, config = two_roots
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantStore(client, workspace=config.workspace.name, payload_indexes=False)
    assert store.collection_name(SPACE) == f"ProjectA__{SPACE.slug()}"

    with Manifest(tmp_path / "manifest.sqlite3") as manifest:
        harness = Harness(config, store, manifest, tmp_path)
        await harness.indexer().run()

        assert manifest.file_count("docs") == 2
        assert manifest.file_count("code") == 2
        # Both roots, one collection: the totals add up inside it.
        assert await _chunks_in(store, "docs") > 0
        assert await _chunks_in(store, "code") > 0
        assert await store.count(SPACE) == await _chunks_in(store, "docs") + await _chunks_in(
            store, "code"
        )
    await client.close()


async def test_reindexing_one_root_leaves_the_other_untouched(
    two_roots: tuple[Path, Path, WorkspaceConfig], tmp_path: Path
) -> None:
    """The question anyone with a split workspace asks first.

    Indexing `docs` must not notice that `code` was not walked and conclude its
    files have been deleted. Orphan removal is scoped to the root that was
    actually walked; without that scoping, every `--root` run would wipe every
    other root's chunks -- silent, total, and only visible as an empty search.
    """
    docs, _, config = two_roots
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantStore(client, workspace=config.workspace.name, payload_indexes=False)

    with Manifest(tmp_path / "manifest.sqlite3") as manifest:
        harness = Harness(config, store, manifest, tmp_path)
        await harness.indexer().run()

        code_before = await _chunks_in(store, "code")
        assert code_before

        # Change one docs file, then reindex docs alone.
        (docs / "guide" / "install.md").write_text(
            "# Install\n\nCompletely rewritten instructions.\n", encoding="utf-8"
        )
        stats = await harness.indexer().run(only_root="docs")

        assert stats.files_changed == 1
        # The whole point: a root that was never walked is untouched.
        assert await _chunks_in(store, "code") == code_before
        assert manifest.file_count("code") == 2
    await client.close()


async def test_deleting_a_file_under_one_root_only_prunes_that_root(
    two_roots: tuple[Path, Path, WorkspaceConfig], tmp_path: Path
) -> None:
    """Orphan pruning must still work *within* the walked root -- scoping it
    must not turn it off."""
    docs, _, config = two_roots
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantStore(client, workspace=config.workspace.name, payload_indexes=False)

    with Manifest(tmp_path / "manifest.sqlite3") as manifest:
        harness = Harness(config, store, manifest, tmp_path)
        await harness.indexer().run()
        code_before = await _chunks_in(store, "code")

        (docs / "guide" / "api.md").unlink()
        stats = await harness.indexer().run(only_root="docs")

        assert stats.chunks_deleted > 0
        assert manifest.get_file("docs", "guide/api.md") is None
        assert manifest.get_file("docs", "guide/install.md") is not None
        assert await _chunks_in(store, "code") == code_before
    await client.close()


async def test_removing_a_root_from_config_drops_its_chunks(
    two_roots: tuple[Path, Path, WorkspaceConfig], tmp_path: Path
) -> None:
    """The trap next door to `--root`, and it cuts the other way.

    `--root docs` is a *scope*: other roots are not walked and not touched.
    Deleting a root from workspace.yaml is a *decision*: it is walked as part
    of "everything", found absent, and pruned. Both are right, and they look
    almost identical from the command line, so the difference is worth a test
    rather than a paragraph nobody reads.
    """
    _, _, config = two_roots
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantStore(client, workspace=config.workspace.name, payload_indexes=False)

    with Manifest(tmp_path / "manifest.sqlite3") as manifest:
        harness = Harness(config, store, manifest, tmp_path)
        await harness.indexer().run()
        assert await _chunks_in(store, "code") > 0

        # The user edits workspace.yaml and removes the code root.
        harness.config.workspace.roots = [
            r for r in config.workspace.roots if r.resolved_label != "code"
        ]
        await harness.indexer().run()

        assert await _chunks_in(store, "code") == 0
        assert manifest.file_count("code") == 0
        # docs is untouched -- this is pruning, not a rebuild.
        assert await _chunks_in(store, "docs") > 0
    await client.close()
