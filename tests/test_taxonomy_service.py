"""The taxonomy reports this index, not the enum's ambitions.

The acceptance criterion this file exists for: a category with nothing in it
must be reported at zero, never omitted. An agent cannot tell "this workspace
has no specifications" from "I forgot to ask about specifications" if the
category simply is not there.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from tests.conftest import make_source
from tests.fake_sparse_backend import FakeSparseBackend
from workspace_indexer.chunking.chunk_factory import build_chunk
from workspace_indexer.mcp import TAXONOMY_VERSION, TaxonomyService
from workspace_indexer.models import Chunk, DocumentType, EmbeddingSpace, FileKind
from workspace_indexer.storage.qdrant_store import QdrantStore

SPACE = EmbeddingSpace(model="fake:model", dimensions=4)

# Deliberately lopsided: several implementation chunks, one normative, and
# nothing at all in most categories.
SEEDED: list[tuple[str, DocumentType]] = [
    ("repo/CONVENTIONS.md", DocumentType.NORMATIVE),
    ("repo/src/a.py", DocumentType.IMPLEMENTATION),
    ("repo/src/b.py", DocumentType.IMPLEMENTATION),
    ("repo/src/c.py", DocumentType.IMPLEMENTATION),
]


async def _seed(store: QdrantStore, docs: list[tuple[str, DocumentType]]) -> None:
    sparse = FakeSparseBackend()
    chunks: list[Chunk] = []
    for index, (rel_path, doc_type) in enumerate(docs):
        source = make_source(
            "body text", kind=FileKind.CODE, language="python", rel_path=rel_path, unit="repo"
        )
        chunk = build_chunk(
            source,
            "labbox",
            source_text="body text",
            start_line=1,
            end_line=1,
            chunker="text",
            version=1,
            chunk_index=index,
        )
        chunk.meta.doc_type = doc_type
        chunks.append(chunk)
    await store.upsert(
        SPACE,
        chunks,
        [[1.0, 0, 0, 0]] * len(chunks),
        sparse.encode_documents(["body text"] * len(chunks)),
    )


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[QdrantStore]:
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantStore(client, workspace="labbox", payload_indexes=False)
    await _seed(store, SEEDED)
    yield store
    await client.close()


async def test_empty_categories_are_reported_at_zero(store: QdrantStore) -> None:
    """The acceptance criterion. `normative: 0` is a real answer -- it means
    read the code, because there is no written guidance to find."""
    taxonomy = await TaxonomyService(store, SPACE).build()
    by_name = {entry.name: entry for entry in taxonomy.types}

    assert set(by_name) == {t.value for t in DocumentType}
    assert by_name["guide"].count == 0
    assert by_name["record"].count == 0


async def test_counts_come_from_the_index(store: QdrantStore) -> None:
    taxonomy = await TaxonomyService(store, SPACE).build()
    by_name = {entry.name: entry.count for entry in taxonomy.types}
    assert by_name["implementation"] == 3
    assert by_name["normative"] == 1
    assert taxonomy.total_chunks == len(SEEDED)


async def test_every_entry_carries_a_definition(store: QdrantStore) -> None:
    """The definition is the only description of a category an agent sees
    before deciding whether to filter on it."""
    taxonomy = await TaxonomyService(store, SPACE).build()
    for entry in taxonomy.types:
        assert entry.definition
        assert entry.definition == DocumentType(entry.name).definition


async def test_examples_are_real_paths_from_this_workspace(store: QdrantStore) -> None:
    taxonomy = await TaxonomyService(store, SPACE).build()
    by_name = {entry.name: entry for entry in taxonomy.types}

    assert by_name["normative"].examples == ["repo/CONVENTIONS.md"]
    assert set(by_name["implementation"].examples) <= {
        "repo/src/a.py",
        "repo/src/b.py",
        "repo/src/c.py",
    }
    # No documents, so nothing to illustrate with -- and nothing invented.
    assert by_name["guide"].examples == []


async def test_examples_are_distinct_files(store: QdrantStore) -> None:
    """Chunks cluster by file, so a naive first-page sample returns the same
    document three times and teaches the model nothing."""
    docs = [("repo/src/big.py", DocumentType.IMPLEMENTATION)] * 20
    docs += [("repo/src/other.py", DocumentType.IMPLEMENTATION)]
    client = AsyncQdrantClient(location=":memory:")
    store2 = QdrantStore(client, workspace="labbox", payload_indexes=False)
    await _seed(store2, docs)

    taxonomy = await TaxonomyService(store2, SPACE).build()
    examples = {e.name: e.examples for e in taxonomy.types}["implementation"]
    assert len(examples) == len(set(examples))
    await client.close()


async def test_version_is_reported(store: QdrantStore) -> None:
    """Once agents filter on `normative`, renaming it is a breaking change."""
    taxonomy = await TaxonomyService(store, SPACE).build()
    assert taxonomy.taxonomy_version == TAXONOMY_VERSION
    assert taxonomy.space == SPACE.slug()


async def test_an_empty_index_still_lists_every_type(tmp_path: Path) -> None:
    """The worst moment to omit a category is when the index is empty."""
    client = AsyncQdrantClient(path=str(tmp_path / "empty"))
    store = QdrantStore(client, workspace="labbox", payload_indexes=False)
    taxonomy = await TaxonomyService(store, SPACE).build()

    assert len(taxonomy.types) == len(list(DocumentType))
    assert all(entry.count == 0 for entry in taxonomy.types)
    assert taxonomy.total_chunks == 0
    await client.close()
