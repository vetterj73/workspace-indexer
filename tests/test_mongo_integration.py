"""The Mongo store against a real Atlas cluster.

Skipped unless MONGODB_CONNECTION_STRING is set, because there is no offline
Atlas: `mongomock` implements neither `$vectorSearch` nor `$search`, and a
plain `mongod` would accept the pipelines and answer them wrongly. Everything
else in the Mongo tests asserts the query documents we send; this is the only
place that finds out whether Atlas agrees with us about them.

Writes into a collection of its own and drops it afterwards, so pointing this
at a cluster holding a real index is safe. Uses four dimensions and a handful
of documents, so it costs no embedding tokens and a trivial amount of storage.

The one thing worth knowing before reading a failure here: Atlas search indexes
build **asynchronously**. A collection can hold documents and answer nothing
for a minute afterwards, which looks exactly like a broken query. `_wait_ready`
is why, and a timeout there is a slow cluster rather than a wrong pipeline.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

import pytest

from workspace_indexer.models import DocumentType, EmbeddingSpace, SearchFilters
from workspace_indexer.storage.mongo_store import MongoStore, build_document
from workspace_indexer.storage.query_spec import QuerySpec

CONNECTION = os.environ.get("MONGODB_CONNECTION_STRING")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not CONNECTION, reason="MONGODB_CONNECTION_STRING is not set"),
]

# Deliberately tiny and deliberately its own space slug, so the collection name
# cannot collide with a real one on the same cluster.
SPACE = EmbeddingSpace(model="itest:model", dimensions=4)
WORKSPACE = "workspace_indexer_itest"

# Orthogonal-ish vectors so "which one is nearest" has one obvious answer and a
# failure means the query is wrong rather than the fixture ambiguous.
DOCUMENTS = [
    ("auth", [1.0, 0.0, 0.0, 0.0], "verifying the bearer token before the handler runs", "impl"),
    ("cake", [0.0, 1.0, 0.0, 0.0], "cream the butter and sugar and bake the sponge", "impl"),
    ("test", [0.0, 0.0, 1.0, 0.0], "asserts the bearer token is rejected when expired", "test"),
]

READY_TIMEOUT_SECONDS = 300


@pytest.fixture
async def store() -> AsyncIterator[MongoStore]:
    from pymongo import AsyncMongoClient

    client: AsyncMongoClient[dict[str, object]] = AsyncMongoClient(CONNECTION)
    made = MongoStore(
        client,
        workspace=WORKSPACE,
        database=os.environ.get("MONGODB_DATABASE", "workspace_indexer"),
    )
    await made.drop_collection(SPACE)
    try:
        yield made
    finally:
        await made.drop_collection(SPACE)
        await made.close()


async def _seed(store: MongoStore) -> None:
    await store.ensure_collection(SPACE)
    await store.upsert_points(
        SPACE,
        [name for name, _, _, _ in DOCUMENTS],
        [vector for _, vector, _, _ in DOCUMENTS],
        [],
        [
            {
                "workspace": WORKSPACE,
                "root_label": "itest",
                "unit": "itest",
                "rel_path": f"src/{name}.py",
                "ancestors": ["src"],
                "kind": "code",
                "language": "python",
                "doc_type": (
                    DocumentType.TEST.value if role == "test" else DocumentType.IMPLEMENTATION.value
                ),
                "source_text": text,
                "context_header": "",
                "start_line": 1,
                "end_line": 5,
                "space_slug": SPACE.slug(),
            }
            for name, _, text, role in DOCUMENTS
        ],
    )
    await _wait_ready(store)


async def _wait_ready(store: MongoStore) -> None:
    """Block until both search indexes answer queries.

    Not a courtesy: a query against a building index returns zero documents
    with no error, so without this the first assertion in every test below
    would fail for a reason that has nothing to do with what it tests.
    """
    for _ in range(READY_TIMEOUT_SECONDS // 5):
        indexes = await store.describe_vectors(SPACE)
        if indexes["dense"] and indexes["sparse"]:
            # Queryable is necessary but the initial sync of documents into
            # mongot lags it slightly.
            await asyncio.sleep(5)
            return
        await asyncio.sleep(5)
    pytest.fail(f"search indexes were not queryable within {READY_TIMEOUT_SECONDS}s")


async def test_atlas_accepts_the_index_definitions_we_generate(store: MongoStore) -> None:
    """The first thing that can be wrong, and the cheapest to find out."""
    await _seed(store)
    assert await store.count(SPACE) == len(DOCUMENTS)


async def test_a_dense_search_finds_the_nearest_vector(store: MongoStore) -> None:
    await _seed(store)
    hits = await store.search(
        SPACE, QuerySpec(dense=[1.0, 0.0, 0.0, 0.0], fusion="dense_only", limit=3)
    )
    assert hits, "no dense hits; the vector index exists but returned nothing"
    assert hits[0].rel_path == "src/auth.py"
    # A zero score means we asked for the wrong $meta name -- the field is
    # simply absent, so nothing errors and the ranking collapses silently.
    assert hits[0].score > 0.0


async def test_a_text_search_finds_the_matching_words(store: MongoStore) -> None:
    """The half that is Lucene rather than our fastembed sparse vector."""
    await _seed(store)
    hits = await store.search(SPACE, QuerySpec(text="sponge", fusion="sparse_only", limit=3))
    assert [h.rel_path for h in hits] == ["src/cake.py"]
    assert hits[0].score > 0.0


async def test_hybrid_search_returns_both_branches(store: MongoStore) -> None:
    """Whether this goes through $rankFusion or the client-side fallback is
    the server's decision; the result must be the same either way."""
    await _seed(store)
    hits = await store.search(SPACE, QuerySpec(dense=[1.0, 0.0, 0.0, 0.0], text="sponge", limit=3))
    paths = {h.rel_path for h in hits}
    assert "src/auth.py" in paths, "the dense branch contributed nothing"
    assert "src/cake.py" in paths, "the text branch contributed nothing"


async def test_a_filter_is_applied_inside_the_search_not_after_it(store: MongoStore) -> None:
    """The assertion that catches a filter Atlas silently ignored, which is
    what an undeclared filter field produces."""
    await _seed(store)
    hits = await store.search(
        SPACE,
        QuerySpec(dense=[0.0, 0.0, 1.0, 0.0], fusion="dense_only", limit=3),
        SearchFilters(exclude_doc_types=[DocumentType.TEST]),
    )
    assert all(h.doc_type is not DocumentType.TEST for h in hits)
    assert "src/test.py" not in {h.rel_path for h in hits}


async def test_a_reindexed_chunk_replaces_rather_than_duplicates(store: MongoStore) -> None:
    await _seed(store)
    await store.upsert_points(
        SPACE, ["auth"], [[1.0, 0.0, 0.0, 0.0]], [], [{"rel_path": "src/auth.py"}]
    )
    assert await store.count(SPACE) == len(DOCUMENTS)


async def test_the_vector_survives_a_round_trip_through_bindata(store: MongoStore) -> None:
    """What `reproject` depends on: vectors read back out have to be the
    vectors we paid to compute."""
    await _seed(store)
    found = {
        point_id: vectors async for point_id, _, vectors in store.scroll(SPACE, with_vectors=True)
    }
    assert found["auth"] is not None
    assert found["auth"]["dense"] == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_the_document_we_build_is_within_the_bson_limit() -> None:
    """16 MB per document. A chunk is nowhere near it, but the failure mode if
    one ever were is a rejected write in the middle of a paid run."""
    import bson

    document = build_document("id", [0.1] * 2048, {"source_text": "x" * 100_000})
    assert len(bson.encode(document)) < 16 * 1024 * 1024
