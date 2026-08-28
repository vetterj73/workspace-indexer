"""One contract, every VectorStore implementation.

Written once and run against Qdrant and MongoDB Atlas, because the thing that
actually goes wrong with two backends is not a backend being wrong on its own
terms -- it is the two drifting apart while each stays internally consistent.
A change to filter translation, payload keys or fusion can leave one store
answering correctly and the other answering plausibly, and nothing above
`storage/` can tell the difference.

Every assertion here is about *observable behaviour through the protocol*.
Nothing reaches for a Qdrant filter object or a Mongo pipeline; those are
tested per backend, in `test_qdrant_filters.py` and `test_mongo_filter.py`.
What this file pins is the meaning both must agree on.

Qdrant runs on every invocation -- embedded, no network, no credentials, so CI
covers it. Atlas is marked `integration` and skipped without a connection
string. That asymmetry is deliberate and is the reason the Qdrant parameter
exists at all: a contract only one backend ever runs is a contract that
notices nothing in CI.

The store is module-scoped and seeded once. Not for speed: a Free Atlas
cluster allows three search indexes in total, this store needs two, and
dropping a collection frees the quota asynchronously -- so per-test setup
fails the second test outright. Mutating tests therefore use ids of their own
and clean up after themselves.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from tests.conftest import make_source
from workspace_indexer.chunking.chunk_factory import build_chunk
from workspace_indexer.config import Settings
from workspace_indexer.embedding.fastembed_sparse_backend import FastembedSparseBackend
from workspace_indexer.models import DocumentType, EmbeddingSpace, SearchFilters, compute_chunk_id
from workspace_indexer.storage.query_spec import QuerySpec
from workspace_indexer.storage.vector_store import VectorStore

# Every test on the module's loop. The store fixture is module-scoped -- it has
# to be, for the Atlas quota reason above -- and AsyncMongoClient binds to the
# loop it was created on, raising rather than reconnecting when used from
# another.
pytestmark = pytest.mark.asyncio(loop_scope="module")

WORKSPACE = "contract"
DIMENSIONS = 4

# Orthogonal vectors, so "which is nearest" has one answer and a failure means
# the query is wrong rather than the fixture ambiguous. The rare literal in
# `auth` is what the keyword branch has to find and the dense branch cannot.
CORPUS: list[dict[str, Any]] = [
    {
        "_id": "auth",
        "vector": [1.0, 0.0, 0.0, 0.0],
        "rel_path": "app/src/auth.py",
        "source_text": "verify the bearer token before the handler runs; ZZQBEACON9 marker",
        "doc_type": DocumentType.IMPLEMENTATION.value,
    },
    {
        "_id": "cake",
        "vector": [0.0, 1.0, 0.0, 0.0],
        "rel_path": "app/src/cake.py",
        "source_text": "cream the butter and sugar and bake the sponge",
        "doc_type": DocumentType.IMPLEMENTATION.value,
    },
    {
        "_id": "test",
        "vector": [0.0, 0.0, 1.0, 0.0],
        "rel_path": "app/tests/test_auth.py",
        "source_text": "asserts the bearer token is rejected once expired",
        "doc_type": DocumentType.TEST.value,
    },
    {
        "_id": "guide",
        "vector": [0.0, 0.0, 0.0, 1.0],
        "rel_path": "docs/guide.md",
        "source_text": "how to run the service locally and roll back a deployment",
        "doc_type": DocumentType.GUIDE.value,
    },
]


def chunk_id(name: str) -> str:
    """A real chunk id, from the real generator.

    Not an arbitrary string, and that is a contract fact rather than
    tidiness: Qdrant accepts only UUIDs or integers as point ids, while Mongo
    takes anything as `_id`. A change that made chunk ids non-UUID would break
    one backend and not the other -- precisely the drift this file exists to
    catch -- so the contract exercises the generator instead of inventing ids.
    """
    return compute_chunk_id("main", f"app/{name}.py", None, 0, name * 8)


def _payload(document: dict[str, Any]) -> dict[str, object]:
    rel_path: str = document["rel_path"]
    parts = rel_path.split("/")[:-1]
    return {
        "workspace": WORKSPACE,
        "root_label": "main",
        "unit": parts[0] if parts else "",
        "rel_path": rel_path,
        "abs_path": f"/tmp/{rel_path}",
        "file_name": rel_path.rsplit("/", 1)[-1],
        "ext": f".{rel_path.rsplit('.', 1)[-1]}",
        "ancestors": ["/".join(parts[: i + 1]) for i in range(len(parts))],
        "kind": "code",
        "language": "python",
        "is_repo": True,
        "repo_name": "app",
        "symbol_path": None,
        "symbol_kind": None,
        "symbol_name": None,
        "start_line": 1,
        "end_line": 5,
        "source_text": document["source_text"],
        "context_header": "",
        "token_count": 12,
        "content_sha": "0" * 64,
        "chunk_index": 0,
        "chunk_total": 1,
        "doc_type": document["doc_type"],
        "doc_confidence": 1.0,
        "space_slug": "contract_4",
    }


# Real BM25, not the fake one used elsewhere: this is the branch the contract
# is about, and both backends have to answer the same keyword question through
# their own mechanism -- Qdrant scoring the sparse vector we hand it, Atlas
# scoring the text with Lucene. A fake encoder would make the Qdrant half agree
# with itself and prove nothing about the pair.
_SPARSE = FastembedSparseBackend()


async def _seed(store: VectorStore, space: EmbeddingSpace) -> None:
    await store.ensure_collection(space)
    await store.upsert_points(
        space,
        [chunk_id(d["_id"]) for d in CORPUS],
        [d["vector"] for d in CORPUS],
        _SPARSE.encode_documents([d["source_text"] for d in CORPUS]),
        [_payload(d) for d in CORPUS],
    )


async def _qdrant() -> tuple[VectorStore, Any]:
    import tempfile

    from qdrant_client import AsyncQdrantClient

    from workspace_indexer.storage.qdrant_store import QdrantStore

    directory = tempfile.mkdtemp(prefix="contract-qdrant-")
    client = AsyncQdrantClient(path=directory)
    # Payload indexes are a no-op in embedded mode and warn per field.
    return QdrantStore(client, workspace=WORKSPACE, payload_indexes=False), directory


async def _mongo() -> tuple[VectorStore, Any]:
    from pymongo import AsyncMongoClient

    from workspace_indexer.storage.mongo_store import MongoStore

    settings = Settings()
    if not settings.mongodb_connection_string:
        pytest.skip("MONGODB_CONNECTION_STRING is not set")
    client: AsyncMongoClient[dict[str, Any]] = AsyncMongoClient(settings.mongodb_connection_string)
    return (
        MongoStore(client, workspace=WORKSPACE, database=settings.mongodb_database),
        None,
    )


@pytest_asyncio.fixture(
    scope="module",
    loop_scope="module",
    params=[
        pytest.param("qdrant", id="qdrant"),
        # Atlas needs credentials and roughly a minute of index build, so it
        # cannot run in CI. Qdrant carries the contract there.
        pytest.param("mongodb", id="mongodb", marks=pytest.mark.integration),
    ],
)
async def store(request: pytest.FixtureRequest) -> AsyncIterator[VectorStore]:
    backend: str = request.param
    made, handle = await (_qdrant() if backend == "qdrant" else _mongo())
    space = EmbeddingSpace(model="contract:model", dimensions=DIMENSIONS)
    await made.drop_collection(space)
    if backend == "mongodb":
        await _drop_every_contract_collection(made)
        await _seed_when_the_quota_allows(made, space)
        await _wait_until_queryable(made, space)
    else:
        await _seed(made, space)
    try:
        yield made
    finally:
        # Drop semantics asserted here rather than in a test of their own.
        # A second indexed collection is impossible on a Free Atlas cluster --
        # three search indexes in total, two per collection -- so the only
        # collection available to drop is this one, and the only moment it is
        # free to drop is after the last test. An assertion in teardown still
        # fails the run.
        await made.drop_collection(space)
        assert made.collection_name(space) not in await made.collection_names()
        # Repeating must be safe: a retry after a partial failure would
        # otherwise die on the cleanup rather than on the cause.
        await made.drop_collection(space)
        await made.close()
        if isinstance(handle, str):
            import shutil

            shutil.rmtree(handle, ignore_errors=True)


@pytest.fixture(scope="module")
def space() -> EmbeddingSpace:
    return EmbeddingSpace(model="contract:model", dimensions=DIMENSIONS)


async def _drop_every_contract_collection(store: VectorStore) -> None:
    """Leftovers from an aborted run hold the quota just as firmly.

    Scoped to this workspace's prefix so pointing the contract at a cluster
    that also holds a real index cannot touch it.
    """
    for name in await store.collection_names():
        if name.startswith(f"{WORKSPACE}__"):
            slug = name.split("__", 1)[1]
            model, _, dimensions = slug.rpartition("_")
            await store.drop_collection(
                EmbeddingSpace(model=model.replace("_", ":", 1), dimensions=int(dimensions))
            )


async def _seed_when_the_quota_allows(store: VectorStore, space: EmbeddingSpace) -> None:
    """Retry the seed until Atlas has released the previous run's indexes.

    A Free cluster allows three search indexes in total and this store needs
    two, and a dropped collection frees them *lazily* -- so recreating too soon
    fails with "The maximum number of FTS indexes has been reached", which
    reads like a quota problem and is a timing one.

    Retrying rather than polling for the absence of indexes, because there is
    no cluster-level view of the quota through the protocol: once the
    collection is gone, asking about its indexes returns "none" immediately
    while Atlas is still letting go of them. Trying to create is the only
    honest question.
    """
    import asyncio

    from pymongo.errors import OperationFailure

    for _ in range(18):
        try:
            await _seed(store, space)
        except OperationFailure as exc:
            if "maximum number of FTS indexes" not in str(exc):
                raise
            await asyncio.sleep(10)
        else:
            return
    pytest.fail(
        "Atlas never released the previous run's search indexes. A Free cluster "
        "allows three in total; check for other collections holding them."
    )


async def _wait_until_queryable(store: VectorStore, space: EmbeddingSpace) -> None:
    """Atlas builds search indexes asynchronously: a collection can hold every
    document and answer nothing for a minute, which looks like a broken query
    rather than an unfinished index."""
    import asyncio

    for _ in range(60):
        indexes = await store.describe_vectors(space)
        if indexes["dense"] and indexes["sparse"]:
            await asyncio.sleep(5)
            return
        await asyncio.sleep(5)
    pytest.fail("search indexes were not queryable in time")


# ---- identity ---------------------------------------------------------------


async def test_the_collection_name_is_the_same_formula_everywhere(
    store: VectorStore, space: EmbeddingSpace
) -> None:
    """A workspace indexed into both backends must be recognisably the same
    collection in each, or `status` and `reproject` describe different things
    depending on which store is configured."""
    assert store.collection_name(space) == f"{WORKSPACE}__{space.slug()}"


async def test_describe_names_something_a_human_can_act_on(store: VectorStore) -> None:
    """Preflight prints this when the index is empty. It used to be
    reconstructed from Qdrant settings, which reported an empty Mongo index as
    an empty Qdrant one."""
    described = store.describe()
    assert described and described.strip() == described


async def test_a_seeded_collection_counts_what_was_put_in_it(
    store: VectorStore, space: EmbeddingSpace
) -> None:
    assert await store.count(space) == len(CORPUS)


async def test_both_named_vectors_are_present(store: VectorStore, space: EmbeddingSpace) -> None:
    """A store missing its keyword half silently degrades hybrid search to
    dense-only, which looks like a quality problem rather than a schema one."""
    described = await store.describe_vectors(space)
    assert described["dense"], "no dense vector or index"
    assert described["sparse"], "no keyword index; hybrid search would be dense-only"


# ---- retrieval --------------------------------------------------------------


async def test_dense_search_returns_the_nearest_vector_with_a_real_score(
    store: VectorStore, space: EmbeddingSpace
) -> None:
    """A zero score means the backend was asked for the wrong relevance
    metadata -- which is not an error in either engine, just a missing field,
    so the ranking collapses silently."""
    hits = await store.search(
        space, QuerySpec(dense=[1.0, 0.0, 0.0, 0.0], text="", fusion="dense_only", limit=4)
    )
    assert hits, "dense search returned nothing"
    assert hits[0].rel_path == "app/src/auth.py"
    assert hits[0].score > 0.0


async def test_the_keyword_branch_finds_a_rare_literal(
    store: VectorStore, space: EmbeddingSpace
) -> None:
    """The half hybrid search exists for. A rare identifier has no semantic
    neighbourhood, so the dense branch cannot find it however good it is --
    this is what earns the second engine its complexity."""
    hits = await store.search(
        space,
        QuerySpec(
            dense=None,
            text="ZZQBEACON9",
            sparse=_SPARSE.encode_query("ZZQBEACON9"),
            fusion="sparse_only",
            limit=4,
        ),
    )
    assert [h.rel_path for h in hits][:1] == ["app/src/auth.py"]


async def test_a_hit_carries_the_payload_the_mcp_layer_reads(
    store: VectorStore, space: EmbeddingSpace
) -> None:
    """The payload is the contract shared by both backends. A missing field
    here surfaces as an MCP result an agent cannot act on."""
    hits = await store.search(
        space, QuerySpec(dense=[1.0, 0.0, 0.0, 0.0], text="", fusion="dense_only", limit=1)
    )
    hit = hits[0]
    assert hit.rel_path == "app/src/auth.py"
    assert hit.doc_type is DocumentType.IMPLEMENTATION
    assert hit.source_text.startswith("verify the bearer token")
    assert hit.start_line == 1 and hit.end_line == 5
    # `path:start-end`, the single most valuable field for an LLM consumer.
    assert hit.location == "app/src/auth.py:1-5"


async def test_limit_is_respected(store: VectorStore, space: EmbeddingSpace) -> None:
    hits = await store.search(
        space, QuerySpec(dense=[1.0, 1.0, 1.0, 1.0], text="", fusion="dense_only", limit=2)
    )
    assert len(hits) == 2


# ---- filtering --------------------------------------------------------------


async def test_an_excluded_document_type_never_comes_back(
    store: VectorStore, space: EmbeddingSpace
) -> None:
    """`search_code` drops tests by default. If the filter were applied after
    the search instead of inside it, this would still pass while quietly
    returning fewer than `limit` hits -- which the next test catches."""
    hits = await store.search(
        space,
        QuerySpec(dense=[0.0, 0.0, 1.0, 0.0], text="", fusion="dense_only", limit=4),
        SearchFilters(exclude_doc_types=[DocumentType.TEST]),
    )
    assert hits
    assert all(h.doc_type is not DocumentType.TEST for h in hits)


async def test_a_filtered_search_still_fills_the_limit(
    store: VectorStore, space: EmbeddingSpace
) -> None:
    """The post-filtering trap, asserted directly.

    Filtering a returned page silently shrinks the result set: ask for three
    hits in one repository and get one because the other two were elsewhere.
    Three of the four documents survive this filter, so a store that
    post-filters returns fewer.
    """
    hits = await store.search(
        space,
        QuerySpec(dense=[1.0, 1.0, 1.0, 1.0], text="", fusion="dense_only", limit=3),
        SearchFilters(exclude_doc_types=[DocumentType.TEST]),
    )
    assert len(hits) == 3


async def test_a_family_of_document_types_is_a_union_not_an_intersection(
    store: VectorStore, space: EmbeddingSpace
) -> None:
    """`find_guidance` asks for normative *or* design *or* guide. Expressed as
    an intersection it asks for a chunk that is somehow all three, and matches
    nothing -- an empty result that reads like "this workspace has no
    guidance"."""
    hits = await store.search(
        space,
        QuerySpec(dense=[0.0, 0.0, 0.0, 1.0], text="", fusion="dense_only", limit=4),
        SearchFilters(doc_types=[DocumentType.GUIDE, DocumentType.NORMATIVE]),
    )
    assert [h.rel_path for h in hits] == ["docs/guide.md"]


async def test_a_directory_restriction_matches_on_a_path_segment(
    store: VectorStore, space: EmbeddingSpace
) -> None:
    hits = await store.search(
        space,
        QuerySpec(dense=[1.0, 1.0, 1.0, 1.0], text="", fusion="dense_only", limit=4),
        SearchFilters(path_prefix="app/src"),
    )
    assert {h.rel_path for h in hits} == {"app/src/auth.py", "app/src/cake.py"}


async def test_count_and_search_agree_about_what_a_filter_means(
    store: VectorStore, space: EmbeddingSpace
) -> None:
    """`status` counts and `search` retrieves. If the two translated filters
    differently, status would report totals no search could reproduce."""
    filters = SearchFilters(exclude_doc_types=[DocumentType.TEST])
    counted = await store.count(space, filters)
    hits = await store.search(
        space,
        QuerySpec(dense=[1.0, 1.0, 1.0, 1.0], text="", fusion="dense_only", limit=10),
        filters,
    )
    assert counted == len(hits) == len(CORPUS) - 1


# ---- reading by path --------------------------------------------------------


async def test_chunks_for_path_takes_an_exact_path(
    store: VectorStore, space: EmbeddingSpace
) -> None:
    hits = await store.chunks_for_path(space, "app/src/auth.py")
    assert [h.rel_path for h in hits] == ["app/src/auth.py"]


async def test_chunks_for_path_takes_a_suffix_on_a_segment_boundary(
    store: VectorStore, space: EmbeddingSpace
) -> None:
    """An agent that typed the path itself usually gives a trailing portion.
    `auth.py` must find `app/src/auth.py` and must not find
    `app/tests/test_auth.py` -- a silently wrong file is worse than no answer.
    """
    hits = await store.chunks_for_path(space, "auth.py")
    assert [h.rel_path for h in hits] == ["app/src/auth.py"]


async def test_an_unknown_path_is_empty_rather_than_an_error(
    store: VectorStore, space: EmbeddingSpace
) -> None:
    assert await store.chunks_for_path(space, "nothing/here.py") == []


# ---- aggregates -------------------------------------------------------------


async def test_facet_counts_every_value_of_a_field(
    store: VectorStore, space: EmbeddingSpace
) -> None:
    """One round trip for the whole distribution. The taxonomy is built from
    this, and a count of zero there is a real answer an agent acts on."""
    counts = await store.facet(space, "doc_type")
    assert counts["implementation"] == 2
    assert counts["test"] == 1
    assert counts["guide"] == 1


async def test_sample_paths_are_distinct_files(store: VectorStore, space: EmbeddingSpace) -> None:
    """Chunks cluster by file, so the first three documents are often three
    chunks of one -- which makes a useless example set for the taxonomy."""
    paths = await store.sample_paths(space, None, limit=3)
    assert len(paths) == len(set(paths)) == 3


async def test_scroll_yields_every_point_with_its_payload(
    store: VectorStore, space: EmbeddingSpace
) -> None:
    found = {pid: payload async for pid, payload, _ in store.scroll(space)}
    assert set(found) == {chunk_id(d["_id"]) for d in CORPUS}
    assert found[chunk_id("auth")]["rel_path"] == "app/src/auth.py"


async def test_a_vector_survives_the_round_trip(store: VectorStore, space: EmbeddingSpace) -> None:
    """What `reproject` depends on: vectors read back have to be the vectors
    we paid to compute."""
    vectors = {pid: vec async for pid, _, vec in store.scroll(space, with_vectors=True)}
    stored = vectors[chunk_id("auth")]
    assert stored is not None, (
        "scroll returned no vector; reproject would write an empty collection"
    )
    assert stored["dense"] == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-6)


# ---- writing ----------------------------------------------------------------


async def test_re_upserting_an_id_replaces_rather_than_duplicates(
    store: VectorStore, space: EmbeddingSpace
) -> None:
    """The incremental path rewrites a chunk whose content changed. A store
    that appended would double the collection on every reindex."""
    before = await store.count(space)
    await store.upsert_points(
        space,
        [chunk_id("auth")],
        [[1.0, 0.0, 0.0, 0.0]],
        _SPARSE.encode_documents([CORPUS[0]["source_text"]]),
        [_payload(CORPUS[0])],
    )
    assert await store.count(space) == before


async def test_delete_by_ids_removes_exactly_those(
    store: VectorStore, space: EmbeddingSpace
) -> None:
    await store.upsert_points(
        space,
        [chunk_id("scratch-1")],
        [[0.5, 0.5, 0.0, 0.0]],
        _SPARSE.encode_documents(["scratch"]),
        [_payload(CORPUS[1])],
    )
    assert await store.count(space) == len(CORPUS) + 1

    await store.delete_by_ids(space, [chunk_id("scratch-1")])
    assert await store.count(space) == len(CORPUS)


async def test_delete_by_path_targets_the_root_and_path_pair(
    store: VectorStore, space: EmbeddingSpace
) -> None:
    """Two roots can hold the same relative path. Deleting on rel_path alone
    would take out a file in a repository nobody touched.

    This is also what makes a deleted or renamed file correct with no manifest
    lookup: everything at that path goes, whatever chunks it produced.
    """
    payload = {**_payload(CORPUS[1]), "rel_path": "app/src/scratch.py"}
    await store.upsert_points(
        space,
        [chunk_id("scratch-2")],
        [[0.5, 0.5, 0.0, 0.0]],
        _SPARSE.encode_documents(["scratch"]),
        [payload],
    )
    assert await store.count(space) == len(CORPUS) + 1

    await store.delete_by_path(space, "other-root", "app/src/scratch.py")
    assert await store.count(space) == len(CORPUS) + 1, "deleted from the wrong root"

    await store.delete_by_path(space, "main", "app/src/scratch.py")
    assert await store.count(space) == len(CORPUS)


async def test_deleting_nothing_is_not_an_error(store: VectorStore, space: EmbeddingSpace) -> None:
    await store.delete_by_ids(space, [])
    await store.delete_by_path(space, "main", "no/such/file.py")
    assert await store.count(space) == len(CORPUS)


async def test_a_collection_that_does_not_exist_counts_zero(
    store: VectorStore,
) -> None:
    """`status` and `preflight` ask before anything has been indexed. Raising
    here would make an empty index indistinguishable from a broken one."""
    absent = EmbeddingSpace(model="contract:absent", dimensions=DIMENSIONS)
    assert await store.count(absent) == 0
    assert await store.facet(absent, "doc_type") == {}
    assert await store.sample_paths(absent) == []


async def test_upsert_accepts_chunks_as_the_pipeline_hands_them_over(
    store: VectorStore, space: EmbeddingSpace
) -> None:
    """The other entry point. `upsert_points` carries an existing payload
    across for reprojection; `upsert` builds one from a Chunk, and the two must
    produce a hit that reads the same -- otherwise a reprojected collection
    answers differently from the one it was derived from.
    """
    source = make_source(
        "def handler():\n    return 1\n",
        rel_path="app/src/fresh.py",
        root_label="main",
        unit="app",
    )
    chunk = build_chunk(
        source,
        WORKSPACE,
        source_text="def handler():\n    return 1\n",
        start_line=1,
        end_line=2,
        chunker="code",
        version=1,
        chunk_index=0,
    )
    await store.upsert(
        space,
        [chunk],
        [[0.0, 0.0, 0.0, 1.0]],
        _SPARSE.encode_documents([chunk.embed_text]),
    )
    try:
        hits = await store.chunks_for_path(space, "app/src/fresh.py")
        assert [h.rel_path for h in hits] == ["app/src/fresh.py"]
        assert hits[0].source_text.startswith("def handler()")
    finally:
        await store.delete_by_path(space, "main", "app/src/fresh.py")


async def test_collection_names_lists_what_exists(
    store: VectorStore, space: EmbeddingSpace
) -> None:
    """`status` enumerates spaces with this, which is how a model swap is
    visible: the old collection must still be there beside the new one."""
    assert store.collection_name(space) in await store.collection_names()


async def test_the_store_satisfies_the_protocol(store: VectorStore) -> None:
    """Structural, so a method added to the protocol without an implementation
    fails here rather than at the call site in production."""
    assert isinstance(store, VectorStore)


async def test_every_protocol_method_is_covered_by_this_file() -> None:
    """The contract has to grow when the protocol does.

    Without this, adding a method to VectorStore leaves a hole that neither
    backend is checked against -- which is exactly how the two drift apart
    while each stays internally consistent.
    """
    import inspect

    covered = Path(__file__).read_text(encoding="utf-8")
    # `close` is exercised by the fixture teardown, where a leaked client shows
    # up as a warning rather than a failed assertion.
    exercised_by_the_fixture = {"close"}
    missing = [
        name
        for name, _ in inspect.getmembers(VectorStore, callable)
        if not name.startswith("_")
        and name not in exercised_by_the_fixture
        and f"store.{name}(" not in covered
    ]
    assert not missing, f"VectorStore methods with no contract test: {missing}"
