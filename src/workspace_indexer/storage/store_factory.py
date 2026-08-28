"""Build the vector store from settings.

The one place that knows which backend is configured. Everything downstream is
typed against the `VectorStore` protocol, so adding a backend is a branch here
and a module beside it -- which is what the protocol was for.
"""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient

from workspace_indexer.config import RerankConfig, Settings
from workspace_indexer.obs.logging import get_logger, log_once
from workspace_indexer.storage.atlas_rerank import AtlasRerank
from workspace_indexer.storage.qdrant_store import QdrantStore
from workspace_indexer.storage.server_reranker import ServerReranker
from workspace_indexer.storage.vector_store import VectorStore

log = get_logger("workspace_indexer.storage.factory")


def build_qdrant_client(settings: Settings) -> AsyncQdrantClient:
    if settings.qdrant_mode == "embedded":
        path = settings.qdrant_path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Embedded mode is a single-process local store: the process holding it
        # takes a lock, so an MCP server cannot read while the indexer writes.
        # Fine for a first slice, and the reason server mode exists.
        log_once(
            log,
            "qdrant:embedded",
            "store.embedded_mode",
            path=str(path),
            detail="single-process only; run the server for concurrent read and write",
        )
        return AsyncQdrantClient(path=str(path))

    log.info("store.server_mode", url=settings.qdrant_url)
    return AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


# Rerank providers that mean "the store does it". Kept in step with the
# reranker factory's own list by a test, because the two deciding differently
# is the one failure this arrangement can have: both would decline to rerank
# and a search would quietly return fusion order.
DATABASE_PROVIDERS = frozenset({"database", "server"})


def build_vector_store(
    settings: Settings, workspace: str, rerank: RerankConfig | None = None
) -> VectorStore:
    server_rerank = _server_rerank(rerank)

    if settings.vector_store == "mongodb":
        return _build_mongo_store(settings, workspace, server_rerank)

    if server_rerank is not None:
        # Qdrant has no server-side reranker. Silently ignoring the setting
        # would leave the reranker factory declining to rerank as well, so
        # every search would return fusion order while the configuration
        # claimed otherwise.
        raise ValueError(
            f"rerank model {rerank.model if rerank else ''!r} asks the database to "
            "rerank, but VECTOR_STORE=qdrant has no server-side reranker. Use "
            "VECTOR_STORE=mongodb, or a client-side model such as "
            "voyageai:rerank-2.5-lite."
        )

    embedded = settings.qdrant_mode == "embedded"
    return QdrantStore(
        build_qdrant_client(settings),
        workspace=workspace,
        on_disk_payload=settings.qdrant_on_disk_payload,
        # Local Qdrant ignores payload indexes and warns once per field.
        payload_indexes=not embedded,
        location=(
            f"embedded qdrant at {settings.qdrant_path.expanduser().resolve()}"
            if embedded
            else f"qdrant server at {settings.qdrant_url}"
        ),
    )


def _server_rerank(rerank: RerankConfig | None) -> ServerReranker | None:
    """The store's reranker, or None when something else does the reranking."""
    if rerank is None or not rerank.enabled or rerank.provider not in DATABASE_PROVIDERS:
        return None
    return AtlasRerank(rerank.model_id, candidates=rerank.candidates)


def _build_mongo_store(
    settings: Settings, workspace: str, rerank: ServerReranker | None = None
) -> VectorStore:
    """Imported here rather than at module scope.

    pymongo is an optional extra. A top-level import would make the whole
    storage package unimportable for everyone running on Qdrant who has not
    installed it -- including CI, which has no reason to.
    """
    if not settings.mongodb_connection_string:
        raise ValueError(
            "VECTOR_STORE=mongodb needs MONGODB_CONNECTION_STRING in .env. "
            "Atlas gives you the string under Connect > Drivers; it carries the "
            "password inline, which is why it belongs in .env and never in "
            "workspace.yaml."
        )
    try:
        from pymongo import AsyncMongoClient

        from workspace_indexer.storage.mongo_store import MongoStore
    except ImportError as exc:  # pragma: no cover - exercised by hand, not CI
        raise ValueError(
            "VECTOR_STORE=mongodb needs the driver: `poetry install --extras mongo`."
        ) from exc

    log.info(
        "store.mongodb",
        database=settings.mongodb_database,
        dtype=settings.mongodb_vector_dtype,
        rerank=rerank.name if rerank else "none",
    )
    return MongoStore(
        AsyncMongoClient(settings.mongodb_connection_string),
        workspace=workspace,
        database=settings.mongodb_database,
        dtype=settings.mongodb_vector_dtype,
        rerank=rerank,
    )
