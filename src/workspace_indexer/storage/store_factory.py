"""Build the vector store from settings."""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient

from workspace_indexer.config import Settings
from workspace_indexer.obs.logging import get_logger, log_once
from workspace_indexer.storage.qdrant_store import QdrantStore

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


def build_vector_store(settings: Settings, workspace: str) -> QdrantStore:
    embedded = settings.qdrant_mode == "embedded"
    return QdrantStore(
        build_qdrant_client(settings),
        workspace=workspace,
        on_disk_payload=settings.qdrant_on_disk_payload,
        # Local Qdrant ignores payload indexes and warns once per field.
        payload_indexes=not embedded,
    )
