"""Wiring: build every layer from config, once, in one place.

The CLI commands then read as what they do rather than as assembly, and the
future MCP server gets the same construction for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from workspace_indexer.chunking import ChunkerRegistry
from workspace_indexer.classification import DocumentClassifier, RuleClassifier
from workspace_indexer.config import (
    LoggingConfig,
    Settings,
    WorkspaceConfig,
    load_workspace_config,
)
from workspace_indexer.embedding import (
    build_embedding_service,
    build_space,
    build_sparse_backend,
)
from workspace_indexer.embedding.embedding_service import EmbeddingService
from workspace_indexer.embedding.sparse_backend import SparseBackend
from workspace_indexer.models import EmbeddingSpace
from workspace_indexer.obs.logging import configure_logging
from workspace_indexer.pipeline import Indexer
from workspace_indexer.rerank import build_reranker
from workspace_indexer.rerank.reranker import Reranker
from workspace_indexer.search import SearchService
from workspace_indexer.state import Manifest
from workspace_indexer.storage import build_vector_store
from workspace_indexer.storage.qdrant_store import QdrantStore


@dataclass(slots=True)
class AppContext:
    config: WorkspaceConfig
    settings: Settings
    space: EmbeddingSpace
    manifest: Manifest
    registry: ChunkerRegistry
    embeddings: EmbeddingService
    sparse: SparseBackend
    store: QdrantStore
    reranker: Reranker
    classifier: DocumentClassifier

    @classmethod
    def build(cls, config_path: Path | None = None) -> AppContext:
        config = load_workspace_config(config_path)
        settings = Settings()

        # Before anything else runs, so a failure during setup is still logged.
        configure_logging(_with_env_overrides(config, settings))

        space = build_space(settings)
        return cls(
            config=config,
            settings=settings,
            space=space,
            manifest=Manifest(settings.state_db),
            registry=ChunkerRegistry(config.workspace.name),
            embeddings=build_embedding_service(settings),
            sparse=build_sparse_backend(settings),
            store=build_vector_store(settings, config.workspace.name),
            reranker=build_reranker(config.search.rerank, settings),
            classifier=RuleClassifier(),
        )

    def indexer(self) -> Indexer:
        return Indexer(
            config=self.config,
            settings=self.settings,
            manifest=self.manifest,
            registry=self.registry,
            embeddings=self.embeddings,
            sparse=self.sparse,
            store=self.store,
            space=self.space,
            classifier=self.classifier,
        )

    def search_service(self, space: EmbeddingSpace | None = None) -> SearchService:
        return SearchService(
            store=self.store,
            embeddings=self.embeddings,
            sparse=self.sparse,
            reranker=self.reranker,
            config=self.config.search,
            space=space or self.space,
        )

    async def close(self) -> None:
        await self.store.close()
        self.manifest.close()


def _with_env_overrides(config: WorkspaceConfig, settings: Settings) -> LoggingConfig:
    """.env wins over workspace.yaml for logging, so LOG_LEVEL=DEBUG works
    without editing a committed file."""
    logging_config = config.logging
    updates: dict[str, object] = {}
    if settings.log_level:
        updates["level"] = settings.log_level
    if settings.logfire_enabled is not None or settings.logfire_send_to_cloud is not None:
        updates["logfire"] = logging_config.logfire.model_copy(
            update={
                k: v
                for k, v in {
                    "enabled": settings.logfire_enabled,
                    "send_to_cloud": settings.logfire_send_to_cloud,
                }.items()
                if v is not None
            }
        )
    return logging_config.model_copy(update=updates) if updates else logging_config
