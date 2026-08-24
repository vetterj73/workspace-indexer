"""Backend selection and credentials, from .env.

Kept separate from workspace.yaml on purpose: this half holds secrets and is
never committed, and it does not need to be hot-reloaded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- dense embeddings ----
    embedding_model: str = "voyageai:voyage-code-4"
    embedding_dimensions: int = 2048
    embedding_quantization: Literal["float32", "int8", "binary"] = "float32"
    embedding_batch_size: int = 64
    embedding_max_concurrency: int = 4
    # Providers cap total tokens per request, not just document count: 64
    # chunks that each happen to be huge fails the whole batch.
    embedding_max_batch_tokens: int = 100_000
    voyage_api_key: str | None = None

    # ---- sparse ----
    sparse_model: str = "Qdrant/bm25"

    # ---- reranking ----
    rerank_enabled: bool = True
    rerank_model: str = "voyageai:rerank-2.5-lite"

    # ---- vector store ----
    vector_store: Literal["qdrant"] = "qdrant"
    qdrant_mode: Literal["embedded", "server"] = "embedded"
    qdrant_path: Path = Path("./data/qdrant")
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_on_disk_payload: bool = True

    # ---- state ----
    state_db: Path = Path("./data/manifest.sqlite3")

    # ---- logging overrides (workspace.yaml wins unless these are set) ----
    log_level: str | None = None
    logfire_enabled: bool | None = None
    logfire_send_to_cloud: bool | None = None
    logfire_token: str | None = Field(default=None, repr=False)

    def config_hash(self) -> str:
        """Fingerprint of everything that affects index *content*.

        Credentials and transport settings are excluded — rotating an API key
        should not look like a configuration change that invalidates an index.
        """
        material = {
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "embedding_quantization": self.embedding_quantization,
            "sparse_model": self.sparse_model,
        }
        blob = json.dumps(material, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
