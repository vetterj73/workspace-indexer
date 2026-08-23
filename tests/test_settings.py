"""Environment-driven settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from dirindex.config import Settings


@pytest.fixture(autouse=True)
def _isolate_env(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Settings reads .env from the cwd, which would otherwise leak the
    developer's real credentials into assertions."""
    monkeypatch.chdir(tmp_path)
    for key in ("EMBEDDING_MODEL", "EMBEDDING_DIMENSIONS", "VOYAGE_API_KEY", "RERANK_ENABLED"):
        monkeypatch.delenv(key, raising=False)


def test_defaults() -> None:
    settings = Settings()
    assert settings.embedding_model == "voyageai:voyage-code-4"
    assert settings.embedding_dimensions == 2048
    assert settings.qdrant_mode == "embedded"


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "openai:text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1536")
    settings = Settings()
    assert settings.embedding_model == "openai:text-embedding-3-small"
    assert settings.embedding_dimensions == 1536


def test_config_hash_tracks_content_affecting_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    before = Settings().config_hash()
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1024")
    assert Settings().config_hash() != before


def test_config_hash_ignores_credentials_and_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rotating an API key must not look like a configuration change that
    invalidates the whole index."""
    before = Settings().config_hash()
    monkeypatch.setenv("VOYAGE_API_KEY", "rotated-key")
    monkeypatch.setenv("QDRANT_URL", "http://elsewhere:6333")
    assert Settings().config_hash() == before
