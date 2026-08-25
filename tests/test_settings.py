"""Environment-driven settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from workspace_indexer.config import Settings


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


def test_api_keys_are_exported_to_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """pydantic-settings loads .env into the object, but provider SDKs read
    credentials from os.environ. Without this bridge the key is loaded and then
    invisible to the thing that needs it."""
    import os

    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    exported = Settings(voyage_api_key="from-dotenv").export_credentials()
    assert "VOYAGE_API_KEY" in exported
    assert os.environ["VOYAGE_API_KEY"] == "from-dotenv"


def test_a_real_environment_variable_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shell is more specific than a file. Overriding it would make
    `VOYAGE_API_KEY=x command` silently do nothing."""
    import os

    monkeypatch.setenv("VOYAGE_API_KEY", "from-shell")
    exported = Settings(voyage_api_key="from-dotenv").export_credentials()
    assert "VOYAGE_API_KEY" not in exported
    assert os.environ["VOYAGE_API_KEY"] == "from-shell"


def test_absent_credentials_are_not_exported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    assert Settings(voyage_api_key=None).export_credentials() == []


def test_export_returns_names_never_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """The return value is logged, so it must not carry the secret."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    exported = Settings(voyage_api_key="super-secret-value").export_credentials()
    assert exported == ["VOYAGE_API_KEY"]
    assert not any("super-secret" in name for name in exported)
