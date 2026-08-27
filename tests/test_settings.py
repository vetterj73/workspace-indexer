"""Environment-driven settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from workspace_indexer.config import Settings, WorkspaceConfig


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


def _config(**index: Any) -> WorkspaceConfig:
    return WorkspaceConfig.model_validate(
        {
            "workspace": {"name": "w", "roots": [{"path": "/tmp/a", "label": "a"}]},
            "index": index,
        }
    )


def test_config_hash_tracks_content_affecting_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    before = Settings().config_hash(config)
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1024")
    assert Settings().config_hash(config) != before


def test_config_hash_ignores_credentials_and_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rotating an API key must not look like a configuration change that
    invalidates the whole index."""
    config = _config()
    before = Settings().config_hash(config)
    monkeypatch.setenv("VOYAGE_API_KEY", "rotated-key")
    monkeypatch.setenv("QDRANT_URL", "http://elsewhere:6333")
    assert Settings().config_hash(config) == before


def test_config_hash_tracks_which_roots_were_indexed() -> None:
    """The bug this parameter exists for.

    The hash claimed to fingerprint "everything that affects index content"
    while omitting *what was indexed*. Two runs over entirely different corpora
    shared a hash, so an eval delta between them was reported as though it
    meant something.
    """
    settings = Settings()
    one = WorkspaceConfig.model_validate(
        {"workspace": {"name": "w", "roots": [{"path": "/tmp/a", "label": "a"}]}}
    )
    two = WorkspaceConfig.model_validate(
        {
            "workspace": {
                "name": "w",
                "roots": [{"path": "/tmp/a", "label": "a"}, {"path": "/tmp/b", "label": "b"}],
            }
        }
    )
    assert settings.config_hash(one) != settings.config_hash(two)


def test_config_hash_tracks_exclusions() -> None:
    """A file excluded is a file not indexed, which moves recall exactly as
    adding a root does."""
    settings = Settings()
    assert settings.config_hash(_config()) != settings.config_hash(
        _config(exclude=["**/vendor/**"])
    )
    assert settings.config_hash(_config()) != settings.config_hash(_config(respect_gitignore=False))
    assert settings.config_hash(_config()) != settings.config_hash(_config(max_file_bytes=1024))


def test_root_order_is_not_a_change() -> None:
    """Reordering roots in the YAML indexes exactly the same files."""
    settings = Settings()
    forward = WorkspaceConfig.model_validate(
        {
            "workspace": {
                "name": "w",
                "roots": [{"path": "/tmp/a", "label": "a"}, {"path": "/tmp/b", "label": "b"}],
            }
        }
    )
    reversed_ = WorkspaceConfig.model_validate(
        {
            "workspace": {
                "name": "w",
                "roots": [{"path": "/tmp/b", "label": "b"}, {"path": "/tmp/a", "label": "a"}],
            }
        }
    )
    assert settings.config_hash(forward) == settings.config_hash(reversed_)


def test_an_equivalent_root_path_is_not_a_change() -> None:
    """`~/src` and the absolute path it expands to are one corpus."""
    settings = Settings()
    tilde = WorkspaceConfig.model_validate(
        {"workspace": {"name": "w", "roots": [{"path": "~/src", "label": "s"}]}}
    )
    absolute = WorkspaceConfig.model_validate(
        {"workspace": {"name": "w", "roots": [{"path": str(Path.home() / "src"), "label": "s"}]}}
    )
    assert settings.config_hash(tilde) == settings.config_hash(absolute)


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
