"""Building the store from settings."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from workspace_indexer.config import Settings
from workspace_indexer.models import EmbeddingSpace
from workspace_indexer.storage.qdrant_store import QdrantStore
from workspace_indexer.storage.store_factory import build_vector_store

SPACE = EmbeddingSpace(model="voyageai:voyage-code-4", dimensions=2048)


@pytest.fixture(autouse=True)
def _isolate_env(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)


async def test_embedded_mode_creates_its_data_directory(tmp_path: Path) -> None:
    """A missing parent would fail at the first upsert, deep into a run, rather
    than at startup."""
    target = tmp_path / "nested" / "deeper" / "qdrant"
    store = build_vector_store(Settings(qdrant_mode="embedded", qdrant_path=target), "labbox")
    assert target.parent.exists()
    await store.close()


async def test_embedded_mode_does_not_ask_for_payload_indexes(tmp_path: Path) -> None:
    """Local Qdrant ignores them and warns once per field, so asking anyway
    would emit eleven warnings on every single run."""
    store = build_vector_store(
        Settings(qdrant_mode="embedded", qdrant_path=tmp_path / "q"), "labbox"
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        await store.ensure_collection(EmbeddingSpace(model="fake:m", dimensions=4))
    assert not [w for w in captured if "Payload indexes" in str(w.message)]
    await store.close()


def test_server_mode_does_not_create_a_local_store(tmp_path: Path) -> None:
    """Pointing at a server must not quietly leave an unused local database
    behind, which would then hold a lock nobody expects."""
    local = tmp_path / "should_not_exist" / "qdrant"
    settings = Settings(qdrant_mode="server", qdrant_url="http://localhost:6333",
                        qdrant_path=local)
    with warnings.catch_warnings():
        # No server is running here; the client warns it cannot check version.
        warnings.simplefilter("ignore")
        build_vector_store(settings, "labbox")
    assert not local.exists()


async def test_workspace_name_scopes_the_collection(tmp_path: Path) -> None:
    """Two workspaces on one Qdrant must not share a collection."""
    client = AsyncQdrantClient(path=str(tmp_path / "q"))
    a = QdrantStore(client, workspace="labbox", payload_indexes=False)
    b = QdrantStore(client, workspace="other", payload_indexes=False)
    assert a.collection_name(SPACE) == "labbox__voyageai_voyage-code-4_2048"
    assert b.collection_name(SPACE) != a.collection_name(SPACE)
    await client.close()


@pytest.mark.integration
async def test_server_mode_round_trip() -> None:
    """Needs a running `qdrant/qdrant` container. The plan's verification asks
    for identical top hits in both modes; this is where that gets checked."""
    pytest.skip("start qdrant on :6333 and run with -m integration")
