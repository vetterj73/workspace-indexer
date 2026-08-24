"""The Qdrant payload contract.

These keys are read by the CLI, the MCP server and the reranker, and the
payload indexes have to name the same fields the filters use. A rename here
breaks all three, so the contract is pinned.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import make_source
from workspace_indexer.chunking.chunk_factory import build_chunk
from workspace_indexer.models import EmbeddingSpace, FileKind, RepoInfo, SearchFilters
from workspace_indexer.storage.payload import (
    INDEXED_FIELDS,
    ancestors_of,
    to_payload,
    to_search_hit,
)

SPACE = EmbeddingSpace(model="voyageai:voyage-code-4", dimensions=2048)
REPO = RepoInfo(name="workspace-indexer", branch="main", head_sha="a" * 40, remote_url="git@x:y")


def _chunk(**overrides: object):
    file = make_source(
        "def upsert(self):\n    pass",
        kind=FileKind.CODE,
        language="python",
        rel_path="src/workspace_indexer/storage/qdrant_store.py",
        repo=REPO,
    )
    kwargs: dict[str, object] = {
        "source_text": "def upsert(self):\n    pass",
        "start_line": 40,
        "end_line": 41,
        "chunker": "code",
        "version": 1,
        "symbol_path": "QdrantStore.upsert",
        "symbol_kind": "function",
        "symbol_name": "upsert",
    }
    kwargs.update(overrides)
    return build_chunk(file, "labbox", **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("rel_path", "expected"),
    [
        ("a.py", []),
        ("src/a.py", ["src"]),
        ("src/pkg/mod/a.py", ["src", "src/pkg", "src/pkg/mod"]),
    ],
)
def test_ancestors_are_every_directory_prefix(rel_path: str, expected: list[str]) -> None:
    """Qdrant cannot prefix-match a keyword, and post-filtering the returned
    page would silently shrink the result set."""
    assert ancestors_of(rel_path) == expected


def test_payload_carries_the_location_fields() -> None:
    payload = to_payload(_chunk(), SPACE)
    assert payload["rel_path"] == "src/workspace_indexer/storage/qdrant_store.py"
    assert payload["file_name"] == "qdrant_store.py"
    assert payload["ext"] == ".py"
    assert payload["start_line"] == 40
    assert payload["end_line"] == 41
    assert "src/workspace_indexer" in payload["ancestors"]


def test_payload_carries_provenance() -> None:
    payload = to_payload(_chunk(), SPACE)
    assert payload["is_repo"] is True
    assert payload["repo_name"] == "workspace-indexer"
    assert payload["repo_branch"] == "main"
    assert payload["repo_head_sha"] == "a" * 40


def test_non_repo_files_record_that_fact_rather_than_omitting_it() -> None:
    file = make_source("body", kind=FileKind.TEXT, language=None, repo=None)
    chunk = build_chunk(file, "labbox", source_text="body", start_line=1, end_line=1,
                        chunker="text", version=1)
    payload = to_payload(chunk, SPACE)
    assert payload["is_repo"] is False
    assert payload["repo_name"] is None


def test_payload_stores_only_the_header_not_a_second_copy_of_the_source() -> None:
    """embed_text is header + source. Storing both would double the payload to
    hold a copy of something already there."""
    chunk = _chunk()
    payload = to_payload(chunk, SPACE)
    assert payload["source_text"] == chunk.source_text
    assert chunk.source_text not in payload["context_header"]
    assert payload["context_header"].startswith("# repo:")


def test_search_hit_reconstructs_embed_text_exactly() -> None:
    """The reranker scores embed_text, so a lossy reconstruction would rerank
    against different text than we embedded."""
    chunk = _chunk()
    hit = to_search_hit(chunk.chunk_id, 0.9, to_payload(chunk, SPACE))
    assert hit.embed_text == chunk.embed_text


def test_search_hit_reconstruction_survives_a_disabled_header() -> None:
    chunk = _chunk(include_header=False)
    hit = to_search_hit(chunk.chunk_id, 0.5, to_payload(chunk, SPACE))
    assert hit.embed_text == chunk.source_text == chunk.embed_text


def test_round_trip_preserves_what_a_result_needs_to_show() -> None:
    chunk = _chunk()
    hit = to_search_hit(chunk.chunk_id, 0.42, to_payload(chunk, SPACE))
    assert hit.chunk_id == chunk.chunk_id
    assert hit.score == 0.42
    assert hit.location == "src/workspace_indexer/storage/qdrant_store.py:40-41"
    assert hit.symbol_path == "QdrantStore.upsert"
    assert hit.kind is FileKind.CODE
    assert hit.language == "python"
    assert hit.content_sha == chunk.meta.content_sha
    assert hit.token_count == chunk.meta.token_estimate


def test_space_slug_is_recorded_for_targeted_invalidation() -> None:
    payload = to_payload(_chunk(), SPACE)
    assert payload["space_slug"] == "voyageai_voyage-code-4_2048"
    assert payload["chunker"] == "code"
    assert payload["chunker_version"] == 1


def test_indexed_at_is_iso_utc() -> None:
    """Staleness questions are answered against this, so it has to be
    unambiguous about timezone."""
    stamp = to_payload(_chunk(), SPACE)["indexed_at"]
    assert stamp.endswith("+00:00")


def test_every_filterable_field_has_an_index() -> None:
    """A filter on an unindexed field makes Qdrant scan, which looks like a
    vector performance problem but is not."""
    filterable = set(SearchFilters.model_fields) - {"path_prefix"}
    payload_keys = set(to_payload(_chunk(), SPACE))
    for field in filterable:
        assert field in payload_keys, field
        assert field in INDEXED_FIELDS, field
    # path_prefix is served by the ancestors list rather than a field of its own.
    assert "ancestors" in INDEXED_FIELDS


def test_indexed_fields_all_exist_in_the_payload() -> None:
    payload_keys = set(to_payload(_chunk(), SPACE))
    assert set(INDEXED_FIELDS).issubset(payload_keys)


def test_missing_payload_keys_degrade_rather_than_raise() -> None:
    """A point written by an older version must still render as a hit."""
    hit = to_search_hit("id", 0.1, {"rel_path": "a.py"})
    assert hit.rel_path == "a.py"
    assert hit.start_line == 1
    assert hit.kind is FileKind.TEXT


def test_abs_path_is_a_string_not_a_path_object() -> None:
    """Qdrant serialises the payload as JSON; a Path would not survive."""
    assert isinstance(to_payload(_chunk(), SPACE)["abs_path"], str)
    assert isinstance(Path(to_payload(_chunk(), SPACE)["abs_path"]), Path)
