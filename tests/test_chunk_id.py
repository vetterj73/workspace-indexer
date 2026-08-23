"""Chunk identity.

The whole incremental-reindex cost story rests on this function's stability:
edit one function and exactly one chunk id must change. Untested, that is a
hope rather than a guarantee.
"""

from __future__ import annotations

import uuid

from dirindex.models import compute_chunk_id


def chunk_id(
    root_label: str = "repo_one",
    rel_path: str = "src/widget.py",
    symbol_path: str | None = "Widget.render",
    chunk_index: int = 0,
    content_sha: str = "a" * 64,
) -> str:
    """Keyword-only variation on one baseline, so each test names just the
    field it is varying. A dict-splat baseline would widen every argument to
    `str | int` and hide real type errors."""
    return compute_chunk_id(root_label, rel_path, symbol_path, chunk_index, content_sha)


def test_is_a_valid_uuid() -> None:
    """Qdrant rejects point ids that are neither an unsigned int nor a UUID."""
    assert uuid.UUID(chunk_id()).version == 5


def test_deterministic_across_calls() -> None:
    assert chunk_id() == chunk_id()


def test_content_change_changes_the_id() -> None:
    assert chunk_id(content_sha="b" * 64) != chunk_id()


def test_each_identifying_field_is_load_bearing() -> None:
    """No two distinct chunks may collide on any single field."""
    variants = {
        chunk_id(),
        chunk_id(root_label="repo_two"),
        chunk_id(rel_path="src/other.py"),
        chunk_id(symbol_path="Widget.__init__"),
        chunk_id(chunk_index=1),
        chunk_id(content_sha="c" * 64),
    }
    assert len(variants) == 6


def test_field_boundaries_cannot_be_forged() -> None:
    """Concatenating fields without a separator would let one chunk's path
    bleed into another's symbol and produce a collision."""
    assert chunk_id(rel_path="src/a", symbol_path="b") != chunk_id(
        rel_path="src/ab", symbol_path=""
    )


def test_missing_symbol_path_is_stable() -> None:
    assert chunk_id(symbol_path=None) == chunk_id(symbol_path=None)
    # None and "" describe the same thing (no enclosing symbol) and must agree,
    # or a markdown chunk's id would flip depending on which the chunker set.
    assert chunk_id(symbol_path=None) == chunk_id(symbol_path="")
