"""Flagging results whose source moved on.

Real files, because the whole question is what is on disk right now.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from workspace_indexer.models import SearchHit
from workspace_indexer.search.staleness import mark_stale

BODY = "def login():\n    return check_token()"


def _hit(path: Path, source_text: str = BODY, chunk_id: str = "id-1") -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        score=1.0,
        rel_path=path.name,
        abs_path=str(path),
        root_label="root",
        source_text=source_text,
        content_sha="sha",
    )


def test_unchanged_file_is_not_stale(tmp_path: Path) -> None:
    path = tmp_path / "a.py"
    path.write_text(f"# header\n{BODY}\n", encoding="utf-8")
    assert mark_stale([_hit(path)])[0].stale is False


def test_edited_chunk_is_stale(tmp_path: Path) -> None:
    """Showing the current text at the old line numbers would display
    something that never matched the query."""
    path = tmp_path / "a.py"
    path.write_text("def login():\n    return check_session()\n", encoding="utf-8")
    assert mark_stale([_hit(path)])[0].stale is True


def test_change_elsewhere_in_the_file_does_not_make_this_hit_stale(tmp_path: Path) -> None:
    """A file hash would flag this; the hit itself is still perfectly accurate."""
    path = tmp_path / "a.py"
    path.write_text(f"# a brand new comment\n{BODY}\n\ndef other(): pass\n", encoding="utf-8")
    assert mark_stale([_hit(path)])[0].stale is False


def test_deleted_file_is_stale(tmp_path: Path) -> None:
    hit = _hit(tmp_path / "gone.py")
    assert mark_stale([hit])[0].stale is True


def test_hits_without_a_path_are_left_alone(tmp_path: Path) -> None:
    """A point written before abs_path was stored should still render."""
    hit = SearchHit(chunk_id="id", score=1.0, rel_path="a.py", root_label="r", source_text=BODY)
    assert mark_stale([hit])[0].stale is False


def test_one_read_per_file_not_per_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Several chunks of one file routinely appear in the same result set."""
    path = tmp_path / "a.py"
    path.write_text(f"{BODY}\n\ndef other():\n    pass\n", encoding="utf-8")

    reads: list[str] = []
    original = Path.read_text

    def counting(self: Path, *args: object, **kwargs: object) -> str:
        reads.append(str(self))
        return original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", counting)
    hits = [
        _hit(path, BODY, "id-1"),
        _hit(path, "def other():\n    pass", "id-2"),
    ]
    marked = mark_stale(hits)
    assert [h.stale for h in marked] == [False, False]
    assert reads.count(str(path)) == 1


def test_empty_input(tmp_path: Path) -> None:
    assert mark_stale([]) == []
