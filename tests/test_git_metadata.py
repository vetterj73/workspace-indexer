"""Git provenance reading."""

from __future__ import annotations

from pathlib import Path

from workspace_indexer.discovery import is_repo, read_repo_info


def test_repo_metadata_is_populated(workspace: Path) -> None:
    info = read_repo_info(workspace / "repo_one")
    assert info is not None
    assert info.name == "repo_one"
    assert info.branch == "main"
    assert info.head_sha is not None
    assert len(info.head_sha) == 40


def test_plain_folder_is_not_a_repo(workspace: Path) -> None:
    """Expected, not an error: a workspace holds plain folders next to repos
    and both get indexed."""
    assert not is_repo(workspace / "plain_folder")
    assert read_repo_info(workspace / "plain_folder") is None


def test_dirty_state_is_detected(workspace: Path) -> None:
    repo = workspace / "repo_one"
    assert read_repo_info(repo).is_dirty is False  # type: ignore[union-attr]
    (repo / "src" / "widget.py").write_text("# changed\n", encoding="utf-8")
    assert read_repo_info(repo).is_dirty is True  # type: ignore[union-attr]


def test_missing_directory_does_not_raise(tmp_path: Path) -> None:
    assert read_repo_info(tmp_path / "nope") is None
