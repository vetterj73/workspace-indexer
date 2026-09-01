"""Counting the reasons written into the code itself."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.conftest import git_init, write
from workspace_indexer.grounding import MarkerScanner

_IDENTITY = [
    "-c",
    "user.email=test@example.com",
    "-c",
    "user.name=Test",
    "-c",
    "commit.gpgsign=false",
]


def commit_all(repo: Path) -> None:
    subprocess.run(["git", *_IDENTITY, "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", *_IDENTITY, "commit", "-q", "-m", "add"],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    write(path / "seed.txt", "seed")
    git_init(path)
    return path


def test_not_a_repository_returns_none(tmp_path: Path) -> None:
    assert MarkerScanner().scan(tmp_path) is None


def test_no_markers_is_zero_not_none(repo: Path) -> None:
    """git grep exits 1 for "no matches", which is an answer.

    Treating that exit code as a failure would report a clean repository as
    unreadable, and the report would omit it rather than score it.
    """
    write(repo / "plain.py", "x = 1\n")
    commit_all(repo)

    assert MarkerScanner().scan(repo) == 0


def test_decision_markers_are_counted(repo: Path) -> None:
    write(
        repo / "store.py",
        "# WHY: cosine ignores magnitude here\n"
        "x = 1\n"
        "# DECISION: one collection per space\n"
        "y = 2\n",
    )
    commit_all(repo)

    assert MarkerScanner().scan(repo) == 2


def test_markers_across_several_files_are_summed(repo: Path) -> None:
    write(repo / "a.py", "# WHY: first\n")
    write(repo / "b.py", "# HACK: second, because the API lies\n")
    commit_all(repo)

    assert MarkerScanner().scan(repo) == 2


def test_todo_is_not_a_decision(repo: Path) -> None:
    """TODO names work outstanding, not a reason.

    Counting it would score a repository well for having a backlog in its
    comments, which is the opposite of what this measures.
    """
    write(repo / "a.py", "# TODO: fix this\n# FIXME: and this\n")
    commit_all(repo)

    assert MarkerScanner().scan(repo) == 0


def test_untracked_files_are_not_counted(repo: Path) -> None:
    """git grep sees the repository's own idea of what exists.

    That is the property being relied on: it is what keeps build output and
    vendored dependencies out of the count without an ignore list here.
    """
    write(repo / "tracked.py", "# WHY: counted\n")
    commit_all(repo)
    write(repo / "scratch.py", "# WHY: not counted\n")

    assert MarkerScanner().scan(repo) == 1
