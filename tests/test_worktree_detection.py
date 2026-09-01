"""Telling a linked worktree apart from everything that looks like one.

Real repositories throughout, including a real submodule. The shorter test --
"`.git` is a file" -- is true of submodules as well, and would quietly drop
vendored code from the index. Nothing but git can settle it, so nothing but git
is used here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.conftest import git_init, write
from workspace_indexer.discovery import is_linked_worktree

_IDENTITY = ["-c", "user.email=t@example.com", "-c", "user.name=T", "-c", "commit.gpgsign=false"]


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *_IDENTITY, *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "main-checkout"
    path.mkdir()
    write(path / "a.txt", "x")
    git_init(path)
    return path


def test_a_main_checkout_is_not_a_worktree(repo: Path) -> None:
    assert is_linked_worktree(repo) is False


def test_a_linked_worktree_is_recognised(repo: Path, tmp_path: Path) -> None:
    """The case the exclusion exists for: the same repository at a second path."""
    linked = tmp_path / "feature-x"
    git(repo, "worktree", "add", "-q", str(linked), "-b", "feature-x")

    assert is_linked_worktree(linked) is True


def test_a_submodule_is_not_a_worktree(tmp_path: Path) -> None:
    """A submodule's `.git` is a file too.

    Excluding vendored submodule code would be a real loss, and the cheap test
    everyone reaches for first gets this wrong.
    """
    lib = tmp_path / "lib"
    lib.mkdir()
    write(lib / "a.txt", "x")
    git_init(lib)

    app = tmp_path / "app"
    app.mkdir()
    write(app / "b.txt", "y")
    git_init(app)
    subprocess.run(
        [
            "git",
            *_IDENTITY,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(lib),
            "vendor/lib",
        ],
        cwd=app,
        check=True,
        capture_output=True,
    )

    assert (app / "vendor" / "lib" / ".git").is_file()
    assert is_linked_worktree(app / "vendor" / "lib") is False


def test_a_plain_directory_is_not_a_worktree(tmp_path: Path) -> None:
    """Answered without running git at all -- this is the common case."""
    plain = tmp_path / "notes"
    plain.mkdir()

    assert is_linked_worktree(plain) is False


def test_a_directory_with_a_git_file_that_git_cannot_read_is_not_claimed(
    tmp_path: Path,
) -> None:
    """A `.git` file pointing nowhere is a broken checkout, not a worktree.

    Claiming it would silently drop the directory from the index on the
    strength of a file we failed to parse.
    """
    broken = tmp_path / "broken"
    broken.mkdir()
    write(broken / ".git", "gitdir: /nowhere/that/exists\n")

    assert is_linked_worktree(broken) is False
