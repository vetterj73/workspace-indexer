"""Refusing to guess which checkout a query is about.

Real repositories and real worktrees: the whole subject is what git reports,
and a stubbed lister would only confirm the porcelain format we assumed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.conftest import git_init, write
from workspace_indexer.models import FileKind, SearchHit
from workspace_indexer.worktrees import (
    DivergenceScanner,
    Worktree,
    WorktreeChoiceError,
    WorktreeGate,
    WorktreeLens,
    WorktreeLister,
    WorktreeRegistry,
)

_IDENTITY = ["-c", "user.email=t@example.com", "-c", "user.name=T", "-c", "commit.gpgsign=false"]


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *_IDENTITY, *args], cwd=repo, check=True, capture_output=True)


class FixedRepositories(WorktreeRegistry):
    """A registry over known checkouts, so no manifest is needed.

    Only the repository list is replaced. Worktree discovery stays real,
    because that is the behaviour under test.
    """

    def __init__(self, repositories: list[Path]) -> None:  # pyright: ignore[reportMissingSuperCall]
        self._fixed = repositories
        self._lister = WorktreeLister()

    def repositories(self) -> list[Path]:
        return self._fixed


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "Product"
    path.mkdir()
    write(path / "auth.py", "def authenticate():\n    return True\n")
    git_init(path)
    return path


def add_worktree(repo: Path, path: Path, branch: str) -> None:
    git(repo, "worktree", "add", "-q", str(path), "-b", branch)


def gate_for(repo: Path) -> WorktreeGate:
    return WorktreeGate(FixedRepositories([repo]))


def test_a_workspace_with_no_worktrees_never_asks(repo: Path) -> None:
    """Nobody pays for a feature they do not use.

    Asking here would put a round trip in front of every search in every
    workspace that has never run `git worktree add`.
    """
    assert gate_for(repo).scope(None) is None


def test_an_unanswered_choice_is_refused_once_a_worktree_exists(repo: Path, tmp_path: Path) -> None:
    add_worktree(repo, tmp_path / "feature-a", "feature-a")

    with pytest.raises(WorktreeChoiceError) as caught:
        gate_for(repo).scope(None)

    message = str(caught.value)
    assert "feature-a" in message
    assert '"none"' in message  # the escape hatch has to be in the refusal


def test_none_means_the_main_checkout(repo: Path, tmp_path: Path) -> None:
    """A reader who is not in a worktree must be able to say so.

    Without this the guard is unsatisfiable for the commonest caller: an agent
    reading code it is not editing has no worktree to name.
    """
    add_worktree(repo, tmp_path / "feature-a", "feature-a")

    assert gate_for(repo).scope("none") is None
    assert gate_for(repo).scope("NONE") is None


def test_a_worktree_is_found_by_name(repo: Path, tmp_path: Path) -> None:
    add_worktree(repo, tmp_path / "feature-a", "feature-a")

    scoped = gate_for(repo).scope("feature-a")

    assert scoped is not None
    assert scoped.path == tmp_path / "feature-a"
    assert scoped.main_checkout == repo


def test_a_worktree_is_found_by_absolute_path(repo: Path, tmp_path: Path) -> None:
    """An agent that created the worktree knows its path, not its label."""
    add_worktree(repo, tmp_path / "feature-a", "feature-a")

    scoped = gate_for(repo).scope(str(tmp_path / "feature-a"))

    assert scoped is not None
    assert scoped.branch == "feature-a"


def test_an_unknown_worktree_names_the_real_ones(repo: Path, tmp_path: Path) -> None:
    add_worktree(repo, tmp_path / "feature-a", "feature-a")

    with pytest.raises(WorktreeChoiceError) as caught:
        gate_for(repo).scope("feature-b")

    assert "feature-a" in str(caught.value)


def test_naming_a_worktree_where_none_exist_is_still_refused(repo: Path) -> None:
    """Silently ignoring it would let the caller believe it was scoped."""
    with pytest.raises(WorktreeChoiceError):
        gate_for(repo).scope("feature-a")


def test_the_main_checkout_is_not_offered_as_a_worktree(repo: Path, tmp_path: Path) -> None:
    """`git worktree list` reports it first; it is the thing others diverge from."""
    add_worktree(repo, tmp_path / "feature-a", "feature-a")

    names = [w.name for w in FixedRepositories([repo]).all_worktrees()]

    assert names == ["feature-a"]


def test_a_worktree_created_mid_session_is_seen(repo: Path, tmp_path: Path) -> None:
    """The list must not be cached.

    The working pattern this exists for is an agent creating a worktree and
    then working in it. A cached list would answer "no worktrees" for exactly
    the session that needed the answer.
    """
    gate = gate_for(repo)
    assert gate.scope(None) is None

    add_worktree(repo, tmp_path / "feature-late", "feature-late")

    with pytest.raises(WorktreeChoiceError):
        gate.scope(None)


def hit(abs_path: Path, rel_path: str) -> SearchHit:
    return SearchHit(
        chunk_id="c1",
        score=1.0,
        rel_path=rel_path,
        abs_path=str(abs_path),
        root_label="src",
        kind=FileKind.CODE,
        source_text="def authenticate():\n    return True\n",
    )


def test_a_hit_the_worktree_changed_is_marked_and_repointed(repo: Path, tmp_path: Path) -> None:
    """The path is rewritten, not merely annotated.

    A path is the one thing an agent acts on without thinking. Handing back the
    main checkout's path while saying "this differs in your worktree" invites
    the exact mistake the flag exists to prevent.
    """
    worktree_path = tmp_path / "feature-a"
    add_worktree(repo, worktree_path, "feature-a")
    write(worktree_path / "auth.py", "def authenticate():\n    return False\n")

    worktree = Worktree(path=worktree_path, main_checkout=repo)
    marked = WorktreeLens().apply([hit(repo / "auth.py", "Product/auth.py")], worktree)

    assert marked[0].stale is True
    assert marked[0].abs_path == str(worktree_path / "auth.py")


def test_a_hit_the_worktree_has_not_touched_is_repointed_but_not_marked(
    repo: Path, tmp_path: Path
) -> None:
    """Repointed regardless, so the agent always reads its own checkout."""
    worktree_path = tmp_path / "feature-a"
    add_worktree(repo, worktree_path, "feature-a")

    worktree = Worktree(path=worktree_path, main_checkout=repo)
    marked = WorktreeLens().apply([hit(repo / "auth.py", "Product/auth.py")], worktree)

    assert marked[0].stale is False
    assert marked[0].abs_path == str(worktree_path / "auth.py")


def test_a_committed_change_still_counts_as_divergence(repo: Path, tmp_path: Path) -> None:
    """Measured against the main checkout's HEAD, not the worktree's.

    Against the worktree's own HEAD a change would stop being visible the
    moment the agent committed it -- which is precisely when it stops being
    visible any other way.
    """
    worktree_path = tmp_path / "feature-a"
    add_worktree(repo, worktree_path, "feature-a")
    write(worktree_path / "auth.py", "def authenticate():\n    return False\n")
    git(worktree_path, "add", "-A")
    git(worktree_path, "commit", "-q", "-m", "change auth")

    worktree = Worktree(path=worktree_path, main_checkout=repo)
    diverged = DivergenceScanner().diverged(worktree)

    assert diverged == {"auth.py"}


def test_a_file_created_in_the_worktree_is_reported_as_diverged(repo: Path, tmp_path: Path) -> None:
    """It has no hit to mark, but it is still part of the divergence set."""
    worktree_path = tmp_path / "feature-a"
    add_worktree(repo, worktree_path, "feature-a")
    write(worktree_path / "new_thing.py", "x = 1\n")

    diverged = DivergenceScanner().diverged(Worktree(path=worktree_path, main_checkout=repo))

    assert diverged == {"new_thing.py"}


def test_a_hit_from_another_repository_is_left_alone(repo: Path, tmp_path: Path) -> None:
    """A workspace holds several repositories.

    Rewriting a hit from one of the others would point at a path that does not
    exist inside this worktree.
    """
    other = tmp_path / "Other"
    other.mkdir()
    write(other / "thing.py", "x = 1\n")
    git_init(other)

    worktree_path = tmp_path / "feature-a"
    add_worktree(repo, worktree_path, "feature-a")

    worktree = Worktree(path=worktree_path, main_checkout=repo)
    marked = WorktreeLens().apply([hit(other / "thing.py", "Other/thing.py")], worktree)

    assert marked[0].abs_path == str(other / "thing.py")
    assert marked[0].stale is False
