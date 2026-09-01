"""A worktree beside its repository must not become a second copy of it.

The failure this prevents is silent. Two checkouts of one repository produce
two chunks of every file, both ranking; and route resolution, which resolves an
endpoint to a *file* and refuses to guess when several match, then resolves to
nothing at all -- so cross-repository edges disappear rather than duplicate.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import git_init, write
from workspace_indexer.config import WorkspaceConfig
from workspace_indexer.discovery import SkipReason, Walker

_IDENTITY = ["-c", "user.email=t@example.com", "-c", "user.name=T", "-c", "commit.gpgsign=false"]


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *_IDENTITY, *args], cwd=repo, check=True, capture_output=True)


def config(**overrides: Any) -> WorkspaceConfig:
    payload: dict[str, Any] = {"workspace": {"name": "test", "roots": []}}
    payload.update(overrides)
    return WorkspaceConfig.model_validate(payload)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A root holding one repository, with a worktree added beside it."""
    root = tmp_path / "Root"
    repo = root / "Product"
    repo.mkdir(parents=True)
    write(repo / "auth.py", "def authenticate():\n    return True\n")
    git_init(repo)
    git(repo, "worktree", "add", "-q", str(root / "Product-feature-a"), "-b", "feature-a")
    return root


def walk(root: Path, *, recurse: bool = True) -> tuple[list[str], Walker]:
    cfg = config(
        workspace={
            "name": "test",
            "roots": [{"path": str(root), "recurse_into_children": recurse}],
        }
    )
    walker = Walker(cfg)
    return [c.rel_path for c in walker.walk()], walker


def test_a_worktree_beside_its_repository_is_not_indexed(workspace: Path) -> None:
    paths, walker = walk(workspace)

    assert "Product/auth.py" in paths
    assert not any(p.startswith("Product-feature-a/") for p in paths)
    assert walker.pruned_dirs[SkipReason.WORKTREE.value] == 1


def test_the_main_checkout_is_still_indexed_in_full(workspace: Path) -> None:
    """The exclusion must not take the repository with it."""
    paths, _ = walk(workspace)

    assert [p for p in paths if p.startswith("Product/")] == ["Product/auth.py"]


def test_a_worktree_configured_as_its_own_root_is_indexed(workspace: Path) -> None:
    """Explicit beats inferred.

    A root is pushed straight onto the walk stack and never passes the
    directory checks, so naming a worktree deliberately still indexes it. That
    is the escape hatch for a developer who works *in* a worktree.
    """
    paths, walker = walk(workspace / "Product-feature-a", recurse=False)

    assert "auth.py" in paths
    assert walker.pruned_dirs[SkipReason.WORKTREE.value] == 0


def test_a_worktree_nested_deeper_than_a_child_is_still_caught(tmp_path: Path) -> None:
    """Worktrees do not have to sit beside their repository.

    The check runs on every directory descended into, not only on the children
    of a root, because `worktrees/feature-a` is just as common a layout.
    """
    root = tmp_path / "Root"
    repo = root / "Product"
    repo.mkdir(parents=True)
    write(repo / "a.py", "x = 1\n")
    git_init(repo)
    (root / "worktrees").mkdir()
    git(repo, "worktree", "add", "-q", str(root / "worktrees" / "feature-a"), "-b", "feature-a")

    paths, walker = walk(root)

    assert "Product/a.py" in paths
    assert not any("feature-a" in p for p in paths)
    assert walker.pruned_dirs[SkipReason.WORKTREE.value] == 1


def test_a_submodule_beside_a_repository_is_still_indexed(tmp_path: Path) -> None:
    """Vendored submodule code is part of the checkout and must survive.

    The cheap test -- "`.git` is a file" -- is true here too, so this is the
    case that would break if the detection were simplified.
    """
    root = tmp_path / "Root"
    lib = tmp_path / "lib"
    lib.mkdir()
    write(lib / "helper.py", "def help_me():\n    pass\n")
    git_init(lib)

    app = root / "App"
    app.mkdir(parents=True)
    write(app / "main.py", "x = 1\n")
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

    paths, walker = walk(root)

    assert "App/vendor/lib/helper.py" in paths
    assert walker.pruned_dirs[SkipReason.WORKTREE.value] == 0
