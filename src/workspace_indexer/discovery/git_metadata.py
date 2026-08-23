"""Read git provenance for a root.

One subprocess batch per root, never per file. `git` is invoked rather than a
binding because it is guaranteed present on a dev box and correctly handles
worktrees, submodules, and detached HEADs without us reimplementing any of it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from workspace_indexer.models import RepoInfo
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.discovery.git")

_TIMEOUT = 10


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.debug("git.failed", args=args, error=str(exc))
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def is_repo(root: Path) -> bool:
    # --git-dir gives a truthy answer for worktrees and submodules too, where a
    # bare `(root / ".git").is_dir()` check would say no.
    return _git(root, "rev-parse", "--git-dir") is not None


def read_repo_info(root: Path) -> RepoInfo | None:
    """None when the directory is not a repository, which is expected — a
    workspace holds plain folders alongside repos and both get indexed."""
    if not is_repo(root):
        return None

    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    head = _git(root, "rev-parse", "HEAD")
    remote = _git(root, "config", "--get", "remote.origin.url")
    status = _git(root, "status", "--porcelain")

    return RepoInfo(
        name=root.name,
        remote_url=remote or None,
        branch=branch or None,
        head_sha=head or None,
        is_dirty=bool(status),
    )
