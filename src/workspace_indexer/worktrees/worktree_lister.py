"""Which worktrees exist for the repositories this index covers."""

from __future__ import annotations

import subprocess
from pathlib import Path

from workspace_indexer.obs.logging import get_logger
from workspace_indexer.worktrees.worktree import Worktree

log = get_logger("workspace_indexer.worktrees.lister")

_TIMEOUT = 15


class WorktreeLister:
    """`git worktree list` per repository, asked fresh every time.

    Not cached, deliberately, and this is the one thing here that must not be.
    The working pattern this exists for is an agent creating a worktree and
    then working in it, inside a single session -- a cached list would answer
    "no worktrees" for the whole of the session that needed the answer most.

    One subprocess per repository, and a workspace holds a handful.
    """

    def worktrees_of(self, main_checkout: Path) -> list[Worktree]:
        """Linked worktrees of `main_checkout`, never the checkout itself."""
        raw = self._list(main_checkout)
        if raw is None:
            return []

        found: list[Worktree] = []
        path: Path | None = None
        branch: str | None = None
        for line in [*raw.splitlines(), ""]:
            if line.startswith("worktree "):
                path = Path(line.removeprefix("worktree ").strip())
            elif line.startswith("branch "):
                branch = line.removeprefix("branch ").strip().removeprefix("refs/heads/")
            elif not line.strip():
                # A blank line closes a record. The first record is always the
                # main checkout, which is not a worktree in the sense that
                # matters here: it is the thing the others diverge from.
                if path is not None and path.resolve() != main_checkout.resolve():
                    found.append(Worktree(path=path, main_checkout=main_checkout, branch=branch))
                path, branch = None, None
        return found

    def _list(self, main_checkout: Path) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(main_checkout), "worktree", "list", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.debug("worktrees.list_failed", repo=str(main_checkout), error=str(exc))
            return None
        if result.returncode != 0:
            return None
        return result.stdout
