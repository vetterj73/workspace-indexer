"""Which files differ between a worktree and the checkout the index was built from."""

from __future__ import annotations

import subprocess
from pathlib import Path

from workspace_indexer.obs.logging import get_logger
from workspace_indexer.worktrees.worktree import Worktree

log = get_logger("workspace_indexer.worktrees.divergence")

_TIMEOUT = 30


class DivergenceScanner:
    """Asks git, and reads no files.

    Cheaper than the staleness check it sits beside: that one re-reads every
    hit off disk to compare text, this one gets the whole answer as a path list
    in two subprocesses. It is also the only way to get this answer at all --
    the indexed text came from a different checkout, so comparing a hit against
    the file it was built from would always say "unchanged".

    Divergence is measured against the **main checkout's HEAD** rather than the
    worktree's own, because HEAD is what the index reflects. Measuring against
    the worktree's HEAD would call a committed change "unchanged" the moment
    the agent committed it, which is precisely when it stops being visible any
    other way.
    """

    def diverged(self, worktree: Worktree) -> set[str] | None:
        """Paths, relative to the repository, that differ in `worktree`.

        None when git could not answer -- distinct from an empty set, which is
        a real finding meaning the worktree matches what was indexed.
        """
        raw_head = self._run(worktree.main_checkout, "rev-parse", "HEAD")
        # Stripped, because git prints a trailing newline and the next call
        # takes this as an argument: an unstripped sha fails every comparison
        # and the whole scan reports "cannot tell", which reads as "nothing
        # diverged" to anyone not looking at the log.
        head = raw_head.strip() if raw_head else None
        if not head:
            return None

        changed = self._run(worktree.path, "diff", "--name-only", head)
        untracked = self._run(worktree.path, "ls-files", "--others", "--exclude-standard")
        if changed is None or untracked is None:
            return None

        paths = {line.strip() for line in changed.splitlines() if line.strip()}
        paths |= {line.strip() for line in untracked.splitlines() if line.strip()}
        log.debug("worktrees.diverged", worktree=worktree.name, files=len(paths))
        return paths

    def _run(self, cwd: Path, *args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(cwd), *args],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.debug("worktrees.git_failed", args=args, error=str(exc))
            return None
        if result.returncode != 0:
            return None
        return result.stdout
