"""Count the places where the code itself states a reason."""

from __future__ import annotations

import subprocess
from pathlib import Path

from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.grounding.markers")

_TIMEOUT = 30

# Deliberately narrow. TODO and FIXME name work outstanding, not a decision
# taken, and including them would score a repository well for having a backlog
# in its comments. HACK is admitted because it almost always carries the reason
# the ugly thing is there, which is exactly the content this looks for.
_MARKERS = r"\b(WHY|DECISION|RATIONALE|DESIGN NOTE|HACK|ADR-[0-9]+)\b:"


class MarkerScanner:
    """`git grep` over tracked files.

    Chosen over walking the tree because it is one process for a whole
    repository, it already skips binaries and honours the repository's own idea
    of what is tracked, and it never sees `node_modules` or build output --
    which is the difference between a fast answer and a slow wrong one.
    """

    def scan(self, repo: Path) -> int | None:
        """Matching lines across the repository, or None if it cannot be read."""
        try:
            result = subprocess.run(
                ["git", "-C", str(repo), "grep", "-I", "-c", "-E", _MARKERS],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.debug("grounding.grep_failed", repo=str(repo), error=str(exc))
            return None

        # git grep exits 1 to say "no matches", which is an answer (zero), not
        # a failure. Anything above that is a real error and must not be
        # reported as an absence of markers.
        if result.returncode > 1:
            return None
        if result.returncode == 1:
            return 0

        total = 0
        for line in result.stdout.splitlines():
            _, _, count = line.rpartition(":")
            if count.isdigit():
                total += int(count)
        return total
