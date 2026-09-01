"""Reporting hits as one worktree sees them."""

from __future__ import annotations

from pathlib import Path

from workspace_indexer.models import SearchHit
from workspace_indexer.obs.logging import get_logger
from workspace_indexer.worktrees.divergence_scanner import DivergenceScanner
from workspace_indexer.worktrees.worktree import Worktree

log = get_logger("workspace_indexer.worktrees.lens")


class WorktreeLens:
    """Marks the hits a worktree has changed, and points them at its copy.

    Rewriting `abs_path` rather than adding a second field is the decision
    worth noticing. A hit's path is the one thing an agent acts on without
    thinking, and handing back the main checkout's path while saying "this
    differs in your worktree" invites exactly the mistake the flag exists to
    prevent -- reading, or worse patching, the wrong copy. The agent should
    always be able to open the path it was given.

    The index still holds the main checkout's *text*, and that is a real limit
    rather than an oversight: this can say a file differs, never what it now
    says. A file created in the worktree has no hit to mark at all.
    """

    def __init__(self, scanner: DivergenceScanner | None = None) -> None:
        self._scanner = scanner or DivergenceScanner()

    def apply(self, hits: list[SearchHit], worktree: Worktree) -> list[SearchHit]:
        diverged = self._scanner.diverged(worktree)
        if diverged is None:
            # Could not be read. Marking nothing beats marking everything: a
            # flag that fires on a git failure is a flag nobody trusts.
            log.warning(
                "worktrees.divergence_unknown",
                worktree=worktree.name,
                detail="could not compare this worktree against the indexed checkout; "
                "hits are reported as the index has them",
            )
            return hits

        seen: list[SearchHit] = []
        for hit in hits:
            copy = worktree.copy_of(Path(hit.abs_path)) if hit.abs_path else None
            if copy is None:
                # A hit from a different repository. It has no copy here, and
                # rewriting it would point at a path that does not exist.
                seen.append(hit)
                continue
            within = str(copy.relative_to(worktree.path).as_posix())
            if within in diverged:
                seen.append(hit.model_copy(update={"abs_path": str(copy), "stale": True}))
            else:
                seen.append(hit.model_copy(update={"abs_path": str(copy)}))

        changed = sum(1 for h in seen if h.stale)
        if changed:
            log.info("worktrees.diverged_hits", worktree=worktree.name, count=changed)
        return seen
