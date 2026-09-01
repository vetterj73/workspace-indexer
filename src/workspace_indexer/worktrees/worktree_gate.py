"""Deciding which checkout a query is about, and refusing to guess."""

from __future__ import annotations

from workspace_indexer.models import SearchHit
from workspace_indexer.worktrees.worktree import Worktree
from workspace_indexer.worktrees.worktree_choice_error import WorktreeChoiceError
from workspace_indexer.worktrees.worktree_lens import WorktreeLens
from workspace_indexer.worktrees.worktree_registry import WorktreeRegistry

# What a caller says to mean "the main checkout". Spelled rather than left as
# absence so that "I am not in a worktree" is a statement the caller makes,
# distinguishable from having forgotten to say anything.
NONE = "none"


class WorktreeGate:
    """Asks only when the answer could change the result.

    A workspace with no worktrees never sees this: no error, no parameter, no
    cost. The moment one exists the question becomes unanswerable from the
    index -- results depend on a checkout the server cannot see -- and asking
    once beats either of the guesses available.
    """

    def __init__(self, registry: WorktreeRegistry, lens: WorktreeLens | None = None) -> None:
        self._registry = registry
        self._lens = lens or WorktreeLens()

    def scope(self, given: str | None) -> Worktree | None:
        """The worktree to report through, or None for the main checkout.

        Raises WorktreeChoiceError when worktrees exist and the caller has not
        chosen, or chose one that is not there.
        """
        wanted = given.strip() if given else None
        if wanted and wanted.lower() == NONE:
            return None

        available = self._registry.all_worktrees()
        if not available:
            if wanted:
                # Naming one that cannot exist is still worth saying out loud:
                # silently ignoring it would let a caller believe it was scoped
                # when it was not.
                raise WorktreeChoiceError(wanted, [])
            return None

        names = [w.name for w in available]
        if wanted is None:
            raise WorktreeChoiceError(None, names)
        found = self._registry.resolve(wanted)
        if found is None:
            raise WorktreeChoiceError(wanted, names)
        return found

    def apply(self, hits: list[SearchHit], worktree: Worktree | None) -> list[SearchHit]:
        return self._lens.apply(hits, worktree) if worktree is not None else hits
