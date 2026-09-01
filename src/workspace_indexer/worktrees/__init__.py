"""Seeing the index as one checkout sees it.

The index is built from main checkouts; a `git worktree add` checkout is a
second copy of the same repository that discovery deliberately skips. This
package is how a caller working in one is told which of its results that
checkout has since changed -- without the other worktrees, whose in-progress
work is nobody else's truth, being consulted at all.
"""

from __future__ import annotations

from workspace_indexer.worktrees.divergence_scanner import DivergenceScanner
from workspace_indexer.worktrees.worktree import Worktree
from workspace_indexer.worktrees.worktree_choice_error import WorktreeChoiceError
from workspace_indexer.worktrees.worktree_gate import NONE, WorktreeGate
from workspace_indexer.worktrees.worktree_lens import WorktreeLens
from workspace_indexer.worktrees.worktree_lister import WorktreeLister
from workspace_indexer.worktrees.worktree_registry import WorktreeRegistry

__all__ = [
    "NONE",
    "DivergenceScanner",
    "Worktree",
    "WorktreeChoiceError",
    "WorktreeGate",
    "WorktreeLens",
    "WorktreeLister",
    "WorktreeRegistry",
]
