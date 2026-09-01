"""The worktrees of every repository this index covers."""

from __future__ import annotations

from pathlib import Path

from workspace_indexer.discovery import repo_root
from workspace_indexer.obs.logging import get_logger
from workspace_indexer.state import Manifest
from workspace_indexer.worktrees.worktree import Worktree
from workspace_indexer.worktrees.worktree_lister import WorktreeLister

log = get_logger("workspace_indexer.worktrees.registry")


class WorktreeRegistry:
    """Two questions with opposite caching needs, which is why they are separate.

    *Which repositories are indexed* changes only when the index is rebuilt, so
    it is resolved once per process -- it costs a git call per distinct
    directory and there is no point paying that per search.

    *What worktrees those repositories have* changes constantly, and by the
    hand of the very agent asking. It is never cached: an agent that creates a
    worktree and then searches must not be told there are none.
    """

    def __init__(self, manifest: Manifest, lister: WorktreeLister | None = None) -> None:
        self._manifest = manifest
        self._lister = lister or WorktreeLister()
        self._repositories: list[Path] | None = None

    def repositories(self) -> list[Path]:
        """Main checkouts covered by this index, resolved once."""
        if self._repositories is None:
            seen: dict[str, Path | None] = {}
            found: set[Path] = set()
            for abs_path, _, _ in self._manifest.indexed_documents():
                directory = str(Path(abs_path).parent)
                if directory not in seen:
                    seen[directory] = repo_root(Path(directory))
                repo = seen[directory]
                if repo is not None:
                    found.add(repo)
            self._repositories = sorted(found)
            log.debug("worktrees.repositories", count=len(self._repositories))
        return self._repositories

    def all_worktrees(self) -> list[Worktree]:
        """Every linked worktree of every indexed repository, read fresh."""
        found: list[Worktree] = []
        for repo in self.repositories():
            found.extend(self._lister.worktrees_of(repo))
        return found

    def resolve(self, given: str) -> Worktree | None:
        """Match `given` against a worktree by path or by directory name.

        Both, because an agent that created the worktree knows its path while
        one told about it from an error message has only the name, and refusing
        either would send the caller back for a round trip it cannot complete.
        """
        candidate = Path(given).expanduser()
        for worktree in self.all_worktrees():
            if worktree.name == given:
                return worktree
            if candidate.is_absolute() and worktree.path.resolve() == candidate.resolve():
                return worktree
        return None
