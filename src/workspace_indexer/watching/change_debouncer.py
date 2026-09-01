"""Turning a burst of filesystem events into one reindex.

A single editor save is rarely one event. Vim writes a temp file, renames it
over the original and deletes a backup; a formatter rewrites forty files in two
seconds; `git checkout` touches a thousand. Reindexing per event would embed
the same file repeatedly and hammer the API for no gain.
"""

from __future__ import annotations

from pathlib import Path

from workspace_indexer.config import WorkspaceConfig
from workspace_indexer.discovery.git_metadata import is_linked_worktree
from workspace_indexer.discovery.ignore_matcher import IgnoreMatcher
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.watching.debounce")


class ChangeDebouncer:
    """Accumulates changed paths and reports which roots they belong to.

    Roots rather than files, deliberately. The indexer's decision ladder and
    its orphan pruning are both built around walking a root, and `--root` is
    already tested to scope pruning correctly. Reindexing the affected root
    reuses that path instead of inventing a second one that could disagree with
    it -- and an unchanged file costs a single stat(), so the walk is cheap.
    """

    def __init__(self, config: WorkspaceConfig, config_file: Path | None = None) -> None:
        self._config = config
        self._config_file = config_file.resolve() if config_file else None
        # One matcher per root: ignore rules are relative to a root, and a
        # nested .gitignore applies to its own subtree the way git does.
        self._ignore = {
            root.resolved_label: IgnoreMatcher(
                root.path.expanduser().resolve(),
                config.all_excludes,
                config.index.respect_gitignore,
            )
            for root in config.workspace.roots
        }
        self._pending: set[str] = set()
        self._config_changed = False
        # Memoised per directory: a save storm in one worktree asks the same
        # question thousands of times, and the answer cannot change while the
        # watcher runs without the worktree itself being added or removed.
        self._in_worktree: dict[Path, bool] = {}

    def add(self, path: Path) -> bool:
        """Record a change. Returns whether it was worth recording.

        Ignored paths are dropped here rather than at the watcher, so our own
        log and data writes cannot wake a reindex that writes more of them.
        """
        resolved = path.resolve()
        if self._config_file is not None and resolved == self._config_file:
            self._config_changed = True
            return True

        root = self._root_for(resolved)
        if root is None:
            return False

        label, rel = root
        matcher = self._ignore.get(label)
        if matcher is not None and matcher.reason(resolved) is not None:
            # Dropped here rather than at the watcher, so our own log and data
            # writes cannot wake a reindex that writes more of them.
            log.debug("watch.ignored", path=rel, root=label)
            return False

        if self._inside_a_worktree(resolved.parent, label):
            # The walk would prune this anyway, so reindexing the root on its
            # account is pure cost -- and it is cost paid per save, by agents
            # whose whole working pattern is saving into a worktree. Without
            # this, one agent editing in a worktree makes the watcher re-walk
            # the entire root, repeatedly, to discover nothing.
            log.debug("watch.worktree_ignored", path=rel, root=label)
            return False

        self._pending.add(label)
        return True

    def _inside_a_worktree(self, directory: Path, label: str) -> bool:
        """Is `directory` at or below the root of a linked worktree?

        Walks up rather than testing only the directory itself: the change is a
        file somewhere inside the worktree, and only its top carries the `.git`
        that identifies it. Stops at the configured root, so the search is
        bounded by the workspace rather than by the filesystem.
        """
        base = self._base_of(label)
        seen: list[Path] = []
        current = directory
        while True:
            if current in self._in_worktree:
                answer = self._in_worktree[current]
                break
            seen.append(current)
            if base is not None and current == base:
                answer = False
                break
            if is_linked_worktree(current):
                answer = True
                break
            if current.parent == current or (base is not None and base not in current.parents):
                answer = False
                break
            current = current.parent
        for path in seen:
            self._in_worktree[path] = answer
        return answer

    def _base_of(self, label: str) -> Path | None:
        for root in self._config.workspace.roots:
            if root.resolved_label == label:
                return root.path.expanduser().resolve()
        return None

    def drain(self) -> tuple[set[str], bool]:
        """Everything accumulated since the last drain, and whether the config
        file was among it. Clears the buffer."""
        roots, config_changed = set(self._pending), self._config_changed
        self._pending.clear()
        self._config_changed = False
        return roots, config_changed

    @property
    def pending(self) -> set[str]:
        return set(self._pending)

    def _root_for(self, path: Path) -> tuple[str, str] | None:
        """Which configured root contains `path`, and the path relative to it.

        Longest match wins, so a root nested inside another is attributed to
        the more specific one.
        """
        best: tuple[int, str, str] | None = None
        for root in self._config.workspace.roots:
            base = root.path.expanduser().resolve()
            if path == base or base in path.parents:
                depth = len(base.parts)
                if best is None or depth > best[0]:
                    best = (depth, root.resolved_label, path.relative_to(base).as_posix())
        return (best[1], best[2]) if best else None
