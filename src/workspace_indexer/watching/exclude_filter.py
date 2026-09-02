"""Keeping the watcher's event stream to the files the index would take."""

from __future__ import annotations

from pathlib import Path

from watchfiles import Change, DefaultFilter

from workspace_indexer.config import WorkspaceConfig
from workspace_indexer.discovery.ignore_matcher import IgnoreMatcher


class ExcludeFilter(DefaultFilter):
    """`index.exclude` applied to filesystem events.

    Subclasses `DefaultFilter` rather than replacing it: watchfiles already
    drops editor scratch files, `.pyc`, `~` backups and `.DS_Store`, and those
    rules are worth keeping. A bare callable would silently discard them and
    wake a reindex on every `.swp` vim writes.

    **This does not stop the watcher descending into an excluded directory.**
    watchfiles applies a filter to changes the Rust watcher has already
    produced -- see `_prep_changes` -- so recursion happens first and filtering
    second. What this buys is the work after that point: no debounce entry, no
    reindex, no walk of a root because something under `node_modules` moved.
    Whether the OS-level watch can even read a path is a separate question the
    watcher has to survive on its own.
    """

    def __init__(self, config: WorkspaceConfig) -> None:
        super().__init__()
        self._config = config
        # One matcher per root: ignore rules are relative to a root, and a
        # nested .gitignore applies to its own subtree the way git does. Built
        # once, because this is called for every event in a burst.
        self._matchers = {
            root.resolved_label: IgnoreMatcher(
                root.path.expanduser().resolve(),
                config.all_excludes,
                config.index.respect_gitignore,
            )
            for root in config.workspace.roots
        }

    def __call__(self, change: Change, path: str) -> bool:
        if not super().__call__(change, path):
            return False

        resolved = Path(path)
        root = self._config.root_containing(resolved)
        if root is None:
            # Outside every root. The config file itself lives here -- it is
            # watched deliberately and usually sits outside the tree it
            # describes -- so this is kept rather than dropped.
            return True

        matcher = self._matchers.get(root.resolved_label)
        return matcher is None or matcher.reason(resolved) is None
