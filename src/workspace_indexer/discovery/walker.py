"""Filesystem traversal.

Uses os.scandir rather than Path.rglob because scandir returns the stat data
the kernel already fetched while listing the directory. rglob re-stats every
path, which doubles the syscall count on a tree this size.

The walker never opens a file. That is deliberate: the manifest's fast path
decides whether a file needs reading based on mtime and size alone, and reading
here would throw that saving away.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from workspace_indexer.config import RootConfig, WorkspaceConfig
from workspace_indexer.discovery.classify import classify, is_lockfile
from workspace_indexer.discovery.file_candidate import FileCandidate
from workspace_indexer.discovery.git_metadata import read_repo_info
from workspace_indexer.discovery.ignore_matcher import IgnoreMatcher
from workspace_indexer.discovery.skip_reason import SkipReason
from workspace_indexer.models import RepoInfo
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.discovery.walker")

# Always skipped, whatever the config says. `.git` is enormous, entirely
# machine-generated, and re-walking it would dwarf the real work.
_ALWAYS_SKIP_DIRS = frozenset({".git"})


class Walker:
    def __init__(self, config: WorkspaceConfig) -> None:
        self._config = config
        # Files dropped, keyed by reason.
        self.skips: Counter[str] = Counter()
        # Directories not descended into. Tracked separately because one entry
        # here can stand for thousands of files, so folding it into `skips`
        # would make the file tally meaningless.
        self.pruned_dirs: Counter[str] = Counter()

    def walk(self, only_root: str | None = None) -> Iterator[FileCandidate]:
        for root in self._config.workspace.roots:
            if only_root and root.resolved_label != only_root:
                continue
            if not root.path.is_dir():
                log.warning("root.missing", path=str(root.path), label=root.resolved_label)
                continue
            yield from self._walk_root(root)

    def _walk_root(self, root: RootConfig) -> Iterator[FileCandidate]:
        matcher = IgnoreMatcher(
            root.path,
            self._config.all_excludes,
            self._config.index.respect_gitignore,
        )
        # Repo metadata is read once per unit directory, never per file.
        repo_cache: dict[str, RepoInfo | None] = {}
        follow = self._config.index.follow_symlinks
        max_bytes = self._config.index.max_file_bytes

        stack: list[Path] = [root.path]
        while stack:
            current = stack.pop()
            try:
                entries = list(os.scandir(current))
            except OSError as exc:
                self._skip(SkipReason.UNREADABLE, current)
                log.warning("scandir.failed", path=str(current), error=str(exc))
                continue

            for entry in entries:
                path = Path(entry.path)
                try:
                    is_dir = entry.is_dir(follow_symlinks=follow)
                except OSError:
                    self._skip(SkipReason.UNREADABLE, path)
                    continue

                if entry.is_symlink() and not follow:
                    self._skip(SkipReason.SYMLINK, path)
                    continue

                if is_dir:
                    # Note: hidden directories are NOT blanket-skipped. `.claude`
                    # is a primary target, so only `.git` is dropped outright.
                    if entry.name in _ALWAYS_SKIP_DIRS:
                        continue
                    prune = matcher.reason(path, is_dir=True)
                    if prune is not None:
                        self.pruned_dirs[prune.value] += 1
                        log.debug("discovery.prune", reason=prune.value, path=str(path))
                        continue
                    stack.append(path)
                    continue

                if not entry.is_file(follow_symlinks=follow):
                    continue

                candidate = self._consider(root, path, entry, matcher, repo_cache, max_bytes)
                if candidate is not None:
                    yield candidate

    def _consider(
        self,
        root: RootConfig,
        path: Path,
        entry: os.DirEntry[str],
        matcher: IgnoreMatcher,
        repo_cache: dict[str, RepoInfo | None],
        max_bytes: int,
    ) -> FileCandidate | None:
        reason = matcher.reason(path)
        if reason is not None:
            self._skip(reason, path)
            return None

        if is_lockfile(path):
            self._skip(SkipReason.LOCKFILE, path)
            return None

        try:
            stat = entry.stat(follow_symlinks=False)
        except OSError:
            self._skip(SkipReason.UNREADABLE, path)
            return None

        if stat.st_size == 0:
            self._skip(SkipReason.EMPTY, path)
            return None
        if stat.st_size > max_bytes:
            self._skip(SkipReason.TOO_LARGE, path)
            return None

        # OPAQUE files are still yielded: they are recorded in the manifest so
        # `status` can say the file is known and deliberately not embedded.
        # That is not a skip, so it is not counted as one.
        kind, language = classify(path)

        rel_path = path.relative_to(root.path).as_posix()
        unit = self._unit_for(root, rel_path)
        if unit not in repo_cache:
            unit_dir = root.path / unit if unit else root.path
            repo_cache[unit] = read_repo_info(unit_dir)

        return FileCandidate(
            root_label=root.resolved_label,
            unit=unit,
            abs_path=path,
            rel_path=rel_path,
            kind=kind,
            language=language,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            repo=repo_cache[unit],
        )

    @staticmethod
    def _unit_for(root: RootConfig, rel_path: str) -> str:
        """The top-level subdirectory a file belongs to.

        With recurse_into_children, a workspace root holds a mix of repos and
        plain folders; the unit is what makes "search only Repo2" expressible
        for both, where a repo-name filter would miss the plain folders.
        """
        if not root.recurse_into_children:
            return ""
        head, _, tail = rel_path.partition("/")
        return head if tail else ""

    def _skip(self, reason: SkipReason, path: Path) -> None:
        self.skips[reason.value] += 1
        log.debug("discovery.skip", reason=reason.value, path=str(path))
