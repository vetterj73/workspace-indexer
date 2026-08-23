"""Ignore-rule evaluation: config excludes plus the .gitignore chain.

Uses pathspec so we parse gitignore syntax in-process. Shelling out to
`git check-ignore` per path would be one subprocess per file, which on a
workspace-sized tree is the difference between seconds and many minutes.

GitIgnoreSpec rather than the generic PathSpec: it is the class built for
.gitignore semantics, where the last matching pattern wins and negation with
`!` has to override an earlier exclusion.
"""

from __future__ import annotations

from pathlib import Path

import pathspec

from workspace_indexer.discovery.skip_reason import SkipReason

_GITIGNORE = ".gitignore"


class IgnoreMatcher:
    """Matches paths relative to one root.

    .gitignore files are discovered per directory as we descend and cached, so
    nested ignore files apply to their own subtree the way git does.
    """

    def __init__(self, root: Path, excludes: list[str], respect_gitignore: bool) -> None:
        self._root = root
        self._respect_gitignore = respect_gitignore
        self._excludes = pathspec.GitIgnoreSpec.from_lines(excludes)
        self._dir_specs: dict[Path, pathspec.GitIgnoreSpec | None] = {}

    def _spec_for_dir(self, directory: Path) -> pathspec.GitIgnoreSpec | None:
        if directory in self._dir_specs:
            return self._dir_specs[directory]
        gitignore = directory / _GITIGNORE
        spec: pathspec.GitIgnoreSpec | None = None
        if gitignore.is_file():
            try:
                lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
                spec = pathspec.GitIgnoreSpec.from_lines(lines)
            except OSError:
                spec = None
        self._dir_specs[directory] = spec
        return spec

    def _gitignored(self, path: Path, is_dir: bool) -> bool:
        """Walk up from the file's directory to the root, applying each
        .gitignore to the path relative to that ignore file's own directory."""
        directory = path.parent
        while True:
            spec = self._spec_for_dir(directory)
            if spec is not None:
                try:
                    relative = path.relative_to(directory).as_posix()
                except ValueError:  # pragma: no cover - defensive
                    break
                if is_dir:
                    relative += "/"
                if spec.match_file(relative):
                    return True
            if directory == self._root:
                break
            parent = directory.parent
            if parent == directory:
                break
            directory = parent
        return False

    def reason(self, path: Path, is_dir: bool = False) -> SkipReason | None:
        """None means "keep it"."""
        try:
            relative = path.relative_to(self._root).as_posix()
        except ValueError:  # pragma: no cover - defensive
            return None
        probe = relative + "/" if is_dir else relative
        if self._excludes.match_file(probe):
            return SkipReason.EXCLUDED
        if self._respect_gitignore and self._gitignored(path, is_dir):
            return SkipReason.GITIGNORED
        return None
