"""Ignore-rule evaluation.

The trickiest logic in discovery: git applies each .gitignore relative to the
directory containing it, so the same pattern means different things at
different depths, and a file's fate depends on every ignore file above it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dirindex.discovery import IgnoreMatcher, SkipReason


def test_config_exclude_matches(workspace: Path) -> None:
    matcher = IgnoreMatcher(workspace, ["**/node_modules/**"], respect_gitignore=False)
    target = workspace / "repo_one" / "node_modules" / "dep" / "index.js"
    assert matcher.reason(target) is SkipReason.EXCLUDED


def test_gitignore_applies_relative_to_its_own_directory(workspace: Path) -> None:
    """repo_one/.gitignore says `secret.txt`, which must not reach repo_two."""
    matcher = IgnoreMatcher(workspace, [], respect_gitignore=True)
    assert matcher.reason(workspace / "repo_one" / "secret.txt") is SkipReason.GITIGNORED
    assert matcher.reason(workspace / "repo_two" / "secret.txt") is None


def test_nested_gitignore_applies_to_its_subtree(workspace: Path) -> None:
    """repo_two/app/.gitignore ignores generated.ts, a sibling of main.ts."""
    matcher = IgnoreMatcher(workspace, [], respect_gitignore=True)
    assert matcher.reason(workspace / "repo_two" / "app" / "generated.ts") is SkipReason.GITIGNORED
    assert matcher.reason(workspace / "repo_two" / "app" / "main.ts") is None


def test_directory_pattern_needs_the_trailing_slash(workspace: Path) -> None:
    """`build/` in .gitignore matches the directory only when we probe it as
    one; probing it as a file would silently fail to prune the subtree."""
    matcher = IgnoreMatcher(workspace, [], respect_gitignore=True)
    build = workspace / "repo_one" / "build"
    assert matcher.reason(build, is_dir=True) is SkipReason.GITIGNORED


def test_gitignore_ignored_when_disabled(workspace: Path) -> None:
    matcher = IgnoreMatcher(workspace, [], respect_gitignore=False)
    assert matcher.reason(workspace / "repo_one" / "secret.txt") is None


def test_config_excludes_win_over_gitignore_being_off(workspace: Path) -> None:
    """Turning off gitignore must not disable our own hardcoded protections."""
    matcher = IgnoreMatcher(workspace, ["logs/**", "**/*.sqlite3"], respect_gitignore=False)
    assert matcher.reason(workspace / "logs" / "dirindex.jsonl") is SkipReason.EXCLUDED
    assert matcher.reason(workspace / "data" / "manifest.sqlite3") is SkipReason.EXCLUDED


def test_clean_paths_are_kept(workspace: Path) -> None:
    matcher = IgnoreMatcher(workspace, ["**/node_modules/**"], respect_gitignore=True)
    assert matcher.reason(workspace / "repo_one" / "src" / "widget.py") is None
    assert matcher.reason(workspace / ".claude" / "settings.json") is None


def test_gitignore_is_read_once_per_directory(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-directory cache is what keeps this O(directories) rather than
    O(files); without it every file in a repo re-reads the same ignore files.

    Asserted by counting actual reads rather than inspecting the cache dict —
    the I/O saving is the thing we care about, and it survives a refactor of
    how the cache is stored.
    """
    reads: list[str] = []
    original = Path.read_text

    def counting_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self.name == ".gitignore":
            reads.append(str(self))
        return original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", counting_read_text)

    matcher = IgnoreMatcher(workspace, [], respect_gitignore=True)
    src = workspace / "repo_one" / "src"
    for _ in range(5):
        matcher.reason(src / "widget.py")

    assert len(reads) == len(set(reads)), f"re-read the same .gitignore: {reads}"


def test_negation_reincludes_a_file(workspace: Path) -> None:
    """`*.tmp` then `!important.tmp`: the later pattern has to win, or a
    deliberate re-inclusion silently does nothing."""
    matcher = IgnoreMatcher(workspace, [], respect_gitignore=True)
    assert matcher.reason(workspace / "repo_one" / "scratch.tmp") is SkipReason.GITIGNORED
    assert matcher.reason(workspace / "repo_one" / "important.tmp") is None
