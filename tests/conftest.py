"""Shared fixtures.

The workspace fixture builds a real directory tree with real git repositories
rather than mocking the filesystem. Discovery is almost entirely about how the
filesystem and git actually behave — mtime granularity, symlink resolution,
where git puts its metadata, how .gitignore nests — and a mock would happily
confirm whatever we assumed instead of what is true.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from workspace_indexer.config import WorkspaceConfig
from workspace_indexer.models import FileKind, RepoInfo, SourceFile, sha256_text
from workspace_indexer.obs.logging import forget_once_only

# What the `config_for` fixture hands back. Named so tests can annotate it
# instead of accepting an untyped fixture argument.
ConfigFactory = Callable[..., WorkspaceConfig]

# Set inline so the tests do not depend on the developer's global git config.
_GIT_IDENTITY = [
    "-c",
    "user.email=test@example.com",
    "-c",
    "user.name=Test",
    "-c",
    "commit.gpgsign=false",
]


def git_init(path: Path, *, commit: bool = True) -> None:
    subprocess.run(
        ["git", "init", "-q", "--initial-branch=main", str(path)],
        check=True,
        capture_output=True,
    )
    if commit:
        subprocess.run(
            ["git", *_GIT_IDENTITY, "add", "-A"], cwd=path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", *_GIT_IDENTITY, "commit", "-q", "-m", "initial"],
            cwd=path,
            check=True,
            capture_output=True,
        )


def write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


SAMPLE_PY = '''"""Widget module."""

import os

MAX_WIDGETS = 10


class Widget:
    """A widget."""

    def __init__(self, name: str) -> None:
        self.name = name

    def render(self, ctx: dict) -> str:
        """Render the widget to a string."""
        parts = []
        for index in range(MAX_WIDGETS):
            parts.append(f"{self.name}-{index}")
        return os.linesep.join(parts)


def free_function(a: int, b: int) -> int:
    return a + b
'''

SAMPLE_MD = """# Deployment Guide

Intro paragraph about deploying.

## Rollbacks

To roll back, run the rollback script.

### Emergency rollback

Page the on-call engineer first.

## Monitoring

Watch the dashboard.
"""


@pytest.fixture(autouse=True)
def _forget_once_only_logs() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Clear the log-once ledger between tests.

    `log_once` dedupes for the life of the *process*, which is right in
    production and wrong across a test session: whichever test happens to run
    first consumes the event, and any later test asserting on it fails
    depending on ordering. Two tests assert on once-only events today
    (`rerank.skipped`, `store.search_indexes_unavailable`), and both were one
    reordering away from being flaky.

    Same family as #47 -- state surviving between tests -- and cheap enough
    that there is no reason to wait for it to bite.
    """
    forget_once_only()
    yield
    forget_once_only()


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[Path]:
    """A workspace root holding two git repos, a plain folder, and .claude/."""
    root = tmp_path / "workspace"

    # repo_one: a normal repo with a .gitignore
    one = root / "repo_one"
    write(one / "src" / "widget.py", SAMPLE_PY)
    write(one / "README.md", SAMPLE_MD)
    write(one / ".gitignore", "secret.txt\nbuild/\n*.tmp\n!important.tmp\n")
    write(one / "scratch.tmp", "throwaway")
    write(one / "important.tmp", "re-included by a negation rule")
    write(one / "secret.txt", "should never be indexed")
    write(one / "build" / "artifact.py", "print('generated')")
    write(one / "package-lock.json", '{"lockfileVersion": 3}')
    write(one / "node_modules" / "dep" / "index.js", "module.exports = {}")
    git_init(one)

    # repo_two: a repo with a nested .gitignore in a subdirectory
    two = root / "repo_two"
    write(two / "app" / "main.ts", "export const run = () => 1;\n")
    write(two / "app" / ".gitignore", "generated.ts\n")
    write(two / "app" / "generated.ts", "export const gen = 2;\n")
    # Same filename as the ignored file in repo_one, but not ignored here.
    write(two / "secret.txt", "not ignored in this repo")
    git_init(two)

    # plain_folder: deliberately not a repo
    write(root / "plain_folder" / "notes.md", "# Notes\n\nSome notes.\n")

    # .claude/: a hidden directory that MUST still be indexed
    write(root / ".claude" / "commands" / "deploy.md", "# Deploy\n\nRun it.\n")
    write(root / ".claude" / "settings.json", '{"model": "opus"}')

    # loose file directly in the root, so unit == ""
    write(root / "TOPLEVEL.md", "# Top level\n")

    # a real binary and a real image
    (root / "plain_folder" / "blob.so").write_bytes(b"\x7fELF\x00\x01\x02binary")
    (root / "plain_folder" / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    yield root


@pytest.fixture
def config_for(workspace: Path) -> ConfigFactory:
    """Build a WorkspaceConfig pointed at the fixture workspace."""

    def build(**overrides: Any) -> WorkspaceConfig:
        payload: dict[str, Any] = {
            "workspace": {
                "name": "test",
                "roots": [{"path": str(workspace), "recurse_into_children": True}],
            },
            "index": {"exclude": ["**/node_modules/**"]},
        }
        payload.update(overrides)
        return WorkspaceConfig.model_validate(payload)

    return build


def make_source(
    text: str,
    *,
    kind: FileKind = FileKind.CODE,
    language: str | None = "python",
    rel_path: str = "src/widget.py",
    root_label: str = "repo_one",
    unit: str = "repo_one",
    repo: RepoInfo | None = None,
) -> SourceFile:
    """A SourceFile without touching the filesystem.

    Chunkers only read `.text`, `.kind`, `.language` and the metadata they copy
    onto chunks, so a real file would add I/O without adding coverage. The
    filesystem-facing behaviour is tested against real files in
    test_file_reader.py instead.
    """
    return SourceFile(
        root_label=root_label,
        unit=unit,
        abs_path=Path("/tmp") / rel_path,
        rel_path=rel_path,
        kind=kind,
        language=language,
        size=len(text.encode()),
        mtime_ns=1,
        sha256=sha256_text(text),
        repo=repo,
        text=text,
    )
