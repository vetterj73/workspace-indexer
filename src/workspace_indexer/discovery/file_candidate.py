"""A path that passed the ignore rules, before its contents are read.

Kept separate from SourceFile on purpose: the manifest's fast path decides
whether to read a file at all based on mtime and size alone, so the walker must
never touch file contents.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from workspace_indexer.models import FileKind, RepoInfo


class FileCandidate(BaseModel):
    root_label: str
    abs_path: Path
    rel_path: str
    unit: str = ""
    kind: FileKind
    language: str | None
    size: int
    mtime_ns: int
    repo: RepoInfo | None = None
