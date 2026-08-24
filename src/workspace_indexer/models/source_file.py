"""A file that survived discovery and is about to be chunked."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from workspace_indexer.models.file_kind import FileKind
from workspace_indexer.models.repo_info import RepoInfo


class SourceFile(BaseModel):
    root_label: str
    # The top-level subdirectory of the root — a repo or a plain folder.
    # Empty for files sitting directly in the root.
    unit: str = ""
    abs_path: Path
    rel_path: str
    kind: FileKind
    language: str | None
    size: int
    mtime_ns: int
    sha256: str
    repo: RepoInfo | None = None
    # None for OPAQUE/IMAGE: those are never read into memory.
    text: str | None = None
