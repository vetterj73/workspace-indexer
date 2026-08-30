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
    # One entry per page, for PDFs only; empty for everything else.
    #
    # Carried beside `text` rather than instead of it because the two have
    # different jobs and one guards the other: `text` is the whole document and
    # is what the secret scanner reads, `pages` is the same content split so a
    # chunk can say which page it came from. A test asserts they hold the same
    # content, which is what makes the scan provably cover everything the
    # chunker will embed.
    pages: list[str] = []
    # Set before chunking when the type is to be embedded, so build_header can
    # see it. Classification is a property of the file, not of a chunk, which
    # is why it belongs here rather than being threaded through every chunker.
    doc_type: str | None = None
