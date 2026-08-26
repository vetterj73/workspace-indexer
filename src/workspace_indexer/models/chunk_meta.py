"""Everything we know about a chunk apart from its text."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from workspace_indexer.models.document_type import DocumentType
from workspace_indexer.models.file_kind import FileKind
from workspace_indexer.models.repo_info import RepoInfo


class ChunkMeta(BaseModel):
    workspace: str
    root_label: str
    # The top-level subdirectory of the root this file belongs to — a repo
    # or a plain folder. Empty for files sitting directly in the root.
    unit: str = ""
    abs_path: Path
    rel_path: str
    kind: FileKind
    language: str | None = None
    repo: RepoInfo | None = None
    symbol_path: str | None = None
    symbol_kind: str | None = None
    symbol_name: str | None = None
    start_line: int = 1
    end_line: int = 1
    chunk_index: int = 0
    chunk_total: int = 1
    content_sha: str = ""
    token_estimate: int = 0
    chunker: str = ""
    chunker_version: int = 1
    # What role this document plays, as distinct from how it was chunked.
    # One classification per file; every chunk of it inherits the verdict.
    doc_type: DocumentType = DocumentType.UNKNOWN
    doc_type_confidence: float = 0.0
    classifier_version: int = 0
    # True when tree-sitter hit error nodes or no grammar was available, so a
    # quality problem is visible in the payload rather than only in the log.
    parse_degraded: bool = False
