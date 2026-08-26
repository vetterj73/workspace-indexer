"""One result returned from the index."""

from __future__ import annotations

from pydantic import BaseModel

from workspace_indexer.models.document_type import DocumentType
from workspace_indexer.models.file_kind import FileKind


class SearchHit(BaseModel):
    chunk_id: str
    score: float
    rel_path: str
    # Where the file actually is, so staleness can be checked without the
    # search layer having to resolve root labels back to paths.
    abs_path: str = ""
    root_label: str
    unit: str = ""
    repo_name: str | None = None
    is_repo: bool = False
    kind: FileKind = FileKind.CODE
    language: str | None = None
    symbol_path: str | None = None
    symbol_name: str | None = None
    doc_type: DocumentType = DocumentType.UNKNOWN
    doc_confidence: float = 0.0
    start_line: int = 1
    end_line: int = 1
    source_text: str = ""
    embed_text: str = ""
    token_count: int = 0
    content_sha: str = ""
    indexed_at: str | None = None
    # Set when the file on disk no longer matches what was indexed, so a caller
    # never silently shows text that never actually matched the query.
    stale: bool = False
    rerank_score: float | None = None

    @property
    def location(self) -> str:
        return f"{self.rel_path}:{self.start_line}-{self.end_line}"
