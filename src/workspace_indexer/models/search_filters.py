"""Payload filters applied inside the vector search, not after it.

Filtering server-side is what makes "search only Repo2" cheap; filtering the
returned page would silently shrink the result set instead.
"""

from __future__ import annotations

from pydantic import BaseModel

from workspace_indexer.models.document_type import DocumentType
from workspace_indexer.models.file_kind import FileKind


class SearchFilters(BaseModel):
    root_label: str | None = None
    unit: str | None = None
    repo_name: str | None = None
    language: str | None = None
    kind: FileKind | None = None
    path_prefix: str | None = None
    symbol_kind: str | None = None
    # What role the document plays. This is what separates a specification
    # from a changelog sitting in the same directory.
    doc_type: DocumentType | None = None
    # Exclusions, so search_code can drop tests and generated output
    # without the caller naming everything it does want.
    exclude_doc_types: list[DocumentType] = []

    def is_empty(self) -> bool:
        """No constraints at all.

        Empty collections count as unset: `exclude_doc_types=[]` is the default
        and constrains nothing, but it is not None, so exclude_none alone gets
        this wrong.
        """
        return not any(
            value for value in self.model_dump(exclude_none=True).values()
        )
