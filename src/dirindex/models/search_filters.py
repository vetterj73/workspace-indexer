"""Payload filters applied inside the vector search, not after it.

Filtering server-side is what makes "search only Repo2" cheap; filtering the
returned page would silently shrink the result set instead.
"""

from __future__ import annotations

from pydantic import BaseModel

from dirindex.models.file_kind import FileKind


class SearchFilters(BaseModel):
    root_label: str | None = None
    unit: str | None = None
    repo_name: str | None = None
    language: str | None = None
    kind: FileKind | None = None
    path_prefix: str | None = None
    symbol_kind: str | None = None

    def is_empty(self) -> bool:
        return not self.model_dump(exclude_none=True)
