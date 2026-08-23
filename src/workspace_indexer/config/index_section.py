"""The `index:` block — traversal and filtering rules."""

from __future__ import annotations

from pydantic import Field

from workspace_indexer.config.strict import Strict


class IndexSection(Strict):
    respect_gitignore: bool = True
    follow_symlinks: bool = False
    max_file_bytes: int = 1_048_576
    exclude: list[str] = Field(default_factory=list)
