"""What document types this index actually holds."""

from __future__ import annotations

from pydantic import BaseModel

from workspace_indexer.mcp.taxonomy_entry import TaxonomyEntry

# Bump when a type is renamed or removed. Once agents filter on `normative`,
# that string is an API: renaming it silently turns every guidance query into
# an error, or worse, into an empty result.
TAXONOMY_VERSION = 1


class Taxonomy(BaseModel):
    taxonomy_version: int = TAXONOMY_VERSION
    # Which embedding space these counts came from. Two collections can hold
    # different content, and a count without that context is not reproducible.
    space: str = ""
    total_chunks: int = 0
    types: list[TaxonomyEntry] = []
