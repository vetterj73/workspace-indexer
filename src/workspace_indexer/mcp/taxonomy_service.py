"""Building the taxonomy from what is in the index, not from the enum."""

from __future__ import annotations

from workspace_indexer.mcp.taxonomy import Taxonomy
from workspace_indexer.mcp.taxonomy_entry import TaxonomyEntry
from workspace_indexer.models import DocumentType, EmbeddingSpace, SearchFilters
from workspace_indexer.storage.vector_store import VectorStore


class TaxonomyService:
    """Reports every type in the enum, with counts from the live collection.

    Two halves, and both matter. Counts come from the store, so the answer
    describes *this* workspace rather than what the code can in principle
    produce. But the type list comes from the enum, so a category with nothing
    in it is still reported, at zero -- the absence is the information.
    """

    def __init__(self, store: VectorStore, space: EmbeddingSpace) -> None:
        self._store = store
        self._space = space

    async def build(self, examples_per_type: int = 3) -> Taxonomy:
        counts = await self._store.facet(self._space, "doc_type")
        entries: list[TaxonomyEntry] = []
        for doc_type in DocumentType:
            count = counts.get(doc_type.value, 0)
            examples: list[str] = []
            if count and examples_per_type:
                examples = await self._store.sample_paths(
                    self._space,
                    SearchFilters(doc_type=doc_type),
                    limit=examples_per_type,
                )
            entries.append(
                TaxonomyEntry(
                    name=doc_type.value,
                    count=count,
                    definition=doc_type.definition,
                    examples=examples,
                )
            )
        return Taxonomy(
            space=self._space.slug(),
            total_chunks=sum(counts.values()),
            types=entries,
        )
