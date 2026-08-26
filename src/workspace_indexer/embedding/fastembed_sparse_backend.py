"""BM25 sparse vectors via fastembed.

Local and free, which is what makes hybrid search affordable enough to include
from the first iteration. BM25 is the half of retrieval that finds an exact
identifier, an error string, or a config key — the queries a dense embedding is
worst at, because a rare literal has no semantic neighbourhood.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from workspace_indexer.models import SparseVec
from workspace_indexer.obs.logging import get_logger

if TYPE_CHECKING:
    from fastembed import SparseTextEmbedding

log = get_logger("workspace_indexer.embedding.sparse")

DEFAULT_SPARSE_MODEL = "Qdrant/bm25"


class FastembedSparseBackend:
    def __init__(self, model: str = DEFAULT_SPARSE_MODEL) -> None:
        from fastembed import SparseTextEmbedding

        self.model = model
        self._encoder: SparseTextEmbedding = SparseTextEmbedding(model_name=model, lazy_load=True)

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVec]:
        if not texts:
            return []
        return [self._convert(e) for e in self._encoder.embed(list(texts))]

    def encode_query(self, text: str) -> SparseVec:
        # A distinct call from encode_documents: BM25 weights document terms by
        # frequency, while a query only needs term presence. Using the document
        # path for a query skews the scores.
        return next(self._convert(e) for e in self._encoder.query_embed([text]))

    @staticmethod
    def _convert(embedding: object) -> SparseVec:
        indices = getattr(embedding, "indices", [])
        values = getattr(embedding, "values", [])
        return SparseVec(
            indices=[int(i) for i in indices],
            values=[float(v) for v in values],
        )
