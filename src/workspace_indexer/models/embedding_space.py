"""Identity of one vector space."""

from __future__ import annotations

from pydantic import BaseModel


class EmbeddingSpace(BaseModel):
    """Vectors from two different spaces are not comparable — cosine distance
    between them is noise — so a space maps one-to-one onto a collection."""

    model: str
    dimensions: int
    sparse_model: str = "Qdrant/bm25"

    def slug(self) -> str:
        safe = self.model.replace(":", "_").replace("/", "_").replace(".", "-")
        return f"{safe}_{self.dimensions}"
