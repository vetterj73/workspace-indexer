"""Identity of one vector space."""

from __future__ import annotations

from pydantic import BaseModel


class EmbeddingSpace(BaseModel):
    """Vectors from two different spaces are not comparable — cosine distance
    between them is noise — so a space maps one-to-one onto a collection."""

    model: str
    dimensions: int
    sparse_model: str = "Qdrant/bm25"
    # Set when this space was derived by Matryoshka truncation rather than
    # embedded natively at this width. Without it, "asked the model for 1024"
    # and "truncated 2048 down to 1024" produce the same slug and therefore the
    # same collection — and a partial run leaves the two silently mixed.
    derived_from: int | None = None

    def slug(self) -> str:
        safe = self.model.replace(":", "_").replace("/", "_").replace(".", "-")
        base = f"{safe}_{self.dimensions}"
        return base if self.derived_from is None else f"{base}_from_{self.derived_from}"

    @property
    def is_derived(self) -> bool:
        return self.derived_from is not None
