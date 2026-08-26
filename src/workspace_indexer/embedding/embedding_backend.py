"""The dense-embedding seam."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from workspace_indexer.models import EmbeddingSpace


@runtime_checkable
class EmbeddingBackend(Protocol):
    """One provider's dense embeddings.

    Deliberately thin: one call here is one request. Batching, retries and
    concurrency belong in EmbeddingService so they are written once rather than
    once per provider.
    """

    space: EmbeddingSpace

    async def max_input_tokens(self) -> int | None: ...

    async def count_tokens(self, text: str) -> int: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...

    def last_cost_usd(self) -> float | None:
        """Cost of the most recent request, when the provider reports it.

        None means "unknown", 0.0 means "free". Collapsing the two is how a
        run against an unpriced provider comes to look like a free one.
        """
        ...

    def last_tokens(self) -> int | None:
        """Input tokens the provider counted for the most recent request.

        The authority, where we can get it. Our own estimate runs 13-22% high
        per call against voyage-code-4, and cumulatively 45% high across a
        manifest, which is far too much drift to bill or budget against.
        """
        ...
