"""Dense embeddings through pydantic-ai.

The whole provider abstraction is the `provider:model` string in .env, so
switching from Voyage to OpenAI or to a local sentence-transformers model is a
config edit rather than a new class.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic_ai import Embedder
from pydantic_ai.embeddings import EmbeddingSettings

from workspace_indexer.models import EmbeddingSpace
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.embedding.dense")


class PydanticAiBackend:
    """A thin adapter. One method call is one HTTP request."""

    def __init__(
        self,
        space: EmbeddingSpace,
        *,
        embedder: Embedder | None = None,
        truncate: bool = True,
    ) -> None:
        self.space = space
        self._settings = EmbeddingSettings(
            dimensions=space.dimensions,
            # Let the provider truncate an over-long input rather than failing
            # the whole batch. EmbeddingService warns when this can happen, so
            # it is a degradation we can see rather than a silent one.
            truncate=truncate,
        )
        self._embedder = embedder or Embedder(space.model)
        self._last_cost: float | None = None
        self._last_tokens: int | None = None

    async def max_input_tokens(self) -> int | None:
        return await self._embedder.max_input_tokens()

    async def count_tokens(self, text: str) -> int:
        return await self._embedder.count_tokens(text)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        # Settings go on the call, not the constructor: an Embedder handed in
        # from outside would otherwise silently lose `dimensions`, which is
        # the entire point of the provider abstraction.
        result = await self._embedder.embed_documents(list(texts), settings=self._settings)
        self._record_cost(result)
        return [list(vector) for vector in result.embeddings]

    async def embed_query(self, text: str) -> list[float]:
        # Not the same call as embed_documents: Voyage and Cohere encode a
        # query differently from a document, and using the document path for a
        # query measurably degrades retrieval.
        result = await self._embedder.embed_query(text, settings=self._settings)
        self._record_cost(result)
        return list(result.embeddings[0])

    def last_cost_usd(self) -> float | None:
        return self._last_cost

    def last_tokens(self) -> int | None:
        return self._last_tokens

    def _record_cost(self, result: object) -> None:
        self._last_cost = None
        self._last_tokens = _reported_tokens(result)
        cost_fn = getattr(result, "cost", None)
        if not callable(cost_fn):
            return
        try:
            calculation = cost_fn()
        except Exception:
            # cost() raises LookupError for a provider genai-prices does not
            # know, and a missing price is not a reason to fail an index.
            return
        total = getattr(calculation, "total_price", None)
        if total is not None:
            self._last_cost = float(total)


def _reported_tokens(result: object) -> int | None:
    """The provider's own input-token count for one request.

    Read defensively: it arrives through pydantic-ai's `usage` object, which
    not every provider populates, and a missing count is a reason to fall back
    to our estimate rather than to fail a request that already succeeded.
    """
    usage = getattr(result, "usage", None)
    tokens = getattr(usage, "input_tokens", None)
    if isinstance(tokens, int) and tokens > 0:
        return tokens
    return None
