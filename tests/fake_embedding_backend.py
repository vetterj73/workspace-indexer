"""A dense backend that records what it was asked and can be made to fail.

The real providers are the wrong tool for testing batching and retry: they cost
money, and a 429 cannot be summoned on demand.
"""

from __future__ import annotations

from collections.abc import Sequence

from workspace_indexer.models import EmbeddingSpace


class FakeEmbeddingBackend:
    def __init__(
        self,
        *,
        dimensions: int = 4,
        max_tokens: int | None = 1000,
        fail_times: int = 0,
        error: Exception | None = None,
        cost_per_call: float | None = 0.001,
        tokens_per_call: int | None = None,
        returned_dimensions: int | None = None,
        drop_last: bool = False,
    ) -> None:
        self.space = EmbeddingSpace(model="fake:model", dimensions=dimensions)
        self._max_tokens = max_tokens
        self._fail_times = fail_times
        self._error = error or RuntimeError("boom")
        self._cost = cost_per_call
        # None models a provider that reports no usage, so the caller has
        # to fall back to its own estimate.
        self._tokens = tokens_per_call
        # Simulate a provider whose vector width disagrees with our config.
        self._returned_dimensions = returned_dimensions or dimensions
        # Simulate a provider returning fewer vectors than inputs.
        self._drop_last = drop_last

        self.batches: list[list[str]] = []
        self.queries: list[str] = []
        self.calls = 0
        self.concurrent = 0
        self.max_concurrent = 0
        self.exact_counts = 0
        self.stats_documents = 0

    async def max_input_tokens(self) -> int | None:
        return self._max_tokens

    async def count_tokens(self, text: str) -> int:
        self.exact_counts += 1
        return len(text.split())

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            if self._fail_times > 0:
                self._fail_times -= 1
                raise self._error
            self.batches.append(list(texts))
            self.stats_documents += len(texts)
            vectors = [self._vector(t) for t in texts]
            if self._drop_last and vectors:
                vectors.pop()
            return vectors
        finally:
            self.concurrent -= 1

    async def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        if self._fail_times > 0:
            self._fail_times -= 1
            raise self._error
        self.queries.append(text)
        return self._vector(text)

    def last_cost_usd(self) -> float | None:
        return self._cost

    def last_tokens(self) -> int | None:
        return self._tokens

    def _vector(self, text: str) -> list[float]:
        # Deterministic and text-dependent, so a misordered result is visible.
        seed = float(len(text))
        return [seed + i for i in range(self._returned_dimensions)]
