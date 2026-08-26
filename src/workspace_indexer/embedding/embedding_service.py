"""Batching, concurrency, retries and accounting around a dense backend.

Written once here rather than once per provider. Everything in this module
exists because embedding calls cost money and fail in ways that are invisible
without instrumentation: a partial batch, a silently truncated input, a retry
storm, a dimension mismatch that poisons a whole collection.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence

from workspace_indexer.chunking.token_estimate import estimate_tokens
from workspace_indexer.embedding.embedding_backend import EmbeddingBackend
from workspace_indexer.embedding.embedding_stats import EmbeddingStats
from workspace_indexer.embedding.retry_policy import RetryPolicy
from workspace_indexer.embedding.token_pricer import TokenPricer
from workspace_indexer.models import EmbeddingSpace, FileKind
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.embedding.service")

# Only pay for an exact token count when the cheap estimate says we are near
# the limit. Counting every chunk exactly would add an await per chunk for a
# check that almost always passes.
_EXACT_COUNT_THRESHOLD = 0.8


class EmbeddingService:
    def __init__(
        self,
        backend: EmbeddingBackend,
        *,
        batch_size: int = 64,
        max_concurrency: int = 4,
        max_batch_tokens: int = 100_000,
        retry: RetryPolicy | None = None,
        pricer: TokenPricer | None = None,
    ) -> None:
        self._backend = backend
        self._pricer = pricer or TokenPricer(None)
        self._warned_unpriced = False
        self._batch_size = max(1, batch_size)
        self._max_batch_tokens = max(1, max_batch_tokens)
        self._retry = retry or RetryPolicy()
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self.stats = EmbeddingStats()

    @property
    def space(self) -> EmbeddingSpace:
        return self._backend.space

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Vectors aligned one-to-one with `texts`, in the same order."""
        if not texts:
            return []

        await self._warn_on_truncation(texts)
        batches = self._batch(texts)
        log.debug("embed.plan", documents=len(texts), batches=len(batches))

        results = await asyncio.gather(*(self._embed_batch(batch) for batch in batches))

        vectors = [vector for batch in results for vector in batch]
        if len(vectors) != len(texts):
            # Silent misalignment is the worst outcome available here: every
            # chunk after the gap would be stored with someone else's vector.
            raise RuntimeError(
                f"embedding backend returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        vector = await self._with_retry(lambda: self._backend.embed_query(text), documents=1)
        self._check_dimensions([vector])
        return vector

    def _batch(self, texts: Sequence[str]) -> list[list[str]]:
        """Split on document count and on a token budget.

        Count alone is not enough: 64 chunks of 512 tokens is fine, but 64
        chunks that each happen to be huge exceeds the provider's per-request
        total and fails the whole batch.
        """
        batches: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0

        for text in texts:
            tokens = estimate_tokens(text, FileKind.CODE)
            over_count = len(current) >= self._batch_size
            over_tokens = current and current_tokens + tokens > self._max_batch_tokens
            if over_count or over_tokens:
                batches.append(current)
                current, current_tokens = [], 0
            current.append(text)
            current_tokens += tokens

        if current:
            batches.append(current)
        return batches

    def _tokens_for(self, batch: list[str]) -> tuple[int, bool]:
        """The provider's count where it gave one, ours otherwise.

        Ours runs high -- 13-22% per call against voyage-code-4, and 45% high
        cumulatively across this manifest once dry runs and retries are in the
        mix. Fine for deciding a batch size, far too loose to bill or budget
        against, which is what the reported count is for.
        """
        reported = self._backend.last_tokens()
        if reported is not None:
            return reported, True
        self.stats.estimated_token_requests += 1
        return sum(estimate_tokens(text, FileKind.CODE) for text in batch), False

    def _price(self, tokens: int) -> float | None:
        """Provider price, then a configured rate, then an honest unknown."""
        cost = self._backend.last_cost_usd()
        if cost is not None:
            self.stats.est_cost_usd += cost
            return cost

        estimated = self._pricer.cost_of(tokens)
        if estimated is not None:
            self.stats.est_cost_usd += estimated
            self.stats.config_priced_requests += 1
            return estimated

        self.stats.unpriced_requests += 1
        self._warn_unpriced_once()
        return None

    def _warn_unpriced_once(self) -> None:
        """Said during the run, not only discovered in the table afterwards.

        Once per process: this fires on every batch of a full index, and a
        thousand copies of the same warning is how a real one gets missed.
        """
        if self._warned_unpriced:
            return
        self._warned_unpriced = True
        log.warning(
            "embed.unpriced",
            model=self._backend.space.model,
            detail="the provider reports no price and EMBEDDING_PRICE_PER_MTOK is "
            "unset, so this run's cost will be recorded as unknown rather than as "
            "zero. Set it to price the run.",
        )

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        async with self._semaphore:
            started = time.monotonic()
            vectors = await self._with_retry(
                lambda: self._backend.embed_documents(batch), documents=len(batch)
            )
            self._check_dimensions(vectors)

            tokens, reported = self._tokens_for(batch)
            cost = self._price(tokens)
            self.stats.documents += len(batch)
            self.stats.tokens += tokens

            log.info(
                "embed.batch",
                documents=len(batch),
                tokens=tokens,
                tokens_reported=reported,
                model=self._backend.space.model,
                duration_ms=round((time.monotonic() - started) * 1000, 1),
                cost_usd=cost,
            )
            return vectors

    async def _with_retry[T](self, call: Callable[[], Awaitable[T]], *, documents: int) -> T:
        attempt = 0
        while True:
            attempt += 1
            try:
                result = await call()
            except Exception as exc:
                if not self._retry.should_retry(exc, attempt):
                    log.error(
                        "error.embed_failed",
                        attempt=attempt,
                        documents=documents,
                        status=self._retry.status_of(exc),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    raise
                delay = self._retry.delay_for(exc, attempt)
                self.stats.retries += 1
                log.warning(
                    "embed.retry",
                    attempt=attempt,
                    max_attempts=self._retry.max_attempts,
                    documents=documents,
                    status=self._retry.status_of(exc),
                    backoff_s=round(delay, 2),
                    error=f"{type(exc).__name__}: {exc}",
                )
                await asyncio.sleep(delay)
            else:
                self.stats.requests += 1
                return result

    async def _warn_on_truncation(self, texts: Sequence[str]) -> None:
        """Silent truncation is the classic invisible quality bug: the call
        succeeds, the vector is wrong, and nothing says so."""
        limit = await self._backend.max_input_tokens()
        if not limit:
            return
        gate = int(limit * _EXACT_COUNT_THRESHOLD)
        for text in texts:
            if estimate_tokens(text, FileKind.CODE) < gate:
                continue
            exact = await self._backend.count_tokens(text)
            if exact > limit:
                self.stats.truncated += 1
                log.warning(
                    "embed.truncated",
                    tokens=exact,
                    max_input_tokens=limit,
                    excess=exact - limit,
                    preview=text[:120],
                )

    def _check_dimensions(self, vectors: list[list[float]]) -> None:
        expected = self._backend.space.dimensions
        for vector in vectors:
            if len(vector) != expected:
                # A mismatch means the configured dimensions and the model
                # disagree. Every vector written to the collection would be
                # wrong, so fail before anything is stored.
                raise RuntimeError(
                    f"model {self._backend.space.model} returned {len(vector)} dimensions, "
                    f"configuration expects {expected}"
                )
