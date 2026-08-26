"""Running totals for one indexing run.

Kept as data rather than log-scraping so `status` can answer "why did this cost
$40" from the manifest, and so a cost regression is visible over time instead
of arriving on an invoice.
"""

from __future__ import annotations

from pydantic import BaseModel


class EmbeddingStats(BaseModel):
    requests: int = 0
    retries: int = 0
    documents: int = 0
    tokens: int = 0
    truncated: int = 0
    est_cost_usd: float = 0.0
    # Providers do not all report cost. Requests the provider would not price
    # *and* config could not price are counted here, so a zero total stays
    # distinguishable from "nobody told us".
    unpriced_requests: int = 0
    # Requests priced from `EMBEDDING_PRICE_PER_MTOK` rather than by the
    # provider. A configured rate goes stale silently, so a total containing
    # any of these is an estimate and has to be labelled as one.
    config_priced_requests: int = 0
    # Requests where the provider gave no token count and we fell back to our
    # own estimate. That estimate runs high -- 13-22% per call against
    # voyage-code-4 -- so a token total containing these is soft.
    estimated_token_requests: int = 0

    def merge(self, other: EmbeddingStats) -> None:
        self.requests += other.requests
        self.retries += other.retries
        self.documents += other.documents
        self.tokens += other.tokens
        self.truncated += other.truncated
        self.est_cost_usd += other.est_cost_usd
        self.unpriced_requests += other.unpriced_requests
        self.config_priced_requests += other.config_priced_requests
        self.estimated_token_requests += other.estimated_token_requests

    @property
    def cost_is_estimate(self) -> bool:
        """Whether `est_cost_usd` came from a configured rate rather than the
        provider. The number is still worth having; it is not worth trusting
        to the cent."""
        return self.config_priced_requests > 0

    @property
    def cost_is_known(self) -> bool:
        """False when at least one request could be priced by neither the
        provider nor config -- the case where `$0.0000` is a lie."""
        return self.unpriced_requests == 0
