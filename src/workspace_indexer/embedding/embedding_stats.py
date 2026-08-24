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
    # Providers do not all report cost. None-cost requests are counted here so
    # a zero total is distinguishable from "nobody told us".
    unpriced_requests: int = 0

    def merge(self, other: EmbeddingStats) -> None:
        self.requests += other.requests
        self.retries += other.retries
        self.documents += other.documents
        self.tokens += other.tokens
        self.truncated += other.truncated
        self.est_cost_usd += other.est_cost_usd
        self.unpriced_requests += other.unpriced_requests
