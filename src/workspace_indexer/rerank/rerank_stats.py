"""Running totals for reranking."""

from __future__ import annotations

from pydantic import BaseModel


class RerankStats(BaseModel):
    calls: int = 0
    documents: int = 0
    degraded: int = 0
    # How often reranking left the top result where it was. Consistently zero
    # churn means we are paying a round trip per search for nothing.
    unchanged_top: int = 0
    est_cost_usd: float = 0.0
