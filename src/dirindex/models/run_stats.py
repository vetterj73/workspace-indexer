"""Summary of one indexing run, persisted so cost regressions are visible."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class RunStats(BaseModel):
    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    mode: str = "index"
    files_seen: int = 0
    files_skipped: int = 0
    files_changed: int = 0
    chunks_upserted: int = 0
    chunks_deleted: int = 0
    tokens_embedded: int = 0
    est_cost_usd: float = 0.0
    errors: int = 0
    config_hash: str = ""
    skip_reasons: dict[str, int] = Field(default_factory=dict)
