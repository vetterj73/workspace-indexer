"""A row in the runs table."""

from __future__ import annotations

from pydantic import BaseModel


class RunRecord(BaseModel):
    run_id: str
    started_at: str
    # None means the run never finished: a crash or an interrupt. Visible as
    # such rather than silently absent.
    finished_at: str | None = None
    mode: str = "index"
    files_seen: int = 0
    files_skipped: int = 0
    files_changed: int = 0
    chunks_upserted: int = 0
    chunks_deleted: int = 0
    tokens_embedded: int = 0
    est_cost_usd: float = 0.0
    errors: int = 0
    config_hash: str | None = None

    @property
    def unfinished(self) -> bool:
        return self.finished_at is None
