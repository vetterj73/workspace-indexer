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
    unpriced_requests: int = 0
    cost_is_estimate: bool = False
    errors: int = 0
    config_hash: str | None = None

    @property
    def unfinished(self) -> bool:
        return self.finished_at is None

    @property
    def cost_display(self) -> str:
        """Three states, told apart.

        `$0.0000` used to mean both "this run was free" and "nobody told us
        what it cost", which is the wrong answer in the more expensive
        direction. Rows written before this column existed default to
        unpriced=0, so they render as costs -- accurate for a local model,
        optimistic for the API runs that predate the fix.
        """
        if self.unpriced_requests:
            return f"unpriced ({self.unpriced_requests})"
        if self.cost_is_estimate:
            return f"~${self.est_cost_usd:.4f}"
        return f"${self.est_cost_usd:.4f}"
