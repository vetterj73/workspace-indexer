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
    # Requests neither the provider nor config could price. Carried this far
    # on purpose: EmbeddingStats drew the distinction correctly and RunStats
    # used to drop it, which is how `$0.0000` came to mean both "free" and
    # "no idea".
    unpriced_requests: int = 0
    # True when the cost above came from a configured rate rather than from
    # the provider.
    cost_is_estimate: bool = False
    # Import edges pointed at a file this run. Not all edges can be: a
    # package, a tsconfig alias and a C# namespace all need more than the
    # file list, and stay unresolved rather than being guessed at.
    imports_resolved: int = 0
    # Client call sites matched to the file declaring the endpoint. Counted
    # separately from imports because the two resolve under opposite rules --
    # an import inside its own repository, a route across them.
    routes_resolved: int = 0
    # Files this run would have removed from the index but did not, because
    # the deletion looked more like a bad checkout than a real removal.
    # Counted rather than logged alone: a warning in CI output is not a brake.
    # Files reindexed because --force was given rather than because they
    # changed. Counted so a completed run can be shown to have been a full
    # rebuild: `force` on run.start says one was *requested*, and a run that
    # died halfway says the same thing.
    forced: int = 0
    deletions_withheld: int = 0
    errors: int = 0
    config_hash: str = ""
    skip_reasons: dict[str, int] = Field(default_factory=dict)
