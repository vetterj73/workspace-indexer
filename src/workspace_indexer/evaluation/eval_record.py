"""One eval run, as it is written to disk.

Flat and versioned on purpose. JSON does not enforce a shape the way a columnar
format would, so the discipline has to come from here plus a test -- otherwise
files drift and anything reading across them gets unreliable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from workspace_indexer.evaluation.eval_result import EvalResult

# Bump when the record shape changes incompatibly, so a reader can tell rather
# than silently misinterpreting an older file.
SCHEMA_VERSION = 1


class EvalRecord(BaseModel):
    schema_version: int = SCHEMA_VERSION
    recorded_at: str
    label: str

    # Everything that affects the numbers, so two runs are only comparable when
    # these match. config_hash deliberately excludes credentials -- rotating an
    # API key is not a change in configuration.
    config_hash: str
    space_slug: str
    embedding_model: str
    dimensions: int
    fusion: str
    reranker: str
    limit: int

    recall_at_k: float
    mrr_at_k: float
    case_count: int
    miss_count: int

    results: list[EvalResult] = Field(default_factory=list[EvalResult])

    def comparable_to(self, other: EvalRecord) -> bool:
        """Whether a delta between these two means anything.

        Different config hashes measure different systems. Reporting a delta
        across them looks authoritative and is not.

        fusion and reranker are checked separately because neither is in
        config_hash: both are per-call overrides rather than settings. Turning
        reranking off is a deliberate experiment, not a regression, and an
        automatic "vs last run" that conflated the two would report a 0.3 drop
        in MRR as though something had broken.
        """
        return (
            self.config_hash == other.config_hash
            and self.fusion == other.fusion
            and self.reranker == other.reranker
        )
