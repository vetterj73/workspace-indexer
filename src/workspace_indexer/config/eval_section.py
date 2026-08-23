"""The `eval:` block — where the retrieval-quality dataset lives."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from workspace_indexer.config.strict import Strict


class EvalSection(Strict):
    dataset: Path = Path("./config/eval.yaml")
    metrics: list[str] = Field(default_factory=lambda: ["recall@10", "mrr@10"])
