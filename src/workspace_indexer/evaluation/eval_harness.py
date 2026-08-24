"""Measuring retrieval quality.

Every knob in this system — dimensions, fusion mode, prefetch limit, rerank
model, chunk size, whether the context header helps — is a plausible-sounding
choice that can only be settled by measurement. Without this, tuning becomes
folklore.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import yaml

from workspace_indexer.evaluation.eval_case import EvalCase
from workspace_indexer.evaluation.eval_report import EvalReport
from workspace_indexer.evaluation.eval_result import EvalResult
from workspace_indexer.obs.logging import get_logger
from workspace_indexer.search.search_request import SearchRequest
from workspace_indexer.search.search_service import SearchService

log = get_logger("workspace_indexer.evaluation")


def load_cases(path: Path) -> list[EvalCase]:
    if not path.is_file():
        raise FileNotFoundError(
            f"no eval dataset at {path}. Write ~20-30 real queries against this "
            "workspace with the files each should return."
        )
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ValueError(f"{path} must be a list of cases")
    # isinstance narrows `object` only to list[Unknown]; each element is
    # validated by EvalCase anyway, which is where a malformed case is caught.
    items = cast("list[Any]", raw)
    return [EvalCase.model_validate(item) for item in items]


class EvalHarness:
    def __init__(self, search: SearchService) -> None:
        self._search = search

    async def run(
        self,
        cases: list[EvalCase],
        *,
        limit: int = 10,
        label: str = "default",
        fusion: Literal["rrf", "dense_only", "sparse_only"] | None = None,
        rerank: bool | None = None,
    ) -> EvalReport:
        results: list[EvalResult] = []

        for case in cases:
            hits = await self._search.search(
                SearchRequest(
                    query=case.query,
                    limit=limit,
                    fusion=fusion,
                    rerank=rerank,
                    # Reading every hit's file to compare text is pure overhead
                    # for a measurement that only looks at paths.
                    check_staleness=False,
                )
            )
            found = [hit.rel_path for hit in hits]
            results.append(
                EvalResult(
                    query=case.query,
                    expected=case.expect,
                    found=found,
                    first_hit_rank=_first_rank(case.expect, found),
                )
            )

        report = EvalReport(label=label, limit=limit, results=results)
        log.info(
            "eval.report",
            label=label,
            cases=len(results),
            recall=round(report.recall_at_k, 3),
            mrr=round(report.mrr_at_k, 3),
            misses=len(report.misses),
        )
        return report


def _first_rank(expected: list[str], found: list[str]) -> int | None:
    for position, path in enumerate(found, start=1):
        if any(want in path for want in expected):
            return position
    return None
