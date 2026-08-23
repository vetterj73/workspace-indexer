"""Reranking settings."""

from __future__ import annotations

from typing import Literal

from workspace_indexer.config.strict import Strict


class RerankConfig(Strict):
    enabled: bool = True
    model: str = "rerank-2.5-lite"
    candidates: int = 50
    top_n: int = 10
    # The reranker benefits from the same context header the embedder gets: a
    # bare `def upsert(...)` body is ambiguous without its file and class.
    rerank_text: Literal["embed_text", "source_text"] = "embed_text"
    # The rerank models follow instructions but expose no instruction
    # parameter, so this is prepended to the query string client-side.
    instruction: str | None = None
    # degrade: an API failure returns the fusion ordering with a WARNING.
    # fail: raise. Only the eval harness wants that — a silent degradation
    # there would quietly corrupt a measurement.
    on_error: Literal["degrade", "fail"] = "degrade"
