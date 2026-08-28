"""The seam for reranking that happens inside the query, not after it.

Separate from `rerank.Reranker` because the two are different shapes, not two
implementations of one. A `Reranker` takes hits that have already been fetched
and returns them reordered. Server-side reranking is not a call at all: it is
stages appended to the aggregation the store was going to run anyway, so there
is no list of hits to hand it.

The pair below is the same trick `NoopReranker` uses, for the same reason. "The
store does not rerank" is a different object rather than a flag, so the search
path never asks the question -- it appends whatever stages it is given, and one
of the two implementations gives it none.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ServerReranker(Protocol):
    name: str

    def depth(self, limit: int) -> int:
        """How many documents the retrieval stages should produce.

        Retrieve deep, return shallow -- the same rule `SearchService` follows
        when it reranks client-side. It lives here as well because with
        server-side reranking the service is holding a NoopReranker and has no
        idea a rerank is coming, so nothing above the store would widen the
        candidate set.
        """
        ...

    def stages(self, query: str, limit: int, score_meta: str) -> list[dict[str, Any]]:
        """The tail of the pipeline: scoring, reranking and the final limit.

        Takes `score_meta` because the metadata name differs per branch --
        `vectorSearchScore`, `searchScore`, or `score` after a fusion -- and an
        implementation that reranks overrides all of them with its own.
        """
        ...
