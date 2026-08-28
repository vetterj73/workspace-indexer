"""Reranking inside the aggregation, via Atlas's `$rerank` stage.

Runs a Voyage reranker model where the documents already are, instead of
shipping fifty candidates to the Voyage API and back. Same models -- this is
Voyage either way -- so what it buys is one round trip rather than two, and
what it costs is that the model choice moves into the database.

Requires Native Reranking to be enabled for the Atlas project. It is not
enabled by default and cannot be turned on from here; a project without it
fails the query rather than quietly skipping the stage, which is the right way
round.
"""

from __future__ import annotations

from typing import Any

# What Atlas accepts. Checked here rather than left to the server because the
# failure arrives mid-query otherwise, one round trip and one confusing error
# message later.
MODELS = frozenset({"rerank-2.5", "rerank-2.5-lite", "rerank-2", "rerank-2-lite"})

# Atlas caps a single `$rerank` at this many documents.
MAX_DOCUMENTS = 1000

# What the reranker reads. `source_text` is the chunk body and is written for
# every document by `to_payload`; `context_header` carries the file and symbol
# trail. Both together are the closest equivalent to the `embed_text` the
# client-side reranker scores, which matters because the two are meant to be
# comparable measurements rather than merely both "reranked".
PATHS = ("context_header", "source_text")


class AtlasRerank:
    def __init__(self, model: str, *, candidates: int = 50) -> None:
        if model not in MODELS:
            raise ValueError(
                f"unknown Atlas reranker model {model!r}; "
                f"expected one of {', '.join(sorted(MODELS))}. "
                "Configure it as `database:rerank-2.5-lite`."
            )
        self.name = f"database:{model}"
        self._model = model
        self._candidates = min(max(1, candidates), MAX_DOCUMENTS)

    def depth(self, limit: int) -> int:
        """Retrieve the candidate set, not the page.

        `SearchService` normally does this widening, but with server-side
        reranking it holds a NoopReranker and does not know a rerank is coming.
        So the store widens for itself, or the reranker reorders the ten
        documents it was going to return anyway and buys nothing.
        """
        return max(limit, self._candidates)

    def stages(self, query: str, limit: int, score_meta: str) -> list[dict[str, Any]]:
        """`score_meta` is deliberately ignored.

        Whatever the retrieval branch scored, `$rerank` replaces the ordering
        and publishes its own score, so carrying the branch's score forward
        would leave results ordered by one number and labelled with another.
        """
        return [
            # `$rerank` fails outright if a path is missing from any document
            # rather than skipping it -- the docs are explicit -- so the fields
            # are defaulted rather than assumed. Written back onto themselves,
            # so nothing new appears in the payload the hit is built from.
            {"$set": {path: {"$ifNull": [f"${path}", ""]} for path in PATHS}},
            {
                "$rerank": {
                    "model": self._model,
                    "query": {"text": query},
                    "path": list(PATHS),
                    "numDocsToRerank": self._candidates,
                }
            },
            {"$addFields": {"score": {"$meta": "score"}}},
            # Deep in, shallow out. Without this the caller gets the whole
            # candidate set, reranked.
            {"$limit": limit},
        ]
