"""A local cross-encoder via fastembed.

The offline path, and the reason the abstraction is proven rather than assumed:
a protocol with one implementation is a guess. It also makes rerank relevance
testable with no API key, the same way the local dense model does for embedding.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from workspace_indexer.config import RerankConfig
from workspace_indexer.rerank.scoring_reranker import ScoringReranker

if TYPE_CHECKING:
    # fastembed declares py.typed but ships no fastembed/rerank/__init__.py, so
    # this is an implicit namespace package that pyright cannot treat as typed.
    # Their packaging gap, not ours; the import works fine at runtime.
    from fastembed.rerank.cross_encoder import (  # pyright: ignore[reportMissingTypeStubs]
        TextCrossEncoder,
    )

DEFAULT_LOCAL_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class LocalCrossEncoderReranker(ScoringReranker):
    name = "local"

    def __init__(self, config: RerankConfig, model: str = DEFAULT_LOCAL_MODEL) -> None:
        super().__init__(config)
        from fastembed.rerank.cross_encoder import (  # pyright: ignore[reportMissingTypeStubs]
            TextCrossEncoder,
        )

        # lazy_load so constructing one for a --dry-run downloads nothing.
        self._model: TextCrossEncoder = TextCrossEncoder(model_name=model, lazy_load=True)

    async def _score(self, query: str, documents: list[str]) -> list[float]:
        # ONNX inference is CPU-bound and synchronous. Running it inline would
        # stall the event loop that every concurrent search shares.
        return await asyncio.to_thread(self._rerank_sync, query, documents)

    def _rerank_sync(self, query: str, documents: list[str]) -> list[float]:
        return [float(score) for score in self._model.rerank(query, documents)]

    def cost_of_last_call(self) -> float | None:
        # Local inference is free, which is different from unknown.
        return 0.0
