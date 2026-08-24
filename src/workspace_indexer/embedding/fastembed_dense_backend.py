"""Local dense embeddings via fastembed.

The offline path. fastembed is already a dependency for BM25 and runs ONNX
rather than PyTorch, so a real semantic model costs ~130 MB instead of the ~2 GB
sentence-transformers would drag in. Used by tests that need retrieval to
actually work, and available as a no-API-key option.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from workspace_indexer.models import EmbeddingSpace
from workspace_indexer.obs.logging import get_logger

if TYPE_CHECKING:
    from fastembed import TextEmbedding

log = get_logger("workspace_indexer.embedding.local")

DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"


class FastembedDenseBackend:
    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL, dimensions: int = 384) -> None:
        from fastembed import TextEmbedding

        # lazy_load defers the model download to first use, so constructing a
        # backend for a --dry-run costs nothing.
        self._model: TextEmbedding = TextEmbedding(model_name=model_name, lazy_load=True)
        self._model_name = model_name
        self.space = EmbeddingSpace(model=f"fastembed:{model_name}", dimensions=dimensions)

    async def max_input_tokens(self) -> int | None:
        # bge-small's context window. Inputs beyond it are truncated by the
        # tokenizer rather than rejected.
        return 512

    async def count_tokens(self, text: str) -> int:
        # No cheap exact tokenizer is exposed, and this backend is never billed,
        # so a word-ish approximation is enough for the truncation warning.
        return max(1, len(text) // 4)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(map(float, vector)) for vector in self._model.embed(list(texts))]

    async def embed_query(self, text: str) -> list[float]:
        return [list(map(float, v)) for v in self._model.query_embed([text])][0]

    def last_cost_usd(self) -> float | None:
        # Local inference is free, which is different from unknown: report 0.
        return 0.0
