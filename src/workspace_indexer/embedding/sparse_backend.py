"""The sparse-embedding seam.

Synchronous on purpose: BM25 runs locally with no network, so an async
interface would add ceremony without buying concurrency.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from workspace_indexer.models import SparseVec


@runtime_checkable
class SparseBackend(Protocol):
    model: str

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVec]: ...

    def encode_query(self, text: str) -> SparseVec: ...
