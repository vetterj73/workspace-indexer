"""A sparse backend with deterministic, hand-controlled term ids.

Real BM25 works offline and is tested directly elsewhere; here the point is to
control exactly which documents the sparse branch favours, so a fusion
assertion means something.
"""

from __future__ import annotations

from collections.abc import Sequence

from workspace_indexer.models import SparseVec


class FakeSparseBackend:
    def __init__(self, model: str = "fake/bm25") -> None:
        self.model = model
        self.queries: list[str] = []

    @staticmethod
    def _vector(text: str) -> SparseVec:
        # One term id per distinct word, so overlap is exactly word overlap.
        terms = sorted({abs(hash(word)) % 10_000 for word in text.lower().split()})
        return SparseVec(indices=terms, values=[1.0] * len(terms))

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVec]:
        return [self._vector(text) for text in texts]

    def encode_query(self, text: str) -> SparseVec:
        self.queries.append(text)
        return self._vector(text)
