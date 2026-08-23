"""A sparse (BM25) vector in Qdrant's index/value form."""

from __future__ import annotations

from pydantic import BaseModel


class SparseVec(BaseModel):
    indices: list[int]
    values: list[float]
