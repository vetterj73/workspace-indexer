"""Matryoshka truncation, shared by reprojection and query embedding.

voyage-code-4 nests its embeddings: the first k entries of a 2048-d vector are
themselves a valid k-d embedding. That is what makes a narrower collection free
to derive -- but it only works if the *query* is truncated the same way the
documents were. Comparing a 2048-d query against 1024-d documents is not a
degraded search, it is an error.
"""

from __future__ import annotations

import math


def truncate(vector: list[float], dimensions: int) -> list[float]:
    """Take the leading entries and re-normalise.

    Cosine distance ignores magnitude, so normalising is not strictly required
    today -- but it costs nothing and keeps the result correct if a collection
    is ever switched to dot-product distance, where it very much is.
    """
    head = vector[:dimensions]
    norm = math.sqrt(sum(value * value for value in head))
    if norm == 0.0:
        return head
    return [value / norm for value in head]
