"""The classification seam."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from workspace_indexer.classification.classification import Classification
from workspace_indexer.models import SourceFile


@runtime_checkable
class DocumentClassifier(Protocol):
    """One way of deciding what a document is for.

    Implementations chain cheapest-first: rules, then embedding prototypes,
    then a model on whatever the first two could not settle. Each returns a
    confidence so the chain knows when to escalate.
    """

    name: str
    # Bump when the rules change, so the manifest reclassifies rather than
    # trusting a cached verdict from an older ruleset.
    version: int

    def classify(self, file: SourceFile) -> Classification: ...
