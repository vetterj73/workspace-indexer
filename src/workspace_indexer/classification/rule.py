"""One signal a rule-based classifier can read."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from workspace_indexer.classification.classification import Classification
from workspace_indexer.models import SourceFile


@runtime_checkable
class Rule(Protocol):
    """Returns None when this rule has nothing to say about the file.

    Rules are consulted in order and the first verdict wins, so ordering
    encodes precedence: an explicit declaration in the document beats its
    location, which beats a guess from its prose.
    """

    name: str

    def apply(self, file: SourceFile) -> Classification | None: ...
