"""What a codebase can say about why it is the way it is.

Retrieval can only return what was written down. This package measures what
was, per repository, so that an empty answer can be told apart from an absent
one -- and so that "generate the missing documentation" is a decision made
against a number rather than an impression.
"""

from __future__ import annotations

from workspace_indexer.grounding.commit_scanner import CommitScanner
from workspace_indexer.grounding.coverage_service import CoverageService
from workspace_indexer.grounding.grounding_source import GroundingSource
from workspace_indexer.grounding.marker_scanner import MarkerScanner
from workspace_indexer.grounding.rationale_signals import RationaleSignals
from workspace_indexer.grounding.source_strength import SourceStrength
from workspace_indexer.grounding.unit_coverage import UnitCoverage

__all__ = [
    "CommitScanner",
    "CoverageService",
    "GroundingSource",
    "MarkerScanner",
    "RationaleSignals",
    "SourceStrength",
    "UnitCoverage",
]
