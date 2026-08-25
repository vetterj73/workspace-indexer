"""Detect specifications by how they talk.

RFC 2119 keywords -- MUST, SHALL, SHOULD, MUST NOT -- are the linguistic
fingerprint of a document that binds behaviour. A document telling you what you
have to do reads measurably differently from one describing what exists, and
that difference survives when the path and filename say nothing useful.

Counted per thousand words rather than absolutely, or a long README that says
"should" a dozen times in passing outranks a short, dense standard.
"""

from __future__ import annotations

import re

from workspace_indexer.classification.classification import Classification
from workspace_indexer.classification.document_type import DocumentType
from workspace_indexer.models import FileKind, SourceFile

# Uppercase forms are the strong signal: RFC 2119 capitalises deliberately, and
# an author writing MUST NOT is making a rule rather than a remark.
_EMPHATIC = re.compile(r"\b(MUST NOT|SHALL NOT|MUST|SHALL|REQUIRED|SHOULD NOT|SHOULD)\b")
_ORDINARY = re.compile(
    r"\b(must not|shall not|must|shall|is required to|should not|never|always)\b",
    re.IGNORECASE,
)

# Tuned so a genuine standards document clears the bar and ordinary prose does
# not. Emphatic forms count for more because they are unambiguous.
_EMPHATIC_WEIGHT = 4.0
_THRESHOLD_PER_1000 = 12.0
_MIN_WORDS = 120

_PROSE = frozenset({FileKind.MARKDOWN, FileKind.TEXT, FileKind.PDF})


def modal_density(text: str) -> float:
    """Weighted RFC-2119 keyword count per thousand words."""
    words = len(text.split())
    if words < _MIN_WORDS:
        return 0.0
    emphatic = len(_EMPHATIC.findall(text))
    ordinary = len(_ORDINARY.findall(text)) - emphatic
    weighted = emphatic * _EMPHATIC_WEIGHT + max(0, ordinary)
    return weighted * 1000.0 / words


class ModalDensityRule:
    name = "modal_density"

    def apply(self, file: SourceFile) -> Classification | None:
        # Code is full of `should` in test names and assertions; this signal
        # only means anything in prose.
        if file.kind not in _PROSE or not file.text:
            return None

        density = modal_density(file.text)
        if density < _THRESHOLD_PER_1000:
            return None

        # Deliberately not 1.0. This is a genuine inference rather than a
        # declaration, so the chain can still be overruled by better evidence
        # and a later rung can revisit it.
        confidence = min(0.85, 0.55 + density / 100.0)
        return Classification(
            doc_type=DocumentType.NORMATIVE,
            confidence=round(confidence, 2),
            reason=f"RFC-2119 modal density {density:.0f} per 1000 words",
        )
