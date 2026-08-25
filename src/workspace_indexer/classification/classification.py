"""One classifier's verdict on one file."""

from __future__ import annotations

from pydantic import BaseModel

from workspace_indexer.classification.document_type import DocumentType


class Classification(BaseModel):
    doc_type: DocumentType
    # 0.0 to 1.0. The chain escalates below a threshold rather than guessing,
    # so a rule that is merely plausible must say so.
    confidence: float = 1.0
    # Why. "under docs/adr/" is checkable; "the model said so" is not, and
    # auditability is a large part of the case for rules over inference.
    reason: str = ""

    @property
    def decided(self) -> bool:
        return self.doc_type is not DocumentType.UNKNOWN
