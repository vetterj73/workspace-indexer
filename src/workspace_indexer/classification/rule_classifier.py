"""Rung one: classify from path, declaration and prose style.

Free, deterministic and auditable. Every verdict carries the rule that produced
it, so "why is this normative" has a checkable answer -- which is most of the
argument for rules ahead of inference.
"""

from __future__ import annotations

from workspace_indexer.classification.classification import Classification
from workspace_indexer.classification.frontmatter_rule import FrontmatterRule
from workspace_indexer.classification.modal_density_rule import ModalDensityRule
from workspace_indexer.classification.path_rule import PathRule
from workspace_indexer.classification.rule import Rule
from workspace_indexer.models import DocumentType, FileKind, SourceFile
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.classification")

# Kinds whose type is settled by what they are, with no inference required.
_BY_KIND: dict[FileKind, DocumentType] = {
    FileKind.CODE: DocumentType.IMPLEMENTATION,
    FileKind.IMAGE: DocumentType.REFERENCE,
    FileKind.OPAQUE: DocumentType.REFERENCE,
}


class RuleClassifier:
    name = "rules"
    version = 1

    def __init__(self, rules: list[Rule] | None = None) -> None:
        # Order encodes precedence: an explicit declaration beats location,
        # which beats a guess from prose style.
        self._rules: list[Rule] = rules or [
            FrontmatterRule(),
            PathRule(),
            ModalDensityRule(),
        ]

    def classify(self, file: SourceFile) -> Classification:
        for rule in self._rules:
            verdict = rule.apply(file)
            if verdict is not None:
                log.debug(
                    "classify.matched",
                    rule=rule.name,
                    doc_type=verdict.doc_type.value,
                    confidence=verdict.confidence,
                )
                return verdict.model_copy(
                    update={"reason": f"{rule.name}: {verdict.reason}"}
                )

        fallback = _BY_KIND.get(file.kind)
        if fallback is not None:
            return Classification(
                doc_type=fallback, confidence=0.9, reason=f"kind: {file.kind.value}"
            )

        # Prose that declared nothing, sits nowhere telling, and does not read
        # like a specification. Left for the next rung rather than guessed at.
        return Classification(
            doc_type=DocumentType.UNKNOWN, confidence=0.0, reason="no rule matched"
        )
