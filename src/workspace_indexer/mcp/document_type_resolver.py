"""Turning whatever word a caller used into a DocumentType."""

from __future__ import annotations

from workspace_indexer.mcp.unknown_document_type_error import UnknownDocumentTypeError
from workspace_indexer.models import DocumentType

# Words a model plausibly reaches for, mapped to the category we actually use.
# Near-misses should just work: the agent is guessing at our vocabulary from a
# one-line tool description, and being right about the *intent* while wrong
# about the *word* is the common case, not an edge case.
ALIASES: dict[str, DocumentType] = {
    "spec": DocumentType.NORMATIVE,
    "specification": DocumentType.NORMATIVE,
    "standard": DocumentType.NORMATIVE,
    "convention": DocumentType.NORMATIVE,
    "conventions": DocumentType.NORMATIVE,
    "adr": DocumentType.NORMATIVE,
    "rule": DocumentType.NORMATIVE,
    "rules": DocumentType.NORMATIVE,
    "policy": DocumentType.NORMATIVE,
    "requirement": DocumentType.NORMATIVE,
    "requirements": DocumentType.NORMATIVE,
    "architecture": DocumentType.DESIGN,
    "rfc": DocumentType.DESIGN,
    "proposal": DocumentType.DESIGN,
    "plan": DocumentType.DESIGN,
    "readme": DocumentType.GUIDE,
    "tutorial": DocumentType.GUIDE,
    "runbook": DocumentType.GUIDE,
    "howto": DocumentType.GUIDE,
    "docs": DocumentType.REFERENCE,
    "api": DocumentType.REFERENCE,
    "changelog": DocumentType.RECORD,
    "history": DocumentType.RECORD,
    "postmortem": DocumentType.RECORD,
    "code": DocumentType.IMPLEMENTATION,
    "source": DocumentType.IMPLEMENTATION,
    "impl": DocumentType.IMPLEMENTATION,
    "tests": DocumentType.TEST,
    "fixture": DocumentType.TEST,
    "lockfile": DocumentType.GENERATED,
}


class DocumentTypeResolver:
    """Maps a caller-supplied string to a DocumentType, or fails loudly.

    Deliberately not a `try: DocumentType(x) except: None`. Returning None on a
    bad input pushes the decision to a caller that will almost certainly turn
    it back into "no filter" or "match nothing", which is exactly the failure
    this class exists to prevent.
    """

    def resolve(self, given: str) -> DocumentType:
        key = given.strip().lower().replace("-", "_")
        try:
            return DocumentType(key)
        except ValueError:
            pass
        if key in ALIASES:
            return ALIASES[key]
        raise UnknownDocumentTypeError(
            given,
            valid=[t.value for t in DocumentType],
            aliases=sorted(ALIASES),
        )

    def resolve_all(self, given: list[str]) -> list[DocumentType]:
        return [self.resolve(item) for item in given]
