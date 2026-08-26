"""What role a document plays.

A second axis to FileKind. FileKind answers "how do I chunk this" and picks a
strategy; this answers "what is this for" and steers retrieval. Two markdown
files can be an architecture decision record and a changelog: identical
chunking, opposite usefulness to an agent asking how something should be built.

Deliberately small. Finer categories make both rules and models unreliable, and
make it impossible to write an eval set a human can agree with.
"""

from __future__ import annotations

from enum import StrEnum


class DocumentType(StrEnum):
    # How it *must* be built: specs, standards, ADRs, conventions.
    NORMATIVE = "normative"
    # How it is shaped, and why: architecture docs, RFCs, design proposals.
    DESIGN = "design"
    # How to use or operate it: README, tutorials, runbooks.
    GUIDE = "guide"
    # What exists: API documentation, generated reference.
    REFERENCE = "reference"
    # What happened: changelogs, postmortems, meeting notes.
    RECORD = "record"
    # The code itself.
    IMPLEMENTATION = "implementation"
    # Verification: tests, fixtures, snapshots.
    TEST = "test"
    # Machine-written.
    GENERATED = "generated"
    # Rules could not decide. Not a category so much as an admission, and the
    # input to the next rung of the classifier chain.
    UNKNOWN = "unknown"

    @property
    def definition(self) -> str:
        """One line a model can act on, not a dictionary gloss.

        Lives on the member rather than in a lookup table beside it, because a
        table can go stale the moment a category is added. The MCP taxonomy
        serves these verbatim: they are the only description of a category an
        agent ever sees before deciding to filter on it.
        """
        return _DEFINITIONS[self]


_DEFINITIONS: dict[DocumentType, str] = {
    DocumentType.NORMATIVE: (
        "Specifies how things must be built: specifications, standards, "
        "architecture decision records, coding conventions. Read these before "
        "writing new code -- they are the rules, not a description of what exists."
    ),
    DocumentType.DESIGN: (
        "Explains how a system is shaped and why: architecture documents, RFCs, "
        "design proposals, plans. Read these for intent and trade-offs."
    ),
    DocumentType.GUIDE: (
        "Explains how to use or operate something: READMEs, tutorials, runbooks, "
        "installation and deployment instructions."
    ),
    DocumentType.REFERENCE: (
        "Describes what exists, exhaustively: API documentation, generated "
        "reference material, configuration schemas."
    ),
    DocumentType.RECORD: (
        "Records what happened: changelogs, release notes, postmortems, meeting "
        "notes. Historical -- rarely what you want when deciding how to build "
        "something now."
    ),
    DocumentType.IMPLEMENTATION: (
        "Source code that does the work, as opposed to code that verifies it."
    ),
    DocumentType.TEST: (
        "Verifies behaviour: test files, fixtures, snapshots. Useful as worked "
        "examples of an API, misleading as a model of how that API is written."
    ),
    DocumentType.GENERATED: (
        "Machine-written and not edited by hand: lockfiles, build output, "
        "generated clients. Changing these by hand is almost always a mistake."
    ),
    DocumentType.UNKNOWN: (
        "The classifier could not decide. Not a category so much as an "
        "admission; these documents are still fully searchable."
    ),
}
