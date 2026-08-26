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
