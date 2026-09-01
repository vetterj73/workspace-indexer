"""How much of a grounding source a repository actually has."""

from __future__ import annotations

from enum import StrEnum


class SourceStrength(StrEnum):
    """Three states, because two would hide the case that matters.

    The interesting answer is not "does this repository document itself" but
    "is there enough here to answer *why*". A binary present/absent collapses
    THIN into PRESENT and tells an agent it can expect an answer it will not
    get, which is the failure this whole report exists to prevent.
    """

    # Nothing. A query against this source will always return empty, and that
    # emptiness is a fact about the repository, not about the query.
    ABSENT = "absent"
    # Real but sparse. Retrieval will sometimes succeed and mostly will not,
    # and a miss says nothing about whether an answer exists elsewhere.
    THIN = "thin"
    # Enough that a miss is informative.
    PRESENT = "present"

    @property
    def definition(self) -> str:
        return _DEFINITIONS[self]


_DEFINITIONS: dict[SourceStrength, str] = {
    SourceStrength.ABSENT: "not present at all; queries against it cannot succeed",
    SourceStrength.THIN: "present but sparse; a miss does not mean no answer exists",
    SourceStrength.PRESENT: "well enough covered that a miss is itself informative",
}
