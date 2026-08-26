"""One document type, as the taxonomy reports it."""

from __future__ import annotations

from pydantic import BaseModel


class TaxonomyEntry(BaseModel):
    name: str
    # How many chunks in this index carry the type. Present even when zero:
    # `normative: 0` tells an agent there is no written guidance here and it
    # should fall back to reading code. Omitting the category would leave it
    # unable to tell "none" from "I forgot to ask".
    count: int
    definition: str
    # Real paths from this workspace. A model calibrates far better on
    # `docs/adr/0007-event-sourcing.md` than on any prose definition.
    examples: list[str] = []
