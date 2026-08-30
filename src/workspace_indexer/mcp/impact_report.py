"""What changing one file would touch, in both directions."""

from __future__ import annotations

from pydantic import BaseModel

from workspace_indexer.graph.dependency import Dependency
from workspace_indexer.graph.dependent import Dependent


class ImpactReport(BaseModel):
    # The file we actually resolved the caller's path to. Echoed back because
    # the path is matched as a suffix: an agent that asked about "store.py"
    # needs to see which store.py it got.
    rel_path: str = ""
    root_label: str = ""
    doc_type: str = ""
    language: str | None = None

    # Outgoing: what this file imports.
    depends_on: list[Dependency] = []
    depends_on_total: int = 0
    # Incoming: what imports this file. The half that is expensive to get any
    # other way, and the reason this tool exists.
    used_by: list[Dependent] = []
    used_by_total: int = 0

    # Files that reach this one over HTTP, and endpoints this one calls.
    #
    # Kept apart from `used_by`/`depends_on` rather than merged into them,
    # because the relationship is different in a way that changes what an agent
    # should do. An importer breaks at compile time; a caller over HTTP breaks
    # at run time, in another repository, possibly deployed separately. Merging
    # them would hide exactly the distinction that makes the question worth
    # asking.
    called_by: list[Dependent] = []
    calls: list[Dependency] = []
    called_by_total: int = 0
    calls_total: int = 0

    # Every dependent counted by doc_type, over the whole result rather than
    # the page that fitted. This is the line an agent can act on without
    # reading the list: `{"test": 3, "implementation": 1}` means changing this
    # signature breaks one caller and three tests.
    used_by_by_type: dict[str, int] = {}

    # Edges omitted to stay inside `limit`, per direction. Reported rather
    # than silently swallowed, for the same reason SearchResponse reports it:
    # a truncated list that looks complete is how an agent concludes it has
    # seen every caller when it has seen the first ten.
    dropped_depends_on: int = 0
    dropped_used_by: int = 0

    # Filled when the path matched more than one indexed file, or none. The
    # report is then empty *because we did not guess*, which is a different
    # thing from the file having no edges -- hence `note` explaining which.
    candidates: list[str] = []

    # Never optional in practice. An empty graph result is ambiguous between
    # "nothing imports this" and "we cannot scan this language", and only the
    # note can tell them apart.
    note: str | None = None
