"""One outgoing edge: something this file imports."""

from __future__ import annotations

from pydantic import BaseModel, computed_field


class Dependency(BaseModel):
    """What a file imports, and the file at the other end when we know it.

    Asymmetric with `Dependent` on purpose. Outgoing edges can point outside
    the index -- a package, a stdlib module, a tsconfig alias we cannot follow
    -- so the target is optional and `resolved` is a first-class field. An
    incoming edge is always a file we have indexed, because that is the only
    way it could have been recorded.
    """

    # Exactly what the source wrote: "workspace_indexer.models", "./sibling",
    # "System.Text". Kept even when resolved, because it is what the agent will
    # see if it opens the file, and what it must edit to change the edge.
    module: str
    # Line of the import statement *in the file being asked about*.
    line: int
    # The indexed file this names, or None when it points outside the index.
    rel_path: str | None = None
    # Of the target. None when unresolved -- deliberately not "unknown", which
    # is a real DocumentType meaning "the classifier could not decide".
    doc_type: str | None = None
    language: str | None = None

    @computed_field
    @property
    def resolved(self) -> bool:
        """False means "points outside the index", never "no such import".

        Spelled out as its own field rather than left implicit in a null
        rel_path, because the two readings lead an agent to opposite next
        moves: chase the file, or go and read the package.
        """
        return self.rel_path is not None
