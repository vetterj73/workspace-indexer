"""One incoming edge: a file that imports the one being asked about."""

from __future__ import annotations

from pydantic import BaseModel, computed_field


class Dependent(BaseModel):
    """A file that imports the subject, anchored at the import statement.

    Always a file we indexed: an incoming edge only exists because we scanned
    the importer and resolved what it named. That is what makes this the
    half a per-project language server structurally cannot answer -- it spans
    every repository in the workspace rather than one project.
    """

    rel_path: str
    # Line of the import statement in the *importer*, so `location` below is
    # somewhere the agent can actually go.
    line: int
    # What the importer wrote to name the subject. Two importers can reach the
    # same file by different spellings -- "./helper" and "pkg.helper" -- and
    # which one they used is what you have to edit when the file moves.
    module: str
    doc_type: str
    language: str | None = None

    @computed_field
    @property
    def location(self) -> str:
        """`path:line`, so the next action is a Read with nothing to work out.

        The same contract as SearchResult.location, for the same reason.
        """
        return f"{self.rel_path}:{self.line}"
