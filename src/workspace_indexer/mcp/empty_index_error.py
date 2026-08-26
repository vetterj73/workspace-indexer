"""Raised when the MCP server is asked to serve an index with nothing in it."""

from __future__ import annotations


class EmptyIndexError(RuntimeError):
    """An empty collection, caught at startup rather than at query time.

    A server over an empty index answers every question with "nothing matched",
    which an agent reads as "this workspace does not contain that" -- the exact
    failure this whole tool surface is designed to prevent, one layer down and
    applied to everything at once.

    It is nearly always a wiring problem rather than an empty workspace. The
    client launches the server from its own working directory, so a relative
    QDRANT_PATH resolves somewhere new and an unreachable `.env` drops the
    process into embedded mode against a directory that does not exist. All of
    which starts up perfectly and serves nothing.
    """

    def __init__(self, *, space: str, mode: str, detail: str) -> None:
        super().__init__(
            f"the collection for {space} is empty ({mode}), so every query would "
            f"return nothing.\n{detail}\n"
            "Run `workspace-indexer status` from the repository to confirm what is "
            "indexed, then check that this process resolves the same store: an MCP "
            "client launches it from its own working directory, so pass --config "
            "with an absolute path and set QDRANT_MODE, QDRANT_URL and the "
            "embedding credentials in the client's env block."
        )
