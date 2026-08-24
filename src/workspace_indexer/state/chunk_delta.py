"""What a file's chunks changed into."""

from __future__ import annotations

from pydantic import BaseModel


class ChunkDelta(BaseModel):
    """The exact set of ids to write and to remove.

    Chunk ids are content-addressed, so editing one function in a
    forty-function file yields one id to add and one to remove rather than
    forty of each. This is where the incremental cost saving actually lands.
    """

    to_upsert: list[str] = []
    to_delete: list[str] = []
    unchanged: list[str] = []

    @property
    def is_noop(self) -> bool:
        return not self.to_upsert and not self.to_delete
