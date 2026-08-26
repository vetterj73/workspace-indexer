"""A file whose chunks are embedded but not yet written."""

from __future__ import annotations

from pydantic import BaseModel

from workspace_indexer.classification import Classification
from workspace_indexer.models import Chunk, SourceFile
from workspace_indexer.state import ChunkDelta


class PendingFile(BaseModel):
    """Work held back so embedding batches span files.

    Embedding one file at a time would mean one API request per file: a
    five-chunk module and a forty-thousand-file workspace is forty thousand
    round trips for work that batches naturally.
    """

    source: SourceFile
    chunker: str
    chunker_version: int
    chunks: list[Chunk]
    delta: ChunkDelta
    classification: Classification | None = None

    @property
    def to_embed(self) -> list[Chunk]:
        wanted = set(self.delta.to_upsert)
        return [chunk for chunk in self.chunks if chunk.chunk_id in wanted]
