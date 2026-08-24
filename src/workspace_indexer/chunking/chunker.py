"""The chunking seam."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from workspace_indexer.config import ChunkingSection
from workspace_indexer.models import Chunk, FileKind, SourceFile


@runtime_checkable
class Chunker(Protocol):
    """One strategy per kind of file.

    `version` participates in the manifest's invalidation decision: bump it
    when a strategy changes and the next run re-chunks that kind, ignoring
    content hashes that would otherwise say "unchanged".
    """

    name: str
    version: int
    kinds: frozenset[FileKind]

    def chunk(self, file: SourceFile, config: ChunkingSection) -> Iterator[Chunk]: ...
