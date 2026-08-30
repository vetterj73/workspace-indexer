"""Resolve a file to the chunker that should handle it."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from workspace_indexer.chunking.chunker import Chunker
from workspace_indexer.chunking.code_chunker import CodeChunker
from workspace_indexer.chunking.markdown_chunker import MarkdownChunker
from workspace_indexer.chunking.opaque_chunker import OpaqueChunker
from workspace_indexer.chunking.pdf_chunker import PdfChunker
from workspace_indexer.chunking.text_chunker import TextChunker
from workspace_indexer.config import ChunkingSection
from workspace_indexer.models import Chunk, FileKind, SourceFile
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.chunking.registry")


class ChunkerRegistry:
    """FileKind -> Chunker, with per-extension overrides from config.

    The registry is the extension point. PDF support was exactly that: one
    entry here plus one new class, plus the extraction the reader had to do
    before the scanner could see the text.
    """

    def __init__(self, workspace: str) -> None:
        text = TextChunker(workspace)
        self._by_name: dict[str, Chunker] = {
            "code": CodeChunker(workspace, fallback=text),
            "markdown": MarkdownChunker(workspace),
            "text": text,
            "opaque": OpaqueChunker(workspace),
            "pdf": PdfChunker(workspace),
        }
        self._by_kind: dict[FileKind, Chunker] = {
            FileKind.CODE: self._by_name["code"],
            FileKind.MARKDOWN: self._by_name["markdown"],
            FileKind.TEXT: text,
            # A PDF only reaches this chunker when read_source extracted a
            # text layer; without one it was downgraded to OPAQUE and never
            # arrives here.
            FileKind.PDF: self._by_name["pdf"],
            FileKind.IMAGE: self._by_name["opaque"],
            FileKind.OPAQUE: self._by_name["opaque"],
        }

    def names(self) -> list[str]:
        return sorted(self._by_name)

    def versions(self) -> dict[str, int]:
        """Chunker versions, for the manifest's invalidation decision."""
        return {name: chunker.version for name, chunker in self._by_name.items()}

    def resolve(self, file: SourceFile, config: ChunkingSection) -> Chunker:
        override = config.overrides.get(Path(file.rel_path).suffix.lower())
        if override is not None:
            chunker = self._by_name.get(override)
            if chunker is not None:
                return chunker
            log.warning(
                "chunk.unknown_override",
                override=override,
                known=self.names(),
                detail="falling back to the kind's default chunker",
            )
        return self._by_kind[file.kind]

    def chunk(self, file: SourceFile, config: ChunkingSection) -> Iterator[Chunk]:
        return self.resolve(file, config).chunk(file, config)
