"""The universal fallback: paragraph packing with overlap."""

from __future__ import annotations

from collections.abc import Iterator

from workspace_indexer.chunking.block_splitter import pack_blocks, split_into_blocks
from workspace_indexer.chunking.chunk_factory import build_chunk
from workspace_indexer.chunking.context_header import header_token_cost
from workspace_indexer.chunking.token_estimate import estimate_tokens
from workspace_indexer.config import ChunkingSection
from workspace_indexer.models import Chunk, FileKind, SourceFile
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.chunking.text")


# Formats whose top-level entries start at column zero. Splitting on that
# gives YAML a `nav:` boundary where the prose splitter sees one paragraph the
# length of the file -- the shape that actually overflowed in practice.
_STRUCTURED = frozenset({"yaml", "toml", "ini", "cfg", "properties"})


class TextChunker:
    name = "text"
    # Bumped: structured files split differently, so their existing chunks are
    # stale and must be re-cut rather than trusted.
    version = 2
    kinds = frozenset({FileKind.TEXT, FileKind.PDF})

    def __init__(self, workspace: str) -> None:
        self._workspace = workspace

    def chunk(
        self,
        file: SourceFile,
        config: ChunkingSection,
        *,
        parse_degraded: bool = False,
    ) -> Iterator[Chunk]:
        """`parse_degraded` is set when this runs as the code chunker's
        fallback, so the payload records that the chunks are not symbol-aware."""
        if not file.text:
            return

        settings = config.text
        # max_tokens applies to what we embed, which is header + source.
        budget = max(1, settings.max_tokens - header_token_cost(file, file.kind))
        blocks = split_into_blocks(
            file.text,
            respect_fences=False,
            structured=(file.language or "") in _STRUCTURED,
        )
        groups = pack_blocks(
            blocks,
            max_tokens=budget,
            kind=file.kind,
            overlap=settings.overlap_paragraphs,
        )
        total = len(groups)
        for index, group in enumerate(groups):
            # Anything still over budget here has survived paragraph packing,
            # structural boundaries and a line-by-line cut, so it genuinely
            # cannot be made smaller.
            oversized = estimate_tokens(group.text, file.kind) > budget
            if oversized:
                log.info(
                    "chunk.indivisible_block",
                    rel_path=file.rel_path,
                    tokens=estimate_tokens(group.text, file.kind),
                    budget=budget,
                    lines=f"{group.start_line}-{group.end_line}",
                    detail="one indivisible block exceeds the chunk budget; it is "
                    "indexed whole and may be truncated by the embedding provider. "
                    "This is the documented tradeoff, not a chunker defect.",
                )
            yield build_chunk(
                file,
                self._workspace,
                source_text=group.text,
                start_line=group.start_line,
                end_line=group.end_line,
                chunker=self.name,
                version=self.version,
                chunk_index=index,
                chunk_total=total,
                parse_degraded=parse_degraded,
                indivisible=oversized,
            )
