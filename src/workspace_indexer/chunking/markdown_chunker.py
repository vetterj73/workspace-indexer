"""Heading-aware markdown chunking.

Headings are the document's own idea of where topics start, so they make better
boundaries than any token window. Each chunk carries its heading trail as the
symbol path, which is what lets a result say "Deployment Guide > Rollbacks >
Emergency rollback" instead of just a line range.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from workspace_indexer.chunking.block_splitter import pack_blocks, split_into_blocks
from workspace_indexer.chunking.chunk_factory import build_chunk
from workspace_indexer.chunking.context_header import header_token_cost
from workspace_indexer.chunking.token_estimate import estimate_tokens
from workspace_indexer.config import ChunkingSection
from workspace_indexer.models import Chunk, FileKind, SourceFile
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.chunking.markdown")

_HEADING = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s{0,3}(?P<fence>`{3,}|~{3,})\s*(?P<info>.*)$")


class MarkdownChunker:
    name = "markdown"
    version = 1
    kinds = frozenset({FileKind.MARKDOWN})

    def __init__(self, workspace: str) -> None:
        self._workspace = workspace

    def chunk(self, file: SourceFile, config: ChunkingSection) -> Iterator[Chunk]:
        if not file.text:
            return

        settings = config.markdown
        # max_tokens applies to what we embed, which is header + source.
        budget = max(1, settings.max_tokens - header_token_cost(file, file.kind))
        sections = self._sections(file.text, settings.split_on_heading_depth)

        for trail, start_line, body in sections:
            symbol_path = " > ".join(trail) if trail else None
            if estimate_tokens(body, file.kind) <= budget:
                yield build_chunk(
                    file,
                    self._workspace,
                    source_text=body,
                    start_line=start_line,
                    end_line=start_line + len(body.splitlines()) - 1,
                    chunker=self.name,
                    version=self.version,
                    symbol_path=symbol_path,
                    symbol_kind="heading" if trail else None,
                    symbol_name=trail[-1] if trail else None,
                )
                continue

            # Oversized section: split on block boundaries, repeating the
            # heading trail so each part still says where it came from.
            blocks = split_into_blocks(body, respect_fences=True)
            for block in blocks:
                block.start_line += start_line - 1
                block.end_line += start_line - 1
            groups = pack_blocks(blocks, max_tokens=budget, kind=file.kind)
            total = len(groups)
            for index, group in enumerate(groups):
                # A fenced block is deliberately never cut, so this is where
                # markdown's oversized chunks come from. Said once, at info,
                # rather than left for the provider to truncate in silence.
                oversized = estimate_tokens(group.text, file.kind) > budget
                if oversized:
                    log.info(
                        "chunk.indivisible_block",
                        rel_path=file.rel_path,
                        tokens=estimate_tokens(group.text, file.kind),
                        budget=budget,
                        lines=f"{group.start_line}-{group.end_line}",
                        detail="a fenced block exceeds the chunk budget and is kept "
                        "whole on purpose -- cutting a fence embeds badly and displays "
                        "worse. It may be truncated by the embedding provider.",
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
                    symbol_path=symbol_path,
                    symbol_kind="heading" if trail else None,
                    symbol_name=trail[-1] if trail else None,
                    indivisible=oversized,
                )

    @staticmethod
    def _sections(text: str, max_depth: int) -> list[tuple[list[str], int, str]]:
        """Split at headings no deeper than max_depth.

        Returns (heading trail, 1-based start line, section text). Headings
        inside fenced code are ignored — a `# comment` in a shell example is
        not a document heading, and treating it as one shatters the document.
        """
        lines = text.splitlines()
        sections: list[tuple[list[str], int, str]] = []

        def flush(body: list[str], trail_names: list[str], first_line: int) -> None:
            if body and any(item.strip() for item in body):
                sections.append((list(trail_names), first_line, "\n".join(body).strip("\n")))

        trail: list[tuple[int, str]] = []
        current: list[str] = []
        current_trail: list[str] = []
        start = 1

        fence_char = ""
        fence_len = 0

        for number, line in enumerate(lines, start=1):
            fence = _FENCE.match(line)
            if fence is not None:
                token = fence.group("fence")
                if not fence_char:
                    fence_char, fence_len = token[0], len(token)
                elif token[0] == fence_char and len(token) >= fence_len:
                    fence_char, fence_len = "", 0
                current.append(line)
                continue

            heading = _HEADING.match(line) if not fence_char else None
            if heading is not None:
                depth = len(heading.group("hashes"))
                if depth <= max_depth:
                    flush(current, current_trail, start)
                    trail = [(d, t) for d, t in trail if d < depth]
                    trail.append((depth, heading.group("title")))
                    current_trail = [title for _, title in trail]
                    current = [line]
                    start = number
                    continue
                # Deeper than the split depth: keep it inside the section body,
                # but it still refines the trail for anything that follows.
            current.append(line)

        flush(current, current_trail, start)
        return sections
