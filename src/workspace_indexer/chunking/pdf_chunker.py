"""One chunk per page, split further when a page is too long.

The page is the natural unit and the only anchor a PDF really has. There are no
line numbers to return -- nobody opens a PDF at line 340 -- so `symbol_path`
carries "page 12" and that is what a reader acts on. `start_line` and
`end_line` still describe the extracted text, so `location` stays well-formed
and monotonic, but the page is the useful half.

Headings are not used to split. A PDF's text layer has no heading structure,
only visual size, and inferring one from font metrics is a different project
with its own failure modes -- a wrong heading trail is worse than none, because
it looks authoritative.

The text is not read here. `read_source` extracts it, so the secret scanner
sees the whole document before any of it reaches this class; see
`discovery/pdf_text.py` for why that matters.
"""

from __future__ import annotations

from collections.abc import Iterator

from workspace_indexer.chunking.block import Block
from workspace_indexer.chunking.block_splitter import pack_blocks, split_into_blocks
from workspace_indexer.chunking.chunk_factory import build_chunk
from workspace_indexer.chunking.context_header import header_token_cost
from workspace_indexer.chunking.token_estimate import estimate_tokens
from workspace_indexer.config import ChunkingSection
from workspace_indexer.models import Chunk, FileKind, SourceFile
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.chunking.pdf")


class PdfChunker:
    name = "pdf"
    version = 1
    kinds = frozenset({FileKind.PDF})

    def __init__(self, workspace: str) -> None:
        self._workspace = workspace

    def chunk(self, file: SourceFile, config: ChunkingSection) -> Iterator[Chunk]:
        if not file.pages:
            # A PDF that reached this chunker with no pages was extracted and
            # found empty, which read_source already downgrades to OPAQUE --
            # so this is defensive rather than expected, and silence is right.
            return

        # PDFs have no dedicated config section: they are prose, and the
        # markdown budget is the prose budget. A separate knob would be one
        # more thing to tune with no evidence that they want different values.
        settings = config.markdown
        budget = max(1, settings.max_tokens - header_token_cost(file, file.kind))

        # Line offsets accumulate across pages so `location` is monotonic
        # through the document, matching how every other chunker numbers.
        line_offset = 1
        for number, page in enumerate(file.pages, start=1):
            page_lines = len(page.splitlines())
            yield from self._page(file, page, number, line_offset, budget)
            # +1 for the blank line `read_source` joins pages with, so offsets
            # here address the same text the secret scanner read.
            line_offset += page_lines + 1

        log.debug("chunk.pdf", pages=len(file.pages))

    def _page(
        self, file: SourceFile, page: str, number: int, offset: int, budget: int
    ) -> Iterator[Chunk]:
        label = f"page {number}"
        if estimate_tokens(page, file.kind) <= budget:
            yield build_chunk(
                file,
                self._workspace,
                source_text=page,
                start_line=offset,
                end_line=offset + max(0, len(page.splitlines()) - 1),
                chunker=self.name,
                version=self.version,
                symbol_path=label,
                symbol_kind="page",
                symbol_name=str(number),
            )
            return

        # A dense page -- a table, an appendix, a page of solid prose. Split on
        # paragraph boundaries and say which part it was, so a hit is still
        # locatable in the document rather than merely somewhere on page 12.
        blocks = _blocks_within(page, budget, file.kind)
        for block in blocks:
            block.start_line += offset - 1
            block.end_line += offset - 1
        groups = pack_blocks(blocks, max_tokens=budget, kind=file.kind)
        total = len(groups)
        for index, group in enumerate(groups):
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
                symbol_path=f"{label} ({index + 1}/{total})" if total > 1 else label,
                symbol_kind="page",
                symbol_name=str(number),
            )


def _blocks_within(page: str, budget: int, kind: FileKind) -> list[Block]:
    """Paragraph blocks, further split by line when one is over budget.

    `pack_blocks` deliberately leaves an oversized block whole -- correct for a
    markdown code fence, which must not be cut. It is wrong here, and the
    difference is that PDF extraction does not reliably preserve blank lines:
    a page of prose often comes back as a single paragraph, so blank-line
    splitting produces one block the size of the whole page.

    Left alone that becomes one chunk far over budget, which the embedder
    truncates -- losing most of the page while reporting success. Splitting on
    lines is safe for a PDF in a way it is not for markdown: the text layer has
    no atomic units, and a line is a real visual boundary in the document.

    Found by a test whose page would not split, not by reading this code.
    """
    blocks: list[Block] = []
    for block in split_into_blocks(page, respect_fences=False):
        if estimate_tokens(block.text, kind) <= budget:
            blocks.append(block)
            continue
        for index, line in enumerate(block.text.splitlines()):
            number = block.start_line + index
            blocks.append(Block(start_line=number, end_line=number, text=line))
    return blocks
