"""Symbol-aware code chunking via tree-sitter.

The heavy lifting is `tree_sitter_language_pack.process()`, which already
splits on definition boundaries, packs adjacent small definitions, and reports
the enclosing symbol trail and whether the parse hit error nodes. That covers
every grammar the pack ships rather than the handful we would have written
node-type tables for.

Two things the library's units differ from ours: `chunk_max_size` is in bytes,
not tokens, and its line numbers are 0-based.
"""

from __future__ import annotations

from collections.abc import Iterator

import tree_sitter_language_pack as tslp

from workspace_indexer.chunking.chunk_factory import build_chunk
from workspace_indexer.chunking.context_header import header_token_cost
from workspace_indexer.chunking.text_chunker import TextChunker
from workspace_indexer.chunking.token_estimate import estimate_tokens, tokens_to_bytes
from workspace_indexer.config import ChunkingSection, CodeChunking
from workspace_indexer.models import Chunk, FileKind, SourceFile
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.chunking.code")


class CodeChunker:
    name = "code"
    version = 1
    kinds = frozenset({FileKind.CODE})

    def __init__(self, workspace: str, fallback: TextChunker) -> None:
        self._workspace = workspace
        # Explicit rather than implicit: a file we cannot parse still gets
        # indexed, just without symbol awareness.
        self._fallback = fallback

    def chunk(self, file: SourceFile, config: ChunkingSection) -> Iterator[Chunk]:
        if not file.text:
            return
        if not file.language:
            yield from self._degrade(file, config, reason="no_language")
            return

        settings = config.code
        try:
            result = tslp.process(
                file.text,
                tslp.ProcessConfig(
                    language=file.language,
                    symbols=True,
                    structure=False,
                    imports=False,
                    exports=False,
                    comments=False,
                    docstrings=False,
                    diagnostics=False,
                    chunk_max_size=tokens_to_bytes(self._budget(file, settings), file.kind),
                ),
            )
        except Exception as exc:
            # Grammars download on demand, so a cache miss with no network
            # lands here alongside genuine parse failures. Either way the file
            # is still worth indexing.
            log.warning(
                "chunk.parse_failed",
                language=file.language,
                error=f"{type(exc).__name__}: {exc}",
            )
            yield from self._degrade(file, config, reason="parse_error")
            return

        symbol_kinds = {symbol.name: str(symbol.kind).lower() for symbol in result.symbols}

        # Filter before counting: chunk_total has to be the number of chunks we
        # actually emit, or "part 3 of 9" points at a 7-chunk file. process()
        # emits a few degenerate fragments (a bare identifier, a stray brace)
        # that carry no retrievable meaning and would dilute the index.
        kept = [
            c
            for c in result.chunks
            if c.content
            and c.content.strip()
            and estimate_tokens(c.content, file.kind) >= settings.min_tokens
        ]

        # A definition too large for one chunk is reported with its name on the
        # first chunk only; the continuations carry just the enclosing class in
        # context_path. Carrying the name forward while the context is
        # unchanged is what keeps the tail of a 400-line method labelled
        # `Big.method` instead of collapsing to `Big`.
        carried_name: str | None = None
        carried_context: tuple[str, ...] = ()

        for index, chunk in enumerate(kept):
            source_text = chunk.content
            names: list[str] = list(chunk.metadata.symbols_defined or [])
            trail: list[str] = list(chunk.metadata.context_path or [])
            context = tuple(trail)

            if names:
                symbol_name: str | None = names[0]
                carried_name, carried_context = names[0], context
            elif carried_name is not None and context == carried_context:
                symbol_name = carried_name
            elif trail:
                symbol_name = trail[-1]
                carried_name, carried_context = None, ()
            else:
                symbol_name = None
                carried_name, carried_context = None, ()

            if symbol_name is None:
                symbol_path = None
            elif trail and trail[-1] == symbol_name:
                symbol_path = ".".join(trail)
            else:
                symbol_path = ".".join([*trail, symbol_name])

            yield build_chunk(
                file,
                self._workspace,
                source_text=source_text,
                # The library counts lines from 0; editors and file:line links
                # count from 1. end_line is derived from the content rather
                # than the reported span, so it stays self-consistent whether
                # or not the library's end is inclusive.
                start_line=chunk.start_line + 1,
                end_line=chunk.start_line + max(1, len(source_text.splitlines())),
                chunker=self.name,
                version=self.version,
                chunk_index=index,
                chunk_total=len(kept),
                symbol_path=symbol_path,
                symbol_kind=symbol_kinds.get(symbol_name or ""),
                symbol_name=symbol_name,
                parse_degraded=bool(chunk.metadata.has_error_nodes),
                include_header=settings.include_context_header,
            )

        if not kept:
            # A grammar that parsed but produced nothing usable is still a
            # file someone may search for.
            log.debug("chunk.no_symbols", language=file.language)
            yield from self._degrade(file, config, reason="no_chunks")

    @staticmethod
    def _budget(file: SourceFile, settings: CodeChunking) -> int:
        """max_tokens applies to what we embed, which is header + source."""
        if not settings.include_context_header:
            return settings.max_tokens
        reserved = header_token_cost(file, file.kind)
        return max(settings.min_tokens, settings.max_tokens - reserved)

    def _degrade(
        self, file: SourceFile, config: ChunkingSection, *, reason: str
    ) -> Iterator[Chunk]:
        log.debug("chunk.degraded_to_text", reason=reason, language=file.language)
        yield from self._fallback.chunk(file, config, parse_degraded=True)


def prefetch_languages(languages: set[str]) -> None:
    """Warm the grammar cache once, before the walk.

    Grammars download on demand. Without this the first file of each language
    pays a network round trip in the middle of indexing, and a transient
    failure silently degrades that language to text chunking.
    """
    wanted = sorted(name for name in languages if name)
    if not wanted:
        return
    try:
        tslp.prefetch(wanted)
    except Exception as exc:
        log.warning(
            "chunk.prefetch_failed",
            languages=wanted,
            error=f"{type(exc).__name__}: {exc}",
            detail="affected languages will degrade to text chunking",
        )
    else:
        log.info("chunk.prefetch_ok", count=len(wanted))
