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
from workspace_indexer.chunking.declaration import Declaration
from workspace_indexer.chunking.declaration_scanner import DeclarationScanner
from workspace_indexer.chunking.text_chunker import TextChunker
from workspace_indexer.chunking.token_estimate import estimate_tokens, tokens_to_bytes
from workspace_indexer.config import ChunkingSection, CodeChunking
from workspace_indexer.models import Chunk, FileKind, SourceFile
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.chunking.code")

# Above this share of chunks carrying parse errors, stop trusting the parse.
#
# Chosen from a real ASP.NET repo, where the split is wide enough that the exact
# value hardly matters: razor 75%, scss 82% and sql 98% of chunks carry error
# nodes, while powershell sits at 17% and every other language at 0%. Half
# separates "this grammar is not really parsing this file" from "one construct
# tripped it".
ERROR_NODE_LIMIT = 0.5


def parse_is_unreliable(degraded: int, total: int) -> bool:
    """Whether enough of a file's chunks carry parse errors to stop trusting it.

    A separate function because it is the decision, and the decision is what is
    worth testing. Reproducing a grammar's failure in a synthetic fixture is
    chasing the grammar rather than our own rule -- and the files that actually
    trigger this are a client's, which cannot be committed to a public
    repository.
    """
    if total <= 0:
        return False
    return degraded / total >= ERROR_NODE_LIMIT


class CodeChunker:
    name = "code"
    # 2: arrow-function and class-property declarations are attributed, which
    #    changes symbol_path on JS/TS chunks and so must force a re-chunk.
    # 3: files whose parse is mostly error nodes fall back to text chunking.
    # 4: bicep and powershell declarations are attributed.
    version = 4
    kinds = frozenset({FileKind.CODE})

    def __init__(self, workspace: str, fallback: TextChunker) -> None:
        self._workspace = workspace
        self._declarations = DeclarationScanner()
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

        # Declarations the library does not report, with their line spans. In
        # JS/TS a function assigned to a const is the dominant idiom and comes
        # back with no symbol at all; the span also lets the JSX fragments a
        # large component splits into inherit its name.
        declared = self._declarations.scan(file.text, file.language)

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

        degraded = sum(1 for c in kept if c.metadata.has_error_nodes)
        if parse_is_unreliable(degraded, len(kept)):
            # The grammar parsed without raising and then produced fragments.
            # Razor, SCSS and SQL all do this: `process()` returns chunks that
            # begin inside an HTML attribute or halfway through a statement,
            # carry no symbol, and are worse than useless -- they dilute the
            # index with text nothing can match meaningfully.
            #
            # Paragraph packing is not clever, but it splits on blank lines
            # rather than mid-token, which is strictly better than a confident
            # parse that is wrong.
            log.info(
                "chunk.error_nodes",
                language=file.language,
                degraded_chunks=degraded,
                total_chunks=len(kept),
                detail="most of this file's chunks contain parse errors; "
                "falling back to text chunking",
            )
            yield from self._degrade(file, config, reason="error_nodes")
            return

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
                # Last resort before giving up: whichever missed declaration
                # this chunk shares the most lines with.
                enclosing = _best_match(
                    declared,
                    chunk.start_line + 1,
                    chunk.start_line + max(1, len(source_text.splitlines())),
                )
                symbol_name = enclosing.name if enclosing else None
                carried_name, carried_context = None, ()
                if enclosing is not None:
                    symbol_kinds.setdefault(enclosing.name, enclosing.kind)

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


def _best_match(declared: list[Declaration], start_line: int, end_line: int) -> Declaration | None:
    """The declaration sharing the most lines with this chunk.

    Overlap rather than containment, because both directions happen. A 400-line
    component splits into chunks that sit *inside* it; a small file becomes one
    chunk that *contains* its component. Requiring containment either way would
    leave half of them unnamed.

    Ties break toward the narrower declaration, so a chunk covering a callback
    inside a component is labelled with the callback -- both are true, and the
    narrower one tells a reader more.
    """
    best: tuple[int, int, Declaration] | None = None
    for declaration in declared:
        overlap = min(end_line, declaration.end_line) - max(start_line, declaration.start_line) + 1
        if overlap <= 0:
            continue
        width = declaration.end_line - declaration.start_line
        if best is None or (overlap, -width) > (best[0], -best[1]):
            best = (overlap, width, declaration)
    return best[2] if best else None


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
