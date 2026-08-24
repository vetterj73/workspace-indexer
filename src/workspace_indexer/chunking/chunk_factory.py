"""Assemble a Chunk, so every chunker fills the metadata the same way."""

from __future__ import annotations

from workspace_indexer.chunking.context_header import apply_header, build_header
from workspace_indexer.chunking.token_estimate import estimate_tokens
from workspace_indexer.models import Chunk, ChunkMeta, SourceFile, sha256_text


def build_chunk(
    file: SourceFile,
    workspace: str,
    *,
    source_text: str,
    start_line: int,
    end_line: int,
    chunker: str,
    version: int,
    chunk_index: int = 0,
    chunk_total: int = 1,
    symbol_path: str | None = None,
    symbol_kind: str | None = None,
    symbol_name: str | None = None,
    parse_degraded: bool = False,
    include_header: bool = True,
) -> Chunk:
    header = build_header(file, symbol_path, symbol_kind) if include_header else ""
    meta = ChunkMeta(
        workspace=workspace,
        root_label=file.root_label,
        unit=file.unit,
        abs_path=file.abs_path,
        rel_path=file.rel_path,
        kind=file.kind,
        language=file.language,
        repo=file.repo,
        symbol_path=symbol_path,
        symbol_kind=symbol_kind,
        symbol_name=symbol_name,
        start_line=start_line,
        end_line=end_line,
        chunk_index=chunk_index,
        chunk_total=chunk_total,
        # Hash the source, never the header: the header carries the git branch,
        # so hashing it would change every chunk id on a branch switch and
        # trigger a full re-embed of an unchanged workspace.
        content_sha=sha256_text(source_text),
        token_estimate=estimate_tokens(source_text, file.kind),
        chunker=chunker,
        chunker_version=version,
        parse_degraded=parse_degraded,
    )
    return Chunk(
        meta=meta,
        source_text=source_text,
        embed_text=apply_header(header, source_text),
    )
