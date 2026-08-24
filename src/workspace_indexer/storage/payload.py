"""Translating between a Chunk and a Qdrant point payload.

Kept in one place because the payload is a contract: the MCP server, the CLI and
the reranker all read these keys, and the payload indexes below have to name the
same fields the filters use.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any

from qdrant_client import models

from workspace_indexer.models import Chunk, EmbeddingSpace, FileKind, SearchHit

# Fields that get an explicit payload index. Without one Qdrant filters by
# scanning, and filtered search over a large index gets slow in a way that looks
# like a vector problem but is not.
INDEXED_FIELDS: dict[str, models.PayloadSchemaType] = {
    "root_label": models.PayloadSchemaType.KEYWORD,
    "unit": models.PayloadSchemaType.KEYWORD,
    "rel_path": models.PayloadSchemaType.KEYWORD,
    "ancestors": models.PayloadSchemaType.KEYWORD,
    "ext": models.PayloadSchemaType.KEYWORD,
    "kind": models.PayloadSchemaType.KEYWORD,
    "language": models.PayloadSchemaType.KEYWORD,
    "repo_name": models.PayloadSchemaType.KEYWORD,
    "repo_branch": models.PayloadSchemaType.KEYWORD,
    "symbol_kind": models.PayloadSchemaType.KEYWORD,
    "content_sha": models.PayloadSchemaType.KEYWORD,
}


def ancestors_of(rel_path: str) -> list[str]:
    """Every directory prefix of a path.

    Stored so a "search only under src/discovery" filter is an exact keyword
    match server-side. Qdrant cannot prefix-match a keyword, and filtering the
    returned page client-side would silently shrink the result set instead.
    """
    parts = PurePosixPath(rel_path).parts[:-1]
    return ["/".join(parts[: i + 1]) for i in range(len(parts))]


def to_payload(chunk: Chunk, space: EmbeddingSpace) -> dict[str, Any]:
    meta = chunk.meta
    repo = meta.repo
    return {
        "workspace": meta.workspace,
        "root_label": meta.root_label,
        "unit": meta.unit,
        "rel_path": meta.rel_path,
        "abs_path": str(meta.abs_path),
        "file_name": PurePosixPath(meta.rel_path).name,
        "ext": PurePosixPath(meta.rel_path).suffix.lower(),
        "ancestors": ancestors_of(meta.rel_path),
        "kind": meta.kind.value,
        "language": meta.language,
        "is_repo": repo is not None,
        "repo_name": repo.name if repo else None,
        "repo_remote": repo.remote_url if repo else None,
        "repo_branch": repo.branch if repo else None,
        "repo_head_sha": repo.head_sha if repo else None,
        "symbol_path": meta.symbol_path,
        "symbol_kind": meta.symbol_kind,
        "symbol_name": meta.symbol_name,
        # The single most valuable field for an LLM consumer: it turns a hit
        # into somewhere you can actually go.
        "start_line": meta.start_line,
        "end_line": meta.end_line,
        "source_text": chunk.source_text,
        # Only the header, not the whole embed_text. embed_text is header +
        # source, so storing both would double the payload to hold a copy of
        # something we already have.
        "context_header": _header_of(chunk),
        "token_count": meta.token_estimate,
        "content_sha": meta.content_sha,
        "chunk_index": meta.chunk_index,
        "chunk_total": meta.chunk_total,
        "chunker": meta.chunker,
        "chunker_version": meta.chunker_version,
        "parse_degraded": meta.parse_degraded,
        "space_slug": space.slug(),
        "indexed_at": datetime.now(UTC).isoformat(),
    }


def _header_of(chunk: Chunk) -> str:
    if chunk.embed_text == chunk.source_text:
        return ""
    return chunk.embed_text[: -len(chunk.source_text)].rstrip("\n")


def to_search_hit(point_id: str, score: float, payload: dict[str, Any]) -> SearchHit:
    source_text = str(payload.get("source_text") or "")
    header = str(payload.get("context_header") or "")
    kind_raw = payload.get("kind")
    return SearchHit(
        chunk_id=point_id,
        score=score,
        rel_path=str(payload.get("rel_path") or ""),
        root_label=str(payload.get("root_label") or ""),
        unit=str(payload.get("unit") or ""),
        repo_name=_optional_str(payload.get("repo_name")),
        is_repo=bool(payload.get("is_repo")),
        kind=FileKind(kind_raw) if kind_raw else FileKind.TEXT,
        language=_optional_str(payload.get("language")),
        symbol_path=_optional_str(payload.get("symbol_path")),
        symbol_name=_optional_str(payload.get("symbol_name")),
        start_line=int(payload.get("start_line") or 1),
        end_line=int(payload.get("end_line") or 1),
        source_text=source_text,
        # Reconstructed rather than stored, which is exact because the header
        # was built by prepending it in the first place.
        embed_text=f"{header}\n{source_text}" if header else source_text,
        token_count=int(payload.get("token_count") or 0),
        content_sha=str(payload.get("content_sha") or ""),
        indexed_at=_optional_str(payload.get("indexed_at")),
    )


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)
