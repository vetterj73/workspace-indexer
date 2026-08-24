"""The synthetic header prepended to a chunk before embedding.

A method lifted out of its class is close to meaningless on its own: `def
upsert(self, chunks)` could belong to anything. Prefixing the repo, path and
enclosing symbol is what lets a dense retriever place it, and the reranker
benefits from exactly the same context. We embed the header and return the
untouched source, which is why Chunk carries both.
"""

from __future__ import annotations

from workspace_indexer.chunking.token_estimate import estimate_tokens
from workspace_indexer.models import FileKind, SourceFile


def build_header(file: SourceFile, symbol_path: str | None, symbol_kind: str | None) -> str:
    lines: list[str] = []

    if file.repo is not None:
        branch = f" ({file.repo.branch})" if file.repo.branch else ""
        lines.append(f"# repo: {file.repo.name}{branch}")
    elif file.root_label:
        # Not a repo, but the unit still tells you where in the workspace it is.
        lines.append(f"# location: {file.root_label}")

    lines.append(f"# file: {file.rel_path}")

    if file.language:
        lines.append(f"# language: {file.language}")

    if symbol_path:
        label = symbol_kind.lower() if symbol_kind else "symbol"
        lines.append(f"# {label}: {symbol_path}")

    return "\n".join(lines)


def header_token_cost(file: SourceFile, kind: FileKind) -> int:
    """Roughly what the header will add, so a chunker can reserve room for it.

    Budgeting max_tokens for source_text and then prepending a header means
    what we actually embed is over budget by the header's size. Harmless
    against a 32K-token API model; against a 512-token local one it is silent
    truncation of the end of every large chunk.

    Uses a worst-case symbol line, since the real symbol is not known until
    after the split has been decided.
    """
    probe = build_header(file, "Some.Reasonably.Long.Symbol.Path", "function")
    return estimate_tokens(probe, kind)


def apply_header(header: str, source_text: str) -> str:
    if not header:
        return source_text
    return f"{header}\n{source_text}"
