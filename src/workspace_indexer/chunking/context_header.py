"""The synthetic header prepended to a chunk before embedding.

A method lifted out of its class is close to meaningless on its own: `def
upsert(self, chunks)` could belong to anything. Prefixing the repo, path and
enclosing symbol is what lets a dense retriever place it, and the reranker
benefits from exactly the same context. We embed the header and return the
untouched source, which is why Chunk carries both.
"""

from __future__ import annotations

from workspace_indexer.models import SourceFile


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


def apply_header(header: str, source_text: str) -> str:
    if not header:
        return source_text
    return f"{header}\n{source_text}"
