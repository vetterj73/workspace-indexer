"""Cheap token estimation.

Chunking needs a token count for every candidate boundary, and the exact
tokenizer lives behind an async call on the embedding backend. Paying that per
boundary would dominate the cost of chunking, so we estimate here from bytes
and enforce the real limit once, at embed time, where `embed.truncated` warns
if an estimate was optimistic.

Divisors are bytes-per-token, measured against Voyage/OpenAI-family BPE
tokenizers: code packs fewer characters per token than prose because
identifiers, punctuation and indentation fragment.
"""

from __future__ import annotations

from workspace_indexer.models import FileKind

_DEFAULT_BYTES_PER_TOKEN = 3.7
_BYTES_PER_TOKEN: dict[FileKind, float] = {
    FileKind.CODE: 3.3,
    FileKind.MARKDOWN: 4.0,
    FileKind.TEXT: 4.0,
    FileKind.PDF: 4.0,
}


def bytes_per_token(kind: FileKind) -> float:
    return _BYTES_PER_TOKEN.get(kind, _DEFAULT_BYTES_PER_TOKEN)


def estimate_tokens(text: str, kind: FileKind) -> int:
    """Never returns 0 for non-empty text; a zero would make min_tokens
    filtering drop every chunk."""
    if not text:
        return 0
    size = len(text.encode("utf-8", "replace"))
    return max(1, round(size / bytes_per_token(kind)))


def tokens_to_bytes(tokens: int, kind: FileKind) -> int:
    """For handing a byte budget to a library that wants one.

    tree_sitter_language_pack's `chunk_max_size` is in bytes, not tokens — an
    easy thing to get wrong by a factor of ~3.
    """
    return max(1, int(tokens * bytes_per_token(kind)))
