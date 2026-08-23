"""Chunking settings for the plain-text fallback."""

from __future__ import annotations

from dirindex.config.strict import Strict


class TextChunking(Strict):
    max_tokens: int = 512
    overlap_paragraphs: int = 1
