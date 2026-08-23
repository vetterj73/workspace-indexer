"""Chunking settings for the plain-text fallback."""

from __future__ import annotations

from workspace_indexer.config.strict import Strict


class TextChunking(Strict):
    max_tokens: int = 512
    overlap_paragraphs: int = 1
