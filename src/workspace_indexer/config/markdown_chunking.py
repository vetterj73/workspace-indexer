"""Chunking settings for markdown."""

from __future__ import annotations

from workspace_indexer.config.strict import Strict


class MarkdownChunking(Strict):
    max_tokens: int = 512
    split_on_heading_depth: int = 3
