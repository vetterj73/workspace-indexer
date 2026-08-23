"""Chunking settings for source code."""

from __future__ import annotations

from dirindex.config.strict import Strict


class CodeChunking(Strict):
    max_tokens: int = 512
    min_tokens: int = 24
    include_context_header: bool = True
