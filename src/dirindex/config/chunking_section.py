"""The `chunking:` block."""

from __future__ import annotations

from pydantic import Field

from dirindex.config.code_chunking import CodeChunking
from dirindex.config.markdown_chunking import MarkdownChunking
from dirindex.config.opaque_chunking import OpaqueChunking
from dirindex.config.strict import Strict
from dirindex.config.text_chunking import TextChunking


class ChunkingSection(Strict):
    code: CodeChunking = Field(default_factory=CodeChunking)
    markdown: MarkdownChunking = Field(default_factory=MarkdownChunking)
    text: TextChunking = Field(default_factory=TextChunking)
    opaque: OpaqueChunking = Field(default_factory=OpaqueChunking)
    # Pin a specific extension to a specific chunker, e.g. {".mdx": "markdown"}
    overrides: dict[str, str] = Field(default_factory=dict)
