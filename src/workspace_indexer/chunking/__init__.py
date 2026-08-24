"""Chunking: one strategy per kind of file, resolved through a registry."""

from workspace_indexer.chunking.block import Block
from workspace_indexer.chunking.block_splitter import pack_blocks, split_into_blocks
from workspace_indexer.chunking.chunk_factory import build_chunk
from workspace_indexer.chunking.chunker import Chunker
from workspace_indexer.chunking.chunker_registry import ChunkerRegistry
from workspace_indexer.chunking.code_chunker import CodeChunker, prefetch_languages
from workspace_indexer.chunking.context_header import apply_header, build_header
from workspace_indexer.chunking.file_reader import read_source
from workspace_indexer.chunking.markdown_chunker import MarkdownChunker
from workspace_indexer.chunking.opaque_chunker import OpaqueChunker
from workspace_indexer.chunking.text_chunker import TextChunker
from workspace_indexer.chunking.token_estimate import estimate_tokens, tokens_to_bytes

__all__ = [
    "Block",
    "Chunker",
    "ChunkerRegistry",
    "CodeChunker",
    "MarkdownChunker",
    "OpaqueChunker",
    "TextChunker",
    "apply_header",
    "build_chunk",
    "build_header",
    "estimate_tokens",
    "pack_blocks",
    "read_source",
    "prefetch_languages",
    "split_into_blocks",
    "tokens_to_bytes",
]
