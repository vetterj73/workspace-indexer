"""The plain-text fallback chunker."""

from __future__ import annotations

from tests.conftest import make_source
from workspace_indexer.chunking.text_chunker import TextChunker
from workspace_indexer.config import ChunkingSection, TextChunking
from workspace_indexer.models import FileKind


def _config(**overrides: object) -> ChunkingSection:
    return ChunkingSection(text=TextChunking(**overrides))  # type: ignore[arg-type]


def _prose(paragraphs: int, words: int = 120) -> str:
    return "\n\n".join(f"para{i} " + "word " * words for i in range(paragraphs))


def test_small_file_is_one_chunk() -> None:
    file = make_source("Just a short note.\n", kind=FileKind.TEXT, language=None)
    chunks = list(TextChunker("ws").chunk(file, _config()))
    assert len(chunks) == 1
    assert chunks[0].source_text == "Just a short note."
    assert chunks[0].meta.chunk_total == 1


def test_empty_text_yields_nothing() -> None:
    file = make_source("", kind=FileKind.TEXT, language=None)
    assert list(TextChunker("ws").chunk(file, _config())) == []


def test_large_file_splits_and_reports_consistent_totals() -> None:
    file = make_source(_prose(8), kind=FileKind.TEXT, language=None)
    chunks = list(TextChunker("ws").chunk(file, _config(max_tokens=120)))
    assert len(chunks) > 1
    assert {c.meta.chunk_total for c in chunks} == {len(chunks)}
    assert [c.meta.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunker_identity_is_recorded() -> None:
    """The manifest keys invalidation off chunker and chunker_version."""
    file = make_source(_prose(2), kind=FileKind.TEXT, language=None)
    chunk = next(iter(TextChunker("ws").chunk(file, _config())))
    assert chunk.meta.chunker == "text"
    assert chunk.meta.chunker_version == TextChunker.version


def test_line_numbers_point_at_real_lines() -> None:
    text = "alpha\n\nbeta\n\ngamma\n"
    file = make_source(text, kind=FileKind.TEXT, language=None)
    lines = text.splitlines()
    for chunk in TextChunker("ws").chunk(file, _config(max_tokens=1)):
        first = chunk.source_text.splitlines()[0]
        assert lines[chunk.meta.start_line - 1] == first


def test_metadata_is_copied_from_the_source_file() -> None:
    file = make_source("body\n", kind=FileKind.TEXT, language=None, rel_path="docs/n.txt")
    chunk = next(iter(TextChunker("ws").chunk(file, _config())))
    assert chunk.meta.workspace == "ws"
    assert chunk.meta.rel_path == "docs/n.txt"
    assert chunk.meta.unit == "repo_one"
    assert chunk.meta.kind is FileKind.TEXT


def test_embed_text_carries_the_header_and_source_does_not() -> None:
    file = make_source("body\n", kind=FileKind.TEXT, language=None)
    chunk = next(iter(TextChunker("ws").chunk(file, _config())))
    assert chunk.embed_text.startswith("# ")
    assert chunk.source_text == "body"
    assert chunk.embed_text.endswith("body")


def test_parse_degraded_flag_is_propagated() -> None:
    """Set when this runs as the code chunker's fallback, so the payload
    records that the chunks are not symbol-aware."""
    file = make_source("body\n", kind=FileKind.TEXT, language=None)
    chunker = TextChunker("ws")
    assert next(iter(chunker.chunk(file, _config()))).meta.parse_degraded is False
    degraded = next(iter(chunker.chunk(file, _config(), parse_degraded=True)))
    assert degraded.meta.parse_degraded is True


def test_overlap_keeps_a_boundary_passage_retrievable() -> None:
    file = make_source(_prose(6), kind=FileKind.TEXT, language=None)
    chunker = TextChunker("ws")
    without = list(chunker.chunk(file, _config(max_tokens=120, overlap_paragraphs=0)))
    with_overlap = list(chunker.chunk(file, _config(max_tokens=120, overlap_paragraphs=1)))
    assert sum(len(c.source_text) for c in with_overlap) > sum(len(c.source_text) for c in without)


def test_chunk_ids_are_unique_within_a_file() -> None:
    """Colliding ids would make one chunk silently overwrite another on
    upsert."""
    file = make_source(_prose(8), kind=FileKind.TEXT, language=None)
    chunks = list(TextChunker("ws").chunk(file, _config(max_tokens=120)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
