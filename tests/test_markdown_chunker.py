"""Heading-aware markdown chunking."""

from __future__ import annotations

from tests.conftest import make_source
from workspace_indexer.chunking.markdown_chunker import MarkdownChunker
from workspace_indexer.config import ChunkingSection, MarkdownChunking
from workspace_indexer.models import Chunk, FileKind

DOC = """# Deployment Guide

Intro paragraph about deploying.

## Rollbacks

To roll back, run the rollback script.

### Emergency rollback

Page the on-call engineer first.

## Monitoring

Watch the dashboard.
"""

WITH_FENCE = """# Guide

## Setup

Run this:

```bash
# This is a shell comment, not a heading
cd /srv

# Another comment
./deploy.sh
```

Done.
"""


def _config(**overrides: object) -> ChunkingSection:
    return ChunkingSection(markdown=MarkdownChunking(**overrides))  # type: ignore[arg-type]


def _chunk(text: str, **overrides: object) -> list[Chunk]:
    file = make_source(text, kind=FileKind.MARKDOWN, language="markdown", rel_path="README.md")
    return list(MarkdownChunker("ws").chunk(file, _config(**overrides)))


def test_splits_at_headings() -> None:
    paths = [c.meta.symbol_path for c in _chunk(DOC)]
    assert paths == [
        "Deployment Guide",
        "Deployment Guide > Rollbacks",
        "Deployment Guide > Rollbacks > Emergency rollback",
        "Deployment Guide > Monitoring",
    ]


def test_heading_trail_pops_back_out_on_a_shallower_heading() -> None:
    """`## Monitoring` must not inherit `### Emergency rollback`."""
    trails = {c.meta.symbol_path for c in _chunk(DOC)}
    assert "Deployment Guide > Monitoring" in trails
    assert not any("Monitoring" in t and "Emergency" in t for t in trails if t)


def test_symbol_name_is_the_deepest_heading() -> None:
    chunks = _chunk(DOC)
    assert chunks[2].meta.symbol_name == "Emergency rollback"
    assert chunks[2].meta.symbol_kind == "heading"


def test_deeper_headings_stay_inside_their_section_when_below_split_depth() -> None:
    chunks = _chunk(DOC, split_on_heading_depth=2)
    assert len(chunks) == 3
    emergency = [c for c in chunks if "Emergency rollback" in c.source_text]
    assert len(emergency) == 1
    assert emergency[0].meta.symbol_path == "Deployment Guide > Rollbacks"


def test_hash_comments_inside_a_fence_are_not_headings() -> None:
    """A `# comment` in a shell example would otherwise shatter the document."""
    chunks = _chunk(WITH_FENCE)
    assert [c.meta.symbol_path for c in chunks] == ["Guide", "Guide > Setup"]
    setup = chunks[1]
    assert "cd /srv" in setup.source_text
    assert "./deploy.sh" in setup.source_text


def test_section_text_includes_its_own_heading_line() -> None:
    chunks = _chunk(DOC)
    assert chunks[1].source_text.startswith("## Rollbacks")


def test_start_line_points_at_the_heading() -> None:
    lines = DOC.splitlines()
    for chunk in _chunk(DOC):
        assert lines[chunk.meta.start_line - 1].lstrip().startswith("#")


def test_oversized_section_splits_but_keeps_the_trail() -> None:
    body = "\n\n".join("word " * 200 for _ in range(6))
    text = f"# Top\n\n## Big\n\n{body}\n"
    chunks = _chunk(text, max_tokens=120)
    big = [c for c in chunks if c.meta.symbol_path == "Top > Big"]
    assert len(big) > 1
    assert {c.meta.chunk_total for c in big} == {len(big)}
    assert [c.meta.chunk_index for c in big] == list(range(len(big)))


def test_split_section_line_numbers_stay_within_the_file() -> None:
    """Part offsets are relative to the section, so they have to be rebased
    onto the file or every link after the first part is wrong."""
    body = "\n\n".join("word " * 200 for _ in range(6))
    text = f"# Top\n\n## Big\n\n{body}\n"
    total_lines = len(text.splitlines())
    for chunk in _chunk(text, max_tokens=120):
        assert 1 <= chunk.meta.start_line <= total_lines
        assert chunk.meta.start_line <= chunk.meta.end_line <= total_lines


def test_content_before_any_heading_is_kept() -> None:
    """A preamble above the first heading is real content, not throwaway."""
    chunks = _chunk("Orphan intro paragraph.\n\n# First\n\nBody.\n")
    assert chunks[0].meta.symbol_path is None
    assert "Orphan intro" in chunks[0].source_text


def test_document_with_no_headings_is_one_chunk() -> None:
    chunks = _chunk("Just prose.\n\nMore prose.\n")
    assert len(chunks) == 1
    assert chunks[0].meta.symbol_path is None


def test_empty_document_yields_nothing() -> None:
    assert _chunk("") == []
    assert _chunk("\n\n   \n") == []


def test_setext_style_hashes_in_body_do_not_break_parsing() -> None:
    """A line of `#` alone is not a heading match and must not crash."""
    chunks = _chunk("# Title\n\n#\n\nBody.\n")
    assert len(chunks) == 1


def test_chunk_ids_unique_within_a_document() -> None:
    chunks = _chunk(DOC)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_embed_text_includes_the_heading_trail() -> None:
    """The trail is the most useful retrieval signal a doc chunk has."""
    chunks = _chunk(DOC)
    assert "# heading: Deployment Guide > Rollbacks" in chunks[1].embed_text
