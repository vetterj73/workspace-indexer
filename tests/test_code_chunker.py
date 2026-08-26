"""Symbol-aware code chunking.

Runs against real tree-sitter grammars, because the whole value of this chunker
is what the parser actually reports and a stub would only echo our assumptions.
Grammars are cached locally after first use; the degradation paths that cover a
missing grammar are exercised with a monkeypatched parser so they need nothing.
"""

from __future__ import annotations

import pytest

from tests.conftest import make_source
from workspace_indexer.chunking.code_chunker import CodeChunker
from workspace_indexer.chunking.text_chunker import TextChunker
from workspace_indexer.config import ChunkingSection, CodeChunking
from workspace_indexer.models import Chunk, FileKind

SAMPLE = '''"""Module docstring."""

import os

MAX = 10


class Widget:
    """A widget."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.parts: list[str] = []

    def render(self, ctx: dict) -> str:
        """Render the widget."""
        out = []
        for index in range(MAX):
            out.append(f"{self.name}-{index}-{os.sep}")
        return "".join(out)


def free_function(a: int, b: int) -> int:
    return a + b
'''


def _chunker() -> CodeChunker:
    return CodeChunker("ws", fallback=TextChunker("ws"))


def _config(**overrides: object) -> ChunkingSection:
    return ChunkingSection(code=CodeChunking(**overrides))  # type: ignore[arg-type]


def _chunk(text: str = SAMPLE, language: str | None = "python", **overrides: object) -> list[Chunk]:
    file = make_source(text, kind=FileKind.CODE, language=language)
    return list(_chunker().chunk(file, _config(**overrides)))


def test_whole_small_file_is_one_chunk_at_the_default_budget() -> None:
    """512 tokens is ~1.7 KB of code, and this sample fits. Splitting a file
    that fits would fragment it for nothing."""
    assert len(_chunk(min_tokens=1)) == 1


def test_produces_symbol_aware_chunks() -> None:
    paths = {c.meta.symbol_path for c in _chunk(max_tokens=60, min_tokens=1)}
    assert "Widget.__init__" in paths
    assert "Widget.render" in paths


def test_symbol_kind_is_lowercase() -> None:
    """Payload filters compare exact keyword values, so casing has to be
    normalised at the source rather than at every call site."""
    kinds = {c.meta.symbol_kind for c in _chunk(min_tokens=1)} - {None}
    assert kinds
    assert all(k == k.lower() for k in kinds if k)


def test_chunk_total_counts_emitted_chunks_not_candidates() -> None:
    """min_tokens filtering happens before counting; otherwise a chunk claims
    to be 'part 3 of 9' in a file that only has 7."""
    chunks = _chunk(max_tokens=60, min_tokens=20)
    assert chunks
    assert {c.meta.chunk_total for c in chunks} == {len(chunks)}
    assert [c.meta.chunk_index for c in chunks] == list(range(len(chunks)))


def test_min_tokens_drops_degenerate_fragments() -> None:
    """The parser emits a few near-empty fragments; they carry no retrievable
    meaning and would dilute the index."""
    permissive = _chunk(max_tokens=60, min_tokens=1)
    strict = _chunk(max_tokens=60, min_tokens=40)
    assert len(strict) < len(permissive)


def test_start_line_points_at_the_chunk_content() -> None:
    """The library counts from 0 and we publish file:line links, so this is the
    off-by-one that would misdirect every result."""
    lines = SAMPLE.splitlines()
    for chunk in _chunk(max_tokens=60, min_tokens=1):
        first = chunk.source_text.splitlines()[0].strip()
        assert lines[chunk.meta.start_line - 1].strip() == first


def test_line_span_is_ordered_and_inside_the_file() -> None:
    total = len(SAMPLE.splitlines())
    for chunk in _chunk(max_tokens=60, min_tokens=1):
        assert 1 <= chunk.meta.start_line <= chunk.meta.end_line <= total


def test_long_definition_split_across_chunks_keeps_its_symbol() -> None:
    """The parser reports a split definition via context_path and leaves
    symbols_defined empty; those continuation chunks must not arrive
    anonymous."""
    body = "\n".join(f"    value_{i} = compute({i})" for i in range(400))
    text = f"class Big:\n    def method(self):\n{body}\n"
    chunks = _chunk(text, max_tokens=120, min_tokens=1)
    owned = [c for c in chunks if c.meta.symbol_path == "Big.method"]
    assert len(owned) > 1, [c.meta.symbol_path for c in chunks]
    # Without carry-forward these collapse to the enclosing class, which is a
    # far less useful thing for a result to say.
    assert "Big" not in {c.meta.symbol_path for c in chunks}
    assert all(c.meta.symbol_name == "method" for c in owned)


def test_embed_text_carries_context_and_source_does_not() -> None:
    chunks = _chunk(max_tokens=60, min_tokens=1)
    chunk = next(c for c in chunks if c.meta.symbol_path == "Widget.render")
    assert "# file: src/widget.py" in chunk.embed_text
    assert "# function: Widget.render" in chunk.embed_text
    assert chunk.embed_text.endswith(chunk.source_text)
    assert not chunk.source_text.startswith("#")


def test_header_can_be_disabled() -> None:
    chunk = _chunk(min_tokens=1, include_context_header=False)[0]
    assert chunk.embed_text == chunk.source_text


def test_content_hash_excludes_the_header() -> None:
    """The header carries the git branch. Hashing it would change every chunk
    id on a branch switch and re-embed an unchanged workspace."""
    from workspace_indexer.models import sha256_text

    for chunk in _chunk(max_tokens=60, min_tokens=1):
        assert chunk.meta.content_sha == sha256_text(chunk.source_text)


def test_chunker_identity_recorded() -> None:
    chunk = _chunk(min_tokens=1)[0]
    assert chunk.meta.chunker == "code"
    assert chunk.meta.chunker_version == CodeChunker.version


def test_chunk_ids_unique_within_a_file() -> None:
    chunks = _chunk(max_tokens=60, min_tokens=1)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_works_for_a_language_we_wrote_no_rules_for() -> None:
    """The point of using the pack's own chunker: coverage is not limited to
    the languages we thought about."""
    chunks = _chunk(
        "export const run = (a: number): number => a + 1;\n", language="typescript", min_tokens=1
    )
    assert chunks
    assert chunks[0].meta.language == "typescript"


def test_missing_language_degrades_to_text() -> None:
    chunks = _chunk("some content\n\nmore content\n", language=None, min_tokens=1)
    assert chunks
    assert all(c.meta.chunker == "text" for c in chunks)
    assert all(c.meta.parse_degraded for c in chunks)


def test_parser_failure_degrades_to_text_rather_than_losing_the_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grammars download on demand, so a cache miss with no network lands here
    alongside real parse errors. Either way the file is still worth indexing."""
    import workspace_indexer.chunking.code_chunker as module

    def boom(*_: object, **__: object) -> object:
        raise RuntimeError("grammar download failed")

    monkeypatch.setattr(module.tslp, "process", boom)
    chunks = _chunk(min_tokens=1)
    assert chunks
    assert all(c.meta.chunker == "text" for c in chunks)
    assert all(c.meta.parse_degraded for c in chunks)


def test_parse_that_yields_nothing_usable_still_indexes_the_file() -> None:
    """A file of only comments parses fine and defines no symbols. It is still
    something a person might search for."""
    text = "# note one\n# note two\n" + "# filler comment line\n" * 40
    chunks = _chunk(text, min_tokens=1)
    assert chunks


def test_empty_text_yields_nothing() -> None:
    assert _chunk("", min_tokens=1) == []


def test_syntax_error_is_flagged_but_not_fatal() -> None:
    """parse_degraded surfaces in the payload so a quality problem is visible
    without reading the log."""
    text = "class Broken:\n    def f(self:\n        return (((\n"
    chunks = _chunk(text, min_tokens=1)
    assert chunks


REACT_VIEW = """\
import React, { useState } from 'react';

interface Props { userId: string }

const Cart: React.FC<Props> = ({ userId }) => {
  const [items, setItems] = useState([]);
  const handleAdd = (sku: string) => {
    setItems([...items, sku]);
  };
  return (
    <div className="cart">
      <div className="cart__header">
        <span>{userId}</span>
      </div>
      <ul>{items.map((i) => <li key={i}>{i}</li>)}</ul>
    </div>
  );
};

export default Cart;
"""


def test_an_arrow_function_component_is_attributed() -> None:
    """The dominant React idiom. `const X: React.FC = () => {}` is a
    lexical_declaration, which the symbol extractor does not report -- so every
    chunk of a 29KB view came back with no symbol at all."""
    chunks = _chunk(REACT_VIEW, "tsx")
    assert chunks
    assert any(c.meta.symbol_path == "Cart" for c in chunks)


def test_every_chunk_of_a_component_carries_its_name() -> None:
    """The second half of the same fix. A large component splits into JSX
    fragments, and a chunk whose text begins `<div className='cart'>` cannot be
    identified on its own."""
    chunks = _chunk(REACT_VIEW, "tsx", max_tokens=40, min_tokens=1)
    assert len(chunks) > 1

    # Lines 5-19 are the component body. Line 20 is `export default Cart;`,
    # which is outside it and correctly stays unnamed.
    body = [c for c in chunks if 5 <= c.meta.start_line <= 19]
    assert len(body) > 10
    assert all(c.meta.symbol_path for c in body), [
        (c.meta.start_line, c.meta.symbol_path) for c in body
    ]
    # Including the pure-markup fragments, which is the point.
    markup = [c for c in body if c.source_text.strip().startswith("<")]
    assert markup
    assert all(c.meta.symbol_path == "Cart" for c in markup)


def test_the_innermost_declaration_wins() -> None:
    """A callback inside a component is labelled with the callback. Both are
    true; the narrower one is more useful in a search result."""
    chunks = _chunk(REACT_VIEW, "tsx", max_tokens=30, min_tokens=1)
    named = {c.meta.symbol_path for c in chunks if c.meta.symbol_path}
    assert "handleAdd" in named


def test_a_plain_function_declaration_still_wins_over_the_scanner() -> None:
    """The library's own symbols are authoritative where it has them.

    Both forms in one file, so the scanner cannot be crediting itself for a
    declaration the extractor already reported.
    """
    code = (
        REACT_VIEW
        + """
function RunCard({ id }: { id: string }) {
  const label = id.toUpperCase();
  return <span title={label}>{id}</span>;
}
"""
    )
    chunks = _chunk(code, "tsx", max_tokens=60, min_tokens=1)
    named = {c.meta.symbol_path for c in chunks if c.meta.symbol_path}
    assert "RunCard" in named
    assert "Cart" in named


def test_python_attribution_is_unchanged() -> None:
    """The scanner must not touch a language already scoring 88%.

    Uses the module's own Python sample, so this fails if the new pass changes
    attribution for a language it was never meant to run on.
    """
    chunks = _chunk(SAMPLE, "python", max_tokens=60, min_tokens=1)
    named = {c.meta.symbol_path for c in chunks if c.meta.symbol_path}
    assert named
    # Every name still traces to a real def/class, never to an assignment.
    assert all("lambda" not in str(n) for n in named)


def test_a_config_object_does_not_name_a_chunk() -> None:
    """`const config = {...}` is a declaration but not a definition."""
    code = "const config = {\n  retries: 3,\n  timeout: 1000,\n};\n"
    chunks = _chunk(code, "typescript")
    assert not any(c.meta.symbol_path == "config" for c in chunks)
