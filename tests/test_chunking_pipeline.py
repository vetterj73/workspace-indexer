"""Discovery -> read -> chunk, over the whole fixture workspace.

The unit tests cover each chunker against hand-written input. This one walks a
real tree and asserts the invariants that have to hold for *every* chunk,
whatever produced it — the checks that were originally a throwaway script.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import ConfigFactory
from workspace_indexer.chunking import ChunkerRegistry, read_source
from workspace_indexer.discovery import Walker
from workspace_indexer.models import Chunk, FileKind


def _index(config_for: ConfigFactory) -> list[Chunk]:
    config = config_for()
    registry = ChunkerRegistry(config.workspace.name)
    chunks: list[Chunk] = []
    for candidate in Walker(config).walk():
        source = read_source(candidate)
        if source is None:
            continue
        chunks.extend(registry.chunk(source, config.chunking))
    return chunks


def test_produces_chunks_across_kinds(config_for: ConfigFactory) -> None:
    kinds = {c.meta.kind for c in _index(config_for)}
    assert FileKind.CODE in kinds
    assert FileKind.MARKDOWN in kinds


def test_chunk_ids_are_globally_unique(config_for: ConfigFactory) -> None:
    """Ids are Qdrant point ids. A collision silently overwrites a chunk from
    a different file."""
    chunks = _index(config_for)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_every_line_span_is_valid_against_its_file(config_for: ConfigFactory) -> None:
    """The single most user-visible property: file:line has to land on real
    lines of the real file."""
    for chunk in _index(config_for):
        total = len(Path(chunk.meta.abs_path).read_text(encoding="utf-8").splitlines())
        assert 1 <= chunk.meta.start_line <= chunk.meta.end_line, chunk.meta.rel_path
        assert chunk.meta.end_line <= total, f"{chunk.meta.rel_path} {chunk.meta.end_line}>{total}"


def test_chunk_indices_are_consistent_per_file(config_for: ConfigFactory) -> None:
    by_file: dict[tuple[str, str | None], list[Chunk]] = {}
    for chunk in _index(config_for):
        by_file.setdefault((chunk.meta.rel_path, chunk.meta.symbol_path), []).append(chunk)
    for key, group in by_file.items():
        indices = sorted(c.meta.chunk_index for c in group)
        assert indices == list(range(len(group))) or len(group) == 1, key


def test_no_chunk_is_empty_or_whitespace(config_for: ConfigFactory) -> None:
    for chunk in _index(config_for):
        assert chunk.source_text.strip(), chunk.meta.rel_path


def test_embed_text_always_contains_the_source(config_for: ConfigFactory) -> None:
    for chunk in _index(config_for):
        assert chunk.source_text in chunk.embed_text


def test_binaries_and_images_contribute_nothing(config_for: ConfigFactory) -> None:
    paths = {c.meta.rel_path for c in _index(config_for)}
    assert "plain_folder/blob.so" not in paths
    assert "plain_folder/logo.png" not in paths


def test_gitignored_content_never_reaches_a_chunk(config_for: ConfigFactory) -> None:
    """The ignore rules matter most here: this is the last point where a secret
    could leak into an index that gets shipped to an embedding API."""
    chunks = _index(config_for)
    paths = {c.meta.rel_path for c in chunks}
    assert "repo_one/secret.txt" not in paths
    assert not any(p.startswith("repo_one/build/") for p in paths)
    assert not any("node_modules" in p for p in paths)
    assert not any("should never be indexed" in c.source_text for c in chunks)


def test_code_chunks_carry_symbols(config_for: ConfigFactory) -> None:
    code = [c for c in _index(config_for) if c.meta.kind is FileKind.CODE]
    assert code
    assert any(c.meta.symbol_path for c in code)


def test_markdown_chunks_carry_heading_trails(config_for: ConfigFactory) -> None:
    docs = [c for c in _index(config_for) if c.meta.kind is FileKind.MARKDOWN]
    assert docs
    assert any(c.meta.symbol_path and ">" in c.meta.symbol_path for c in docs)


def test_repo_provenance_is_attached_where_it_exists(config_for: ConfigFactory) -> None:
    chunks = _index(config_for)
    from_repo = [c for c in chunks if c.meta.unit == "repo_one"]
    from_plain = [c for c in chunks if c.meta.unit == "plain_folder"]
    assert from_repo and all(c.meta.repo is not None for c in from_repo)
    # Plain folders are indexed too; they simply have no provenance.
    assert from_plain and all(c.meta.repo is None for c in from_plain)


def test_hidden_claude_directory_is_indexed(config_for: ConfigFactory) -> None:
    """The primary reason discovery does not skip dotfiles."""
    paths = {c.meta.rel_path for c in _index(config_for)}
    assert ".claude/commands/deploy.md" in paths


def test_chunking_is_deterministic(config_for: ConfigFactory) -> None:
    """Re-chunking unchanged files must produce identical ids, or the manifest
    would see churn on every run and re-embed the whole workspace."""
    first = {c.chunk_id for c in _index(config_for)}
    second = {c.chunk_id for c in _index(config_for)}
    assert first == second
