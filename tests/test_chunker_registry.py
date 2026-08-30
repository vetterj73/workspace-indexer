"""Resolving a file to its chunker.

The registry is the seam the plan promised: adding PDF support later should be
one entry plus one class, with nothing else touched.
"""

from __future__ import annotations

from tests.conftest import make_source
from workspace_indexer.chunking.chunker import Chunker
from workspace_indexer.chunking.chunker_registry import ChunkerRegistry
from workspace_indexer.config import ChunkingSection
from workspace_indexer.models import FileKind

MD = "# Title\n\nBody text.\n"
PY_SRC = "def f():\n    return 1\n"


def test_each_kind_resolves_to_its_strategy() -> None:
    registry = ChunkerRegistry("ws")
    config = ChunkingSection()
    expected = {
        FileKind.CODE: "code",
        FileKind.MARKDOWN: "markdown",
        FileKind.TEXT: "text",
        FileKind.PDF: "pdf",
        FileKind.IMAGE: "opaque",
        FileKind.OPAQUE: "opaque",
    }
    for kind, name in expected.items():
        file = make_source("body", kind=kind, language=None)
        assert registry.resolve(file, config).name == name, kind


def test_every_kind_is_covered() -> None:
    """A missing entry would be a KeyError at index time, on a real file."""
    registry = ChunkerRegistry("ws")
    config = ChunkingSection()
    for kind in FileKind:
        registry.resolve(make_source("body", kind=kind, language=None), config)


def test_extension_override_wins_over_the_kind_default() -> None:
    """`.mdx` classifies as markdown but a user may want it as plain text."""
    registry = ChunkerRegistry("ws")
    config = ChunkingSection(overrides={".mdx": "text"})
    file = make_source(MD, kind=FileKind.MARKDOWN, language="markdown", rel_path="d/page.mdx")
    assert registry.resolve(file, config).name == "text"


def test_override_is_case_insensitive_on_the_extension() -> None:
    registry = ChunkerRegistry("ws")
    config = ChunkingSection(overrides={".mdx": "text"})
    file = make_source(MD, kind=FileKind.MARKDOWN, language="markdown", rel_path="d/PAGE.MDX")
    assert registry.resolve(file, config).name == "text"


def test_unknown_override_falls_back_instead_of_crashing() -> None:
    """A typo in config should degrade to the default, not abort a long run."""
    registry = ChunkerRegistry("ws")
    config = ChunkingSection(overrides={".py": "treesitter"})
    file = make_source(PY_SRC, kind=FileKind.CODE, language="python")
    assert registry.resolve(file, config).name == "code"


def test_chunk_delegates_to_the_resolved_chunker() -> None:
    registry = ChunkerRegistry("ws")
    file = make_source(MD, kind=FileKind.MARKDOWN, language="markdown", rel_path="README.md")
    chunks = list(registry.chunk(file, ChunkingSection()))
    assert chunks
    assert all(c.meta.chunker == "markdown" for c in chunks)


def test_opaque_files_produce_no_chunks() -> None:
    """Recorded in the manifest, deliberately not embedded."""
    registry = ChunkerRegistry("ws")
    for kind in (FileKind.IMAGE, FileKind.OPAQUE):
        file = make_source("", kind=kind, language=None)
        assert list(registry.chunk(file, ChunkingSection())) == []


def test_unsupported_opaque_mode_does_not_raise() -> None:
    """Setting mode: multimodal should warn once, not abort indexing."""
    registry = ChunkerRegistry("ws")
    config = ChunkingSection.model_validate({"opaque": {"mode": "multimodal"}})
    file = make_source("", kind=FileKind.IMAGE, language=None)
    assert list(registry.chunk(file, config)) == []


def test_versions_are_exposed_for_manifest_invalidation() -> None:
    """Bumping a chunker's version is how a strategy change forces a re-chunk
    of that kind despite unchanged content hashes."""
    versions = ChunkerRegistry("ws").versions()
    assert set(versions) == {"code", "markdown", "text", "opaque", "pdf"}
    assert all(v >= 1 for v in versions.values())


def test_all_chunkers_satisfy_the_protocol() -> None:
    registry = ChunkerRegistry("ws")
    config = ChunkingSection()
    for kind in FileKind:
        chunker = registry.resolve(make_source("body", kind=kind, language=None), config)
        assert isinstance(chunker, Chunker)


def test_workspace_name_reaches_the_chunks() -> None:
    registry = ChunkerRegistry("labbox")
    file = make_source(PY_SRC, kind=FileKind.CODE, language="python")
    chunks = list(registry.chunk(file, ChunkingSection()))
    assert chunks
    assert all(c.meta.workspace == "labbox" for c in chunks)
