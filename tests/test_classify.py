"""Classification precedence.

Every case in the first test corresponds to a bug found while building this,
originally with a throwaway shell command. They live here now.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from workspace_indexer.discovery.classify import classify, is_lockfile
from workspace_indexer.models import FileKind


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        # An unknown extension is not evidence of binary content.
        ("weird.qqq", FileKind.TEXT),
        # `.env.example` is an env file, not a file of type "example".
        (".env.example", FileKind.CODE),
        ("config.json.template", FileKind.TEXT),
        # Line-delimited JSON and snapshot files are text.
        ("events.jsonl", FileKind.TEXT),
        ("layout.snap", FileKind.TEXT),
        # Genuinely binary.
        ("libfoo.so", FileKind.OPAQUE),
        ("model.safetensors", FileKind.OPAQUE),
    ],
)
def test_misclassified_as_binary_regressions(name: str, kind: FileKind) -> None:
    assert classify(Path(name))[0] is kind


def test_filename_beats_extension() -> None:
    """CMakeLists.txt is cmake, not a .txt document."""
    kind, language = classify(Path("CMakeLists.txt"))
    assert (kind, language) == (FileKind.CODE, "cmake")


def test_extensionless_build_files_get_a_language() -> None:
    assert classify(Path("Makefile")) == (FileKind.CODE, "make")
    assert classify(Path("Dockerfile")) == (FileKind.CODE, "dockerfile")


def test_modifier_suffix_recurses_only_once_per_layer() -> None:
    """A stacked modifier suffix still resolves to the real type."""
    assert classify(Path("settings.yaml.dist"))[0] is FileKind.TEXT


def test_markdown_and_images() -> None:
    assert classify(Path("README.md")) == (FileKind.MARKDOWN, "markdown")
    assert classify(Path("docs/guide.mdx"))[0] is FileKind.MARKDOWN
    assert classify(Path("logo.png")) == (FileKind.IMAGE, None)
    assert classify(Path("paper.pdf")) == (FileKind.PDF, None)


def test_structured_data_is_text_not_code() -> None:
    """Symbol-level chunking of a config file buys nothing over paragraphs."""
    assert classify(Path("pyproject.toml"))[0] is FileKind.TEXT
    assert classify(Path("tsconfig.json"))[0] is FileKind.TEXT


def test_code_detection() -> None:
    assert classify(Path("src/a/b.py")) == (FileKind.CODE, "python")
    assert classify(Path("app/main.ts")) == (FileKind.CODE, "typescript")
    assert classify(Path("lib/mod.rs")) == (FileKind.CODE, "rust")


@pytest.mark.parametrize(
    "name",
    ["package-lock.json", "poetry.lock", "Cargo.lock", "go.sum", "uv.lock", "anything.lock"],
)
def test_lockfiles_recognised(name: str) -> None:
    assert is_lockfile(Path(name))


def test_ordinary_files_are_not_lockfiles() -> None:
    assert not is_lockfile(Path("package.json"))
    assert not is_lockfile(Path("locked_down.py"))
