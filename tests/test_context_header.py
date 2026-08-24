"""The synthetic context header.

Everything here protects one property: the header goes into embed_text and
never into source_text or the content hash. Hashing the header would put the
git branch into every chunk id, so switching branches would re-embed an
otherwise unchanged workspace.
"""

from __future__ import annotations

from tests.conftest import make_source
from workspace_indexer.chunking.context_header import apply_header, build_header
from workspace_indexer.models import FileKind, RepoInfo

REPO = RepoInfo(name="workspace-indexer", branch="main", head_sha="a" * 40)


def test_header_names_repo_file_language_and_symbol() -> None:
    file = make_source("body", repo=REPO, rel_path="src/storage/qdrant.py")
    header = build_header(file, "QdrantStore.upsert", "Function")
    assert "# repo: workspace-indexer (main)" in header
    assert "# file: src/storage/qdrant.py" in header
    assert "# language: python" in header
    assert "# function: QdrantStore.upsert" in header


def test_non_repo_files_still_say_where_they_are() -> None:
    """A plain folder in the workspace has no provenance but still has a
    location, and dropping it would leave the chunk unplaceable."""
    file = make_source("body", repo=None, root_label="claude-config")
    header = build_header(file, None, None)
    assert "# location: claude-config" in header
    assert "repo:" not in header


def test_detached_head_omits_the_branch_parenthetical() -> None:
    file = make_source("body", repo=RepoInfo(name="r", branch=None))
    assert build_header(file, None, None).splitlines()[0] == "# repo: r"


def test_symbol_kind_is_lowercased_for_display() -> None:
    file = make_source("body", repo=REPO)
    assert "# class: Widget" in build_header(file, "Widget", "Class")


def test_symbol_without_a_kind_gets_a_neutral_label() -> None:
    file = make_source("body", repo=REPO)
    assert "# symbol: Widget" in build_header(file, "Widget", None)


def test_no_language_line_for_prose() -> None:
    file = make_source("body", kind=FileKind.TEXT, language=None)
    assert "language:" not in build_header(file, None, None)


def test_apply_header_prepends_and_leaves_source_untouched() -> None:
    combined = apply_header("# file: a.py", "def f():\n    pass")
    assert combined == "# file: a.py\ndef f():\n    pass"
    assert combined.endswith("def f():\n    pass")


def test_apply_header_is_a_no_op_when_disabled() -> None:
    """include_context_header: false must yield exactly the source."""
    assert apply_header("", "def f(): pass") == "def f(): pass"


def test_header_cost_is_reserved_from_the_chunk_budget() -> None:
    """max_tokens has to apply to what we embed, which is header + source.
    Budgeting only the source silently overshoots by the header's size --
    harmless against a 32K-token API model, but the tail of every large chunk
    against a 512-token local one."""
    from workspace_indexer.chunking.context_header import header_token_cost

    file = make_source("body", repo=REPO)
    cost = header_token_cost(file, FileKind.CODE)
    assert 5 < cost < 100


def test_header_cost_scales_with_the_header_that_will_be_built() -> None:
    from workspace_indexer.chunking.context_header import header_token_cost

    short = make_source("body", repo=None, root_label="a", rel_path="a.py")
    long = make_source(
        "body",
        repo=RepoInfo(name="a-very-long-repository-name", branch="feature/some-long-branch"),
        rel_path="src/deeply/nested/module/path/implementation.py",
    )
    assert header_token_cost(long, FileKind.CODE) > header_token_cost(short, FileKind.CODE)
