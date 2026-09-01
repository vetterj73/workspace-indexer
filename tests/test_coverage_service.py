"""Assembling a coverage report from the manifest plus what git can add."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests.conftest import git_init, write
from workspace_indexer.classification import Classification
from workspace_indexer.grounding import CommitScanner, CoverageService, MarkerScanner
from workspace_indexer.grounding.source_strength import SourceStrength
from workspace_indexer.models import DocumentType, FileKind, SourceFile
from workspace_indexer.state import Manifest


def record(
    manifest: Manifest,
    *,
    abs_path: Path,
    rel_path: str,
    doc_type: DocumentType = DocumentType.IMPLEMENTATION,
    root_label: str = "src",
) -> None:
    source = SourceFile(
        root_label=root_label,
        unit=rel_path.split("/")[0],
        abs_path=abs_path,
        rel_path=rel_path,
        kind=FileKind.CODE,
        language="python",
        size=1,
        mtime_ns=1,
        sha256=hashlib.sha256(rel_path.encode()).hexdigest(),
        repo=None,
        text="x = 1\n",
    )
    manifest.record_file(
        source,
        chunker="code",
        chunker_version=1,
        classification=Classification(doc_type=doc_type, confidence=1.0, reason="test"),
        classifier_version=1,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real repository nested below its workspace root."""
    path = tmp_path / "workspace" / "product"
    path.mkdir(parents=True)
    write(path / "seed.py", "x = 1\n")
    git_init(path)
    return path


def test_documents_are_counted_against_the_code_they_explain(tmp_path: Path, repo: Path) -> None:
    with Manifest(tmp_path / "m.sqlite3") as manifest:
        for n in range(50):
            record(manifest, abs_path=repo / f"mod{n}.py", rel_path=f"workspace/product/mod{n}.py")
        record(
            manifest,
            abs_path=repo / "design.md",
            rel_path="workspace/product/design.md",
            doc_type=DocumentType.DESIGN,
        )

        units = CoverageService(manifest).coverage()

    assert len(units) == 1
    assert units[0].label == "product"
    assert units[0].code_files == 50
    design = next(s for s in units[0].sources if s.name == "design docs")
    assert design.found == 1
    assert design.strength is SourceStrength.PRESENT  # 1 per 50 clears one per hundred


def test_a_repository_nested_below_the_root_is_grouped_under_the_repository(
    tmp_path: Path, repo: Path
) -> None:
    """Not under the first path segment.

    The two coincide until a workspace is rearranged, and the label people read
    should name the thing git actually versions.
    """
    with Manifest(tmp_path / "m.sqlite3") as manifest:
        record(manifest, abs_path=repo / "seed.py", rel_path="workspace/product/seed.py")
        units = CoverageService(manifest).coverage()

    assert units[0].label == "product"
    assert units[0].signals is not None
    assert units[0].signals.commits == 1


def test_loose_files_above_a_repository_do_not_hide_it(tmp_path: Path, repo: Path) -> None:
    """The bug this grouping exists to fix.

    A workspace directory holding both a repository and files of its own was
    reported as a single unit, graded by whichever file the database returned
    first -- so 274 files inside a repository were reported as being in none.
    """
    loose = tmp_path / "workspace" / "notes.md"
    write(loose, "hello")

    with Manifest(tmp_path / "m.sqlite3") as manifest:
        record(manifest, abs_path=loose, rel_path="workspace/notes.md")
        for n in range(3):
            record(manifest, abs_path=repo / f"m{n}.py", rel_path=f"workspace/product/m{n}.py")
        units = CoverageService(manifest).coverage()

    by_label = {u.label: u for u in units}
    assert set(by_label) == {"product", "workspace"}
    assert by_label["product"].signals is not None
    assert by_label["workspace"].signals is None


def test_two_repositories_are_never_merged(tmp_path: Path, repo: Path) -> None:
    other = tmp_path / "workspace" / "second"
    other.mkdir(parents=True)
    write(other / "seed.py", "y = 2\n")
    git_init(other)

    with Manifest(tmp_path / "m.sqlite3") as manifest:
        record(manifest, abs_path=repo / "a.py", rel_path="workspace/product/a.py")
        record(manifest, abs_path=other / "b.py", rel_path="workspace/second/b.py")
        units = CoverageService(manifest).coverage()

    assert {u.label for u in units} == {"product", "second"}


def test_a_unit_outside_any_repository_has_no_commit_sources(tmp_path: Path) -> None:
    """Not a repository, so history and markers are absent rather than zero.

    Reporting them as zero would grade a plain directory of documents on its
    commit discipline, which it cannot have.
    """
    plain = tmp_path / "plain"
    plain.mkdir()
    write(plain / "note.md", "hello")

    with Manifest(tmp_path / "m.sqlite3") as manifest:
        record(
            manifest,
            abs_path=plain / "note.md",
            rel_path="plain/note.md",
            doc_type=DocumentType.DESIGN,
        )
        units = CoverageService(manifest).coverage()

    assert units[0].signals is None
    assert {s.name for s in units[0].sources} == {"design docs", "normative docs"}
    assert units[0].notes == ["not a git repository, so no history was read"]


def test_units_are_ordered_by_size(tmp_path: Path, repo: Path) -> None:
    other = tmp_path / "workspace" / "second"
    other.mkdir(parents=True)
    write(other / "seed.py", "y = 2\n")
    git_init(other)

    with Manifest(tmp_path / "m.sqlite3") as manifest:
        record(manifest, abs_path=other / "a.py", rel_path="workspace/second/a.py")
        for n in range(5):
            record(manifest, abs_path=repo / f"b{n}.py", rel_path=f"workspace/product/b{n}.py")
        units = CoverageService(manifest).coverage()

    assert [u.label for u in units] == ["product", "second"]


def test_scanners_are_injectable_so_the_report_can_be_tested_without_git(
    tmp_path: Path, repo: Path
) -> None:
    """The seam that keeps the assembly logic testable in isolation."""

    class NoHistory(CommitScanner):
        def scan(self, repo: Path) -> None:
            return None

    class NoMarkers(MarkerScanner):
        def scan(self, repo: Path) -> None:
            return None

    with Manifest(tmp_path / "m.sqlite3") as manifest:
        record(manifest, abs_path=repo / "a.py", rel_path="workspace/product/a.py")
        units = CoverageService(manifest, commits=NoHistory(), markers=NoMarkers()).coverage()

    assert {s.name for s in units[0].sources} == {"design docs", "normative docs"}


def test_an_empty_index_reports_nothing_rather_than_failing(tmp_path: Path) -> None:
    with Manifest(tmp_path / "m.sqlite3") as manifest:
        assert CoverageService(manifest).coverage() == []


def test_files_indexed_at_a_path_that_no_longer_exists_are_flagged(tmp_path: Path) -> None:
    """A workspace that moved leaves the manifest pointing at nothing.

    Without this the row reports a confident "absent" for every source, which
    is a statement about the index masquerading as one about the code.
    """
    with Manifest(tmp_path / "m.sqlite3") as manifest:
        record(
            manifest,
            abs_path=tmp_path / "gone" / "old" / "a.py",
            rel_path="gone/old/a.py",
        )
        units = CoverageService(manifest).coverage()

    assert units[0].on_disk is False
    assert "reindex" in units[0].notes[0]


def test_one_surviving_file_keeps_a_unit_off_the_stale_list(tmp_path: Path, repo: Path) -> None:
    """A single deleted file is ordinary; a whole unit vanishing is not."""
    with Manifest(tmp_path / "m.sqlite3") as manifest:
        record(manifest, abs_path=repo / "here.py", rel_path="workspace/product/here.py")
        record(
            manifest,
            abs_path=repo / "deleted" / "gone.py",
            rel_path="workspace/product/deleted/gone.py",
        )
        units = CoverageService(manifest).coverage()

    assert units[0].on_disk is True


def test_loose_files_at_a_root_share_one_unit_rather_than_one_each(tmp_path: Path) -> None:
    """A rel_path with no directory names no unit.

    Splitting unconditionally makes each such file its own unit named after
    itself, which turned one stale index into dozens of rows called
    `.dockerignore` and `PRTester.txt`.
    """
    loose = tmp_path / "loose"
    loose.mkdir()
    for name in (".dockerignore", "PRTester.txt", "CONTRIBUTING.md"):
        write(loose / name, "x")

    with Manifest(tmp_path / "m.sqlite3") as manifest:
        for name in (".dockerignore", "PRTester.txt", "CONTRIBUTING.md"):
            record(manifest, abs_path=loose / name, rel_path=name)
        units = CoverageService(manifest).coverage()

    assert [u.label for u in units] == ["(root)"]
