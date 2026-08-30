"""PDFs, end to end: real files, real extraction, real chunks.

Every fixture here writes an actual PDF with pymupdf rather than stubbing the
extraction. The project's rule is that a mock confirms our assumptions instead
of the truth, and PDF text extraction is exactly the kind of thing our
assumptions are wrong about -- what a page returns when it holds only an image,
what an encrypted document does, whether a blank page comes back as "" or None.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import make_source
from workspace_indexer.chunking.chunker_registry import ChunkerRegistry
from workspace_indexer.chunking.file_reader import read_source
from workspace_indexer.chunking.pdf_chunker import PdfChunker
from workspace_indexer.config import ChunkingSection
from workspace_indexer.discovery.file_candidate import FileCandidate
from workspace_indexer.discovery.pdf_text import MIN_PAGE_CHARS, extract_pages
from workspace_indexer.models import FileKind

pymupdf = pytest.importorskip("pymupdf", reason="needs `poetry install --extras pdf`")

CONFIG = ChunkingSection()


def write_pdf(path: Path, pages: list[str], *, password: str | None = None) -> Path:
    """A real PDF with real text, one page per entry."""
    document = pymupdf.open()
    for body in pages:
        page = document.new_page()
        # `insert_textbox` returns the space left over, and a negative value
        # means the text did not fit and *nothing was written*. Left unchecked
        # that produces an empty PDF and a later assertion failing for a reason
        # with nothing to do with the code under test -- which is exactly what
        # it did the first time this file ran.
        remaining = page.insert_textbox(pymupdf.Rect(36, 36, 560, 780), body, fontsize=8)
        if remaining < 0:
            raise AssertionError(
                f"fixture text does not fit on one page (short by {-remaining:.0f}pt); "
                "shorten it or split it across pages"
            )
    if password:
        document.save(
            path, encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw=password, user_pw=password
        )
    else:
        document.save(path)
    document.close()
    return path


def candidate_for(path: Path, kind: FileKind = FileKind.PDF) -> FileCandidate:
    stat = path.stat()
    return FileCandidate(
        root_label="main",
        unit="docs",
        abs_path=path,
        rel_path=f"docs/{path.name}",
        kind=kind,
        language=None,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        repo=None,
    )


# ---- extraction -------------------------------------------------------------


def test_a_text_pdf_yields_one_string_per_page(tmp_path: Path) -> None:
    path = write_pdf(
        tmp_path / "guide.pdf",
        [
            "Rolling back a deployment requires the rollback script.",
            "Authentication verifies the bearer token before the handler runs.",
        ],
    )
    pages = extract_pages(path)

    assert pages is not None
    assert len(pages) == 2
    assert "rollback script" in pages[0]
    assert "bearer token" in pages[1]


def test_a_scanned_pdf_with_no_text_layer_is_not_indexed(tmp_path: Path) -> None:
    """The plan's rule: record it rather than index an empty document, so
    `status` can tell "we have this and cannot read it" from "we never saw
    it". Those call for OCR and for fixing the walk respectively."""
    document = pymupdf.open()
    document.new_page()  # no text at all
    path = tmp_path / "scan.pdf"
    document.save(path)
    document.close()

    assert extract_pages(path) is None


def test_a_page_with_almost_no_text_is_dropped_but_the_document_survives(
    tmp_path: Path,
) -> None:
    """A scanned figure between two pages of prose is normal. Losing the prose
    over it would be the wrong trade."""
    path = write_pdf(
        tmp_path / "mixed.pdf",
        ["7", "Deployment rollback is covered in the operations runbook."],
    )
    pages = extract_pages(path)

    assert pages is not None
    assert len(pages) == 1
    assert "runbook" in pages[0]
    assert len("7") < MIN_PAGE_CHARS


def test_an_encrypted_pdf_is_not_an_error(tmp_path: Path) -> None:
    """An ordinary thing to find in a workspace. It must cost that file, not
    the run."""
    path = write_pdf(tmp_path / "locked.pdf", ["Confidential deployment notes."], password="x")
    assert extract_pages(path) is None


def test_a_corrupt_file_costs_one_file_not_the_run(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.7\nnot actually a pdf at all")
    assert extract_pages(path) is None


# ---- reading ----------------------------------------------------------------


def test_reading_a_pdf_populates_both_text_and_pages(tmp_path: Path) -> None:
    path = write_pdf(
        tmp_path / "d.pdf",
        [
            "First page, covering deployment rollback procedures.",
            "Second page, covering authentication and bearer tokens.",
        ],
    )
    source = read_source(candidate_for(path))

    assert source is not None
    assert source.kind is FileKind.PDF
    assert len(source.pages) == 2
    assert source.text is not None


def test_the_secret_scanner_sees_every_page(tmp_path: Path) -> None:
    """The whole reason extraction happens in the reader.

    `text` is what the scanner reads and `pages` is what the chunker embeds, so
    anything in a page that is not in the text would reach the embedding API
    unscanned. Asserted rather than assumed, because the two are built
    separately and could drift apart silently.
    """
    path = write_pdf(
        tmp_path / "d.pdf",
        ["Page one covers deployment.", "Page two covers authentication.", "Page three: rollback."],
    )
    source = read_source(candidate_for(path))

    assert source is not None and source.text is not None
    for page in source.pages:
        assert page in source.text


def test_a_pdf_without_a_text_layer_is_recorded_as_opaque(tmp_path: Path) -> None:
    """Known and counted, not mysteriously absent."""
    document = pymupdf.open()
    document.new_page()
    path = tmp_path / "scan.pdf"
    document.save(path)
    document.close()

    source = read_source(candidate_for(path))

    assert source is not None
    assert source.kind is FileKind.OPAQUE
    assert source.text is None
    assert source.pages == []


def test_a_credential_in_a_pdf_is_withheld(tmp_path: Path) -> None:
    """A PDF is a document people paste credentials into, and before this it
    could not be scanned because it was never read."""
    from workspace_indexer.secrets import SecretWithheldError

    path = write_pdf(
        tmp_path / "setup.pdf",
        ["Deployment guide.", 'api_key = "sk-liveAKIA5FAKE0EXAMPLE7TOKEN"'],
    )
    with pytest.raises(SecretWithheldError):
        read_source(candidate_for(path))


# ---- chunking ---------------------------------------------------------------


def source_from(tmp_path: Path, pages: list[str]) -> object:
    path = write_pdf(tmp_path / "d.pdf", pages)
    source = read_source(candidate_for(path))
    assert source is not None
    return source


def test_each_page_becomes_a_chunk_anchored_to_its_page_number(tmp_path: Path) -> None:
    """The page is the only anchor a PDF has -- nobody opens one at line 340."""
    source = source_from(
        tmp_path,
        ["Rollback is covered here in detail.", "Authentication is covered here instead."],
    )
    chunks = list(PdfChunker("w").chunk(source, CONFIG))  # pyright: ignore[reportArgumentType]

    assert [c.meta.symbol_path for c in chunks] == ["page 1", "page 2"]
    assert [c.meta.symbol_kind for c in chunks] == ["page", "page"]
    assert [c.meta.symbol_name for c in chunks] == ["1", "2"]


def test_line_numbers_advance_through_the_document(tmp_path: Path) -> None:
    """`location` has to stay monotonic, as it is for every other chunker,
    even though the page is the useful half of the anchor."""
    source = source_from(
        tmp_path,
        ["First page with enough text to matter.", "Second page with enough text to matter."],
    )
    chunks = list(PdfChunker("w").chunk(source, CONFIG))  # pyright: ignore[reportArgumentType]

    assert chunks[0].meta.start_line == 1
    assert chunks[1].meta.start_line > chunks[0].meta.end_line


def test_a_long_page_is_split_and_says_which_part(tmp_path: Path) -> None:
    """A hit in a dense appendix should still be locatable in the document,
    not merely "somewhere on page 1"."""
    # Sized to fill one page and no more -- `write_pdf` refuses anything that
    # would overflow, so this is text that genuinely reached the PDF.
    paragraphs = "\n\n".join(
        f"Paragraph {i} covers deployment rollback and the operations runbook." for i in range(20)
    )
    source = source_from(tmp_path, [paragraphs])
    narrow = ChunkingSection.model_validate({"markdown": {"max_tokens": 48}})

    chunks = list(PdfChunker("w").chunk(source, narrow))  # pyright: ignore[reportArgumentType]

    assert len(chunks) > 1
    labels = [c.meta.symbol_path for c in chunks]
    assert all(label is not None for label in labels)
    assert all(str(label).startswith("page 1") for label in labels)
    assert "(1/" in str(labels[0])
    assert chunks[0].meta.chunk_total == len(chunks)


def test_no_pages_yields_nothing_rather_than_an_empty_chunk(tmp_path: Path) -> None:
    empty = make_source("", kind=FileKind.PDF, language=None, rel_path="docs/x.pdf")
    assert list(PdfChunker("w").chunk(empty, CONFIG)) == []


# ---- wiring -----------------------------------------------------------------


def test_the_registry_routes_pdfs_to_the_pdf_chunker(tmp_path: Path) -> None:
    source = source_from(tmp_path, ["Some deployment prose that is long enough."])
    resolved = ChunkerRegistry("w").resolve(source, CONFIG)  # pyright: ignore[reportArgumentType]
    assert resolved.name == "pdf"


def test_an_unreadable_pdf_routes_to_opaque_not_pdf(tmp_path: Path) -> None:
    """It was downgraded on read, so the registry sees OPAQUE and records it
    without embedding -- the behaviour PDFs had before this feature."""
    document = pymupdf.open()
    document.new_page()
    path = tmp_path / "scan.pdf"
    document.save(path)
    document.close()

    source = read_source(candidate_for(path))
    assert source is not None
    assert ChunkerRegistry("w").resolve(source, CONFIG).name == "opaque"
