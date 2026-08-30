"""Pulling the text layer out of a PDF.

Lives beside the reader rather than in `chunking/` on purpose, and the reason
is the secret scanner. `read_source` is the single point where bytes become
text, and the one place every path to the embedding API passes through. A
chunker that opened the file for itself would be a second such path, one the
scanner never sees -- and the scanner is what stands between a credential in a
document and Voyage's servers.

So the reader extracts, the scanner sees the whole document, and `PdfChunker`
splits text that has already been cleared.

pymupdf is an optional extra. Absent, a PDF is recorded as opaque -- known,
counted, deliberately unembedded -- exactly as it was before this existed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from workspace_indexer.obs.logging import get_logger, log_once

log = get_logger("workspace_indexer.discovery.pdf")

# A page whose extracted text is shorter than this carries no prose worth
# embedding -- a page number, a rule, a caption fragment under a scan. Kept low
# because a genuinely short page (a section divider, a title page) is still
# worth finding, and the cost of an extra small chunk is a few tokens.
MIN_PAGE_CHARS = 24


def extract_pages(path: Path) -> list[str] | None:
    """One string per page, or None when there is nothing to index.

    None is returned for three different situations, and they are logged
    differently because they call for different responses: the extra is not
    installed, the file cannot be opened, or the document genuinely has no text
    layer. Only the first is fixable by the person running it, and only the
    third means the PDF itself is the problem.

    Pages that are empty individually are dropped rather than failing the
    document -- a scanned figure between two pages of prose is normal, and
    losing the prose over it would be the wrong trade.
    """
    try:
        import pymupdf
    except ImportError:
        log_once(
            log,
            "pdf:no-extra",
            "pdf.extra_missing",
            detail=(
                "PDFs are recorded but not indexed; install the extra to read "
                "them: poetry install --extras pdf"
            ),
        )
        return None

    try:
        # pymupdf ships no py.typed marker, so every attribute off a Document
        # is Unknown under strict mode. Cast once, here, rather than scattering
        # suppressions: past this line the module is ordinary typed Python, and
        # the one thing that could actually change under us -- what `get_text`
        # returns -- is checked at runtime below instead of being assumed.
        opened = cast("Any", pymupdf.open(path))
        with opened as document:
            if bool(document.needs_pass):
                # Encrypted. Not an error -- a perfectly ordinary thing to find
                # in a workspace -- but nothing can be read from it.
                log.info("pdf.encrypted", pages=int(document.page_count))
                return None
            # `get_text()` is typed as returning str, list or dict because it
            # takes a mode argument we do not pass. Narrowed by construction
            # rather than suppressed: the default mode is "text", and anything
            # that is not a string here is a pymupdf change we want to notice.
            pages: list[str] = []
            for page in document:
                extracted: object = page.get_text()
                if not isinstance(extracted, str):
                    log.warning(
                        "pdf.unexpected_extraction",
                        got=type(extracted).__name__,
                        detail="pymupdf returned a non-string from get_text()",
                    )
                    return None
                pages.append(extracted.strip())
    except Exception as exc:
        # Deliberately broad: pymupdf raises its own exception types for a
        # damaged file, and a malformed PDF must cost that one file rather than
        # the run it was found in.
        log.warning("pdf.unreadable", error=f"{type(exc).__name__}: {exc}")
        return None

    usable = [text for text in pages if len(text) >= MIN_PAGE_CHARS]
    if not usable:
        # A scan with no OCR layer. Recorded as opaque rather than indexed as
        # an empty document, so `status` can tell "we have this and cannot read
        # it" from "we never saw it" -- which is the difference between needing
        # OCR and needing to fix the walk.
        log.info("pdf.no_text_layer", pages=len(pages))
        return None

    log.debug("pdf.extracted", pages=len(pages), with_text=len(usable))
    return usable
