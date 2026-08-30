"""Flagging results whose source has changed since it was indexed.

The alternative is worse than it sounds: silently re-reading the file would
show its current text at the line numbers of the version that actually matched
the query.

The check is "is this chunk's exact text still in the file", not "does the file
hash match". A file hash and a chunk hash never compare directly, and a change
elsewhere in the file does not make *this* hit wrong.

"Still in the file" has to mean the same text the chunk was built from, which
for a PDF is the extracted layer rather than the bytes. Reading a PDF as UTF-8
yields replacement characters, so every PDF hit came back flagged stale --
found by searching a real one, not by reading this.
"""

from __future__ import annotations

from pathlib import Path

from workspace_indexer.discovery.pdf_text import extract_pages
from workspace_indexer.models import FileKind, SearchHit
from workspace_indexer.obs.logging import get_logger, log_once

log = get_logger("workspace_indexer.search.staleness")


def mark_stale(hits: list[SearchHit]) -> list[SearchHit]:
    """One read per distinct file, not per hit: several chunks of one file
    routinely appear in the same result set."""
    cache: dict[str, str | None] = {}
    marked: list[SearchHit] = []

    for hit in hits:
        if not hit.abs_path or not hit.source_text:
            marked.append(hit)
            continue
        if hit.abs_path not in cache:
            cache[hit.abs_path] = _read(hit.abs_path, hit.kind)
        text = cache[hit.abs_path]
        if text is None and Path(hit.abs_path).exists():
            # Present but unreadable in the form the chunk was built from --
            # a PDF with the extra uninstalled, say. We cannot tell whether it
            # changed, and "stale" is a claim rather than an absence of one.
            log_once(
                log,
                "staleness:unjudgeable",
                "search.staleness_unknown",
                path=hit.rel_path,
                detail="cannot re-read this file in its indexed form; not flagging it either way",
            )
            marked.append(hit)
            continue
        # A file that has vanished is stale in the sense that matters: the hit
        # no longer points anywhere real.
        stale = text is None or hit.source_text not in text
        marked.append(hit.model_copy(update={"stale": True}) if stale else hit)

    if any(h.stale for h in marked):
        log.info("search.stale_hits", count=sum(1 for h in marked if h.stale))
    return marked


def _read(path: str, kind: FileKind) -> str | None:
    """The file as the chunker saw it, which is not always its bytes.

    Re-extracting a PDF on every search is real work, and the reason
    `search.check_staleness` exists as a setting. One extraction per file per
    search, not per hit.
    """
    if kind is FileKind.PDF:
        pages = extract_pages(Path(path))
        return None if pages is None else "\n\n".join(pages)
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.debug("staleness.unreadable", path=path, error=str(exc))
        return None
