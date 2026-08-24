"""Flagging results whose source has changed since it was indexed.

The alternative is worse than it sounds: silently re-reading the file would
show its current text at the line numbers of the version that actually matched
the query.

The check is "is this chunk's exact text still in the file", not "does the file
hash match". A file hash and a chunk hash never compare directly, and a change
elsewhere in the file does not make *this* hit wrong.
"""

from __future__ import annotations

from pathlib import Path

from workspace_indexer.models import SearchHit
from workspace_indexer.obs.logging import get_logger

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
            cache[hit.abs_path] = _read(hit.abs_path)
        text = cache[hit.abs_path]
        # A file that has vanished is stale in the sense that matters: the hit
        # no longer points anywhere real.
        stale = text is None or hit.source_text not in text
        marked.append(hit.model_copy(update={"stale": True}) if stale else hit)

    if any(h.stale for h in marked):
        log.info("search.stale_hits", count=sum(1 for h in marked if h.stale))
    return marked


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.debug("staleness.unreadable", path=path, error=str(exc))
        return None
