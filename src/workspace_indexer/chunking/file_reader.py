"""Turn a discovery candidate into a SourceFile.

This is the first point at which a file is opened. Discovery deliberately
stops short of it, so the manifest's mtime/size fast path can skip a file
without paying for a read.

Classification up to here was extension-only. This is where a claim like "this
.txt is text" gets checked against the actual bytes, and a file that does not
decode is downgraded to OPAQUE rather than indexed as mojibake.
"""

from __future__ import annotations

from workspace_indexer.discovery.file_candidate import FileCandidate
from workspace_indexer.models import FileKind, SourceFile, sha256_text
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.chunking.reader")

# Kinds we never decode: reading them as text is meaningless and, for a large
# binary, wasteful.
_NEVER_TEXT = frozenset({FileKind.IMAGE, FileKind.OPAQUE, FileKind.PDF})

# A NUL byte in the first block is the same heuristic `git diff` uses to call a
# file binary, and it is far cheaper than attempting a full decode.
_SNIFF_BYTES = 8192


def read_source(candidate: FileCandidate) -> SourceFile | None:
    """None when the file vanished or became unreadable between the walk and
    now — a normal race on a live workspace, not an error."""
    try:
        raw = candidate.abs_path.read_bytes()
    except OSError as exc:
        log.warning("read.failed", error=str(exc))
        return None

    kind = candidate.kind
    text: str | None = None

    if kind not in _NEVER_TEXT:
        if b"\x00" in raw[:_SNIFF_BYTES]:
            log.debug("read.binary_downgrade", reason="nul_byte", declared=kind.value)
            kind = FileKind.OPAQUE
        else:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                log.debug("read.binary_downgrade", reason="undecodable", error=str(exc.reason))
                kind = FileKind.OPAQUE

    return SourceFile(
        root_label=candidate.root_label,
        unit=candidate.unit,
        abs_path=candidate.abs_path,
        rel_path=candidate.rel_path,
        kind=kind,
        language=candidate.language if kind is candidate.kind else None,
        size=len(raw),
        mtime_ns=candidate.mtime_ns,
        # Hash the bytes, not the decoded text, so a file is identified the
        # same way whether or not it happens to be decodable.
        sha256=sha256_text(raw.decode("utf-8", "surrogateescape")),
        repo=candidate.repo,
        text=text,
    )
