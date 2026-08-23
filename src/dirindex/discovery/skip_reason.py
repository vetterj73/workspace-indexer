"""Why a file was not indexed.

Recorded per-file in the log and tallied per-run, because "why isn't this file
in the index" is the most common question a user will have.
"""

from __future__ import annotations

from enum import StrEnum


class SkipReason(StrEnum):
    GITIGNORED = "gitignored"
    EXCLUDED = "excluded"
    TOO_LARGE = "too_large"
    SYMLINK = "symlink"
    LOCKFILE = "lockfile"
    UNREADABLE = "unreadable"
    BINARY = "binary"
    EMPTY = "empty"
    UNCHANGED = "unchanged"
