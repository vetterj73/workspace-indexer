"""How a root is watched."""

from __future__ import annotations

from enum import StrEnum


class WatchMode(StrEnum):
    # Decide per root from the filesystem backing it. Almost always right.
    AUTO = "auto"
    # Force inotify. Fails loudly on a filesystem that cannot deliver events,
    # which is the point: an override that silently does nothing is worse than
    # no override.
    NATIVE = "native"
    # Force polling. Slower and burns CPU proportional to the tree, but works
    # on anything that can be stat()ed.
    POLL = "poll"
