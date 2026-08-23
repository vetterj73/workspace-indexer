"""Exclusions that apply no matter what the user configures.

These are our own runtime state. A log file inside a watched root means the
watcher fires on our own writes, reindexes, writes more log lines, and spins
forever — not something a user should be able to configure themselves into.
"""

from __future__ import annotations

HARDCODED_EXCLUDES: tuple[str, ...] = (
    "**/.git/**",
    "logs/**",
    "**/logs/**",
    "data/**",
    "**/data/qdrant/**",
    "**/*.sqlite3",
    "**/*.sqlite3-journal",
    "**/workspace-indexer.jsonl*",
)
