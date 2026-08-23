"""Rolling flat-file log settings."""

from __future__ import annotations

from pathlib import Path

from workspace_indexer.config.strict import Strict


class FileLogConfig(Strict):
    path: Path = Path("./logs/workspace-indexer.jsonl")
    max_bytes: int = 20_971_520
    backup_count: int = 10
