"""One configured directory to index."""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator

from dirindex.config.strict import Strict


class RootConfig(Strict):
    path: Path
    label: str | None = None
    recurse_into_children: bool = False

    @field_validator("path")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return v.expanduser()

    @property
    def resolved_label(self) -> str:
        return self.label or self.path.name
