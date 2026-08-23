"""The `workspace:` block."""

from __future__ import annotations

from pydantic import Field, model_validator

from dirindex.config.root_config import RootConfig
from dirindex.config.strict import Strict


class WorkspaceSection(Strict):
    name: str
    roots: list[RootConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_labels(self) -> WorkspaceSection:
        labels = [r.resolved_label for r in self.roots]
        dupes = {label for label in labels if labels.count(label) > 1}
        if dupes:
            raise ValueError(
                f"duplicate root labels {sorted(dupes)}; labels key the manifest and the "
                "payload filter, so they must be unique — set `label:` explicitly"
            )
        return self
