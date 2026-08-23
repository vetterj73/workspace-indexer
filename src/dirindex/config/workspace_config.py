"""Top-level workspace.yaml model."""

from __future__ import annotations

from pydantic import Field

from dirindex.config.chunking_section import ChunkingSection
from dirindex.config.eval_section import EvalSection
from dirindex.config.excludes import HARDCODED_EXCLUDES
from dirindex.config.index_section import IndexSection
from dirindex.config.logging_config import LoggingConfig
from dirindex.config.root_config import RootConfig
from dirindex.config.search_section import SearchSection
from dirindex.config.strict import Strict
from dirindex.config.workspace_section import WorkspaceSection


class WorkspaceConfig(Strict):
    workspace: WorkspaceSection
    index: IndexSection = Field(default_factory=IndexSection)
    chunking: ChunkingSection = Field(default_factory=ChunkingSection)
    search: SearchSection = Field(default_factory=SearchSection)
    eval: EvalSection = Field(default_factory=EvalSection)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @property
    def all_excludes(self) -> list[str]:
        return [*HARDCODED_EXCLUDES, *self.index.exclude]

    def root_by_label(self, label: str) -> RootConfig:
        for root in self.workspace.roots:
            if root.resolved_label == label:
                return root
        known = ", ".join(r.resolved_label for r in self.workspace.roots)
        raise KeyError(f"no root labelled {label!r}; known roots: {known}")
