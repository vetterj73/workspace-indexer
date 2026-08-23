"""Top-level workspace.yaml model."""

from __future__ import annotations

from pydantic import Field

from workspace_indexer.config.chunking_section import ChunkingSection
from workspace_indexer.config.eval_section import EvalSection
from workspace_indexer.config.excludes import HARDCODED_EXCLUDES
from workspace_indexer.config.index_section import IndexSection
from workspace_indexer.config.logging_config import LoggingConfig
from workspace_indexer.config.root_config import RootConfig
from workspace_indexer.config.search_section import SearchSection
from workspace_indexer.config.strict import Strict
from workspace_indexer.config.workspace_section import WorkspaceSection


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
