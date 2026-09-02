"""Top-level workspace.yaml model."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from workspace_indexer.config.chunking_section import ChunkingSection
from workspace_indexer.config.eval_section import EvalSection
from workspace_indexer.config.excludes import HARDCODED_EXCLUDES
from workspace_indexer.config.graph_section import GraphSection
from workspace_indexer.config.index_section import IndexSection
from workspace_indexer.config.logging_config import LoggingConfig
from workspace_indexer.config.root_config import RootConfig
from workspace_indexer.config.search_section import SearchSection
from workspace_indexer.config.strict import Strict
from workspace_indexer.config.watch_section import WatchSection
from workspace_indexer.config.workspace_section import WorkspaceSection


class WorkspaceConfig(Strict):
    workspace: WorkspaceSection
    index: IndexSection = Field(default_factory=IndexSection)
    chunking: ChunkingSection = Field(default_factory=ChunkingSection)
    search: SearchSection = Field(default_factory=SearchSection)
    watch: WatchSection = Field(default_factory=WatchSection)
    graph: GraphSection = Field(default_factory=GraphSection)
    eval: EvalSection = Field(default_factory=EvalSection)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @property
    def all_excludes(self) -> list[str]:
        return [*HARDCODED_EXCLUDES, *self.index.exclude]

    @property
    def excluded_paths(self) -> set[Path]:
        """Specific files that must never be indexed, whatever the patterns say.

        The eval dataset contains the query text of every case, which makes it a
        perfect lexical *and* semantic match for its own queries. Indexing it
        does not merely add noise: it puts the dataset at the top of its own
        results and corrupts every measurement taken afterwards.

        Derived rather than hardcoded because the path is configurable, but not
        user-overridable for the same reason `logs/` is not: it is a correctness
        rule, not a preference.
        """
        return {self.eval.dataset.expanduser().resolve()}

    def root_containing(self, path: Path) -> RootConfig | None:
        """The configured root that holds `path`, or None if no root does.

        Longest match wins, so a root nested inside another is attributed to
        the more specific one. Lives here rather than in the watcher because
        two callers now need the same answer -- the debouncer deciding which
        root to reindex, and the watch filter deciding whether to care at all
        -- and two copies of a prefix comparison is two chances to disagree
        about which root owns a file.
        """
        best: tuple[int, RootConfig] | None = None
        for root in self.workspace.roots:
            base = root.path.expanduser().resolve()
            if path == base or base in path.parents:
                depth = len(base.parts)
                if best is None or depth > best[0]:
                    best = (depth, root)
        return best[1] if best else None

    def root_by_label(self, label: str) -> RootConfig:
        for root in self.workspace.roots:
            if root.resolved_label == label:
                return root
        known = ", ".join(r.resolved_label for r in self.workspace.roots)
        raise KeyError(f"no root labelled {label!r}; known roots: {known}")
