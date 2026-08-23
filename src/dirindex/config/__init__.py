"""Configuration models.

workspace.yaml is *what to index* and is safe to commit; .env is *how to
index* and holds credentials. Only the former needs hot-reloading, which keeps
the watcher simple.

One class per module, re-exported here for readable call sites.
"""

from dirindex.config.chunking_section import ChunkingSection
from dirindex.config.code_chunking import CodeChunking
from dirindex.config.eval_section import EvalSection
from dirindex.config.excludes import HARDCODED_EXCLUDES
from dirindex.config.file_log_config import FileLogConfig
from dirindex.config.index_section import IndexSection
from dirindex.config.logfire_config import LogfireConfig
from dirindex.config.logging_config import LoggingConfig
from dirindex.config.markdown_chunking import MarkdownChunking
from dirindex.config.opaque_chunking import OpaqueChunking
from dirindex.config.rerank_config import RerankConfig
from dirindex.config.root_config import RootConfig
from dirindex.config.search_section import SearchSection
from dirindex.config.settings import Settings
from dirindex.config.strict import Strict
from dirindex.config.text_chunking import TextChunking
from dirindex.config.workspace_config import WorkspaceConfig
from dirindex.config.workspace_section import WorkspaceSection

__all__ = [
    "HARDCODED_EXCLUDES",
    "ChunkingSection",
    "CodeChunking",
    "EvalSection",
    "FileLogConfig",
    "IndexSection",
    "LogfireConfig",
    "LoggingConfig",
    "MarkdownChunking",
    "OpaqueChunking",
    "RerankConfig",
    "RootConfig",
    "SearchSection",
    "Settings",
    "Strict",
    "TextChunking",
    "WorkspaceConfig",
    "WorkspaceSection",
]
