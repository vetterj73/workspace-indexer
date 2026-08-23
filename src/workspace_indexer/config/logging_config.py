"""The `logging:` block."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from workspace_indexer.config.file_log_config import FileLogConfig
from workspace_indexer.config.logfire_config import LogfireConfig
from workspace_indexer.config.strict import Strict


class LoggingConfig(Strict):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    console: Literal["pretty", "json", "off"] = "pretty"
    file: FileLogConfig | None = Field(default_factory=FileLogConfig)
    logfire: LogfireConfig = Field(default_factory=LogfireConfig)
