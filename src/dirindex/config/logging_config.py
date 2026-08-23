"""The `logging:` block."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from dirindex.config.file_log_config import FileLogConfig
from dirindex.config.logfire_config import LogfireConfig
from dirindex.config.strict import Strict


class LoggingConfig(Strict):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    console: Literal["pretty", "json", "off"] = "pretty"
    file: FileLogConfig | None = Field(default_factory=FileLogConfig)
    logfire: LogfireConfig = Field(default_factory=LogfireConfig)
