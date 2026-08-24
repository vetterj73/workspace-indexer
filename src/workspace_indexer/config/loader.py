"""Loading workspace.yaml.

Config errors surface here with the file and the offending key named, rather
than as a mysteriously empty index three layers down.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from workspace_indexer.config.config_error import ConfigError
from workspace_indexer.config.workspace_config import WorkspaceConfig

DEFAULT_CONFIG_PATH = Path("config/workspace.yaml")


def load_workspace_config(path: Path | None = None) -> WorkspaceConfig:
    target = (path or DEFAULT_CONFIG_PATH).expanduser()

    if not target.is_file():
        raise ConfigError(
            f"no config at {target}. Copy config/workspace.example.yaml to "
            f"{DEFAULT_CONFIG_PATH} and edit it for your machine."
        )

    try:
        raw: Any = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{target} is not valid YAML: {exc}") from exc

    if raw is None:
        raise ConfigError(f"{target} is empty")
    if not isinstance(raw, dict):
        raise ConfigError(f"{target} must be a mapping at the top level, got {type(raw).__name__}")

    try:
        return WorkspaceConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{target} is invalid:\n{_explain(exc)}") from exc


def _explain(error: ValidationError) -> str:
    """Pydantic's default rendering buries the key path in noise."""
    lines: list[str] = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "(root)"
        lines.append(f"  {location}: {item['msg']}")
    return "\n".join(lines)
