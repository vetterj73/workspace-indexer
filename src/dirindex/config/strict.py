"""Base model for config sections."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Strict(BaseModel):
    """Reject unknown keys, so a typo in YAML is an error rather than a
    silently ignored setting that leaves you debugging the wrong thing."""

    model_config = ConfigDict(extra="forbid")
