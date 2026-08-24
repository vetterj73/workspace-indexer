"""Configuration failures that are the user's to fix."""

from __future__ import annotations


class ConfigError(RuntimeError):
    """Carries a message meant to be shown to a person, not a traceback."""
