"""The shipped example config must stay valid.

Documentation that no longer parses is worse than no documentation: a new user
copies it, gets a validation error, and cannot tell whether they mistyped or we
shipped something broken. `extra="forbid"` means any renamed or removed field
fails this test immediately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from workspace_indexer.config import WorkspaceConfig

EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "workspace.example.yaml"


def _load() -> dict[str, Any]:
    return yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))


def test_example_exists() -> None:
    """README and CLAUDE.md both tell the user to copy this file."""
    assert EXAMPLE.is_file()


def test_example_validates() -> None:
    config = WorkspaceConfig.model_validate(_load())
    assert config.workspace.name
    assert config.workspace.roots


def test_example_exercises_every_section() -> None:
    """If a section is missing from the example, its defaults never get
    documented and a user has no idea the knob exists.

    Derived from the model rather than hardcoded, so adding a section fails
    this test until the example documents it -- rather than failing it until
    someone updates a list here, which teaches people to edit the list.
    """
    assert set(_load()) == set(WorkspaceConfig.model_fields)


def test_example_does_not_ship_cloud_logging_enabled() -> None:
    """Source text leaving the machine must never be the copy-paste default."""
    config = WorkspaceConfig.model_validate(_load())
    assert config.logging.logfire.send_to_cloud is False


def test_example_labels_are_unique_after_defaulting() -> None:
    """One root sets `label:` and one does not; the defaulted label must not
    collide with the explicit one."""
    config = WorkspaceConfig.model_validate(_load())
    labels = [r.resolved_label for r in config.workspace.roots]
    assert len(labels) == len(set(labels))
