"""Keeping the watcher's event stream to files the index would actually take.

What this cannot do is as important as what it can: watchfiles applies a filter
to changes the Rust watcher has *already produced*, so an excluded directory is
still descended into. These tests pin the reachable half -- no debounce entry,
no reindex -- and `test_watcher.py` covers surviving the half that is not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from watchfiles import Change

from workspace_indexer.config import WorkspaceConfig
from workspace_indexer.watching import ExcludeFilter


@pytest.fixture
def config(tmp_path: Path) -> WorkspaceConfig:
    (tmp_path / "code" / "src").mkdir(parents=True)
    (tmp_path / "code" / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "code" / ".ralph" / "tasks").mkdir(parents=True)
    payload: dict[str, Any] = {
        "workspace": {"name": "w", "roots": [{"path": str(tmp_path / "code"), "label": "code"}]},
        "index": {"exclude": ["**/node_modules/**", "**/.ralph/**"]},
    }
    return WorkspaceConfig.model_validate(payload)


def test_a_file_the_index_would_take_is_kept(config: WorkspaceConfig, tmp_path: Path) -> None:
    keep = ExcludeFilter(config)

    assert keep(Change.modified, str(tmp_path / "code" / "src" / "app.py")) is True


def test_an_excluded_tree_is_dropped(config: WorkspaceConfig, tmp_path: Path) -> None:
    """The reported case: `.ralph/**` is excluded, yet still woke a reindex."""
    keep = ExcludeFilter(config)

    assert keep(Change.modified, str(tmp_path / "code" / ".ralph" / "tasks" / "x.json")) is False
    assert keep(Change.added, str(tmp_path / "code" / "node_modules" / "pkg" / "i.js")) is False


def test_the_default_editor_noise_rules_are_kept(config: WorkspaceConfig, tmp_path: Path) -> None:
    """Subclassing DefaultFilter rather than replacing it.

    A bare callable would lose watchfiles' own rules and wake a reindex on
    every `.swp` vim writes beside the file being edited.
    """
    keep = ExcludeFilter(config)

    assert keep(Change.modified, str(tmp_path / "code" / "src" / "app.py.swp")) is False
    assert keep(Change.modified, str(tmp_path / "code" / "src" / "app.pyc")) is False
    assert keep(Change.modified, str(tmp_path / "code" / "src" / "__pycache__" / "a.pyc")) is False


def test_a_path_outside_every_root_is_kept(config: WorkspaceConfig, tmp_path: Path) -> None:
    """workspace.yaml is watched deliberately and usually sits outside the tree.

    Dropping it would silently disable config hot-reload.
    """
    keep = ExcludeFilter(config)

    assert keep(Change.modified, str(tmp_path / "workspace.yaml")) is True


def test_the_nearest_root_decides(tmp_path: Path) -> None:
    """A root nested inside another is attributed to the more specific one.

    Exclude patterns are global -- one list, matched relative to whichever
    root owns the file -- so which root owns it changes what the pattern sees.
    `inner/**` matches `inner/src/a.py` relative to `outer`, and matches
    nothing relative to `inner`, where the same file is just `src/a.py`.
    """
    (tmp_path / "outer" / "inner" / "src").mkdir(parents=True)
    config = WorkspaceConfig.model_validate(
        {
            "workspace": {
                "name": "w",
                "roots": [
                    {"path": str(tmp_path / "outer"), "label": "outer"},
                    {"path": str(tmp_path / "outer" / "inner"), "label": "inner"},
                ],
            },
            "index": {"exclude": ["inner/**"]},
        }
    )
    keep = ExcludeFilter(config)

    # Owned by `inner`, where the pattern cannot reach it.
    assert keep(Change.modified, str(tmp_path / "outer" / "inner" / "src" / "a.py")) is True
    # Owned by `outer`, where it does.
    assert keep(Change.modified, str(tmp_path / "outer" / "inner_notes.md")) is True
