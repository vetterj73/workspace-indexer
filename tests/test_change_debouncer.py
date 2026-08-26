"""Turning a burst of filesystem events into a set of roots to reindex.

The coalescing matters more than it looks. One editor save is several events --
vim writes a temp file, renames it over the original, deletes a backup -- and a
formatter run is hundreds. Reindexing per event would embed the same file over
and over.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from workspace_indexer.config import WorkspaceConfig
from workspace_indexer.watching import ChangeDebouncer


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, Path, WorkspaceConfig]:
    docs = tmp_path / "doc"
    code = tmp_path / "src"
    (docs / "guide").mkdir(parents=True)
    (code / "app").mkdir(parents=True)
    (code / "node_modules" / "pkg").mkdir(parents=True)
    (code / "logs").mkdir()
    config = WorkspaceConfig.model_validate(
        {
            "workspace": {
                "name": "w",
                "roots": [
                    {"path": str(docs), "label": "docs"},
                    {"path": str(code), "label": "code"},
                ],
            },
            "index": {"exclude": ["**/node_modules/**"]},
        }
    )
    return docs, code, config


def test_a_burst_of_edits_becomes_one_root(roots: tuple[Path, Path, WorkspaceConfig]) -> None:
    docs, _, config = roots
    debouncer = ChangeDebouncer(config)

    for name in ("a.md", "a.md", "b.md", "c.md"):
        debouncer.add(docs / "guide" / name)

    assert debouncer.drain() == ({"docs"}, False)


def test_changes_in_two_roots_report_both(roots: tuple[Path, Path, WorkspaceConfig]) -> None:
    docs, code, config = roots
    debouncer = ChangeDebouncer(config)

    debouncer.add(docs / "guide" / "a.md")
    debouncer.add(code / "app" / "main.py")

    assert debouncer.drain() == ({"docs", "code"}, False)


def test_draining_clears_the_buffer(roots: tuple[Path, Path, WorkspaceConfig]) -> None:
    """Otherwise every batch reindexes everything that has ever changed."""
    docs, _, config = roots
    debouncer = ChangeDebouncer(config)
    debouncer.add(docs / "guide" / "a.md")

    assert debouncer.drain()[0] == {"docs"}
    assert debouncer.drain()[0] == set()


def test_excluded_paths_do_not_wake_a_reindex(
    roots: tuple[Path, Path, WorkspaceConfig],
) -> None:
    _, code, config = roots
    debouncer = ChangeDebouncer(config)

    assert debouncer.add(code / "node_modules" / "pkg" / "index.js") is False
    assert debouncer.drain()[0] == set()


def test_our_own_writes_cannot_wake_a_reindex(
    roots: tuple[Path, Path, WorkspaceConfig],
) -> None:
    """The loop-prevention guarantee, at the watcher this time.

    `logs/` and `data/` are in HARDCODED_EXCLUDES precisely so a watcher cannot
    fire on our own log writes, reindex, write more log lines and spin forever.
    """
    _, code, config = roots
    debouncer = ChangeDebouncer(config)

    assert debouncer.add(code / "logs" / "workspace-indexer.jsonl") is False
    assert debouncer.add(code / "data" / "manifest.sqlite3") is False
    assert debouncer.drain()[0] == set()


def test_a_change_outside_every_root_is_ignored(
    roots: tuple[Path, Path, WorkspaceConfig], tmp_path: Path
) -> None:
    _, _, config = roots
    debouncer = ChangeDebouncer(config)

    assert debouncer.add(tmp_path / "elsewhere" / "x.py") is False


def test_the_config_file_is_reported_separately(
    roots: tuple[Path, Path, WorkspaceConfig], tmp_path: Path
) -> None:
    """It usually sits outside every root, so nothing else would notice it."""
    _, _, config = roots
    config_file = tmp_path / "workspace.yaml"
    config_file.write_text("workspace: {}", encoding="utf-8")
    debouncer = ChangeDebouncer(config, config_file)

    assert debouncer.add(config_file) is True
    roots_changed, config_changed = debouncer.drain()
    assert config_changed is True
    # A config change is not a file change: nothing to reindex from it alone.
    assert roots_changed == set()


def test_a_nested_root_wins_over_its_parent(tmp_path: Path) -> None:
    """Longest match, so a root inside another is attributed to the specific
    one -- otherwise editing it reindexes the wrong, larger tree."""
    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    config = WorkspaceConfig.model_validate(
        {
            "workspace": {
                "name": "w",
                "roots": [
                    {"path": str(outer), "label": "outer"},
                    {"path": str(inner), "label": "inner"},
                ],
            }
        }
    )
    debouncer = ChangeDebouncer(config)
    debouncer.add(inner / "file.py")

    assert debouncer.drain()[0] == {"inner"}
