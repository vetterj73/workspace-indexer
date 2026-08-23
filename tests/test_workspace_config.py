"""Config validation.

Config errors should surface as clear messages at load time, not as a
mysteriously empty index three layers down.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from dirindex.config import HARDCODED_EXCLUDES, WorkspaceConfig


def _minimal(**index: Any) -> dict[str, Any]:
    return {
        "workspace": {"name": "w", "roots": [{"path": "/tmp/a", "label": "a"}]},
        "index": index,
    }


def test_duplicate_labels_rejected() -> None:
    """Labels key the manifest and the payload filter, so a collision would
    silently merge two roots into one."""
    with pytest.raises(ValidationError, match="duplicate root labels"):
        WorkspaceConfig.model_validate(
            {
                "workspace": {
                    "name": "w",
                    "roots": [{"path": "/tmp/a"}, {"path": "/other/a"}],
                }
            }
        )


def test_label_defaults_to_directory_name() -> None:
    config = WorkspaceConfig.model_validate(
        {"workspace": {"name": "w", "roots": [{"path": "/tmp/myrepo"}]}}
    )
    assert config.workspace.roots[0].resolved_label == "myrepo"


def test_tilde_is_expanded() -> None:
    """`~/src` in YAML has to become a real path; the shell is not involved."""
    config = WorkspaceConfig.model_validate(
        {"workspace": {"name": "w", "roots": [{"path": "~/src", "label": "src"}]}}
    )
    assert config.workspace.roots[0].path == Path.home() / "src"


def test_unknown_key_is_an_error_not_a_silent_no_op() -> None:
    with pytest.raises(ValidationError):
        WorkspaceConfig.model_validate(_minimal(respect_gitgnore=True))  # typo


def test_at_least_one_root_required() -> None:
    with pytest.raises(ValidationError):
        WorkspaceConfig.model_validate({"workspace": {"name": "w", "roots": []}})


def test_hardcoded_excludes_always_apply() -> None:
    """A user must not be able to configure our own log directory into a
    watched root; that is an infinite reindex loop."""
    config = WorkspaceConfig.model_validate(_minimal(exclude=["**/custom/**"]))
    excludes = config.all_excludes
    assert "**/custom/**" in excludes
    for pattern in HARDCODED_EXCLUDES:
        assert pattern in excludes


def test_hardcoded_excludes_survive_an_empty_exclude_list() -> None:
    config = WorkspaceConfig.model_validate(_minimal())
    assert set(HARDCODED_EXCLUDES).issubset(config.all_excludes)


def test_root_by_label_error_names_the_alternatives() -> None:
    config = WorkspaceConfig.model_validate(
        {
            "workspace": {
                "name": "w",
                "roots": [{"path": "/tmp/a", "label": "one"}, {"path": "/tmp/b", "label": "two"}],
            }
        }
    )
    assert config.root_by_label("two").label == "two"
    with pytest.raises(KeyError, match="one, two"):
        config.root_by_label("three")


def test_defaults_are_usable_without_any_optional_sections() -> None:
    config = WorkspaceConfig.model_validate(
        {"workspace": {"name": "w", "roots": [{"path": "/tmp/a"}]}}
    )
    assert config.search.fusion == "rrf"
    assert config.search.rerank.model == "rerank-2.5-lite"
    assert config.chunking.code.max_tokens == 512
    assert config.logging.file is not None
    assert config.logging.logfire.send_to_cloud is False
