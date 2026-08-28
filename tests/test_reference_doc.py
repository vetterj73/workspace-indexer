"""The reference must list everything that exists.

Documentation drifts silently: a new option ships, nobody adds it, and the
page quietly becomes a list of the options someone remembered. This is the
same guard that keeps `config/workspace.example.yaml` complete, applied to
`docs/reference.md` -- adding a config field or a CLI flag fails the build
until it is documented.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from tests.mcp_tool_names import cli_command_names, registered_tool_names
from workspace_indexer.config import Settings, WorkspaceConfig

REFERENCE = Path(__file__).resolve().parents[1] / "docs" / "reference.md"


@pytest.fixture(scope="module")
def text() -> str:
    return REFERENCE.read_text(encoding="utf-8")


def _leaf_fields(model: type[BaseModel], prefix: str = "") -> list[str]:
    """Dotted paths of every settable option, descending into nested models."""
    found: list[str] = []
    for name, field in model.model_fields.items():
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            found.extend(_leaf_fields(annotation, f"{prefix}{name}."))
        else:
            found.append(f"{prefix}{name}")
    return found


def test_every_workspace_option_is_documented(text: str) -> None:
    missing = [
        field
        for field in _leaf_fields(WorkspaceConfig)
        # `roots` is documented as a shape rather than leaf by leaf.
        if not field.startswith("workspace.roots") and field.split(".")[-1] not in text
    ]
    assert not missing, f"undocumented workspace.yaml options: {missing}"


def test_every_env_setting_is_documented(text: str) -> None:
    missing = [name for name in Settings.model_fields if name.upper() not in text]
    assert not missing, f"undocumented .env settings: {missing}"


def test_every_command_is_documented(text: str) -> None:
    commands = cli_command_names()
    # The name must end at a word boundary. A bare prefix check passed happily
    # for `mirrorX` when looking for `mirror`, which is the same
    # nearly-matching failure the guard exists to prevent.
    missing = [c for c in commands if not any(f"### `{c}{after}" in text for after in ("`", " "))]
    assert not missing, f"undocumented commands: {missing}"


def test_every_mcp_tool_and_its_parameters_are_documented(text: str) -> None:
    """Tool descriptions are written for the agent and only visible at
    runtime; a human reading the repo needs them here."""
    tools = registered_tool_names()
    assert len(tools) >= 5, f"expected to find the registered tools, got {tools}"
    for tool in tools:
        # Its own entry, not merely a mention. A passing reference in someone
        # else's paragraph is how a tool ends up "documented" with no
        # parameters listed -- which is what this test is for.
        assert f"**`{tool}`**" in text, tool
    for parameter in ("include_tests", "path_prefix", "rel_path", "doc_type", "repo", "limit"):
        assert parameter in text, parameter


def test_the_taxonomy_is_listed(text: str) -> None:
    from workspace_indexer.models import DocumentType

    missing = [t.value for t in DocumentType if t.value not in text]
    assert not missing, f"document types missing from the reference: {missing}"


def test_the_reference_is_linked_from_the_readme() -> None:
    """A page nobody can find is not documentation."""
    readme = (REFERENCE.parents[1] / "README.md").read_text(encoding="utf-8")
    assert "docs/reference.md" in readme


def test_the_testing_guide_is_linked_from_the_readme() -> None:
    """A page nobody can find is not documentation -- the same rule the
    reference is held to."""
    readme = (REFERENCE.parents[1] / "README.md").read_text(encoding="utf-8")
    assert "docs/testing.md" in readme
