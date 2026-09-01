"""Loading workspace.yaml.

Config errors have to surface here, naming the file and the key, rather than as
a mysteriously empty index three layers down.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from workspace_indexer.config import ConfigError, load_workspace_config

VALID = """
workspace:
  name: labbox
  roots:
    - path: /tmp/a
      label: a
"""


def test_loads_a_valid_file(tmp_path: Path) -> None:
    path = tmp_path / "workspace.yaml"
    path.write_text(VALID, encoding="utf-8")
    config = load_workspace_config(path)
    assert config.workspace.name == "labbox"
    assert config.search.fusion == "rrf"


def test_missing_file_says_how_to_fix_it(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="workspace.example.yaml"):
        load_workspace_config(tmp_path / "nope.yaml")


def test_malformed_yaml_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "workspace.yaml"
    path.write_text("workspace: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_workspace_config(path)


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "workspace.yaml"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="is empty"):
        load_workspace_config(path)


def test_non_mapping_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "workspace.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping at the top level"):
        load_workspace_config(path)


def test_validation_error_names_the_offending_key(tmp_path: Path) -> None:
    """Pydantic's default rendering buries the key path in noise."""
    path = tmp_path / "workspace.yaml"
    path.write_text("workspace:\n  name: labbox\n  roots: []\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="workspace.roots"):
        load_workspace_config(path)


def test_typo_in_a_key_is_an_error_not_a_silent_default(tmp_path: Path) -> None:
    path = tmp_path / "workspace.yaml"
    path.write_text(VALID + "\nindex:\n  respect_gitgnore: true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="respect_gitgnore"):
        load_workspace_config(path)


def test_a_repeated_key_is_rejected_rather_than_silently_discarded(tmp_path: Path) -> None:
    """YAML keeps the last of two identical keys and says nothing.

    So an edited block -- and every comment in it -- can have no effect at all,
    with no symptom: the file parses and the program runs on a setting its
    author is not reading. Found in a real config where a `file:` section was
    duplicated and the annotated half was dead text.
    """
    path = tmp_path / "workspace.yaml"
    path.write_text(
        "workspace:\n"
        "  name: t\n"
        "  roots: [{path: /tmp}]\n"
        "logging:\n"
        "  file: {path: FIRST}\n"
        "  level: INFO\n"
        "  file: {path: SECOND}\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as caught:
        load_workspace_config(path)

    message = str(caught.value)
    assert "duplicate key 'file'" in message
    # The line number matters: a config can repeat a key hundreds of lines apart.
    assert "line 7" in message


def test_the_same_key_in_different_mappings_is_fine(tmp_path: Path) -> None:
    """`path` appears under several sections, and always will."""
    path = tmp_path / "workspace.yaml"
    path.write_text(
        "workspace:\n"
        "  name: t\n"
        "  roots:\n"
        "    - {path: /tmp}\n"
        "    - {path: /var}\n"
        "logging:\n"
        "  file: {path: x.jsonl}\n",
        encoding="utf-8",
    )

    config = load_workspace_config(path)

    assert len(config.workspace.roots) == 2
