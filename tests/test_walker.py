"""Filesystem traversal."""

from __future__ import annotations

import os
from pathlib import Path

from tests.conftest import ConfigFactory
from workspace_indexer.config import WorkspaceConfig
from workspace_indexer.discovery import Walker
from workspace_indexer.discovery.file_candidate import FileCandidate
from workspace_indexer.models import FileKind


def _by_path(config: WorkspaceConfig) -> dict[str, FileCandidate]:
    return {c.rel_path: c for c in Walker(config).walk()}


def test_hidden_directories_are_not_blanket_skipped(config_for: ConfigFactory) -> None:
    """`.claude` is a primary target. A generic "skip dotfiles" rule would drop
    the single most valuable directory in the workspace."""
    found = _by_path(config_for())
    assert ".claude/commands/deploy.md" in found
    assert ".claude/settings.json" in found


def test_git_directory_is_never_walked(config_for: ConfigFactory) -> None:
    found = _by_path(config_for())
    assert not [p for p in found if "/.git/" in p or p.startswith(".git/")]


def test_unit_is_the_top_level_subdirectory(config_for: ConfigFactory) -> None:
    found = _by_path(config_for())
    assert found["repo_one/src/widget.py"].unit == "repo_one"
    assert found["repo_two/app/main.ts"].unit == "repo_two"
    assert found["plain_folder/notes.md"].unit == "plain_folder"
    # A file sitting directly in the root belongs to no unit.
    assert found["TOPLEVEL.md"].unit == ""


def test_unit_is_empty_when_not_recursing_into_children(config_for: ConfigFactory) -> None:
    config = config_for()
    config.workspace.roots[0].recurse_into_children = False
    assert {c.unit for c in Walker(config).walk()} == {""}


def test_repo_metadata_attached_per_unit(config_for: ConfigFactory) -> None:
    found = _by_path(config_for())
    repo = found["repo_one/src/widget.py"].repo
    assert repo is not None
    assert repo.name == "repo_one"
    # Plain folders are indexed too; they just have no provenance.
    assert found["plain_folder/notes.md"].repo is None


def test_same_filename_ignored_in_one_repo_and_kept_in_another(config_for: ConfigFactory) -> None:
    found = _by_path(config_for())
    assert "repo_one/secret.txt" not in found
    assert "repo_two/secret.txt" in found


def test_gitignored_directory_is_pruned_and_counted(config_for: ConfigFactory) -> None:
    walker = Walker(config_for())
    found = {c.rel_path for c in walker.walk()}
    assert not [p for p in found if p.startswith("repo_one/build/")]
    # Pruning must be visible: "why isn't this indexed" is the common question.
    # Counted separately from file skips because one pruned directory can stand
    # for thousands of files.
    assert walker.pruned_dirs["gitignored"] >= 1


def test_excluded_directory_prune_is_counted_separately(config_for: ConfigFactory) -> None:
    walker = Walker(config_for())
    list(walker.walk())
    assert walker.pruned_dirs["excluded"] >= 1
    # node_modules is pruned as a directory, so it must not inflate the
    # per-file skip tally.
    assert walker.skips.get("excluded", 0) == 0


def test_config_excludes_prune_node_modules(config_for: ConfigFactory) -> None:
    found = _by_path(config_for())
    assert not [p for p in found if "node_modules" in p]


def test_lockfiles_skipped(config_for: ConfigFactory) -> None:
    walker = Walker(config_for())
    found = {c.rel_path for c in walker.walk()}
    assert "repo_one/package-lock.json" not in found
    assert walker.skips["lockfile"] >= 1


def test_opaque_files_are_yielded_not_skipped(config_for: ConfigFactory) -> None:
    """A binary is recorded so `status` can say it is known and deliberately
    not embedded. Counting it as a skip would double-report it."""
    walker = Walker(config_for())
    found = _by_path(config_for())
    list(walker.walk())
    assert found["plain_folder/blob.so"].kind is FileKind.OPAQUE
    assert found["plain_folder/logo.png"].kind is FileKind.IMAGE
    assert walker.skips.get("binary", 0) == 0


def test_size_cap_skips_large_files(config_for: ConfigFactory, workspace: Path) -> None:
    (workspace / "plain_folder" / "big.py").write_text("x = 1\n" * 5000, encoding="utf-8")
    config = config_for()
    config.index.max_file_bytes = 100
    walker = Walker(config)
    found = {c.rel_path for c in walker.walk()}
    assert "plain_folder/big.py" not in found
    assert walker.skips["too_large"] >= 1


def test_empty_files_skipped(config_for: ConfigFactory, workspace: Path) -> None:
    (workspace / "plain_folder" / "blank.py").touch()
    walker = Walker(config_for())
    found = {c.rel_path for c in walker.walk()}
    assert "plain_folder/blank.py" not in found
    assert walker.skips["empty"] >= 1


def test_symlinks_skipped_by_default(config_for: ConfigFactory, workspace: Path) -> None:
    os.symlink(workspace / "TOPLEVEL.md", workspace / "plain_folder" / "link.md")
    walker = Walker(config_for())
    found = {c.rel_path for c in walker.walk()}
    assert "plain_folder/link.md" not in found
    assert walker.skips["symlink"] >= 1


def test_only_root_filter(config_for: ConfigFactory, workspace: Path) -> None:
    config = config_for(
        workspace={
            "name": "test",
            "roots": [
                {"path": str(workspace / "repo_one"), "label": "one"},
                {"path": str(workspace / "repo_two"), "label": "two"},
            ],
        }
    )
    labels = {c.root_label for c in Walker(config).walk(only_root="two")}
    assert labels == {"two"}


def test_missing_root_warns_and_continues(config_for: ConfigFactory, workspace: Path) -> None:
    """One bad path in the config must not abort the whole run."""
    config = config_for(
        workspace={
            "name": "test",
            "roots": [
                {"path": str(workspace / "does_not_exist"), "label": "ghost"},
                {"path": str(workspace / "repo_two"), "label": "two"},
            ],
        }
    )
    assert {c.root_label for c in Walker(config).walk()} == {"two"}


def test_mtime_and_size_are_captured_for_the_manifest_fast_path(config_for: ConfigFactory) -> None:
    """The incremental fast path is a single stat() per file; if the walker did
    not carry these forward it would have to re-stat everything."""
    candidate = _by_path(config_for())["repo_one/src/widget.py"]
    assert candidate.mtime_ns > 0
    assert candidate.size == candidate.abs_path.stat().st_size


def test_configured_eval_dataset_is_never_walked(
    config_for: ConfigFactory, workspace: Path
) -> None:
    """Our own operational files, excluded by absolute path rather than by
    pattern, because the path is configurable."""
    dataset = workspace / "plain_folder" / "eval.yaml"
    dataset.write_text("- query: anything\n  expect: [x.py]\n", encoding="utf-8")
    config = config_for(eval={"dataset": str(dataset)})
    found = {c.rel_path for c in Walker(config).walk()}
    assert "plain_folder/eval.yaml" not in found


def test_our_own_eval_artefacts_are_never_discovered(
    workspace: Path, config_for: ConfigFactory
) -> None:
    """Every eval artefact quotes the query text of every case verbatim.

    Indexing one makes it a near-perfect lexical *and* semantic match for the
    queries used to score retrieval, so the harness ends up measuring our own
    output instead of the workspace. This has now happened three times, in
    three different files, which is why the rule is enforced in code rather
    than left to the user-editable exclude list.
    """
    (workspace / "evals").mkdir()
    (workspace / "evals" / "2026-01-01-run.json").write_text(
        '{"results": [{"query": "how does the walker skip files"}]}', encoding="utf-8"
    )
    (workspace / "repo_one" / "evals").mkdir()
    (workspace / "repo_one" / "evals" / "nested-run.json").write_text("{}", encoding="utf-8")
    (workspace / "docs").mkdir(exist_ok=True)
    (workspace / "docs" / "eval-baselines.md").write_text("# Baselines\n", encoding="utf-8")

    # The exclude list a user would plausibly write mentions none of these.
    found = _by_path(config_for())

    assert "evals/2026-01-01-run.json" not in found
    assert "repo_one/evals/nested-run.json" not in found
    assert "docs/eval-baselines.md" not in found
