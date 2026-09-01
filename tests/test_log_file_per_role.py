"""Two processes must not share one rotating log file.

`RotatingFileHandler` renames the live file on rollover, and Windows refuses to
rename a file another process holds open -- so a long-running `serve` and any
reindex collide with WinError 32 and one of them dies. POSIX permits the
rename, so this never surfaces on Linux: the orphaned handle keeps writing to
an inode nobody can find, which is a quieter failure and not a better one.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from workspace_indexer.config import FileLogConfig, LoggingConfig
from workspace_indexer.obs.logging import configure_logging, get_logger, reset_for_tests


@pytest.fixture(autouse=True)
def _clean_logging() -> None:  # pyright: ignore[reportUnusedFunction]
    reset_for_tests()


def config_for(path: Path) -> LoggingConfig:
    return LoggingConfig(
        level="INFO",
        console="off",
        file=FileLogConfig(path=path, max_bytes=1_000_000, backup_count=2),
    )


def emit(message: str) -> None:
    get_logger("test").info(message)
    for handler in logging.getLogger().handlers:
        handler.flush()


def test_a_role_lands_in_the_file_name(tmp_path: Path) -> None:
    configure_logging(config_for(tmp_path / "workspace-indexer.jsonl"), "serve")
    emit("hello")

    assert (tmp_path / "workspace-indexer-serve.jsonl").is_file()
    assert not (tmp_path / "workspace-indexer.jsonl").exists()


def test_two_roles_never_share_a_file(tmp_path: Path) -> None:
    """The whole point: `serve` holds its handle open while `index` rolls its own."""
    path = tmp_path / "workspace-indexer.jsonl"

    configure_logging(config_for(path), "serve")
    emit("from serve")
    reset_for_tests()
    configure_logging(config_for(path), "index")
    emit("from index")

    served = (tmp_path / "workspace-indexer-serve.jsonl").read_text(encoding="utf-8")
    indexed = (tmp_path / "workspace-indexer-index.jsonl").read_text(encoding="utf-8")

    assert "from serve" in served and "from index" not in served
    assert "from index" in indexed and "from serve" not in indexed


def test_no_role_leaves_the_configured_path_alone(tmp_path: Path) -> None:
    """A library caller that names no command still gets the file it asked for."""
    configure_logging(config_for(tmp_path / "workspace-indexer.jsonl"), None)
    emit("hello")

    assert (tmp_path / "workspace-indexer.jsonl").is_file()


def test_a_path_without_a_suffix_still_gets_the_role(tmp_path: Path) -> None:
    configure_logging(config_for(tmp_path / "logfile"), "watch")
    emit("hello")

    assert (tmp_path / "logfile-watch").is_file()


def test_a_command_that_logs_nothing_leaves_no_file(tmp_path: Path) -> None:
    """`delay=True`.

    Without it every invocation creates an empty file for its role, including
    runs that abort on a config error before logging anything.
    """
    configure_logging(config_for(tmp_path / "workspace-indexer.jsonl"), "status")

    assert list(tmp_path.iterdir()) == []
