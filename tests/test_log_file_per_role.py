"""Two processes must not share one rotating log file.

`RotatingFileHandler` renames the live file on rollover, and Windows refuses to
rename a file another process holds open -- so a long-running `serve` and any
reindex collide with WinError 32 and one of them dies. POSIX permits the
rename, so this never surfaces on Linux: the orphaned handle keeps writing to
an inode nobody can find, which is a quieter failure and not a better one.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
from pathlib import Path
from typing import cast

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


def _writer(path_str: str, tag: str, lines: int) -> None:
    """A separate *process* writing to one role's log.

    Threads would not test this: the failure is between processes, where the
    stdlib handler's rollover rename and its unserialised appends both break.
    """
    from pathlib import Path as P

    from workspace_indexer.config import FileLogConfig, LoggingConfig
    from workspace_indexer.obs.logging import configure_logging, get_logger, reset_for_tests

    reset_for_tests()
    configure_logging(
        LoggingConfig(
            level="INFO",
            console="off",
            # Small enough that the processes roll over repeatedly while the
            # others still hold the file open -- which is the actual bug. The
            # backup count is generous on purpose: rotation is *supposed* to
            # discard the oldest lines, and a tighter one made this look like
            # lost writes when it was working correctly.
            file=FileLogConfig(path=P(path_str), max_bytes=4_000, backup_count=40),
        ),
        "serve",
    )
    log = get_logger("writer")
    for n in range(lines):
        log.info("line", tag=tag, n=n, filler="x" * 200)


def test_two_processes_in_the_same_role_do_not_corrupt_the_log(tmp_path: Path) -> None:
    """Issue #67: every Claude Code session spawns its own `serve`.

    Per-role naming separated `serve` from `index`, but four sessions are four
    `serve` processes on one file.

    This is *not* a Windows-only bug, which is what the issue and I both
    assumed. Measured here on Linux with the stdlib RotatingFileHandler: 346 of
    360 lines survived, and rollover raised `FileNotFoundError` as two
    processes raced to rename the same file and the loser found it already
    gone. Windows adds a fatal WinError 32 on top; POSIX just loses the lines
    quietly, which is the worse half.

    The line count is the assertion that discriminates. Every earlier version
    of this test -- intact JSON, all writers represented, rollover happened --
    passed with the broken handler.
    """
    target = tmp_path / "workspace-indexer.jsonl"
    procs = [
        multiprocessing.Process(target=_writer, args=(str(target), tag, 120))
        for tag in ("alpha", "beta", "gamma")
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)

    assert [p.exitcode for p in procs] == [0, 0, 0], "a writer died"

    written = tmp_path / "workspace-indexer-serve.jsonl"
    rotated = sorted(tmp_path.glob("workspace-indexer-serve.jsonl.*"))
    assert written.is_file()
    assert rotated, "expected rollover at this volume"

    # Every line intact JSON, and -- the part that actually discriminates --
    # every line still there. Nothing may be dropped: the backup count is sized
    # so rotation never legitimately discards any.
    tags: set[str] = set()
    kept = 0
    for path in [written, *rotated]:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw: object = json.loads(line)
            # Checked at runtime, then cast: json.loads is Any, and an
            # unnarrowed dict makes .get() untyped under strict mode.
            assert isinstance(raw, dict)
            entry = cast("dict[str, object]", raw)
            kept += 1
            tag = entry.get("tag")
            if isinstance(tag, str):
                tags.add(tag)

    assert tags == {"alpha", "beta", "gamma"}
    assert kept == 360, f"lost {360 - kept} lines to concurrent rollover"
