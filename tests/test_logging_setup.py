"""Logging sinks.

The file sink is the forensic record, so its two guarantees are worth pinning:
it captures DEBUG regardless of the console level, and it rotates rather than
growing without bound.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from dirindex.config import FileLogConfig, LoggingConfig
from dirindex.obs.context import bound, new_run_id
from dirindex.obs.logging import configure_logging, get_logger, log_once, reset_for_tests


@pytest.fixture(autouse=True)
def _clean_logging() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    reset_for_tests()
    yield
    reset_for_tests()


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _config(tmp_path: Path, **overrides: Any) -> LoggingConfig:
    payload: dict[str, Any] = {
        "level": "WARNING",
        "console": "off",
        "file": FileLogConfig(path=tmp_path / "dirindex.jsonl"),
    }
    payload.update(overrides)
    return LoggingConfig(**payload)


def test_file_sink_captures_debug_even_when_console_is_quiet(tmp_path: Path) -> None:
    """You cannot retroactively raise a log level after the failure you needed
    to see, so the file always gets everything."""
    configure_logging(_config(tmp_path))
    get_logger("dirindex.test").debug("embed.batch", tokens=1200)

    events = _read(tmp_path / "dirindex.jsonl")
    assert [e["event"] for e in events] == ["embed.batch"]
    assert events[0]["tokens"] == 1200
    assert events[0]["level"] == "debug"


def test_output_is_json_lines(tmp_path: Path) -> None:
    """jq over the log file is the intended debugging workflow."""
    configure_logging(_config(tmp_path))
    log = get_logger("dirindex.test")
    for i in range(3):
        log.info("embed.batch", duration_ms=i * 10)

    durations = [e["duration_ms"] for e in _read(tmp_path / "dirindex.jsonl")]
    assert durations == [0, 10, 20]


def test_contextvars_reach_lines_logged_deeper_in_the_call_stack(tmp_path: Path) -> None:
    """The whole point: a failure logged inside the chunker still says which
    file caused it, without that being threaded through every signature."""
    configure_logging(_config(tmp_path))
    log = get_logger("dirindex.test")

    def deep_helper() -> None:
        log.error("chunk.parse_failed", language="python")

    with bound(run_id="run123"), bound(root_label="repo_one", rel_path="src/widget.py"):
        deep_helper()

    event = _read(tmp_path / "dirindex.jsonl")[0]
    assert event["run_id"] == "run123"
    assert event["rel_path"] == "src/widget.py"
    assert event["root_label"] == "repo_one"


def test_context_is_unbound_on_exit(tmp_path: Path) -> None:
    """A leaked rel_path would misattribute every later line to the wrong file."""
    configure_logging(_config(tmp_path))
    log = get_logger("dirindex.test")
    with bound(rel_path="a.py"):
        pass
    log.info("run.end")
    assert "rel_path" not in _read(tmp_path / "dirindex.jsonl")[0]


def test_exception_is_recorded_as_structured_data(tmp_path: Path) -> None:
    configure_logging(_config(tmp_path))
    log = get_logger("dirindex.test")
    try:
        raise ValueError("boom")
    except ValueError:
        log.exception("error.chunking")

    event = _read(tmp_path / "dirindex.jsonl")[0]
    assert event["event"] == "error.chunking"
    assert "boom" in json.dumps(event["exception"])


def test_rotation_creates_backups_and_bounds_total_size(tmp_path: Path) -> None:
    rolling = FileLogConfig(path=tmp_path / "x.jsonl", max_bytes=800, backup_count=2)
    configure_logging(_config(tmp_path, file=rolling))
    log = get_logger("dirindex.test")
    for i in range(400):
        log.info("embed.batch", index=i, filler="p" * 100)

    assert (tmp_path / "x.jsonl.1").exists()
    # backup_count=2 means the oldest is discarded rather than accumulating.
    assert not (tmp_path / "x.jsonl.3").exists()


def test_log_once_suppresses_repeats(tmp_path: Path) -> None:
    """rerank.skipped is permanent for the life of the run; repeating it on
    every one of ten thousand searches would bury real events."""
    configure_logging(_config(tmp_path))
    log = get_logger("dirindex.test")
    for _ in range(5):
        log_once(log, "rerank:no_key", "rerank.skipped", reason="no_api_key")

    events = _read(tmp_path / "dirindex.jsonl")
    assert len(events) == 1
    assert events[0]["reason"] == "no_api_key"


def test_noisy_third_party_loggers_are_turned_down(tmp_path: Path) -> None:
    configure_logging(_config(tmp_path))
    logging.getLogger("httpx").info("chatty request detail")
    assert _read(tmp_path / "dirindex.jsonl") == []


def test_console_off_leaves_only_the_file_handler(tmp_path: Path) -> None:
    configure_logging(_config(tmp_path))
    handlers = logging.getLogger().handlers
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.handlers.RotatingFileHandler)


def test_run_ids_are_distinct() -> None:
    assert new_run_id() != new_run_id()
