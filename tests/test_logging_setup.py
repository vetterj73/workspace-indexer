"""Logging sinks.

The file sink is the forensic record, so its two guarantees are worth pinning:
it captures DEBUG regardless of the console level, and it rotates rather than
growing without bound.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import structlog.testing
from concurrent_log_handler import ConcurrentRotatingFileHandler

from workspace_indexer.config import FileLogConfig, LoggingConfig
from workspace_indexer.obs.context import bound, new_run_id
from workspace_indexer.obs.logging import (
    configure_logging,
    get_logger,
    log_once,
    reset_for_tests,
)


@pytest.fixture(autouse=True)
def _clean_logging() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    reset_for_tests()
    yield
    reset_for_tests()


def _read(path: Path) -> list[dict[str, Any]]:
    """Lines written, treating a missing file as none.

    The handler opens its file on first write, so "nothing was logged" and "no
    file exists" are the same statement -- and a test asserting the first
    should not fail on the second.
    """
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _config(tmp_path: Path, **overrides: Any) -> LoggingConfig:
    payload: dict[str, Any] = {
        "level": "WARNING",
        "console": "off",
        "file": FileLogConfig(path=tmp_path / "workspace-indexer.jsonl"),
    }
    payload.update(overrides)
    return LoggingConfig(**payload)


def test_file_sink_captures_debug_even_when_console_is_quiet(tmp_path: Path) -> None:
    """You cannot retroactively raise a log level after the failure you needed
    to see, so the file always gets everything."""
    configure_logging(_config(tmp_path))
    get_logger("workspace_indexer.test").debug("embed.batch", tokens=1200)

    events = _read(tmp_path / "workspace-indexer.jsonl")
    assert [e["event"] for e in events] == ["embed.batch"]
    assert events[0]["tokens"] == 1200
    assert events[0]["level"] == "debug"


def test_output_is_json_lines(tmp_path: Path) -> None:
    """jq over the log file is the intended debugging workflow."""
    configure_logging(_config(tmp_path))
    log = get_logger("workspace_indexer.test")
    for i in range(3):
        log.info("embed.batch", duration_ms=i * 10)

    durations = [e["duration_ms"] for e in _read(tmp_path / "workspace-indexer.jsonl")]
    assert durations == [0, 10, 20]


def test_contextvars_reach_lines_logged_deeper_in_the_call_stack(tmp_path: Path) -> None:
    """The whole point: a failure logged inside the chunker still says which
    file caused it, without that being threaded through every signature."""
    configure_logging(_config(tmp_path))
    log = get_logger("workspace_indexer.test")

    def deep_helper() -> None:
        log.error("chunk.parse_failed", language="python")

    with bound(run_id="run123"), bound(root_label="repo_one", rel_path="src/widget.py"):
        deep_helper()

    event = _read(tmp_path / "workspace-indexer.jsonl")[0]
    assert event["run_id"] == "run123"
    assert event["rel_path"] == "src/widget.py"
    assert event["root_label"] == "repo_one"


def test_context_is_unbound_on_exit(tmp_path: Path) -> None:
    """A leaked rel_path would misattribute every later line to the wrong file."""
    configure_logging(_config(tmp_path))
    log = get_logger("workspace_indexer.test")
    with bound(rel_path="a.py"):
        pass
    log.info("run.end")
    assert "rel_path" not in _read(tmp_path / "workspace-indexer.jsonl")[0]


def test_exception_is_recorded_as_structured_data(tmp_path: Path) -> None:
    configure_logging(_config(tmp_path))
    log = get_logger("workspace_indexer.test")
    try:
        raise ValueError("boom")
    except ValueError:
        log.exception("error.chunking")

    event = _read(tmp_path / "workspace-indexer.jsonl")[0]
    assert event["event"] == "error.chunking"
    assert "boom" in json.dumps(event["exception"])


def test_rotation_creates_backups_and_bounds_total_size(tmp_path: Path) -> None:
    rolling = FileLogConfig(path=tmp_path / "x.jsonl", max_bytes=800, backup_count=2)
    configure_logging(_config(tmp_path, file=rolling))
    log = get_logger("workspace_indexer.test")
    for i in range(400):
        log.info("embed.batch", index=i, filler="p" * 100)

    assert (tmp_path / "x.jsonl.1").exists()
    # backup_count=2 means the oldest is discarded rather than accumulating.
    assert not (tmp_path / "x.jsonl.3").exists()


def test_log_once_suppresses_repeats(tmp_path: Path) -> None:
    """rerank.skipped is permanent for the life of the run; repeating it on
    every one of ten thousand searches would bury real events."""
    configure_logging(_config(tmp_path))
    log = get_logger("workspace_indexer.test")
    for _ in range(5):
        log_once(log, "rerank:no_key", "rerank.skipped", reason="no_api_key")

    events = _read(tmp_path / "workspace-indexer.jsonl")
    assert len(events) == 1
    assert events[0]["reason"] == "no_api_key"


def test_noisy_third_party_loggers_are_turned_down(tmp_path: Path) -> None:
    configure_logging(_config(tmp_path))
    logging.getLogger("httpx").info("chatty request detail")
    assert _read(tmp_path / "workspace-indexer.jsonl") == []


def test_console_off_leaves_only_the_file_handler(tmp_path: Path) -> None:
    configure_logging(_config(tmp_path))
    handlers = logging.getLogger().handlers
    assert len(handlers) == 1
    # The concurrent handler, not the stdlib one: several processes can share
    # a role's file, and the stdlib's rollover cannot survive that on Windows.
    assert isinstance(handlers[0], ConcurrentRotatingFileHandler)


def test_run_ids_are_distinct() -> None:
    assert new_run_id() != new_run_id()


def test_capture_logs_still_sees_a_logger_bound_before_a_reconfigure(tmp_path: Path) -> None:
    """The bug behind #47, in five lines.

    `cache_logger_on_first_use` freezes a bound logger against the processor
    list that was live when it was first used. structlog's own `capture_logs`
    mutates that list *in place* rather than replacing it, specifically so
    cached loggers keep seeing the current chain -- its source says so.

    Handing structlog a brand-new list on a second `configure` breaks that
    contract: the cached logger holds the old list, `capture_logs` mutates the
    new one, and the event escapes to the real sinks. The test that asserts on
    it then sees an empty list, which reads exactly like "the code never
    logged" -- so this fails in the direction that looks like a passing
    assertion about absence.

    Every module in this codebase binds its logger at import, so this is not a
    hypothetical: it took out two watcher tests in the full suite while CI,
    which deselects integration tests, stayed green.
    """
    configure_logging(_config(tmp_path))
    log = get_logger("workspace_indexer.probe")
    log.info("bind me")  # caches the logger against the first processor list

    reset_for_tests()
    configure_logging(_config(tmp_path))  # a second list would orphan the cache

    with structlog.testing.capture_logs() as captured:
        log.info("after.reconfigure", n=1)

    assert [e["event"] for e in captured] == ["after.reconfigure"]


def test_the_processor_list_keeps_its_identity_across_reconfigures(tmp_path: Path) -> None:
    """The invariant the test above depends on, asserted directly.

    Stated as identity rather than equality on purpose: equal contents in a new
    list is exactly the state that breaks caching, and is what the code did
    before.
    """
    configure_logging(_config(tmp_path))
    first = structlog.get_config()["processors"]

    reset_for_tests()
    configure_logging(_config(tmp_path))

    assert structlog.get_config()["processors"] is first
