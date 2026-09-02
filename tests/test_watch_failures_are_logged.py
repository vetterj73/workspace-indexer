"""A watcher's failures have to survive the terminal that started it.

Reported firsthand: `watch` crashed on a machine it had never run on, and the
JSONL ended at `watch.start` with nothing after it -- no error, no shutdown,
nothing. The only evidence was on screen, and the screen was gone. A watcher is
meant to run unattended, so a console-only report is a report to nobody.

Assertions read the **log file**, not `structlog.testing.capture_logs`. The CLI
calls `configure_logging` inside the invocation, which replaces the processor
list `capture_logs` installed and silently undoes the capture -- and the file is
what the bug was about anyway.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from workspace_indexer.cli import app
from workspace_indexer.obs.logging import reset_for_tests


@pytest.fixture(autouse=True)
def _fresh_logging() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    # configure_logging is idempotent, so without this the first test in the
    # module pins the log path for every one after it.
    reset_for_tests()
    yield
    reset_for_tests()


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "code").mkdir()
    config = tmp_path / "workspace.yaml"
    config.write_text(
        f"workspace:\n  name: w\n  roots:\n    - path: {tmp_path / 'code'}\n"
        f"logging:\n  console: 'off'\n  file:\n    path: {tmp_path / 'l.jsonl'}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("STATE_DB", str(tmp_path / "m.sqlite3"))
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "q"))
    return config


def watch_log(tmp_path: Path) -> list[dict[str, Any]]:
    """The JSONL the `watch` command writes -- per-role, so `-watch` suffixed."""
    path = tmp_path / "l-watch.jsonl"
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parsed: object = json.loads(line)
            assert isinstance(parsed, dict)
            out.append(dict(parsed))  # pyright: ignore[reportUnknownArgumentType]
    return out


def events(entries: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [e for e in entries if e.get("event") == name]


def run_watch(config: Path, monkeypatch: pytest.MonkeyPatch, behaviour: object) -> None:
    import workspace_indexer.watching.watcher as watcher_module

    monkeypatch.setattr(watcher_module.Watcher, "run", behaviour)
    CliRunner().invoke(app, ["watch", "--config", str(config)])


def test_a_crash_inside_the_watcher_reaches_the_log(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Previously an unhandled exception killed the process with a traceback on
    stderr and left nothing in the JSONL at all."""

    async def explode(self: object, stop: object = None) -> None:
        raise RuntimeError("the file cannot be accessed by the system")

    run_watch(workspace, monkeypatch, explode)

    crashed = events(watch_log(tmp_path), "watch.crashed")
    assert len(crashed) == 1
    assert "cannot be accessed" in crashed[0]["error"]
    assert crashed[0]["error_type"] == "RuntimeError"


def test_a_clean_stop_is_recorded_too(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a terminator, a log ending at `watch.start` is ambiguous:
    crashed, killed and still-running are indistinguishable after the fact."""

    async def stop_at_once(self: object, stop: object = None) -> None:
        return None

    run_watch(workspace, monkeypatch, stop_at_once)

    stopped = events(watch_log(tmp_path), "watch.stopped")
    assert len(stopped) == 1
    assert stopped[0]["reason"] == "completed"


def test_an_interrupt_is_a_clean_stop_not_a_crash(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl-C is how this is meant to end. Logging it as a failure would train
    the reader to ignore the one event that means something went wrong."""

    async def interrupt(self: object, stop: object = None) -> None:
        raise KeyboardInterrupt

    run_watch(workspace, monkeypatch, interrupt)

    entries = watch_log(tmp_path)
    assert not events(entries, "watch.crashed")
    stopped = events(entries, "watch.stopped")
    assert len(stopped) == 1
    assert stopped[0]["reason"] == "interrupted"
