"""Recording what an agent asked and what it got back.

The eval dataset is sixteen queries someone invented. These records turn the
queries an agent *actually* asks into the dataset, with what it received --
which is a categorically better source of cases.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog.testing

from workspace_indexer.mcp.tool_call_recorder import ToolCallRecorder
from workspace_indexer.models import ToolCall
from workspace_indexer.state.manifest import Manifest


@pytest.fixture
def manifest(tmp_path: Path) -> Iterator[Manifest]:
    with Manifest(tmp_path / "m.sqlite3") as m:
        yield m


def _call(tool: str = "search_code", paths: list[str] | None = None, **over: object) -> ToolCall:
    base: dict[str, object] = {
        "tool": tool,
        "query": "how does indexing decide to skip a file",
        "parameters": {"limit": "8"},
        "returned_paths": ["state/manifest.py"] if paths is None else paths,
        "total_matches": 3,
        "duration_ms": 42.0,
    }
    base.update(over)
    return ToolCall.model_validate(base)


# --- the log half ---------------------------------------------------------


def test_the_call_is_logged_with_tool_parameters_and_paths() -> None:
    """`search.query` records that a search happened. None of it says which
    tool ran, with what arguments, or what came back."""
    with structlog.testing.capture_logs() as logs:
        ToolCallRecorder().record(_call())

    entry = next(e for e in logs if e["event"] == "mcp.tool_call")
    assert entry["tool"] == "search_code"
    assert entry["paths"] == ["state/manifest.py"]
    assert entry["parameters"] == {"limit": "8"}
    assert entry["returned"] == 1


def test_an_empty_result_is_recorded_as_such() -> None:
    with structlog.testing.capture_logs() as logs:
        ToolCallRecorder().record(_call(paths=[], total_matches=0, note="Nothing matched."))

    entry = next(e for e in logs if e["event"] == "mcp.tool_call")
    assert entry["returned"] == 0
    assert entry["note"] == "Nothing matched."


# --- the manifest half ----------------------------------------------------


def test_calls_round_trip_through_the_manifest(manifest: Manifest) -> None:
    ToolCallRecorder(manifest).record(_call())

    stored = manifest.tool_calls()
    assert len(stored) == 1
    assert stored[0].tool == "search_code"
    assert stored[0].returned_paths == ["state/manifest.py"]
    assert stored[0].parameters == {"limit": "8"}


def test_disappointing_calls_are_the_harvesting_query(manifest: Manifest) -> None:
    """Calls that returned nothing, or had to drop results, are the ones worth
    turning into eval cases."""
    recorder = ToolCallRecorder(manifest)
    recorder.record(_call())
    recorder.record(_call(query="nothing matches this", paths=[], total_matches=0))
    recorder.record(_call(query="too much matched", dropped_for_budget=4))

    disappointing = manifest.tool_calls(disappointing_only=True)
    assert {c.query for c in disappointing} == {"nothing matches this", "too much matched"}
    assert len(manifest.tool_calls()) == 3


def test_stats_separate_empty_calls_from_the_rest(manifest: Manifest) -> None:
    recorder = ToolCallRecorder(manifest)
    recorder.record(_call(tool="find_guidance"))
    recorder.record(_call(tool="find_guidance", paths=[]))
    recorder.record(_call(tool="search_code"))

    assert manifest.tool_call_stats() == {"find_guidance": (2, 1), "search_code": (1, 0)}


def test_every_call_is_an_event_not_an_update(manifest: Manifest) -> None:
    """The same query asked twice is two facts, not one row overwritten."""
    recorder = ToolCallRecorder(manifest)
    recorder.record(_call())
    recorder.record(_call())
    assert len(manifest.tool_calls()) == 2


# --- failure must not propagate -------------------------------------------


def test_a_recording_failure_never_fails_the_call() -> None:
    """A search that worked must not error because bookkeeping did."""

    class Broken:
        def record_tool_call(self, call: ToolCall) -> None:
            raise sqlite3.OperationalError("database is locked")

    with structlog.testing.capture_logs() as logs:
        ToolCallRecorder(Broken()).record(_call())

    # The forensic half still succeeded, so nothing the caller needs is lost.
    assert [e for e in logs if e["event"] == "mcp.tool_call"]
    assert [e for e in logs if e["event"] == "mcp.tool_call_unrecorded"]


def test_no_sink_is_a_normal_configuration() -> None:
    with structlog.testing.capture_logs() as logs:
        ToolCallRecorder().record(_call())
    assert [e for e in logs if e["event"] == "mcp.tool_call"]
    assert not [e for e in logs if e["event"] == "mcp.tool_call_unrecorded"]


# --- the contamination guard ----------------------------------------------


def test_the_record_is_never_indexable() -> None:
    """This table holds query text verbatim, which has corrupted a measurement
    three times before -- config/eval.yaml, docs/eval-baselines.md, evals/.

    It lives under data/, which is hardcoded-excluded rather than left to the
    user-editable list, for exactly that reason.
    """
    from workspace_indexer.config import HARDCODED_EXCLUDES

    assert "data/**" in HARDCODED_EXCLUDES
    assert "**/*.sqlite3" in HARDCODED_EXCLUDES
