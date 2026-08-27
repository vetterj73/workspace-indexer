"""Recording what an agent asked and what it got back.

Two sinks, deliberately. The log is for forensics -- it already carries run_id
and rotates, and `jq` over it answers "what happened in that session". The
manifest is for harvesting: finding the calls that returned nothing is a query
rather than a log scrape, and those calls are eval cases waiting to be written.

The eval dataset is sixteen queries someone invented. This turns the queries an
agent actually asks into the dataset, which is a categorically better source.
"""

from __future__ import annotations

from workspace_indexer.mcp.tool_call_sink import ToolCallSink
from workspace_indexer.models import ToolCall
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.mcp.calls")


class ToolCallRecorder:
    def __init__(self, sink: ToolCallSink | None = None) -> None:
        # None is a normal configuration, not a degraded one: the log half
        # works without a manifest, and a test does not need a database to
        # assert what was recorded.
        self._sink = sink

    def record(self, call: ToolCall) -> None:
        log.info(
            "mcp.tool_call",
            tool=call.tool,
            query=call.query,
            parameters=call.parameters,
            returned=call.returned,
            paths=call.returned_paths,
            total_matches=call.total_matches,
            dropped_for_budget=call.dropped_for_budget,
            note=call.note,
            duration_ms=round(call.duration_ms, 1),
        )
        if self._sink is None:
            return
        try:
            self._sink.record_tool_call(call)
        except Exception as exc:
            # A search that worked must never fail because bookkeeping did.
            # The log half above has already succeeded, so nothing is lost
            # that matters to the caller.
            log.warning("mcp.tool_call_unrecorded", error=f"{type(exc).__name__}: {exc}")
