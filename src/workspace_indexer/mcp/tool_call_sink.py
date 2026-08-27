"""Where a recorded tool call goes."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from workspace_indexer.models import ToolCall


@runtime_checkable
class ToolCallSink(Protocol):
    """The manifest, as far as the recorder needs to know.

    A protocol so a test can assert what was recorded without a database, and
    so the durable half can be swapped without touching the tools.
    """

    def record_tool_call(self, call: ToolCall) -> None: ...
