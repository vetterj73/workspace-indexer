"""The server as a client actually meets it: a subprocess on a pipe.

Everything else about the MCP layer is tested in-process, which cannot catch
the failures unique to stdio -- a tool that only registers under the real
entry point, or, the one that would be silent and fatal, anything printing to
stdout and corrupting the protocol stream.

Marked integration: it needs the configured index and whatever credentials the
embedding provider wants, so it does not run in CI.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from workspace_indexer.mcp import TAXONOMY_URI
from workspace_indexer.models import DocumentType

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[1]
CLI = REPO / ".venv" / "bin" / "workspace-indexer"


def registered_tool_names() -> list[str]:
    """Tool names parsed out of server_factory.

    Static rather than by building a server: this module deliberately talks to
    a subprocess, and importing the server here to enumerate it would assert
    against a different object than the one under test.
    """
    import ast

    source = REPO / "src" / "workspace_indexer" / "mcp" / "server_factory.py"
    found: list[str] = []
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(target, ast.Attribute) and target.attr == "tool":
                found.append(node.name)
    assert found, "no @server.tool() functions found; the parser needs updating"
    return found


def _params() -> StdioServerParameters:
    command = str(CLI) if CLI.exists() else shutil.which("workspace-indexer") or ""
    if not command:
        pytest.skip("workspace-indexer is not installed in this environment")
    return StdioServerParameters(command=command, args=["serve"], env=dict(os.environ))


def _body(result: Any) -> dict[str, Any]:
    if result.structured_content is not None:
        return dict(result.structured_content)
    text = "".join(b.text for b in result.content if isinstance(b, TextContent))
    loaded: object = json.loads(text)
    assert isinstance(loaded, dict)
    return dict(loaded)  # pyright: ignore[reportUnknownArgumentType]


async def test_a_client_can_use_every_tool_over_stdio() -> None:
    async with stdio_client(_params()) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        tools = await session.list_tools()
        # Read out of server_factory rather than typed in here. The hand-written
        # list was already wrong once: `impact_of` shipped and this test kept
        # asserting the four tools that existed before it, and stayed green
        # because CI runs -m "not integration" and nobody ran it by hand.
        assert sorted(t.name for t in tools.tools) == sorted(registered_tool_names())

        resources = await session.list_resources()
        assert [str(r.uri) for r in resources.resources] == [TAXONOMY_URI]

        taxonomy = _body(await session.call_tool("list_document_types", {}))
        assert {t["name"] for t in taxonomy["types"]} == {t.value for t in DocumentType}
        # A real index behind it, not an empty collection answering politely.
        assert sum(t["count"] for t in taxonomy["types"]) > 0

        hits = _body(await session.call_tool("search_code", {"query": "hybrid search", "limit": 3}))
        assert hits["results"]
        for result in hits["results"]:
            assert result["location"].count(":") >= 1

        # Every tool, not merely every registration: the graph tools answer
        # from the manifest rather than the vector store, so a serve process
        # that opened one and not the other would pass everything above.
        impact = _body(
            await session.call_tool(
                "impact_of", {"rel_path": hits["results"][0]["rel_path"], "limit": 5}
            )
        )
        assert impact["rel_path"] == hits["results"][0]["rel_path"]
        # Both directions present as keys even when empty, and `note` is what
        # keeps an empty answer from reading as "nothing depends on this".
        assert "depends_on" in impact and "used_by" in impact


async def test_stdout_is_not_corrupted_by_our_own_logging() -> None:
    """The failure mode unique to stdio, and a silent one: a print or a console
    log handler on stdout makes the stream unparseable and every call fail.

    Completing a round trip at all is the assertion -- the client would raise
    on the first unparseable line.
    """
    async with stdio_client(_params()) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        for _ in range(3):
            body = _body(await session.call_tool("list_document_types", {}))
            assert body["taxonomy_version"] >= 1


async def test_an_unknown_type_reaches_the_client_as_an_actionable_error() -> None:
    """End to end, because the SDK strips the message of an unrecognised
    exception and this is the one place that conversion can silently break."""
    async with stdio_client(_params()) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool(
            "find_guidance", {"query": "anything", "doc_type": "blueprint"}
        )

        assert result.is_error
        text = "".join(b.text for b in result.content if isinstance(b, TextContent))
        assert "blueprint" in text
        assert "normative" in text
