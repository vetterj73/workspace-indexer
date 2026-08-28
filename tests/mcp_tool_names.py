"""The registered MCP tool names, read out of the code that registers them.

One copy, because the whole point is that there should not be two. Three
places need this list -- the in-process server test, the stdio test, and the
reference-doc guard -- and a hand-written copy in any of them is a copy that
drifts. It already did: `impact_of` shipped while the stdio test went on
asserting the four tools that existed before it, green because CI deselects
integration tests.

Parsed statically rather than by building a server, so it stays usable from a
test that talks to a subprocess and from one that only reads files.
"""

from __future__ import annotations

import ast
from pathlib import Path

SERVER_FACTORY = (
    Path(__file__).resolve().parents[1] / "src" / "workspace_indexer" / "mcp" / "server_factory.py"
)


def registered_tool_names() -> list[str]:
    found: list[str] = []
    for node in ast.walk(ast.parse(SERVER_FACTORY.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(target, ast.Attribute) and target.attr == "tool":
                found.append(node.name)
    assert found, f"no @server.tool() functions found in {SERVER_FACTORY.name}"
    return found
