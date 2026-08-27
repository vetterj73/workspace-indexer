"""What each file imports, taken from the source and not resolved.

Rung 1 of the dependency graph (#34). Deliberately stops at the string the
source actually wrote. Resolving `@/hooks/useThing` to a file needs tsconfig
paths, barrel files and extension inference; resolving `Intertech.Unifi.Data`
needs a workspace-wide namespace table. Both are real projects, and the point
of this rung is to measure coverage first so that decision is made on evidence.

Our own tree-sitter pass rather than `process(imports=True)`, for two reasons.
The library reports the whole statement text rather than the module specifier,
so `import { A } from '@/hooks/x'` arrives as that entire string. And it
extracts nothing at all for C#, where `using` is the whole story.
"""

from __future__ import annotations

import tree_sitter_language_pack as tslp
from tree_sitter import Node

from workspace_indexer.graph.import_edge import ImportEdge
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.graph.imports")

# Languages with a resolver worth writing later. Everything else returns
# nothing rather than a partial answer -- see `coverage` below for why that
# distinction has to survive into the report.
SUPPORTED = frozenset({"python", "typescript", "tsx", "javascript", "csharp"})

_JS = frozenset({"typescript", "tsx", "javascript"})


class ImportScanner:
    def scan(self, text: str, language: str) -> list[ImportEdge]:
        if language not in SUPPORTED or not text:
            return []
        try:
            tree = tslp.get_parser(language).parse(text.encode())  # pyright: ignore[reportArgumentType]
        except Exception as exc:
            # A grammar cache miss with no network must cost the edges, never
            # the file.
            log.debug("imports.parse_failed", language=language, error=str(exc))
            return []

        found: list[ImportEdge] = []
        _walk(tree.root_node, language, found)
        return found


def _walk(node: Node, language: str, out: list[ImportEdge]) -> None:
    if language == "python":
        _python(node, out)
    elif language in _JS:
        _javascript(node, out)
    elif language == "csharp":
        _csharp(node, out)
    for child in node.children:
        _walk(child, language, out)


def _python(node: Node, out: list[ImportEdge]) -> None:
    if node.type == "import_from_statement":
        module = node.child_by_field_name("module_name")
        if module is not None:
            _add(out, _text(module), "from", node)
    elif node.type == "import_statement":
        # `import os, sys.path` is two edges from one statement.
        for index in range(node.child_count):
            if node.field_name_for_child(index) == "name":
                child = node.child(index)
                if child is not None:
                    # `import x as y` wraps the name in aliased_import.
                    target = child.child_by_field_name("name") or child
                    _add(out, _text(target), "import", node)


def _javascript(node: Node, out: list[ImportEdge]) -> None:
    if node.type not in ("import_statement", "export_statement"):
        return
    source = node.child_by_field_name("source")
    if source is None:
        # `export const x = 1` is an export_statement with no source. Not an
        # edge; it is the other end of one.
        return
    module = _text(source).strip("'\"`")
    kind = "import" if node.type == "import_statement" else "reexport"
    _add(out, module, kind, node)


def _csharp(node: Node, out: list[ImportEdge]) -> None:
    if node.type != "using_directive":
        return
    # `using Alias = My.Name;` puts the alias in the `name` field and the
    # target unnamed, so position beats field here: the last qualified name is
    # the target in every form -- plain, static, aliased and global.
    names = [c for c in node.children if c.type in ("qualified_name", "identifier")]
    if names:
        _add(out, _text(names[-1]), "using", node)


def _add(out: list[ImportEdge], module: str, kind: str, node: Node) -> None:
    module = module.strip()
    if not module:
        return
    out.append(
        ImportEdge(
            module=module,
            kind=kind,
            is_relative=module.startswith("."),
            # tree-sitter counts from 0; everything we expose counts from 1.
            line=node.start_point[0] + 1,
        )
    )


def _text(node: Node) -> str:
    return node.text.decode("utf-8", errors="replace") if node.text else ""
