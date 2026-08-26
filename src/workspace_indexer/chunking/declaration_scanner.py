"""Named declarations the symbol extractor does not report.

`tree_sitter_language_pack.process()` reports `function_declaration` and
`class_declaration` but not a function assigned to a variable. In JavaScript
and TypeScript that is not an edge case -- it is the dominant idiom:

    const Cart: React.FC<Props> = ({ userId }) => { ... }

Measured on a real 1,563-file polyglot repo, 134 of 320 top-level components
were declared that way, and every chunk of them came back with no symbol at
all. TSX scored 34% named against Python's 88%.

Spans, not just names, because they fix a second problem at the same time. A
large component splits into many chunks, and the ones after the first are
fragments of JSX -- a chunk whose text begins `<div className='grid'>` is
unidentifiable on its own. Knowing the enclosing declaration's line range lets
every one of those fragments be labelled with the component it came from.
"""

from __future__ import annotations

import tree_sitter_language_pack as tslp
from tree_sitter import Node

from workspace_indexer.chunking.declaration import Declaration
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.chunking.declarations")

# Only the JS family needs this. Python assigns lambdas but they are one-liners
# nobody searches for by name; C# scores 83% on the library alone. Scanning a
# language that does not need it would cost a second parse for nothing.
# No "jsx" entry: there is no jsx grammar, and `.jsx` files are classified
# as javascript. Listing it would only ever log a download failure.
SUPPORTED = frozenset({"javascript", "typescript", "tsx"})

# The declaration shapes that hold a callable.
_CALLABLE = frozenset({"arrow_function", "function_expression"})


class DeclarationScanner:
    """A second, cheap pass over the same source for missed declarations."""

    def scan(self, text: str, language: str) -> list[Declaration]:
        if language not in SUPPORTED or not text:
            return []
        try:
            parser = tslp.get_parser(language)  # pyright: ignore[reportArgumentType]
            tree = parser.parse(text.encode())
        except Exception as exc:
            # Same reasoning as the chunker: a grammar cache miss with no
            # network must not cost us the file, only the extra attribution.
            log.debug("declarations.parse_failed", language=language, error=str(exc))
            return []

        found: list[Declaration] = []
        _walk(tree.root_node, found)
        return found


def _walk(node: Node, out: list[Declaration]) -> None:
    if node.type == "variable_declarator":
        _record(node, "name", "function", out)
    elif node.type == "public_field_definition":
        # A class property holding an arrow function: `fetchAll = async () => {}`
        _record(node, "name", "method", out)
    for child in node.children:
        _walk(child, out)


def _record(node: Node, name_field: str, kind: str, out: list[Declaration]) -> None:
    value = node.child_by_field_name("value")
    if value is None or value.type not in _CALLABLE:
        # `const config = { a: 1 }` is a declaration but not a definition, and
        # labelling a chunk with it would be worse than leaving it unnamed.
        return
    name_node = node.child_by_field_name(name_field)
    if name_node is None or name_node.text is None:
        return
    out.append(
        Declaration(
            name=name_node.text.decode("utf-8", errors="replace"),
            kind=kind,
            # tree-sitter counts lines from 0; everything we expose counts
            # from 1.
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
        )
    )
