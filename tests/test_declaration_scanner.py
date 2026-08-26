"""Declarations the symbol extractor misses.

`tree_sitter_language_pack.process()` reports `function_declaration` but not a
function assigned to a variable. In JS/TS that is the dominant idiom, not an
edge case: measured on a real 1,563-file polyglot repo, 134 of 320 top-level
components were declared that way and every chunk of them came back unnamed.
"""

from __future__ import annotations

import pytest

from workspace_indexer.chunking.declaration_scanner import SUPPORTED, DeclarationScanner


@pytest.fixture
def scanner() -> DeclarationScanner:
    return DeclarationScanner()


def _names(scanner: DeclarationScanner, code: str, language: str = "tsx") -> list[str]:
    return [d.name for d in scanner.scan(code, language)]


def test_the_react_idiom_is_found(scanner: DeclarationScanner) -> None:
    """The case that motivated all of this."""
    code = "const Cart: React.FC<Props> = ({ userId }) => { return <div/>; };"
    assert _names(scanner, code) == ["Cart"]


def test_an_exported_arrow_is_found(scanner: DeclarationScanner) -> None:
    assert _names(scanner, "export const Header = () => <h1/>;") == ["Header"]


def test_a_function_expression_is_found(scanner: DeclarationScanner) -> None:
    assert _names(scanner, "const useThing = function () { return 1; };") == ["useThing"]


def test_an_async_arrow_is_found(scanner: DeclarationScanner) -> None:
    assert _names(scanner, "const load = async (id) => { await go(id); };") == ["load"]


def test_a_class_property_arrow_is_a_method(scanner: DeclarationScanner) -> None:
    code = "class Svc {\n  fetchAll = async () => { return []; };\n}"
    found = scanner.scan(code, "tsx")
    assert [(d.name, d.kind) for d in found] == [("fetchAll", "method")]


def test_a_plain_value_is_not_a_declaration(scanner: DeclarationScanner) -> None:
    """`const config = { a: 1 }` is a declaration but not a definition.
    Labelling a chunk with it is worse than leaving the chunk unnamed."""
    code = "const config = { a: 1 };\nconst n = 5;\nconst s = 'x';"
    assert _names(scanner, code) == []


def test_spans_cover_a_whole_multiline_component(scanner: DeclarationScanner) -> None:
    """The span is what lets JSX fragments inherit the component's name -- a
    chunk whose text begins `<div className='grid'>` is otherwise
    unidentifiable."""
    code = "\n".join(["const Big = () => {", *[f"  const x{i} = {i};" for i in range(10)], "};"])
    found = scanner.scan(code, "tsx")
    big = next(d for d in found if d.name == "Big")
    assert big.start_line == 1
    assert big.end_line == 12
    assert big.contains(6)


def test_line_numbers_are_one_based(scanner: DeclarationScanner) -> None:
    """tree-sitter counts from 0; everything we expose counts from 1."""
    found = scanner.scan("\n\nconst A = () => 1;", "tsx")
    assert found[0].start_line == 3


@pytest.mark.parametrize("language", sorted(SUPPORTED))
def test_every_supported_language_actually_parses(
    scanner: DeclarationScanner, language: str
) -> None:
    assert _names(scanner, "const A = () => 1;", language) == ["A"]


def test_unsupported_languages_do_no_work(scanner: DeclarationScanner) -> None:
    """Python assigns lambdas but they are one-liners nobody searches by name,
    and C# already scores 83% on the library alone. A second parse for nothing
    is a cost on every file of those languages."""
    assert scanner.scan("f = lambda: 1", "python") == []
    assert scanner.scan("var f = () => 1;", "csharp") == []


def test_empty_and_broken_input_do_not_raise(scanner: DeclarationScanner) -> None:
    assert scanner.scan("", "tsx") == []
    # Half-typed code is a normal thing to index; tree-sitter recovers.
    assert isinstance(scanner.scan("const A = () => { const", "tsx"), list)
