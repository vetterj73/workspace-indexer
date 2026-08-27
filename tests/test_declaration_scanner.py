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


# A declaration in each language, so a listed language with a broken grammar
# fails here rather than silently reporting nothing forever.
_SMOKE = {
    "javascript": "const A = () => 1;",
    "typescript": "const A = () => 1;",
    "tsx": "const A = () => 1;",
    "bicep": "resource A 'Microsoft.KeyVault/vaults@2023-02-01' = { name: 'x' }",
    "powershell": "function A { }",
}


@pytest.mark.parametrize("language", sorted(SUPPORTED))
def test_every_supported_language_actually_parses(
    scanner: DeclarationScanner, language: str
) -> None:
    assert _names(scanner, _SMOKE[language], language) == ["A"]


def test_the_smoke_cases_cover_every_supported_language() -> None:
    """Otherwise adding a language silently skips its own smoke test."""
    assert set(_SMOKE) == set(SUPPORTED)


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


# --- bicep ---------------------------------------------------------------

BICEP = """\
param location string = resourceGroup().location
var prefix = 'app'
resource keyVault 'Microsoft.KeyVault/vaults@2023-02-01' = {
  name: '${prefix}-kv'
}
module network './network.bicep' = { name: 'net' }
output vaultUri string = keyVault.properties.vaultUri
"""


def test_bicep_resources_are_named(scanner: DeclarationScanner) -> None:
    """ "Where is the Key Vault defined" is a real infrastructure question, and
    bicep chunks carried no symbol at all before this."""
    found = {d.name: d.kind for d in scanner.scan(BICEP, "bicep")}
    assert found["keyVault"] == "resource"
    assert found["network"] == "module"


def test_bicep_params_vars_and_outputs_are_named(scanner: DeclarationScanner) -> None:
    found = {d.name: d.kind for d in scanner.scan(BICEP, "bicep")}
    assert found["location"] == "param"
    assert found["prefix"] == "var"
    assert found["vaultUri"] == "output"


def test_a_bicep_resource_spans_its_whole_body(scanner: DeclarationScanner) -> None:
    """The span is what lets a chunk inside a long resource block inherit its
    name rather than coming back unlabelled."""
    resource = next(d for d in scanner.scan(BICEP, "bicep") if d.name == "keyVault")
    assert resource.start_line == 3
    assert resource.end_line == 5
    assert resource.contains(4)


# --- powershell ----------------------------------------------------------


def test_powershell_functions_are_named(scanner: DeclarationScanner) -> None:
    code = "function Get-Remittance {\n  param($Id)\n}\n"
    assert [(d.name, d.kind) for d in scanner.scan(code, "powershell")] == [
        ("Get-Remittance", "function")
    ]


@pytest.mark.parametrize(
    ("code", "name"),
    [
        ("filter Format-Row { $_ }", "Format-Row"),
        ("workflow Deploy-All { }", "Deploy-All"),
    ],
)
def test_filter_and_workflow_are_functions_too(
    scanner: DeclarationScanner, code: str, name: str
) -> None:
    """The grammar reports all three as function_statement."""
    assert [d.name for d in scanner.scan(code, "powershell")] == [name]


def test_powershell_classes_and_methods(scanner: DeclarationScanner) -> None:
    code = "class Employer {\n  [void] Save() { }\n}\n"
    found = {d.name: d.kind for d in scanner.scan(code, "powershell")}
    assert found == {"Employer": "class", "Save": "method"}


def test_a_script_with_no_declarations_yields_none(scanner: DeclarationScanner) -> None:
    """Most deployment scripts are a sequence of top-level commands with
    nothing to name. Measured: 13 of 17 real .ps1 files contain no function at
    all, which is why powershell attribution stays low and should."""
    code = '$ErrorActionPreference = "Stop"\naz stack sub create -n Thing\n'
    assert scanner.scan(code, "powershell") == []


def test_javascript_behaviour_is_unchanged(scanner: DeclarationScanner) -> None:
    """The per-language split must not disturb the languages already at 82%."""
    code = "const Cart = () => 1;\nconst config = { a: 1 };\n"
    assert _names(scanner, code, "tsx") == ["Cart"]
