"""Pointing an import string at the file it names.

Rung 2 of the dependency graph. What it declines matters as much as what it
resolves: a confident edge pointing at the wrong file is worse than no edge,
because nothing downstream can tell.
"""

from __future__ import annotations

import pytest

from workspace_indexer.graph import ImportEdge
from workspace_indexer.graph.import_resolver import ImportResolver

FILES = {
    ("src", "repo"): {
        "repo/src/pkg/__init__.py",
        "repo/src/pkg/models.py",
        "repo/src/pkg/store/__init__.py",
        "repo/src/pkg/store/qdrant.py",
        "repo/tools/e2e/helpers/__init__.py",
        "repo/tools/e2e/helpers/llm_judge.py",
        "repo/tools/e2e/test_x.py",
        "repo/web/App.tsx",
        "repo/web/components/Cart.tsx",
        "repo/web/components/index.ts",
        "repo/web/hooks/useThing.ts",
        "repo/api/LogBroadcaster.ts",
    },
    # A second repo holding the same package name, to prove scoping.
    ("src", "other"): {"other/src/pkg/models.py"},
}


@pytest.fixture
def resolver() -> ImportResolver:
    return ImportResolver(FILES)


def _resolve(resolver: ImportResolver, module: str, frm: str, language: str) -> str | None:
    edge = ImportEdge(module=module, kind="import", is_relative=module.startswith("."), line=1)
    return resolver.resolve(edge, from_path=frm, root_label="src", language=language)


# --- python --------------------------------------------------------------


@pytest.mark.parametrize(
    ("module", "frm", "expected"),
    [
        (".helpers", "repo/tools/e2e/test_x.py", "repo/tools/e2e/helpers/__init__.py"),
        (".helpers.llm_judge", "repo/tools/e2e/test_x.py", "repo/tools/e2e/helpers/llm_judge.py"),
        ("..models", "repo/src/pkg/store/qdrant.py", "repo/src/pkg/models.py"),
        ("pkg.store", "repo/src/pkg/models.py", "repo/src/pkg/store/__init__.py"),
        ("pkg.store.qdrant", "repo/src/pkg/models.py", "repo/src/pkg/store/qdrant.py"),
    ],
)
def test_python_imports_resolve(
    resolver: ImportResolver, module: str, frm: str, expected: str
) -> None:
    assert _resolve(resolver, module, frm, "python") == expected


def test_a_single_dot_means_this_package(resolver: ImportResolver) -> None:
    """`.helpers` is a sibling of the importing file, not of its parent. Off by
    one level here silently resolves half a codebase to the wrong directory."""
    assert (
        _resolve(resolver, ".helpers", "repo/tools/e2e/test_x.py", "python")
        == "repo/tools/e2e/helpers/__init__.py"
    )


def test_stdlib_and_third_party_stay_unresolved(resolver: ImportResolver) -> None:
    """`os` is not a file in this workspace, and pretending otherwise would be
    a wrong answer rather than a missing one."""
    for module in ("os", "pathlib", "pydantic"):
        assert _resolve(resolver, module, "repo/src/pkg/models.py", "python") is None


# --- javascript family ---------------------------------------------------


@pytest.mark.parametrize(
    ("module", "frm", "expected"),
    [
        ("./components/Cart", "repo/web/App.tsx", "repo/web/components/Cart.tsx"),
        ("./components", "repo/web/App.tsx", "repo/web/components/index.ts"),
        ("../hooks/useThing", "repo/web/components/Cart.tsx", "repo/web/hooks/useThing.ts"),
    ],
)
def test_javascript_relative_imports_resolve(
    resolver: ImportResolver, module: str, frm: str, expected: str
) -> None:
    assert _resolve(resolver, module, frm, "tsx") == expected


def test_the_typescript_esm_extension_convention(resolver: ImportResolver) -> None:
    """TypeScript ESM writes `import './x.js'` for a file that is `x.ts` -- the
    specifier names the emitted file, not the one in the repository. Five real
    edges were unresolved by this alone."""
    assert (
        _resolve(resolver, "./LogBroadcaster.js", "repo/api/test.ts", "typescript")
        == "repo/api/LogBroadcaster.ts"
    )


def test_packages_and_aliases_stay_unresolved(resolver: ImportResolver) -> None:
    """`@/lib/utils` needs tsconfig paths parsed; `react` is a package. Both
    resolve to nothing rather than to something plausible."""
    for module in ("react", "@/hooks/useThing", "lucide-react"):
        assert _resolve(resolver, module, "repo/web/App.tsx", "tsx") is None


# --- scope and ambiguity -------------------------------------------------


def test_resolution_does_not_cross_repositories(resolver: ImportResolver) -> None:
    """Two repos routinely hold the same package name. Reaching across them
    needs a workspace-wide symbol table, which is rung 3."""
    assert (
        _resolve(resolver, "pkg.models", "repo/src/pkg/store/qdrant.py", "python")
        == "repo/src/pkg/models.py"
    )
    assert (
        _resolve(resolver, "pkg.models", "other/src/pkg/models.py", "python")
        == "other/src/pkg/models.py"
    )


def test_an_ambiguous_module_resolves_to_nothing() -> None:
    """Two files matching one module name means picking one would produce a
    confident edge pointing at the wrong file."""
    ambiguous = ImportResolver({("src", "repo"): {"repo/a/pkg/models.py", "repo/b/pkg/models.py"}})
    assert _resolve(ambiguous, "pkg.models", "repo/a/main.py", "python") is None


def test_an_unknown_unit_resolves_to_nothing(resolver: ImportResolver) -> None:
    assert _resolve(resolver, ".helpers", "nowhere/test.py", "python") is None


def test_languages_without_a_resolver_decline(resolver: ImportResolver) -> None:
    """C# namespaces name no path at all; bicep and the rest have no rule yet.
    An empty answer must mean "not supported", never "no dependency"."""
    for language in ("csharp", "bicep", "rust"):
        assert _resolve(resolver, "MyApp.Data.Core", "repo/src/x.cs", language) is None
