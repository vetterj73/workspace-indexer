"""Process-global state must have a reset, and must not survive a test.

The bug behind #47 was not really about structlog. It was about state living
for the life of the process while the test suite assumes each test starts
fresh -- and failing in the direction that looks like a pass, because
`capture_logs` returning `[]` reads exactly like "the code never logged".

Two guards, aimed at the two halves of that.

The first is structural: find every module-level name this codebase actually
*mutates*, and require it to be listed below with the mechanism that clears it.
Adding new process-global state is then a deliberate act with a reset attached,
rather than something discovered a month later by a test that fails on ordering.

The second is behavioural: assert the specific things that already leaked stay
un-leaked, including the contextvars every log line is decorated with.
"""

from __future__ import annotations

import ast
from pathlib import Path

import structlog

from workspace_indexer.obs.context import bound, new_run_id

SRC = Path(__file__).resolve().parents[1] / "src" / "workspace_indexer"

# Module-level names that are mutated at runtime, each with how it is reset.
# A name here is a promise that something clears it between tests; a name
# missing from here is state nobody has thought about yet.
MUTABLE_GLOBALS: dict[str, str] = {
    "obs/logging.py:_configured": "reset_for_tests()",
    "obs/logging.py:_seen_once": "forget_once_only(), autouse in conftest",
    # Refilled rather than replaced, which is the whole point -- see #47.
    "obs/logging.py:_PROCESSORS": "configure_logging() refills it in place",
}

# Names mutated in place at import time to build a constant, never afterwards.
# `X[a] = b` at module scope is a table being written, not state being kept.
_BUILT_AT_IMPORT = ("__all__",)

_MUTATORS = frozenset({"add", "clear", "append", "extend", "update", "pop", "discard", "remove"})


def _mutated_globals(path: Path) -> set[str]:
    """Module-level names the module changes after import."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module_level = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign | ast.AnnAssign)
        for target in ([node.target] if isinstance(node, ast.AnnAssign) else node.targets)
        if isinstance(target, ast.Name)
    }
    inside_a_function = {
        node
        for definition in ast.walk(tree)
        if isinstance(definition, ast.FunctionDef | ast.AsyncFunctionDef)
        for node in ast.walk(definition)
    }

    found: set[str] = set()
    for node in ast.walk(tree):
        # `global X` is an unambiguous declaration of intent to rebind.
        if isinstance(node, ast.Global):
            found.update(name for name in node.names if name in module_level)
        # `X.add(...)`, `X.clear()` and friends, but only from inside a
        # function: at module scope the same call is building a constant.
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _MUTATORS
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in module_level
            and node in inside_a_function
        ):
            found.add(node.func.value.id)
        # `X[...] = ...` and `X[:] = ...`, again only inside a function.
        if isinstance(node, ast.Assign) and node in inside_a_function:
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in module_level
                ):
                    found.add(target.value.id)
    return {name for name in found if name not in _BUILT_AT_IMPORT}


def test_every_mutable_global_has_a_documented_reset() -> None:
    """New process-global state must arrive with a way to clear it.

    This is the guard that would have made #47 a five-minute conversation
    rather than an afternoon: the state was there, nothing owned resetting it,
    and the consequence surfaced in an unrelated test two features later.
    """
    undeclared: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        for name in sorted(_mutated_globals(path)):
            key = f"{path.relative_to(SRC)}:{name}"
            if key not in MUTABLE_GLOBALS:
                undeclared.append(key)
    assert not undeclared, (
        "module-level state is mutated at runtime with no reset recorded in "
        "MUTABLE_GLOBALS. Anything here survives between tests and will "
        "eventually fail one on ordering alone:\n  " + "\n  ".join(undeclared)
    )


def test_the_declared_resets_still_refer_to_real_state() -> None:
    """The allowlist is a list too, so it drifts like any other.

    Without this it would quietly accumulate entries for state that no longer
    exists, and stop being a description of anything.
    """
    stale: list[str] = []
    for key in MUTABLE_GLOBALS:
        module, _, name = key.partition(":")
        if name not in _mutated_globals(SRC / module):
            stale.append(key)
    assert not stale, f"MUTABLE_GLOBALS names state that is no longer mutated: {stale}"


def test_logging_contextvars_do_not_survive_their_block() -> None:
    """`run_id` and `rel_path` decorate every log line emitted while bound.

    Leaking them would attribute one test's events to another's file -- and
    worse, attribute a production failure to whichever file happened to be
    processed before it, which is the exact debugging aid the binding exists
    to provide.
    """
    assert structlog.contextvars.get_contextvars() == {}

    with bound(run_id=new_run_id(), rel_path="src/a.py"):
        assert structlog.contextvars.get_contextvars()["rel_path"] == "src/a.py"

    assert structlog.contextvars.get_contextvars() == {}


def test_a_nested_binding_restores_the_outer_one() -> None:
    """The indexer binds a run and then a file inside it. If the inner block
    cleared rather than restored, every line after the first file would lose
    its run_id -- and the log's one reliable join key with it."""
    run = new_run_id()
    with bound(run_id=run):
        with bound(rel_path="src/a.py"):
            assert structlog.contextvars.get_contextvars()["run_id"] == run
        assert structlog.contextvars.get_contextvars() == {"run_id": run}
