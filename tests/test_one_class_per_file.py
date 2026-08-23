"""Enforce the one-class-per-file mandate.

A rule that lives only in CLAUDE.md drifts the moment someone is in a hurry.
This makes it fail the build instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "dirindex"

# Deliberate exceptions go here, with a reason. Empty on purpose.
ALLOWED_MULTI_CLASS: dict[str, str] = {}


def _module_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if p.name != "__init__.py")


def test_at_most_one_top_level_class_per_module() -> None:
    offenders: list[str] = []
    for path in _module_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
        rel = str(path.relative_to(SRC))
        if len(classes) > 1 and rel not in ALLOWED_MULTI_CLASS:
            offenders.append(f"{rel}: {classes}")
    assert not offenders, "modules with more than one top-level class:\n  " + "\n  ".join(
        offenders
    )


def test_module_is_named_after_its_class() -> None:
    """`class SearchHit` belongs in `search_hit.py`."""

    def snake(name: str) -> str:
        out: list[str] = []
        for i, ch in enumerate(name):
            if ch.isupper() and i and not name[i - 1].isupper():
                out.append("_")
            out.append(ch.lower())
        return "".join(out)

    offenders: list[str] = []
    for path in _module_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
        if len(classes) != 1:
            continue
        expected = snake(classes[0])
        if path.stem != expected:
            offenders.append(
                f"{path.relative_to(SRC)} defines {classes[0]}, expected {expected}.py"
            )
    assert not offenders, "misnamed modules:\n  " + "\n  ".join(offenders)
