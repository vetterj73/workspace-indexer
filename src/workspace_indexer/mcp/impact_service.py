"""Answering "what would changing this file touch", from the manifest alone.

No vector store and no embedding call. The dependency graph is relational --
"which files import this one" is a WHERE clause -- and the whole point of
recording it in SQLite rather than the vector payload was to be able to ask it
this way.

The design constraint that shapes everything below: an empty answer must never
be readable as "nothing depends on this file". It can equally mean the
language has no import scanner, or that every edge naming this file is spelled
in a way we cannot resolve. Those call for opposite next moves, so every empty
result carries a note saying which one it is.
"""

from __future__ import annotations

import time

from workspace_indexer.graph.dependency import Dependency
from workspace_indexer.graph.dependent import Dependent
from workspace_indexer.graph.import_scanner import SUPPORTED
from workspace_indexer.mcp.impact_report import ImpactReport
from workspace_indexer.mcp.tool_call_recorder import ToolCallRecorder
from workspace_indexer.models import DocumentType, ToolCall
from workspace_indexer.state.manifest import Manifest

# What to keep when there are more dependents than fit. Callers before
# verifiers: the question behind "who uses this" is almost always "what breaks
# if I change it", and a test that breaks is a signal, while a caller that
# breaks is the actual damage. Generated files come last because editing them
# by hand is a mistake anyway.
_KEEP_ORDER = {
    DocumentType.IMPLEMENTATION.value: 0,
    DocumentType.TEST.value: 1,
    DocumentType.REFERENCE.value: 2,
    DocumentType.GENERATED.value: 4,
}
_DEFAULT_RANK = 3

# Files that exist to re-export something else. An import that lands on one of
# these is a hop, not a destination: the file that actually uses the symbol
# imported the barrel, so it never appears as a dependent of the module the
# symbol lives in. This project's own re-export mandate guarantees the case,
# and a TypeScript codebase with index.ts barrels has exactly the same shape.
_BARRELS = frozenset(
    {"__init__.py", "index.ts", "index.tsx", "index.js", "index.jsx", "index.mjs", "index.cjs"}
)


class ImpactService:
    def __init__(self, manifest: Manifest, *, recorder: ToolCallRecorder | None = None) -> None:
        self._manifest = manifest
        self._recorder = recorder or ToolCallRecorder()

    def impact_of(self, path: str, *, limit: int = 25) -> ImpactReport:
        started = time.monotonic()
        report = self._build(path, limit=limit)
        self._record(started, path, report)
        return report

    def _build(self, path: str, *, limit: int) -> ImpactReport:
        matches = self._manifest.find_paths(path)
        resolved = _pick(path, matches)
        if resolved is None:
            return _unresolved(path, matches)

        root_label, rel_path = resolved
        record = self._manifest.get_file(root_label, rel_path)
        language = record.language if record else None

        dependencies = self._manifest.dependencies_of(root_label, rel_path)
        dependents = self._manifest.dependents_of(root_label, rel_path)
        counts = _count_by_type(dependents)
        kept_out = dependencies[:limit]
        kept_in = sorted(dependents, key=_keep_rank)[:limit]

        return ImpactReport(
            rel_path=rel_path,
            root_label=root_label,
            doc_type=record.doc_type.value if record else DocumentType.UNKNOWN.value,
            language=language,
            depends_on=kept_out,
            depends_on_total=len(dependencies),
            used_by=kept_in,
            used_by_total=len(dependents),
            used_by_by_type=counts,
            dropped_depends_on=len(dependencies) - len(kept_out),
            dropped_used_by=len(dependents) - len(kept_in),
            note=self._note(
                language=language,
                dependencies=dependencies,
                dependents=dependents,
                dropped=(len(dependencies) - len(kept_out)) + (len(dependents) - len(kept_in)),
            ),
        )

    def _note(
        self,
        *,
        language: str | None,
        dependencies: list[Dependency],
        dependents: list[Dependent],
        dropped: int,
    ) -> str | None:
        """The half of the answer that is about what we could not see.

        Ordered by how badly each thing misleads. Unscanned language first:
        that is the case where every number above is zero for a reason that has
        nothing to do with the file.
        """
        parts: list[str] = []
        if language is None or language not in SUPPORTED:
            named = language or "this file type"
            parts.append(
                f"Imports are not scanned for {named}, so both lists are empty by "
                "construction. This says nothing about whether anything depends on "
                f"this file. Scanned languages: {', '.join(sorted(SUPPORTED))}."
            )
        else:
            if not dependents:
                parts.append(
                    "No indexed file resolves an import to this one. Edges naming a "
                    "package, a build alias or a namespace are recorded but not "
                    "resolved to a file, so a dependency expressed that way would "
                    "not appear here."
                )
            barrels = sorted({d.rel_path for d in dependents if _is_barrel(d.rel_path)})
            if barrels:
                parts.append(
                    f"{len(barrels)} of the importers are re-export files ({', '.join(barrels)}). "
                    "Anything importing the symbol through one of those is NOT counted "
                    "here, so the real number of callers is higher -- run impact_of on "
                    "the re-export file to follow the next hop."
                )
            unresolved = sum(1 for d in dependencies if not d.resolved)
            if unresolved:
                parts.append(
                    f"{unresolved} of {len(dependencies)} imports point outside the "
                    "index -- packages, stdlib, or aliases we cannot follow -- and "
                    "are listed with rel_path null."
                )
            coverage = self._manifest.resolution_coverage().get(language)
            if coverage and coverage[1]:
                got, total = coverage
                parts.append(
                    f"Workspace-wide, {got} of {total} {language} import edges resolve "
                    "to an indexed file."
                )
        if dropped:
            parts.append(
                f"{dropped} further edge(s) were omitted to stay inside limit; "
                "callers were kept ahead of tests. Raise limit to see the rest."
            )
        return " ".join(parts) if parts else None

    def _record(self, started: float, path: str, report: ImpactReport) -> None:
        self._recorder.record(
            ToolCall(
                tool="impact_of",
                query=path,
                parameters={"rel_path": report.rel_path} if report.rel_path else {},
                # Both directions, so a harvested call shows what the agent was
                # actually handed rather than only half of it.
                returned_paths=[d.rel_path for d in report.used_by]
                + [d.rel_path for d in report.depends_on if d.rel_path],
                total_matches=report.used_by_total + report.depends_on_total,
                dropped_for_budget=report.dropped_used_by + report.dropped_depends_on,
                note=report.note,
                duration_ms=(time.monotonic() - started) * 1000,
            )
        )


def _pick(path: str, matches: list[tuple[str, str]]) -> tuple[str, str] | None:
    """One match, or nothing.

    An exact path always wins, even when it is also a suffix of longer ones:
    someone who typed the whole path meant that file. Short of that, guessing
    between two candidates is the failure this tool exists to avoid -- an agent
    told "nothing imports this" about the wrong file will delete working code.
    """
    exact = [m for m in matches if m[1] == path]
    if len(exact) == 1:
        return exact[0]
    if len(matches) == 1:
        return matches[0]
    return None


def _unresolved(path: str, matches: list[tuple[str, str]]) -> ImpactReport:
    if not matches:
        return ImpactReport(
            note=(
                f"No indexed file matches {path!r}. The path is matched as a whole "
                "path or a trailing portion of one, on a directory boundary -- try a "
                "longer or shorter portion, or the file may be excluded from the index."
            )
        )
    return ImpactReport(
        candidates=[rel for _, rel in matches],
        note=(
            f"{len(matches)} indexed files end with {path!r}. Nothing was guessed: "
            "call again with one of the candidate paths."
        ),
    )


def _is_barrel(rel_path: str) -> bool:
    return rel_path.rsplit("/", 1)[-1] in _BARRELS


def _count_by_type(dependents: list[Dependent]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for dependent in dependents:
        counts[dependent.doc_type] = counts.get(dependent.doc_type, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _keep_rank(dependent: Dependent) -> tuple[int, str, int]:
    return (
        _KEEP_ORDER.get(dependent.doc_type, _DEFAULT_RANK),
        dependent.rel_path,
        dependent.line,
    )
