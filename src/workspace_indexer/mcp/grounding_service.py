"""Answering "does this codebase record why, and where should I look".

The tool an agent reaches for after a `find_guidance` call comes back empty.
That emptiness has two causes -- the index missed it, or nobody ever wrote it
-- and they call for opposite next moves: search again with different words, or
stop searching and read the implementation. Nothing else in the index can tell
them apart, because both look like zero hits.
"""

from __future__ import annotations

import time

from workspace_indexer.grounding import CoverageService, SourceStrength, UnitCoverage
from workspace_indexer.mcp.grounding_report import GroundingReport
from workspace_indexer.mcp.tool_call_recorder import ToolCallRecorder
from workspace_indexer.mcp.unknown_repository_error import UnknownRepositoryError
from workspace_indexer.models import ToolCall

_ABSENT_NOTE = (
    "No source of recorded intent was found. An empty find_guidance result here means "
    "the reason was never written down, not that the search failed -- read the "
    "implementation instead of searching again, and say the rationale is unrecorded "
    "rather than inferring one."
)
_THIN_NOTE = (
    "Grounding is sparse. A miss here is uninformative: it may simply not be written "
    "down for this part of the code. Prefer the implementation over a confident answer."
)
_PRESENT_NOTE = (
    "Grounding is well covered, so a miss is itself informative: if find_guidance "
    "returns nothing for a topic here, the decision probably was not recorded, and "
    "rephrasing the query is worth one more attempt first."
)
_EMPTY_NOTE = "Nothing is indexed yet, so there is nothing to report."

_BY_VERDICT = {
    SourceStrength.ABSENT: _ABSENT_NOTE,
    SourceStrength.THIN: _THIN_NOTE,
    SourceStrength.PRESENT: _PRESENT_NOTE,
}


class GroundingService:
    def __init__(
        self, coverage: CoverageService, *, recorder: ToolCallRecorder | None = None
    ) -> None:
        self._coverage = coverage
        self._recorder = recorder or ToolCallRecorder()

    def grounding(self, repo: str | None = None) -> GroundingReport:
        started = time.monotonic()
        units = self._coverage.coverage(only=repo)
        if repo is not None and not units:
            # Distinguished from "this repository has no grounding", which is
            # what an empty result would otherwise say -- the most misleading
            # answer this tool could give, since it is the answer it exists to
            # deliver truthfully.
            raise UnknownRepositoryError(repo, self._coverage.repository_labels())

        report = GroundingReport(repositories=units, scoped_to=repo, note=_note_for(units))
        self._record(started, repo, report)
        return report

    def _record(self, started: float, repo: str | None, report: GroundingReport) -> None:
        self._recorder.record(
            ToolCall(
                tool="grounding",
                query=repo or "",
                parameters={"repo": repo} if repo else {},
                # The labels rather than file paths: this tool answers about
                # repositories, so a recorded call replays as a repository list.
                returned_paths=[u.label for u in report.repositories],
                total_matches=len(report.repositories),
                note=report.note or None,
                duration_ms=(time.monotonic() - started) * 1000,
            )
        )


def _note_for(units: list[UnitCoverage]) -> str:
    """Guidance keyed to the weakest repository reported.

    The weakest rather than the best, which inverts how a single repository's
    own verdict is computed. Within one repository the sources are alternatives
    and the best one answers the question; across several they are separate
    codebases, and an agent told "well covered" because one of five is would
    trust the four that are not.
    """
    if not units:
        return _EMPTY_NOTE
    weakest = min(units, key=lambda u: _RANK[u.verdict])
    return _BY_VERDICT[weakest.verdict]


_RANK = {
    SourceStrength.ABSENT: 0,
    SourceStrength.THIN: 1,
    SourceStrength.PRESENT: 2,
}
