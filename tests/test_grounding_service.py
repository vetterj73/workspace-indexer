"""Shaping coverage into an answer an agent can act on.

Stubbed coverage rather than a real manifest: what is under test here is the
framing -- which note is chosen, and whether a typo can masquerade as a finding
-- and the measurement itself is covered against real repositories in
test_coverage_service.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from workspace_indexer.grounding import CoverageService, GroundingSource, UnitCoverage
from workspace_indexer.mcp import GroundingService, UnknownRepositoryError
from workspace_indexer.mcp.tool_call_recorder import ToolCallRecorder
from workspace_indexer.mcp.tool_call_sink import ToolCallSink
from workspace_indexer.models import ToolCall


class StubCoverage(CoverageService):
    def __init__(self, units: list[UnitCoverage]) -> None:  # pyright: ignore[reportMissingSuperCall]
        # Deliberately does not call super().__init__: the base needs a manifest
        # to measure, and nothing here measures anything.
        self._units = units

    def coverage(self, only: str | None = None) -> list[UnitCoverage]:
        return [u for u in self._units if only is None or u.label == only]

    def repository_labels(self) -> list[str]:
        return sorted(u.label for u in self._units)


class Recorded(ToolCallSink):
    def __init__(self) -> None:
        self.calls: list[ToolCall] = []

    def record_tool_call(self, call: ToolCall) -> None:
        self.calls.append(call)


def unit(label: str, *, design: int = 0, code: int = 100) -> UnitCoverage:
    return UnitCoverage(
        label=label,
        code_files=code,
        sources=[GroundingSource.by_density("design docs", design, code, detail="")],
    )


def test_a_repository_with_no_recorded_intent_is_told_not_to_infer_one() -> None:
    """The whole reason the tool exists.

    An agent that reads "absent" and then writes a plausible rationale has done
    the one thing this is meant to prevent, so the instruction is in the
    payload rather than left to inference.
    """
    service = GroundingService(StubCoverage([unit("api", design=0)]))

    report = service.grounding()

    assert "never written down" in report.note
    assert "unrecorded" in report.note


def test_a_well_covered_repository_says_a_miss_is_informative() -> None:
    service = GroundingService(StubCoverage([unit("api", design=50)]))

    report = service.grounding()

    assert "informative" in report.note


def test_the_note_follows_the_weakest_repository_not_the_best() -> None:
    """Across repositories the sources are not alternatives.

    Within one repository the best source answers the question, so the verdict
    is the maximum. Across several they are separate codebases, and an agent
    told "well covered" because one of two is would trust the one that is not.
    """
    service = GroundingService(StubCoverage([unit("rich", design=50), unit("bare", design=0)]))

    report = service.grounding()

    assert "never written down" in report.note


def test_scoping_narrows_to_one_repository_and_says_so() -> None:
    service = GroundingService(StubCoverage([unit("api", design=50), unit("web", design=0)]))

    report = service.grounding("api")

    assert [u.label for u in report.repositories] == ["api"]
    assert report.scoped_to == "api"
    # Scoped to the covered one, so the weakest-of-those is that one.
    assert "informative" in report.note


def test_an_unknown_repository_is_an_error_naming_the_real_ones() -> None:
    """Never an empty result.

    Empty from this tool reads as "this repository records no reasons" -- the
    strongest claim it can make. A typo must not manufacture it.
    """
    service = GroundingService(StubCoverage([unit("api"), unit("web")]))

    with pytest.raises(UnknownRepositoryError) as caught:
        service.grounding("ap")

    assert "api" in str(caught.value)
    assert "web" in str(caught.value)


def test_an_empty_index_is_reported_as_such_rather_than_as_absent_grounding() -> None:
    """Nothing indexed is a fact about the index, not about any codebase."""
    service = GroundingService(StubCoverage([]))

    report = service.grounding()

    assert report.repositories == []
    assert "Nothing is indexed" in report.note


def test_the_call_is_recorded_with_the_repositories_it_named() -> None:
    sink = Recorded()
    service = GroundingService(
        StubCoverage([unit("api"), unit("web")]), recorder=ToolCallRecorder(sink)
    )

    service.grounding()

    assert len(sink.calls) == 1
    assert sink.calls[0].tool == "grounding"
    assert set(sink.calls[0].returned_paths) == {"api", "web"}


def test_an_unknown_repository_records_nothing() -> None:
    """A rejected call is not a call the index answered.

    Recording it would put a repository that does not exist into the harvest of
    queries worth turning into eval cases.
    """
    sink = Recorded()
    service = GroundingService(StubCoverage([unit("api")]), recorder=ToolCallRecorder(sink))

    with pytest.raises(UnknownRepositoryError):
        service.grounding("nope")

    assert sink.calls == []


def test_the_verdict_and_notes_survive_serialisation() -> None:
    """Both are computed properties, and the agent only ever sees the JSON.

    Without `@computed_field` they exist in Python and vanish on the wire,
    leaving the agent the raw counts and none of the interpretation.
    """
    service = GroundingService(StubCoverage([unit("api", design=0)]))

    payload = service.grounding().model_dump(mode="json")

    first = payload["repositories"][0]
    assert first["verdict"] == "absent"
    assert isinstance(first["notes"], list)
    assert first["notes"]


def test_a_stale_repository_is_not_reported_as_undocumented(tmp_path: Path) -> None:
    """Guards the distinction end to end, through the MCP shape.

    A moved workspace makes every source unreadable, which looks exactly like a
    codebase that documents nothing.
    """
    moved = UnitCoverage(
        label="moved",
        code_files=10,
        sources=[GroundingSource.by_density("design docs", 0, 10, detail="")],
        on_disk=False,
    )
    service = GroundingService(StubCoverage([moved]))

    report = service.grounding()

    assert any("reindex" in note for note in report.repositories[0].notes)
