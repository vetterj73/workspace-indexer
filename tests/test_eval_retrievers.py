"""The seam that lets a *tool* be scored, not just the search path underneath.

EvalHarness.run had no direct coverage before this: the existing tests exercise
the metric arithmetic and the dataset loader, and stop at the class boundary.
Since the harness now decides which surface it is measuring, that gap is the
one place a wrong answer would go unnoticed.
"""

from __future__ import annotations

from workspace_indexer.evaluation import EvalCase, EvalHarness, ToolRetriever
from workspace_indexer.mcp import SearchResponse, SearchResult


class RecordingRetriever:
    """A stand-in that returns fixed paths and remembers how it was called."""

    def __init__(self, paths: list[str], name: str = "fake") -> None:
        self.name = name
        self._paths = paths
        self.calls: list[tuple[str, int]] = []

    async def retrieve(self, query: str, limit: int) -> list[str]:
        self.calls.append((query, limit))
        return self._paths


CASES = [
    EvalCase(query="where is auth", expect=["src/auth.py"]),
    EvalCase(query="how must modules be structured", expect=["CONVENTIONS.md"], group="guidance"),
]


async def test_harness_scores_whatever_retriever_it_is_given() -> None:
    retriever = RecordingRetriever(["src/auth.py", "other.py"])
    report = await EvalHarness(retriever).run(CASES, limit=5)

    assert len(report.results) == 2
    # First case hits at rank 1; the second finds nothing.
    assert report.results[0].first_hit_rank == 1
    assert report.results[1].first_hit_rank is None
    assert report.recall_at_k == 0.5


async def test_harness_passes_the_limit_through() -> None:
    """A tool scored at a different depth than the baseline is not a
    comparison, and the difference would be invisible in the numbers."""
    retriever = RecordingRetriever([])
    await EvalHarness(retriever).run(CASES, limit=3)
    assert [limit for _, limit in retriever.calls] == [3, 3]


async def test_harness_asks_each_case_once_in_order() -> None:
    retriever = RecordingRetriever([])
    await EvalHarness(retriever).run(CASES, limit=10)
    assert [query for query, _ in retriever.calls] == [c.query for c in CASES]


async def test_label_survives_onto_the_report() -> None:
    report = await EvalHarness(RecordingRetriever([])).run(CASES, limit=5, label="tool=x")
    assert report.label == "tool=x"


async def test_tool_retriever_unwraps_a_response_to_paths() -> None:
    async def call(query: str, limit: int) -> SearchResponse:
        assert (query, limit) == ("q", 4)
        return SearchResponse(
            results=[
                SearchResult(
                    location="a.py:1-2",
                    rel_path="a.py",
                    start_line=1,
                    end_line=2,
                    doc_type="implementation",
                ),
            ]
        )

    retriever = ToolRetriever("find_guidance", call)
    assert retriever.name == "find_guidance"
    assert await retriever.retrieve("q", 4) == ["a.py"]


async def test_tool_retriever_reports_an_empty_response_as_a_miss() -> None:
    """A tool that filtered everything out must score as a miss, not as an
    error swallowed somewhere in the harness."""

    async def call(query: str, limit: int) -> SearchResponse:
        return SearchResponse(results=[], note="filtered to nothing")

    report = await EvalHarness(ToolRetriever("t", call)).run(CASES, limit=5)
    assert report.recall_at_k == 0.0
    assert len(report.misses) == 2
