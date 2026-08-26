"""Packing hits into a token budget without lying about what was cut."""

from __future__ import annotations

from workspace_indexer.mcp import ResultBudget
from workspace_indexer.models import DocumentType, SearchHit


def _hit(index: int, tokens: int, text: str | None = None) -> SearchHit:
    return SearchHit(
        chunk_id=f"id-{index}",
        score=1.0 - index / 100,
        rel_path=f"src/file{index}.py",
        root_label="r",
        doc_type=DocumentType.IMPLEMENTATION,
        start_line=index * 10 + 1,
        end_line=index * 10 + 9,
        source_text=text if text is not None else "x" * (tokens * 4),
        token_count=tokens,
    )


def test_everything_fits_when_the_budget_allows() -> None:
    results, dropped = ResultBudget(1000).pack([_hit(i, 100) for i in range(5)])
    assert len(results) == 5
    assert dropped == 0
    assert all(not r.text_truncated for r in results)


def test_overflow_drops_whole_hits_and_counts_them() -> None:
    """Three complete chunks beat ten fragments, so whole hits go rather than
    every hit being shaved to uselessness."""
    results, dropped = ResultBudget(250, min_chunk_tokens=64).pack([_hit(i, 100) for i in range(5)])
    assert len(results) + dropped == 5
    assert dropped > 0
    # The ones kept are the highest ranked, in order.
    assert [r.rel_path for r in results] == [f"src/file{i}.py" for i in range(len(results))]


def test_a_single_oversized_chunk_is_truncated_and_flagged() -> None:
    """An agent that thinks it read a whole function will confidently describe
    the half it got."""
    results, _ = ResultBudget(100).pack([_hit(0, 5000)])
    assert len(results) == 1
    assert results[0].text_truncated
    assert len(results[0].text) < 5000 * 4
    assert "truncated" in results[0].text


def test_truncation_cuts_on_a_line_boundary() -> None:
    body = "\n".join(f"line {i} of the function body" for i in range(200))
    results, _ = ResultBudget(40).pack([_hit(0, 2000, text=body)])
    kept = results[0].text.split("\n... ")[0]
    assert kept
    # Every retained line is whole, so the result is still readable code.
    assert all(line in body.splitlines() for line in kept.splitlines())


def test_location_is_always_anchored() -> None:
    results, _ = ResultBudget(1000).pack([_hit(3, 10)])
    assert results[0].location == "src/file3.py:31-39"


def test_staleness_survives_packing() -> None:
    """An agent editing from stale text writes a patch that will not apply, so
    this flag must never be dropped in formatting."""
    hit = _hit(0, 10)
    hit.stale = True
    results, _ = ResultBudget(1000).pack([hit])
    assert results[0].stale is True


def test_rerank_score_wins_when_present() -> None:
    """The displayed score should be the one that decided the order."""
    hit = _hit(0, 10)
    hit.rerank_score = 0.42
    results, _ = ResultBudget(1000).pack([hit])
    assert results[0].score == 0.42


def test_missing_token_count_is_estimated_not_treated_as_free() -> None:
    """A zero token_count on every hit would make the budget unbounded."""
    hit = _hit(0, 0, text="y" * 8000)
    results, dropped = ResultBudget(100).pack([hit, _hit(1, 50)])
    assert results[0].text_truncated
    assert dropped == 1


def test_empty_input_is_not_an_error() -> None:
    assert ResultBudget(100).pack([]) == ([], 0)


def test_the_top_hit_survives_a_budget_too_small_to_hold_it() -> None:
    """Returning nothing is indistinguishable from "no matches", and sends the
    agent looking elsewhere for something we actually found."""
    results, dropped = ResultBudget(10, min_chunk_tokens=64).pack([_hit(i, 500) for i in range(3)])
    assert len(results) == 1
    assert results[0].text_truncated
    assert dropped == 2
