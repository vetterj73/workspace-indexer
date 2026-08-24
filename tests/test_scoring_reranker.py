"""The shared reranking template.

Everything providers would otherwise copy-paste: candidate capping, text
selection, instruction prepending, re-sorting, the degrade paths, and the churn
accounting that tells you whether reranking is earning its round trip.
"""

from __future__ import annotations

import pytest

from tests.fake_scoring_reranker import FakeScoringReranker
from workspace_indexer.config import RerankConfig
from workspace_indexer.models import FileKind, SearchHit


def _hits(count: int) -> list[SearchHit]:
    return [
        SearchHit(
            chunk_id=f"id-{i}",
            score=1.0 - i / 100,
            rel_path=f"src/f{i}.py",
            root_label="root",
            kind=FileKind.CODE,
            source_text=f"body {i}",
            embed_text=f"# file: src/f{i}.py\nbody {i}",
        )
        for i in range(count)
    ]


def _config(**overrides: object) -> RerankConfig:
    base: dict[str, object] = {"model": "fake:model"}
    base.update(overrides)
    return RerankConfig(**base)  # type: ignore[arg-type]


async def test_empty_hits_short_circuits() -> None:
    reranker = FakeScoringReranker(_config())
    assert await reranker.rerank("q", [], 10) == []
    assert reranker.seen_queries == []


async def test_reordering_actually_happens() -> None:
    """A no-op bug would pass every other test in this file."""
    reranker = FakeScoringReranker(_config(), reverse=True)
    ranked = await reranker.rerank("q", _hits(4), top_n=4)
    assert [h.chunk_id for h in ranked] == ["id-3", "id-2", "id-1", "id-0"]


async def test_scores_are_attached_to_the_hits() -> None:
    reranker = FakeScoringReranker(_config(), scores=[0.1, 0.9])
    ranked = await reranker.rerank("q", _hits(2), top_n=2)
    assert ranked[0].chunk_id == "id-1"
    assert ranked[0].rerank_score == pytest.approx(0.9)
    # The original fusion score survives, so both are available to a caller.
    assert ranked[0].score == pytest.approx(0.99)


async def test_candidates_cap_what_is_sent() -> None:
    """Reranking 500 candidates costs latency on every search for recall the
    fusion stage already decided it did not have."""
    reranker = FakeScoringReranker(_config(candidates=3))
    await reranker.rerank("q", _hits(10), top_n=5)
    assert len(reranker.seen_documents[0]) == 3


async def test_top_n_truncates_the_result() -> None:
    reranker = FakeScoringReranker(_config(), reverse=True)
    assert len(await reranker.rerank("q", _hits(10), top_n=3)) == 3


async def test_embed_text_is_scored_by_default() -> None:
    """The reranker benefits from the same context header the embedder got; a
    bare method body is ambiguous."""
    reranker = FakeScoringReranker(_config())
    await reranker.rerank("q", _hits(1), top_n=1)
    assert reranker.seen_documents[0][0].startswith("# file: src/f0.py")


async def test_source_text_can_be_scored_instead() -> None:
    reranker = FakeScoringReranker(_config(rerank_text="source_text"))
    await reranker.rerank("q", _hits(1), top_n=1)
    assert reranker.seen_documents[0] == ["body 0"]


async def test_falls_back_to_source_when_embed_text_is_missing() -> None:
    """A point written before context headers existed still has to rerank."""
    hit = _hits(1)[0].model_copy(update={"embed_text": ""})
    reranker = FakeScoringReranker(_config())
    await reranker.rerank("q", [hit], top_n=1)
    assert reranker.seen_documents[0] == ["body 0"]


async def test_instruction_is_prepended_to_the_query() -> None:
    """rerank-2.5* follow instructions but expose no instruction parameter."""
    reranker = FakeScoringReranker(_config(instruction="Prefer implementations."))
    await reranker.rerank("how does auth work", _hits(1), top_n=1)
    assert reranker.seen_queries[0] == "Prefer implementations.\nhow does auth work"


async def test_no_instruction_leaves_the_query_alone() -> None:
    reranker = FakeScoringReranker(_config())
    await reranker.rerank("plain query", _hits(1), top_n=1)
    assert reranker.seen_queries[0] == "plain query"


async def test_blank_instruction_is_not_prepended() -> None:
    reranker = FakeScoringReranker(_config(instruction="   "))
    await reranker.rerank("plain query", _hits(1), top_n=1)
    assert reranker.seen_queries[0] == "plain query"


async def test_api_failure_degrades_to_the_fusion_order() -> None:
    """Results get worse; nothing errors."""
    reranker = FakeScoringReranker(_config(), error=RuntimeError("503"))
    ranked = await reranker.rerank("q", _hits(5), top_n=3)
    assert [h.chunk_id for h in ranked] == ["id-0", "id-1", "id-2"]
    assert reranker.stats.degraded == 1


async def test_on_error_fail_raises_instead() -> None:
    """The eval harness wants this: a silent degradation there would quietly
    corrupt a measurement."""
    reranker = FakeScoringReranker(_config(on_error="fail"), error=RuntimeError("503"))
    with pytest.raises(RuntimeError, match="503"):
        await reranker.rerank("q", _hits(5), top_n=3)
    assert reranker.stats.degraded == 1


async def test_mismatched_score_count_is_treated_as_a_failure() -> None:
    """Silently misaligned scores would reorder results by nothing at all."""
    reranker = FakeScoringReranker(_config(), wrong_length=True)
    ranked = await reranker.rerank("q", _hits(4), top_n=4)
    assert [h.chunk_id for h in ranked] == ["id-0", "id-1", "id-2", "id-3"]
    assert reranker.stats.degraded == 1


async def test_equal_scores_keep_the_fusion_order() -> None:
    """A stable sort, so identical runs do not shuffle results."""
    reranker = FakeScoringReranker(_config(), scores=[0.5, 0.5, 0.5])
    ranked = await reranker.rerank("q", _hits(3), top_n=3)
    assert [h.chunk_id for h in ranked] == ["id-0", "id-1", "id-2"]


async def test_unchanged_top_is_tracked() -> None:
    """Consistently zero churn means we pay a round trip per search and change
    nothing, which the log should be able to tell you."""
    reranker = FakeScoringReranker(_config(), scores=[0.9, 0.1, 0.05])
    await reranker.rerank("q", _hits(3), top_n=3)
    assert reranker.stats.unchanged_top == 1

    mover = FakeScoringReranker(_config(), scores=[0.1, 0.9, 0.05])
    await mover.rerank("q", _hits(3), top_n=3)
    assert mover.stats.unchanged_top == 0


async def test_stats_accumulate() -> None:
    reranker = FakeScoringReranker(_config(), cost=0.002)
    await reranker.rerank("q", _hits(4), top_n=2)
    await reranker.rerank("q", _hits(4), top_n=2)
    assert reranker.stats.calls == 2
    assert reranker.stats.documents == 8
    assert reranker.stats.est_cost_usd == pytest.approx(0.004)


async def test_unknown_cost_is_not_counted_as_zero() -> None:
    reranker = FakeScoringReranker(_config(), cost=None)
    await reranker.rerank("q", _hits(2), top_n=2)
    assert reranker.stats.est_cost_usd == 0.0
    assert reranker.stats.calls == 1
