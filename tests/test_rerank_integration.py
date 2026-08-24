"""A real cross-encoder, no API key.

The point of this module is that the abstraction is proven rather than assumed:
a protocol with one implementation is a guess. It also answers the question the
whole reranking layer exists for — does reranking actually improve the order —
without spending anything.

Marked `integration` because the first run downloads the ONNX model (cached
after). Needs no key, no credentials, no paid request.
"""

from __future__ import annotations

import pytest

from workspace_indexer.config import RerankConfig
from workspace_indexer.models import FileKind, SearchHit
from workspace_indexer.rerank.local_cross_encoder_reranker import LocalCrossEncoderReranker
from workspace_indexer.rerank.reranker import Reranker

pytestmark = pytest.mark.integration

ROLLBACK = (
    "To roll back a failed deployment, run scripts/rollback.sh with the previous "
    "release tag, then confirm the health checks pass before paging anyone."
)
CAKE = (
    "Cream the butter and sugar together, fold in the flour, and bake the sponge "
    "for forty minutes until the top is golden."
)
AUTH = (
    "Authentication verifies the bearer token on every request and rejects "
    "expired credentials before the handler runs."
)


def _hit(index: int, path: str, body: str) -> SearchHit:
    return SearchHit(
        chunk_id=f"id-{index}",
        # Deliberately descending, so the incoming order is CAKE first: the
        # reranker has to overcome it rather than agree with it.
        score=1.0 - index / 10,
        rel_path=path,
        root_label="root",
        kind=FileKind.MARKDOWN,
        source_text=body,
        embed_text=f"# file: {path}\n{body}",
    )


@pytest.fixture(scope="module")
def reranker() -> LocalCrossEncoderReranker:
    return LocalCrossEncoderReranker(RerankConfig(model="fastembed:x"))


async def test_it_satisfies_the_protocol(reranker: LocalCrossEncoderReranker) -> None:
    assert isinstance(reranker, Reranker)


async def test_reranking_fixes_a_wrong_order(reranker: LocalCrossEncoderReranker) -> None:
    """The payoff: retrieval handed us the cake recipe first, and the reranker
    has to promote the document that actually answers the question."""
    hits = [
        _hit(0, "docs/cake.md", CAKE),
        _hit(1, "docs/auth.md", AUTH),
        _hit(2, "docs/rollback.md", ROLLBACK),
    ]
    ranked = await reranker.rerank("how do I undo a bad release", hits, top_n=3)
    assert ranked[0].rel_path == "docs/rollback.md"


async def test_scores_are_attached_and_ordered(reranker: LocalCrossEncoderReranker) -> None:
    hits = [_hit(0, "docs/cake.md", CAKE), _hit(1, "docs/rollback.md", ROLLBACK)]
    ranked = await reranker.rerank("rolling back a deploy", hits, top_n=2)
    scores = [h.rerank_score for h in ranked]
    assert all(s is not None for s in scores)
    assert scores == sorted(scores, key=lambda s: s or 0.0, reverse=True)


async def test_churn_is_recorded_when_the_top_changes(
    reranker: LocalCrossEncoderReranker,
) -> None:
    hits = [_hit(0, "docs/cake.md", CAKE), _hit(1, "docs/rollback.md", ROLLBACK)]
    before = reranker.stats.unchanged_top
    await reranker.rerank("how do I undo a bad release", hits, top_n=2)
    assert reranker.stats.unchanged_top == before


async def test_local_inference_is_free_not_unknown(
    reranker: LocalCrossEncoderReranker,
) -> None:
    assert reranker.cost_of_last_call() == 0.0


async def test_candidate_cap_is_honoured_against_a_real_model() -> None:
    reranker = LocalCrossEncoderReranker(RerankConfig(model="fastembed:x", candidates=2))
    hits = [
        _hit(0, "docs/cake.md", CAKE),
        _hit(1, "docs/auth.md", AUTH),
        _hit(2, "docs/rollback.md", ROLLBACK),
    ]
    ranked = await reranker.rerank("how do I undo a bad release", hits, top_n=5)
    # rollback was candidate 3 and therefore never scored.
    assert len(ranked) == 2
    assert "docs/rollback.md" not in [h.rel_path for h in ranked]
