"""Where a run's cost comes from, and whether we admit not knowing.

Four real index runs against `voyageai:voyage-code-4` each reported `$0.0000`
while embedding ~3.15M tokens. Nothing was wrong with the arithmetic: the
provider is unpriced, `cost()` raises, and the distinction between "free" and
"nobody told us" was thrown away between EmbeddingStats and RunStats.
"""

from __future__ import annotations

import pytest
import structlog.testing

from tests.fake_embedding_backend import FakeEmbeddingBackend
from workspace_indexer.embedding.embedding_service import EmbeddingService
from workspace_indexer.embedding.token_pricer import TokenPricer

TEXTS = ["def a(): return 1", "def b(): return 2"]


def _service(
    *,
    cost: float | None,
    tokens: int | None = None,
    price_per_mtok: float | None = None,
) -> EmbeddingService:
    return EmbeddingService(
        FakeEmbeddingBackend(dimensions=4, cost_per_call=cost, tokens_per_call=tokens),
        pricer=TokenPricer(price_per_mtok),
    )


# --- which of the three states a run lands in ----------------------------


async def test_a_provider_reported_price_is_a_cost_not_an_estimate() -> None:
    service = _service(cost=0.25)
    await service.embed_documents(TEXTS)

    assert service.stats.est_cost_usd == pytest.approx(0.25)
    assert service.stats.unpriced_requests == 0
    assert service.stats.cost_is_estimate is False
    assert service.stats.cost_is_known is True


async def test_an_unpriced_provider_with_no_configured_rate_is_unknown() -> None:
    """The reported bug. A zero here would read as "this run was free"."""
    service = _service(cost=None)
    await service.embed_documents(TEXTS)

    assert service.stats.unpriced_requests == 1
    assert service.stats.est_cost_usd == 0.0
    assert service.stats.cost_is_known is False


async def test_a_configured_rate_prices_an_unpriced_provider() -> None:
    service = _service(cost=None, tokens=1_000_000, price_per_mtok=0.12)
    await service.embed_documents(TEXTS)

    assert service.stats.est_cost_usd == pytest.approx(0.12)
    assert service.stats.cost_is_estimate is True
    # Priced, so not unknown -- but flagged, because a config rate goes stale.
    assert service.stats.unpriced_requests == 0
    assert service.stats.cost_is_known is True


async def test_a_free_local_backend_is_not_the_same_as_unpriced() -> None:
    """Local inference genuinely costs nothing. `$0.0000` is the right answer
    here and the wrong one for an unpriced API."""
    service = _service(cost=0.0)
    await service.embed_documents(TEXTS)

    assert service.stats.est_cost_usd == 0.0
    assert service.stats.unpriced_requests == 0
    assert service.stats.cost_is_known is True


# --- token counting ------------------------------------------------------


async def test_the_providers_token_count_wins_over_our_estimate() -> None:
    """Ours runs 13-22% high per call against voyage-code-4 and 45% high
    cumulatively -- far too loose to bill or budget against."""
    service = _service(cost=None, tokens=159)
    await service.embed_documents(TEXTS)

    assert service.stats.tokens == 159
    assert service.stats.estimated_token_requests == 0


async def test_we_fall_back_to_estimating_when_the_provider_is_silent() -> None:
    service = _service(cost=None, tokens=None)
    await service.embed_documents(TEXTS)

    assert service.stats.tokens > 0
    # Flagged, so a total built from estimates is not mistaken for a measurement.
    assert service.stats.estimated_token_requests == 1


# --- saying so during the run --------------------------------------------


async def test_an_unpriced_run_warns_once_not_once_per_batch() -> None:
    """A full index is hundreds of batches. A thousand copies of the same
    warning is how a real one gets missed."""
    service = _service(cost=None)
    with structlog.testing.capture_logs() as logs:
        for _ in range(5):
            await service.embed_documents(TEXTS)

    warnings = [entry for entry in logs if entry["event"] == "embed.unpriced"]
    assert len(warnings) == 1
    assert "EMBEDDING_PRICE_PER_MTOK" in str(warnings[0]["detail"])


async def test_no_unpriced_warning_when_the_provider_reports_a_price() -> None:
    service = _service(cost=0.25)
    with structlog.testing.capture_logs() as logs:
        await service.embed_documents(TEXTS)

    assert not [entry for entry in logs if entry["event"] == "embed.unpriced"]


async def test_no_unpriced_warning_when_config_can_price_it() -> None:
    service = _service(cost=None, price_per_mtok=0.12)
    with structlog.testing.capture_logs() as logs:
        await service.embed_documents(TEXTS)

    assert not [entry for entry in logs if entry["event"] == "embed.unpriced"]


# --- merging across services ---------------------------------------------


async def test_merge_carries_every_price_field() -> None:
    """Stats merge across concurrent services; a field that does not merge
    silently reports zero for all but one of them."""
    priced = _service(cost=0.25)
    unpriced = _service(cost=None)
    await priced.embed_documents(TEXTS)
    await unpriced.embed_documents(TEXTS)

    priced.stats.merge(unpriced.stats)
    assert priced.stats.unpriced_requests == 1
    assert priced.stats.est_cost_usd == pytest.approx(0.25)
    assert priced.stats.cost_is_known is False
