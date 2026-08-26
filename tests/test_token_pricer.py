"""Pricing tokens when the provider will not.

The distinction under test is not arithmetic -- it is that "free", "estimated"
and "unknown" are three different answers, and that collapsing them into
`$0.0000` reports the wrong one in the expensive direction.
"""

from __future__ import annotations

import pytest

from workspace_indexer.embedding.token_pricer import TokenPricer


def test_no_rate_configured_returns_none_not_zero() -> None:
    """None means "nobody told us", 0.0 means "free". Collapsing them is the
    whole bug."""
    assert TokenPricer(None).cost_of(1_000_000) is None
    assert TokenPricer(None).configured is False


def test_a_configured_rate_prices_a_million_tokens() -> None:
    assert TokenPricer(0.12).cost_of(1_000_000) == pytest.approx(0.12)


def test_pricing_is_linear_below_and_above_a_million() -> None:
    pricer = TokenPricer(0.12)
    assert pricer.cost_of(500_000) == pytest.approx(0.06)
    assert pricer.cost_of(3_190_000) == pytest.approx(0.3828)


def test_zero_tokens_costs_zero_when_a_rate_is_known() -> None:
    """Distinct from the None above: we know the rate and know it was free."""
    assert TokenPricer(0.12).cost_of(0) == 0.0


def test_a_free_provider_can_be_priced_at_zero() -> None:
    """A rate of 0.0 is a real answer, not a missing one."""
    pricer = TokenPricer(0.0)
    assert pricer.configured is True
    assert pricer.cost_of(5_000_000) == 0.0
