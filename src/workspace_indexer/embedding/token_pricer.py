"""Pricing tokens when the provider will not.

`genai-prices`, which pydantic-ai delegates to, has no entry for
`voyage-code-4`, so `cost()` raises and every run records $0.0000. That reads
as "this run was free" when it means "nobody told us", and it fails in the
worst direction: silently under-reporting spend.

A configured rate is not as good as a provider-reported price -- it goes stale
without saying so -- which is why anything priced this way is reported as an
estimate rather than as a cost.
"""

from __future__ import annotations

_TOKENS_PER_MILLION = 1_000_000


class TokenPricer:
    """Converts a token count to dollars at a configured rate.

    Deliberately knows nothing about free tiers. A tier is a property of an
    *account* and is drawn down by everything using the key, including work
    this manifest never saw; pretending otherwise here would produce a
    confidently wrong number. `status` reports the drawdown separately, where
    it can be labelled with what it actually measures.
    """

    def __init__(self, price_per_mtok: float | None) -> None:
        self._price_per_mtok = price_per_mtok

    @property
    def configured(self) -> bool:
        return self._price_per_mtok is not None

    def cost_of(self, tokens: int) -> float | None:
        """Dollars for `tokens`, or None when no rate is configured.

        None rather than 0.0, because those mean opposite things and collapsing
        them is the entire bug this class exists to fix.
        """
        if self._price_per_mtok is None:
            return None
        return tokens / _TOKENS_PER_MILLION * self._price_per_mtok
