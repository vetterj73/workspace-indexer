"""Retry classification and backoff.

Rate limits are the normal case on a full reindex, not an exception. Getting
the classification wrong either aborts a run that would have succeeded, or
spends money retrying a request that can never succeed.
"""

from __future__ import annotations

import pytest
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError

from workspace_indexer.embedding.retry_policy import RetryPolicy


def _http(status: int, headers: dict[str, str] | None = None) -> ModelHTTPError:
    return ModelHTTPError(status_code=status, model_name="m", body=None, headers=headers)


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_transient_statuses_are_retried(status: int) -> None:
    assert RetryPolicy().should_retry(_http(status), attempt=1)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retried(status: int) -> None:
    """A malformed request or a bad key will fail identically forever, and
    each retry is billable."""
    assert not RetryPolicy().should_retry(_http(status), attempt=1)


def test_transport_failures_are_retried() -> None:
    assert RetryPolicy().should_retry(ConnectionError("reset"), attempt=1)
    assert RetryPolicy().should_retry(TimeoutError(), attempt=1)
    assert RetryPolicy().should_retry(ModelAPIError("m", "socket closed"), attempt=1)


def test_unrelated_errors_are_not_retried() -> None:
    """A bug in our own code should surface, not be retried five times."""
    assert not RetryPolicy().should_retry(ValueError("our bug"), attempt=1)


def test_attempts_are_capped() -> None:
    policy = RetryPolicy(max_attempts=3)
    assert policy.should_retry(_http(429), attempt=2)
    assert not policy.should_retry(_http(429), attempt=3)


def test_backoff_grows_exponentially() -> None:
    policy = RetryPolicy(base_delay=1.0, jitter=0.0)
    delays = [policy.delay_for(_http(429), attempt=n) for n in (1, 2, 3, 4)]
    assert delays == [1.0, 2.0, 4.0, 8.0]


def test_backoff_is_capped() -> None:
    policy = RetryPolicy(base_delay=1.0, max_delay=5.0, jitter=0.0)
    assert policy.delay_for(_http(429), attempt=10) == 5.0


def test_jitter_stays_within_bounds() -> None:
    """Without jitter, concurrent requests back off in lockstep and hit the
    limit again together."""
    policy = RetryPolicy(base_delay=1.0, jitter=0.5)
    delays = {policy.delay_for(_http(429), attempt=1) for _ in range(50)}
    assert all(1.0 <= d <= 1.5 for d in delays)
    assert len(delays) > 1


def test_retry_after_header_wins_over_our_backoff() -> None:
    """The provider knows when it will accept traffic again; guessing shorter
    just burns another request."""
    policy = RetryPolicy(base_delay=1.0, jitter=0.0)
    assert policy.delay_for(_http(429, {"retry-after": "17"}), attempt=1) == 17.0


def test_retry_after_is_still_capped() -> None:
    policy = RetryPolicy(max_delay=30.0)
    assert policy.delay_for(_http(429, {"retry-after": "9999"}), attempt=1) == 30.0


def test_unparseable_retry_after_falls_back_to_backoff() -> None:
    """Retry-After may be an HTTP date; our own backoff beats parsing formats
    for a hint."""
    policy = RetryPolicy(base_delay=2.0, jitter=0.0)
    header = {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}
    assert policy.delay_for(_http(429, header), attempt=1) == 2.0


def test_status_of_reports_none_for_non_http_errors() -> None:
    assert RetryPolicy.status_of(_http(429)) == 429
    assert RetryPolicy.status_of(ValueError()) is None
