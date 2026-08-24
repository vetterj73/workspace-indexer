"""When to retry a failed embedding request, and how long to wait.

Rate limits are the normal case, not an exception: a full reindex of a large
workspace will hit 429 repeatedly, and a run that aborts there wastes every
request already paid for.
"""

from __future__ import annotations

import random

from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError

# 429 is the rate limit; 5xx is the provider having a bad moment. A 4xx that is
# not 429 means the request itself is wrong, and retrying it just spends money
# to fail again.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class RetryPolicy:
    def __init__(
        self,
        max_attempts: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: float = 0.25,
    ) -> None:
        self.max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._jitter = jitter

    @staticmethod
    def status_of(error: BaseException) -> int | None:
        if isinstance(error, ModelHTTPError):
            return error.status_code
        return None

    def should_retry(self, error: BaseException, attempt: int) -> bool:
        if attempt >= self.max_attempts:
            return False
        status = self.status_of(error)
        if status is not None:
            return status in _RETRYABLE_STATUS
        # A transport-level failure with no status: a dropped connection or a
        # timeout, both worth one more try.
        return isinstance(error, ModelAPIError | TimeoutError | ConnectionError)

    def delay_for(self, error: BaseException, attempt: int) -> float:
        """Exponential backoff, with the provider's Retry-After winning.

        Jitter matters because a reindex fires many concurrent requests: without
        it they all back off in lockstep and hit the limit again together.
        """
        retry_after = self._retry_after(error)
        if retry_after is not None:
            return min(retry_after, self._max_delay)
        delay = min(self._base_delay * (2 ** (attempt - 1)), self._max_delay)
        return delay * (1 + random.uniform(0, self._jitter))  # noqa: S311

    @staticmethod
    def _retry_after(error: BaseException) -> float | None:
        headers = getattr(error, "headers", None)
        if not headers:
            return None
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            # Retry-After may be an HTTP date. Falling back to our own backoff
            # is better than parsing date formats for a hint.
            return None
