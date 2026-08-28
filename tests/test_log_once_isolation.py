"""That a once-per-process log event does not leak between tests.

Its own module deliberately. `test_logging_setup.py` has an autouse fixture
calling `reset_for_tests()`, which clears the ledger as a side effect and so
hides the very leak this is about -- the hazard is *across* modules, where
nothing local resets anything.

`log_once` dedupes for the life of the process, which is correct in production
and wrong across a test session: whichever test runs first consumes the event,
and any later test asserting on it fails on ordering alone. Two tests already
depend on this (`rerank.skipped` in test_logging_setup, and
`store.search_indexes_unavailable` in test_mongo_store); both were one
reordering away from flaking. Same family as #47 -- state surviving between
tests -- and the autouse fixture in conftest is what stops it.

The two tests below share a key on purpose. Without that fixture the second
one to run fails, whichever order they run in.
"""

from __future__ import annotations

import structlog.testing

from workspace_indexer.obs.logging import get_logger, log_once

SHARED_KEY = "probe:shared-once-key"


def _emit() -> list[str]:
    log = get_logger("workspace_indexer.probe")
    with structlog.testing.capture_logs() as captured:
        log_once(log, SHARED_KEY, "probe.event")
    return [entry["event"] for entry in captured]


def test_a_once_only_event_fires_for_this_test() -> None:
    assert _emit() == ["probe.event"]


def test_the_same_once_only_event_fires_again_for_the_next_test() -> None:
    """The assertion the ledger reset exists for."""
    assert _emit() == ["probe.event"]


def test_a_repeat_within_one_test_is_still_suppressed() -> None:
    """The reset must not defeat what log_once is for: inside a single run the
    second call is still silence, which is the whole point of the mechanism."""
    log = get_logger("workspace_indexer.probe")
    with structlog.testing.capture_logs() as captured:
        log_once(log, SHARED_KEY, "probe.event")
        log_once(log, SHARED_KEY, "probe.event")
    assert [entry["event"] for entry in captured] == ["probe.event"]
