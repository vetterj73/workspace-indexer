"""Batching, ordering, retries and accounting.

Everything here guards against a failure that costs money or corrupts the index
silently: a misaligned result, a dimension mismatch, a retry storm, an input
truncated without anyone noticing.
"""

from __future__ import annotations

import pytest
import structlog.testing
from pydantic_ai.exceptions import ModelHTTPError

from tests.fake_embedding_backend import FakeEmbeddingBackend
from workspace_indexer.embedding.embedding_service import EmbeddingService
from workspace_indexer.embedding.retry_policy import RetryPolicy


def _service(backend: FakeEmbeddingBackend, **kwargs: object) -> EmbeddingService:
    kwargs.setdefault("retry", RetryPolicy(base_delay=0.0, jitter=0.0))
    return EmbeddingService(backend, **kwargs)  # type: ignore[arg-type]


async def test_empty_input_makes_no_request() -> None:
    backend = FakeEmbeddingBackend()
    assert await _service(backend).embed_documents([]) == []
    assert backend.calls == 0


async def test_splits_on_document_count() -> None:
    backend = FakeEmbeddingBackend()
    await _service(backend, batch_size=2).embed_documents(["a", "b", "c", "d", "e"])
    assert [len(b) for b in backend.batches] == [2, 2, 1]


async def test_splits_on_token_budget_even_under_the_count_limit() -> None:
    """64 chunks of 512 tokens is fine; 64 chunks that each happen to be huge
    exceeds the provider's per-request total and fails the whole batch."""
    backend = FakeEmbeddingBackend()
    big = "w" * 4000
    await _service(backend, batch_size=100, max_batch_tokens=2000).embed_documents([big] * 4)
    assert len(backend.batches) > 1


async def test_order_is_preserved_across_concurrent_batches() -> None:
    """Batches run concurrently; if results came back reordered every chunk
    would be stored with someone else's vector."""
    texts = [f"{'x' * n}" for n in range(1, 21)]
    backend = FakeEmbeddingBackend()
    vectors = await _service(backend, batch_size=3, max_concurrency=4).embed_documents(texts)
    assert [v[0] for v in vectors] == [float(len(t)) for t in texts]


async def test_concurrency_is_capped() -> None:
    backend = FakeEmbeddingBackend()
    await _service(backend, batch_size=1, max_concurrency=2).embed_documents(["a"] * 10)
    assert backend.max_concurrent <= 2


async def test_retries_a_rate_limit_then_succeeds() -> None:
    backend = FakeEmbeddingBackend(
        fail_times=2, error=ModelHTTPError(status_code=429, model_name="m")
    )
    service = _service(backend)
    vectors = await service.embed_documents(["a"])
    assert len(vectors) == 1
    assert service.stats.retries == 2
    assert service.stats.requests == 1


async def test_gives_up_after_max_attempts() -> None:
    backend = FakeEmbeddingBackend(
        fail_times=99, error=ModelHTTPError(status_code=429, model_name="m")
    )
    service = _service(backend, retry=RetryPolicy(max_attempts=3, base_delay=0.0, jitter=0.0))
    with pytest.raises(ModelHTTPError):
        await service.embed_documents(["a"])
    assert service.stats.retries == 2


async def test_client_error_fails_immediately_without_burning_retries() -> None:
    backend = FakeEmbeddingBackend(
        fail_times=99, error=ModelHTTPError(status_code=401, model_name="m")
    )
    service = _service(backend)
    with pytest.raises(ModelHTTPError):
        await service.embed_documents(["a"])
    assert service.stats.retries == 0
    assert backend.calls == 1


async def test_dimension_mismatch_fails_before_anything_is_stored() -> None:
    """The configured dimensions and the model disagree. Every vector written
    to the collection would be wrong."""
    backend = FakeEmbeddingBackend(dimensions=2048, returned_dimensions=1024)
    with pytest.raises(RuntimeError, match="1024 dimensions"):
        await _service(backend).embed_documents(["a"])


async def test_short_result_is_caught_rather_than_misaligned() -> None:
    """Silent misalignment is the worst outcome available: every chunk after
    the gap gets someone else's vector."""
    backend = FakeEmbeddingBackend(drop_last=True)
    with pytest.raises(RuntimeError, match="returned 2 vectors for 3 inputs"):
        await _service(backend, batch_size=3).embed_documents(["a", "b", "c"])


async def test_truncation_is_warned_about() -> None:
    """Silent truncation is the classic invisible quality bug: the call
    succeeds, the vector is wrong, nothing says so."""
    backend = FakeEmbeddingBackend(max_tokens=10)
    service = _service(backend)
    await service.embed_documents(["word " * 200])
    assert service.stats.truncated == 1


async def test_short_inputs_skip_the_exact_token_count() -> None:
    """Counting every chunk exactly would add an await per chunk for a check
    that almost always passes."""
    backend = FakeEmbeddingBackend(max_tokens=100_000)
    await _service(backend).embed_documents(["tiny"] * 20)
    assert backend.exact_counts == 0


async def test_no_truncation_warning_when_the_limit_is_unknown() -> None:
    backend = FakeEmbeddingBackend(max_tokens=None)
    service = _service(backend)
    await service.embed_documents(["word " * 500])
    assert service.stats.truncated == 0
    assert backend.exact_counts == 0


async def test_stats_accumulate_documents_tokens_and_cost() -> None:
    backend = FakeEmbeddingBackend(cost_per_call=0.25)
    service = _service(backend, batch_size=2)
    await service.embed_documents(["alpha", "beta", "gamma"])
    assert service.stats.documents == 3
    assert service.stats.requests == 2
    assert service.stats.tokens > 0
    assert service.stats.est_cost_usd == pytest.approx(0.5)


async def test_unpriced_requests_are_counted_separately_from_zero_cost() -> None:
    """A zero total should be distinguishable from "nobody told us"."""
    backend = FakeEmbeddingBackend(cost_per_call=None)
    service = _service(backend)
    await service.embed_documents(["a"])
    assert service.stats.est_cost_usd == 0.0
    assert service.stats.unpriced_requests == 1


async def test_query_uses_the_query_path_not_the_document_path() -> None:
    """Voyage and Cohere encode a query differently from a document, and using
    the document path for a query measurably degrades retrieval."""
    backend = FakeEmbeddingBackend()
    await _service(backend).embed_query("how does auth work")
    assert backend.queries == ["how does auth work"]
    assert backend.batches == []


async def test_query_retries_too() -> None:
    backend = FakeEmbeddingBackend(
        fail_times=1, error=ModelHTTPError(status_code=503, model_name="m")
    )
    service = _service(backend)
    assert await service.embed_query("q")
    assert service.stats.retries == 1


async def test_space_is_exposed_for_collection_naming() -> None:
    backend = FakeEmbeddingBackend(dimensions=8)
    assert _service(backend).space.dimensions == 8


# --- issue #3: telling the two truncations apart -------------------------


def _long(word: str = "word") -> str:
    """Comfortably past a 10-token limit, and identifiable in the preview."""
    return f"{word} " * 200


async def test_truncation_of_an_indivisible_chunk_is_reported_as_the_tradeoff() -> None:
    """A warning that fires on a known, accepted tradeoff is a warning that
    gets filtered out -- and it takes the real ones with it."""
    service = _service(FakeEmbeddingBackend(max_tokens=10))

    with structlog.testing.capture_logs() as logs:
        await service.embed_documents([_long()], indivisible=[True])

    truncated = [e for e in logs if e["event"] == "embed.truncated"]
    assert len(truncated) == 1
    assert truncated[0]["log_level"] == "info"
    assert truncated[0]["cause"] == "indivisible_block"
    # Still counted: the vector is still partial, whatever the reason.
    assert service.stats.truncated == 1


async def test_truncation_of_a_divisible_chunk_stays_a_warning() -> None:
    """The chunker produced something larger than its own budget. That is a
    defect and has to keep looking like one."""
    service = _service(FakeEmbeddingBackend(max_tokens=10))

    with structlog.testing.capture_logs() as logs:
        await service.embed_documents([_long()], indivisible=[False])

    truncated = [e for e in logs if e["event"] == "embed.truncated"]
    assert len(truncated) == 1
    assert truncated[0]["log_level"] == "warning"
    assert truncated[0]["cause"] == "chunker_overshoot"


async def test_an_unlabelled_truncation_is_reported_as_the_louder_case() -> None:
    """Callers that say nothing keep the old behaviour.

    Unknown must not be silently optimistic: a real chunker defect reported at
    info is a defect nobody sees.
    """
    service = _service(FakeEmbeddingBackend(max_tokens=10))

    with structlog.testing.capture_logs() as logs:
        await service.embed_documents([_long()])

    truncated = [e for e in logs if e["event"] == "embed.truncated"]
    assert truncated and truncated[0]["log_level"] == "warning"


async def test_the_flags_are_matched_to_their_own_text() -> None:
    """Position matters: mislabelling would report the wrong chunk's cause."""
    service = _service(FakeEmbeddingBackend(max_tokens=10))

    with structlog.testing.capture_logs() as logs:
        await service.embed_documents([_long("alpha"), _long("beta")], indivisible=[True, False])

    causes = {
        str(e["preview"]).split()[0]: e["cause"] for e in logs if e["event"] == "embed.truncated"
    }
    assert causes == {"alpha": "indivisible_block", "beta": "chunker_overshoot"}
