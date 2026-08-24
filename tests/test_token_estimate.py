"""Token estimation.

Estimates, not exact counts: the real tokenizer is an async call on the
embedding backend and calling it per candidate boundary would dominate the cost
of chunking. What matters here is that the estimate is monotonic, never zero
for real text, and that the bytes/tokens conversion is not off by a factor of
three — which is the mistake that would silently produce chunks three times
larger than the model accepts.
"""

from __future__ import annotations

from workspace_indexer.chunking.token_estimate import (
    bytes_per_token,
    estimate_tokens,
    tokens_to_bytes,
)
from workspace_indexer.models import FileKind


def test_empty_text_is_zero() -> None:
    assert estimate_tokens("", FileKind.CODE) == 0


def test_short_text_never_estimates_zero() -> None:
    """A zero would make min_tokens filtering drop every chunk."""
    assert estimate_tokens("x", FileKind.CODE) >= 1


def test_monotonic_in_length() -> None:
    small = estimate_tokens("a" * 100, FileKind.TEXT)
    large = estimate_tokens("a" * 1000, FileKind.TEXT)
    assert large > small


def test_code_estimates_more_tokens_than_prose_for_equal_bytes() -> None:
    """Identifiers, punctuation and indentation fragment more than prose, so
    the same byte count is more tokens in code."""
    body = "x" * 1000
    assert estimate_tokens(body, FileKind.CODE) > estimate_tokens(body, FileKind.MARKDOWN)


def test_unknown_kind_falls_back_to_a_sane_divisor() -> None:
    assert 3.0 < bytes_per_token(FileKind.IMAGE) < 5.0


def test_multibyte_counts_bytes_not_characters() -> None:
    """A tokenizer sees bytes; ten emoji are not ten characters' worth."""
    assert estimate_tokens("🙂" * 10, FileKind.TEXT) > estimate_tokens("a" * 10, FileKind.TEXT)


def test_round_trip_is_approximately_stable() -> None:
    for kind in (FileKind.CODE, FileKind.MARKDOWN, FileKind.TEXT):
        budget = tokens_to_bytes(512, kind)
        recovered = estimate_tokens("a" * budget, kind)
        assert abs(recovered - 512) <= 1, kind


def test_tokens_to_bytes_is_larger_than_the_token_count() -> None:
    """The units differ by roughly 3-4x. tree-sitter's chunk_max_size is in
    bytes, and passing a token count straight through would ask for chunks a
    third of the intended size."""
    assert tokens_to_bytes(512, FileKind.CODE) > 512 * 3


def test_tokens_to_bytes_never_zero() -> None:
    assert tokens_to_bytes(0, FileKind.CODE) >= 1
