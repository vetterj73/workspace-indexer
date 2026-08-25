"""Space identity.

A space maps one-to-one onto a collection, so two different things must never
produce the same slug.
"""

from __future__ import annotations

from workspace_indexer.models import EmbeddingSpace


def test_slug_is_filesystem_and_collection_safe() -> None:
    space = EmbeddingSpace(model="voyageai:voyage-code-4", dimensions=2048)
    assert space.slug() == "voyageai_voyage-code-4_2048"


def test_dimensions_change_the_slug() -> None:
    a = EmbeddingSpace(model="voyageai:voyage-code-4", dimensions=2048)
    b = EmbeddingSpace(model="voyageai:voyage-code-4", dimensions=1024)
    assert a.slug() != b.slug()


def test_model_change_the_slug() -> None:
    a = EmbeddingSpace(model="voyageai:voyage-code-4", dimensions=1024)
    b = EmbeddingSpace(model="openai:text-embedding-3-small", dimensions=1024)
    assert a.slug() != b.slug()


def test_derived_space_is_distinct_from_a_native_one() -> None:
    """"Asked the model for 1024" and "truncated 2048 down to 1024" are
    different vector spaces. Sharing a slug means sharing a collection, and a
    partial run then leaves the two silently mixed with nothing able to detect
    it."""
    native = EmbeddingSpace(model="voyageai:voyage-code-4", dimensions=1024)
    derived = EmbeddingSpace(
        model="voyageai:voyage-code-4", dimensions=1024, derived_from=2048
    )
    assert native.slug() != derived.slug()
    assert not native.is_derived
    assert derived.is_derived


def test_derived_slug_names_its_source() -> None:
    """So `status` can say what a reprojection came from."""
    derived = EmbeddingSpace(
        model="voyageai:voyage-code-4", dimensions=1024, derived_from=2048
    )
    assert "2048" in derived.slug()
    assert "1024" in derived.slug()


def test_reprojections_from_different_widths_are_distinct() -> None:
    from_2048 = EmbeddingSpace(model="m", dimensions=512, derived_from=2048)
    from_1024 = EmbeddingSpace(model="m", dimensions=512, derived_from=1024)
    assert from_2048.slug() != from_1024.slug()
