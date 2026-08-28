"""Reranking inside the aggregation.

The stages are the whole of it, so the stages are what these assert. Atlas
rejects a malformed `$rerank` outright, which is the pleasant failure; the
unpleasant one is a well-formed stage that reranks the wrong documents, the
wrong text, or the wrong number of them, and comes back looking fine.
"""

from __future__ import annotations

from typing import Any

import pytest

from workspace_indexer.storage.atlas_rerank import MAX_DOCUMENTS, PATHS, AtlasRerank
from workspace_indexer.storage.no_server_rerank import NoServerRerank
from workspace_indexer.storage.server_reranker import ServerReranker


def stage(stages: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(s[name] for s in stages if name in s)


def test_both_implementations_satisfy_the_protocol() -> None:
    assert isinstance(AtlasRerank("rerank-2.5-lite"), ServerReranker)
    assert isinstance(NoServerRerank(), ServerReranker)


def test_off_by_default_adds_only_the_score_projection() -> None:
    """The default must produce exactly the pipeline tail the store wrote
    before any of this existed, or turning reranking off changes retrieval."""
    assert NoServerRerank().stages("q", 10, "vectorSearchScore") == [
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}}
    ]


def test_off_by_default_does_not_widen_the_candidate_set() -> None:
    """Retrieving fifty documents to return ten is only worth paying for when
    something is going to reorder them."""
    assert NoServerRerank().depth(10) == 10


def test_reranking_widens_the_candidate_set() -> None:
    """`SearchService` holds a NoopReranker when the store reranks, so it does
    not widen -- nothing above the store knows a rerank is coming. Without this
    the reranker reorders the ten documents that were going to be returned
    anyway and buys nothing.
    """
    assert AtlasRerank("rerank-2.5-lite", candidates=50).depth(10) == 50


def test_a_limit_larger_than_the_candidate_set_still_wins() -> None:
    """`get_file_context` asks for more than the rerank candidate count.
    Narrowing to the candidates would silently drop chunks the caller asked
    for."""
    assert AtlasRerank("rerank-2.5-lite", candidates=50).depth(100) == 100


def test_the_stage_carries_the_query_the_model_needs() -> None:
    stages = AtlasRerank("rerank-2.5-lite", candidates=25).stages("how does auth work", 10, "score")
    rerank = stage(stages, "$rerank")

    assert rerank["model"] == "rerank-2.5-lite"
    assert rerank["query"] == {"text": "how does auth work"}
    assert rerank["numDocsToRerank"] == 25
    assert rerank["path"] == list(PATHS)


def test_missing_text_fields_are_defaulted_before_reranking() -> None:
    """`$rerank` fails the whole query if a path is missing from any document
    -- it does not skip that document. `context_header` is empty for a chunk
    that needed no header, and an older payload may not carry it at all."""
    stages = AtlasRerank("rerank-2.5-lite").stages("q", 10, "score")
    defaults = stage(stages, "$set")

    assert set(defaults) == set(PATHS)
    assert defaults["source_text"] == {"$ifNull": ["$source_text", ""]}


def test_the_defaults_are_written_before_the_rerank_stage() -> None:
    """Order is the whole point of the previous test: after the stage, the
    query has already failed."""
    names = [next(iter(s)) for s in AtlasRerank("rerank-2.5-lite").stages("q", 10, "score")]
    assert names.index("$set") < names.index("$rerank")


def test_the_page_is_cut_after_reranking_not_before() -> None:
    """Limiting first would hand the reranker ten documents and reorder those,
    which is the one thing this is not for."""
    names = [next(iter(s)) for s in AtlasRerank("rerank-2.5-lite").stages("q", 10, "score")]
    assert names.index("$rerank") < names.index("$limit")
    assert stage(AtlasRerank("rerank-2.5-lite").stages("q", 10, "score"), "$limit") == 10


def test_the_rerank_score_replaces_the_branch_score() -> None:
    """Carrying the retrieval score forward would leave results ordered by one
    number and labelled with another."""
    stages = AtlasRerank("rerank-2.5-lite").stages("q", 10, "vectorSearchScore")
    assert stage(stages, "$addFields") == {"score": {"$meta": "score"}}
    assert "vectorSearchScore" not in str(stages)


def test_an_unknown_model_is_refused_here_rather_than_by_atlas() -> None:
    """Otherwise the failure arrives mid-query, one round trip and one
    confusing error message later."""
    with pytest.raises(ValueError, match="unknown Atlas reranker model"):
        AtlasRerank("rerank-9000")


def test_the_candidate_count_is_clamped_to_what_atlas_accepts() -> None:
    assert AtlasRerank("rerank-2.5-lite", candidates=5000).depth(10) == MAX_DOCUMENTS
    assert AtlasRerank("rerank-2.5-lite", candidates=0).depth(1) == 1


def test_the_name_says_where_it_runs() -> None:
    """`search.store` logs this. "rerank-2.5-lite" alone would not distinguish
    a rerank that happened in the database from one that happened here, and
    those have very different latency."""
    assert AtlasRerank("rerank-2.5-lite").name == "database:rerank-2.5-lite"
