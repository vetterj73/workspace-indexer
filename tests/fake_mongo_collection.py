"""A collection that records what was asked of it.

This project prefers real things to mocks, and every other store test runs
against a real engine. There is no offline equivalent here: Atlas Search is a
managed service, `mongomock` does not implement `$vectorSearch` or `$search`,
and a plain `mongod` container would accept the pipelines and answer them
wrongly, which is worse than not running them.

So the seam is the driver, and what these fakes assert is the *document we
send*. That is not a stand-in for the real thing -- `test_mongo_integration.py`
is, and it runs against Atlas when a connection string is configured. It is
coverage of the half that is wrong most often: two bugs in this store's first
draft were a wrong `$meta` name and a stage `$rankFusion` rejects, and both are
visible in the pipeline and invisible in anything a mocked *result* could show.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, cast


class FakeMongoCollection:
    def __init__(self, name: str, documents: list[dict[str, Any]] | None = None) -> None:
        self.name = name
        self.documents: list[dict[str, Any]] = list(documents or [])
        # Every aggregation pipeline sent, in order. The assertions live here.
        self.pipelines: list[list[dict[str, Any]]] = []
        self.created_indexes: list[Any] = []
        self.created_search_indexes: list[Any] = []
        self.search_indexes: list[dict[str, Any]] = []
        self.deleted: list[dict[str, Any]] = []
        self.written: list[dict[str, Any]] = []
        # Name a stage here to make only pipelines containing it fail, which
        # is what a server missing $rankFusion actually does. Failing every
        # aggregation instead would make the fallback path untestable, because
        # the fallback is itself an aggregation.
        self.unsupported_stage: str | None = None
        self.search_index_error: Exception | None = None
        self.aggregate_results: list[list[dict[str, Any]]] = []

    async def create_index(self, keys: Any, **_: Any) -> str:
        self.created_indexes.append(keys)
        return "ok"

    async def list_search_indexes(self) -> AsyncIterator[dict[str, Any]]:
        if self.search_index_error is not None:
            raise self.search_index_error
        return _aiter(self.search_indexes)

    async def create_search_indexes(self, models: Sequence[Any]) -> list[str]:
        self.created_search_indexes.extend(models)
        self.search_indexes.extend({"name": m.document["name"], "queryable": True} for m in models)
        return [m.document["name"] for m in models]

    async def bulk_write(self, operations: Sequence[Any], **_: Any) -> Any:
        # Only the count. What the documents look like is asserted against
        # `build_document`, which is a pure function -- reading them back out
        # of a driver operation would mean touching pymongo's private
        # attributes to test our own code.
        self.written.extend([cast(dict[str, Any], {}) for _ in operations])
        return _BulkResult(len(operations))

    async def delete_many(self, condition: dict[str, Any]) -> Any:
        self.deleted.append(condition)
        return _DeleteResult(0)

    async def count_documents(self, condition: dict[str, Any]) -> int:
        self.pipelines.append([{"$match": condition}])
        return len(self.documents)

    async def aggregate(self, pipeline: list[dict[str, Any]], **_: Any) -> AsyncIterator[Any]:
        self.pipelines.append(pipeline)
        if self.unsupported_stage and any(self.unsupported_stage in s for s in pipeline):
            from pymongo.errors import OperationFailure

            raise OperationFailure(f"Unrecognized pipeline stage name: '{self.unsupported_stage}'")
        if self.aggregate_results:
            return _aiter(self.aggregate_results.pop(0))
        return _aiter(self.documents)

    def find(self, condition: dict[str, Any] | None = None, **_: Any) -> _FakeCursor:
        return _FakeCursor(self.documents, condition or {})


class _BulkResult:
    def __init__(self, count: int) -> None:
        self.upserted_count = count
        self.modified_count = 0


class _DeleteResult:
    def __init__(self, count: int) -> None:
        self.deleted_count = count


class _FakeCursor:
    def __init__(self, documents: list[dict[str, Any]], condition: dict[str, Any]) -> None:
        self._documents = [d for d in documents if _matches(d, condition)]

    def limit(self, count: int) -> _FakeCursor:
        self._documents = self._documents[:count]
        return self

    def batch_size(self, _: int) -> _FakeCursor:
        return self

    async def to_list(self) -> list[dict[str, Any]]:
        return list(self._documents)

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return _aiter(self._documents)


def _matches(document: dict[str, Any], condition: dict[str, Any]) -> bool:
    """Equality and `$regex` only -- the two forms `chunks_for_path` uses."""
    import re

    for key, expected in condition.items():
        actual = document.get(key)
        if isinstance(expected, dict) and "$regex" in expected:
            pattern = cast(dict[str, Any], expected)["$regex"]
            if not isinstance(actual, str) or not re.search(str(pattern), actual):
                return False
        elif actual != expected:
            return False
    return True


async def _aiter(items: Sequence[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item
