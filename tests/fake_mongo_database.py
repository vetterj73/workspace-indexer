"""The database half of the driver fake. See `fake_mongo_collection`."""

from __future__ import annotations

from typing import Any

from tests.fake_mongo_collection import FakeMongoCollection


class FakeMongoDatabase:
    def __init__(self, name: str) -> None:
        self.name = name
        self.collections: dict[str, FakeMongoCollection] = {}

    def __getitem__(self, name: str) -> FakeMongoCollection:
        # Created on access, like the real driver: a collection in Mongo does
        # not exist until something is written to it, and code that assumed
        # otherwise would work here and fail against Atlas.
        return self.collections.setdefault(name, FakeMongoCollection(name))

    async def list_collection_names(self) -> list[str]:
        return sorted(self.collections)

    async def create_collection(self, name: str, **_: Any) -> FakeMongoCollection:
        return self[name]

    async def drop_collection(self, name: str) -> None:
        self.collections.pop(name, None)
