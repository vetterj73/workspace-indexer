"""The client half of the driver fake. See `fake_mongo_collection`."""

from __future__ import annotations

from tests.fake_mongo_database import FakeMongoDatabase


class FakeMongoClient:
    def __init__(self) -> None:
        self.databases: dict[str, FakeMongoDatabase] = {}
        self.closed = False

    def __getitem__(self, name: str) -> FakeMongoDatabase:
        return self.databases.setdefault(name, FakeMongoDatabase(name))

    async def close(self) -> None:
        self.closed = True
