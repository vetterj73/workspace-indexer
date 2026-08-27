"""The protocol layer: what a client actually sees.

Thin by design, but three things can only go wrong here -- a tool that is not
registered, a description that does not tell the agent the vocabulary, and an
unknown document type that comes back as an empty list instead of an error.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError
from mcp.types import CallToolResult, InputRequiredResult, TextContent
from qdrant_client import AsyncQdrantClient

from tests.conftest import make_source
from tests.fake_embedding_backend import FakeEmbeddingBackend
from tests.fake_sparse_backend import FakeSparseBackend
from workspace_indexer.app_context import AppContext
from workspace_indexer.chunking.chunk_factory import build_chunk
from workspace_indexer.config import RerankConfig, SearchSection, Settings
from workspace_indexer.embedding.embedding_service import EmbeddingService
from workspace_indexer.mcp import (
    TAXONOMY_URI,
    EmptyIndexError,
    ImpactService,
    QueryService,
    TaxonomyService,
    build_mcp_server,
)
from workspace_indexer.mcp.server_factory import preflight
from workspace_indexer.models import Chunk, DocumentType, EmbeddingSpace, FileKind
from workspace_indexer.rerank.noop_reranker import NoopReranker
from workspace_indexer.search.search_service import SearchService
from workspace_indexer.state import Manifest
from workspace_indexer.storage.qdrant_store import QdrantStore

SPACE = EmbeddingSpace(model="fake:model", dimensions=4)

DOCS: list[tuple[str, DocumentType]] = [
    ("repo/CONVENTIONS.md", DocumentType.NORMATIVE),
    ("repo/src/store.py", DocumentType.IMPLEMENTATION),
    ("repo/tests/test_store.py", DocumentType.TEST),
]


@pytest.fixture
async def populated(tmp_path: Path) -> AsyncIterator[QdrantStore]:
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantStore(client, workspace="labbox", payload_indexes=False)
    sparse = FakeSparseBackend()
    chunks: list[Chunk] = []
    for index, (rel_path, doc_type) in enumerate(DOCS):
        source = make_source(
            "store the thing",
            kind=FileKind.CODE,
            language="python",
            rel_path=rel_path,
            unit="repo",
        )
        chunk = build_chunk(
            source,
            "labbox",
            source_text="store the thing",
            start_line=1,
            end_line=1,
            chunker="text",
            version=1,
            chunk_index=index,
        )
        chunk.meta.doc_type = doc_type
        chunks.append(chunk)
    await store.upsert(
        SPACE,
        chunks,
        [[1.0, 0, 0, 0]] * len(chunks),
        sparse.encode_documents(["store the thing"] * len(chunks)),
    )
    yield store
    await client.close()


@pytest.fixture
def queries(populated: QdrantStore) -> QueryService:
    search = SearchService(
        store=populated,
        embeddings=EmbeddingService(FakeEmbeddingBackend(dimensions=4)),
        sparse=FakeSparseBackend(),
        reranker=NoopReranker(),
        config=SearchSection(rerank=RerankConfig(enabled=False, model="fake:m")),
        space=SPACE,
    )
    return QueryService(search=search, taxonomy=TaxonomyService(populated, SPACE))


@pytest.fixture
def server(queries: QueryService, tmp_path: Path) -> MCPServer:
    """The real server over a real (empty) manifest.

    A real SQLite file rather than a stub: the graph tools are SQL, and a mock
    would confirm our assumptions about the query instead of the query.
    """
    return build_mcp_server(queries, ImpactService(Manifest(tmp_path / "manifest.sqlite3")))


async def test_every_tool_is_registered(server: MCPServer) -> None:
    tools = await server.list_tools()
    assert {t.name for t in tools} == {
        "search_code",
        "find_guidance",
        "get_file_context",
        "list_document_types",
        "impact_of",
    }


async def test_tool_descriptions_carry_the_vocabulary(server: MCPServer) -> None:
    """The type list must be in context *before* the agent picks one. A round
    trip to discover it is a round trip it will skip, and then it guesses."""
    tools = {t.name: t for t in await server.list_tools()}
    guidance = tools["find_guidance"]
    schema = json.dumps(guidance.input_schema)
    for doc_type in DocumentType:
        assert doc_type.value in schema


async def test_search_code_dispatches_and_excludes_tests(server: MCPServer) -> None:
    result = await server.call_tool("search_code", {"query": "store", "limit": 10})
    body = json.dumps(_payload(result))
    assert "repo/src/store.py" in body
    assert "repo/tests/test_store.py" not in body


async def test_unknown_doc_type_is_an_error_not_an_empty_list(
    server: MCPServer,
) -> None:
    """The acceptance criterion that stops the silent-empty-result failure mode
    from shipping.

    Asserting the *message*, not just the failure. The SDK strips the text of
    an unrecognised exception and sends the model a bare "Error executing tool
    find_guidance"; only a ToolError carries its own words through. An error
    the agent cannot act on is barely better than the empty list.
    """
    with pytest.raises(ToolError) as caught:
        await server.call_tool("find_guidance", {"query": "structure", "doc_type": "blueprint"})

    message = str(caught.value)
    assert "blueprint" in message
    assert "normative" in message
    assert "spec" in message


async def test_a_plain_exception_reaches_the_model_with_its_text_removed() -> None:
    """Why the conversion above is deliberate rather than incidental.

    Pins the SDK behaviour the design works around, on a throwaway server so
    the assertion is about the SDK and not about which of our tools happens to
    fail. If a future version stops stripping the message, this test fails and
    the boundary conversion can be simplified away.
    """
    probe = MCPServer(name="probe")

    @probe.tool()
    async def explode() -> str:
        raise ValueError("a very specific and useful explanation")

    _ = explode
    with pytest.raises(UnexpectedToolError) as caught:
        await probe.call_tool("explode", {})
    assert "very specific" not in str(caught.value)


async def test_an_alias_is_not_an_error(server: MCPServer) -> None:
    result = await server.call_tool("find_guidance", {"query": "structure", "doc_type": "spec"})
    assert getattr(result, "isError", False) is False


async def test_taxonomy_resource_is_served_as_json(server: MCPServer) -> None:
    resources = await server.list_resources()
    assert str(resources[0].uri) == TAXONOMY_URI

    payload = json.loads(await _resource(server, TAXONOMY_URI))
    assert payload["taxonomy_version"] >= 1
    assert {e["name"] for e in payload["types"]} == {t.value for t in DocumentType}


async def test_resource_and_tool_agree(server: MCPServer) -> None:
    """Both surfaces exist because clients differ in which one a model reliably
    sees. They must never disagree about what is in the workspace."""
    from_resource = json.loads(await _resource(server, TAXONOMY_URI))

    result = await server.call_tool("list_document_types", {})
    from_tool = _payload(result)

    assert from_resource["types"] == from_tool["types"]


async def _resource(server: MCPServer, uri: str) -> str:
    """The body of a resource read, as text.

    The SDK returns an iterable of content parts whose payload is `str | bytes`
    depending on the mime type; ours is JSON, so anything else is a bug in the
    server rather than something to accommodate here.
    """
    result = await server.read_resource(uri)
    assert not isinstance(result, InputRequiredResult)
    parts = [part.content for part in result]
    assert all(isinstance(part, str) for part in parts)
    return "".join(part for part in parts if isinstance(part, str))


def _payload(result: CallToolResult | InputRequiredResult) -> dict[str, Any]:
    """Structured output where the SDK provides it, decoded text otherwise."""
    assert isinstance(result, CallToolResult)
    if result.structured_content is not None:
        return dict(result.structured_content)
    text = "".join(b.text for b in result.content if isinstance(b, TextContent))
    loaded: object = json.loads(text)
    assert isinstance(loaded, dict)
    return dict(loaded)  # pyright: ignore[reportUnknownArgumentType]


async def test_preflight_refuses_to_serve_an_empty_index(tmp_path: Path) -> None:
    """The deployment failure this catches is silent and total.

    An MCP client launches the server from its own working directory, so a
    relative QDRANT_PATH resolves somewhere new and an unreachable `.env`
    drops the process into embedded mode against a directory that does not
    exist. It starts up perfectly and answers every question with "nothing
    found", which the agent believes.
    """
    client = AsyncQdrantClient(path=str(tmp_path / "empty"))
    store = QdrantStore(client, workspace="labbox", payload_indexes=False)
    ctx = _stub_context(store, tmp_path)

    with pytest.raises(EmptyIndexError) as caught:
        await preflight(ctx)

    message = str(caught.value)
    # Actionable on its own: which store it looked at, and what to check.
    assert "labbox__fake_model_4" in message
    assert "status" in message
    assert "--config" in message
    await client.close()


async def test_preflight_passes_over_a_populated_index(
    populated: QdrantStore, tmp_path: Path
) -> None:
    await preflight(_stub_context(populated, tmp_path))


def _stub_context(store: QdrantStore, tmp_path: Path) -> AppContext:
    """The three fields preflight reads, with the rest left unbuilt.

    Constructing a real AppContext would load config, configure logging and
    build an embedding backend -- none of which preflight touches.
    """
    return cast(
        "AppContext",
        SimpleNamespace(
            store=store,
            space=SPACE,
            settings=Settings(state_db=tmp_path / "manifest.sqlite3"),
        ),
    )
