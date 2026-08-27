"""Binding the query service to the MCP protocol.

Deliberately thin. Everything here is a decorator, a docstring and a type hint;
the behaviour lives in QueryService, which is testable without a client
session. Named `server_factory` rather than `server` so that a traceback never
leaves you wondering whether `mcp.server` is ours or the SDK's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from workspace_indexer.app_context import AppContext
from workspace_indexer.mcp.document_type_resolver import DocumentTypeResolver
from workspace_indexer.mcp.empty_index_error import EmptyIndexError
from workspace_indexer.mcp.query_service import QueryService
from workspace_indexer.mcp.search_response import SearchResponse
from workspace_indexer.mcp.taxonomy import Taxonomy
from workspace_indexer.mcp.taxonomy_service import TaxonomyService
from workspace_indexer.mcp.tool_call_recorder import ToolCallRecorder
from workspace_indexer.mcp.unknown_document_type_error import UnknownDocumentTypeError
from workspace_indexer.models import DocumentType

TAXONOMY_URI = "workspace-indexer://taxonomy"

# Spliced into the tool descriptions. The types have to be in context *before*
# the agent picks one: a round trip to discover the vocabulary is a round trip
# it will usually skip, and then it guesses.
_TYPE_LIST = ", ".join(t.value for t in DocumentType)

_INSTRUCTIONS = f"""\
A semantic + keyword index over this workspace. Use it instead of grepping when
you know *what* you want but not *where* it is.

- search_code -- implementation, with tests and generated files excluded.
- find_guidance -- specifications and design documents only. Reach for this
  before writing new code, especially when there is nothing yet to imitate.
- get_file_context -- every indexed chunk of one file, in order.
- list_document_types -- what kinds of document this workspace actually holds,
  with counts. A count of zero is a real answer: it means look at the code.

Document types: {_TYPE_LIST}.
"""


def build_query_service(ctx: AppContext) -> QueryService:
    return QueryService(
        search=ctx.search_service(),
        taxonomy=TaxonomyService(ctx.store, ctx.space),
        check_staleness=ctx.config.search.check_staleness,
        # The manifest is the harvesting sink: finding the calls that returned
        # nothing is a query rather than a log scrape, and those calls are eval
        # cases waiting to be written.
        recorder=ToolCallRecorder(ctx.manifest),
    )


def build_mcp_server(queries: QueryService) -> MCPServer:
    """Wrap a QueryService in the protocol.

    Takes the service rather than the AppContext so a test can build the real
    server -- real tool schemas, real dispatch -- over a store it seeded
    itself, without constructing every layer of the application.
    """
    resolver = DocumentTypeResolver()
    server = MCPServer(name="workspace-indexer", instructions=_INSTRUCTIONS)

    @server.tool()
    async def search_code(
        query: Annotated[str, Field(description="What you are looking for, in plain language.")],
        limit: Annotated[int, Field(description="Maximum hits.", ge=1, le=50)] = 8,
        repo: Annotated[str | None, Field(description="Restrict to one repository.")] = None,
        language: Annotated[
            str | None, Field(description="Restrict to one language, e.g. python.")
        ] = None,
        path_prefix: Annotated[
            str | None, Field(description="Restrict to a directory, e.g. src/auth.")
        ] = None,
        include_tests: Annotated[
            bool, Field(description="Include tests and generated files.")
        ] = False,
    ) -> SearchResponse:
        """Search implementation code and documentation by meaning.

        Tests and generated files are excluded unless include_tests is set,
        because a test naming a symbol twenty times otherwise outranks the one
        file that defines it. Every result is anchored as path:start-end.
        """
        return await queries.search_code(
            query,
            limit=limit,
            repo=repo,
            language=language,
            path_prefix=path_prefix,
            include_tests=include_tests,
        )

    @server.tool()
    async def find_guidance(
        query: Annotated[str, Field(description="The decision or topic you need rules for.")],
        limit: Annotated[int, Field(description="Maximum hits.", ge=1, le=50)] = 8,
        repo: Annotated[str | None, Field(description="Restrict to one repository.")] = None,
        doc_type: Annotated[
            str | None,
            Field(description=f"Narrow to one type. One of: {_TYPE_LIST}."),
        ] = None,
    ) -> SearchResponse:
        """Find the specs, conventions and design documents governing a topic.

        Searches normative and design documents only, so a changelog describing
        how something used to work cannot outrank the standard saying how it
        must be built. Use this before writing new code -- particularly when
        there is no existing implementation to copy.

        An unrecognised doc_type is an error naming the valid ones, never an
        empty result.
        """
        try:
            selected = resolver.resolve(doc_type) if doc_type else None
        except UnknownDocumentTypeError as exc:
            # Converted at the boundary, deliberately. The SDK strips the text
            # of an arbitrary exception and sends the model a bare "Error
            # executing tool find_guidance" -- which is exactly the useless,
            # unactionable failure this whole design is built to avoid. Only a
            # ToolError carries its message through to the agent.
            raise ToolError(str(exc)) from exc
        return await queries.find_guidance(query, limit=limit, repo=repo, doc_type=selected)

    @server.tool()
    async def get_file_context(
        rel_path: Annotated[
            str, Field(description="Path from a search result, or a trailing portion of one.")
        ],
        limit: Annotated[int, Field(description="Maximum chunks.", ge=1, le=100)] = 20,
    ) -> SearchResponse:
        """Return every indexed chunk of one file, in file order.

        Use this to expand around a hit: the search returns the matching
        function, this returns the rest of the file as it was indexed.
        """
        return await queries.get_file_context(rel_path, limit=limit)

    @server.tool()
    async def list_document_types() -> Taxonomy:
        """List the document types in this workspace, with counts and examples.

        Counts describe this index, not the code's vocabulary. A type reported
        with count 0 genuinely has no documents of that kind here -- if
        `normative` is 0, this workspace has no written standards and you
        should read the implementation instead of hunting for specs.
        """
        return await queries.taxonomy()

    @server.resource(
        TAXONOMY_URI,
        name="Document taxonomy",
        description="Document types in this workspace, with counts and example paths.",
        mime_type="application/json",
    )
    async def taxonomy_resource() -> str:
        taxonomy = await queries.taxonomy()
        return json.dumps(taxonomy.model_dump(), indent=2)

    # Registered by decoration; naming them here keeps the linter honest about
    # the fact that these are the server's public surface.
    _ = (search_code, find_guidance, get_file_context, list_document_types, taxonomy_resource)
    return server


async def preflight(ctx: AppContext) -> None:
    """Fail loudly at startup rather than serving nothing.

    Deliberately a hard failure, not a warning. A warning goes to stderr, which
    an MCP client files under logs nobody reads, and the session then proceeds
    with a server that answers every question with "nothing found" -- worse
    than no server at all, because the agent believes the answer.
    """
    if await ctx.store.count(ctx.space):
        return
    raise EmptyIndexError(
        space=ctx.space.slug(),
        mode=(
            f"qdrant server at {ctx.settings.qdrant_url}"
            if ctx.settings.qdrant_mode == "server"
            else f"embedded qdrant at {Path(ctx.settings.qdrant_path).resolve()}"
        ),
        detail=f"collection {ctx.store.collection_name(ctx.space)} holds no points.",
    )
