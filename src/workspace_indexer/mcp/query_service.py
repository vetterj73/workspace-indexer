"""What the MCP tools actually do, with no MCP in sight.

The protocol layer above is decoration: decorators, docstrings and JSON
schemas. Everything worth testing -- which document types each tool selects,
how an unknown type is reported, what happens when nothing matches -- lives
here, so the tests need a store and a search service rather than a client
session and a subprocess.
"""

from __future__ import annotations

from workspace_indexer.mcp.result_budget import ResultBudget
from workspace_indexer.mcp.search_response import SearchResponse
from workspace_indexer.mcp.taxonomy import Taxonomy
from workspace_indexer.mcp.taxonomy_service import TaxonomyService
from workspace_indexer.models import DocumentType, SearchFilters, SearchHit
from workspace_indexer.obs.logging import get_logger
from workspace_indexer.search.search_request import SearchRequest
from workspace_indexer.search.search_service import SearchService

log = get_logger("workspace_indexer.mcp")

# What `search_code` drops unless asked otherwise. Tests mention a symbol far
# more often than the one file defining it, so they crowd out the definition on
# exactly the queries where the definition is what you wanted.
CODE_EXCLUDES = [DocumentType.TEST, DocumentType.GENERATED]

# What `find_guidance` keeps. Narrow enough to exclude a changelog describing
# how something once worked, wide enough to keep CONTRIBUTING.md.
#
# `guide` is in this list because of a measurement, not a hunch. Normative and
# design alone scored recall 0.812 / MRR 0.792 over the eight guidance cases --
# no better than plain search, because filtering out `guide` lost
# CONTRIBUTING.md entirely on "how do I set up a development environment".
# Adding it gives 0.938 / 0.900. The eval case that caught this was written
# before the tool existed, specifically to catch a type filter over-filtering.
GUIDANCE_TYPES = [DocumentType.NORMATIVE, DocumentType.DESIGN, DocumentType.GUIDE]


class QueryService:
    def __init__(
        self,
        *,
        search: SearchService,
        taxonomy: TaxonomyService,
        max_response_tokens: int = 6000,
        check_staleness: bool = True,
    ) -> None:
        self._search = search
        self._taxonomy = taxonomy
        self._budget = ResultBudget(max_response_tokens)
        self._check_staleness = check_staleness

    async def search_code(
        self,
        query: str,
        *,
        limit: int = 8,
        repo: str | None = None,
        language: str | None = None,
        path_prefix: str | None = None,
        include_tests: bool = False,
    ) -> SearchResponse:
        filters = SearchFilters(
            repo_name=repo,
            language=language,
            path_prefix=path_prefix,
            exclude_doc_types=[] if include_tests else CODE_EXCLUDES,
        )
        hits = await self._run(query, filters, limit)
        return self._respond(
            query,
            filters,
            hits,
            empty_note=(
                "Nothing matched. Tests and generated files were excluded; "
                "retry with include_tests=true if the answer may live in a test."
                if not include_tests
                else "Nothing matched. Try a broader query, or drop the filters."
            ),
        )

    async def find_guidance(
        self,
        query: str,
        *,
        limit: int = 8,
        repo: str | None = None,
        doc_type: DocumentType | None = None,
    ) -> SearchResponse:
        """Specs and design documents only.

        The motivating case is an agent starting greenfield work: there is no
        existing code to imitate, so what it needs is the rules, and a plain
        semantic search hands it the nearest *paragraph* instead.
        """
        filters = SearchFilters(
            repo_name=repo,
            doc_types=[doc_type] if doc_type else GUIDANCE_TYPES,
        )
        hits = await self._run(query, filters, limit)
        wanted = doc_type.value if doc_type else " or ".join(t.value for t in GUIDANCE_TYPES)
        return self._respond(
            query,
            filters,
            hits,
            empty_note=(
                f"No {wanted} documents matched. This workspace may have no written "
                "guidance on the topic -- call list_document_types to see whether it "
                "has any at all, and fall back to reading the implementation if not."
            ),
        )

    async def get_file_context(self, rel_path: str, *, limit: int = 20) -> SearchResponse:
        """Every indexed chunk of one file, in file order.

        Not a search: an agent that has a hit and wants the surrounding code
        should not have to guess a query that retrieves its neighbours.
        """
        hits = await self._search.chunks_for_path(
            rel_path, limit=limit, check_staleness=self._check_staleness
        )
        hits.sort(key=lambda h: (h.rel_path, h.start_line))
        results, dropped = self._budget.pack(hits)
        return SearchResponse(
            query=rel_path,
            applied_filters={"rel_path": rel_path},
            results=results,
            total_matches=len(hits),
            returned=len(results),
            dropped_for_budget=dropped,
            note=(
                f"No indexed chunks for {rel_path!r}. The path is matched as a suffix, "
                "so try a longer or shorter portion of it -- or the file may be "
                "excluded from the index."
                if not hits
                else None
            ),
        )

    async def taxonomy(self) -> Taxonomy:
        return await self._taxonomy.build()

    async def _run(self, query: str, filters: SearchFilters, limit: int) -> list[SearchHit]:
        return await self._search.search(
            SearchRequest(
                query=query,
                filters=filters,
                limit=limit,
                check_staleness=self._check_staleness,
            )
        )

    def _respond(
        self,
        query: str,
        filters: SearchFilters,
        hits: list[SearchHit],
        *,
        empty_note: str,
    ) -> SearchResponse:
        results, dropped = self._budget.pack(hits)
        note: str | None = None
        if not hits:
            note = empty_note
        elif dropped:
            note = (
                f"{dropped} further match(es) were dropped to stay inside the "
                "response token budget; narrow the query or raise the limit."
            )
        return SearchResponse(
            query=query,
            applied_filters=_describe(filters),
            results=results,
            total_matches=len(hits),
            returned=len(results),
            dropped_for_budget=dropped,
            note=note,
        )


def _describe(filters: SearchFilters) -> dict[str, str]:
    """Echo the filters back in the caller's terms.

    An empty result set means two very different things depending on whether a
    filter was applied, and the agent cannot see the filters we defaulted it
    into unless we say so.
    """
    described: dict[str, str] = {}
    for key, value in filters.model_dump(exclude_none=True).items():
        if isinstance(value, list):
            if value:
                described[key] = ", ".join(str(v) for v in value)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        elif value:
            described[key] = str(value)
    return described
