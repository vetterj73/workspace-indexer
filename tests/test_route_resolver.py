"""Matching a called URL to the file that declares it.

The failure to guard hardest against is not a missed edge -- it is a confident
wrong one. An `impact_of` that names the wrong controller sends someone to
change a file nothing was calling, and unlike an empty answer it carries no
hint that it might be wrong.
"""

from __future__ import annotations

from workspace_indexer.graph.route_resolver import RouteResolver, segments
from workspace_indexer.graph.route_target import RouteTarget

REMITTANCE = "Api/RemittanceController.cs"
RATES = "Api/RatesController.cs"


def target(rel_path: str, template: str, root: str = "api") -> RouteTarget:
    return RouteTarget(root_label=root, rel_path=rel_path, template=template)


RESOLVER = RouteResolver(
    [
        target(REMITTANCE, "api/Remittance"),
        target(REMITTANCE, "api/Remittance/{id}"),
        target(REMITTANCE, "api/Remittance/archive/{id:int}"),
        target(RATES, "api/Rates"),
        target("Web/Pages/Remittance/Detail.cshtml", "Remittance/Detail", root="web"),
    ]
)


# ---- exact URLs -------------------------------------------------------------


def test_an_exact_url_finds_its_endpoint() -> None:
    hit = RESOLVER.resolve("/api/Remittance", exact=True)
    assert hit is not None and hit.rel_path == REMITTANCE


def test_a_route_parameter_matches_any_one_segment() -> None:
    hit = RESOLVER.resolve("/api/Remittance/42", exact=True)
    assert hit is not None and hit.template == "api/Remittance/{id}"


def test_matching_ignores_case() -> None:
    """A client writes `/api/remittance`; the attribute says
    `[Route("api/[controller]")]` on `RemittanceController`. Routing is
    case-insensitive in every framework this targets, and insisting on a match
    would drop most real edges."""
    hit = RESOLVER.resolve("/API/remittance", exact=True)
    assert hit is not None and hit.rel_path == REMITTANCE


def test_a_query_string_is_not_part_of_the_path() -> None:
    hit = RESOLVER.resolve("/api/Rates?live=1&at=now", exact=True)
    assert hit is not None and hit.rel_path == RATES


def test_an_absolute_url_resolves_by_its_path() -> None:
    hit = RESOLVER.resolve("https://api.example.test/api/Rates", exact=True)
    assert hit is not None and hit.rel_path == RATES


def test_a_url_with_no_matching_endpoint_stays_unresolved() -> None:
    assert RESOLVER.resolve("/api/Unknown", exact=True) is None


def test_a_longer_url_does_not_match_a_shorter_route() -> None:
    """`api/Rates` has no parameter to absorb the extra segment."""
    assert RESOLVER.resolve("/api/Rates/extra", exact=True) is None


def test_a_razor_page_is_a_target_too() -> None:
    hit = RESOLVER.resolve("/Remittance/Detail", exact=True)
    assert hit is not None and hit.root_label == "web"


# ---- prefixes from template literals ----------------------------------------


def test_a_prefix_resolves_when_one_file_declares_everything_under_it() -> None:
    """What makes the prefix-only half usable, and it is over half of the call
    sites measured. `/api/Remittance/${id}` cannot say which action it calls,
    but every candidate is in one file -- and "which file" is the question
    impact_of asks."""
    hit = RESOLVER.resolve("/api/Remittance/", exact=False)
    assert hit is not None and hit.rel_path == REMITTANCE


def test_a_prefix_spanning_two_files_stays_unresolved() -> None:
    """`/api/` reaches both controllers. Picking one would be a guess wearing
    the clothes of an answer."""
    assert RESOLVER.resolve("/api/", exact=False) is None


def test_a_prefix_must_be_shorter_than_the_route_it_matches() -> None:
    """The prefix came from a template literal, so something was interpolated
    after it -- there has to be a further segment for it to go in."""
    assert RESOLVER.resolve("/api/Rates/", exact=False) is None


def test_an_empty_url_resolves_to_nothing() -> None:
    assert RESOLVER.resolve("", exact=True) is None
    assert RESOLVER.resolve("/", exact=False) is None


# ---- normalisation ----------------------------------------------------------


def test_segments_strips_scheme_host_query_and_slashes() -> None:
    assert segments("https://h.test/api/Remittance/?q=1#x") == ["api", "remittance"]
    assert segments("//api//Remittance//") == ["api", "remittance"]


def test_a_catch_all_absorbs_the_rest_of_the_path() -> None:
    resolver = RouteResolver([target("Api/FilesController.cs", "api/files/{*path}")])
    hit = resolver.resolve("/api/files/a/b/c.txt", exact=True)
    assert hit is not None and hit.rel_path == "Api/FilesController.cs"


def test_two_repositories_declaring_the_same_path_are_ambiguous() -> None:
    """Why the target carries its root. Without it the two are one row and the
    resolver would silently pick whichever came first."""
    resolver = RouteResolver(
        [
            target("Api/HomeController.cs", "api/home", root="one"),
            target("Api/HomeController.cs", "api/home", root="two"),
        ]
    )
    assert resolver.resolve("/api/home", exact=True) is None
