"""Extracting endpoints and the calls that reach them.

Fixtures are written to the shapes a real ASP.NET + React workspace actually
uses, because measuring one first is what showed the obvious guesses were
wrong: every route lives on a separate `[Route]` attribute rather than on
`[HttpGet]`, every Razor `@page` directive was bare, and the dominant client
function was a project wrapper rather than `fetch`.
"""

from __future__ import annotations

from workspace_indexer.graph.route_scanner import RouteScanner

CONTROLLER = """
[ApiController]
[Route("api/[controller]")]
public class RemittanceController : ControllerBase
{
    [HttpGet]
    [Route("{id}")]
    public IActionResult Get(int id) => Ok();

    [HttpPost]
    [Route("")]
    public IActionResult Create() => Ok();
}
"""


def templates(scanner: RouteScanner, text: str, language: str, path: str) -> list[str]:
    return [d.template for d in scanner.declarations(text, language, path)]


# ---- server -----------------------------------------------------------------


def test_a_class_route_is_joined_to_each_action_route() -> None:
    """The whole of ASP.NET extraction. Measured on a real workspace, *no*
    HttpGet/HttpPost carried a template of its own -- 0 of 51 -- so the class
    attribute joined to the action attribute is the entire route."""
    found = templates(RouteScanner(), CONTROLLER, "csharp", "Api/RemittanceController.cs")
    assert sorted(found) == ["api/Remittance", "api/Remittance/{id}"]


def test_an_empty_action_route_means_exactly_the_class_route() -> None:
    """`[Route("")]` was 11 of 66 in the workspace measured. Treated as a
    missing route it would drop those endpoints entirely."""
    found = templates(RouteScanner(), CONTROLLER, "csharp", "Api/RemittanceController.cs")
    assert "api/Remittance" in found


def test_the_controller_token_expands_to_the_class_name() -> None:
    """Left in, every controller's route reads identically to every other's,
    and nothing could ever match one."""
    found = templates(RouteScanner(), CONTROLLER, "csharp", "Api/RemittanceController.cs")
    assert not any("[controller]" in template for template in found)


def test_the_verb_is_recorded_where_the_code_states_one() -> None:
    declarations = RouteScanner().declarations(CONTROLLER, "csharp", "Api/X.cs")
    verbs = {d.template: d.method for d in declarations}
    assert verbs["api/Remittance/{id}"] == "GET"
    assert verbs["api/Remittance"] == "POST"


def test_a_class_route_is_not_handed_to_every_method_wholesale() -> None:
    """A class's attribute lists and its methods' attribute lists all sit
    inside the class node, so a subtree walk would give every action the same
    wrong prefix."""
    text = """
    [Route("api/first")]
    public class FirstController {
        [HttpGet]
        [Route("a")]
        public IActionResult A() => Ok();
    }
    """
    assert templates(RouteScanner(), text, "csharp", "Api/FirstController.cs") == ["api/first/a"]


def test_a_class_with_no_route_attribute_is_not_a_failure() -> None:
    text = """
    public class Helper {
        public int Add(int a, int b) => a + b;
    }
    """
    assert templates(RouteScanner(), text, "csharp", "Helper.cs") == []


# ---- razor ------------------------------------------------------------------


def test_a_bare_page_directive_takes_its_route_from_the_file_path() -> None:
    """The finding that nearly sank the Razor half: all twenty `@page`
    directives in a real workspace were bare. Reading only the directive finds
    nothing and reports an absence of routes rather than of templates."""
    found = templates(
        RouteScanner(), "@page\n<h1>Detail</h1>\n", "html", "Web/Pages/Remittance/Detail.cshtml"
    )
    assert found == ["Remittance/Detail"]


def test_an_explicit_template_wins_over_the_path() -> None:
    found = templates(
        RouteScanner(), '@page "/remit/{id}"\n', "html", "Web/Pages/Remittance/Detail.cshtml"
    )
    assert found == ["/remit/{id}"]


def test_index_drops_out_of_the_route() -> None:
    """`Pages/Foo/Index` answers `/Foo`, which is the convention."""
    found = templates(RouteScanner(), "@page\n", "html", "Web/Pages/Remittance/Index.cshtml")
    assert found == ["Remittance"]


def test_a_cshtml_without_a_page_directive_declares_nothing() -> None:
    """A layout, a partial or a view component is not addressable."""
    assert (
        templates(RouteScanner(), "<div>partial</div>\n", "html", "Web/Pages/_Layout.cshtml") == []
    )


def test_a_page_outside_the_pages_directory_is_not_guessed_at() -> None:
    assert templates(RouteScanner(), "@page\n", "html", "Web/Views/Thing.cshtml") == []


def test_the_pages_directory_is_configurable() -> None:
    scanner = RouteScanner(razor_pages_dir="Screens")
    assert templates(scanner, "@page\n", "html", "Web/Screens/Detail.cshtml") == ["Detail"]


# ---- client -----------------------------------------------------------------


def test_a_string_literal_gives_the_whole_url() -> None:
    calls = RouteScanner().calls('fetch("/api/remittance/submit");', "typescript")
    assert [(c.target, c.exact) for c in calls] == [("/api/remittance/submit", True)]


def test_a_template_literal_gives_its_static_prefix_and_says_so() -> None:
    """A prefix names an endpoint but cannot separate two routes that share
    it. Recording it as exact would overstate what the graph knows."""
    calls = RouteScanner().calls("fetch(`/api/remittance/${id}/status`);", "typescript")
    assert [(c.target, c.exact) for c in calls] == [("/api/remittance/", False)]


def test_a_url_built_entirely_from_a_variable_yields_nothing() -> None:
    """Inventing a URL for it would be worse than recording nothing -- 18 of
    32 call sites measured were this, which is why rung 3 exists."""
    assert RouteScanner().calls("fetch(url);", "typescript") == []


def test_a_leading_interpolation_yields_nothing() -> None:
    """`${BASE}/api/x` has a static part that names no endpoint on its own."""
    assert RouteScanner().calls("fetch(`${BASE}/api/x`);", "typescript") == []


def test_a_configured_object_covers_its_request_methods() -> None:
    calls = RouteScanner().calls('axios.get("/api/rates");', "typescript")
    assert [c.target for c in calls] == ["/api/rates"]


def test_an_unconfigured_wrapper_is_invisible_until_it_is_named() -> None:
    """The single most valuable finding of this rung. A real workspace called
    its own wrapper 66 times against `fetch`'s 27, so the default list saw 6
    call sites where naming the wrapper saw 71.
    """
    source = 'customFetch("/api/remittance/submit");'
    assert RouteScanner().calls(source, "typescript") == []
    named = RouteScanner(http_clients=["fetch", "customFetch"]).calls(source, "typescript")
    assert [c.target for c in named] == ["/api/remittance/submit"]


def test_a_language_with_no_scanner_returns_nothing_rather_than_guessing() -> None:
    assert RouteScanner().calls('fetch("/api/x")', "python") == []
    assert RouteScanner().declarations('[Route("x")]', "python", "a.py") == []


def test_unparseable_source_costs_the_edges_not_the_file() -> None:
    """Same rule the import scanner follows."""
    assert RouteScanner().calls("fetch( ( ( unclosed", "typescript") == []
    assert RouteScanner().declarations("public class {{{", "csharp", "a.cs") == []
