"""Configuration for the dependency and route graph."""

from __future__ import annotations

from pydantic import Field

from workspace_indexer.config.strict import Strict

# Functions that make an HTTP request, as shipped. Deliberately short, because
# the useful answer is project-specific and pretending otherwise is what makes
# a graph look empty.
#
# Measured on a real React workspace: `fetch` appeared 27 times and a project
# wrapper called `customFetch` 66 times, with `fetchJson` behind it. A scanner
# that knew only `fetch` therefore found under a tenth of the call sites and
# reported that as coverage. Whatever your codebase wraps it in has to be
# nameable, or the numbers describe our guess rather than your code.
DEFAULT_HTTP_CLIENTS = ("fetch", "axios", "ky", "superagent")


class GraphSection(Strict):
    # Bare function names (`fetch`, `customFetch`) and object names whose
    # method calls count (`axios` covers `axios.get`, `axios.post`).
    http_clients: list[str] = Field(default_factory=lambda: list(DEFAULT_HTTP_CLIENTS))
    # Directory whose contents map to routes by file path. Razor Pages declares
    # `@page` with no template and derives the URL from location -- measured on
    # a real workspace, all 20 directives were bare -- so the convention is the
    # only thing there is to read.
    razor_pages_dir: str = "Pages"
    # Path segments the client prepends that the server never declares --
    # added by a reverse proxy, a dev-server rewrite, or UsePathBase.
    #
    # Tried only as a fallback, after the URL as written fails, so a workspace
    # whose routes genuinely start with `api` is unaffected. Measured on a real
    # React + minimal-API codebase: every client call began `/api/` and no
    # endpoint did, and allowing the fallback took resolution from 4% to 23%.
    client_base_paths: list[str] = Field(default_factory=lambda: ["api"])
