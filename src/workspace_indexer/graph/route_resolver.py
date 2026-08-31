"""Matching a client call to the endpoint it reaches.

Rung 2 of #53. Rung 1 established there is something to match: 53 declarations
and 71 call sites in one real workspace, 34 of them carrying a whole URL and 37
only a static prefix.

**This resolver crosses repositories, and the import resolver deliberately does
not.** That inversion is the whole point. An import names a symbol, and a
symbol belongs to the project that compiles it, so searching outside that
project would invent edges. A URL names a running service, and the service is
in a different repository more often than not -- confining the search would
find nothing worth finding.

Resolution is to a *file*, not to an endpoint, and that choice makes the
prefix-only half usable. `fetch(\\`/api/remit/${id}\\`)` cannot say which of a
controller's actions it calls, but every one of them is in the same file, so
the question `impact_of` actually asks -- what breaks if I change this file --
has an unambiguous answer even when the endpoint does not.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from workspace_indexer.graph.route_target import RouteTarget
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.graph.routes")

# A route parameter: {id}, {id:int}, {*catchAll}. Matched by any one segment,
# except the catch-all which swallows the rest.
_PARAMETER = re.compile(r"^\{.*\}$")
_CATCH_ALL = re.compile(r"^\{\*.*\}$")

# Everything before the path in an absolute URL, and everything after it.
_SCHEME_HOST = re.compile(r"^[a-z][a-z0-9+.-]*://[^/]*", re.IGNORECASE)
_QUERY_OR_FRAGMENT = re.compile(r"[?#].*$")


def segments(url: str) -> list[str]:
    """A URL or template reduced to comparable path segments.

    Lowercased because routing is case-insensitive in every framework this
    targets, and a client that wrote `/api/remittance` against
    `[Route("api/[controller]")]` on `RemittanceController` differs only in
    case. Insisting on a match there would drop most real edges.
    """
    path = _QUERY_OR_FRAGMENT.sub("", _SCHEME_HOST.sub("", url.strip()))
    return [part for part in path.lower().split("/") if part]


class RouteResolver:
    def __init__(
        self, declarations: list[RouteTarget], client_base_paths: Sequence[str] = ()
    ) -> None:
        self._targets = [(segments(target.template), target) for target in declarations]
        self._base_paths = [segments(path) for path in client_base_paths if segments(path)]

    def resolve(self, target_url: str, *, exact: bool) -> RouteTarget | None:
        """The file this call reaches, or None when that cannot be decided.

        None covers three genuinely different situations -- nothing matched,
        several files matched, or the URL named something outside the
        workspace -- and they are all unresolvable in the same way. What they
        must never become is a guess: an `impact_of` that names the wrong
        controller is worse than one that names none.
        """
        wanted = segments(target_url)
        if not wanted:
            return None

        hit = self._match(wanted, exact=exact)
        if hit is not None:
            return hit
        # Only now: a base path the server never declares, added by a proxy or
        # UsePathBase. Tried as a fallback rather than up front so a workspace
        # whose routes really do begin with that segment keeps its own answer.
        for base in self._base_paths:
            if wanted[: len(base)] == base and len(wanted) > len(base):
                stripped = self._match(wanted[len(base) :], exact=exact)
                if stripped is not None:
                    return stripped
        return None

    def _match(self, wanted: list[str], *, exact: bool) -> RouteTarget | None:
        matches = [
            target
            for template, target in self._targets
            if (_prefixes(wanted, template) if not exact else _matches(wanted, template))
        ]
        if not matches:
            return None

        # Ambiguous at endpoint level is still decided at file level, which is
        # the granularity every consumer of this works at. A prefix that hits
        # four actions of one controller has one answer to "which file".
        files = {(m.root_label, m.rel_path) for m in matches}
        if len(files) > 1:
            log.debug(
                "routes.ambiguous",
                url="/".join(wanted),
                files=len(files),
                detail="the same path is declared in more than one file; left unresolved",
            )
            return None
        return matches[0]


def _matches(wanted: list[str], template: list[str]) -> bool:
    """A complete URL against a template, segment by segment."""
    for index, part in enumerate(template):
        if _CATCH_ALL.match(part):
            # `{*rest}` takes everything remaining, including nothing.
            return True
        if index >= len(wanted):
            return False
        if _PARAMETER.match(part):
            continue
        if part != wanted[index]:
            return False
    return len(wanted) == len(template)


def _prefixes(wanted: list[str], template: list[str]) -> bool:
    """A known prefix against a template.

    The template must be *longer*, not merely start the same way. The prefix
    came from a template literal, so something was interpolated after it --
    `fetch(\\`/api/remit/${id}\\`)` cannot be the route `api/remit`, which has
    no further segment to interpolate into.
    """
    if len(template) <= len(wanted):
        return False
    for index, part in enumerate(wanted):
        candidate = template[index]
        if _CATCH_ALL.match(candidate):
            return True
        if _PARAMETER.match(candidate):
            continue
        if candidate != part:
            return False
    return True
