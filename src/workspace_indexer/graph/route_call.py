"""A place in client code that calls an HTTP endpoint."""

from __future__ import annotations

from pydantic import BaseModel


class RouteCall(BaseModel):
    """One `fetch` or `axios` call site, with as much of the URL as is knowable
    without running the program.

    This *is* an edge, and the reason the whole idea is worth building: it
    crosses repositories, which no import edge does. A React page and the C#
    controller it calls share a string and nothing else -- no symbol, no import,
    nothing a language server could follow.
    """

    # What could be recovered statically. For a plain string that is the whole
    # URL; for a template literal it is the part before the first `${`.
    target: str
    line: int
    # False when `target` is only the leading part of the URL, because the rest
    # was interpolated. A prefix still identifies an endpoint -- `/api/remit/`
    # names exactly one controller -- but it cannot distinguish two routes that
    # share it, and the resolver must not pretend otherwise.
    #
    # Measured against a real React workspace, only 3 of 32 call sites used a
    # plain literal and 11 more were template literals, so most edges that
    # resolve at all will resolve through a prefix. Treating them as exact
    # would quietly overstate the graph's precision.
    exact: bool = True
