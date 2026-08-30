"""An HTTP endpoint a file exposes."""

from __future__ import annotations

from pydantic import BaseModel


class RouteDeclaration(BaseModel):
    """One endpoint, as the server declares it.

    Not an edge. An edge points from a caller to a callee, and a route
    declaration is the other end -- a thing that can be pointed at. It is a
    symbol table entry, which is why these live in their own table rather than
    beside import edges.
    """

    # The effective template, already assembled from whatever the framework
    # split it across. In ASP.NET that is the class attribute joined to the
    # action attribute; measured against a real workspace, *no* HttpGet/HttpPost
    # attribute carried a template of its own, so the join is the whole story.
    #
    # Stored as written rather than normalised, because normalisation is a
    # matching decision and this is a record of what the code says. The
    # resolver normalises both sides when it compares them.
    template: str
    # GET/POST/..., or None when the declaration does not say. A bare
    # `[Route(...)]` with no verb attribute answers every method, and guessing
    # GET would invent a constraint the code does not have.
    method: str | None = None
    line: int
    # What declared it: `controller` for an ASP.NET action, `page` for a Razor
    # page. Kept because the two are found and matched differently, and a
    # coverage report that merged them would hide one of them going to zero.
    kind: str
