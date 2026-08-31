"""One endpoint a call could have reached."""

from __future__ import annotations

from pydantic import BaseModel


class RouteTarget(BaseModel):
    """A declaration, addressed well enough to resolve to.

    Carries the root as well as the path, which the import graph never needs
    to: an import resolves inside its own repository by design, and a route
    edge is worth having *because* it crosses one. A bare rel_path would be
    ambiguous the moment two repositories both have `Api/HomeController.cs`.
    """

    root_label: str
    rel_path: str
    template: str
