"""Raised when a caller scopes grounding to a repository that is not indexed."""

from __future__ import annotations


class UnknownRepositoryError(ValueError):
    """A repository name that matches nothing, reported rather than returned empty.

    The same failure the taxonomy guards against, and worse here. An empty
    result from *this* tool reads as "this repository records no reasons" --
    which is precisely the finding it exists to deliver, so a typo would
    manufacture the strongest possible claim out of nothing and the agent would
    stop looking for guidance that exists.
    """

    def __init__(self, given: str, known: list[str]) -> None:
        self.given = given
        self.known = known
        listed = ", ".join(known) if known else "none are indexed yet"
        super().__init__(f"unknown repository {given!r}. Indexed repositories: {listed}.")
