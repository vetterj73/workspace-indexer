"""Raised when a caller must name a worktree and has not, or named a wrong one."""

from __future__ import annotations


class WorktreeChoiceError(ValueError):
    """One error for both halves of the same question, deliberately.

    "You did not say which checkout you are in" and "the one you named does not
    exist" want the same reply -- here are the choices, pick one -- and
    splitting them would mean two chances to word that reply badly.

    Raised rather than defaulted because neither default is safe. Answering
    from the index would serve a developer working in a worktree the wrong
    checkout, silently and confidently. Reporting divergence across *all*
    worktrees would flag a file because some other agent is mid-edit in a
    branch this caller has never heard of -- and with abandoned branches that
    noise never goes away, so the flag stops being read at all.

    One round trip, once, and the caller then knows for the rest of the
    session. That is the same trade `UnknownDocumentTypeError` makes.
    """

    def __init__(self, given: str | None, available: list[str]) -> None:
        self.given = given
        self.available = available
        listed = ", ".join(available) if available else "none are indexed"
        if given is None:
            super().__init__(
                f"this repository has worktrees ({listed}), so results depend on which "
                "checkout you are working in. Pass worktree=<name> to see it as that "
                'worktree has it, or worktree="none" for the main checkout.'
            )
        else:
            super().__init__(
                f"unknown worktree {given!r}. Available: {listed}. "
                'Pass worktree="none" if you are working in the main checkout.'
            )
