"""Raised when a caller names a document type that does not exist."""

from __future__ import annotations


class UnknownDocumentTypeError(ValueError):
    """An unrecognised type name, reported as an error rather than a filter.

    The alternative -- treating an unknown type as "match nothing" -- is the
    single worst failure mode this server has. An agent asks for `type=spec`,
    the taxonomy calls it `normative`, and an empty result set reads as "this
    workspace has no specifications". It then proceeds to write code with no
    guidance and no idea it missed any. A loud error costs one round trip; a
    silent empty result costs the whole task.
    """

    def __init__(self, given: str, *, valid: list[str], aliases: list[str]) -> None:
        self.given = given
        self.valid = valid
        message = f"unknown document type {given!r}. Valid types: {', '.join(valid)}."
        if aliases:
            message += f" Accepted aliases: {', '.join(aliases)}."
        super().__init__(message)
