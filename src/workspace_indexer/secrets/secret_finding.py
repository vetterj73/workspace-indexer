"""A credential spotted in a file, described without quoting it."""

from __future__ import annotations

from pydantic import BaseModel


class SecretFinding(BaseModel):
    """Deliberately carries no value.

    The whole point is to keep the credential out of the index; putting it in a
    log line or an exception message would defeat that, and logs are shipped
    and shared like anything else.
    """

    rule: str
    # 1-based, so it matches what an editor shows.
    line: int
    # A short description of the shape found, e.g. "GitHub personal access
    # token". Never the token.
    description: str

    def __str__(self) -> str:
        return f"{self.description} (rule {self.rule}, line {self.line})"
