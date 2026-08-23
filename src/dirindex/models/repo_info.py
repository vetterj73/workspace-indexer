"""Git provenance for a root, when the root happens to be a repository."""

from __future__ import annotations

from pydantic import BaseModel


class RepoInfo(BaseModel):
    name: str
    remote_url: str | None = None
    branch: str | None = None
    head_sha: str | None = None
    is_dirty: bool = False
