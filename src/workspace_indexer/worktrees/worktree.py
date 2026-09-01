"""One `git worktree add` checkout of an indexed repository."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class Worktree(BaseModel):
    """A second checkout, and the repository it belongs to.

    Carries the main checkout as well as its own path because every use here
    is a comparison between the two: which files differ, and where the copy of
    an indexed file lives. Holding only one of the pair would mean resolving
    the other through git again on every hit.
    """

    path: Path
    # The checkout the index was built from. Its HEAD is what divergence is
    # measured against, because that is what the indexed text reflects.
    main_checkout: Path
    branch: str | None = None

    @property
    def name(self) -> str:
        return self.path.name

    def copy_of(self, indexed_abs_path: Path) -> Path | None:
        """Where this worktree keeps the file indexed at `indexed_abs_path`.

        None when the file is not in this worktree's repository at all -- a
        workspace holds several, and a hit from one of the others has no copy
        here and must be left pointing at itself.
        """
        try:
            within = indexed_abs_path.relative_to(self.main_checkout)
        except ValueError:
            return None
        return self.path / within
