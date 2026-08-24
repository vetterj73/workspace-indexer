"""Why a file is or is not being reindexed."""

from __future__ import annotations

from enum import StrEnum


class IndexDecision(StrEnum):
    """The rungs of the decision ladder, cheapest test first.

    Recorded per file in the log and tallied per run, because "why isn't this
    file in the index" and "why did this run cost so much" are the two
    questions a user actually asks.
    """

    # Rung 1: mtime and size both unchanged. One stat(), no read.
    SKIP_UNCHANGED = "skip_unchanged"
    # Rung 2: content read and hashed, hash identical. A formatter pass or a
    # `git checkout` of the same bytes lands here, and costs no embedding.
    SKIP_SAME_CONTENT = "skip_same_content"
    # Rung 3: genuinely new content.
    REINDEX = "reindex"
    # Rung 4: we changed a chunking strategy, so content hashes are irrelevant.
    RECHUNK_STRATEGY = "rechunk_strategy"
    # Rung 6: the file is current but has nothing in the active embedding
    # space. This is the model-swap backfill path.
    BACKFILL_SPACE = "backfill_space"
    # Never seen before.
    NEW = "new"
    # --force
    FORCED = "forced"

    @property
    def needs_read(self) -> bool:
        return self is not IndexDecision.SKIP_UNCHANGED

    @property
    def needs_embedding(self) -> bool:
        return self not in {
            IndexDecision.SKIP_UNCHANGED,
            IndexDecision.SKIP_SAME_CONTENT,
        }
