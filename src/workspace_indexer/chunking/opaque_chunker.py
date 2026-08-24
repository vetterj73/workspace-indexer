"""Files we deliberately do not embed."""

from __future__ import annotations

from collections.abc import Iterator

from workspace_indexer.config import ChunkingSection
from workspace_indexer.models import Chunk, FileKind, SourceFile
from workspace_indexer.obs.logging import get_logger, log_once

log = get_logger("workspace_indexer.chunking.opaque")


class OpaqueChunker:
    """Yields nothing, on purpose.

    The file still gets a manifest row, so `status` can report that it is known
    and deliberately unembedded rather than mysteriously absent. Images would
    need a multimodal model, which is a different vector space and therefore a
    different collection — see the plan's "On images" section.
    """

    name = "opaque"
    version = 1
    kinds = frozenset({FileKind.IMAGE, FileKind.OPAQUE})

    def __init__(self, workspace: str) -> None:
        self._workspace = workspace

    def chunk(self, file: SourceFile, config: ChunkingSection) -> Iterator[Chunk]:
        mode = config.opaque.mode
        if mode != "metadata_only":
            # Fail loudly rather than silently indexing nothing: a user who set
            # this expects images in the index.
            log_once(
                log,
                f"opaque:{mode}",
                "chunk.opaque_mode_unsupported",
                mode=mode,
                detail="only metadata_only is implemented; files are recorded but not embedded",
            )
        log.debug("chunk.opaque_recorded", kind=file.kind.value, size=file.size)
        return iter(())
