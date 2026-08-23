"""What to do with files we cannot read as text (images, binaries)."""

from __future__ import annotations

from typing import Literal

from workspace_indexer.config.strict import Strict


class OpaqueChunking(Strict):
    # metadata_only: record the file exists, embed nothing.
    # caption / multimodal: declared upgrade paths, not implemented yet.
    mode: Literal["metadata_only", "caption", "multimodal"] = "metadata_only"
