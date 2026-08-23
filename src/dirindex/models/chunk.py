"""One indexable unit of a file."""

from __future__ import annotations

from pydantic import BaseModel

from dirindex.models.chunk_id import compute_chunk_id
from dirindex.models.chunk_meta import ChunkMeta


class Chunk(BaseModel):
    meta: ChunkMeta
    # The exact bytes from the file. This is what we show the caller.
    source_text: str
    # Context header + source_text. This is what we embed and rerank against,
    # because a method lifted out of its class is meaningless on its own.
    embed_text: str
    chunk_id: str = ""

    def model_post_init(self, _: object) -> None:
        if not self.chunk_id:
            self.chunk_id = compute_chunk_id(
                self.meta.root_label,
                self.meta.rel_path,
                self.meta.symbol_path,
                self.meta.chunk_index,
                self.meta.content_sha,
            )
