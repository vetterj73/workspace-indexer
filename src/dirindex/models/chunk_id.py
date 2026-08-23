"""Deterministic chunk identity.

Qdrant point ids must be an unsigned int or a UUID — a raw sha256 hex string is
rejected — so we hash the identifying material into a uuid5 and use that as the
point id directly.
"""

from __future__ import annotations

import uuid

# Stable namespace so ids are reproducible across processes and machines.
CHUNK_NAMESPACE = uuid.UUID("b4f1c0de-0000-4000-8000-646972696e64")


def compute_chunk_id(
    root_label: str,
    rel_path: str,
    symbol_path: str | None,
    chunk_index: int,
    content_sha: str,
) -> str:
    """Content-addressed and stable.

    Edit one function and only that function's id changes, so the delete/upsert
    set on reindex is exactly the chunks that actually changed.
    """
    key = f"{root_label}\x00{rel_path}\x00{symbol_path or ''}\x00{chunk_index}\x00{content_sha}"
    return str(uuid.uuid5(CHUNK_NAMESPACE, key))
