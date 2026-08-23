"""Core data models.

One class per module, re-exported here so callers write
`from dirindex.models import Chunk` rather than reaching into submodules.
"""

from dirindex.models.chunk import Chunk
from dirindex.models.chunk_id import CHUNK_NAMESPACE, compute_chunk_id
from dirindex.models.chunk_meta import ChunkMeta
from dirindex.models.embedding_space import EmbeddingSpace
from dirindex.models.file_kind import FileKind
from dirindex.models.hashing import sha256_text
from dirindex.models.repo_info import RepoInfo
from dirindex.models.run_stats import RunStats
from dirindex.models.search_filters import SearchFilters
from dirindex.models.search_hit import SearchHit
from dirindex.models.source_file import SourceFile
from dirindex.models.sparse_vec import SparseVec

__all__ = [
    "CHUNK_NAMESPACE",
    "Chunk",
    "ChunkMeta",
    "EmbeddingSpace",
    "FileKind",
    "RepoInfo",
    "RunStats",
    "SearchFilters",
    "SearchHit",
    "SourceFile",
    "SparseVec",
    "compute_chunk_id",
    "sha256_text",
]
