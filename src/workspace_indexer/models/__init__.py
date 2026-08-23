"""Core data models.

One class per module, re-exported here so callers write
`from workspace_indexer.models import Chunk` rather than reaching into submodules.
"""

from workspace_indexer.models.chunk import Chunk
from workspace_indexer.models.chunk_id import CHUNK_NAMESPACE, compute_chunk_id
from workspace_indexer.models.chunk_meta import ChunkMeta
from workspace_indexer.models.embedding_space import EmbeddingSpace
from workspace_indexer.models.file_kind import FileKind
from workspace_indexer.models.hashing import sha256_text
from workspace_indexer.models.repo_info import RepoInfo
from workspace_indexer.models.run_stats import RunStats
from workspace_indexer.models.search_filters import SearchFilters
from workspace_indexer.models.search_hit import SearchHit
from workspace_indexer.models.source_file import SourceFile
from workspace_indexer.models.sparse_vec import SparseVec

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
