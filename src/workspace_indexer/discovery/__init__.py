"""Filesystem discovery: what to index, and what we know about it."""

from workspace_indexer.discovery.classify import classify, is_lockfile
from workspace_indexer.discovery.file_candidate import FileCandidate
from workspace_indexer.discovery.git_metadata import is_repo, read_repo_info, repo_root
from workspace_indexer.discovery.ignore_matcher import IgnoreMatcher
from workspace_indexer.discovery.skip_reason import SkipReason
from workspace_indexer.discovery.walker import Walker

__all__ = [
    "FileCandidate",
    "IgnoreMatcher",
    "SkipReason",
    "Walker",
    "classify",
    "is_lockfile",
    "is_repo",
    "read_repo_info",
    "repo_root",
]
