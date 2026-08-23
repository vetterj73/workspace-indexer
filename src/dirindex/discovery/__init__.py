"""Filesystem discovery: what to index, and what we know about it."""

from dirindex.discovery.classify import classify, is_lockfile
from dirindex.discovery.file_candidate import FileCandidate
from dirindex.discovery.git_metadata import is_repo, read_repo_info
from dirindex.discovery.ignore_matcher import IgnoreMatcher
from dirindex.discovery.skip_reason import SkipReason
from dirindex.discovery.walker import Walker

__all__ = [
    "FileCandidate",
    "IgnoreMatcher",
    "SkipReason",
    "Walker",
    "classify",
    "is_lockfile",
    "is_repo",
    "read_repo_info",
]
