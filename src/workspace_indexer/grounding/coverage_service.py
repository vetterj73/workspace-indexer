"""Assemble, per repository, what the index can say about why."""

from __future__ import annotations

from pathlib import Path

from workspace_indexer.discovery import repo_root
from workspace_indexer.grounding.commit_scanner import CommitScanner
from workspace_indexer.grounding.grounding_source import GroundingSource
from workspace_indexer.grounding.marker_scanner import MarkerScanner
from workspace_indexer.grounding.unit_coverage import UnitCoverage
from workspace_indexer.models import DocumentType
from workspace_indexer.obs.logging import get_logger
from workspace_indexer.state import Manifest

log = get_logger("workspace_indexer.grounding")


class CoverageService:
    """Reads the manifest, then asks git what the manifest cannot know.

    Deliberately reports on what is *indexed* rather than what is on disk. The
    report's whole purpose is to tell a reader -- or an agent -- whether an
    empty result means "not written down" or "not retrieved", and that is a
    question about the index. A report over files the index never took in would
    promise answers no query can reach.
    """

    def __init__(
        self,
        manifest: Manifest,
        commits: CommitScanner | None = None,
        markers: MarkerScanner | None = None,
    ) -> None:
        self._manifest = manifest
        self._commits = commits or CommitScanner()
        self._markers = markers or MarkerScanner()
        self._resolved: dict[str, Path | None] = {}
        self._present: dict[str, bool] = {}

    def coverage(self, only: str | None = None) -> list[UnitCoverage]:
        """One entry per repository, largest first.

        `only` narrows to a single repository by label. Grouping still runs over
        every file -- a repository cannot be identified without resolving the
        files that might belong to it -- and only the per-repository `git log`
        and `git grep` are skipped for the rest.

        That is a smaller saving than it sounds: measured on a 1,121-file
        workspace it was 0.44s to 0.37s, because resolving each directory to
        its repository dominates and happens either way. Scoping is here to
        answer about one repository, not to make the answer fast.

        Grouped by the repository each file actually belongs to, resolved from
        git, rather than by the first segment of its path. Those coincide only
        until a workspace is rearranged: a root can hold a repository nested
        below it *and* loose files above that repository, and grouping by path
        would report the two together under whichever the first row happened to
        be -- an answer that changes with row order.
        """
        groups, repos, labels, present = self._group()
        results = [
            self._for_unit(labels[key], counts, repos[key], on_disk=present[key])
            for key, counts in groups.items()
            if only is None or labels[key] == only
        ]
        results.sort(key=lambda c: c.code_files, reverse=True)
        return results

    def repository_labels(self) -> list[str]:
        """Every label `coverage(only=...)` would accept, sorted.

        Shares the grouping pass rather than re-deriving names, so a label that
        selects nothing here cannot be one the report would have printed. Used
        only to name the valid options when a caller asks for a repository that
        does not exist -- an empty result would otherwise read as "this
        repository has no grounding", which is a very different answer.
        """
        _, _, labels, _ = self._group()
        return sorted(set(labels.values()))

    def _group(
        self,
    ) -> tuple[dict[str, dict[str, int]], dict[str, Path | None], dict[str, str], dict[str, bool]]:
        """Assign every indexed file to the repository that versions it."""
        groups: dict[str, dict[str, int]] = {}
        repos: dict[str, Path | None] = {}
        labels: dict[str, str] = {}
        present: dict[str, bool] = {}

        for abs_path, rel_path, doc_type in self._manifest.indexed_documents():
            directory = Path(abs_path).parent
            repo = self._repo_for(directory)
            # Files in no repository stay separated by their top-level
            # directory: merging every loose file in the workspace into one
            # bucket would report a coverage figure for nothing in particular.
            # Only a leading *directory* names a unit. Splitting unconditionally
            # makes every loose file at a root its own unit named after itself,
            # which turned a stale index into forty rows called `.dockerignore`
            # and `PRTester.txt`. Matches how `files_by_unit` derives the same
            # thing, so the two cannot disagree.
            segment = rel_path.split("/")[0] if "/" in rel_path else ""
            key = str(repo) if repo is not None else f"\0{segment}"
            if key not in labels:
                labels[key] = repo.name if repo is not None else (segment or "(root)")
                repos[key] = repo
            # One file still on disk is enough: a unit is stale only when the
            # whole of it has moved, and a single deleted file is ordinary.
            present[key] = present.get(key, False) or self._exists(directory)
            counts = groups.setdefault(key, {})
            counts[doc_type] = counts.get(doc_type, 0) + 1

        return groups, repos, labels, present

    def _repo_for(self, directory: Path) -> Path | None:
        """`repo_root`, memoised per directory.

        Memoised rather than short-circuited by "this directory sits under a
        repository we already found", which would be faster and wrong: a
        submodule's toplevel is itself, not its parent, and the shortcut would
        silently fold one repository's history into another's.
        """
        key = str(directory)
        if key not in self._resolved:
            self._resolved[key] = repo_root(directory)
        return self._resolved[key]

    def _exists(self, directory: Path) -> bool:
        key = str(directory)
        if key not in self._present:
            self._present[key] = directory.is_dir()
        return self._present[key]

    def _for_unit(
        self, label: str, counts: dict[str, int], repo: Path | None, *, on_disk: bool
    ) -> UnitCoverage:
        code_files = counts.get(DocumentType.IMPLEMENTATION.value, 0)
        sources = [
            GroundingSource.by_density(
                "design docs",
                counts.get(DocumentType.DESIGN.value, 0),
                code_files,
                detail="architecture, RFCs, plans -- intent and trade-offs",
            ),
            GroundingSource.by_density(
                "normative docs",
                counts.get(DocumentType.NORMATIVE.value, 0),
                code_files,
                detail="specs, standards, ADRs, conventions -- the rules",
            ),
        ]

        signals = self._commits.scan(repo) if repo is not None else None
        if signals is not None:
            sources.append(
                GroundingSource.by_share(
                    "commit rationale",
                    signals.with_rationale,
                    signals.commits,
                    detail="commits that state a reason, not just a change",
                )
            )

        markers = self._markers.scan(repo) if repo is not None else None
        if markers is not None:
            sources.append(
                GroundingSource.by_density(
                    "decision markers",
                    markers,
                    code_files,
                    detail="WHY: / DECISION: / HACK: comments in tracked code",
                )
            )

        coverage = UnitCoverage(
            label=label,
            code_files=code_files,
            sources=sources,
            signals=signals,
            on_disk=on_disk,
        )
        log.debug(
            "grounding.unit",
            unit=label,
            code_files=code_files,
            verdict=coverage.verdict.value,
        )
        return coverage
