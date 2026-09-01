"""The thresholds that turn counts into a verdict.

These are judgment calls, which is exactly why they get tests: a threshold
nobody has pinned down drifts, and the whole report is an argument about where
the line sits.
"""

from __future__ import annotations

from workspace_indexer.grounding import GroundingSource, SourceStrength


def test_nothing_found_is_absent_however_small_the_codebase() -> None:
    source = GroundingSource.by_density("design docs", 0, 3, detail="")

    assert source.strength is SourceStrength.ABSENT
    assert source.share == 0.0


def test_one_document_against_a_large_codebase_is_thin_not_present() -> None:
    """The case the report exists for.

    A single architecture document in a two-thousand-file repository is real
    and nearly useless, and calling it "present" would teach an agent to read
    its own empty result as "no such decision was made".
    """
    source = GroundingSource.by_density("design docs", 1, 2_000, detail="")

    assert source.strength is SourceStrength.THIN
    assert source.per_100 == 0.05


def test_a_document_per_hundred_files_is_present() -> None:
    source = GroundingSource.by_density("design docs", 5, 500, detail="")

    assert source.strength is SourceStrength.PRESENT
    assert source.per_100 == 1.0


def test_documents_with_no_code_are_not_graded_thin() -> None:
    """A docs-only unit has nothing to be thin *relative to*.

    Without this the division guard would grade a directory of pure
    documentation as poorly covered, which inverts the finding.
    """
    source = GroundingSource.by_density("design docs", 4, 0, detail="")

    assert source.strength is SourceStrength.PRESENT
    assert source.share == 0.0


def test_rationale_below_the_share_is_thin() -> None:
    # The measured shape of a repository whose history records what changed and
    # not why: real commits, almost no stated reasons.
    source = GroundingSource.by_share("commit rationale", 11, 170, detail="")

    assert source.strength is SourceStrength.THIN
    assert round(source.share, 3) == 0.065


def test_rationale_above_the_share_is_present() -> None:
    source = GroundingSource.by_share("commit rationale", 96, 311, detail="")

    assert source.strength is SourceStrength.PRESENT


def test_no_commits_cannot_be_present() -> None:
    """An empty history divides by nothing and must not score as covered."""
    source = GroundingSource.by_share("commit rationale", 0, 0, detail="")

    assert source.strength is SourceStrength.ABSENT
    assert source.share == 0.0


def test_a_share_over_too_few_commits_cannot_reach_present() -> None:
    """Found by running the report on a real workspace.

    A four-commit repository scored "present" because one commit happened to
    contain the word "because". 25% of four is arithmetic, not a habit.
    """
    source = GroundingSource.by_share("commit rationale", 1, 4, detail="")

    assert source.strength is SourceStrength.THIN
    assert source.share == 0.25


def test_too_few_commits_still_reports_thin_rather_than_absent() -> None:
    """A young repository has not failed to explain itself, only not yet."""
    source = GroundingSource.by_share("commit rationale", 3, 5, detail="")

    assert source.strength is SourceStrength.THIN


def test_a_strong_share_over_enough_commits_is_present() -> None:
    source = GroundingSource.by_share("commit rationale", 5, 20, detail="")

    assert source.strength is SourceStrength.PRESENT
