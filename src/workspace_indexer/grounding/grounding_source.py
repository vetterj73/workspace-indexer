"""One measured source of "why", and how much of it there is."""

from __future__ import annotations

from pydantic import BaseModel

from workspace_indexer.grounding.source_strength import SourceStrength

# Documents per 100 code files. A repository with one design document and two
# thousand code files has a design document; it does not have design coverage,
# and an agent told "design docs: present" would read the miss on its own
# module as "no such decision" rather than "not written down".
#
# One-per-hundred is a judgment, not a measurement. It is set where it is
# because a hundred files is roughly a subsystem, and a subsystem with no
# document explaining it is the case this report exists to name.
_DENSE_ENOUGH_PER_100 = 1.0

# Share of non-merge commits carrying a stated reason. Measured across two real
# client repositories the figures were 6.5% and 30.9% -- close to an order of
# magnitude apart, in codebases owned by the same developer. The threshold sits
# between them deliberately: it is calibrated to separate observed cases, not
# derived from anything, and two points is not a curve.
_RATIONALE_SHARE = 0.15

# Below this many commits a share is not evidence of a habit. Found by running
# the report on a real workspace: a four-commit repository scored "present" on
# one commit that happened to say "because", which is arithmetic rather than a
# finding. Too few commits caps the verdict at THIN -- never ABSENT, because
# the reasons may simply not have been written yet.
_MIN_COMMITS_TO_JUDGE = 20


class GroundingSource(BaseModel):
    """A count, its denominator, and a verdict -- in that order of importance.

    The verdict is a convenience over the numbers, never a replacement for
    them. Both are rendered because a threshold nobody can see is a threshold
    nobody can argue with, and this one deserves arguing with.
    """

    name: str
    found: int
    # What `found` is out of. Carried rather than assumed because "3" means
    # opposite things against 12 files and 12,000.
    population: int
    population_unit: str
    strength: SourceStrength
    detail: str

    @property
    def share(self) -> float:
        """`found` as a fraction of its population, or 0.0 when there is none."""
        if self.population <= 0:
            return 0.0
        return self.found / self.population

    @property
    def per_100(self) -> float:
        return self.share * 100

    @classmethod
    def by_density(cls, name: str, found: int, code_files: int, *, detail: str) -> GroundingSource:
        """For sources counted against the code they are supposed to explain."""
        strength = SourceStrength.ABSENT
        if found > 0:
            # No code is not thin coverage of code -- it is a unit holding only
            # documents, where the question this measures does not arise. Left
            # explicit because the arithmetic alone would divide by zero and
            # the guard against that would silently grade it THIN.
            dense = code_files == 0 or (found / code_files) * 100 >= _DENSE_ENOUGH_PER_100
            strength = SourceStrength.PRESENT if dense else SourceStrength.THIN
        return cls(
            name=name,
            found=found,
            population=code_files,
            population_unit="code files",
            strength=strength,
            detail=detail,
        )

    @classmethod
    def by_share(cls, name: str, found: int, commits: int, *, detail: str) -> GroundingSource:
        """For sources counted against commits rather than files."""
        strength = SourceStrength.ABSENT
        if found > 0:
            enough = commits >= _MIN_COMMITS_TO_JUDGE and found / commits >= _RATIONALE_SHARE
            strength = SourceStrength.PRESENT if enough else SourceStrength.THIN
        return cls(
            name=name,
            found=found,
            population=commits,
            population_unit="commits",
            strength=strength,
            detail=detail,
        )
