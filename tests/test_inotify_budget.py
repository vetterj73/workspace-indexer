"""Counting inotify watches against a limit that is shared, not ours alone.

Exhausting `max_user_watches` surfaces as `OSError: [Errno 28] No space left on
device`, which has nothing to do with disk space and sends people looking in
entirely the wrong place.
"""

from __future__ import annotations

from pathlib import Path

import structlog.testing

from workspace_indexer.watching import InotifyBudget


def _tree(root: Path, depth: int, width: int) -> None:
    if depth == 0:
        return
    for i in range(width):
        child = root / f"d{i}"
        child.mkdir()
        (child / "file.txt").write_text("x", encoding="utf-8")
        _tree(child, depth - 1, width)


def test_counts_directories_not_files(tmp_path: Path) -> None:
    """inotify watches directories: one watch reports changes to a directory's
    immediate children. So a tree's cost is its directory count, and files --
    however many -- are free."""
    (tmp_path / "one").mkdir()
    for i in range(50):
        (tmp_path / "one" / f"f{i}.py").write_text("x", encoding="utf-8")

    assert InotifyBudget(limit=100).count_directories(tmp_path) == 2


def test_counts_nested_directories(tmp_path: Path) -> None:
    _tree(tmp_path, depth=2, width=3)
    # root + 3 children + 9 grandchildren
    assert InotifyBudget(limit=100).count_directories(tmp_path) == 13


def test_excluded_directories_are_not_counted(tmp_path: Path) -> None:
    """The reason ignore rules keep the watcher functional, not just the index
    tidy: node_modules is deep and wide, and it is what exhausts the limit."""
    _tree(tmp_path, depth=2, width=2)
    heavy = tmp_path / "node_modules"
    heavy.mkdir()
    _tree(heavy, depth=3, width=4)

    with_heavy = InotifyBudget(limit=10_000).count_directories(tmp_path)
    without = InotifyBudget(limit=10_000).count_directories(tmp_path, excluded={"node_modules"})
    assert without < with_heavy
    assert without == 7


def test_a_symlinked_directory_is_not_descended(tmp_path: Path) -> None:
    """Otherwise a link back up the tree counts forever."""
    real = tmp_path / "real"
    real.mkdir()
    (real / "child").mkdir()
    (tmp_path / "link").symlink_to(real, target_is_directory=True)

    assert InotifyBudget(limit=100).count_directories(tmp_path) == 3


def test_exceeding_the_limit_is_reported_with_the_real_cause(tmp_path: Path) -> None:
    with structlog.testing.capture_logs() as logs:
        fits = InotifyBudget(limit=2).check(500)

    assert fits is False
    error = next(e for e in logs if e["event"] == "watch.budget_exceeded")
    # The Errno 28 message is the least helpful in Linux; say what it means.
    assert "No space left on device" in str(error["detail"])
    assert "node_modules" in str(error["detail"])


def test_approaching_the_limit_warns(tmp_path: Path) -> None:
    """Warns well below 1.0: the limit is shared with every editor and language
    server already running, so 80% ours means the next one cannot watch."""
    with structlog.testing.capture_logs() as logs:
        assert InotifyBudget(limit=100).check(85) is True

    warning = next(e for e in logs if e["event"] == "watch.budget_tight")
    assert warning["used_share"] == 0.85


def test_comfortable_headroom_is_logged_at_info(tmp_path: Path) -> None:
    with structlog.testing.capture_logs() as logs:
        assert InotifyBudget(limit=100_000).check(300) is True

    assert [e for e in logs if e["event"] == "watch.budget"]
    assert not [e for e in logs if e["event"].startswith("watch.budget_")]


def test_an_unreadable_limit_does_not_block_the_watch() -> None:
    """Not Linux, or no /proc. Reported as unknown rather than treated as zero."""
    budget = InotifyBudget(limit=None)
    with structlog.testing.capture_logs() as logs:
        assert budget.check(10_000) is True

    assert budget.limit is None
    assert [e for e in logs if e["event"] == "watch.budget_unknown"]


def test_an_unreadable_directory_is_skipped_not_fatal(tmp_path: Path) -> None:
    """A directory we cannot list is a permissions fact, not a crash."""
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "inner").mkdir()
    blocked.chmod(0o000)
    try:
        assert InotifyBudget(limit=100).count_directories(tmp_path) == 2
    finally:
        blocked.chmod(0o755)
