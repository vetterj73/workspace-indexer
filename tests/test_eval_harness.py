"""Retrieval-quality scoring.

The arithmetic matters: a harness that reports the wrong number is worse than
none, because decisions get made on it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from workspace_indexer.evaluation import EvalCase, EvalReport, EvalResult, load_cases


def _result(expected: list[str], found: list[str]) -> EvalResult:
    rank = next(
        (i for i, path in enumerate(found, 1) if any(w in path for w in expected)), None
    )
    return EvalResult(query="q", expected=expected, found=found, first_hit_rank=rank)


def test_recall_is_the_fraction_of_expected_files_found() -> None:
    result = _result(["a.py", "b.py"], ["src/a.py", "src/z.py"])
    assert result.recall == pytest.approx(0.5)


def test_recall_of_a_complete_miss_is_zero() -> None:
    assert _result(["a.py"], ["src/z.py"]).recall == 0.0


def test_expected_paths_match_as_substrings() -> None:
    """So a case survives a file moving within a directory without the dataset
    needing a rewrite."""
    assert _result(["widget.py"], ["src/pkg/widget.py"]).recall == 1.0


def test_reciprocal_rank_rewards_ranking_first() -> None:
    """Recall alone cannot see the difference between rank 1 and rank 10."""
    assert _result(["a.py"], ["a.py", "b.py"]).reciprocal_rank == 1.0
    assert _result(["a.py"], ["b.py", "a.py"]).reciprocal_rank == pytest.approx(0.5)


def test_reciprocal_rank_of_a_miss_is_zero() -> None:
    assert _result(["a.py"], ["z.py"]).reciprocal_rank == 0.0


def test_report_averages_across_cases() -> None:
    report = EvalReport(
        label="t",
        limit=10,
        results=[_result(["a.py"], ["a.py"]), _result(["b.py"], ["z.py"])],
    )
    assert report.recall_at_k == pytest.approx(0.5)
    assert report.mrr_at_k == pytest.approx(0.5)


def test_empty_report_does_not_divide_by_zero() -> None:
    report = EvalReport(label="t", limit=10, results=[])
    assert report.recall_at_k == 0.0
    assert report.mrr_at_k == 0.0


def test_misses_are_listed_for_inspection() -> None:
    """A score without the failing queries is not actionable."""
    report = EvalReport(
        label="t",
        limit=10,
        results=[_result(["a.py"], ["a.py"]), _result(["b.py"], ["z.py"])],
    )
    assert [m.expected for m in report.misses] == [["b.py"]]


def test_loading_a_dataset(tmp_path: Path) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text(
        "- query: how does the watcher pick a backend\n"
        "  expect: [watcher.py]\n"
        "- query: where is the IDF modifier set\n"
        "  expect: [qdrant_store.py]\n"
        "  note: the easiest thing here to get wrong\n",
        encoding="utf-8",
    )
    cases = load_cases(path)
    assert len(cases) == 2
    assert cases[0].expect == ["watcher.py"]
    assert cases[1].note


def test_missing_dataset_explains_what_to_write(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="real queries"):
        load_cases(tmp_path / "nope.yaml")


def test_empty_dataset_is_not_an_error(tmp_path: Path) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text("\n", encoding="utf-8")
    assert load_cases(path) == []


def test_malformed_case_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text("- query: missing the expect key\n", encoding="utf-8")
    with pytest.raises(Exception, match="expect"):
        load_cases(path)


def test_case_model_requires_a_query() -> None:
    with pytest.raises(Exception, match="query"):
        EvalCase.model_validate({"expect": ["a.py"]})


def test_expect_matching_is_case_insensitive() -> None:
    """A dataset saying CONTRIBUTING.md must match docs/contributing.md.
    Scoring a successful retrieval as a miss makes the measurement lie."""
    assert _result(["CONTRIBUTING.md"], ["ralph/docs/contributing.md"]).recall == 1.0


def test_first_hit_rank_is_also_case_insensitive() -> None:
    """recall and rank iterate in opposite directions; both have to agree, or a
    case gets full recall and a reciprocal rank of zero."""
    result = _result(["CONTRIBUTING.md"], ["a.md", "docs/CONTRIBUTING.md"])
    assert result.recall == 1.0
    assert result.first_hit_rank == 2


def test_match_direction_is_not_reversible() -> None:
    """The expectation is a substring of the path, never the other way round.
    A (str, list[str]) signature would let the arguments be swapped silently,
    because both orderings typecheck."""
    from workspace_indexer.evaluation.eval_result import path_matches

    assert path_matches("manifest.py", "src/state/manifest.py")
    assert not path_matches("src/state/manifest.py", "manifest.py")
