"""Persisting eval runs.

The measurements are the one artefact in this project that is not derived: the
index regenerates in minutes, but what recall *was* before a change already
made cannot be recovered at all. So they live in the repository as committed
files rather than in the gitignored manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from workspace_indexer.evaluation import (
    SCHEMA_VERSION,
    EvalRecord,
    EvalResult,
    compare,
    latest_comparable,
    read_records,
    write_record,
)


def _record(
    *,
    stamp: str = "2026-08-26T04:00:00+00:00",
    recall: float = 0.8,
    mrr: float = 0.6,
    config_hash: str = "abc123",
    fusion: str = "rrf",
    reranker: str = "voyageai:rerank-2.5-lite",
    ranks: dict[str, int | None] | None = None,
) -> EvalRecord:
    ranks = ranks or {"query one": 1, "query two": 3}
    return EvalRecord(
        recorded_at=stamp,
        label="test-run",
        config_hash=config_hash,
        space_slug="voyageai_voyage-code-4_1024",
        embedding_model="voyageai:voyage-code-4",
        dimensions=1024,
        fusion=fusion,
        reranker=reranker,
        limit=10,
        recall_at_k=recall,
        mrr_at_k=mrr,
        case_count=len(ranks),
        miss_count=0,
        results=[
            EvalResult(query=q, expected=["a.py"], found=["a.py"], first_hit_rank=r)
            for q, r in ranks.items()
        ],
    )


def test_a_run_round_trips(tmp_path: Path) -> None:
    written = write_record(_record(), tmp_path)
    (loaded,) = read_records(tmp_path)
    assert loaded.recall_at_k == 0.8
    assert loaded.schema_version == SCHEMA_VERSION
    assert written.exists()


def test_the_file_is_human_readable_and_diffable(tmp_path: Path) -> None:
    """The whole reason for files over a database: a pull request should show
    that a change moved the number."""
    path = write_record(_record(), tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "\n" in text
    assert '"recall_at_k"' in text
    # Sorted keys, so a diff shows a change rather than a reordering.
    parsed = json.loads(text)
    assert list(parsed) == sorted(parsed)


def test_the_written_shape_matches_the_declared_schema(tmp_path: Path) -> None:
    """JSON does not enforce a shape the way a columnar format would, so the
    discipline has to come from here."""
    path = write_record(_record(), tmp_path)
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert set(parsed) == set(EvalRecord.model_fields)
    assert parsed["schema_version"] == SCHEMA_VERSION


def test_filenames_are_unique_and_sortable(tmp_path: Path) -> None:
    a = write_record(_record(stamp="2026-08-26T04:00:00+00:00"), tmp_path)
    b = write_record(_record(stamp="2026-08-26T05:00:00+00:00"), tmp_path)
    assert a != b
    assert sorted([a.name, b.name]) == [a.name, b.name]


def test_filenames_survive_a_label_with_slashes(tmp_path: Path) -> None:
    """Model names contain colons and slashes; a filename cannot."""
    record = _record().model_copy(update={"label": "voyageai:voyage-code-4 fusion=rrf"})
    path = write_record(record, tmp_path)
    assert path.exists()
    assert "/" not in path.name


def test_reading_an_empty_directory(tmp_path: Path) -> None:
    assert read_records(tmp_path) == []
    assert read_records(tmp_path / "missing") == []


def test_an_unreadable_file_is_skipped_not_fatal(tmp_path: Path) -> None:
    """One corrupt file must not stop you comparing the others."""
    write_record(_record(), tmp_path)
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert len(read_records(tmp_path)) == 1


def test_records_come_back_oldest_first(tmp_path: Path) -> None:
    write_record(_record(stamp="2026-08-26T05:00:00+00:00"), tmp_path)
    write_record(_record(stamp="2026-08-26T04:00:00+00:00"), tmp_path)
    stamps = [r.recorded_at for r in read_records(tmp_path)]
    assert stamps == sorted(stamps)


# ---- comparability -----------------------------------------------------


def test_a_different_config_is_not_comparable() -> None:
    """Different config hashes measure different systems. Reporting a delta
    across them looks authoritative and is not."""
    assert not _record(config_hash="aaa").comparable_to(_record(config_hash="bbb"))


def test_a_different_fusion_mode_is_not_comparable() -> None:
    assert not _record(fusion="rrf").comparable_to(_record(fusion="dense_only"))


def test_a_different_reranker_is_not_comparable() -> None:
    """Turning reranking off is a deliberate experiment, not a regression. An
    automatic "vs last run" that conflated them would report a 0.3 drop in MRR
    as though something had broken -- which is exactly what happened before
    reranker was part of this check."""
    assert not _record(reranker="voyageai:rerank-2.5-lite").comparable_to(_record(reranker="none"))


def test_the_same_configuration_is_comparable() -> None:
    assert _record().comparable_to(_record(stamp="2026-08-26T05:00:00+00:00"))


def test_latest_comparable_ignores_a_different_configuration(tmp_path: Path) -> None:
    write_record(_record(stamp="2026-08-26T01:00:00+00:00"), tmp_path)
    write_record(_record(stamp="2026-08-26T02:00:00+00:00", reranker="none"), tmp_path)
    current = _record(stamp="2026-08-26T03:00:00+00:00")
    found = latest_comparable(read_records(tmp_path), current)
    assert found is not None
    assert found.recorded_at == "2026-08-26T01:00:00+00:00"


def test_latest_comparable_never_returns_a_later_run(tmp_path: Path) -> None:
    write_record(_record(stamp="2026-08-26T09:00:00+00:00"), tmp_path)
    current = _record(stamp="2026-08-26T03:00:00+00:00")
    assert latest_comparable(read_records(tmp_path), current) is None


def test_latest_comparable_with_no_history(tmp_path: Path) -> None:
    assert latest_comparable(read_records(tmp_path), _record()) is None


# ---- comparison --------------------------------------------------------


def test_aggregate_deltas() -> None:
    result = compare(_record(recall=0.7, mrr=0.5), _record(recall=0.8, mrr=0.6))
    assert result.recall_delta == pytest.approx(0.1)
    assert result.mrr_delta == pytest.approx(0.1)


def test_a_case_moving_up_is_an_improvement() -> None:
    result = compare(_record(ranks={"q": 5}), _record(ranks={"q": 2}))
    assert [m.query for m in result.improved] == ["q"]
    assert result.regressed == []


def test_a_case_moving_down_is_a_regression() -> None:
    result = compare(_record(ranks={"q": 2}), _record(ranks={"q": 5}))
    assert [m.query for m in result.regressed] == ["q"]


def test_a_case_that_stopped_matching_is_a_regression() -> None:
    result = compare(_record(ranks={"q": 3}), _record(ranks={"q": None}))
    assert result.regressed


def test_a_case_that_started_matching_is_an_improvement() -> None:
    result = compare(_record(ranks={"q": None}), _record(ranks={"q": 3}))
    assert result.improved


def test_an_unchanged_case_is_neither() -> None:
    result = compare(_record(ranks={"q": 3}), _record(ranks={"q": 3}))
    assert not result.improved
    assert not result.regressed


def test_an_average_can_improve_while_cases_regress() -> None:
    """The reason per-case movement is reported at all: an aggregate that went
    up while two cases broke is a result worth seeing."""
    before = _record(ranks={"a": 5, "b": 1, "c": 1}, mrr=0.5)
    after = _record(ranks={"a": 1, "b": 4, "c": 4}, mrr=0.6)
    result = compare(before, after)
    assert result.mrr_delta > 0
    assert len(result.regressed) == 2


def test_cases_are_matched_by_query_not_position() -> None:
    """So reordering the dataset does not look like a change."""
    before = _record(ranks={"first": 1, "second": 2})
    after = _record(ranks={"second": 2, "first": 1})
    assert compare(before, after).movements
    assert not compare(before, after).regressed


def test_a_new_case_is_not_reported_as_movement() -> None:
    """It has no earlier rank to have moved from."""
    result = compare(_record(ranks={"old": 1}), _record(ranks={"old": 1, "new": 2}))
    assert [m.query for m in result.movements] == ["old"]


def test_the_report_is_generated_not_hand_written(tmp_path: Path) -> None:
    """The previous version of this document was prose I typed, with nothing
    stopping it drifting from what the runs actually produced."""
    from workspace_indexer.evaluation import render

    text = render([_record(recall=0.875, mrr=0.745)])
    assert "0.875" in text
    assert "Do not edit by hand" in text


def test_the_report_omits_query_text(tmp_path: Path) -> None:
    """A document quoting the eval queries becomes a perfect match for them,
    which puts it at the top of its own results. That bug has now happened
    twice: once via config/eval.yaml, once via this document."""
    from workspace_indexer.evaluation import render

    text = render([_record(ranks={"how should I structure a new module": 1})])
    assert "how should I structure a new module" not in text


def test_the_report_survives_regeneration(tmp_path: Path) -> None:
    """The DuckDB guidance is part of the generator, not appended by hand --
    anything appended is lost on the next run."""
    from workspace_indexer.evaluation import render

    assert "read_json_auto" in render([])
    assert "read_json_auto" in render([_record()])


def test_an_empty_report_says_so(tmp_path: Path) -> None:
    from workspace_indexer.evaluation import render

    assert "No runs recorded yet" in render([])
