"""Reading and writing eval runs as committed files.

Deliberately not the SQLite manifest. `data/` is gitignored derived state whose
documented recovery story is "re-index" -- results kept there would be deleted
on every rebuild, lost when this moves machines, and invisible in git.

Eval results are the one artefact here that is not derived. The index
regenerates in minutes; what recall *was* before a change already made cannot
be recovered at all. So they live in the repository, where a pull request can
show that a change moved the number.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from workspace_indexer.evaluation.eval_record import EvalRecord
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.evaluation.store")

DEFAULT_EVAL_DIR = Path("evals")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _filename(record: EvalRecord) -> str:
    stamp = record.recorded_at.replace(":", "-").replace("+00:00", "").rstrip("Z")
    label = _UNSAFE.sub("-", record.label).strip("-")[:80]
    return f"{stamp}-{label}.json"


def write_record(record: EvalRecord, directory: Path = DEFAULT_EVAL_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _filename(record)
    # Indented and key-sorted so a diff shows what actually changed rather than
    # a reordering. The whole reason for choosing files over a database.
    path.write_text(
        json.dumps(record.model_dump(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log.info("eval.recorded", path=str(path), recall=record.recall_at_k, mrr=record.mrr_at_k)
    return path


def read_records(directory: Path = DEFAULT_EVAL_DIR) -> list[EvalRecord]:
    """Every run on disk, oldest first. Unreadable files are skipped loudly
    rather than aborting a comparison."""
    if not directory.is_dir():
        return []
    records: list[EvalRecord] = []
    for path in sorted(directory.glob("*.json")):
        try:
            records.append(EvalRecord.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception as exc:
            log.warning("eval.unreadable", path=str(path), error=f"{type(exc).__name__}: {exc}")
    records.sort(key=lambda r: r.recorded_at)
    return records


def latest_comparable(records: list[EvalRecord], to: EvalRecord) -> EvalRecord | None:
    """The most recent earlier run measuring the same system."""
    for record in reversed(records):
        if record.recorded_at >= to.recorded_at:
            continue
        if record.comparable_to(to):
            return record
    return None
