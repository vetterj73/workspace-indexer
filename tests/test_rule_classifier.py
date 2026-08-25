"""Rung one of the classifier chain.

The cases that matter are the ones where two documents about the same subject
sit in the same directory with the same vocabulary. That is the discrimination
no embedding model provided, and it is what this rung has to get right.
"""

from __future__ import annotations

import pytest

from tests.conftest import make_source
from workspace_indexer.classification import (
    Classification,
    DocumentType,
    ModalDensityRule,
    PathRule,
    RuleClassifier,
    modal_density,
)
from workspace_indexer.models import FileKind, SourceFile


def _file(rel_path: str, text: str = "body text", kind: FileKind = FileKind.MARKDOWN) -> SourceFile:
    return make_source(text, kind=kind, language=None, rel_path=rel_path)


def _classify(rel_path: str, text: str = "body", **kwargs: object) -> Classification:
    return RuleClassifier().classify(_file(rel_path, text, **kwargs))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("docs/adr/0007-event-sourcing.md", DocumentType.NORMATIVE),
        ("docs/decisions/use-postgres.md", DocumentType.NORMATIVE),
        ("CLAUDE.md", DocumentType.NORMATIVE),
        (".claude/commands/deploy.md", DocumentType.NORMATIVE),
        ("ai-docs/PythonEngineeringStandards.md", DocumentType.NORMATIVE),
        ("specs/feature/requirements.md", DocumentType.NORMATIVE),
        ("specs/feature/design.md", DocumentType.DESIGN),
        ("docs/architecture.md", DocumentType.DESIGN),
        ("specs/feature/rough-idea.md", DocumentType.DESIGN),
        ("README.md", DocumentType.GUIDE),
        ("CONTRIBUTING.md", DocumentType.GUIDE),
        ("docs/runbook-oncall.md", DocumentType.GUIDE),
        ("CHANGELOG.md", DocumentType.RECORD),
        ("docs/postmortem-2026-01.md", DocumentType.RECORD),
        ("specs/feature/summary.md", DocumentType.RECORD),
    ],
)
def test_path_classification(path: str, expected: DocumentType) -> None:
    assert _classify(path).doc_type is expected


def test_filename_beats_directory() -> None:
    """A changelog living among specifications is still a changelog. This is
    the exact case that motivated the classifier: one directory holding
    requirements, design, plan, summary and changelog for one feature."""
    assert _classify("specs/agent-waves/CHANGELOG.md").doc_type is DocumentType.RECORD
    assert _classify("specs/agent-waves/design.md").doc_type is DocumentType.DESIGN
    assert _classify("specs/agent-waves/requirements.md").doc_type is DocumentType.NORMATIVE
    assert _classify("specs/agent-waves/summary.md").doc_type is DocumentType.RECORD


def test_a_test_about_a_spec_is_still_a_test() -> None:
    """Tests are matched ahead of everything else on purpose."""
    assert _classify("tests/test_adr_parser.py", kind=FileKind.CODE).doc_type is DocumentType.TEST
    assert _classify("docs/adr/tests/test_x.py", kind=FileKind.CODE).doc_type is DocumentType.TEST


def test_code_defaults_to_implementation() -> None:
    assert _classify("src/pkg/manifest.py", kind=FileKind.CODE).doc_type is (
        DocumentType.IMPLEMENTATION
    )


def test_generated_output_is_recognised() -> None:
    for path in ("src/generated/client.ts", "api/service_pb2.py", "proto/user.pb.go"):
        assert _classify(path, kind=FileKind.CODE).doc_type is DocumentType.GENERATED, path


def test_unremarkable_prose_is_left_undecided() -> None:
    """UNKNOWN is an admission, not a category. It is the input to the next
    rung rather than a guess."""
    verdict = _classify("notes/random-thoughts.md", "Some ordinary prose about nothing.")
    assert verdict.doc_type is DocumentType.UNKNOWN
    assert not verdict.decided
    assert verdict.confidence == 0.0


def test_every_verdict_explains_itself() -> None:
    """Auditability is most of the argument for rules over inference: "under
    docs/adr/" is checkable, "the model said so" is not."""
    verdict = _classify("docs/adr/0001-x.md")
    assert verdict.reason.startswith("path:")
    assert "adr" in verdict.reason


def test_confidence_is_lower_for_weaker_signals() -> None:
    """`specs/` is a weak directory hint; a filename match is not."""
    weak = _classify("specs/feature/notes.md")
    strong = _classify("CHANGELOG.md")
    assert weak.confidence < strong.confidence


# ---- frontmatter -------------------------------------------------------


def test_frontmatter_declaration_wins() -> None:
    """An author stating intent beats any inference from location."""
    text = "---\ntype: adr\nstatus: accepted\n---\n\n# Something\n\nBody."
    assert _classify("random/place.md", text).doc_type is DocumentType.NORMATIVE


def test_frontmatter_overrides_a_conflicting_path() -> None:
    text = "---\ntype: changelog\n---\n\n# Notes"
    assert _classify("docs/adr/0001-x.md", text).doc_type is DocumentType.RECORD


def test_malformed_frontmatter_is_ignored_not_fatal() -> None:
    text = "---\ntype: [unclosed\n---\n\nBody"
    assert _classify("README.md", text).doc_type is DocumentType.GUIDE


def test_frontmatter_without_a_type_key_falls_through() -> None:
    text = "---\ntitle: Something\nauthor: someone\n---\n\nBody"
    assert _classify("CHANGELOG.md", text).doc_type is DocumentType.RECORD


# ---- modal density -----------------------------------------------------

SPEC_PROSE = (
    "The service MUST validate every token before dispatch. Implementations "
    "MUST NOT cache credentials beyond the session. Clients SHOULD retry with "
    "backoff, and SHALL abort after five attempts. The handler MUST reject "
    "expired tokens. Callers MUST NOT assume ordering. Responses SHOULD "
    "include a correlation identifier. Servers MUST log every rejection. "
) * 4

DESCRIPTIVE_PROSE = (
    "The service validates tokens before dispatch and caches nothing beyond "
    "the session. Clients retry with backoff and give up after five attempts. "
    "The handler rejects expired tokens, and responses carry a correlation "
    "identifier so the logs can be joined up afterwards. "
) * 4


def test_specification_prose_is_detected_by_how_it_talks() -> None:
    """The linguistic fingerprint of a document that binds behaviour, for
    files whose path and name say nothing useful."""
    assert _classify("notes/unlabelled.md", SPEC_PROSE).doc_type is DocumentType.NORMATIVE


def test_descriptive_prose_is_not() -> None:
    """Same subject, same length, no obligations. This is the discrimination
    the rule exists to make."""
    assert _classify("notes/unlabelled.md", DESCRIPTIVE_PROSE).doc_type is DocumentType.UNKNOWN


def test_modal_density_is_relative_to_length() -> None:
    """Otherwise a long README that says "should" a dozen times in passing
    outranks a short, dense standard."""
    dense = modal_density(SPEC_PROSE)
    diluted = modal_density(SPEC_PROSE + " filler words. " * 500)
    assert dense > diluted


def test_short_documents_are_not_judged_on_style() -> None:
    """Two sentences is not enough evidence to call something a specification."""
    assert modal_density("You MUST do this.") == 0.0


def test_modal_density_ignores_code() -> None:
    """Source is full of `should` in test names and assertions."""
    rule = ModalDensityRule()
    assert rule.apply(_file("src/x.py", SPEC_PROSE, kind=FileKind.CODE)) is None


def test_modal_confidence_stays_below_certainty() -> None:
    """It is an inference, not a declaration, so a later rung can revisit it."""
    verdict = _classify("notes/unlabelled.md", SPEC_PROSE)
    assert 0.5 < verdict.confidence < 1.0


# ---- chain -------------------------------------------------------------


def test_rules_are_consulted_in_order() -> None:
    """Ordering encodes precedence and is the whole design of the chain."""
    calls: list[str] = []

    class Recorder:
        def __init__(self, name: str, verdict: Classification | None) -> None:
            self.name = name
            self._verdict = verdict

        def apply(self, file: SourceFile) -> Classification | None:
            calls.append(self.name)
            return self._verdict

    classifier = RuleClassifier(
        rules=[
            Recorder("first", None),
            Recorder("second", Classification(doc_type=DocumentType.GUIDE, reason="hit")),
            Recorder("third", Classification(doc_type=DocumentType.RECORD, reason="never")),
        ]
    )
    verdict = classifier.classify(_file("x.md"))
    assert verdict.doc_type is DocumentType.GUIDE
    assert calls == ["first", "second"]


def test_classification_is_deterministic() -> None:
    """Rules answer identically every run. That is most of the argument for
    this rung, and what stops payload churn on reindex."""
    a = _classify("docs/adr/0001-x.md", SPEC_PROSE)
    b = _classify("docs/adr/0001-x.md", SPEC_PROSE)
    assert a == b


def test_empty_file_does_not_crash() -> None:
    assert _classify("docs/adr/0001-x.md", "").doc_type is DocumentType.NORMATIVE


def test_path_rule_returns_none_when_nothing_matches() -> None:
    assert PathRule().apply(_file("random/thing.md")) is None
