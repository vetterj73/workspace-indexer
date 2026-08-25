"""Classify by where a file sits and what it is called.

The strongest cheap signal available. Ordering matters and is the point:
filename beats directory, because `specs/feature/CHANGELOG.md` is a record that
happens to live among specifications.
"""

from __future__ import annotations

from fnmatch import fnmatch

from workspace_indexer.classification.classification import Classification
from workspace_indexer.classification.document_type import DocumentType
from workspace_indexer.models import SourceFile

# (glob, type, confidence, why). Consulted in order; first match wins, so more
# specific patterns come first.
_PATTERNS: list[tuple[str, DocumentType, float, str]] = [
    # Generated and vendored output: machine-written, no retrievable intent.
    ("**/generated/**", DocumentType.GENERATED, 1.0, "under generated/"),
    ("**/__generated__/**", DocumentType.GENERATED, 1.0, "under __generated__/"),
    ("**/migrations/**", DocumentType.GENERATED, 0.8, "under migrations/"),
    ("**/*.pb.*", DocumentType.GENERATED, 1.0, "protobuf output"),
    ("**/*_pb2.py", DocumentType.GENERATED, 1.0, "protobuf output"),
    ("**/*.snap", DocumentType.TEST, 1.0, "snapshot fixture"),
    # Tests. Deliberately ahead of everything else: a test *about* an ADR is
    # still a test.
    ("**/tests/**", DocumentType.TEST, 1.0, "under tests/"),
    ("**/test/**", DocumentType.TEST, 1.0, "under test/"),
    ("**/test_*.*", DocumentType.TEST, 1.0, "test_ prefix"),
    ("**/*_test.*", DocumentType.TEST, 1.0, "_test suffix"),
    ("**/*.test.*", DocumentType.TEST, 1.0, ".test. infix"),
    ("**/*.spec.ts", DocumentType.TEST, 1.0, "spec file"),
    ("**/*.spec.js", DocumentType.TEST, 1.0, "spec file"),
    ("**/conftest.py", DocumentType.TEST, 1.0, "pytest fixtures"),
    ("**/fixtures/**", DocumentType.TEST, 0.9, "under fixtures/"),
    ("**/cassettes/**", DocumentType.TEST, 0.9, "recorded fixtures"),
    # Records: things that describe what happened. Filename beats directory,
    # which is why these precede the specs/ and docs/ patterns below.
    ("**/CHANGELOG*", DocumentType.RECORD, 1.0, "changelog"),
    ("**/HISTORY*", DocumentType.RECORD, 1.0, "history file"),
    ("**/RELEASE_NOTES*", DocumentType.RECORD, 1.0, "release notes"),
    ("**/summary.md", DocumentType.RECORD, 0.8, "summary document"),
    ("**/postmortem*", DocumentType.RECORD, 0.9, "postmortem"),
    ("**/retrospective*", DocumentType.RECORD, 0.9, "retrospective"),
    ("**/meeting*", DocumentType.RECORD, 0.8, "meeting notes"),
    # Guides: how to use or operate. CONTRIBUTING is deliberately a guide, not
    # a normative document -- it explains a workflow rather than binding how
    # code must be written.
    ("**/README*", DocumentType.GUIDE, 1.0, "readme"),
    ("**/CONTRIBUTING*", DocumentType.GUIDE, 1.0, "contributing guide"),
    ("**/contributing*", DocumentType.GUIDE, 0.9, "contributing guide"),
    ("**/INSTALL*", DocumentType.GUIDE, 0.9, "install guide"),
    ("**/tutorial*/**", DocumentType.GUIDE, 0.9, "under tutorials/"),
    ("**/runbook*", DocumentType.GUIDE, 0.9, "runbook"),
    ("**/getting-started*", DocumentType.GUIDE, 0.9, "getting started"),
    ("**/quickstart*", DocumentType.GUIDE, 0.9, "quickstart"),
    # Normative: how it must be built.
    ("**/adr/**", DocumentType.NORMATIVE, 1.0, "under adr/"),
    ("**/adrs/**", DocumentType.NORMATIVE, 1.0, "under adrs/"),
    ("**/decisions/**", DocumentType.NORMATIVE, 0.9, "under decisions/"),
    ("**/conventions*", DocumentType.NORMATIVE, 0.9, "conventions document"),
    ("**/standards*", DocumentType.NORMATIVE, 0.9, "standards document"),
    ("**/*Standards*", DocumentType.NORMATIVE, 0.9, "standards document"),
    ("**/*standards*", DocumentType.NORMATIVE, 0.9, "standards document"),
    ("**/requirements.md", DocumentType.NORMATIVE, 0.9, "requirements document"),
    ("**/policy*", DocumentType.NORMATIVE, 0.9, "policy document"),
    ("**/style-guide*", DocumentType.NORMATIVE, 0.9, "style guide"),
    # Agent instructions. They bind how work is done in a repository, which is
    # what `find_guidance` should surface for "what conventions apply here".
    ("**/CLAUDE.md", DocumentType.NORMATIVE, 1.0, "agent instructions"),
    ("**/AGENTS.md", DocumentType.NORMATIVE, 1.0, "agent instructions"),
    ("**/.claude/**", DocumentType.NORMATIVE, 0.8, "under .claude/"),
    ("**/.cursor/**", DocumentType.NORMATIVE, 0.8, "under .cursor/"),
    # Design: how it is shaped and why.
    ("**/design.md", DocumentType.DESIGN, 1.0, "design document"),
    ("**/*-design.md", DocumentType.DESIGN, 0.9, "design document"),
    ("**/architecture*", DocumentType.DESIGN, 0.9, "architecture document"),
    ("**/rfc*/**", DocumentType.DESIGN, 0.9, "under rfcs/"),
    ("**/proposal*", DocumentType.DESIGN, 0.8, "proposal"),
    ("**/rough-idea*", DocumentType.DESIGN, 0.8, "early design note"),
    ("**/plan.md", DocumentType.DESIGN, 0.7, "plan document"),
    ("**/*-plan.md", DocumentType.DESIGN, 0.7, "plan document"),
    # Reference.
    ("**/api/**", DocumentType.REFERENCE, 0.7, "under api/"),
    ("**/reference/**", DocumentType.REFERENCE, 0.8, "under reference/"),
    # Weakest directory signals last, so anything above wins.
    ("**/specs/**", DocumentType.NORMATIVE, 0.7, "under specs/"),
    ("**/spec/**", DocumentType.NORMATIVE, 0.7, "under spec/"),
]


class PathRule:
    name = "path"

    def apply(self, file: SourceFile) -> Classification | None:
        # Match on the full path so directory patterns work, but anchor the
        # leading slash so `**/x` matches at the root too.
        probe = "/" + file.rel_path.lstrip("/")
        for pattern, doc_type, confidence, why in _PATTERNS:
            if fnmatch(probe, pattern) or fnmatch(probe, pattern.replace("**/", "/", 1)):
                return Classification(doc_type=doc_type, confidence=confidence, reason=why)
        return None

