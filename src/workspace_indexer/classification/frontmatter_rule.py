"""Believe the document when it declares its own type.

ADR and specification templates routinely carry `type: adr` or
`status: accepted` in YAML frontmatter. That is an author stating intent
explicitly, which beats any inference we could make, so this rule runs first.
"""

from __future__ import annotations

import re
from typing import Any, cast

import yaml

from workspace_indexer.classification.classification import Classification
from workspace_indexer.models import DocumentType, SourceFile

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(\n|\Z)", re.DOTALL)

# What authors actually write, mapped to our categories.
_DECLARED: dict[str, DocumentType] = {
    "adr": DocumentType.NORMATIVE,
    "architecture-decision-record": DocumentType.NORMATIVE,
    "spec": DocumentType.NORMATIVE,
    "specification": DocumentType.NORMATIVE,
    "standard": DocumentType.NORMATIVE,
    "convention": DocumentType.NORMATIVE,
    "policy": DocumentType.NORMATIVE,
    "requirements": DocumentType.NORMATIVE,
    "rfc": DocumentType.DESIGN,
    "design": DocumentType.DESIGN,
    "proposal": DocumentType.DESIGN,
    "architecture": DocumentType.DESIGN,
    "guide": DocumentType.GUIDE,
    "tutorial": DocumentType.GUIDE,
    "howto": DocumentType.GUIDE,
    "how-to": DocumentType.GUIDE,
    "runbook": DocumentType.GUIDE,
    "reference": DocumentType.REFERENCE,
    "api": DocumentType.REFERENCE,
    "changelog": DocumentType.RECORD,
    "postmortem": DocumentType.RECORD,
    "retrospective": DocumentType.RECORD,
    "notes": DocumentType.RECORD,
}

_KEYS = ("type", "doc_type", "kind", "category", "layout")


class FrontmatterRule:
    name = "frontmatter"

    def apply(self, file: SourceFile) -> Classification | None:
        if not file.text:
            return None
        match = _FRONTMATTER.match(file.text)
        if match is None:
            return None

        try:
            parsed: Any = yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            # Malformed frontmatter is common and not our problem to report.
            return None
        if not isinstance(parsed, dict):
            return None

        # isinstance narrows `object` only to dict[Unknown, Unknown]; the
        # values are validated individually below.
        fields = cast("dict[str, Any]", parsed)
        for key in _KEYS:
            raw = fields.get(key)
            if not isinstance(raw, str):
                continue
            declared = _DECLARED.get(raw.strip().lower())
            if declared is not None:
                return Classification(
                    doc_type=declared,
                    confidence=1.0,
                    reason=f"frontmatter declares {key}: {raw.strip()}",
                )
        return None
