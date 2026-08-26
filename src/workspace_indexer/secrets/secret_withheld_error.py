"""Raised when a file is refused because it contains a credential."""

from __future__ import annotations

from workspace_indexer.secrets.secret_finding import SecretFinding


class SecretWithheldError(Exception):
    """An exception rather than a None return, deliberately.

    "The file vanished" and "the file contains a secret" both mean no
    SourceFile, but they demand opposite handling: a vanished file is a normal
    race to shrug at, while a withheld one must have any previously-indexed
    chunks purged. Detecting a secret and leaving the old copy in the index
    defeats the entire point.

    Carries findings that describe what was seen, never the value.
    """

    def __init__(self, rel_path: str, findings: list[SecretFinding]) -> None:
        self.rel_path = rel_path
        self.findings = findings
        super().__init__(f"{rel_path}: {len(findings)} secret finding(s)")
