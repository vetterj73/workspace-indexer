"""How a file is classified, which decides the chunking strategy."""

from __future__ import annotations

from enum import StrEnum


class FileKind(StrEnum):
    CODE = "code"
    MARKDOWN = "markdown"
    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"
    OPAQUE = "opaque"
