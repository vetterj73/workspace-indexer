"""Contextvar binding — the reason a traceback tells you *which* file broke.

Bind once at the top of a scope and every log line emitted anywhere below it,
including from deep inside the chunker or an embedding retry loop, carries the
run id and the file being processed. Without this you get a stack trace and no
indication which of forty thousand files produced it.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import structlog


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


@contextmanager
def bound(**fields: Any) -> Generator[None]:
    tokens = structlog.contextvars.bind_contextvars(**fields)
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)


@contextmanager
def file_context(root_label: str, rel_path: str) -> Generator[None]:
    with bound(root_label=root_label, rel_path=rel_path):
        yield
