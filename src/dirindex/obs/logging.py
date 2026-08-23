"""Logging setup: pretty console + rolling JSONL file.

Two sinks, deliberately at different levels. The console follows the
configured level because a human is watching it; the file is always DEBUG
because you cannot retroactively raise a log level after the failure you
needed to see. The file is JSON lines so it can be queried with jq:

    jq 'select(.event=="embed.batch") | .duration_ms' logs/dirindex.jsonl
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Any

import structlog

from dirindex.config import LoggingConfig

# Third-party loggers that are chatty enough to bury our own events.
_NOISY = (
    "httpx",
    "httpcore",
    "urllib3",
    "qdrant_client",
    "onnxruntime",
    "filelock",
    "huggingface_hub",
    "sentence_transformers",
    "opentelemetry",
)

_configured = False
_seen_once: set[str] = set()


def _shared_processors() -> list[Any]:
    """Processors applied to every event, whatever the sink."""
    return [
        # merge_contextvars must come first: it injects run_id/rel_path bound by
        # obs.context so every downstream processor and renderer sees them.
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]


def configure_logging(cfg: LoggingConfig) -> None:
    """Install the console and file sinks. Idempotent."""
    global _configured
    if _configured:
        return

    shared = _shared_processors()

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    root = logging.getLogger()
    # The root sits at DEBUG so the file handler can see everything; each
    # handler filters independently below.
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    if cfg.console != "off":
        root.addHandler(_console_handler(cfg, shared))
    if cfg.file is not None:
        root.addHandler(_file_handler(cfg, shared))

    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)

    if cfg.logfire.enabled:
        from dirindex.obs.logfire_sink import attach_logfire

        attach_logfire(cfg.logfire, root)

    _configured = True


def _console_handler(cfg: LoggingConfig, shared: list[Any]) -> logging.Handler:
    if cfg.console == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, cfg.level))
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
        )
    )
    return handler


def _file_handler(cfg: LoggingConfig, shared: list[Any]) -> logging.Handler:
    assert cfg.file is not None
    path = Path(cfg.file.path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Size-based rather than time-based rotation: indexer output is bursty. A
    # full reindex writes everything in minutes then goes quiet for days, so
    # rotating daily would give one enormous file and nine empty ones.
    handler = logging.handlers.RotatingFileHandler(
        path,
        maxBytes=cfg.file.max_bytes,
        backupCount=cfg.file.backup_count,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                # dict_tracebacks renders exceptions as structured data rather
                # than one giant embedded string, which keeps them greppable.
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer(),
            ],
        )
    )
    return handler


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.stdlib.get_logger(name)


def log_once(logger: structlog.stdlib.BoundLogger, key: str, event: str, **fields: Any) -> None:
    """Log an event at most once per process.

    For conditions that are permanent for the life of the run — reranking is
    unconfigured, a grammar is missing — repeating the message on every one of
    ten thousand items is noise that hides real events.
    """
    if key in _seen_once:
        return
    _seen_once.add(key)
    logger.info(event, **fields)


def reset_for_tests() -> None:
    """Allow a test to reconfigure logging from scratch."""
    global _configured
    _configured = False
    _seen_once.clear()
    structlog.reset_defaults()
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.close()
        root.removeHandler(handler)
