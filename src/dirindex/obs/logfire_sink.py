"""Optional Logfire sink.

Off by default, and the reason is data egress rather than taste: pydantic-ai's
instrumentation captures call inputs, and for an embedding call the input is
your source code. With send_to_cloud enabled, chunks of private repositories
leave the machine. That has to be a deliberate choice.

send_to_cloud=False still buys you local spans over the embedding calls, which
is genuinely useful and costs nothing.
"""

from __future__ import annotations

import logging

from dirindex.config import LogfireConfig


def attach_logfire(cfg: LogfireConfig, root: logging.Logger) -> None:
    try:
        import logfire
    except ImportError:
        logging.getLogger("dirindex.obs").warning(
            "logfire.unavailable", extra={"hint": "poetry install --extras logfire"}
        )
        return

    if cfg.send_to_cloud:
        # Deliberately loud: this is the point where source text starts leaving
        # the machine, and it should never be a surprise.
        logging.getLogger("dirindex.obs").warning(
            "logfire.cloud_enabled: source text from indexed files will be "
            "transmitted to Pydantic's hosted service"
        )

    logfire.configure(
        send_to_logfire=cfg.send_to_cloud,
        service_name="dirindex",
        console=False,
    )
    logfire.instrument_pydantic_ai()

    from logfire.integrations.logging import LogfireLoggingHandler

    # An *additional* handler. The console and file sinks are untouched, so
    # nothing in the app depends on Logfire being installed or reachable.
    root.addHandler(LogfireLoggingHandler())
