"""Optional Logfire tracing settings."""

from __future__ import annotations

from dirindex.config.strict import Strict


class LogfireConfig(Strict):
    enabled: bool = False
    # Off by default because pydantic-ai instrumentation captures call inputs,
    # and for an embedding call the input is your source code.
    send_to_cloud: bool = False
