"""One query and the files it should surface."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# Retrieval cases ask "can the index find a thing at all" and are the
# regression net. Guidance cases ask the question an agent asks on greenfield
# work -- "how am I supposed to build this?" -- where the tempting wrong
# answers sit in the same directory with the same vocabulary. They are the
# document-classification argument, and they are scored separately because
# averaging them together hides exactly the movement worth seeing.
Group = Literal["retrieval", "guidance"]


class EvalCase(BaseModel):
    query: str
    group: Group = "retrieval"
    # Substring match against rel_path, so a case survives a file moving
    # within a directory without needing the dataset rewritten.
    expect: list[str]
    note: str | None = None
