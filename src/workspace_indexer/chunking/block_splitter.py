"""Split text into blocks, then pack blocks into chunk-sized groups.

Shared by the markdown and plain-text chunkers. Splitting on blank lines is the
only structure plain prose offers; markdown adds fenced code, which has to
survive as one unit — a code fence cut in half embeds badly and displays worse.
"""

from __future__ import annotations

import re

from workspace_indexer.chunking.block import Block
from workspace_indexer.chunking.token_estimate import estimate_tokens
from workspace_indexer.models import FileKind

# An opening fence is at least three backticks or tildes, optionally indented,
# optionally followed by an info string. The closing fence must use the same
# character and be at least as long, which is what lets ```` appear inside a
# ``` block in real documents.
_FENCE = re.compile(r"^(?P<indent>\s{0,3})(?P<fence>`{3,}|~{3,})\s*(?P<info>.*)$")


def _closes(line: str, char: str, length: int) -> bool:
    match = _FENCE.match(line)
    if match is None:
        return False
    fence = match.group("fence")
    return fence[0] == char and len(fence) >= length and not match.group("info").strip()


def split_into_blocks(text: str, *, respect_fences: bool = False) -> list[Block]:
    """Blank-line separated paragraphs, with fenced code kept whole."""
    lines = text.splitlines()
    blocks: list[Block] = []
    current: list[str] = []
    start = 1

    index = 0
    while index < len(lines):
        line = lines[index]

        if respect_fences:
            opening = _FENCE.match(line)
            if opening is not None:
                if current:
                    blocks.append(_finish(current, start))
                    current = []
                fence = opening.group("fence")
                fence_start = index + 1
                body = [line]
                index += 1
                while index < len(lines):
                    body.append(lines[index])
                    if _closes(lines[index], fence[0], len(fence)):
                        index += 1
                        break
                    index += 1
                blocks.append(
                    Block(
                        start_line=fence_start,
                        end_line=fence_start + len(body) - 1,
                        text="\n".join(body),
                        atomic=True,
                    )
                )
                start = index + 1
                continue

        if line.strip():
            if not current:
                start = index + 1
            current.append(line)
        elif current:
            blocks.append(_finish(current, start))
            current = []
        index += 1

    if current:
        blocks.append(_finish(current, start))
    return blocks


def _finish(lines: list[str], start: int) -> Block:
    return Block(start_line=start, end_line=start + len(lines) - 1, text="\n".join(lines))


def pack_blocks(
    blocks: list[Block],
    *,
    max_tokens: int,
    kind: FileKind,
    overlap: int = 0,
) -> list[Block]:
    """Greedily group blocks up to max_tokens.

    A single block larger than the budget becomes its own oversized group
    rather than being cut: for an atomic fence that is required, and for a
    lone enormous paragraph it is the least-bad option. The embed-time
    `truncate` setting is the backstop.
    """
    if not blocks:
        return []

    packed: list[Block] = []
    group: list[Block] = []
    tokens = 0

    for block in blocks:
        block_tokens = estimate_tokens(block.text, kind)
        if group and tokens + block_tokens > max_tokens:
            packed.append(_merge(group))
            group = _carry_over(group, overlap)
            tokens = sum(estimate_tokens(b.text, kind) for b in group)
        group.append(block)
        tokens += block_tokens

    if group:
        packed.append(_merge(group))
    return packed


def _carry_over(group: list[Block], overlap: int) -> list[Block]:
    """Repeat trailing blocks at the head of the next chunk.

    Overlap exists so a passage split across a boundary is still retrievable
    from either side. Atomic blocks are never carried over — duplicating a
    whole code fence wastes far more budget than the continuity is worth.
    """
    if overlap <= 0:
        return []
    tail = [b for b in group[-overlap:] if not b.atomic]
    return list(tail)


def _merge(group: list[Block]) -> Block:
    return Block(
        start_line=group[0].start_line,
        end_line=group[-1].end_line,
        text="\n\n".join(b.text for b in group),
        atomic=all(b.atomic for b in group),
    )
