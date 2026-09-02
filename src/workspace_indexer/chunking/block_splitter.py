"""Split text into blocks, then pack blocks into chunk-sized groups.

Shared by the markdown and plain-text chunkers. Splitting on blank lines is the
only structure plain prose offers; markdown adds fenced code, which has to
survive as one unit — a code fence cut in half embeds badly and displays worse.
"""

from __future__ import annotations

import re

from workspace_indexer.chunking.block import Block
from workspace_indexer.chunking.token_estimate import estimate_tokens, tokens_to_bytes
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


def split_into_blocks(
    text: str, *, respect_fences: bool = False, structured: bool = False
) -> list[Block]:
    """Blank-line separated paragraphs, with fenced code kept whole.

    `structured` additionally starts a new block at every unindented line,
    which is the one piece of structure YAML, TOML and INI all share: a
    top-level key, a `[section]` header, a list item at column zero. Without it
    a `nav:` tree with no blank lines in it is a single paragraph the length of
    the file -- the shape that actually overflowed the budget in practice.
    """
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
            if structured and current and not line[:1].isspace():
                # A new top-level entry begins. Blank lines still separate as
                # usual; this only adds boundaries where the format has
                # structure and the prose splitter would see none.
                blocks.append(_finish(current, start))
                current = []
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

    An **atomic** block larger than the budget becomes its own oversized
    group rather than being cut: splitting a code fence embeds badly and
    displays worse, so the tradeoff is deliberate and the embed-time
    `truncate` setting is the backstop.

    Anything else that overflows is cut at line boundaries. That guarantee was
    never worth keeping for a paragraph: the provider truncates silently rather
    than failing, so an uncut 1,200-token block does not stay whole, it loses
    its tail with no error. Cutting it is the only option that keeps the
    content. Fences are untouched, so the reason the rule existed still holds
    where it applies.

    A single *line* longer than the budget -- minified JSON, a base64 blob --
    cannot be cut further and is returned oversized. Callers detect that by
    measuring the group they get back.
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

    divided: list[Block] = []
    for block in packed:
        if block.atomic or estimate_tokens(block.text, kind) <= max_tokens:
            divided.append(block)
            continue
        divided.extend(_by_lines(block, max_tokens=max_tokens, kind=kind))
    return divided


def _by_lines(block: Block, *, max_tokens: int, kind: FileKind) -> list[Block]:
    """Cut an oversized non-atomic block at line boundaries, greedily.

    Lines rather than characters because every format this runs on is
    line-oriented, and a cut mid-line produces a fragment that is wrong in both
    halves rather than merely partial.
    """
    # Measured in bytes rather than by summing per-line token estimates.
    # `estimate_tokens` rounds, so a sum of rounded lines drifts from the
    # rounded sum -- and the newlines rejoining them are not in either. That
    # drift put 19-line parts six tokens over a budget they were built to fit.
    budget_bytes = tokens_to_bytes(max_tokens, kind)
    lines = block.text.splitlines()
    if len(lines) <= 1:
        # Nothing to cut on. Returned as-is and over budget: the caller can see
        # that and say so, which beats pretending it fits.
        return [block]

    if len(lines) != block.end_line - block.start_line + 1:
        # A merged group: `_merge` rejoins with a blank line whatever the
        # original spacing was, so its text no longer maps line-for-line onto
        # the file. Cutting it would hand back line numbers pointing at the
        # wrong place, and a hit that cites the wrong lines is worse than one
        # that is too long. Left whole for the caller to report.
        return [block]

    parts: list[Block] = []
    current: list[str] = []
    used = 0
    start = block.start_line

    def flush(end: int) -> None:
        # Blank-only runs are dropped rather than emitted. A merged group is
        # rejoined with a blank line, so cutting one at a boundary can leave a
        # part holding nothing but that separator -- an empty chunk with a real
        # line number, which is worse than no chunk at all.
        if any(line.strip() for line in current):
            parts.append(Block(start_line=start, end_line=end, text="\n".join(current)))

    for offset, line in enumerate(lines):
        # +1 for the newline that will rejoin it to the line before.
        cost = len(line.encode("utf-8", "replace")) + 1
        if current and used + cost > budget_bytes:
            flush(block.start_line + offset - 1)
            current = []
            used = 0
            start = block.start_line + offset
        current.append(line)
        used += cost

    if current:
        flush(block.start_line + len(lines) - 1)
    # Everything was blank: hand back the original rather than nothing.
    return parts or [block]


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
    """Rejoin blocks, restoring the blank lines that separated them.

    Joining with a fixed blank line was close enough while nothing measured
    the result, but it makes the merged text stop mapping line-for-line onto
    the file -- so a later attempt to cut it cannot trust its own line numbers,
    and a group that overflowed by fifteen tokens had to be left whole and
    mislabelled indivisible. Reconstructing the real gaps keeps the mapping,
    and makes `source_text` what it claims to be: the bytes from the file.

    Overlap can repeat an earlier block, which puts the group out of order. The
    gap is then meaningless, so a single blank line is used and the mismatch
    that produces is what tells `_by_lines` not to trust the offsets.
    """
    parts = [group[0].text]
    for previous, following in zip(group, group[1:], strict=False):
        gap = following.start_line - previous.end_line - 1
        parts.append("\n" * (gap + 1) if gap >= 0 else "\n\n")
        parts.append(following.text)
    return Block(
        start_line=group[0].start_line,
        end_line=group[-1].end_line,
        text="".join(parts),
        atomic=all(b.atomic for b in group),
    )
