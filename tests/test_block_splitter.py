"""Block splitting and packing, shared by the markdown and text chunkers."""

from __future__ import annotations

from workspace_indexer.chunking.block import Block
from workspace_indexer.chunking.block_splitter import pack_blocks, split_into_blocks
from workspace_indexer.models import FileKind

FENCED = """Intro paragraph.

```python
def f():
    return 1

def g():
    return 2
```

Trailing paragraph.
"""


def test_paragraphs_split_on_blank_lines() -> None:
    blocks = split_into_blocks("one\nstill one\n\ntwo\n\n\nthree\n")
    assert [b.text for b in blocks] == ["one\nstill one", "two", "three"]


def test_line_numbers_are_one_based_and_inclusive() -> None:
    """These become file:line links, so an off-by-one is user-visible."""
    blocks = split_into_blocks("alpha\n\nbeta\ngamma\n")
    assert (blocks[0].start_line, blocks[0].end_line) == (1, 1)
    assert (blocks[1].start_line, blocks[1].end_line) == (3, 4)


def test_leading_blank_lines_do_not_shift_numbering() -> None:
    blocks = split_into_blocks("\n\nalpha\n")
    assert (blocks[0].start_line, blocks[0].end_line) == (3, 3)


def test_empty_input_yields_nothing() -> None:
    assert split_into_blocks("") == []
    assert split_into_blocks("\n\n  \n") == []


def test_fences_are_kept_whole_despite_internal_blank_lines() -> None:
    """A blank line inside a code fence is not a paragraph break; splitting
    there would cut the block in half."""
    blocks = split_into_blocks(FENCED, respect_fences=True)
    fenced = [b for b in blocks if b.atomic]
    assert len(fenced) == 1
    assert "def f()" in fenced[0].text
    assert "def g()" in fenced[0].text


def test_fences_ignored_when_not_respecting_them() -> None:
    """Plain text has no fence concept, so the same input splits differently."""
    blocks = split_into_blocks(FENCED, respect_fences=False)
    assert not any(b.atomic for b in blocks)
    assert len(blocks) > len(split_into_blocks(FENCED, respect_fences=True))


def test_longer_fence_can_contain_a_shorter_one() -> None:
    """````  wrapping ```  is how real docs show fenced markdown."""
    text = "````\n```python\nx = 1\n```\n````\n\nafter\n"
    blocks = split_into_blocks(text, respect_fences=True)
    atomic = [b for b in blocks if b.atomic]
    assert len(atomic) == 1
    assert atomic[0].text.count("```") >= 3
    assert blocks[-1].text == "after"


def test_unterminated_fence_consumes_to_end_without_raising() -> None:
    blocks = split_into_blocks("```python\nx = 1\n", respect_fences=True)
    assert len(blocks) == 1
    assert blocks[0].atomic


def _blocks(count: int, size: int) -> list[Block]:
    """Each block carries a unique marker.

    Identical filler would make `text.count(...)` meaningless: 800 repeats of
    one character contain a 400-run at many offsets, so a merged group would
    look like it held duplicates when it did not.
    """
    return [
        Block(start_line=i * 2 + 1, end_line=i * 2 + 1, text=f"<b{i}>" + "w" * size)
        for i in range(count)
    ]


def test_packing_groups_up_to_the_budget() -> None:
    packed = pack_blocks(_blocks(4, 400), max_tokens=250, kind=FileKind.TEXT)
    assert len(packed) == 2


def test_packing_preserves_span_across_a_group() -> None:
    packed = pack_blocks(_blocks(3, 40), max_tokens=1000, kind=FileKind.TEXT)
    assert len(packed) == 1
    assert (packed[0].start_line, packed[0].end_line) == (1, 5)


def test_oversized_block_becomes_its_own_group_rather_than_being_cut() -> None:
    blocks = [Block(start_line=1, end_line=1, text="w" * 20_000)]
    packed = pack_blocks(blocks, max_tokens=100, kind=FileKind.TEXT)
    assert len(packed) == 1
    assert packed[0].text == blocks[0].text


def test_overlap_repeats_the_previous_tail() -> None:
    """A passage split across a boundary should be retrievable from either
    side."""
    blocks = _blocks(4, 400)
    packed = pack_blocks(blocks, max_tokens=250, kind=FileKind.TEXT, overlap=1)
    assert blocks[1].text in packed[1].text


def test_no_overlap_by_default() -> None:
    blocks = _blocks(4, 400)
    packed = pack_blocks(blocks, max_tokens=250, kind=FileKind.TEXT)
    # Every block appears exactly once across all groups.
    joined = "".join(group.text for group in packed)
    assert [joined.count(f"<b{i}>") for i in range(4)] == [1, 1, 1, 1]


def test_atomic_blocks_are_never_carried_into_the_overlap() -> None:
    """Duplicating a whole code fence costs more budget than the continuity is
    worth."""
    blocks = [
        Block(start_line=1, end_line=1, text="<a>" + "a" * 400),
        Block(start_line=3, end_line=6, text="```\n<fence>" + "b" * 400 + "\n```", atomic=True),
        Block(start_line=8, end_line=8, text="<c>" + "c" * 400),
    ]
    packed = pack_blocks(blocks, max_tokens=200, kind=FileKind.TEXT, overlap=1)
    joined = "".join(group.text for group in packed)
    assert joined.count("<fence>") == 1


def test_packing_empty_input() -> None:
    assert pack_blocks([], max_tokens=100, kind=FileKind.TEXT) == []
