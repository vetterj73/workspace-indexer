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


# --- issue #3: oversized blocks ------------------------------------------


def test_structured_mode_splits_on_unindented_lines() -> None:
    """A YAML tree with no blank lines is one paragraph to the prose splitter.

    This is the shape that actually overflowed: `mkdocs.yml`'s `nav:` section
    ran the length of the file with nothing for blank-line splitting to find.
    """
    yaml = "nav:\n  - Home: a.md\n  - Guide: b.md\ntheme:\n  name: material\nplugins:\n  - search\n"

    assert len(split_into_blocks(yaml)) == 1
    assert [b.text.split(":")[0] for b in split_into_blocks(yaml, structured=True)] == [
        "nav",
        "theme",
        "plugins",
    ]


def test_structured_mode_keeps_indented_bodies_with_their_key() -> None:
    yaml = "nav:\n  - Home: a.md\n  - Guide: b.md\ntheme:\n  name: material\n"

    first = split_into_blocks(yaml, structured=True)[0]

    assert first.text.startswith("nav:")
    assert "Guide" in first.text
    assert "theme" not in first.text


def test_an_oversized_paragraph_is_cut_at_line_boundaries() -> None:
    """The data-loss half of #3.

    An uncut 1,200-token block does not stay whole -- the provider truncates it
    silently and the tail is simply gone. Cutting is the only option that keeps
    the content.
    """
    lines = [f"line {n} with enough words on it to cost real tokens" for n in range(200)]
    blocks = [Block(start_line=1, end_line=200, text="\n".join(lines))]

    packed = pack_blocks(blocks, max_tokens=100, kind=FileKind.TEXT)

    assert len(packed) > 1
    assert all(not b.atomic for b in packed)
    # Nothing lost: every original line survives somewhere.
    rejoined = "\n".join(b.text for b in packed)
    assert all(line in rejoined for line in lines)


def test_line_numbers_survive_the_cut() -> None:
    """A hit is useless if it points at the wrong lines."""
    lines = [f"line {n} with enough words on it to cost real tokens" for n in range(60)]
    blocks = [Block(start_line=10, end_line=69, text="\n".join(lines))]

    packed = pack_blocks(blocks, max_tokens=100, kind=FileKind.TEXT)

    assert packed[0].start_line == 10
    assert packed[-1].end_line == 69
    # Contiguous: no line falls into a gap between two chunks.
    for index in range(len(packed) - 1):
        assert packed[index + 1].start_line == packed[index].end_line + 1


def test_an_oversized_fence_is_never_cut() -> None:
    """The guarantee that still holds where it applies.

    Splitting a code fence embeds badly and displays worse, so it is kept whole
    and reported instead.
    """
    body = "\n".join(f"    step_{n}()" for n in range(200))
    fence = Block(start_line=1, end_line=202, text=f"```python\n{body}\n```", atomic=True)

    packed = pack_blocks([fence], max_tokens=100, kind=FileKind.MARKDOWN)

    assert len(packed) == 1
    assert packed[0].text.startswith("```python")
    assert packed[0].text.rstrip().endswith("```")


def test_a_single_line_longer_than_the_budget_comes_back_oversized() -> None:
    """Minified JSON, a base64 blob. There is no boundary to cut on.

    Returned over budget rather than mangled, so the caller can measure it and
    say so -- which is what makes the truncation reportable as deliberate.
    """
    blob = "x" * 20_000
    packed = pack_blocks(
        [Block(start_line=1, end_line=1, text=blob)], max_tokens=100, kind=FileKind.TEXT
    )

    assert len(packed) == 1
    assert packed[0].text == blob


def test_cutting_never_yields_an_empty_block() -> None:
    """A merged group is rejoined with a blank line.

    Cutting one at that boundary produced a block holding nothing but the
    separator -- an empty chunk carrying a real line number, which is worse
    than no chunk at all. Caught by an existing line-number test, which then
    indexed past the end of the file.
    """
    merged = Block(start_line=1, end_line=3, text="alpha\n\nbeta")

    packed = pack_blocks([merged], max_tokens=1, kind=FileKind.TEXT)

    assert all(b.text.strip() for b in packed)


def test_a_merged_block_whose_lines_do_not_map_is_left_whole() -> None:
    """`_merge` rejoins with one blank line whatever the original spacing was.

    When that does not reproduce the file, the offsets are untrustworthy, and a
    hit citing the wrong lines is worse than one that is merely too long.
    """
    # Declares a 10-line span but carries 3 lines: the original had wider gaps.
    mismatched = Block(start_line=1, end_line=10, text="alpha\n\nbeta")

    packed = pack_blocks([mismatched], max_tokens=1, kind=FileKind.TEXT)

    assert len(packed) == 1
    assert packed[0].text == "alpha\n\nbeta"
