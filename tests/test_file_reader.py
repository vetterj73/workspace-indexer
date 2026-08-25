"""Reading a candidate into a SourceFile.

Real files here rather than fakes: this is the layer whose whole job is to find
out what the bytes actually are, and a stubbed read would only confirm the
classification we already guessed from the extension.
"""

from __future__ import annotations

from pathlib import Path

from workspace_indexer.chunking.file_reader import read_source
from workspace_indexer.discovery.file_candidate import FileCandidate
from workspace_indexer.models import FileKind


def _candidate(path: Path, kind: FileKind, language: str | None = None) -> FileCandidate:
    stat = path.stat()
    return FileCandidate(
        root_label="root",
        unit="unit",
        abs_path=path,
        rel_path=path.name,
        kind=kind,
        language=language,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def test_text_is_decoded(tmp_path: Path) -> None:
    path = tmp_path / "a.py"
    path.write_text("x = 1\n", encoding="utf-8")
    source = read_source(_candidate(path, FileKind.CODE, "python"))
    assert source is not None
    assert source.text == "x = 1\n"
    assert source.kind is FileKind.CODE
    assert source.language == "python"


def test_metadata_is_carried_through(tmp_path: Path) -> None:
    path = tmp_path / "a.py"
    path.write_text("x = 1\n", encoding="utf-8")
    source = read_source(_candidate(path, FileKind.CODE, "python"))
    assert source is not None
    assert (source.root_label, source.unit, source.rel_path) == ("root", "unit", "a.py")


def test_nul_byte_downgrades_to_opaque(tmp_path: Path) -> None:
    """The same heuristic git uses. A .txt full of NULs is not text, whatever
    the extension claimed."""
    path = tmp_path / "a.txt"
    path.write_bytes(b"hello\x00\x01\x02world")
    source = read_source(_candidate(path, FileKind.TEXT))
    assert source is not None
    assert source.kind is FileKind.OPAQUE
    assert source.text is None


def test_undecodable_bytes_downgrade_to_opaque(tmp_path: Path) -> None:
    """Latin-1 bytes with no NUL still are not UTF-8; indexing them would
    store mojibake."""
    path = tmp_path / "a.txt"
    path.write_bytes(b"caf\xe9 na\xefve \xff\xfe")
    source = read_source(_candidate(path, FileKind.TEXT))
    assert source is not None
    assert source.kind is FileKind.OPAQUE
    assert source.text is None


def test_language_is_cleared_on_downgrade(tmp_path: Path) -> None:
    """Leaving 'python' on an unparseable blob would send it to the code
    chunker, which would then fail and log a spurious parse warning."""
    path = tmp_path / "a.py"
    path.write_bytes(b"\x00\x01binary")
    source = read_source(_candidate(path, FileKind.CODE, "python"))
    assert source is not None
    assert source.language is None


def test_images_are_never_decoded(tmp_path: Path) -> None:
    path = tmp_path / "logo.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x10" * 64)
    source = read_source(_candidate(path, FileKind.IMAGE))
    assert source is not None
    assert source.kind is FileKind.IMAGE
    assert source.text is None


def test_valid_utf8_multibyte_survives(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    path.write_text("# Café 🙂 naïve\n", encoding="utf-8")
    source = read_source(_candidate(path, FileKind.MARKDOWN, "markdown"))
    assert source is not None
    assert source.text is not None
    assert "🙂" in source.text
    assert source.kind is FileKind.MARKDOWN


def test_nul_beyond_the_sniff_window_is_caught_by_the_decode(tmp_path: Path) -> None:
    """Sniffing only the first 8 KB is a speed choice, not a correctness one —
    the decode still has to reject what the sniff missed."""
    path = tmp_path / "a.txt"
    path.write_bytes(b"a" * 9000 + b"\xff\xfe\xfd")
    source = read_source(_candidate(path, FileKind.TEXT))
    assert source is not None
    assert source.kind is FileKind.OPAQUE


def test_hash_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    path = tmp_path / "a.py"
    path.write_text("x = 1\n", encoding="utf-8")
    first = read_source(_candidate(path, FileKind.CODE, "python"))
    second = read_source(_candidate(path, FileKind.CODE, "python"))
    assert first is not None and second is not None
    assert first.sha256 == second.sha256

    path.write_text("x = 2\n", encoding="utf-8")
    changed = read_source(_candidate(path, FileKind.CODE, "python"))
    assert changed is not None
    assert changed.sha256 != first.sha256


def test_binary_files_still_get_a_hash(tmp_path: Path) -> None:
    """Change detection has to work for files we never embed, or every run
    reports them as modified."""
    path = tmp_path / "blob.so"
    path.write_bytes(b"\x7fELF\x00\x01")
    source = read_source(_candidate(path, FileKind.OPAQUE))
    assert source is not None
    assert len(source.sha256) == 64


def test_vanished_file_returns_none_rather_than_raising(tmp_path: Path) -> None:
    """A normal race on a live workspace: the walk saw it, it is gone now."""
    path = tmp_path / "gone.py"
    path.write_text("x = 1\n", encoding="utf-8")
    candidate = _candidate(path, FileKind.CODE, "python")
    path.unlink()
    assert read_source(candidate) is None


def test_a_file_holding_a_credential_is_not_indexed(tmp_path: Path) -> None:
    """The file is withheld entirely rather than redacted. A file holding a
    live credential is not something to ship to an embedding API, and partial
    redaction invites being clever about it."""
    path = tmp_path / "settings.json"
    token = "github" + "_pat_" + "11ABCDEFG0" + "z" * 30 + "Qw7"
    path.write_text(f'{{"Authorization": "Bearer {token}"}}', encoding="utf-8")
    assert read_source(_candidate(path, FileKind.TEXT)) is None


def test_a_credential_in_source_code_is_withheld_too(tmp_path: Path) -> None:
    path = tmp_path / "client.py"
    key = "AK" + "IA" + "IOSFODNN7EXAMPLE"
    path.write_text(f'AWS_KEY = "{key}"\n', encoding="utf-8")
    assert read_source(_candidate(path, FileKind.CODE, "python")) is None


def test_an_env_template_still_indexes(tmp_path: Path) -> None:
    """Templates are useful to search and hold nothing."""
    path = tmp_path / ".env.example"
    path.write_text("VOYAGE_API_KEY=\nQDRANT_MODE=embedded\n", encoding="utf-8")
    source = read_source(_candidate(path, FileKind.TEXT))
    assert source is not None
    assert source.text is not None


def test_the_allow_list_waives_the_scan(tmp_path: Path) -> None:
    """A high-entropy heuristic will occasionally flag a fixture. Silently
    refusing to index it, with no way to override, is user-hostile."""
    path = tmp_path / "fixture.json"
    key = "AK" + "IA" + "IOSFODNN7EXAMPLE"
    path.write_text(f'{{"token": "{key}"}}', encoding="utf-8")
    candidate = _candidate(path, FileKind.TEXT)
    assert read_source(candidate) is None
    assert read_source(candidate, ["fixture.json"]) is not None


def test_the_allow_list_is_scoped_not_global(tmp_path: Path) -> None:
    """Allowing one file must not disable the check everywhere."""
    path = tmp_path / "other.json"
    key = "AK" + "IA" + "IOSFODNN7EXAMPLE"
    path.write_text(f'{{"token": "{key}"}}', encoding="utf-8")
    assert read_source(_candidate(path, FileKind.TEXT), ["fixture.json"]) is None
