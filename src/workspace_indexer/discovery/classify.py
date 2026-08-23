"""Decide a file's kind and language from its path.

Extension-only, deliberately: classification runs on every path in the tree and
must stay at zero I/O. Whether a file is *actually* decodable text is settled by
the reader, which downgrades to OPAQUE if it finds NUL bytes or invalid UTF-8.
"""

from __future__ import annotations

from pathlib import Path

from tree_sitter_language_pack import detect_language_from_path

from workspace_indexer.models import FileKind

MARKDOWN_EXTS = frozenset({".md", ".markdown", ".mdx", ".mdc"})
PDF_EXTS = frozenset({".pdf"})
IMAGE_EXTS = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".ico", ".tif", ".tiff", ".avif"}
)
# Structured data parses fine with tree-sitter, but symbol-level chunking of a
# config file buys nothing over paragraph packing, so route it to the text
# chunker and keep the code path focused on code.
DATA_EXTS = frozenset(
    {
        ".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".toml",
        ".ini", ".cfg", ".conf", ".properties",
    }
)
TEXT_EXTS = frozenset(
    {".txt", ".rst", ".adoc", ".org", ".csv", ".tsv", ".log", ".env", ".snap", ".diff", ".patch"}
)

# Suffixes that only mark a file as a copy or a stub. `.env.example` is an
# env file, not a file of type "example", so we strip these and classify the
# name underneath.
MODIFIER_EXTS = frozenset(
    {".example", ".sample", ".template", ".dist", ".in", ".orig", ".bak", ".tmpl"}
)
BINARY_EXTS = frozenset(
    {
        ".zip", ".gz", ".bz2", ".xz", ".zst", ".tar", ".7z", ".rar",
        ".so", ".dylib", ".dll", ".exe", ".bin", ".o", ".a", ".class", ".jar",
        ".pyc", ".pyo", ".wasm", ".pdb", ".db", ".sqlite", ".sqlite3",
        ".mp3", ".mp4", ".wav", ".mov", ".avi", ".mkv", ".ogg", ".flac",
        ".woff", ".woff2", ".ttf", ".otf", ".eot",
        ".parquet", ".arrow", ".npy", ".npz", ".pkl", ".pt", ".onnx", ".safetensors",
    }
)

# Generated dependency manifests: enormous, churn constantly, and contain no
# information a human or an agent would ever search for by meaning.
LOCKFILE_NAMES = frozenset(
    {
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
        "Cargo.lock", "Gemfile.lock", "composer.lock", "go.sum", "uv.lock",
        "flake.lock", "Pipfile.lock",
    }
)


# Extensionless files that do have a tree-sitter grammar; detect_language_from_path
# keys off the extension, so these need naming explicitly.
FILENAME_LANGUAGES: dict[str, str] = {
    "Makefile": "make",
    "makefile": "make",
    "GNUmakefile": "make",
    "Dockerfile": "dockerfile",
    "Containerfile": "dockerfile",
    "CMakeLists.txt": "cmake",
    "Gemfile": "ruby",
    "Rakefile": "ruby",
    "Vagrantfile": "ruby",
    "BUILD": "starlark",
    "WORKSPACE": "starlark",
}


def is_lockfile(path: Path) -> bool:
    return path.name in LOCKFILE_NAMES or path.suffix == ".lock"


def classify(path: Path) -> tuple[FileKind, str | None]:
    """Return the kind and, for code, the tree-sitter language name."""
    known = FILENAME_LANGUAGES.get(path.name)
    if known:
        return FileKind.CODE, known

    ext = path.suffix.lower()
    if ext in MODIFIER_EXTS:
        return classify(path.with_suffix(""))

    if ext in MARKDOWN_EXTS:
        return FileKind.MARKDOWN, "markdown"
    if ext in PDF_EXTS:
        return FileKind.PDF, None
    if ext in IMAGE_EXTS:
        return FileKind.IMAGE, None
    if ext in BINARY_EXTS:
        return FileKind.OPAQUE, None
    if ext in DATA_EXTS or ext in TEXT_EXTS:
        return FileKind.TEXT, detect_language_from_path(str(path))

    language = detect_language_from_path(str(path))
    if language in {"markdown", "markdown_inline"}:
        return FileKind.MARKDOWN, "markdown"
    if language:
        return FileKind.CODE, language

    known = FILENAME_LANGUAGES.get(path.name)
    if known:
        return FileKind.CODE, known
    # An unknown extension is not evidence of binary content. Default to
    # TEXT; the reader downgrades to OPAQUE if the bytes do not decode.
    return FileKind.TEXT, None
