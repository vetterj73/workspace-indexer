"""Refuse to embed files that contain credentials.

Indexed content is sent to an embedding provider as request input, so anything
reaching a chunk reaches a third party. `.gitignore` covers the conventional
cases and misses three real ones: non-repo folders have no ignore file at all,
committed secrets are tracked by definition, and the case that actually bit
this project was a file named `.mcp.json` holding a GitHub token — a name no
deny-list would flag.

So the check is on content, not on filenames, and it protects source files as
much as configuration.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from workspace_indexer.secrets.secret_finding import SecretFinding

# Token shapes that are unambiguous: these prefixes are issued by a provider
# and do not occur by accident.
_SIGNATURES: list[tuple[str, re.Pattern[str], str]] = [
    ("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), "GitHub personal access token"),
    ("github_classic", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"), "GitHub token"),
    ("openai", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"), "OpenAI-style API key"),
    ("anthropic", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"), "Anthropic API key"),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "AWS access key id"),
    ("slack", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"), "Slack token"),
    ("google_api", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "Google API key"),
    ("voyage", re.compile(r"\bpa-[A-Za-z0-9_-]{30,}"), "Voyage API key"),
    ("private_key", re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"), "private key block"),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\."), "JWT"),
]

# An assignment to a credential-shaped name. The value still has to look random
# before we act on it, or every `API_KEY=` line in a template trips the check.
#
# Two shapes this missed, both found against a real repository:
#
#   "password": "..."     A quoted key. The pattern ran key straight into
#                         `[:=]`, so JSON -- the single commonest way a
#                         credential is written down -- never matched at all.
#   PASSWORD="a~b"        `~` was absent from the value class. Azure service
#                         principal passwords routinely contain it, and the
#                         value simply failed to match rather than failing the
#                         entropy test.
#
# The value class is now "printable, not whitespace or a quote", which is what
# a generated credential actually looks like. Entropy remains the thing that
# decides, so widening this cannot by itself produce a false positive.
_ASSIGNMENT = re.compile(
    r"""(?ix)
    (?P<key>[A-Za-z0-9_.-]*
        (?:api[_-]?key|secret|token|password|passwd|credential|auth|
           connection[_-]?string|conn[_-]?str|sas|pwd)
        [A-Za-z0-9_.-]*)
    ["']?              # a quoted key: "password": "..."
    \s* [:=] \s*
    ["']?
    (?P<value>[^\s"'`,;)\]}]{16,})
    """
)

# Values below this Shannon entropy read as prose or a placeholder rather than
# a generated credential.
_ENTROPY_THRESHOLD = 3.6

# Obvious non-secrets that would otherwise clear the entropy bar.
_PLACEHOLDERS = frozenset(
    {
        "changeme",
        "your-api-key-here",
        "xxxxxxxxxxxxxxxx",
        "insert-key-here",
        "replace-me",
        "todo",
        "none",
        "null",
        "example",
        "placeholder",
    }
)


def shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    total = len(value)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


# Code referring to a credential rather than the credential itself, as in
# `api_key=settings.qdrant_api_key`. Found by this scanner withholding one of
# our own source files.
#
# Identifier *validity* is the wrong test -- a generated secret like
# k7Fq2mZx9RtVwLpA3nBcYdQe is a valid identifier too. What separates them is
# naming convention: references are dotted, or consistently snake_case or
# SCREAMING_CASE. A generated value interleaves case and digits with no word
# structure at all.
_ATTRIBUTE = re.compile(r"\A[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\Z")
_CONVENTIONAL_NAME = re.compile(r"\A(?:[a-z_][a-z0-9_]*|[A-Z_][A-Z0-9_]*|[a-zA-Z][a-zA-Z]*)\Z")

# An expression or a type rather than a literal. Widening the value class to
# catch `~` started catching all of these, and withholding a whole file over a
# variable reference is a silent loss of content -- the worse error here.
#
#   builder.Configuration["x"]          brackets
#   AuthMode::TrustedLocal              a Rust or C++ path
#   Option<AuthMeta>                    a generic type
#   appInsights.properties.Connection   a dotted path
_EXPRESSION = re.compile(r"[()\[\]{}<>]|::|\A[A-Za-z_][A-Za-z0-9_]*\.")

# Word separators inside an identifier. `docker-hub-credentials` is the *name*
# of a credential, not the credential; stripping these before the all-letters
# test recognises it as a name rather than a generated value.
_WORD_SEPARATORS = str.maketrans("", "", "-_")


def _looks_generated(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in _PLACEHOLDERS or lowered.startswith("<") or lowered.startswith("${"):
        return False
    # A path, a URL or a dotted module name is structured, not random.
    if "/" in value or value.count(".") > 2:
        return False
    bare = value.strip()
    if _ATTRIBUTE.match(bare) or _CONVENTIONAL_NAME.match(bare):
        return False
    if _EXPRESSION.search(bare):
        return False
    # A generated credential mixes letters with digits or symbols. All-letters
    # is an identifier -- `sqlAdminPassword`, `docker-hub-credentials` --
    # however long and however cased, and both clear the entropy bar easily.
    if bare.translate(_WORD_SEPARATORS).isalpha():
        return False
    return shannon_entropy(value) >= _ENTROPY_THRESHOLD


def scan(text: str, *, max_findings: int = 5) -> list[SecretFinding]:
    """Findings describe what was seen; they never carry the value."""
    findings: list[SecretFinding] = []

    for number, line in enumerate(text.splitlines(), start=1):
        for rule, pattern, description in _SIGNATURES:
            if pattern.search(line):
                findings.append(SecretFinding(rule=rule, line=number, description=description))
                break
        else:
            match = _ASSIGNMENT.search(line)
            if match and _looks_generated(match.group("value")):
                findings.append(
                    SecretFinding(
                        rule="high_entropy_assignment",
                        line=number,
                        description=f"high-entropy value assigned to {match.group('key')}",
                    )
                )
        if len(findings) >= max_findings:
            break

    return findings
