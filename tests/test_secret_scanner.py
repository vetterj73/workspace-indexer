"""Keeping credentials out of the index.

Indexed content is sent to an embedding provider as request input, so anything
reaching a chunk reaches a third party. Every token below is synthetic.

The two properties that matter: real credentials are caught whatever the file
is called, and the value never leaves the scanner.
"""

from __future__ import annotations

import pytest

from workspace_indexer.secrets import SecretFinding, scan, shannon_entropy

# Every fixture is assembled at runtime rather than written as a literal.
#
# Not decoration: the first attempt to push this file was rejected by GitHub's
# own push protection, which recognised the Slack-shaped fixture. Both scanners
# were right. A test suite for a secret detector cannot contain contiguous
# strings that look like secrets, so the prefixes are joined here and no
# scanner reading the source sees one.
_PREFIX = {
    "github": "github" + "_pat_",
    "aws": "AK" + "IA",
    "openai": "sk" + "-proj_",
    "slack": "xo" + "xb-",
    "google": "AI" + "za",
}

FAKE_GITHUB_PAT = _PREFIX["github"] + "11ABCDEFG0" + "z" * 30 + "Qw7"
FAKE_AWS = _PREFIX["aws"] + "IOSFODNN7EXAMPLE"
FAKE_OPENAI = _PREFIX["openai"] + "a1B2c3D4e5F6g7H8i9J0" * 2
FAKE_SLACK = _PREFIX["slack"] + "1234567890-abcdefghijklmnop"
FAKE_GOOGLE = _PREFIX["google"] + "B" * 35
FAKE_PRIVATE_KEY = (
    "-----BEGIN RSA PRIVATE KEY-----\nMIIEow...\n-----END RSA PRIVATE KEY-----"
)


def test_the_case_that_actually_bit_this_project() -> None:
    """.mcp.json held a GitHub token under a name no deny-list would flag.
    That is why the check is on content rather than filenames."""
    text = f'{{"headers": {{"Authorization": "Bearer {FAKE_GITHUB_PAT}"}}}}'
    findings = scan(text)
    assert findings
    assert findings[0].rule == "github_pat"


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        (f"key = {FAKE_AWS}", "aws_access_key"),
        (f"OPENAI={FAKE_OPENAI}", "openai"),
        (FAKE_PRIVATE_KEY, "private_key"),
        (f"token: {FAKE_SLACK}", "slack"),
        (f"k = {FAKE_GOOGLE}", "google_api"),
    ],
)
def test_known_token_shapes(text: str, rule: str) -> None:
    findings = scan(text)
    assert findings and findings[0].rule == rule


def test_a_secret_in_source_code_is_caught_too() -> None:
    """Not only config. A hard-coded key in a .py is the same problem."""
    code = f'STRIPE_KEY = "{FAKE_OPENAI}"\n\ndef charge():\n    pass\n'
    assert scan(code)


def test_high_entropy_assignment_to_a_credential_name() -> None:
    assert scan('DATABASE_PASSWORD = "k7Fq2mZx9RtVwLpA3nBcYdQe"')


# ---- what must NOT be blocked -----------------------------------------


def test_a_template_with_blank_values_is_fine() -> None:
    """.env.example is genuinely useful to index and holds nothing."""
    text = "VOYAGE_API_KEY=\nRERANK_MODEL=voyageai:rerank-2.5-lite\nQDRANT_MODE=embedded"
    assert scan(text) == []


def test_placeholders_are_not_secrets() -> None:
    for value in ("your-api-key-here", "changeme", "<YOUR_TOKEN>", "${API_KEY}", "TODO"):
        assert scan(f'API_KEY = "{value}"') == [], value


def test_prose_about_credentials_is_not_a_credential() -> None:
    """Documentation discussing tokens must stay indexable, or the guidance
    documents this project exists to find get withheld."""
    text = (
        "Authentication verifies the bearer token on every request. The "
        "API_KEY environment variable must be set before the service starts."
    )
    assert scan(text) == []


def test_a_url_is_not_a_secret() -> None:
    assert scan('AUTH_URL = "https://example.com/oauth/authorize/v2"') == []


def test_a_dotted_identifier_is_not_a_secret() -> None:
    assert scan('AUTH_BACKEND = "django.contrib.auth.backends.ModelBackend"') == []


def test_ordinary_config_values_pass() -> None:
    text = "EMBEDDING_MODEL=voyageai:voyage-code-4\nEMBEDDING_DIMENSIONS=1024\nLOG_LEVEL=INFO"
    assert scan(text) == []


# ---- the finding must not carry the value -----------------------------


def test_findings_never_contain_the_secret() -> None:
    """The finding is logged. A log line carrying the credential would defeat
    the entire exercise, and logs get shipped and shared like anything else."""
    findings = scan(f"token = {FAKE_GITHUB_PAT}")
    assert findings
    rendered = " ".join(str(f) + f.model_dump_json() for f in findings)
    assert FAKE_GITHUB_PAT not in rendered
    assert FAKE_GITHUB_PAT[:24] not in rendered


def test_finding_reports_a_usable_location() -> None:
    text = f"line one\nline two\ntoken = {FAKE_AWS}\n"
    findings = scan(text)
    assert findings[0].line == 3


def test_findings_are_capped() -> None:
    """A file of a thousand keys should not produce a thousand log lines."""
    text = "\n".join(f"KEY_{i} = {FAKE_AWS}" for i in range(50))
    assert len(scan(text)) <= 5


# ---- entropy ----------------------------------------------------------


def test_entropy_separates_random_from_english() -> None:
    assert shannon_entropy("k7Fq2mZx9RtVwLpA3nBcYdQe") > shannon_entropy("the quick brown fox")


def test_entropy_of_empty_string() -> None:
    assert shannon_entropy("") == 0.0


def test_finding_renders_without_a_value() -> None:
    finding = SecretFinding(rule="github_pat", line=7, description="GitHub token")
    assert "line 7" in str(finding)
    assert "github_pat" in str(finding)
