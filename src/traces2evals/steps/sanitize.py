"""
Response sanitization — clean production artifacts from reference responses.

Removes or replaces:
1. Signed URLs (S3, GCS, Azure Blob) — expire and will 404
2. Inline base64 data — bloats eval files
3. Long hex/UUID sequences in URLs
4. API keys or tokens accidentally in responses

Appends temporal drift note to judge remarks when reference contains
data that may change over time.
"""

import re
from typing import Optional

from traces2evals.models.eval_item import EvalItem

# Patterns to sanitize
_SIGNED_URL_PATTERN = re.compile(
    r"https?://[^\s\"']+(?:X-Amz-Signature|x-amz-credential|"
    r"Signature=|sig=|token=|sv=\d{4})[^\s\"']*",
    re.IGNORECASE,
)

_S3_URL_PATTERN = re.compile(
    r"https?://[a-z0-9._-]+\.s3[.\-][a-z0-9-]+\.amazonaws\.com/[^\s\"']+",
    re.IGNORECASE,
)

_GCS_URL_PATTERN = re.compile(
    r"https?://storage\.googleapis\.com/[^\s\"']+",
    re.IGNORECASE,
)

_BASE64_PATTERN = re.compile(
    r"data:[a-zA-Z]+/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]{100,}",
)

_LONG_TOKEN_PATTERN = re.compile(
    r"(?:Bearer\s+|token[=:]\s*)[A-Za-z0-9._\-]{40,}",
    re.IGNORECASE,
)

TEMPORAL_DRIFT_NOTE = (
    "Note: The reference response contains data from the original session date. "
    "The agent's new response should demonstrate the same capability, structure, "
    "and depth — but may reference different or updated data. Do not penalize for "
    "different specific values, dates, or entity states."
)


def sanitize_response(text: str) -> tuple[str, bool]:
    """Sanitize a reference response. Returns (cleaned_text, was_modified)."""
    original = text

    text = _SIGNED_URL_PATTERN.sub("[Generated document link]", text)
    text = _S3_URL_PATTERN.sub("[Generated document link]", text)
    text = _GCS_URL_PATTERN.sub("[Generated document link]", text)
    text = _BASE64_PATTERN.sub("[Embedded content]", text)
    text = _LONG_TOKEN_PATTERN.sub("[Redacted token]", text)

    return text, text != original


def sanitize_eval_items(evals: list[EvalItem]) -> list[EvalItem]:
    """Sanitize all eval items in place. Appends temporal drift note where needed."""
    for item in evals:
        cleaned, was_modified = sanitize_response(item.reference_response)
        item.reference_response = cleaned

        if _contains_temporal_data(item.reference_response):
            if TEMPORAL_DRIFT_NOTE not in item.judge_remarks:
                item.judge_remarks = item.judge_remarks.rstrip() + " " + TEMPORAL_DRIFT_NOTE

    return evals


def _contains_temporal_data(text: str) -> bool:
    """Heuristic: does the response contain data that will likely change over time?"""
    temporal_signals = [
        re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),  # dates like 3/15/2025
        re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),  # ISO dates
        re.compile(r"\b(?:today|yesterday|last week|this month|currently)\b", re.IGNORECASE),
        re.compile(r"\$[\d,]+\.?\d*"),  # dollar amounts
        re.compile(r"\b\d+\s+(?:\w+\s+)?(?:tickets?|issues?|items?|records?)\b", re.IGNORECASE),
    ]
    return any(p.search(text) for p in temporal_signals)
