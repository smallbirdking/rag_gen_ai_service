from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class SafetyResult:
    blocked: bool
    reason: str | None = None


PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"reveal\s+(the\s+)?(hidden\s+)?(system|developer)\s+prompt",
    r"disregard\s+(the\s+)?(system|developer)",
    r"泄露.*(系统|开发者).*提示",
    r"忽略.*(之前|以上).*指令",
    r"绕过.*安全",
]

PII_PATTERNS = [
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("phone", re.compile(r"(?<!\d)(?:\+?\d{1,3}[-\s]?)?(?:\d[-\s]?){8,14}\d(?!\d)")),
    ("cn_id", re.compile(r"\b\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[0-9Xx]\b")),
]


def detect_prompt_injection(text: str) -> SafetyResult:
    lowered = text.lower()
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return SafetyResult(blocked=True, reason="prompt_injection_detected")
    return SafetyResult(blocked=False)


def redact_pii(text: str) -> tuple[str, int]:
    count = 0
    redacted = text
    for label, pattern in PII_PATTERNS:
        redacted, n = pattern.subn(f"[{label.upper()}_REDACTED]", redacted)
        count += n
    return redacted, count
