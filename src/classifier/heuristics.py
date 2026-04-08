"""
Secondary pattern-based heuristic rules for injection detection.

Fallback when the RoBERTa classifier confidence is below the block threshold.
Targets structural patterns of injection: imperative overrides, role assumptions,
meta-queries about the system prompt.
"""
import re
from dataclasses import dataclass


@dataclass
class HeuristicResult:
    flagged: bool
    reason: str = ""


_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(
        r"\b(?:ignore|forget|disregard|override|bypass|skip)\b"
        r".*\b(?:all|previous|prior|above|earlier|original)\b"
        r".*\b(?:instructions?|rules?|guidelines?|constraints?|directives?|prompts?)\b",
        re.IGNORECASE,
    ), "imperative override pattern"),

    (re.compile(
        r"\b(?:you are now|act as|pretend (?:to be|you are)|behave as|roleplay as|"
        r"assume the (?:role|identity|persona))\b",
        re.IGNORECASE,
    ), "role assumption pattern"),

    (re.compile(
        r"\b(?:describe|output|reveal|show|display|print|repeat|list)\b"
        r".*\b(?:system prompt|instructions? (?:you were|that were) given|"
        r"rules? (?:you were|that were) given|your (?:full |complete )?(?:rules?|prompt|directives?))\b",
        re.IGNORECASE,
    ), "system prompt extraction pattern"),

    (re.compile(
        r"\b(?:extended|diagnostic|debug|admin|maintenance|developer) mode\b"
        r".*\b(?:enabled|activated|authorized)\b",
        re.IGNORECASE,
    ), "policy update language"),

    (re.compile(
        r"\b(?:hypothetical(?:ly)?|fictional|imaginary|thought experiment)\b"
        r".*\b(?:no (?:rules?|restrictions?|limitations?|filters?)|anything)\b",
        re.IGNORECASE,
    ), "jailbreak framing pattern"),
]


def check_heuristics(text: str) -> HeuristicResult:
    """Check text against heuristic injection patterns."""
    if not text.strip():
        return HeuristicResult(flagged=False)

    for pattern, reason in _PATTERNS:
        if pattern.search(text):
            return HeuristicResult(flagged=True, reason=reason)

    return HeuristicResult(flagged=False)
