"""
Strips structured policy injection formats and model special tokens from input.

Targets XML and JSON structured injection formats and ChatML special tokens
that could manipulate Qwen3's chat template.
"""
import re


_XML_POLICY_PATTERN = re.compile(
    r"<(?:system[-_]?\w*|source|directive|permission|policy|update|config)\b[^>]*>"
    r".*?"
    r"</(?:system[-_]?\w*|source|directive|permission|policy|update|config)>",
    re.DOTALL | re.IGNORECASE,
)

_JSON_POLICY_KEYS = re.compile(
    r'"(?:system_update|directive|permission|source|policy_override|config)"',
    re.IGNORECASE,
)

_JSON_POLICY_PATTERN = re.compile(
    r'\{(?:[^{}]|\{[^{}]*\})*"(?:system_update|directive|permission|source|policy_override|config)'
    r'"(?:[^{}]|\{[^{}]*\})*\}',
    re.DOTALL | re.IGNORECASE,
)

_SPECIAL_TOKENS = re.compile(
    r"<\|(?:im_start|im_end|endoftext|pad|end|en|zh|ja|ko|de|fr)\|>",
    re.IGNORECASE,
)

_THINKING_TAGS = re.compile(
    r"</?think>",
    re.IGNORECASE,
)


def strip_policy_formats(text: str) -> str:
    """Remove XML and JSON structured policy injection formats from text."""
    text = _XML_POLICY_PATTERN.sub("", text)
    text = _JSON_POLICY_PATTERN.sub("", text)
    return text.strip()


def strip_special_tokens(text: str) -> str:
    """Remove ChatML, Qwen3, and model-specific special tokens from text."""
    text = _SPECIAL_TOKENS.sub("", text)
    text = _THINKING_TAGS.sub("", text)
    text = re.sub(r"^\s*(system|user|assistant)\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
