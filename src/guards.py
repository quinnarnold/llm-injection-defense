# src/guards.py
"""
LLM Guard scanner configuration for input and output filtering.

Replaces custom output_filter/ modules with LLM Guard's battle-tested scanners.
Custom preprocessor and injection classifier are layered on top as defense-in-depth.
"""
from llm_guard import scan_prompt, scan_output
from llm_guard.input_scanners import (
    PromptInjection,
    TokenLimit,
    InvisibleText,
)
from llm_guard.output_scanners import (
    BanCode,
    Regex as OutputRegex,
)
from llm_guard.model import Model

_pi_model = Model(
    path="protectai/deberta-v3-base-prompt-injection-v2",
    pipeline_kwargs={"device": "cpu", "truncation": True, "max_length": 512},
)
_ban_model = Model(
    path="huggingface/CodeBERTa-small-v1",
    pipeline_kwargs={"device": "cpu", "truncation": True, "max_length": 512},
)

_input_scanners = [
    InvisibleText(),
    PromptInjection(model=_pi_model, threshold=0.85),
    TokenLimit(limit=2000),
]

_EMAIL_PATTERN = r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"
_PHONE_PATTERN = r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"

_ban_code_scanner = BanCode(model=_ban_model, threshold=0.9)
_regex_scanner = OutputRegex(patterns=[_EMAIL_PATTERN, _PHONE_PATTERN], is_blocked=True)

_CODE_KEYWORDS = ("def ", "function ", "import ", "var ", "class ", "const ")


def _looks_like_code(text: str) -> bool:
    return "`" in text or any(kw in text for kw in _CODE_KEYWORDS)


def scan_input(text: str) -> tuple[str, bool, dict]:
    """Scan user input with LLM Guard. Returns (sanitized, is_safe, scores)."""
    try:
        sanitized, results_valid, results_score = scan_prompt(_input_scanners, text)
        is_safe = all(results_valid.values())
        return sanitized, is_safe, results_score
    except Exception:
        return text, False, {}


def scan_response(prompt: str, response: str) -> tuple[str, bool, dict]:
    """Scan model output with LLM Guard. Returns (sanitized, is_safe, scores)."""
    try:
        output_scanners = (
            [_ban_code_scanner, _regex_scanner]
            if _looks_like_code(response)
            else [_regex_scanner]
        )
        sanitized, results_valid, results_score = scan_output(
            output_scanners, prompt, response
        )
        is_safe = all(results_valid.values())
        if not is_safe:
            sanitized += "\n\nFor contact information, visit info.bryant.edu"
        return sanitized, is_safe, results_score
    except Exception:
        return response, False, {}
