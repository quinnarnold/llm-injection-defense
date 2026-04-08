"""
Iterative multi-encoding decoder.

Decodes Base64, ROT13, reversed text, URL encoding, HTML entities, and hex
escapes. Runs up to max_depth iterations with change detection to handle
nested/stacked encodings.
"""
import base64
import codecs
import html
import re
from urllib.parse import unquote


def _try_base64(text: str) -> str:
    stripped = text.strip()
    if not re.match(r"^[A-Za-z0-9+/\n\r]+=*$", stripped):
        return text
    if len(stripped) < 8:
        return text
    try:
        decoded = base64.b64decode(stripped, validate=True).decode("utf-8")
        if decoded.isprintable() or "\n" in decoded:
            return decoded
    except Exception:
        pass
    return text


def _try_rot13(text: str) -> str:
    candidate = codecs.decode(text, "rot_13")
    common_words = {"the", "and", "you", "your", "were", "given", "before", "every",
                    "rule", "describe", "system", "prompt", "what", "how", "this"}
    original_hits = sum(1 for w in text.lower().split() if w in common_words)
    candidate_hits = sum(1 for w in candidate.lower().split() if w in common_words)
    if candidate_hits > original_hits:
        return candidate
    return text


def _try_reverse(text: str) -> str:
    candidate = text[::-1]
    common_words = {"the", "and", "you", "your", "were", "given", "before", "every",
                    "rule", "describe", "system", "prompt", "what", "how", "this",
                    "started", "conversation"}
    original_hits = sum(1 for w in text.lower().split() if w in common_words)
    candidate_hits = sum(1 for w in candidate.lower().split() if w in common_words)
    if candidate_hits > original_hits:
        return candidate
    return text


def _decode_url(text: str) -> str:
    return unquote(text)


def _decode_html_entities(text: str) -> str:
    return html.unescape(text)


def _decode_hex_escapes(text: str) -> str:
    def _replace(m: re.Match) -> str:
        try:
            return chr(int(m.group(1), 16))
        except ValueError:
            return m.group(0)
    return re.sub(r"\\x([0-9a-fA-F]{2})", _replace, text)


def iterative_decode(text: str, max_depth: int = 3) -> str:
    """Iteratively decode text through multiple encoding layers.

    Applies all known decoders in sequence, repeating up to max_depth times.
    Stops early if no decoder changes the text (fixed point reached).
    """
    for _ in range(max_depth):
        previous = text
        text = _decode_html_entities(text)
        text = _decode_url(text)
        text = _decode_hex_escapes(text)
        text = _try_base64(text)
        text = _try_rot13(text)
        text = _try_reverse(text)
        if text == previous:
            break
    return text
