"""
Unicode normalization with confusable-character mapping.

Uses NFKC normalization plus a subset of Unicode TR39 skeleton algorithm
to neutralize cross-script homoglyph attacks (e.g., Cyrillic 'a' -> Latin 'a').
"""
import unicodedata


_CONFUSABLES: dict[str, str] = {
    "\u0430": "a",  # Cyrillic
    "\u0435": "e",
    "\u043e": "o",
    "\u0440": "p",
    "\u0441": "c",
    "\u0443": "y",
    "\u0455": "s",
    "\u0456": "i",
    "\u04bb": "h",
    "\u0501": "d",
    "\u051b": "q",
    "\u051d": "w",
    "\u0410": "A",
    "\u0412": "B",
    "\u0415": "E",
    "\u041a": "K",
    "\u041c": "M",
    "\u041d": "H",
    "\u041e": "O",
    "\u0420": "P",
    "\u0421": "C",
    "\u0422": "T",
    "\u0425": "X",
}


def _apply_confusable_mapping(text: str) -> str:
    return "".join(_CONFUSABLES.get(ch, ch) for ch in text)


def normalize_text(text: str) -> str:
    """Apply NFKC normalization then confusable-character mapping."""
    text = unicodedata.normalize("NFKC", text)
    text = _apply_confusable_mapping(text)
    return text
