"""
Orchestrates the full input preprocessing pipeline.

Applies normalization, decoding, and stripping bidirectionally to handle
both wrapped encodings (XML around Base64) and encoded wrappers (Base64
around XML).
"""
from src.preprocessor.normalizer import normalize_text
from src.preprocessor.decoder import iterative_decode
from src.preprocessor.stripper import strip_policy_formats, strip_special_tokens

DEFAULT_MAX_INPUT_LENGTH = 2000


def preprocess(text: str, max_length: int = DEFAULT_MAX_INPUT_LENGTH) -> str:
    """Run the full preprocessing pipeline.

    Order: normalize -> (decode -> strip) + (strip -> decode) -> pick shorter -> truncate.
    """
    if len(text) > max_length * 2:
        text = text[:max_length * 2]

    text = normalize_text(text)
    text = strip_special_tokens(text)

    # Bidirectional: decode-then-strip vs strip-then-decode.
    # We pick the shorter result because stripping removes attack content,
    # so the path that removed more suspicious material produces the shorter string.
    path_a = strip_policy_formats(iterative_decode(text))
    path_b = iterative_decode(strip_policy_formats(text))

    text = path_a if len(path_a) <= len(path_b) else path_b

    if len(text) > max_length:
        text = text[:max_length]

    return text.strip()
