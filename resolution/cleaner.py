"""Team-name normalization utilities for entity resolution.

Provides deterministic, accent-insensitive, punctuation-free team name
normalization so that the same club is matched identically across
heterogeneous sportsbook feeds.
"""

import re
import unicodedata
from typing import Optional

# Tokens that carry no disambiguating signal for club identity.
# NOTE: candidate for migration into `sportsbook_lexicon.json` if the
# lexicon is ever made a permitted dependency of this package.
_CLUB_NOISE_TOKENS = frozenset(
    {
        "fc",
        "cf",
        "sc",
        "afc",
        "ac",
        "as",
        "ss",
        "sss",
        "sk",
        "fk",
        "bk",
        "if",
        "sv",
        "tsv",
        "vfl",
        "vfb",
        "sad",
        "u19",
        "u20",
        "u21",
        "u23",
        "ii",
        "b",
        "reserves",
        "reserve",
        "women",
        "w",
    }
)

_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_team_name(name: Optional[str]) -> str:
    """Normalize a raw team name into a canonical comparison form.

    The transformation pipeline is: safe string coercion, Unicode NFKD
    decomposition with diacritic stripping, lowercasing, punctuation
    removal, whitespace collapsing, and removal of club noise tokens.

    Args:
        name: The raw team name as provided by the source feed.

    Returns:
        The normalized team name. Returns an empty string when the input
        is ``None`` or contains no usable characters.
    """
    if name is None:
        return ""

    try:
        text = str(name)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Team name could not be coerced to string: {exc}") from exc

    # Strip accents/diacritics via NFKD decomposition.
    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )

    lowered = without_accents.lower()
    without_punctuation = _PUNCTUATION_PATTERN.sub(" ", lowered)
    collapsed = _WHITESPACE_PATTERN.sub(" ", without_punctuation).strip()

    if not collapsed:
        return ""

    kept_tokens = [
        token for token in collapsed.split(" ") if token not in _CLUB_NOISE_TOKENS
    ]
    return " ".join(kept_tokens)
