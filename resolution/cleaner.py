"""Team-name normalization utilities for entity resolution.

Provides deterministic, accent-insensitive, punctuation-free team name
normalization so that the same club is matched identically across
heterogeneous sportsbook feeds. Vocabulary is loaded from
`resolution/lexicon.json`.
"""

import re
import unicodedata
from typing import Optional, Set

from resolution.lexicon import get_lexicon

_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _normalize_tokens(name: Optional[str], drop_cosmetic: bool) -> list:
    """Shared normalization pipeline returning the kept token list."""
    if name is None:
        return []

    try:
        text = str(name)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Team name could not be coerced to string: {exc}") from exc

    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )

    lowered = without_accents.lower()
    without_punctuation = _PUNCTUATION_PATTERN.sub(" ", lowered)
    collapsed = _WHITESPACE_PATTERN.sub(" ", without_punctuation).strip()

    if not collapsed:
        return []

    tokens = collapsed.split(" ")
    if drop_cosmetic:
        cosmetic = get_lexicon().cosmetic_noise_tokens
        tokens = [token for token in tokens if token not in cosmetic]
    return tokens

def _apply_aliases(tokens: list) -> list:
    """Expand known team aliases and token synonyms.

    A full-name alias match (e.g. "psg") replaces the entire token
    list with its canonical expansion; otherwise each token is checked
    against the single-token synonym map (e.g. "utd" -> "united").
    """
    lexicon = get_lexicon()
    joined = " ".join(tokens)
    canonical = lexicon.canonical_for_alias(joined)
    if canonical:
        return canonical.split(" ")
    synonyms = lexicon.token_synonyms
    return [synonyms.get(token, token) for token in tokens]

def normalize_team_name(name: Optional[str]) -> str:
    """Normalize a raw team name into a canonical comparison form.

    Cosmetic club noise tokens are removed; semantic category
    discriminators (youth/reserve/gender markers) are intentionally
    preserved. Known aliases and abbreviations are expanded to their
    canonical form so that e.g. "Man Utd" and "Manchester United"
    normalize to the same string, while "Arsenal U21" and "Arsenal"
    still never collide.
    """
    tokens = _normalize_tokens(name, drop_cosmetic=True)
    return " ".join(_apply_aliases(tokens))

def extract_category_markers(name: Optional[str]) -> Set[str]:
    """Extract semantic category markers present in a team name.

    Args:
        name: Raw team name.

    Returns:
        Set of category identifiers (e.g. ``{"youth"}``, ``{"reserve",
        "gender"}``). Empty when no discriminator tokens are present.
    """
    lexicon = get_lexicon()
    return {
        lexicon.category_for(token)
        for token in _normalize_tokens(name, drop_cosmetic=False)
        if lexicon.category_for(token) is not None
    }
