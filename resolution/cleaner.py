"""Self-contained text normalization and qualifier extraction utilities.

Provides deterministic, accent-insensitive, punctuation-free normalization
for team names and regex-driven extraction of semantic qualifiers (youth,
reserve, gender) so that the same club resolves identically across
heterogeneous sportsbook feeds. This module has zero external file or
lexicon dependencies; all semantic detection is purely algorithmic via
compiled regular expressions.
"""

import re
import unicodedata
from typing import Optional, Set

_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_PATTERN = re.compile(r"\s+")

# Qualifier extraction patterns (applied to already-normalized text, i.e.
# lowercase with punctuation collapsed to spaces).
_YOUTH_AGE_PATTERN = re.compile(r"\b(?:u|sub)\s*(\d{1,2})\b", re.UNICODE)
_YOUTH_GENERIC_PATTERN = re.compile(r"\byouth\b", re.UNICODE)
_RESERVE_PATTERN = re.compile(
    r"\b(?:reserves|reserve|res|ii|b)\b", re.UNICODE
)
_GENDER_FEMALE_PATTERN = re.compile(
    r"\b(?:women|femenil|femenino|fem|ladies|w)\b", re.UNICODE
)


def normalize_text(text: Optional[str]) -> str:
    """Normalize a raw string into a canonical comparison form.

    Pipeline: NFKD decomposition with combining-mark removal (strips
    diacritics/accents), lowercasing, punctuation replacement with
    whitespace, and whitespace collapsing.

    Args:
        text: Raw input string (e.g. a team name from a feed).

    Returns:
        The normalized string; an empty string for ``None`` or empty input.
    """
    if text is None:
        return ""
    try:
        raw = str(text)
    except (TypeError, ValueError):
        return ""

    decomposed = unicodedata.normalize("NFKD", raw)
    without_accents = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )
    lowered = without_accents.lower()
    without_punctuation = _PUNCTUATION_PATTERN.sub(" ", lowered)
    return _WHITESPACE_PATTERN.sub(" ", without_punctuation).strip()


def normalize_team_name(text: Optional[str]) -> str:
    """Backward-compatible alias for :func:`normalize_text`."""
    return normalize_text(text)


def extract_qualifiers(text: Optional[str]) -> Set[str]:
    """Extract semantic qualifier markers present in a raw string.

    Detection is purely algorithmic via compiled regular expressions:
      * Youth: age-grade forms such as ``u17``/``u 21``/``sub-20``/``sub21``
        yield ``youth:<age>``; the bare word ``youth`` yields ``youth``.
      * Reserve: isolated tokens such as ``b``, ``ii``, ``res``,
        ``reserve``, ``reserves`` yield ``reserve``.
      * Gender: female markers such as ``women``, ``w``, ``fem``,
        ``femenil``, ``femenino``, ``ladies`` yield ``gender:female``.

    Args:
        text: Raw input string (e.g. a team name from a feed).

    Returns:
        Set of normalized qualifier tags (e.g. ``{"youth:21"}``,
        ``{"reserve"}``, ``{"gender:female"}``). Empty when no qualifier
        is present.
    """
    normalized = normalize_text(text)
    if not normalized:
        return set()

    qualifiers: Set[str] = set()

    for age_match in _YOUTH_AGE_PATTERN.finditer(normalized):
        try:
            age = int(age_match.group(1))
        except (TypeError, ValueError):
            continue
        qualifiers.add(f"youth:{age}")
    if _YOUTH_GENERIC_PATTERN.search(normalized):
        qualifiers.add("youth")

    if _RESERVE_PATTERN.search(normalized):
        qualifiers.add("reserve")

    if _GENDER_FEMALE_PATTERN.search(normalized):
        qualifiers.add("gender:female")

    return qualifiers


def extract_category_markers(text: Optional[str]) -> Set[str]:
    """Backward-compatible alias for :func:`extract_qualifiers`."""
    return extract_qualifiers(text)


def generate_char_ngrams(text: Optional[str], n: int = 3) -> Set[str]:
    """Generate character-level n-grams from the normalized text.

    The input is normalized via :func:`normalize_text` and whitespace is
    removed before the sliding window is applied, so n-grams capture
    sub-word structure suitable for fuzzy similarity comparison.

    Args:
        text: Raw input string.
        n: N-gram window size; must be a positive integer.

    Returns:
        Set of character n-grams. For inputs shorter than ``n``, a set
        containing the whole compacted string is returned (when non-empty).

    Raises:
        ValueError: If ``n`` is not a positive integer.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer.")

    compact = normalize_text(text).replace(" ", "")
    if not compact:
        return set()
    if len(compact) <= n:
        return {compact}
    return {compact[index:index + n] for index in range(len(compact) - n + 1)}


def tokenize(text: Optional[str]) -> list:
    """Split normalized text into clean alphanumeric word tokens.

    Args:
        text: Raw input string.

    Returns:
        List of lowercase alphanumeric tokens; empty for ``None`` or
        empty input.
    """
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split(" ")
