"""Utility layer for odds normalization.

Defensive payload inspection/coercion helpers, target schema field-mapping
resolution, and raw-to-decimal odds conversion. Depends exclusively on the
Python standard library.
"""

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

class SafePayloadUtils:
    """Defensive lookup and coercion helpers for arbitrary JSON payloads."""

    _CAMEL_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")
    _TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
    _MAX_TRAVERSAL_DEPTH = 12
    _BRACKETED_SEGMENT = re.compile(r"\([^()]*\)|\[[^\[\]]*\]|\{[^{}]*\}")
    # Language-agnostic structural delimiters (punctuation only, never words).
    _SEPARATOR_RUN = re.compile(r"([\s\-:;|/,\\@\u2013\u2014\u00d7\u2022\u00b7]+)")
    _EDGE_PUNCTUATION = re.compile(r"^\W+|\W+$")
    _NUMERIC_WORD = re.compile(r"^\d+$")
    _MAX_CONNECTOR_LENGTH = 3

    @classmethod
    def tokenize_key(cls, key: Any) -> List[str]:
        """Normalize any key (camelCase, snake_case, kebab-case) into tokens."""
        raw = str(key)
        snake = cls._CAMEL_BOUNDARY.sub(r"\1_\2", raw).lower()
        return [t for t in cls._TOKEN_SPLIT.split(snake) if t]

    @classmethod
    def key_matches(cls, key: Any, term: str) -> bool:
        """Token-subset match between a raw key and a lexicon term."""
        key_tokens = set(cls.tokenize_key(key))
        term_tokens = cls.tokenize_key(term)
        if not key_tokens or not term_tokens:
            return False
        return set(term_tokens).issubset(key_tokens)

    @classmethod
    def find_first(cls, node: Any, terms: Set[str], max_depth: int = _MAX_TRAVERSAL_DEPTH) -> Any:
        """Depth-limited BFS for the first value stored under a matching key."""
        if max_depth < 0:
            return None
        queue: List[Tuple[Any, int]] = [(node, 0)]
        while queue:
            current, depth = queue.pop(0)
            if isinstance(current, dict):
                for k, v in current.items():
                    for term in terms:
                        if cls.key_matches(k, term) and v is not None:
                            return v
                if depth + 1 <= max_depth:
                    for v in current.values():
                        queue.append((v, depth + 1))
            elif isinstance(current, list) and depth + 1 <= max_depth:
                for item in current:
                    queue.append((item, depth + 1))
        return None

    @classmethod
    def find_first_list(cls, node: Any, terms: Set[str], max_depth: int = _MAX_TRAVERSAL_DEPTH) -> Optional[List[Any]]:
        """Depth-limited BFS for the first list stored under a matching key."""
        if max_depth < 0:
            return None
        queue: List[Tuple[Any, int]] = [(node, 0)]
        while queue:
            current, depth = queue.pop(0)
            if isinstance(current, dict):
                for k, v in current.items():
                    if isinstance(v, list):
                        for term in terms:
                            if cls.key_matches(k, term):
                                return v
                if depth + 1 <= max_depth:
                    for v in current.values():
                        queue.append((v, depth + 1))
            elif isinstance(current, list) and depth + 1 <= max_depth:
                for item in current:
                    queue.append((item, depth + 1))
        return None

    @classmethod
    def has_key(cls, node: Any, term: str) -> bool:
        """Check whether a dict contains a key matching the given term."""
        if not isinstance(node, dict):
            return False
        return any(cls.key_matches(k, term) for k in node.keys())

    @staticmethod
    def coerce_float(value: Any) -> Optional[float]:
        """Safely coerce a value to float, returning None on failure."""
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip().replace(",", ".")
            if not re.match(r"^[+-]?\d+(\.\d+)?$", cleaned):
                return None
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

    @staticmethod
    def coerce_str(value: Any) -> Optional[str]:
        """Safely coerce a value to a non-empty stripped string."""
        if value is None or isinstance(value, (dict, list, bool)):
            return None
        text = str(value).strip()
        return text if text else None

    @classmethod
    def decompose_fixture_label(
        cls,
        label: Any,
        separator_terms: Optional[Set[str]] = None,
        inverting_terms: Optional[Set[str]] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Decompose a raw event label into ``(home_team, away_team)``.

        Splits on structural invariants only: punctuation delimiters, very
        short alphabetic connectors, and optional lexicon-supplied connector
        terms. Venue-first ordering is the default (first entity is home);
        separators flagged as inverting (venue-last syntax such as ``"A @ B"``
        or a term from ``inverting_terms``) swap the assignment so the second
        entity becomes the home team. Bracketed annotations and trailing
        numeric runs (dates, scores) are stripped before splitting.

        Args:
            label: Raw event label (match header/title) of any type.
            separator_terms: Optional lexicon-derived order-preserving
                connector terms (e.g. equivalents of "vs").
            inverting_terms: Optional lexicon-derived venue-last connector
                terms (e.g. equivalents of "at").

        Returns:
            ``(home_team, away_team)``; ``(None, None)`` when no confident
            two-entity decomposition exists.
        """
        try:
            text = SafePayloadUtils.coerce_str(label)
            if not text:
                return None, None
            cleaned = cls._BRACKETED_SEGMENT.sub(" ", text)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if not cleaned:
                return None, None
            items = cls._strip_numeric_tail(cls._tokenize_matchup(cleaned))
            words = [payload for kind, payload in items if kind == "word"]
            if len(words) < 2:
                return None, None
            candidates = cls._collect_separator_candidates(items, separator_terms, inverting_terms)
            home, away, _ = cls._split_matchup_items(words, candidates)
            return home, away
        except Exception:
            return None, None

    @classmethod
    def infer_competitors_from_outcomes(
        cls,
        outcome_labels: Any,
        non_competitor_terms: Optional[Set[str]] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Infer ``(home, away)`` competitor names from head-to-head outcomes.

        Non-competitor selections (draws, ties, neutral placeholders) are
        filtered via lexicon-supplied ``non_competitor_terms`` using
        token-subset matching, so localized terminology never enters source
        code. Structural invariants back the filtering when terms are absent:
        a two-outcome market has no draw selection, and in a three-outcome
        market the unique shortest label is treated as the non-competitor.
        Remaining participants are returned in listed order (home first,
        matching the 1X2 listing convention).

        Args:
            outcome_labels: Raw outcome labels of one primary market.
            non_competitor_terms: Optional lexicon-derived draw/tie/placeholder
                vocabulary.

        Returns:
            ``(home, away)`` or ``(None, None)`` when inference is impossible.
        """
        try:
            if outcome_labels is None or isinstance(outcome_labels, (str, bytes, dict)):
                return None, None
            try:
                raw_labels = list(outcome_labels)
            except TypeError:
                return None, None
            distinct: List[str] = []
            seen: Set[str] = set()
            for raw in raw_labels:
                text = SafePayloadUtils.coerce_str(raw)
                if not text:
                    continue
                fingerprint = text.lower()
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                distinct.append(text)
            if len(distinct) < 2:
                return None, None
            terms = {
                normalized
                for normalized in (
                    SafePayloadUtils.coerce_str(term) for term in (non_competitor_terms or set())
                )
                if normalized
            }
            participants = [
                text for text in distinct
                if not any(cls.key_matches(text, term.lower()) for term in terms)
            ]
            if len(participants) >= 2:
                return participants[0], participants[1]
            if len(distinct) == 2:
                return distinct[0], distinct[1]
            if len(distinct) == 3:
                token_counts = [len(cls.tokenize_key(text)) for text in distinct]
                minimum = min(token_counts)
                if token_counts.count(minimum) == 1:
                    remaining = [text for i, text in enumerate(distinct) if token_counts[i] != minimum]
                    return remaining[0], remaining[1]
            return None, None
        except Exception:
            return None, None

    @classmethod
    def _tokenize_matchup(cls, text: str) -> List[Tuple[str, str]]:
        """Split cleaned text into ``("word", text)`` / ``("sep", text)`` items."""
        items: List[Tuple[str, str]] = []
        for part in cls._SEPARATOR_RUN.split(text):
            if not part:
                continue
            if cls._SEPARATOR_RUN.fullmatch(part):
                if part.strip():
                    items.append(("sep", part))
            else:
                items.append(("word", part.strip()))
        return items

    @classmethod
    def _strip_numeric_tail(cls, items: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Drop trailing numeric runs (scores/dates) delimited by punctuation."""
        trimmed = list(items)
        while True:
            while trimmed and trimmed[-1][0] == "sep":
                trimmed.pop()
            if not trimmed or trimmed[-1][0] != "word" or not cls._NUMERIC_WORD.match(trimmed[-1][1]):
                break
            preceding_is_separator = len(trimmed) >= 2 and trimmed[-2][0] == "sep"
            numeric_run = 0
            for kind, payload in reversed(trimmed):
                if kind == "sep":
                    continue
                if cls._NUMERIC_WORD.match(payload):
                    numeric_run += 1
                else:
                    break
            if preceding_is_separator or numeric_run >= 2:
                trimmed.pop()
                continue
            break
        return trimmed

    @classmethod
    def _collect_separator_candidates(
        cls,
        items: List[Tuple[str, str]],
        separator_terms: Optional[Set[str]],
        inverting_terms: Optional[Set[str]],
    ) -> List[Tuple[int, int, int, bool]]:
        """Collect ``(rank, word_index, skip, inverts)`` split candidates.

        Rank 0: punctuation delimiters. Rank 1: lexicon connector terms.
        Rank 2: very short alphabetic connector words (structural heuristic).
        """
        separator_norm = {
            term.strip().lower()
            for term in (separator_terms or set())
            if SafePayloadUtils.coerce_str(term)
        }
        inverting_norm = {
            term.strip().lower()
            for term in (inverting_terms or set())
            if SafePayloadUtils.coerce_str(term)
        }
        candidates: List[Tuple[int, int, int, bool]] = []
        words = [payload for kind, payload in items if kind == "word"]
        word_index = 0
        for kind, payload in items:
            if kind == "sep":
                candidates.append((0, word_index, 0, "@" in payload))
                continue
            core = cls._EDGE_PUNCTUATION.sub("", payload).lower()
            if core and core.isalpha() and len(core) <= cls._MAX_CONNECTOR_LENGTH:
                candidates.append((2, word_index, 1, core in inverting_norm))
            word_index += 1
        for term in separator_norm | inverting_norm:
            term_tokens = cls.tokenize_key(term)
            if not term_tokens:
                continue
            length = len(term_tokens)
            for start in range(len(words) - length + 1):
                if all(
                    cls.tokenize_key(words[start + offset]) == [term_tokens[offset]]
                    for offset in range(length)
                ):
                    candidates.append((1, start, length, term in inverting_norm))
        return sorted(set(candidates))

    @classmethod
    def _split_matchup_items(
        cls,
        words: List[str],
        candidates: List[Tuple[int, int, int, bool]],
    ) -> Tuple[Optional[str], Optional[str], bool]:
        """Recursively select the best split, returning ``(home, away, inverts)``.

        Preference order: strongest rank, sides free of other separators,
        balanced side sizes, earliest position. Sides still containing
        separators are recursively refined; a refined pair (produced by an
        explicit matchup connector) is preferred over its unrefined sibling.
        """
        if not candidates or len(words) < 2:
            return None, None, False
        for rank in sorted({candidate[0] for candidate in candidates}):
            valid = []
            for candidate in candidates:
                if candidate[0] != rank:
                    continue
                _, idx, skip, _ = candidate
                if idx >= 1 and idx + skip <= len(words) - 1:
                    valid.append(candidate)
            if not valid:
                continue

            def contamination(candidate: Tuple[int, int, int, bool]) -> int:
                _, idx, skip, _ = candidate
                return sum(
                    1 for other in candidates
                    if other != candidate
                    and (other[1] + other[2] <= idx or other[1] >= idx + skip)
                )

            chosen = min(
                valid,
                key=lambda c: (
                    contamination(c),
                    abs(len(words[:c[1]]) - len(words[c[1] + c[2]:])),
                    c[1],
                ),
            )
            _, idx, skip, inverts = chosen
            left_words = words[:idx]
            right_words = words[idx + skip:]
            left_candidates = [c for c in candidates if c != chosen and c[1] + c[2] <= idx]
            right_candidates = [c for c in candidates if c != chosen and c[1] >= idx + skip]
            if left_candidates:
                left_pair = cls._split_matchup_items(left_words, left_candidates)
                if left_pair[0] and left_pair[1]:
                    return left_pair
            if right_candidates:
                right_pair = cls._split_matchup_items(right_words, right_candidates)
                if right_pair[0] and right_pair[1]:
                    return right_pair
            if inverts:
                return " ".join(right_words), " ".join(left_words), True
            return " ".join(left_words), " ".join(right_words), False
        return None, None, False

class FieldMappingResolver:
    """Resolves raw payload keys from a target's stored ``field_mappings``.

    Canonical keys (e.g. ``home_team``, ``odds_value``, ``match_container``)
    are mapped to the raw key names discovered for a specific target during
    schema discovery. An empty resolver means "no schema knowledge" and every
    consumer must degrade to lexicon-based semantic matching.
    """

    ALLOWED_KEYS: Tuple[str, ...] = (
        "event_id", "home_team", "away_team", "league", "sport", "start_time",
        "event_label", "match_container", "market_container", "outcome_container",
        "market_label", "outcome_label", "outcome_id", "odds_value", "line",
    )

    def __init__(self, mappings: Optional[Dict[str, Any]] = None) -> None:
        self._mappings: Dict[str, str] = {}
        self.update(mappings)

    def update(self, mappings: Optional[Dict[str, Any]]) -> None:
        """Replace the current mappings with a sanitized copy of ``mappings``."""
        cleaned: Dict[str, str] = {}
        if isinstance(mappings, dict):
            for raw_key, raw_value in mappings.items():
                key = SafePayloadUtils.coerce_str(raw_key)
                value = SafePayloadUtils.coerce_str(raw_value)
                if key and value and key.strip().lower() in self.ALLOWED_KEYS:
                    cleaned[key.strip().lower()] = value.strip()
        self._mappings = cleaned

    def get(self, canonical_key: str) -> Optional[str]:
        """Return the raw key mapped to ``canonical_key``, or None."""
        return self._mappings.get(canonical_key)

    def as_dict(self) -> Dict[str, str]:
        return dict(self._mappings)

    def __bool__(self) -> bool:
        return bool(self._mappings)

    def __len__(self) -> int:
        return len(self._mappings)

class OddsConverter:
    """Heuristic detection and conversion of raw odds values to decimal format."""

    DECIMAL_MIN = 1.01
    DECIMAL_MAX = 1000.0
    AMERICAN_ABS_MIN = 100
    AMERICAN_ABS_MAX = 100000

    @classmethod
    def to_decimal(cls, raw: Any) -> Tuple[Optional[float], str]:
        """Convert a raw odds value to decimal odds.

        Returns a ``(decimal_value, detected_format)`` tuple where the format is
        one of ``decimal``, ``american``, ``fractional`` or ``unknown``.
        """
        if isinstance(raw, bool) or raw is None:
            return None, "unknown"

        if isinstance(raw, (int, float)):
            return cls._from_number(float(raw))

        if isinstance(raw, str):
            text = raw.strip()
            fractional = cls._from_fraction_string(text)
            if fractional[0] is not None:
                return fractional
            american = cls._from_american_string(text)
            if american[0] is not None:
                return american
            numeric = SafePayloadUtils.coerce_float(text)
            if numeric is not None:
                return cls._from_number(numeric)

        return None, "unknown"

    @classmethod
    def _from_number(cls, value: float) -> Tuple[Optional[float], str]:
        if value == math.floor(value) and cls.AMERICAN_ABS_MIN <= abs(value) <= cls.AMERICAN_ABS_MAX:
            return cls._american_to_decimal(value), "american"
        if cls.DECIMAL_MIN <= value <= cls.DECIMAL_MAX:
            return round(value, 4), "decimal"
        return None, "unknown"

    @staticmethod
    def _american_to_decimal(value: float) -> float:
        if value > 0:
            return round(1.0 + value / 100.0, 4)
        return round(1.0 + 100.0 / abs(value), 4)

    @classmethod
    def _from_american_string(cls, text: str) -> Tuple[Optional[float], str]:
        if re.match(r"^[+-]\d+$", text):
            value = float(text)
            if cls.AMERICAN_ABS_MIN <= abs(value) <= cls.AMERICAN_ABS_MAX:
                return cls._american_to_decimal(value), "american"
        return None, "unknown"

    @classmethod
    def _from_fraction_string(cls, text: str) -> Tuple[Optional[float], str]:
        match = re.match(r"^(\d+)\s*/\s*(\d+)$", text)
        if match:
            numerator = float(match.group(1))
            denominator = float(match.group(2))
            if denominator > 0:
                return round(1.0 + numerator / denominator, 4), "fractional"
        return None, "unknown"
