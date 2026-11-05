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
