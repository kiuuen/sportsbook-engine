"""Canonical odds normalization engine for heterogeneous sportsbook payloads.

Translates raw, schema-agnostic sportsbook API responses into a uniform
canonical entity structure (events -> markets -> outcomes with decimal odds).
Market semantics are resolved dynamically through the lexicon-driven
:class:`MarketClassifier` instead of hardcoded provider translation tables,
and provider-specific layouts are handled by pluggable
:class:`ProviderAdapter` implementations registered into :class:`OddsNormalizer`.
"""

import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from heuristic_classifier import LexiconManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("odds_normalizer")


class CanonicalMarketType:
    """Namespace of canonical market group identifiers."""

    MONEYLINE = "moneyline"
    TOTALS = "totals"
    SPREADS = "spreads"
    UNKNOWN = "unknown"

    ALL: Tuple[str, ...] = (MONEYLINE, TOTALS, SPREADS, UNKNOWN)


@dataclass
class CanonicalOutcome:
    """A single normalized selection with decimal odds."""

    label: str
    odds_decimal: Optional[float]
    odds_raw: Any = None
    odds_format: str = "unknown"
    line: Optional[float] = None
    provider_outcome_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalMarket:
    """A normalized market belonging to a canonical group."""

    canonical_type: str
    raw_label: str
    market_id: Optional[str] = None
    classification_confidence: float = 0.0
    outcomes: List[CanonicalOutcome] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


@dataclass
class CanonicalEvent:
    """A normalized event with its normalized markets."""

    event_id: Optional[str]
    home_team: Optional[str]
    away_team: Optional[str]
    start_time: Optional[str]
    raw_label: Optional[str]
    source_provider: str = "unknown"
    league: Optional[str] = None
    sport: Optional[str] = None
    markets: List[CanonicalMarket] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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


class MarketClassifier:
    """Lexicon-driven semantic classifier mapping raw market labels to canonical groups.

    Canonical group vocabularies are built dynamically from terms present in
    ``sportsbook_lexicon.json``; only generic token patterns are additive.
    """

    _SEED_CANONICAL_TERMS: Dict[str, Set[str]] = {
        CanonicalMarketType.MONEYLINE: {"moneyline", "draw_no_bet", "outright", "to_qualify", "futures"},
        CanonicalMarketType.TOTALS: {"totals", "over_under", "over_price", "under_price"},
        CanonicalMarketType.SPREADS: {
            "spreads", "spread", "spread_value", "handicap", "handicap_value",
            "asian_handicap", "european_handicap", "line", "lines",
        },
    }

    _TOKEN_HINT_RULES: List[Tuple[re.Pattern, str]] = [
        (re.compile(r"\b(over|under|total|totals)\b", re.IGNORECASE), CanonicalMarketType.TOTALS),
        (re.compile(r"\b(handicap|handicaps|spread|spreads)\b", re.IGNORECASE), CanonicalMarketType.SPREADS),
        (re.compile(r"\b(moneyline|ml|winner|win|1x2|match|draw)\b", re.IGNORECASE), CanonicalMarketType.MONEYLINE),
    ]

    def __init__(self, lexicon_mgr: Optional[LexiconManager]) -> None:
        self.lexicon_mgr = lexicon_mgr
        self.canonical_terms: Dict[str, Set[str]] = {
            canonical: set() for canonical in CanonicalMarketType.ALL if canonical != CanonicalMarketType.UNKNOWN
        }
        self._build_vocabularies()

    def _build_vocabularies(self) -> None:
        """Intersect seed terms with the live lexicon so unknown lexicons degrade gracefully."""
        betting_terms: Set[str] = set()
        if self.lexicon_mgr is not None:
            betting_terms = self.lexicon_mgr.positive_keywords.get("betting_markets_and_odds", set())

        for canonical, seed_terms in self._SEED_CANONICAL_TERMS.items():
            if betting_terms:
                self.canonical_terms[canonical] = {t for t in seed_terms if t in betting_terms}
            else:
                logger.warning("[MarketClassifier] Lexicon betting terms unavailable; using seed vocabulary only.")
                self.canonical_terms[canonical] = set(seed_terms)

    def classify(self, market_label: str) -> Tuple[str, float]:
        """Classify a raw market label into a canonical group with a confidence score."""
        if not market_label:
            return CanonicalMarketType.UNKNOWN, 0.0

        tokens = SafePayloadUtils.tokenize_key(market_label)
        token_set = set(tokens)
        label_lower = market_label.lower()

        best_group = CanonicalMarketType.UNKNOWN
        best_hits = 0
        for canonical, terms in self.canonical_terms.items():
            hits = sum(1 for term in terms if term in token_set or term in label_lower)
            if hits > best_hits:
                best_group, best_hits = canonical, hits

        if best_hits > 0:
            return best_group, min(1.0, best_hits / 3.0)

        for pattern, canonical in self._TOKEN_HINT_RULES:
            if pattern.search(label_lower):
                return canonical, 0.4

        return CanonicalMarketType.UNKNOWN, 0.0


@dataclass
class NormalizationContext:
    """Shared dependencies injected into every provider adapter."""

    lexicon_mgr: Optional[LexiconManager]
    classifier: MarketClassifier
    converter: OddsConverter
    utils: SafePayloadUtils
    mappings: FieldMappingResolver = field(default_factory=FieldMappingResolver)


class ProviderAdapter:
    """Base class for pluggable provider layout handlers.

    Subclasses override :meth:`can_handle` and :meth:`normalize` to translate a
    specific payload family into canonical events. The base implementation is a
    safe no-op so misbehaving adapters can never crash the pipeline.
    """

    name: str = "base_adapter"
    priority: int = 100

    EVENT_ID_TERMS: Set[str] = {"event_id", "match_id", "fixture_id", "game_id"}
    GENERIC_ID_TERMS: Set[str] = {"id"}
    HOME_TERMS: Set[str] = {"home_team", "home_competitor", "home", "home_name"}
    AWAY_TERMS: Set[str] = {"away_team", "away_competitor", "away", "away_name"}
    START_TERMS: Set[str] = {"commence_time", "start_time", "kickoff", "scheduled_at", "event_date"}
    EVENT_LABEL_TERMS: Set[str] = {"name", "event", "match", "title"}
    MARKET_LABEL_TERMS: Set[str] = {"bet_offer_type", "market_name", "market_type", "market", "sub_market"}
    OUTCOME_LABEL_TERMS: Set[str] = {"selection_name", "selection", "runner", "name", "label", "outcome"}
    ODDS_TERMS: Tuple[str, ...] = (
        "decimal_odds", "odds", "price", "over_price", "under_price", "american_odds", "fractional_odds",
    )
    LINE_TERMS: Set[str] = {"line", "spread", "spread_value", "handicap", "handicap_value"}
    OUTCOME_ID_TERMS: Set[str] = {"outcome_id", "selection_id", "runner_id", "id"}
    MARKET_LIST_TERMS: Set[str] = {"markets", "market", "betoffers", "betoffer"}
    OUTCOME_LIST_TERMS: Set[str] = {"outcomes", "outcome", "selections", "selection", "runners", "runner"}
    EVENT_LIST_TERMS: Set[str] = {"events", "fixtures", "matches"}

    LEAGUE_TERMS: Set[str] = {"league", "league_id", "competition", "tournament"}
    SPORT_TERMS: Set[str] = {"sport", "sport_id", "sport_key", "sport_name"}

    def __init__(self, ctx: NormalizationContext) -> None:
        self.ctx = ctx

    def can_handle(self, payload: Any) -> bool:
        """Return True when this adapter recognizes the payload layout."""
        return False

    def normalize(self, payload: Any) -> List[CanonicalEvent]:
        """Translate a recognized payload into canonical events."""
        logger.debug(f"[{self.name}] normalize() invoked but not implemented for this adapter.")
        return []

    # ------------------------------------------------------------------
    # Shared defensive extraction helpers available to all adapters.
    # ------------------------------------------------------------------
    def _extract_event_identity(self, node: Any) -> Dict[str, Optional[str]]:
        utils = self.ctx.utils
        specific_id = utils.coerce_str(self._mapped_lookup(node, "event_id", self.EVENT_ID_TERMS))
        generic_id = utils.coerce_str(utils.find_first(node, self.GENERIC_ID_TERMS))
        return {
            "event_id": specific_id or generic_id,
            "home_team": utils.coerce_str(self._mapped_lookup(node, "home_team", self.HOME_TERMS)),
            "away_team": utils.coerce_str(self._mapped_lookup(node, "away_team", self.AWAY_TERMS)),
            "start_time": utils.coerce_str(self._mapped_lookup(node, "start_time", self.START_TERMS)),
            "league": utils.coerce_str(self._mapped_lookup(node, "league", self.LEAGUE_TERMS)),
            "sport": utils.coerce_str(self._mapped_lookup(node, "sport", self.SPORT_TERMS)),
            "raw_label": utils.coerce_str(self._mapped_lookup(node, "event_label", self.EVENT_LABEL_TERMS)),
        }

    def _mapped_lookup(self, node: Any, canonical_key: str, fallback_terms: Set[str]) -> Any:
        """Resolve a canonical key via stored field mappings, else semantic fallback."""
        if not isinstance(node, dict):
            return None
        mapped_key = self.ctx.mappings.get(canonical_key)
        if mapped_key:
            value = self._direct_key_lookup(node, mapped_key)
            if value is not None:
                return self._unwrap_named(value)
        return self.ctx.utils.find_first(node, fallback_terms)

    @staticmethod
    def _direct_key_lookup(node: Dict[str, Any], raw_key: str) -> Any:
        """Exact-match then token-subset lookup for a raw mapped key at this level."""
        if raw_key in node:
            return node[raw_key]
        for key, value in node.items():
            if SafePayloadUtils.key_matches(key, raw_key):
                return value
        return None

    @staticmethod
    def _unwrap_named(value: Any) -> Any:
        """Unwrap nested ``{"name": ...}``-style objects produced by mapped keys."""
        if isinstance(value, dict):
            for name_key in ("name", "label", "title"):
                if name_key in value and value[name_key] is not None:
                    return value[name_key]
        return value

    def _find_container(self, node: Any, canonical_key: str, terms: Set[str]) -> Optional[List[Any]]:
        """Resolve a list container via field mappings, else lexicon-based BFS."""
        mapped_key = self.ctx.mappings.get(canonical_key)
        if mapped_key and isinstance(node, dict):
            value = self._direct_key_lookup(node, mapped_key)
            if isinstance(value, list):
                return value
        return self.ctx.utils.find_first_list(node, terms)

    def _odds_candidate_terms(self) -> Tuple[str, ...]:
        """Odds key candidates, prioritizing the schema-mapped key when present."""
        mapped = self.ctx.mappings.get("odds_value")
        if mapped:
            return (mapped,) + self.ODDS_TERMS
        return self.ODDS_TERMS

    def _lookup_by_term(self, node: Any, term: str) -> Any:
        """Direct mapped-key lookup first, then depth-limited semantic search."""
        if isinstance(node, dict):
            direct = self._direct_key_lookup(node, term)
            if direct is not None:
                return direct
        return self.ctx.utils.find_first(node, {term})

    def _looks_like_event(self, node: Any) -> bool:
        if not isinstance(node, dict):
            return False
        utils = self.ctx.utils
        has_specific_id = any(utils.has_key(node, t) for t in self.EVENT_ID_TERMS)
        has_generic_id = utils.has_key(node, "id")
        has_context = any(
            utils.has_key(node, t) for t in (self.HOME_TERMS | self.AWAY_TERMS | self.START_TERMS)
        )
        return has_context and (has_specific_id or has_generic_id)

    def _extract_market_label(self, node: Any) -> Optional[str]:
        """Derive a market label only from nodes that also carry outcome signals."""
        if not isinstance(node, dict):
            return None
        has_outcome_signals = any(
            self.ctx.utils.has_key(node, t) for t in self.OUTCOME_LIST_TERMS
        ) or self._find_odds_raw(node) is not None
        if not has_outcome_signals:
            return None
        return self.ctx.utils.coerce_str(
            self._mapped_lookup(node, "market_label", self.MARKET_LABEL_TERMS)
        )

    def _find_odds_raw(self, node: Any) -> Any:
        if not isinstance(node, dict):
            return None
        for term in self._odds_candidate_terms():
            if self.ctx.utils.has_key(node, term):
                value = self._lookup_by_term(node, term)
                decimal, fmt = self.ctx.converter.to_decimal(value)
                if decimal is not None:
                    return value
        return None

    def _build_outcome(self, node: Any) -> Optional[CanonicalOutcome]:
        if not isinstance(node, dict):
            return None
        utils = self.ctx.utils

        odds_raw: Any = None
        odds_decimal: Optional[float] = None
        odds_format = "unknown"
        for term in self._odds_candidate_terms():
            if utils.has_key(node, term):
                candidate = self._lookup_by_term(node, term)
                decimal, fmt = self.ctx.converter.to_decimal(candidate)
                if decimal is not None:
                    odds_raw, odds_decimal, odds_format = candidate, decimal, fmt
                    break
        if odds_decimal is None:
            return None

        line_value = utils.coerce_float(self._mapped_lookup(node, "line", self.LINE_TERMS))
        outcome_label = (
            utils.coerce_str(self._mapped_lookup(node, "outcome_label", self.OUTCOME_LABEL_TERMS))
            or "unknown_selection"
        )
        outcome_id = utils.coerce_str(self._mapped_lookup(node, "outcome_id", self.OUTCOME_ID_TERMS))

        return CanonicalOutcome(
            label=outcome_label,
            odds_decimal=odds_decimal,
            odds_raw=odds_raw,
            odds_format=odds_format,
            line=line_value,
            provider_outcome_id=outcome_id,
        )

    def _build_market(self, node: Any, fallback_label: str = "unknown_market") -> Optional[CanonicalMarket]:
        if not isinstance(node, dict):
            return None
        utils = self.ctx.utils
        raw_label = (
            utils.coerce_str(self._mapped_lookup(node, "market_label", self.MARKET_LABEL_TERMS))
            or fallback_label
        )
        canonical_type, confidence = self.ctx.classifier.classify(raw_label)
        market = CanonicalMarket(
            canonical_type=canonical_type,
            raw_label=raw_label,
            market_id=utils.coerce_str(self._mapped_lookup(node, "market_label", {"market_id", "id"})),
            classification_confidence=confidence,
        )
        outcome_list = self._find_container(node, "outcome_container", self.OUTCOME_LIST_TERMS) or []
        for outcome_node in outcome_list:
            outcome = self._build_outcome(outcome_node)
            if outcome is not None:
                market.outcomes.append(outcome)
        if not market.outcomes:
            direct_outcome = self._build_outcome(node)
            if direct_outcome is not None:
                market.outcomes.append(direct_outcome)
        return market if market.outcomes else None


class KambiStyleAdapter(ProviderAdapter):
    """Adapter for betoffer/outcome layouts (e.g. Kambi-style provider families)."""

    name = "kambi_style"
    priority = 10

    def can_handle(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        betoffers = self._find_container(payload, "market_container", self.MARKET_LIST_TERMS)
        if not betoffers:
            return False
        for betoffer in betoffers[:5]:
            if self.ctx.utils.find_first_list(betoffer, self.OUTCOME_LIST_TERMS):
                return True
        return False

    def normalize(self, payload: Any) -> List[CanonicalEvent]:
        utils = self.ctx.utils
        events_nodes = self._find_container(payload, "match_container", self.EVENT_LIST_TERMS)
        if not events_nodes:
            events_nodes = [payload]

        canonical_events: List[CanonicalEvent] = []
        for event_node in events_nodes:
            if not isinstance(event_node, dict):
                continue
            identity = self._extract_event_identity(event_node)
            event = CanonicalEvent(
                event_id=identity["event_id"],
                home_team=identity["home_team"],
                away_team=identity["away_team"],
                start_time=identity["start_time"],
                raw_label=identity["raw_label"],
                source_provider=self.name,
                league=identity.get("league"),
                sport=identity.get("sport"),
            )
            betoffers = self._find_container(event_node, "market_container", self.MARKET_LIST_TERMS) or []
            for betoffer in betoffers:
                market = self._build_market(betoffer)
                if market is not None:
                    event.markets.append(market)
            if event.markets or event.event_id:
                canonical_events.append(event)
        return canonical_events


class GenericMarketsAdapter(ProviderAdapter):
    """Adapter for payloads exposing a markets/outcomes collection at the root."""

    name = "generic_markets"
    priority = 20

    def can_handle(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        if self._find_container(payload, "market_container", self.MARKET_LIST_TERMS):
            return True
        return bool(self._find_container(payload, "outcome_container", self.OUTCOME_LIST_TERMS))

    def normalize(self, payload: Any) -> List[CanonicalEvent]:
        identity = self._extract_event_identity(payload)
        event = CanonicalEvent(
            event_id=identity["event_id"],
            home_team=identity["home_team"],
            away_team=identity["away_team"],
            start_time=identity["start_time"],
            raw_label=identity["raw_label"],
            source_provider=self.name,
            league=identity.get("league"),
            sport=identity.get("sport"),
        )
        markets = self._find_container(payload, "market_container", self.MARKET_LIST_TERMS) or []
        for market_node in markets:
            market = self._build_market(market_node)
            if market is not None:
                event.markets.append(market)

        if not event.markets:
            market = self._build_market(payload)
            if market is not None:
                event.markets.append(market)

        return [event] if event.markets else []


class GenericEventsAdapter(ProviderAdapter):
    """Adapter for payloads exposing an events/fixtures/matches collection."""

    name = "generic_events"
    priority = 30

    def can_handle(self, payload: Any) -> bool:
        if isinstance(payload, list):
            return any(isinstance(item, dict) for item in payload)
        if isinstance(payload, dict):
            events = self._find_container(payload, "match_container", self.EVENT_LIST_TERMS)
            return bool(events)
        return False

    def normalize(self, payload: Any) -> List[CanonicalEvent]:
        event_nodes: List[Any] = []
        if isinstance(payload, list):
            event_nodes = list(payload)
        else:
            found = self._find_container(payload, "match_container", self.EVENT_LIST_TERMS)
            if found:
                event_nodes = list(found)

        canonical_events: List[CanonicalEvent] = []
        for event_node in event_nodes:
            if not isinstance(event_node, dict):
                continue
            identity = self._extract_event_identity(event_node)
            event = CanonicalEvent(
                event_id=identity["event_id"],
                home_team=identity["home_team"],
                away_team=identity["away_team"],
                start_time=identity["start_time"],
                raw_label=identity["raw_label"],
                source_provider=self.name,
                league=identity.get("league"),
                sport=identity.get("sport"),
            )
            market_lists = self._find_container(event_node, "market_container", self.MARKET_LIST_TERMS) or []
            for market_node in market_lists:
                market = self._build_market(market_node)
                if market is not None:
                    event.markets.append(market)
            if event.markets or event.event_id:
                canonical_events.append(event)
        return canonical_events

class RelationalEntitiesAdapter(ProviderAdapter):
    """Adapter for relational/columnar payloads with sibling entity collections.

    Targets layouts such as ``{"events": [...], "markets": [...], "odds": [...]}``
    where events, markets, and odds/price rows live side by side at the root and
    reference each other through identifier pointers instead of nesting.
    """

    name = "relational_entities"
    priority = 15

    ODDS_LIST_TERMS: Set[str] = {"odds", "prices", "odds_list", "oddslist", "odd_ids", "oddids"}
    MARKET_POINTER_TERMS: Tuple[str, ...] = ("market_ids", "marketids", "markets")
    ODD_POINTER_TERMS: Tuple[str, ...] = (
        "odd_ids", "oddids", "odds", "outcome_ids", "outcomeids",
        "selection_ids", "selectionids", "selections", "runners",
    )
    EVENT_REF_TERMS: Set[str] = {
        "event_id", "eventid", "match_id", "matchid", "fixture_id", "fixtureid",
        "parent_id", "parentid", "parent_event_id",
    }

    def can_handle(self, payload: Any) -> bool:
        """Claim payloads with root-level sibling event and market/odds collections."""
        if not isinstance(payload, dict):
            return False
        utils = self.ctx.utils
        has_events = any(utils.has_key(payload, term) for term in self.EVENT_LIST_TERMS)
        if not has_events:
            return False
        has_markets = any(utils.has_key(payload, term) for term in self.MARKET_LIST_TERMS)
        has_outcomes = any(utils.has_key(payload, term) for term in self.OUTCOME_LIST_TERMS)
        return has_markets or has_outcomes

    def normalize(self, payload: Any) -> List[CanonicalEvent]:
        """Reconstruct canonical events from relational sibling collections."""
        if not isinstance(payload, dict):
            return []
        try:
            outcomes_index = self._build_outcomes_index(payload)
            markets_index, market_event_refs = self._build_markets_index(payload, outcomes_index)
            return self._stitch_events(payload, markets_index, market_event_refs)
        except Exception as err:
            logger.warning(f"[{self.name}] Failed to normalize relational payload: {err}")
            return []

    def _build_outcomes_index(self, payload: Dict[str, Any]) -> Dict[str, CanonicalOutcome]:
        """Index outcomes from the odds/prices collection keyed by stringified identifier."""
        index: Dict[str, CanonicalOutcome] = {}
        odds_nodes = self._find_container(payload, "outcome_container", self.ODDS_LIST_TERMS)
        if not odds_nodes:
            odds_nodes = self._find_container(payload, "outcome_container", self.OUTCOME_LIST_TERMS)
        for node in odds_nodes or []:
            if not isinstance(node, dict):
                continue
            try:
                outcome = self._build_outcome(node)
            except Exception as err:
                logger.warning(f"[{self.name}] Failed to build outcome from odds node: {err}")
                continue
            if outcome is None:
                continue
            outcome_id = outcome.provider_outcome_id or self.ctx.utils.coerce_str(
                self._lookup_by_term(node, "id")
            )
            if outcome_id is None:
                continue
            index[outcome_id] = outcome
        return index

    def _build_markets_index(
        self, payload: Dict[str, Any], outcomes_index: Dict[str, CanonicalOutcome]
    ) -> Tuple[Dict[str, CanonicalMarket], Dict[str, str]]:
        """Index markets keyed by stringified identifier, resolving outcomes via ``outcomes_index``.

        Returns ``(markets_index, market_event_refs)`` where ``market_event_refs``
        maps each market key (real or synthetic) to its stripped-string event
        foreign key when one is present. Markets are only discarded when they
        resolve zero outcomes after both pointer and nested-row lookups.
        """
        index: Dict[str, CanonicalMarket] = {}
        event_refs: Dict[str, str] = {}
        market_nodes = self._find_container(payload, "market_container", self.MARKET_LIST_TERMS) or []
        synthetic_counter = 0
        for node in market_nodes:
            if not isinstance(node, dict):
                continue
            try:
                raw_label = (
                    self.ctx.utils.coerce_str(
                        self._mapped_lookup(node, "market_label", self.MARKET_LABEL_TERMS)
                    )
                    or "unknown_market"
                )
                canonical_type, confidence = self.ctx.classifier.classify(raw_label)
                market = CanonicalMarket(
                    canonical_type=canonical_type,
                    raw_label=raw_label,
                    market_id=self.ctx.utils.coerce_str(
                        self._mapped_lookup(node, "market_label", {"market_id", "id"})
                    ),
                    classification_confidence=confidence,
                )
                for outcome in self._resolve_indexed_outcomes(node, outcomes_index):
                    market.outcomes.append(outcome)
            except Exception as err:
                logger.warning(f"[{self.name}] Failed to build market from node: {err}")
                continue
            if not market.outcomes:
                continue
            if market.market_id is not None:
                market_key = market.market_id
            else:
                synthetic_counter += 1
                market_key = f"anon_market_{synthetic_counter}"
            index[market_key] = market
            event_ref = self.ctx.utils.coerce_str(
                self._mapped_lookup(node, "event_id", self.EVENT_REF_TERMS)
            )
            if event_ref is not None:
                event_refs[market_key] = event_ref
        return index, event_refs

    def _resolve_indexed_outcomes(
        self, node: Dict[str, Any], outcomes_index: Dict[str, CanonicalOutcome]
    ) -> List[CanonicalOutcome]:
        """Resolve child outcomes via pointer arrays/IDs, then nested rows.

        Pointer keys are matched against ``ODD_POINTER_TERMS`` (covering
        ``odd_ids``/``oddids``/``odds``/``selections`` and camelCase variants via
        token matching). All references are normalized to stripped strings before
        lookup so integer and string identifiers compare seamlessly. Nested
        outcome containers are consulted only when no pointer resolved.
        """
        resolved: List[CanonicalOutcome] = []
        seen_ids: Set[str] = set()

        def _resolve_ref(raw_ref: Any) -> None:
            ref_id = SafePayloadUtils.coerce_str(raw_ref)
            if ref_id is None or ref_id in seen_ids:
                return
            outcome = outcomes_index.get(ref_id)
            if outcome is not None:
                seen_ids.add(ref_id)
                resolved.append(outcome)

        for term in self.ODD_POINTER_TERMS:
            for key, value in node.items():
                if not SafePayloadUtils.key_matches(key, term):
                    continue
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            outcome = self._build_outcome(item)
                            if outcome is None:
                                continue
                            dedupe_key = outcome.provider_outcome_id or (
                                f"{outcome.label}|{outcome.odds_decimal}"
                            )
                            if dedupe_key in seen_ids:
                                continue
                            seen_ids.add(dedupe_key)
                            resolved.append(outcome)
                        else:
                            _resolve_ref(item)
                else:
                    # Scalar pointer (e.g. ``"oddIds": 12345``).
                    _resolve_ref(value)
            if resolved:
                return resolved

        if not resolved:
            nested = self._find_container(node, "outcome_container", self.OUTCOME_LIST_TERMS) or []
            for nested_node in nested:
                outcome = self._build_outcome(nested_node)
                if outcome is not None:
                    resolved.append(outcome)
        return resolved

    def _stitch_events(
        self,
        payload: Dict[str, Any],
        markets_index: Dict[str, CanonicalMarket],
        market_event_refs: Dict[str, str],
    ) -> List[CanonicalEvent]:
        """Stitch canonical events to indexed markets via top-down and bottom-up links.

        Top-down: ID lists (or scalars) on each event row whose keys match
        ``MARKET_POINTER_TERMS`` are resolved against ``markets_index``.
        Bottom-up: any indexed market whose event foreign key equals the event's
        ``event_id`` is attached. All identifier comparisons use stripped strings.
        """
        events: List[CanonicalEvent] = []
        if not isinstance(payload, dict):
            return events

        try:
            event_nodes = self._find_container(payload, "match_container", self.EVENT_LIST_TERMS) or []
        except Exception as err:
            logger.warning(f"[{self.name}] Failed to locate events collection: {err}")
            return events

        for event_node in event_nodes:
            if not isinstance(event_node, dict):
                continue
            try:
                identity = self._extract_event_identity(event_node)
                event = CanonicalEvent(
                    event_id=identity["event_id"],
                    home_team=identity["home_team"],
                    away_team=identity["away_team"],
                    start_time=identity["start_time"],
                    raw_label=identity["raw_label"],
                    source_provider=self.name,
                    league=identity.get("league"),
                    sport=identity.get("sport"),
                )
                attached_ids: Set[str] = set()

                # Top-down linking: market ID pointers on the event row.
                for term in self.MARKET_POINTER_TERMS:
                    for key, value in event_node.items():
                        if not SafePayloadUtils.key_matches(key, term):
                            continue
                        candidates = value if isinstance(value, list) else [value]
                        for item in candidates:
                            if isinstance(item, (dict, list)):
                                continue
                            ref_id = SafePayloadUtils.coerce_str(item)
                            if ref_id is None or ref_id in attached_ids:
                                continue
                            market = markets_index.get(ref_id)
                            if market is not None:
                                attached_ids.add(ref_id)
                                event.markets.append(market)

                # Bottom-up linking: markets whose event FK matches this event id.
                event_id = SafePayloadUtils.coerce_str(event.event_id)
                if event_id is not None:
                    for market_key, market in markets_index.items():
                        if market_key in attached_ids:
                            continue
                        if market_event_refs.get(market_key) == event_id:
                            attached_ids.add(market_key)
                            event.markets.append(market)

                events.append(event)
            except Exception as err:
                logger.warning(f"[{self.name}] Failed to stitch event node: {err}")
                continue
        return events


class DeepTraversalAdapter(ProviderAdapter):
    """Fallback adapter that structurally infers events and outcomes from any nesting.

    Used when no specialized adapter claims the payload. It walks the document,
    treating dicts with identity + team/time context as events and dicts with a
    convertible odds value as outcomes, grouping outcomes under the nearest
    enclosing event and market label context.
    """

    name = "deep_traversal_fallback"
    priority = 999

    def __init__(self, ctx: NormalizationContext) -> None:
        super().__init__(ctx)
        self._synthetic_event_counter = 0

    def can_handle(self, payload: Any) -> bool:
        return isinstance(payload, (dict, list))

    def normalize(self, payload: Any) -> List[CanonicalEvent]:
        self._synthetic_event_counter = 0
        event_index: Dict[str, CanonicalEvent] = {}
        ordered_events: List[CanonicalEvent] = []
        self._walk(payload, None, None, event_index, ordered_events, depth=0)
        return ordered_events

    def _walk(
        self,
        node: Any,
        event_ctx: Optional[CanonicalEvent],
        market_label: Optional[str],
        event_index: Dict[str, CanonicalEvent],
        ordered_events: List[CanonicalEvent],
        depth: int,
    ) -> None:
        if depth > 25:
            return
        if isinstance(node, dict):
            if self._looks_like_event(node):
                event_ctx = self._get_or_create_event(node, event_index, ordered_events)
            derived_label = self._extract_market_label(node)
            if derived_label:
                market_label = derived_label
            if self._find_odds_raw(node) is not None:
                self._attach_outcome(node, event_ctx, market_label, event_index, ordered_events)
            for child in node.values():
                self._walk(child, event_ctx, market_label, event_index, ordered_events, depth + 1)
        elif isinstance(node, list):
            for item in node:
                self._walk(item, event_ctx, market_label, event_index, ordered_events, depth + 1)

    def _get_or_create_event(
        self,
        node: Dict[str, Any],
        event_index: Dict[str, CanonicalEvent],
        ordered_events: List[CanonicalEvent],
    ) -> CanonicalEvent:
        identity = self._extract_event_identity(node)
        key = identity["event_id"] or f"{identity['home_team']}|{identity['away_team']}"
        if not key or key == "None|None":
            self._synthetic_event_counter += 1
            key = f"synthetic_event_{self._synthetic_event_counter}"
        existing = event_index.get(key)
        if existing is not None:
            return existing
        event = CanonicalEvent(
            event_id=identity["event_id"],
            home_team=identity["home_team"],
            away_team=identity["away_team"],
            start_time=identity["start_time"],
            raw_label=identity["raw_label"],
            source_provider=self.name,
            league=identity.get("league"),
            sport=identity.get("sport"),
        )
        event_index[key] = event
        ordered_events.append(event)
        return event

    def _attach_outcome(
        self,
        node: Dict[str, Any],
        event_ctx: Optional[CanonicalEvent],
        market_label: Optional[str],
        event_index: Dict[str, CanonicalEvent],
        ordered_events: List[CanonicalEvent],
    ) -> None:
        outcome = self._build_outcome(node)
        if outcome is None:
            return
        if event_ctx is None:
            event_ctx = self._get_or_create_event({}, event_index, ordered_events)
        resolved_label = market_label or "unknown_market"
        market = next((m for m in event_ctx.markets if m.raw_label == resolved_label), None)
        if market is None:
            canonical_type, confidence = self.ctx.classifier.classify(resolved_label)
            market = CanonicalMarket(
                canonical_type=canonical_type,
                raw_label=resolved_label,
                classification_confidence=confidence,
            )
            event_ctx.markets.append(market)
        market.outcomes.append(outcome)


class OddsNormalizer:
    """Facade orchestrating adapter selection and canonical normalization."""

    def __init__(
        self,
        lexicon_path: str = "sportsbook_lexicon.json",
        field_mappings: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            self.lexicon_mgr: Optional[LexiconManager] = LexiconManager(lexicon_path)
        except Exception as err:
            logger.error(f"[OddsNormalizer] Failed to load lexicon '{lexicon_path}': {err}")
            raise RuntimeError(f"OddsNormalizer requires a valid lexicon file: {err}") from err

        self.ctx = NormalizationContext(
            lexicon_mgr=self.lexicon_mgr,
            classifier=MarketClassifier(self.lexicon_mgr),
            converter=OddsConverter(),
            utils=SafePayloadUtils(),
        )
        if field_mappings is not None:
            self.set_field_mappings(field_mappings)
        self._adapters: List[ProviderAdapter] = []
        self._register_default_adapters()

    def _register_default_adapters(self) -> None:
        for adapter_cls in (
            KambiStyleAdapter,
            RelationalEntitiesAdapter,
            GenericMarketsAdapter,
            GenericEventsAdapter,
            DeepTraversalAdapter,
        ):
            self._adapters.append(adapter_cls(self.ctx))
        self._sort_adapters()

    def _sort_adapters(self) -> None:
        self._adapters.sort(key=lambda a: a.priority)

    def register_adapter(self, adapter: ProviderAdapter, priority: Optional[int] = None) -> None:
        """Dynamically register a new provider adapter without modifying core logic."""
        if not isinstance(adapter, ProviderAdapter):
            raise TypeError("register_adapter expects a ProviderAdapter instance.")
        if priority is not None:
            adapter.priority = int(priority)
        self._adapters.append(adapter)
        self._sort_adapters()
        logger.info(f"[OddsNormalizer] Registered adapter '{adapter.name}' with priority {adapter.priority}.")

    def set_field_mappings(self, mappings: Optional[Dict[str, Any]]) -> None:
        """Install (or clear, with ``None``/``{}``) the target's field mappings.

        Mappings take precedence over lexicon-based semantic matching for every
        canonical key they cover; unmapped keys keep falling back gracefully.
        """
        try:
            self.ctx.mappings.update(mappings)
            logger.info(
                f"[OddsNormalizer] Field mappings installed: {self.ctx.mappings.as_dict()}"
            )
        except Exception as err:
            logger.warning(f"[OddsNormalizer] Failed to install field mappings: {err}")

    def load_field_mappings_from_db(self, domain: str, db_manager: "DatabaseManager") -> bool:
        """Load stored ``field_mappings`` for ``domain`` from the control-plane database.

        Returns True when non-empty mappings were installed; returns False (never
        raises) when the target is unknown, has no mappings, or the DB fails.
        """
        if not isinstance(domain, str) or not domain.strip():
            logger.warning("[OddsNormalizer] load_field_mappings_from_db requires a non-empty domain.")
            return False
        try:
            target = db_manager.get_target_by_domain(domain.strip())
        except Exception as err:
            logger.warning(f"[OddsNormalizer] Failed to read target schema for '{domain}': {err}")
            return False
        if not target:
            logger.warning(f"[OddsNormalizer] No stored target schema for domain '{domain}'.")
            return False
        mappings = target.get("field_mappings")
        if not isinstance(mappings, dict) or not mappings:
            logger.warning(f"[OddsNormalizer] Target '{domain}' has no usable field_mappings; using lexicon fallback.")
            self.ctx.mappings.update({})
            return False
        self.ctx.mappings.update(mappings)
        logger.info(
            f"[OddsNormalizer] Loaded {len(self.ctx.mappings)} field mappings for '{domain}'."
        )
        return True

    def normalize(
        self,
        payload: Any,
        provider_hint: Optional[str] = None,
        field_mappings: Optional[Dict[str, Any]] = None,
    ) -> List[CanonicalEvent]:
        """Normalize an arbitrary raw payload into canonical events.

        Args:
            payload: Raw payload (parsed JSON object/array or a JSON string).
            provider_hint: Optional adapter name substring to prioritize.

        Returns:
            A list of :class:`CanonicalEvent`. Never raises on malformed input.
        """
        parsed = self._coerce_payload(payload)
        if parsed is None:
            return []

        previous_mappings = self.ctx.mappings.as_dict()
        if field_mappings is not None:
            try:
                self.ctx.mappings.update(field_mappings)
            except Exception as err:
                logger.warning(f"[OddsNormalizer] Invalid field_mappings ignored: {err}")

        try:
            hint = (provider_hint or "").strip().lower()
            ordered = list(self._adapters)
            if hint:
                ordered.sort(key=lambda a: (0 if hint in a.name.lower() else 1, a.priority))

            for adapter in ordered:
                try:
                    if not adapter.can_handle(parsed):
                        continue
                    results = adapter.normalize(parsed)
                    if results:
                        logger.info(
                            f"[OddsNormalizer] Adapter '{adapter.name}' normalized "
                            f"{len(results)} canonical event(s)."
                        )
                        return results
                except Exception as err:
                    logger.warning(f"[OddsNormalizer] Adapter '{adapter.name}' failed: {err}")
                    continue
        finally:
            # Always restore the prior mapping scope so per-call mappings never leak.
            self.ctx.mappings.update(previous_mappings)

        logger.warning("[OddsNormalizer] No adapter could extract canonical events from the payload.")
        return []

    def normalize_to_json(
        self,
        payload: Any,
        provider_hint: Optional[str] = None,
        indent: int = 2,
        field_mappings: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Normalize a payload and serialize the canonical result to JSON text."""
        events = self.normalize(
            payload, provider_hint=provider_hint, field_mappings=field_mappings
        )
        return json.dumps([e.to_dict() for e in events], indent=indent, ensure_ascii=False)

    def _coerce_payload(self, payload: Any) -> Any:
        if isinstance(payload, str):
            try:
                return json.loads(payload)
            except json.JSONDecodeError as err:
                logger.warning(f"[OddsNormalizer] Payload string is not valid JSON: {err}")
                return None
        if isinstance(payload, (dict, list)):
            return payload
        logger.warning("[OddsNormalizer] Unsupported payload type; expected JSON object/array or JSON string.")
        return None


if __name__ == "__main__":
    SAMPLE_KAMBI_STYLE = {
        "events": [
            {
                "id": 1001,
                "name": "Arsenal vs Chelsea",
                "homeTeam": {"name": "Arsenal"},
                "awayTeam": {"name": "Chelsea"},
                "start_time": "2024-06-01T18:00:00Z",
                "betOffers": [
                    {
                        "bet_offer_type": "moneyline",
                        "outcomes": [
                            {"selection_name": "Arsenal", "odds": 1.85, "outcome_id": "o1"},
                            {"selection_name": "Chelsea", "odds": 2.10, "outcome_id": "o2"},
                        ],
                    },
                    {
                        "bet_offer_type": "asian_handicap",
                        "handicap_value": -0.5,
                        "outcomes": [
                            {"selection_name": "Arsenal -0.5", "price": "+120", "outcome_id": "o3"},
                            {"selection_name": "Chelsea +0.5", "price": "-140", "outcome_id": "o4"},
                        ],
                    },
                ],
            }
        ]
    }

    SAMPLE_GENERIC_NESTED = {
        "fixture_id": "fx-7788",
        "home_team": "Real Madrid",
        "away_team": "Barcelona",
        "scheduled_at": "2024-06-02T20:00:00Z",
        "markets": [
            {
                "market_name": "Total Goals 2.5",
                "selections": [
                    {"selection_name": "Over 2.5", "price": "3/2"},
                    {"selection_name": "Under 2.5", "price": "1/2"},
                ],
            }
        ],
    }

    normalizer = OddsNormalizer("sportsbook_lexicon.json")

    # Schema-driven pass: install the target's discovered field mappings.
    normalizer.set_field_mappings({
        "home_team": "homeTeam",
        "away_team": "awayTeam",
        "match_container": "events",
        "market_container": "betOffers",
        "outcome_container": "outcomes",
        "odds_value": "odds",
        "league": "league",
        "sport": "sport",
    })

    per_call_mappings = {
        "home_team": "homeTeam",
        "away_team": "awayTeam",
        "match_container": "events",
        "market_container": "betOffers",
        "outcome_container": "outcomes",
        "odds_value": "odds",
        "league": "league",
        "sport": "sport",
    }

    for label, sample in (("kambi_style", SAMPLE_KAMBI_STYLE), ("generic_nested", SAMPLE_GENERIC_NESTED)):
        print(f"\n=== Normalizing sample: {label} ===")
        print(normalizer.normalize_to_json(sample, provider_hint=label, field_mappings=per_call_mappings))

    # Resilient fallback pass: clear mappings and rely on lexicon matching only.
    normalizer.set_field_mappings({})
    print("\n=== Fallback normalization without field mappings (generic_nested) ===")
    print(normalizer.normalize_to_json(SAMPLE_GENERIC_NESTED))
