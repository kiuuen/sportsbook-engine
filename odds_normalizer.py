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
        specific_id = utils.coerce_str(utils.find_first(node, self.EVENT_ID_TERMS))
        generic_id = utils.coerce_str(utils.find_first(node, self.GENERIC_ID_TERMS))
        return {
            "event_id": specific_id or generic_id,
            "home_team": utils.coerce_str(utils.find_first(node, self.HOME_TERMS)),
            "away_team": utils.coerce_str(utils.find_first(node, self.AWAY_TERMS)),
            "start_time": utils.coerce_str(utils.find_first(node, self.START_TERMS)),
            "raw_label": utils.coerce_str(utils.find_first(node, self.EVENT_LABEL_TERMS)),
        }

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
        return self.ctx.utils.coerce_str(self.ctx.utils.find_first(node, self.MARKET_LABEL_TERMS))

    def _find_odds_raw(self, node: Any) -> Any:
        if not isinstance(node, dict):
            return None
        for term in self.ODDS_TERMS:
            if self.ctx.utils.has_key(node, term):
                value = self.ctx.utils.find_first(node, {term})
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
        for term in self.ODDS_TERMS:
            if utils.has_key(node, term):
                candidate = utils.find_first(node, {term})
                decimal, fmt = self.ctx.converter.to_decimal(candidate)
                if decimal is not None:
                    odds_raw, odds_decimal, odds_format = candidate, decimal, fmt
                    break
        if odds_decimal is None:
            return None

        line_value = utils.coerce_float(utils.find_first(node, self.LINE_TERMS))
        outcome_label = utils.coerce_str(utils.find_first(node, self.OUTCOME_LABEL_TERMS)) or "unknown_selection"
        outcome_id = utils.coerce_str(utils.find_first(node, self.OUTCOME_ID_TERMS))

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
        raw_label = utils.coerce_str(utils.find_first(node, self.MARKET_LABEL_TERMS)) or fallback_label
        canonical_type, confidence = self.ctx.classifier.classify(raw_label)
        market = CanonicalMarket(
            canonical_type=canonical_type,
            raw_label=raw_label,
            market_id=utils.coerce_str(utils.find_first(node, {"market_id", "id"})),
            classification_confidence=confidence,
        )
        outcome_list = utils.find_first_list(node, self.OUTCOME_LIST_TERMS) or []
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
        betoffers = self.ctx.utils.find_first_list(payload, self.MARKET_LIST_TERMS)
        if not betoffers:
            return False
        for betoffer in betoffers[:5]:
            if self.ctx.utils.find_first_list(betoffer, self.OUTCOME_LIST_TERMS):
                return True
        return False

    def normalize(self, payload: Any) -> List[CanonicalEvent]:
        utils = self.ctx.utils
        events_nodes = utils.find_first_list(payload, self.EVENT_LIST_TERMS)
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
            )
            betoffers = utils.find_first_list(event_node, self.MARKET_LIST_TERMS) or []
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
        markets = self.ctx.utils.find_first_list(payload, self.MARKET_LIST_TERMS)
        if markets:
            return True
        return bool(self.ctx.utils.find_first_list(payload, self.OUTCOME_LIST_TERMS))

    def normalize(self, payload: Any) -> List[CanonicalEvent]:
        identity = self._extract_event_identity(payload)
        event = CanonicalEvent(
            event_id=identity["event_id"],
            home_team=identity["home_team"],
            away_team=identity["away_team"],
            start_time=identity["start_time"],
            raw_label=identity["raw_label"],
            source_provider=self.name,
        )
        markets = self.ctx.utils.find_first_list(payload, self.MARKET_LIST_TERMS) or []
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
            events = self.ctx.utils.find_first_list(payload, self.EVENT_LIST_TERMS)
            return bool(events)
        return False

    def normalize(self, payload: Any) -> List[CanonicalEvent]:
        event_nodes: List[Any] = []
        if isinstance(payload, list):
            event_nodes = list(payload)
        else:
            found = self.ctx.utils.find_first_list(payload, self.EVENT_LIST_TERMS)
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
            )
            market_lists = self.ctx.utils.find_first_list(event_node, self.MARKET_LIST_TERMS) or []
            for market_node in market_lists:
                market = self._build_market(market_node)
                if market is not None:
                    event.markets.append(market)
            if event.markets or event.event_id:
                canonical_events.append(event)
        return canonical_events


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

    def __init__(self, lexicon_path: str = "sportsbook_lexicon.json") -> None:
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
        self._adapters: List[ProviderAdapter] = []
        self._register_default_adapters()

    def _register_default_adapters(self) -> None:
        for adapter_cls in (KambiStyleAdapter, GenericMarketsAdapter, GenericEventsAdapter, DeepTraversalAdapter):
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

    def normalize(self, payload: Any, provider_hint: Optional[str] = None) -> List[CanonicalEvent]:
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

        logger.warning("[OddsNormalizer] No adapter could extract canonical events from the payload.")
        return []

    def normalize_to_json(self, payload: Any, provider_hint: Optional[str] = None, indent: int = 2) -> str:
        """Normalize a payload and serialize the canonical result to JSON text."""
        events = self.normalize(payload, provider_hint=provider_hint)
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
    for label, sample in (("kambi_style", SAMPLE_KAMBI_STYLE), ("generic_nested", SAMPLE_GENERIC_NESTED)):
        print(f"\n=== Normalizing sample: {label} ===")
        print(normalizer.normalize_to_json(sample, provider_hint=label))
