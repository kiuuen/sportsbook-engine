"""Base adapter contract for odds normalization provider layouts.

Defines :class:`NormalizationContext` (shared dependencies injected into every
adapter) and :class:`ProviderAdapter` (the pluggable base class plus shared
defensive extraction helpers). This module contains no concrete adapters and
no normalization orchestration; concrete provider adapters must live in
sibling modules and subclass :class:`ProviderAdapter`.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from heuristic_classifier import LexiconManager
from odds_normalizer.classifier import MarketClassifier
from odds_normalizer.models import CanonicalEvent, CanonicalMarket, CanonicalOutcome
from odds_normalizer.utils import FieldMappingResolver, OddsConverter, SafePayloadUtils

logger = logging.getLogger("odds_normalizer")

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
    specific payload family into canonical events. Term vocabularies are
    resolved dynamically per instance: baseline seeds are merged with matching
    positive categories from the injected :class:`LexiconManager`, and learned
    target mappings (``FieldMappingResolver``) always take lookup priority.
    """

    name: str = "base_adapter"
    priority: int = 100

    # ------------------------------------------------------------------
    # Private baseline fallback seeds (used when the lexicon adds nothing).
    # ------------------------------------------------------------------
    _DEFAULT_EVENT_ID_TERMS: Set[str] = {"event_id", "match_id", "fixture_id", "game_id"}
    _DEFAULT_GENERIC_ID_TERMS: Set[str] = {"id"}
    _DEFAULT_HOME_TERMS: Set[str] = {"home_team", "home_competitor", "home", "home_name"}
    _DEFAULT_AWAY_TERMS: Set[str] = {"away_team", "away_competitor", "away", "away_name"}
    _DEFAULT_START_TERMS: Set[str] = {"commence_time", "start_time", "kickoff", "scheduled_at", "event_date"}
    _DEFAULT_EVENT_LABEL_TERMS: Set[str] = {"name", "event", "match", "title"}
    _DEFAULT_MARKET_LABEL_TERMS: Set[str] = {"bet_offer_type", "market_name", "market_type", "market", "sub_market"}
    _DEFAULT_OUTCOME_LABEL_TERMS: Set[str] = {"selection_name", "selection", "runner", "name", "label", "outcome"}
    _DEFAULT_ODDS_TERMS: Tuple[str, ...] = (
        "decimal_odds", "odds", "price", "over_price", "under_price", "american_odds", "fractional_odds",
    )
    _DEFAULT_LINE_TERMS: Set[str] = {"line", "spread", "spread_value", "handicap", "handicap_value"}
    _DEFAULT_OUTCOME_ID_TERMS: Set[str] = {"outcome_id", "selection_id", "runner_id", "id"}
    _DEFAULT_MARKET_LIST_TERMS: Set[str] = {"markets", "market", "betoffers", "betoffer"}
    _DEFAULT_OUTCOME_LIST_TERMS: Set[str] = {"outcomes", "outcome", "selections", "selection", "runners", "runner"}
    _DEFAULT_EVENT_LIST_TERMS: Set[str] = {"events", "fixtures", "matches"}
    _DEFAULT_LEAGUE_TERMS: Set[str] = {"league", "league_id", "competition", "tournament"}
    _DEFAULT_SPORT_TERMS: Set[str] = {"sport", "sport_id", "sport_key", "sport_name"}

    _LEXICON_CATEGORIES: Tuple[str, ...] = ("sports_event_metadata", "betting_markets_and_odds")

    # Routes each working vocabulary to the lexicon category it augments.
    _LEXICON_CATEGORY_BY_FIELD: Dict[str, str] = {
        "event_id_terms": "sports_event_metadata",
        "generic_id_terms": "sports_event_metadata",
        "home_terms": "sports_event_metadata",
        "away_terms": "sports_event_metadata",
        "start_terms": "sports_event_metadata",
        "event_label_terms": "sports_event_metadata",
        "league_terms": "sports_event_metadata",
        "sport_terms": "sports_event_metadata",
        "event_list_terms": "sports_event_metadata",
        "market_label_terms": "betting_markets_and_odds",
        "outcome_label_terms": "betting_markets_and_odds",
        "odds_terms": "betting_markets_and_odds",
        "line_terms": "betting_markets_and_odds",
        "outcome_id_terms": "betting_markets_and_odds",
        "market_list_terms": "betting_markets_and_odds",
        "outcome_list_terms": "betting_markets_and_odds",
    }

    def __init__(self, ctx: NormalizationContext) -> None:
        self.ctx = ctx
        self._lexicon_terms_by_category: Dict[str, Set[str]] = self._load_lexicon_terms()
        self.event_id_terms: Set[str] = self._build_working_terms("event_id_terms", self._DEFAULT_EVENT_ID_TERMS)
        self.generic_id_terms: Set[str] = self._build_working_terms("generic_id_terms", self._DEFAULT_GENERIC_ID_TERMS)
        self.home_terms: Set[str] = self._build_working_terms("home_terms", self._DEFAULT_HOME_TERMS)
        self.away_terms: Set[str] = self._build_working_terms("away_terms", self._DEFAULT_AWAY_TERMS)
        self.start_terms: Set[str] = self._build_working_terms("start_terms", self._DEFAULT_START_TERMS)
        self.event_label_terms: Set[str] = self._build_working_terms("event_label_terms", self._DEFAULT_EVENT_LABEL_TERMS)
        self.market_label_terms: Set[str] = self._build_working_terms("market_label_terms", self._DEFAULT_MARKET_LABEL_TERMS)
        self.outcome_label_terms: Set[str] = self._build_working_terms("outcome_label_terms", self._DEFAULT_OUTCOME_LABEL_TERMS)
        self.line_terms: Set[str] = self._build_working_terms("line_terms", self._DEFAULT_LINE_TERMS)
        self.outcome_id_terms: Set[str] = self._build_working_terms("outcome_id_terms", self._DEFAULT_OUTCOME_ID_TERMS)
        self.market_list_terms: Set[str] = self._build_working_terms("market_list_terms", self._DEFAULT_MARKET_LIST_TERMS)
        self.outcome_list_terms: Set[str] = self._build_working_terms("outcome_list_terms", self._DEFAULT_OUTCOME_LIST_TERMS)
        self.event_list_terms: Set[str] = self._build_working_terms("event_list_terms", self._DEFAULT_EVENT_LIST_TERMS)
        self.league_terms: Set[str] = self._build_working_terms("league_terms", self._DEFAULT_LEAGUE_TERMS)
        self.sport_terms: Set[str] = self._build_working_terms("sport_terms", self._DEFAULT_SPORT_TERMS)
        odds_working = self._build_working_terms("odds_terms", set(self._DEFAULT_ODDS_TERMS))
        seed_order = [t for t in self._DEFAULT_ODDS_TERMS if t in odds_working]
        self.odds_terms: Tuple[str, ...] = tuple(seed_order + sorted(odds_working.difference(seed_order)))

    def _load_lexicon_terms(self) -> Dict[str, Set[str]]:
        """Collect cleaned positive lexicon terms grouped by category.

        Terms are lower-cased, coerced to non-empty strings, and filtered
        through ``LexiconManager.is_generic_or_blacklisted`` so generic tokens
        never pollute the semantic vocabularies. Degrades to empty groups when
        the lexicon manager is unavailable or malformed.
        """
        grouped: Dict[str, Set[str]] = {}
        mgr = self.ctx.lexicon_mgr
        if mgr is None:
            logger.warning(f"[{self.name}] No LexiconManager injected; using seed vocabularies only.")
            return grouped
        positive = getattr(mgr, "positive_keywords", None)
        if not isinstance(positive, dict):
            logger.warning(f"[{self.name}] LexiconManager exposes no positive keywords; using seed vocabularies only.")
            return grouped
        is_generic = getattr(mgr, "is_generic_or_blacklisted", None)
        for category in self._LEXICON_CATEGORIES:
            raw_terms = positive.get(category)
            if not isinstance(raw_terms, (set, frozenset, list, tuple)):
                continue
            cleaned: Set[str] = set()
            for raw_term in raw_terms:
                term = SafePayloadUtils.coerce_str(raw_term)
                if term is None:
                    continue
                lowered = term.lower()
                if callable(is_generic) and is_generic(lowered):
                    continue
                cleaned.add(lowered)
            if cleaned:
                grouped[category] = cleaned
        return grouped

    def _build_working_terms(self, field_name: str, seeds: Set[str]) -> Set[str]:
        """Merge baseline seeds with lexicon terms from the field's category."""
        category = self._LEXICON_CATEGORY_BY_FIELD.get(field_name)
        lexicon_terms = (
            self._lexicon_terms_by_category.get(category, set()) if category else set()
        )
        return set(seeds) | lexicon_terms

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
        specific_id = utils.coerce_str(self._mapped_lookup(node, "event_id", self.event_id_terms))
        generic_id = utils.coerce_str(utils.find_first(node, self.generic_id_terms))
        return {
            "event_id": specific_id or generic_id,
            "home_team": utils.coerce_str(self._mapped_lookup(node, "home_team", self.home_terms)),
            "away_team": utils.coerce_str(self._mapped_lookup(node, "away_team", self.away_terms)),
            "start_time": utils.coerce_str(self._mapped_lookup(node, "start_time", self.start_terms)),
            "league": utils.coerce_str(self._mapped_lookup(node, "league", self.league_terms)),
            "sport": utils.coerce_str(self._mapped_lookup(node, "sport", self.sport_terms)),
            "raw_label": utils.coerce_str(self._mapped_lookup(node, "event_label", self.event_label_terms)),
        }

    def _mapped_lookup(self, node: Any, canonical_key: str, fallback_terms: Set[str]) -> Any:
        """Two-tier resolution: learned target mapping first, dynamic lexicon second.

        Tier 1 queries ``self.ctx.mappings`` (the target-specific dialect
        learned from SQLite) with an exact/token-subset key match at this node.
        Tier 2 falls back to depth-limited semantic matching against
        ``fallback_terms`` — callers must pass the instance-level dynamic sets
        (e.g. ``self.home_terms``) so lexicon-learned terms participate.
        """
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
        """Odds key candidates: schema-mapped key first, then dynamic lexicon terms."""
        mapped = self.ctx.mappings.get("odds_value")
        if mapped:
            return (mapped,) + self.odds_terms
        return self.odds_terms

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
        has_specific_id = any(utils.has_key(node, t) for t in self.event_id_terms)
        has_generic_id = any(utils.has_key(node, t) for t in self.generic_id_terms)
        has_context = any(
            utils.has_key(node, t) for t in (self.home_terms | self.away_terms | self.start_terms)
        )
        return has_context and (has_specific_id or has_generic_id)

    def _extract_market_label(self, node: Any) -> Optional[str]:
        """Derive a market label only from nodes that also carry outcome signals."""
        if not isinstance(node, dict):
            return None
        has_outcome_signals = any(
            self.ctx.utils.has_key(node, t) for t in self.outcome_list_terms
        ) or self._find_odds_raw(node) is not None
        if not has_outcome_signals:
            return None
        return self.ctx.utils.coerce_str(
            self._mapped_lookup(node, "market_label", self.market_label_terms)
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

        line_value = utils.coerce_float(self._mapped_lookup(node, "line", self.line_terms))
        outcome_label = (
            utils.coerce_str(self._mapped_lookup(node, "outcome_label", self.outcome_label_terms))
            or "unknown_selection"
        )
        outcome_id = utils.coerce_str(self._mapped_lookup(node, "outcome_id", self.outcome_id_terms))

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
            utils.coerce_str(self._mapped_lookup(node, "market_label", self.market_label_terms))
            or fallback_label
        )
        canonical_type, confidence = self.ctx.classifier.classify(raw_label)
        market = CanonicalMarket(
            canonical_type=canonical_type,
            raw_label=raw_label,
            market_id=utils.coerce_str(self._mapped_lookup(node, "market_label", {"market_id", "id"})),
            classification_confidence=confidence,
        )
        outcome_list = self._find_container(node, "outcome_container", self.outcome_list_terms) or []
        for outcome_node in outcome_list:
            outcome = self._build_outcome(outcome_node)
            if outcome is not None:
                market.outcomes.append(outcome)
        if not market.outcomes:
            direct_outcome = self._build_outcome(node)
            if direct_outcome is not None:
                market.outcomes.append(direct_outcome)
        return market if market.outcomes else None
