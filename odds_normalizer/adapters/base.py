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
