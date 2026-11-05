"""Standard provider adapters extracted from the legacy normalizer.

Contains the four baseline layout handlers: Kambi-style betoffer layouts,
root-level market/outcome collections, event-list collections, and the
deep-traversal structural fallback. All adapters subclass
:class:`odds_normalizer.adapters.base.ProviderAdapter` and inherit its
dynamic lexicon-driven vocabularies and defensive extraction helpers.
"""

import logging
from typing import Any, Dict, List, Optional

from odds_normalizer.adapters.base import NormalizationContext, ProviderAdapter
from odds_normalizer.models import CanonicalEvent, CanonicalMarket

logger = logging.getLogger("odds_normalizer")


class KambiStyleAdapter(ProviderAdapter):
    """Adapter for betoffer/outcome layouts (e.g. Kambi-style provider families)."""

    name = "kambi_style"
    priority = 10

    def can_handle(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        betoffers = self._find_container(payload, "market_container", self.market_list_terms)
        if not betoffers:
            return False
        for betoffer in betoffers[:5]:
            if self.ctx.utils.find_first_list(betoffer, self.outcome_list_terms):
                return True
        return False

    def normalize(self, payload: Any) -> List[CanonicalEvent]:
        events_nodes = self._find_container(payload, "match_container", self.event_list_terms)
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
            betoffers = self._find_container(event_node, "market_container", self.market_list_terms) or []
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
        if self._find_container(payload, "market_container", self.market_list_terms):
            return True
        return bool(self._find_container(payload, "outcome_container", self.outcome_list_terms))

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
        markets = self._find_container(payload, "market_container", self.market_list_terms) or []
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
            events = self._find_container(payload, "match_container", self.event_list_terms)
            return bool(events)
        return False

    def normalize(self, payload: Any) -> List[CanonicalEvent]:
        event_nodes: List[Any] = []
        if isinstance(payload, list):
            event_nodes = list(payload)
        else:
            found = self._find_container(payload, "match_container", self.event_list_terms)
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
            market_lists = self._find_container(event_node, "market_container", self.market_list_terms) or []
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
