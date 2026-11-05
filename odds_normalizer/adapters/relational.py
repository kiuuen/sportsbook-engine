"""Relational/columnar provider adapter for sibling entity collections.

Handles payloads such as ``{"events": [...], "markets": [...], "odds": [...]}``
where events, markets, and odds/price rows live side by side at the root and
reference each other through identifier pointers instead of nesting. Outcomes
and markets are indexed by stringified identifier, then stitched onto events
via top-down pointer arrays and bottom-up event foreign keys.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from odds_normalizer.adapters.base import ProviderAdapter
from odds_normalizer.models import CanonicalEvent, CanonicalMarket, CanonicalOutcome
from odds_normalizer.utils import SafePayloadUtils

logger = logging.getLogger("odds_normalizer")

class RelationalEntitiesAdapter(ProviderAdapter):
    """Adapter for relational/columnar payloads with sibling entity collections.

    Targets layouts such as ``{"events": [...], "markets": [...], "odds": [...]}``
    where events, markets, and odds/price rows live side by side at the root and
    reference each other through identifier pointers instead of nesting.
    """

    name = "relational_entities"
    priority = 15

    ODDS_LIST_TERMS: Set[str] = {"odds", "prices", "odds_list", "oddslist", "odd_ids", "oddids"}

    # Exact key names only — token-subset matching is forbidden for identifier
    # resolution because bare terms like "id" token-match secondary keys
    # such as "typeId" or "sportMarketId".
    ODD_POINTER_KEYS: Tuple[str, ...] = (
        "oddIds", "odd_ids", "oddids", "odds",
        "outcomeIds", "outcome_ids", "outcomeids",
        "selectionIds", "selection_ids", "selectionids",
        "selections", "runners", "runnerIds", "runner_ids",
    )
    MARKET_POINTER_KEYS: Tuple[str, ...] = (
        "marketIds", "market_ids", "marketids", "marketId", "market_id", "markets",
    )
    EVENT_REF_KEYS: Tuple[str, ...] = (
        "eventId", "event_id", "eventid",
        "matchId", "match_id", "matchid",
        "fixtureId", "fixture_id", "fixtureid",
        "parentEventId", "parent_event_id", "parentId", "parent_id", "parentid",
    )

    def can_handle(self, payload: Any) -> bool:
        """Claim payloads with root-level sibling event and market/odds collections."""
        if not isinstance(payload, dict):
            return False
        utils = self.ctx.utils
        has_events = any(utils.has_key(payload, term) for term in self.event_list_terms)
        if not has_events:
            return False
        has_markets = any(utils.has_key(payload, term) for term in self.market_list_terms)
        has_outcomes = any(utils.has_key(payload, term) for term in self.outcome_list_terms)
        return has_markets or has_outcomes

    @staticmethod
    def _exact_str_id(node: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[str]:
        """Return the first exact-key identifier coerced to a stripped string.

        Only literal key equality is used; no token matching and no learned
        mapping tier, so secondary keys can never hijack primary-key resolution.
        """
        if not isinstance(node, dict):
            return None
        for key in keys:
            if key in node:
                value = SafePayloadUtils.coerce_str(node[key])
                if value is not None:
                    return value
        return None

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
        odds_nodes = SafePayloadUtils.find_first_list(payload, self.ODDS_LIST_TERMS)
        if not odds_nodes:
            odds_nodes = SafePayloadUtils.find_first_list(payload, self.outcome_list_terms)
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
            outcome_id = self._exact_str_id(node, ("id",))
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
        market_nodes = self._find_container(payload, "market_container", self.market_list_terms) or []
        if not market_nodes:
            market_nodes = SafePayloadUtils.find_first_list(payload, self.market_list_terms) or []
        synthetic_counter = 0
        for node in market_nodes:
            if not isinstance(node, dict):
                continue
            try:
                raw_label = (
                    self.ctx.utils.coerce_str(
                        self._mapped_lookup(node, "market_label", self.market_label_terms)
                    )
                    or "unknown_market"
                )
                canonical_type, confidence = self.ctx.classifier.classify(raw_label)
                market = CanonicalMarket(
                    canonical_type=canonical_type,
                    raw_label=raw_label,
                    market_id=self._exact_str_id(node, ("id",)),
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
            event_ref = self._exact_str_id(node, self.EVENT_REF_KEYS)
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

        for key in self.ODD_POINTER_KEYS:
            if key not in node:
                continue
            value = node[key]
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        outcome = self._build_outcome(item)
                        if outcome is None:
                            continue
                        dedupe_key = (
                            SafePayloadUtils.coerce_str(item.get("id"))
                            or outcome.provider_outcome_id
                            or f"{outcome.label}|{outcome.odds_decimal}"
                        )
                        if dedupe_key in seen_ids:
                            continue
                        seen_ids.add(dedupe_key)
                        resolved.append(outcome)
                    else:
                        _resolve_ref(item)
            else:
                _resolve_ref(value)
            if resolved:
                return resolved

        if not resolved:
            nested = self._find_container(node, "outcome_container", self.outcome_list_terms) or []
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
            event_nodes = self._find_container(payload, "match_container", self.event_list_terms) or []
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
                for key in self.MARKET_POINTER_KEYS:
                    if key not in event_node:
                        continue
                    value = event_node[key]
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
