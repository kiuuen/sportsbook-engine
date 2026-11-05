"""Package facade for the modularized odds normalization engine.

Exposes :class:`OddsNormalizer` — the orchestration entry point that selects a
registered :class:`~odds_normalizer.adapters.base.ProviderAdapter` for a raw
payload and translates it into canonical events — together with the canonical
entity models. Provider-specific layout handling lives in the
:mod:`odds_normalizer.adapters` subpackage; this module contains no adapter
logic itself.
"""

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from heuristic_classifier import LexiconManager
from odds_normalizer.adapters.base import NormalizationContext, ProviderAdapter
from odds_normalizer.adapters.relational import RelationalEntitiesAdapter
from odds_normalizer.adapters.standard import (
    DeepTraversalAdapter,
    GenericEventsAdapter,
    GenericMarketsAdapter,
    KambiStyleAdapter,
)
from odds_normalizer.classifier import MarketClassifier
from odds_normalizer.models import (
    CanonicalEvent,
    CanonicalMarket,
    CanonicalMarketType,
    CanonicalOutcome,
)
from odds_normalizer.utils import FieldMappingResolver, OddsConverter, SafePayloadUtils

if TYPE_CHECKING:
    from database_manager import DatabaseManager

logger = logging.getLogger("odds_normalizer")

__all__ = [
    "OddsNormalizer",
    "CanonicalEvent",
    "CanonicalMarket",
    "CanonicalOutcome",
    "CanonicalMarketType",
]


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
            field_mappings: Optional per-call mappings; installed for this call
                only and always restored afterwards.

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
