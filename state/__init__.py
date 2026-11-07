"""Facade package for the in-memory real-time market state store.

This package consolidates cross-provider market data into canonical events and
exposes a strictly immutable, consumer-safe view of that state. It sits at the
boundary between the harvesting/normalization pipeline and downstream consumer
tasks (e.g. arbitrage evaluation, persistence, or API serialization).

Architectural role
------------------
- :class:`MarketStateStore` is the single synchronous mutation point, intended
  to be owned by the engine running on the main asyncio event loop. Harvest and
  detail workers call ``register_event`` / ``ingest_quotes`` on it; consumer
  tasks read exclusively through frozen snapshot objects.
- Snapshot dataclasses (:class:`EventSnapshot`, :class:`MarketSnapshot`,
  :class:`OutcomeSnapshot`) are decoupled, frozen views built from fresh copies,
  so consumers can never observe or induce concurrent mutation of internal
  dictionaries.
- Type aliases :data:`MarketKey` and :data:`ProviderIdentity` document the
  canonical market-line key and the provider reverse-index identity used
  throughout the store's public API.

Public API contract
-------------------
Only the names listed in ``__all__`` are part of the supported interface.
Internal mutable representations (``_UnifiedEvent``, ``_UnifiedMarket``,
``_ProviderOutcome``) are deliberately not exported; downstream code must rely
solely on the store methods and the frozen snapshot types.
"""

from odds_normalizer.models import CanonicalMarket, CanonicalMarketType, CanonicalOutcome
from state.store import (
    EventSnapshot,
    MarketKey,
    MarketSnapshot,
    MarketStateStore,
    OutcomeSnapshot,
    ProviderIdentity,
)

__all__ = [
    "CanonicalMarket",
    "CanonicalMarketType",
    "CanonicalOutcome",
    "EventSnapshot",
    "MarketKey",
    "MarketSnapshot",
    "MarketStateStore",
    "OutcomeSnapshot",
    "ProviderIdentity",
]
