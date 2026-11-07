                                                                                                                              
"""In-memory real-time state store for unified cross-provider market data.                                                            
                                                                                                                                      
Design invariants                                                                                                                     
-----------------                                                                                                                     
- All mutations are strictly synchronous and must be invoked on the single                                                            
  asyncio event loop thread. Cooperative scheduling only preempts at ``await``                                                        
  points, so synchronous methods are atomic without lock overhead.                                                                    
- Provider quotes are isolated per ``(canonical_type, line, provider_domain)``                                                        
  bucket; one provider's ingest can never overwrite or delete another                                                                 
  provider's quotes for the same market line.                                                                                         
- Read accessors return decoupled, frozen snapshots built from fresh copies so                                                        
  consumer tasks never observe concurrent mutation of internal dictionaries.                                                          
- An O(1) reverse index maps ``(provider_domain, provider_event_id)`` to the                                                          
  canonical event id; evicting a canonical event removes every index entry                                                            
  pointing at it.                                                                                                                     
                                                                                                                                      
Complexity                                                                                                                            
----------                                                                                                                            
- ``register_event`` / ``ingest_quotes`` / ``get_event_id_for_provider``: O(1)                                                        
  index lookups plus O(outcomes) quote upserts.                                                                                       
- Snapshot reads: O(size of the requested subtree).                                                                                   
- ``evict``: O(number of events + number of provider identities evicted).                                                             
"""                                                                                                                                   
                                                                                                                                      
from __future__ import annotations                                                                                                    
                                                                                                                                      
import copy                                                                                                                           
import time                                                                                                                           
import uuid                                                                                                                           
from dataclasses import dataclass, field                                                                                              
from datetime import datetime, timezone                                                                                               
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple                                                               
                                                                                                                                      
from odds_normalizer.models import (                                                                                                  
    CanonicalMarket,                                                                                                                  
    CanonicalMarketType,                                                                                                              
    CanonicalOutcome,                                                                                                                 
)                                                                                                                                     
                                                                                                                                      
MarketKey = Tuple[str, Optional[float]]                                                                                               
ProviderIdentity = Tuple[str, str]                                                                                                    
                                                                                                                                      
                                                                                                                                      
def _require_non_empty_str(value: Any, name: str) -> str:                                                                             
    """Validate that *value* is a non-empty string and return it stripped."""                                                         
    if not isinstance(value, str) or not value.strip():                                                                               
        raise ValueError(f"'{name}' must be a non-empty string, got: {value!r}")                                                      
    return value.strip()                                                                                                              
                                                                                                                                      
                                                                                                                                      
def _validate_optional_float(value: Any, name: str) -> Optional[float]:                                                               
    """Validate that *value* is ``None`` or a real number (bools rejected)."""                                                        
    if value is None:                                                                                                                 
        return None                                                                                                                   
    if isinstance(value, bool) or not isinstance(value, (int, float)):                                                                
        raise TypeError(f"'{name}' must be None or a number, got: {value!r}")                                                         
    return float(value)                                                                                                               
                                                                                                                                      
                                                                                                                                      
@dataclass                                                                                                                            
class _ProviderOutcome:                                                                                                               
    """Internal mutable quote owned by exactly one provider."""                                                                       
                                                                                                                                      
    provider_domain: str                                                                                                              
    label: str                                                                                                                        
    odds_decimal: Optional[float]                                                                                                     
    odds_raw: Any                                                                                                                     
    odds_format: str                                                                                                                  
    line: Optional[float]                                                                                                             
    provider_outcome_id: Optional[str]                                                                                                
    updated_at: float                                                                                                                 
                                                                                                                                      
                                                                                                                                      
@dataclass                                                                                                                            
class _UnifiedMarket:                                                                                                                 
    """Internal mutable market: quotes bucketed per provider, keyed by label."""                                                      
                                                                                                                                      
    canonical_type: str                                                                                                               
    line: Optional[float]                                                                                                             
    raw_labels: Dict[str, str] = field(default_factory=dict)                                                                          
    quotes: Dict[str, Dict[str, _ProviderOutcome]] = field(default_factory=dict)                                                      
                                                                                                                                      
                                                                                                                                      
@dataclass                                                                                                                            
class _UnifiedEvent:                                                                                                                  
    """Internal mutable canonical event aggregating all provider quotes."""                                                           
                                                                                                                                      
    event_id: str                                                                                                                     
    home_team: Optional[str]                                                                                                          
    away_team: Optional[str]                                                                                                          
    start_time: Optional[str]                                                                                                         
    league: Optional[str]                                                                                                             
    sport: Optional[str]                                                                                                              
    provider_ids: Dict[str, str] = field(default_factory=dict)                                                                        
    markets: Dict[MarketKey, _UnifiedMarket] = field(default_factory=dict)                                                            
    created_at: float = 0.0                                                                                                           
    last_update: float = 0.0                                                                                                          
                                                                                                                                      
                                                                                                                                      
@dataclass(frozen=True)                                                                                                               
class OutcomeSnapshot:                                                                                                                
    """Immutable view of a single provider's quote for one selection."""                                                              
                                                                                                                                      
    provider_domain: str                                                                                                              
    label: str                                                                                                                        
    odds_decimal: Optional[float]                                                                                                     
    odds_raw: Any                                                                                                                     
    odds_format: str                                                                                                                  
    line: Optional[float]                                                                                                             
    provider_outcome_id: Optional[str]                                                                                                
    updated_at: float                                                                                                                 
                                                                                                                                      
                                                                                                                                      
@dataclass(frozen=True)                                                                                                               
class MarketSnapshot:                                                                                                                 
    """Immutable view of one market line with per-provider quote tuples."""                                                           
                                                                                                                                      
    canonical_type: str                                                                                                               
    line: Optional[float]                                                                                                             
    quotes: Dict[str, Tuple[OutcomeSnapshot, ...]]                                                                                    
                                                                                                                                      
                                                                                                                                      
@dataclass(frozen=True)                                                                                                               
class EventSnapshot:                                                                                                                  
    """Immutable, decoupled view of a canonical event and all its markets."""                                                         
                                                                                                                                      
    event_id: str                                                                                                                     
    home_team: Optional[str]                                                                                                          
    away_team: Optional[str]                                                                                                          
    start_time: Optional[str]                                                                                                         
    league: Optional[str]                                                                                                             
    sport: Optional[str]                                                                                                              
    providers: Tuple[str, ...]                                                                                                        
    markets: Tuple[MarketSnapshot, ...]                                                                                               
    created_at: float                                                                                                                 
    last_update: float                                                                                                                
                                                                                                                                      
                                                                                                                                      
class MarketStateStore:                                                                                                               
    """Synchronous in-memory store consolidating quotes across providers.                                                             
                                                                                                                                      
    Intended to be owned by the engine running on the main asyncio event loop;                                                        
    harvest and detail workers call ``register_event`` / ``ingest_quotes``                                                            
    synchronously, and consumer tasks read via snapshot accessors.                                                                    
    """                                                                                                                               
                                                                                                                                      
    def __init__(self, clock: Optional[Callable[[], float]] = None) -> None:                                                          
        """Initialize empty indexes.                                                                                                  
                                                                                                                                      
        Args:                                                                                                                         
            clock: Monotonic-ish wall-clock callable returning POSIX seconds;                                                         
                defaults to :func:`time.time`. Injectable for deterministic                                                           
                TTL testing.                                                                                                          
        """                                                                                                                           
        self._clock: Callable[[], float] = clock if clock is not None else time.time                                                  
        self._events: Dict[str, _UnifiedEvent] = {}                                                                                   
        self._provider_index: Dict[ProviderIdentity, str] = {}                                                                        
                                                                                                                                      
    # ------------------------------------------------------------------                                                              
    # Registration (provider-to-canonical identity mapping)                                                                           
    # ------------------------------------------------------------------                                                              
                                                                                                                                      
    def register_event(                                                                                                               
        self,                                                                                                                         
        provider_domain: str,                                                                                                         
        provider_event_id: str,                                                                                                       
        event_id: Optional[str] = None,                                                                                               
        home_team: Optional[str] = None,                                                                                              
        away_team: Optional[str] = None,                                                                                              
        start_time: Optional[str] = None,                                                                                             
        league: Optional[str] = None,                                                                                                 
        sport: Optional[str] = None,                                                                                                  
    ) -> str:                                                                                                                         
        """Register a provider event identity and return its canonical id.                                                            
                                                                                                                                      
        Idempotent: re-registering an already-known ``(provider_domain,                                                               
        provider_event_id)`` pair returns the existing canonical id and                                                               
        refreshes any metadata fields supplied as non-``None``.                                                                       
                                                                                                                                      
        Args:                                                                                                                         
            provider_domain: Domain of the harvesting provider.                                                                       
            provider_event_id: Provider-native event identifier.                                                                      
            event_id: Optional canonical id to adopt or attach to. When it                                                            
                already exists, the provider identity is attached to it                                                               
                (multi-provider merge); otherwise a new event is created.                                                             
            home_team / away_team / start_time / league / sport: Optional                                                             
                metadata; non-``None`` values overwrite existing metadata.                                                            
                                                                                                                                      
        Returns:                                                                                                                      
            The canonical event id backing this provider identity.                                                                    
                                                                                                                                      
        Raises:                                                                                                                       
            ValueError: On empty identifiers, or when the provider identity is                                                        
                already mapped to a *different* canonical event id.                                                                   
            TypeError: On non-string identifiers.                                                                                     
        """                                                                                                                           
        domain = _require_non_empty_str(provider_domain, "provider_domain")                                                           
        native_id = _require_non_empty_str(provider_event_id, "provider_event_id")                                                    
        identity: ProviderIdentity = (domain, native_id)                                                                              
                                                                                                                                      
        existing_event_id = self._provider_index.get(identity)                                                                        
        if existing_event_id is not None:                                                                                             
            if event_id is not None and event_id != existing_event_id:                                                                
                raise ValueError(                                                                                                     
                    f"Provider identity {identity!r} is already mapped to "                                                           
                    f"canonical event {existing_event_id!r}; conflicting "                                                            
                    f"event_id {event_id!r}."                                                                                         
                )                                                                                                                     
            event = self._events[existing_event_id]                                                                                   
            self._refresh_metadata(                                                                                                   
                event,                                                                                                                
                home_team=home_team,                                                                                                  
                away_team=away_team,                                                                                                  
                start_time=start_time,                                                                                                
                league=league,                                                                                                        
                sport=sport,                                                                                                          
            )                                                                                                                         
            return existing_event_id                                                                                                  
                                                                                                                                      
        if event_id is not None:                                                                                                      
            canonical_id = _require_non_empty_str(event_id, "event_id")                                                               
            event = self._events.get(canonical_id)                                                                                    
            if event is None:                                                                                                         
                event = self._create_event(                                                                                           
                    canonical_id,                                                                                                     
                    home_team=home_team,                                                                                              
                    away_team=away_team,                                                                                              
                    start_time=start_time,                                                                                            
                    league=league,                                                                                                    
                    sport=sport,                                                                                                      
                )                                                                                                                     
        else:                                                                                                                         
            canonical_id = uuid.uuid4().hex                                                                                           
            event = self._create_event(                                                                                               
                canonical_id,                                                                                                         
                home_team=home_team,                                                                                                  
                away_team=away_team,                                                                                                  
                start_time=start_time,                                                                                                
                league=league,                                                                                                        
                sport=sport,                                                                                                          
            )                                                                                                                         
                                                                                                                                      
        event.provider_ids[domain] = native_id                                                                                        
        self._provider_index[identity] = canonical_id                                                                                 
        return canonical_id                                                                                                           
                                                                                                                                      
    def get_event_id_for_provider(                                                                                                    
        self, provider_domain: str, provider_event_id: str                                                                            
    ) -> Optional[str]:                                                                                                               
        """O(1) reverse lookup of the canonical id for a provider identity."""                                                        
        domain = _require_non_empty_str(provider_domain, "provider_domain")                                                           
        native_id = _require_non_empty_str(provider_event_id, "provider_event_id")                                                    
        return self._provider_index.get((domain, native_id))                                                                          
                                                                                                                                      
    # ------------------------------------------------------------------                                                              
    # Quote ingestion (high-frequency detail workers)                                                                                 
    # ------------------------------------------------------------------                                                              
                                                                                                                                      
    def ingest_quotes(                                                                                                                
        self,                                                                                                                         
        provider_domain: str,                                                                                                         
        provider_event_id: str,                                                                                                       
        markets: Iterable[CanonicalMarket],                                                                                           
    ) -> str:                                                                                                                         
        """Ingest canonical markets using only the provider's native event id.                                                        
                                                                                                                                      
        Non-destructive: only the ingesting provider's quote bucket for each                                                          
        ``(canonical_type, line)`` key is touched; competing providers' quotes                                                        
        for the same line are preserved.                                                                                              
                                                                                                                                      
        Args:                                                                                                                         
            provider_domain: Domain of the detail worker's provider.                                                                  
            provider_event_id: Provider-native event identifier.                                                                      
            markets: Canonical markets produced by the normalizer.                                                                    
                                                                                                                                      
        Returns:                                                                                                                      
            The canonical event id the quotes were merged into.                                                                       
                                                                                                                                      
        Raises:                                                                                                                       
            KeyError: If the provider identity was never registered via                                                               
                :meth:`register_event`.                                                                                               
            TypeError: If ``markets`` yields non-``CanonicalMarket`` items or                                                         
                an outcome carries non-numeric fields.                                                                                
            ValueError: On empty provider identifiers or an unknown canonical                                                         
                market type.                                                                                                          
        """                                                                                                                           
        domain = _require_non_empty_str(provider_domain, "provider_domain")                                                           
        native_id = _require_non_empty_str(provider_event_id, "provider_event_id")                                                    
        identity: ProviderIdentity = (domain, native_id)                                                                              
                                                                                                                                      
        canonical_id = self._provider_index.get(identity)                                                                             
        if canonical_id is None:                                                                                                      
            raise KeyError(                                                                                                           
                f"Unregistered provider identity {identity!r}; call "                                                                 
                f"register_event() before ingesting quotes."                                                                          
            )                                                                                                                         
        event = self._events[canonical_id]                                                                                            
        now = self._clock()                                                                                                           
                                                                                                                                      
        for market in markets:                                                                                                        
            if not isinstance(market, CanonicalMarket):                                                                               
                raise TypeError(                                                                                                      
                    f"Expected CanonicalMarket, got: {type(market).__name__}"                                                         
                )                                                                                                                     
            if market.canonical_type not in CanonicalMarketType.ALL:                                                                  
                raise ValueError(                                                                                                     
                    f"Unknown canonical market type: {market.canonical_type!r}"                                                       
                )                                                                                                                     
                                                                                                                                      
            line_groups: Dict[Optional[float], List[CanonicalOutcome]] = {}                                                           
            for outcome in market.outcomes:                                                                                           
                if not isinstance(outcome, CanonicalOutcome):                                                                         
                    raise TypeError(                                                                                                  
                        f"Expected CanonicalOutcome, got: {type(outcome).__name__}"                                                   
                    )                                                                                                                 
                _validate_optional_float(outcome.line, "outcome.line")                                                                
                _validate_optional_float(outcome.odds_decimal, "outcome.odds_decimal")                                                
                line_groups.setdefault(outcome.line, []).append(outcome)                                                              
                                                                                                                                      
            for line, outcomes in line_groups.items():                                                                                
                key: MarketKey = (market.canonical_type, line)                                                                        
                unified = event.markets.get(key)                                                                                      
                if unified is None:                                                                                                   
                    unified = _UnifiedMarket(                                                                                         
                        canonical_type=market.canonical_type, line=line                                                               
                    )                                                                                                                 
                    event.markets[key] = unified                                                                                      
                unified.raw_labels[domain] = market.raw_label                                                                         
                                                                                                                                      
                bucket = unified.quotes.setdefault(domain, {})                                                                        
                for outcome in outcomes:                                                                                              
                    bucket[outcome.label] = _ProviderOutcome(                                                                         
                        provider_domain=domain,                                                                                       
                        label=outcome.label,                                                                                          
                        odds_decimal=outcome.odds_decimal,                                                                            
                        odds_raw=copy.deepcopy(outcome.odds_raw),                                                                     
                        odds_format=outcome.odds_format,                                                                              
                        line=outcome.line,                                                                                            
                        provider_outcome_id=outcome.provider_outcome_id,                                                              
                        updated_at=now,                                                                                               
                    )                                                                                                                 
                                                                                                                                      
        event.provider_ids.setdefault(domain, native_id)                                                                              
        event.last_update = now                                                                                                       
        return canonical_id                                                                                                           
                                                                                                                                      
    # ------------------------------------------------------------------                                                              
    # Snapshot reads (consumer-safe, decoupled)                                                                                       
    # ------------------------------------------------------------------                                                              
                                                                                                                                      
    def list_event_ids(self) -> Tuple[str, ...]:                                                                                      
        """Return a tuple of all live canonical event ids."""                                                                         
        return tuple(self._events.keys())                                                                                             
                                                                                                                                      
    def get_event_snapshot(self, event_id: str) -> Optional[EventSnapshot]:                                                           
        """Return a frozen snapshot of one event, or ``None`` if unknown.                                                             
                                                                                                                                      
        The snapshot is built from fresh copies, so mutating any returned                                                             
        container cannot affect internal store state.                                                                                 
        """                                                                                                                           
        canonical_id = _require_non_empty_str(event_id, "event_id")                                                                   
        event = self._events.get(canonical_id)                                                                                        
        if event is None:                                                                                                             
            return None                                                                                                               
        return self._build_event_snapshot(event)                                                                                      
                                                                                                                                      
    def snapshot_all(self) -> Tuple[EventSnapshot, ...]:                                                                              
        """Return frozen snapshots of every live event, sorted by event id."""                                                        
        return tuple(                                                                                                                 
            self._build_event_snapshot(self._events[event_id])                                                                        
            for event_id in sorted(self._events)                                                                                      
        )                                                                                                                             
                                                                                                                                      
    # ------------------------------------------------------------------                                                              
    # Eviction / memory lifecycle                                                                                                     
    # ------------------------------------------------------------------                                                              
                                                                                                                                      
    def evict(                                                                                                                        
        self,                                                                                                                         
        now: Optional[float] = None,                                                                                                  
        start_time_grace_seconds: float = 3600.0,                                                                                     
        update_ttl_seconds: float = 900.0,                                                                                            
    ) -> Tuple[str, ...]:                                                                                                             
        """Remove stale events and prune their reverse-index entries.                                                                 
                                                                                                                                      
        An event is evicted when either:                                                                                              
        - its parsed ``start_time`` (UTC) plus ``start_time_grace_seconds`` has                                                       
          elapsed, or                                                                                                                 
        - it has not received any quote ingest for ``update_ttl_seconds``.                                                            
                                                                                                                                      
        Events with an absent or unparseable ``start_time`` are only eligible                                                         
        via the update-TTL rule, so malformed metadata can never strand data.                                                         
                                                                                                                                      
        Args:                                                                                                                         
            now: Current POSIX timestamp; defaults to the injected clock.                                                             
            start_time_grace_seconds: Grace period after kickoff before the                                                           
                event is considered finished. Must be ``>= 0``.                                                                       
            update_ttl_seconds: Maximum age of the last ingest before the                                                             
                event is considered stale. Must be ``>= 0``.                                                                          
                                                                                                                                      
        Returns:                                                                                                                      
            Tuple of evicted canonical event ids.                                                                                     
                                                                                                                                      
        Raises:                                                                                                                       
            ValueError: If either threshold is negative.                                                                              
        """                                                                                                                           
        if start_time_grace_seconds < 0:                                                                                              
            raise ValueError("start_time_grace_seconds must be >= 0")                                                                 
        if update_ttl_seconds < 0:                                                                                                    
            raise ValueError("update_ttl_seconds must be >= 0")                                                                       
        now_ts: float = now if now is not None else self._clock()                                                                     
                                                                                                                                      
        evicted: List[str] = []                                                                                                       
        for event_id, event in list(self._events.items()):                                                                            
            kickoff = self._parse_start_time(event.start_time)                                                                        
            start_stale = kickoff is not None and now_ts > (                                                                          
                kickoff.timestamp() + start_time_grace_seconds                                                                        
            )                                                                                                                         
            ttl_stale = (now_ts - event.last_update) > update_ttl_seconds                                                             
            if not (start_stale or ttl_stale):                                                                                        
                continue                                                                                                              
                                                                                                                                      
            del self._events[event_id]                                                                                                
            for provider_domain, native_id in event.provider_ids.items():                                                             
                identity: ProviderIdentity = (provider_domain, native_id)                                                             
                if self._provider_index.get(identity) == event_id:                                                                    
                    del self._provider_index[identity]                                                                                
            evicted.append(event_id)                                                                                                  
        return tuple(evicted)                                                                                                         
                                                                                                                                      
    # ------------------------------------------------------------------                                                              
    # Internal helpers                                                                                                                
    # ------------------------------------------------------------------                                                              
                                                                                                                                      
    def _create_event(                                                                                                                
        self,                                                                                                                         
        event_id: str,                                                                                                                
        home_team: Optional[str],                                                                                                     
        away_team: Optional[str],                                                                                                     
        start_time: Optional[str],                                                                                                    
        league: Optional[str],                                                                                                        
        sport: Optional[str],                                                                                                         
    ) -> _UnifiedEvent:                                                                                                               
        """Create and index a new canonical event; caller wires provider ids."""                                                      
        now = self._clock()                                                                                                           
        event = _UnifiedEvent(                                                                                                        
            event_id=event_id,                                                                                                        
            home_team=home_team,                                                                                                      
            away_team=away_team,                                                                                                      
            start_time=start_time,                                                                                                    
            league=league,                                                                                                            
            sport=sport,                                                                                                              
            created_at=now,                                                                                                           
            last_update=now,                                                                                                          
        )                                                                                                                             
        self._events[event_id] = event                                                                                                
        return event                                                                                                                  
                                                                                                                                      
    @staticmethod                                                                                                                     
    def _refresh_metadata(                                                                                                            
        event: _UnifiedEvent,                                                                                                         
        home_team: Optional[str],                                                                                                     
        away_team: Optional[str],                                                                                                     
        start_time: Optional[str],                                                                                                    
        league: Optional[str],                                                                                                        
        sport: Optional[str],                                                                                                         
    ) -> None:                                                                                                                        
        """Overwrite event metadata fields for which a new value was supplied."""                                                     
        if home_team is not None:                                                                                                     
            event.home_team = home_team                                                                                               
        if away_team is not None:                                                                                                     
            event.away_team = away_team                                                                                               
        if start_time is not None:                                                                                                    
            event.start_time = start_time                                                                                             
        if league is not None:                                                                                                        
            event.league = league                                                                                                     
        if sport is not None:                                                                                                         
            event.sport = sport                                                                                                       
                                                                                                                                      
    @staticmethod                                                                                                                     
    def _parse_start_time(raw: Optional[str]) -> Optional[datetime]:                                                                  
        """Parse an ISO-8601 ``start_time`` to aware UTC, or ``None``.                                                                
                                                                                                                                      
        Naive timestamps are assumed UTC; unparseable values return ``None``                                                          
        so eviction falls back to the update-TTL rule.                                                                                
        """                                                                                                                           
        if not isinstance(raw, str) or not raw.strip():                                                                               
            return None                                                                                                               
        text = raw.strip()                                                                                                            
        if text.endswith("Z"):                                                                                                        
            text = text[:-1] + "+00:00"                                                                                               
        try:                                                                                                                          
            parsed = datetime.fromisoformat(text)                                                                                     
        except ValueError:                                                                                                            
            return None                                                                                                               
        if parsed.tzinfo is None:                                                                                                     
            parsed = parsed.replace(tzinfo=timezone.utc)                                                                              
        return parsed.astimezone(timezone.utc)                                                                                        
                                                                                                                                      
    def _build_event_snapshot(self, event: _UnifiedEvent) -> EventSnapshot:                                                           
        """Build a fully decoupled frozen snapshot of an internal event."""                                                           
        markets = tuple(                                                                                                              
            self._build_market_snapshot(event.markets[key])                                                                           
            for key in sorted(event.markets, key=lambda k: (k[0], k[1] is not None, k[1] or 0.0))                                     
        )                                                                                                                             
        return EventSnapshot(                                                                                                         
            event_id=event.event_id,                                                                                                  
            home_team=event.home_team,                                                                                                
            away_team=event.away_team,                                                                                                
            start_time=event.start_time,                                                                                              
            league=event.league,                                                                                                      
            sport=event.sport,                                                                                                        
            providers=tuple(sorted(event.provider_ids)),                                                                              
            markets=markets,                                                                                                          
            created_at=event.created_at,                                                                                              
            last_update=event.last_update,                                                                                            
        )                                                                                                                             
                                                                                                                                      
    @staticmethod                                                                                                                     
    def _build_market_snapshot(market: _UnifiedMarket) -> MarketSnapshot:                                                             
        """Build a frozen market snapshot with per-provider outcome tuples."""                                                        
        quotes: Dict[str, Tuple[OutcomeSnapshot, ...]] = {}                                                                           
        for provider_domain, outcomes in market.quotes.items():                                                                       
            quotes[provider_domain] = tuple(                                                                                          
                OutcomeSnapshot(                                                                                                      
                    provider_domain=outcome.provider_domain,                                                                          
                    label=outcome.label,                                                                                              
                    odds_decimal=outcome.odds_decimal,                                                                                
                    odds_raw=copy.deepcopy(outcome.odds_raw),                                                                         
                    odds_format=outcome.odds_format,                                                                                  
                    line=outcome.line,                                                                                                
                    provider_outcome_id=outcome.provider_outcome_id,                                                                  
                    updated_at=outcome.updated_at,                                                                                    
                )                                                                                                                     
                for outcome in sorted(outcomes.values(), key=lambda o: o.label)                                                       
            )                                                                                                                         
        return MarketSnapshot(                                                                                                        
            canonical_type=market.canonical_type,                                                                                     
            line=market.line,                                                                                                         
            quotes=quotes,                                                                                                            
        )                                                                                                                             
