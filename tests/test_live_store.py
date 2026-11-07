"""Unit tests for the synchronous in-memory :class:`MarketStateStore`.

Covers registration/identity indexing, non-destructive multi-provider quote
ingestion, market-line separation, snapshot isolation/immutability,
deterministic TTL-based eviction via an injected clock, and defensive input
validation. All tests exercise real store instances through the public API;
time-dependent behavior is driven deterministically by an injectable clock.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest

from odds_normalizer.models import (
    CanonicalMarket,
    CanonicalMarketType,
    CanonicalOutcome,
)
from state import (
    EventSnapshot,
    MarketSnapshot,
    MarketStateStore,
    OutcomeSnapshot,
)


class FakeClock:
    """Deterministic stand-in for the store's injectable clock callable."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now: float = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the simulated wall clock forward by *seconds*."""
        self.now += seconds

    def iso_now(self) -> str:
        """Return the current simulated time as an ISO-8601 UTC string."""
        return datetime.fromtimestamp(self.now, tz=timezone.utc).isoformat()


@pytest.fixture
def fake_clock() -> FakeClock:
    """Provide a deterministic clock starting at a fixed POSIX timestamp."""
    return FakeClock()


@pytest.fixture
def store(fake_clock: FakeClock) -> MarketStateStore:
    """Provide a real store wired to the deterministic clock."""
    return MarketStateStore(clock=fake_clock)


def _outcome(
    label: str,
    odds_decimal: Optional[float],
    line: Optional[float] = None,
    odds_raw: Any = None,
) -> CanonicalOutcome:
    """Build a canonical outcome with sensible defaults."""
    return CanonicalOutcome(
        label=label,
        odds_decimal=odds_decimal,
        odds_raw=odds_raw,
        odds_format="decimal",
        line=line,
    )


def _moneyline_market(
    home_odds: float,
    away_odds: float,
    raw_label: str = "Match Winner",
) -> CanonicalMarket:
    """Build a two-way moneyline market."""
    return CanonicalMarket(
        canonical_type=CanonicalMarketType.MONEYLINE,
        raw_label=raw_label,
        outcomes=[
            _outcome("Home", home_odds),
            _outcome("Away", away_odds),
        ],
    )


def _totals_market(
    line: float,
    over_odds: float,
    under_odds: float,
) -> CanonicalMarket:
    """Build a parameterized totals market at the given line."""
    return CanonicalMarket(
        canonical_type=CanonicalMarketType.TOTALS,
        raw_label=f"Totals {line}",
        outcomes=[
            _outcome("Over", over_odds, line=line),
            _outcome("Under", under_odds, line=line),
        ],
    )


def _find_market(
    snapshot: EventSnapshot,
    canonical_type: str,
    line: Optional[float],
) -> Optional[MarketSnapshot]:
    """Locate a market snapshot by canonical type and line, or ``None``."""
    for market in snapshot.markets:
        if market.canonical_type == canonical_type and market.line == line:
            return market
    return None


def _quote_odds(
    market: MarketSnapshot,
    provider_domain: str,
    label: str,
) -> Optional[float]:
    """Extract one provider's decimal odds for a label from a market snapshot."""
    for outcome in market.quotes.get(provider_domain, ()):
        if outcome.label == label:
            return outcome.odds_decimal
    return None


# ---------------------------------------------------------------------------
# 1. Registration & identity indexing
# ---------------------------------------------------------------------------


def test_register_event_creates_event_and_populates_reverse_index(
    store: MarketStateStore,
) -> None:
    """Scenario: first-time registration of a provider identity.

    Invariant: a new canonical event id is returned, the event is live in the
    store, and the reverse index resolves the identity back to that id.
    Failure modes: wrong id returned, event missing from listings, or reverse
    lookup failing would break the harvest-to-canonical mapping contract.
    """
    canonical_id = store.register_event("bookmaker-a.com", "raw-1")

    assert isinstance(canonical_id, str) and canonical_id
    assert store.list_event_ids() == (canonical_id,)
    assert (
        store.get_event_id_for_provider("bookmaker-a.com", "raw-1")
        == canonical_id
    )


def test_register_event_is_idempotent_and_refreshes_metadata(
    store: MarketStateStore,
) -> None:
    """Scenario: re-registering the same (provider, native id) pair.

    Invariant: the canonical id is stable across re-registration and any
    non-``None`` metadata supplied later overwrites the stored metadata.
    Failure modes: a second canonical event being created, or metadata not
    being refreshed, would fragment or stale the unified view.
    """
    first_id = store.register_event(
        "bookmaker-a.com", "raw-1", home_team="Old Home"
    )

    second_id = store.register_event(
        "bookmaker-a.com",
        "raw-1",
        home_team="New Home",
        away_team="New Away",
    )

    assert second_id == first_id
    assert len(store.list_event_ids()) == 1
    snapshot = store.get_event_snapshot(first_id)
    assert snapshot is not None
    assert snapshot.home_team == "New Home"
    assert snapshot.away_team == "New Away"


def test_register_event_merges_multiple_providers_into_existing_event(
    store: MarketStateStore,
) -> None:
    """Scenario: two providers attach to one explicit canonical event id.

    Invariant: both identities map to the same canonical id and the event
    snapshot lists both provider domains.
    Failure modes: a second event being created or one provider domain missing
    from the snapshot would break cross-provider consolidation.
    """
    canonical_id = store.register_event(
        "bookmaker-a.com", "raw-a", event_id="evt-1"
    )
    merged_id = store.register_event(
        "bookmaker-b.com", "raw-b", event_id="evt-1"
    )

    assert merged_id == canonical_id == "evt-1"
    snapshot = store.get_event_snapshot("evt-1")
    assert snapshot is not None
    assert snapshot.providers == ("bookmaker-a.com", "bookmaker-b.com")
    assert store.get_event_id_for_provider("bookmaker-b.com", "raw-b") == "evt-1"


def test_register_event_conflicting_canonical_id_raises_value_error(
    store: MarketStateStore,
) -> None:
    """Scenario: an already-mapped identity is pointed at a different id.

    Invariant: registration raises ``ValueError`` and the store state is left
    untouched (original mapping and event survive).
    Failure modes: silently remapping the identity or corrupting the original
    event would silently re-route another provider's quote stream.
    """
    original_id = store.register_event("bookmaker-a.com", "raw-1")

    with pytest.raises(ValueError):
        store.register_event("bookmaker-a.com", "raw-1", event_id="evt-other")

    assert store.list_event_ids() == (original_id,)
    assert store.get_event_id_for_provider("bookmaker-a.com", "raw-1") == original_id


def test_get_event_id_for_provider_returns_none_for_unknown_identity(
    store: MarketStateStore,
) -> None:
    """Scenario: reverse lookup for an identity that was never registered.

    Invariant: unknown identities resolve to ``None`` rather than raising.
    Failure modes: raising or returning a stale id would break callers that
    probe the index before registering.
    """
    assert store.get_event_id_for_provider("unknown.com", "nope") is None


# ---------------------------------------------------------------------------
# 2. Non-destructive multi-provider ingestion
# ---------------------------------------------------------------------------


def test_ingest_preserves_quotes_from_multiple_providers_same_line(
    store: MarketStateStore,
) -> None:
    """Scenario: providers A and B ingest moneyline quotes for the same event.

    Invariant: both providers' quote buckets coexist side-by-side in
    ``MarketSnapshot.quotes`` with their own odds values.
    Failure modes: one provider's ingest overwriting or deleting the other's
    quotes would silently destroy competing odds data.
    """
    store.register_event("bookmaker-a.com", "raw-a", event_id="evt-1")
    store.register_event("bookmaker-b.com", "raw-b", event_id="evt-1")

    store.ingest_quotes(
        "bookmaker-a.com", "raw-a", [_moneyline_market(1.5, 3.0)]
    )
    store.ingest_quotes(
        "bookmaker-b.com", "raw-b", [_moneyline_market(1.6, 2.9)]
    )

    snapshot = store.get_event_snapshot("evt-1")
    assert snapshot is not None
    market = _find_market(snapshot, CanonicalMarketType.MONEYLINE, None)
    assert market is not None
    assert set(market.quotes) == {"bookmaker-a.com", "bookmaker-b.com"}
    assert _quote_odds(market, "bookmaker-a.com", "Home") == 1.5
    assert _quote_odds(market, "bookmaker-b.com", "Home") == 1.6
    assert _quote_odds(market, "bookmaker-a.com", "Away") == 3.0
    assert _quote_odds(market, "bookmaker-b.com", "Away") == 2.9


def test_provider_update_replaces_only_its_own_bucket(
    store: MarketStateStore,
) -> None:
    """Scenario: provider A re-ingests after both A and B had quoted.

    Invariant: only provider A's odds are replaced; provider B's bucket is
    byte-for-byte unchanged (labels, odds, and ordering).
    Failure modes: cross-provider clobbering or partial bucket deletion would
    violate the per-provider isolation invariant.
    """
    store.register_event("bookmaker-a.com", "raw-a", event_id="evt-1")
    store.register_event("bookmaker-b.com", "raw-b", event_id="evt-1")
    store.ingest_quotes(
        "bookmaker-a.com", "raw-a", [_moneyline_market(1.5, 3.0)]
    )
    store.ingest_quotes(
        "bookmaker-b.com", "raw-b", [_moneyline_market(1.6, 2.9)]
    )

    store.ingest_quotes(
        "bookmaker-a.com", "raw-a", [_moneyline_market(1.55, 2.95)]
    )

    snapshot = store.get_event_snapshot("evt-1")
    assert snapshot is not None
    market = _find_market(snapshot, CanonicalMarketType.MONEYLINE, None)
    assert market is not None
    assert _quote_odds(market, "bookmaker-a.com", "Home") == 1.55
    assert _quote_odds(market, "bookmaker-a.com", "Away") == 2.95
    assert _quote_odds(market, "bookmaker-b.com", "Home") == 1.6
    assert _quote_odds(market, "bookmaker-b.com", "Away") == 2.9


def test_ingest_quotes_for_unregistered_identity_raises_key_error(
    store: MarketStateStore,
) -> None:
    """Scenario: quotes arrive for a provider identity never registered.

    Invariant: ``ingest_quotes`` raises ``KeyError`` and no event is created.
    Failure modes: auto-vivifying an event or silently dropping quotes would
    hide integration bugs in the harvest pipeline.
    """
    with pytest.raises(KeyError):
        store.ingest_quotes(
            "ghost.com", "raw-x", [_moneyline_market(1.5, 3.0)]
        )

    assert store.list_event_ids() == ()


# ---------------------------------------------------------------------------
# 3. Market line separation
# ---------------------------------------------------------------------------


def test_distinct_lines_create_distinct_non_colliding_markets(
    store: MarketStateStore,
) -> None:
    """Scenario: totals ingested at lines 2.5 and 3.5 from one provider.

    Invariant: each ``(canonical_type, line)`` key yields its own
    ``MarketSnapshot`` with only its own outcomes; neither line's quotes are
    overwritten by the other.
    Failure modes: line collision would merge or overwrite parameterized
    markets and corrupt odds attribution.
    """
    store.register_event("bookmaker-a.com", "raw-a", event_id="evt-1")
    store.ingest_quotes(
        "bookmaker-a.com",
        "raw-a",
        [_totals_market(2.5, 1.9, 1.9), _totals_market(3.5, 2.1, 1.75)],
    )

    snapshot = store.get_event_snapshot("evt-1")
    assert snapshot is not None
    lines = sorted(
        market.line
        for market in snapshot.markets
        if market.canonical_type == CanonicalMarketType.TOTALS
    )
    assert lines == [2.5, 3.5]

    market_25 = _find_market(snapshot, CanonicalMarketType.TOTALS, 2.5)
    market_35 = _find_market(snapshot, CanonicalMarketType.TOTALS, 3.5)
    assert market_25 is not None and market_35 is not None
    assert _quote_odds(market_25, "bookmaker-a.com", "Over") == 1.9
    assert _quote_odds(market_35, "bookmaker-a.com", "Over") == 2.1
    assert _quote_odds(market_35, "bookmaker-a.com", "Under") == 1.75


# ---------------------------------------------------------------------------
# 4. Snapshot isolation & immutability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "attribute, value",
    [
        ("event_id", "tampered"),
        ("home_team", "tampered"),
        ("last_update", 0.0),
    ],
)
def test_event_snapshot_is_frozen(
    store: MarketStateStore, attribute: str, value: Any
) -> None:
    """Scenario: mutating any attribute of an ``EventSnapshot``.

    Invariant: frozen dataclasses raise ``FrozenInstanceError`` on attribute
    assignment.
    Failure modes: a mutable snapshot would let consumers corrupt shared state.
    """
    store.register_event("bookmaker-a.com", "raw-a", event_id="evt-1")
    snapshot = store.get_event_snapshot("evt-1")
    assert snapshot is not None

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(snapshot, attribute, value)


def test_market_and_outcome_snapshots_are_frozen(
    store: MarketStateStore,
) -> None:
    """Scenario: mutating nested ``MarketSnapshot``/``OutcomeSnapshot`` fields.

    Invariant: every snapshot layer is frozen, not just the event root.
    Failure modes: mutable nested views would expose an indirect write path
    into consumer-visible state.
    """
    store.register_event("bookmaker-a.com", "raw-a", event_id="evt-1")
    store.ingest_quotes(
        "bookmaker-a.com", "raw-a", [_moneyline_market(1.5, 3.0)]
    )
    snapshot = store.get_event_snapshot("evt-1")
    assert snapshot is not None
    market = snapshot.markets[0]
    outcome = market.quotes["bookmaker-a.com"][0]

    with pytest.raises(dataclasses.FrozenInstanceError):
        market.line = 99.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.odds_decimal = 99.0


def test_snapshot_decoupled_from_subsequent_ingest(
    store: MarketStateStore,
) -> None:
    """Scenario: a snapshot is taken, then new quotes are ingested.

    Invariant: the previously returned snapshot still shows the old odds while
    a freshly taken snapshot reflects the update; the old snapshot object was
    never mutated in place.
    Failure modes: aliasing internal dictionaries would let later ingests
    rewrite history in already-issued snapshots.
    """
    store.register_event("bookmaker-a.com", "raw-a", event_id="evt-1")
    store.ingest_quotes(
        "bookmaker-a.com", "raw-a", [_moneyline_market(1.5, 3.0)]
    )
    old_snapshot = store.get_event_snapshot("evt-1")
    assert old_snapshot is not None

    store.ingest_quotes(
        "bookmaker-a.com", "raw-a", [_moneyline_market(9.99, 9.99)]
    )

    old_market = _find_market(old_snapshot, CanonicalMarketType.MONEYLINE, None)
    assert old_market is not None
    assert _quote_odds(old_market, "bookmaker-a.com", "Home") == 1.5
    assert _quote_odds(old_market, "bookmaker-a.com", "Away") == 3.0

    new_snapshot = store.get_event_snapshot("evt-1")
    assert new_snapshot is not None
    new_market = _find_market(new_snapshot, CanonicalMarketType.MONEYLINE, None)
    assert new_market is not None
    assert _quote_odds(new_market, "bookmaker-a.com", "Home") == 9.99


def test_snapshot_quotes_container_is_decoupled_from_store(
    store: MarketStateStore,
) -> None:
    """Scenario: a consumer mutates the containers inside a snapshot.

    Invariant: inserting a fake provider key into the snapshot's quotes dict
    does not leak into the store; a fresh snapshot is unaffected.
    Failure modes: shared dict references would let consumers inject phantom
    providers into live state.
    """
    store.register_event("bookmaker-a.com", "raw-a", event_id="evt-1")
    store.ingest_quotes(
        "bookmaker-a.com", "raw-a", [_moneyline_market(1.5, 3.0)]
    )
    snapshot = store.get_event_snapshot("evt-1")
    assert snapshot is not None

    snapshot.markets[0].quotes["phantom-provider.com"] = ()

    fresh = store.get_event_snapshot("evt-1")
    assert fresh is not None
    assert "phantom-provider.com" not in fresh.markets[0].quotes


def test_snapshot_odds_raw_is_deep_copied_from_payload(
    store: MarketStateStore,
) -> None:
    """Scenario: the caller mutates a nested ``odds_raw`` payload after ingest.

    Invariant: the store deep-copies ``odds_raw`` at ingest time, so mutating
    the caller's original object never alters snapshot contents.
    Failure modes: shallow retention would let upstream payload mutation
    rewrite already-ingested odds history.
    """
    payload: Dict[str, Any] = {"price": 1.5}
    market = CanonicalMarket(
        canonical_type=CanonicalMarketType.MONEYLINE,
        raw_label="Match Winner",
        outcomes=[_outcome("Home", 1.5, odds_raw=payload)],
    )
    store.register_event("bookmaker-a.com", "raw-a", event_id="evt-1")
    store.ingest_quotes("bookmaker-a.com", "raw-a", [market])
    snapshot = store.get_event_snapshot("evt-1")
    assert snapshot is not None

    payload["price"] = 999.0

    stored = snapshot.markets[0].quotes["bookmaker-a.com"][0]
    assert stored.odds_raw == {"price": 1.5}


# ---------------------------------------------------------------------------
# 5. Deterministic TTL and eviction lifecycle
# ---------------------------------------------------------------------------


def test_evict_by_kickoff_grace_period(store: MarketStateStore, fake_clock: FakeClock) -> None:
    """Scenario: clock advances past ``start_time + start_time_grace_seconds``.

    Invariant: the event is evicted exactly when the kickoff grace window
    elapses; before that point eviction is a no-op. The update-TTL rule is
    neutralized with a huge TTL so only the kickoff rule can trigger.
    Failure modes: premature eviction or failure to evict finished events
    would either drop live data or leak memory.
    """
    kickoff_iso = fake_clock.iso_now()
    store.register_event(
        "bookmaker-a.com", "raw-a", event_id="evt-1", start_time=kickoff_iso
    )

    assert store.evict(update_ttl_seconds=10**9) == ()

    fake_clock.advance(3600.0)
    assert store.evict(update_ttl_seconds=10**9) == ()

    fake_clock.advance(1.0)
    assert store.evict(update_ttl_seconds=10**9) == ("evt-1",)
    assert store.list_event_ids() == ()


def test_evict_by_update_ttl_for_event_without_start_time(
    store: MarketStateStore, fake_clock: FakeClock
) -> None:
    """Scenario: event with no parseable ``start_time`` goes quiet.

    Invariant: the update-TTL rule alone evicts the event once
    ``now - last_update`` exceeds ``update_ttl_seconds``; absent metadata can
    never strand data.
    Failure modes: unparseable start times permanently protecting stale events
    would cause unbounded memory growth.
    """
    store.register_event("bookmaker-a.com", "raw-a", event_id="evt-1")

    assert store.evict(update_ttl_seconds=600.0) == ()

    fake_clock.advance(600.0)
    assert store.evict(update_ttl_seconds=600.0) == ()

    fake_clock.advance(1.0)
    assert store.evict(update_ttl_seconds=600.0) == ("evt-1",)


def test_evict_keeps_active_event_while_evicting_stale_one(
    store: MarketStateStore, fake_clock: FakeClock
) -> None:
    """Scenario: one stale and one actively-updated event coexist.

    Invariant: eviction is per-event; refreshing one event's quotes resets its
    TTL so it survives while the untouched event is evicted.
    Failure modes: global (all-or-nothing) eviction would drop live events.
    """
    store.register_event("bookmaker-a.com", "raw-a", event_id="evt-stale")
    store.register_event("bookmaker-b.com", "raw-b", event_id="evt-live")

    fake_clock.advance(500.0)
    store.ingest_quotes(
        "bookmaker-b.com",
        "raw-b",
        [_moneyline_market(1.6, 2.9)],
    )
    fake_clock.advance(500.0)

    evicted = store.evict(update_ttl_seconds=900.0)

    assert evicted == ("evt-stale",)
    assert store.list_event_ids() == ("evt-live",)


def test_evict_purges_reverse_index_entries(store: MarketStateStore) -> None:
    """Scenario: a multi-provider event is evicted.

    Invariant: every ``(provider_domain, native_id)`` entry pointing at the
    evicted canonical id is removed, so reverse lookups return ``None``.
    Failure modes: orphaned index entries would let later ingests resurrect or
    misroute quotes into a deleted event.
    """
    store.register_event("bookmaker-a.com", "raw-a", event_id="evt-1")
    store.register_event("bookmaker-b.com", "raw-b", event_id="evt-1")

    evicted = store.evict(update_ttl_seconds=0.0)

    assert evicted == ("evt-1",)
    assert store.get_event_id_for_provider("bookmaker-a.com", "raw-a") is None
    assert store.get_event_id_for_provider("bookmaker-b.com", "raw-b") is None


# ---------------------------------------------------------------------------
# 6. Defensive input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"provider_domain": ""},
        {"provider_domain": "   "},
        {"provider_event_id": ""},
        {"provider_event_id": "   "},
    ],
)
def test_register_event_rejects_empty_or_whitespace_identifiers(
    store: MarketStateStore, kwargs: Dict[str, str]
) -> None:
    """Scenario: registration with empty or whitespace-only identifiers.

    Invariant: ``ValueError`` is raised and no event or index entry is created.
    Failure modes: accepting blank identifiers would poison the reverse index
    with unusable keys.
    """
    with pytest.raises(ValueError):
        store.register_event(
            kwargs.get("provider_domain", "bookmaker-a.com"),
            kwargs.get("provider_event_id", "raw-1"),
        )

    assert store.list_event_ids() == ()


def test_register_event_rejects_empty_explicit_event_id(
    store: MarketStateStore,
) -> None:
    """Scenario: an explicitly supplied canonical id is blank.

    Invariant: ``ValueError`` is raised; no event is created.
    Failure modes: blank canonical ids would create unreachable store entries.
    """
    with pytest.raises(ValueError):
        store.register_event("bookmaker-a.com", "raw-1", event_id="  ")

    assert store.list_event_ids() == ()


@pytest.mark.parametrize(
    "bad_field, bad_value",
    [
        ("line", "2.5"),
        ("line", True),
        ("odds_decimal", "1.5"),
        ("odds_decimal", True),
    ],
)
def test_ingest_rejects_non_numeric_line_and_odds_types(
    store: MarketStateStore, bad_field: str, bad_value: Any
) -> None:
    """Scenario: outcomes carry strings or booleans for numeric fields.

    Invariant: ``TypeError`` is raised for non-``None`` non-numeric values
    (booleans explicitly rejected) and nothing is ingested.
    Failure modes: silent coercion would store corrupted odds (e.g. ``True``
    becoming ``1.0``) and poison arbitrage computations.
    """
    store.register_event("bookmaker-a.com", "raw-a", event_id="evt-1")
    outcome_kwargs: Dict[str, Any] = {"label": "Home", "odds_decimal": 1.5}
    outcome_kwargs[bad_field] = bad_value
    market = CanonicalMarket(
        canonical_type=CanonicalMarketType.MONEYLINE,
        raw_label="Match Winner",
        outcomes=[CanonicalOutcome(**outcome_kwargs)],
    )

    with pytest.raises(TypeError):
        store.ingest_quotes("bookmaker-a.com", "raw-a", [market])

    snapshot = store.get_event_snapshot("evt-1")
    assert snapshot is not None
    assert snapshot.markets == ()


def test_ingest_rejects_unknown_canonical_market_type(
    store: MarketStateStore,
) -> None:
    """Scenario: a market carries a canonical type outside the known set.

    Invariant: ``ValueError`` is raised and no market is stored.
    Failure modes: accepting unknown types would create markets that
    downstream consumers cannot interpret.
    """
    store.register_event("bookmaker-a.com", "raw-a", event_id="evt-1")
    market = CanonicalMarket(
        canonical_type="corners",
        raw_label="Corners",
        outcomes=[_outcome("Over", 1.9)],
    )

    with pytest.raises(ValueError):
        store.ingest_quotes("bookmaker-a.com", "raw-a", [market])

    snapshot = store.get_event_snapshot("evt-1")
    assert snapshot is not None
    assert snapshot.markets == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_time_grace_seconds": -1.0},
        {"update_ttl_seconds": -1.0},
    ],
)
def test_evict_rejects_negative_thresholds(
    store: MarketStateStore, kwargs: Dict[str, float]
) -> None:
    """Scenario: eviction called with a negative grace or TTL threshold.

    Invariant: ``ValueError`` is raised before any state mutation.
    Failure modes: negative thresholds would define inverted staleness logic
    and evict healthy events.
    """
    store.register_event("bookmaker-a.com", "raw-a", event_id="evt-1")

    with pytest.raises(ValueError):
        store.evict(**kwargs)

    assert store.list_event_ids() == ("evt-1",)


def test_get_event_snapshot_returns_none_for_unknown_event(
    store: MarketStateStore,
) -> None:
    """Scenario: snapshot requested for an unknown or blank event id.

    Invariant: unknown ids return ``None``; blank ids raise ``ValueError``.
    Failure modes: raising on unknown ids would complicate consumer polling,
    while accepting blank ids would mask caller bugs.
    """
    assert store.get_event_snapshot("does-not-exist") is None
    with pytest.raises(ValueError):
        store.get_event_snapshot("   ")
