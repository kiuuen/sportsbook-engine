"""Integration tests for PipelineWorkers, PipelineEngine, EventClusterResolver,
and MarketStateStore.

Collaborator policy (per CONVENTIONS.md):
- ``MarketStateStore`` and ``EventClusterResolver`` are exercised as real
  instances to verify genuine cross-component integration end to end.
- ``OddsNormalizer`` is replaced post-construction by a deterministic stub
  returning prebuilt canonical model objects: normalization itself is out of
  scope here, and no synthetic provider JSON payloads are fabricated.
- ``ReplayExecutor`` is stubbed out for the engine lifecycle test because
  signal-decoupling verification requires no captured target schema.
"""

import asyncio
import signal
import sys
from typing import Any, Callable, Dict, List, Optional

import pytest

from database_manager import DatabaseManager
from odds_normalizer.models import (
    CanonicalEvent,
    CanonicalMarket,
    CanonicalMarketType,
    CanonicalOutcome,
)
from pipeline.engine import PipelineEngine
from pipeline.healing import FailureHealer
from pipeline.workers import PipelineWorkers
from resolution import EventClusterResolver, generate_canonical_event_id
from session_manager import SessionManager
from state import EventSnapshot, MarketStateStore

TARGET_DOMAIN = "integration_test_target.example"


# ------------------------------------------------------------------ #
# Test doubles
# ------------------------------------------------------------------ #

class _StubNormalizer:
    """Deterministic stand-in for ``OddsNormalizer``.

    Returns prebuilt ``CanonicalEvent`` model objects instead of performing
    real payload normalization, or raises a configured error to exercise the
    defensive degradation path.
    """

    def __init__(
        self,
        events: Optional[List[CanonicalEvent]] = None,
        error: Optional[Exception] = None,
    ) -> None:
        self._events: List[CanonicalEvent] = list(events or [])
        self._error: Optional[Exception] = error
        self.received_payloads: List[Any] = []

    def normalize(self, response: Any) -> List[CanonicalEvent]:
        self.received_payloads.append(response)
        if self._error is not None:
            raise self._error
        return list(self._events)


class _MethodRecorder:
    """Wraps a bound method, recording each call before delegating to it."""

    def __init__(self, owner: Any, method_name: str) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._bound: Callable[..., Any] = getattr(owner, method_name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append({"args": args, "kwargs": kwargs})
        return self._bound(*args, **kwargs)


# ------------------------------------------------------------------ #
# Canonical model builders
# ------------------------------------------------------------------ #

def _build_outcome(
    label: str, odds: float, line: Optional[float] = None
) -> CanonicalOutcome:
    return CanonicalOutcome(
        label=label,
        odds_decimal=odds,
        odds_raw=odds,
        odds_format="decimal",
        line=line,
        provider_outcome_id=None,
    )


def _build_market(
    canonical_type: str,
    outcomes: List[CanonicalOutcome],
    raw_label: str = "raw_market",
) -> CanonicalMarket:
    return CanonicalMarket(
        canonical_type=canonical_type,
        raw_label=raw_label,
        outcomes=list(outcomes),
    )


def _build_event(
    event_id: str,
    home: str,
    away: str,
    start_time: str,
    markets: Optional[List[CanonicalMarket]] = None,
) -> CanonicalEvent:
    return CanonicalEvent(
        event_id=event_id,
        home_team=home,
        away_team=away,
        start_time=start_time,
        raw_label=f"{home} vs {away}",
        source_provider=TARGET_DOMAIN,
        league="premier_league",
        sport="football",
        markets=list(markets or []),
    )


# ------------------------------------------------------------------ #
# Component builders
# ------------------------------------------------------------------ #

def _build_components(
    db_manager: DatabaseManager,
) -> Any:
    shutdown_event = asyncio.Event()
    session_manager = SessionManager(db_manager=db_manager)
    healer = FailureHealer(
        session_manager=session_manager,
        target_domain=TARGET_DOMAIN,
        shutdown_event=shutdown_event,
    )
    return shutdown_event, healer


def _build_workers(
    db_manager: DatabaseManager,
    healer: FailureHealer,
    shutdown_event: asyncio.Event,
    store: Optional[MarketStateStore] = None,
    resolver: Optional[EventClusterResolver] = None,
) -> PipelineWorkers:
    return PipelineWorkers(
        target_domain=TARGET_DOMAIN,
        db_manager=db_manager,
        healer=healer,
        shutdown_event=shutdown_event,
        store=store,
        resolver=resolver,
    )


# ------------------------------------------------------------------ #
# 1. Dependency injection and state exposure
# ------------------------------------------------------------------ #

def test_injected_store_and_resolver_are_assigned(tmp_path: Any) -> None:
    async def scenario() -> None:
        db = DatabaseManager(db_path=tmp_path / "db_injected.db")
        try:
            shutdown_event, healer = _build_components(db)
            store = MarketStateStore()
            resolver = EventClusterResolver()
            workers = _build_workers(
                db, healer, shutdown_event, store=store, resolver=resolver
            )
            assert workers.store is store
            assert workers.resolver is resolver
        finally:
            db.close()

    asyncio.run(scenario())


def test_default_store_and_resolver_initialized_when_omitted(tmp_path: Any) -> None:
    async def scenario() -> None:
        db = DatabaseManager(db_path=tmp_path / "db_defaults.db")
        try:
            shutdown_event, healer = _build_components(db)
            workers = _build_workers(db, healer, shutdown_event)
            assert isinstance(workers.store, MarketStateStore)
            assert isinstance(workers.resolver, EventClusterResolver)
        finally:
            db.close()

    asyncio.run(scenario())


def test_engine_properties_delegate_to_workers(tmp_path: Any) -> None:
    async def scenario() -> None:
        db = DatabaseManager(db_path=tmp_path / "db_engine_delegation.db")
        try:
            engine = PipelineEngine(target_domain=TARGET_DOMAIN, db_manager=db)
            assert engine.store is engine.workers.store
            assert engine.resolver is engine.workers.resolver
            assert isinstance(engine.store, MarketStateStore)
            assert isinstance(engine.resolver, EventClusterResolver)
        finally:
            db.close()

    asyncio.run(scenario())


# ------------------------------------------------------------------ #
# 2. Normalization return contract
# ------------------------------------------------------------------ #

def test_normalize_response_returns_canonical_event_list(tmp_path: Any) -> None:
    async def scenario() -> None:
        db = DatabaseManager(db_path=tmp_path / "db_normalize.db")
        try:
            shutdown_event, healer = _build_components(db)
            workers = _build_workers(db, healer, shutdown_event)

            expected = [
                _build_event(
                    "evt-norm-1", "Alpha FC", "Beta United", "2025-06-01T18:00:00Z"
                )
            ]
            workers.normalizer = _StubNormalizer(events=expected)
            result = workers._normalize_response({"harvest": "payload"}, "unit-harvest")
            assert isinstance(result, list)
            assert result == expected
            assert all(isinstance(item, CanonicalEvent) for item in result)

            # Degradation paths must yield empty lists, never None.
            workers.normalizer = None
            assert workers._normalize_response({"k": 1}, "no-normalizer") == []
            workers.normalizer = _StubNormalizer(error=RuntimeError("boom"))
            assert workers._normalize_response({"k": 1}, "failing-normalizer") == []
        finally:
            db.close()

    asyncio.run(scenario())


# ------------------------------------------------------------------ #
# 3. Harvest event registration (resolve -> register -> ingest)
# ------------------------------------------------------------------ #

def test_harvest_cycle_resolves_registers_and_ingests(tmp_path: Any) -> None:
    async def scenario() -> None:
        db = DatabaseManager(db_path=tmp_path / "db_harvest.db")
        try:
            shutdown_event, healer = _build_components(db)
            store = MarketStateStore()
            resolver = EventClusterResolver()
            workers = _build_workers(
                db, healer, shutdown_event, store=store, resolver=resolver
            )

            market = _build_market(
                CanonicalMarketType.MONEYLINE,
                [_build_outcome("home", 2.10), _build_outcome("away", 3.40)],
            )
            event = _build_event(
                "evt-100", "Alpha FC", "Beta United", "2025-06-01T18:00:00Z",
                markets=[market],
            )

            resolve_recorder = _MethodRecorder(resolver, "resolve")
            register_recorder = _MethodRecorder(store, "register_event")
            ingest_recorder = _MethodRecorder(store, "ingest_quotes")
            resolver.resolve = resolve_recorder
            store.register_event = register_recorder
            store.ingest_quotes = ingest_recorder

            workers._process_harvest_events([event])

            # a) resolver.resolve invoked once with provider identity + metadata.
            assert len(resolve_recorder.calls) == 1
            resolve_args = resolve_recorder.calls[0]["args"]
            assert resolve_args[0] == TARGET_DOMAIN
            assert resolve_args[1] == "evt-100"
            assert resolve_args[2] == event.home_team
            assert resolve_args[3] == event.away_team
            assert resolve_args[4] == event.start_time
            assert resolve_args[5] == event.sport

            canonical_id = store.get_event_id_for_provider(TARGET_DOMAIN, "evt-100")
            assert canonical_id is not None
            assert canonical_id == generate_canonical_event_id(
                event.home_team, event.away_team, event.start_time, event.sport
            )

            # b) register_event invoked under the resolved canonical id.
            assert len(register_recorder.calls) == 1
            assert register_recorder.calls[0]["args"] == (TARGET_DOMAIN, "evt-100")
            register_kwargs = register_recorder.calls[0]["kwargs"]
            assert register_kwargs.get("event_id") == canonical_id
            assert register_kwargs.get("home_team") == event.home_team
            assert register_kwargs.get("away_team") == event.away_team

            # c) embedded harvest markets ingested via ingest_quotes.
            assert len(ingest_recorder.calls) == 1
            assert ingest_recorder.calls[0]["args"][:2] == (TARGET_DOMAIN, "evt-100")
            assert ingest_recorder.calls[0]["args"][2] == [market]

            snapshot = store.get_event_snapshot(canonical_id)
            assert snapshot is not None
            assert snapshot.home_team == event.home_team
            assert snapshot.away_team == event.away_team
            moneyline = next(
                m for m in snapshot.markets
                if m.canonical_type == CanonicalMarketType.MONEYLINE
            )
            quotes = moneyline.quotes[TARGET_DOMAIN]
            assert {q.label for q in quotes} == {"home", "away"}
        finally:
            db.close()

    asyncio.run(scenario())


# ------------------------------------------------------------------ #
# 4. Detail quote ingestion reflected in snapshots
# ------------------------------------------------------------------ #

def test_detail_cycle_ingests_quotes_into_snapshot(tmp_path: Any) -> None:
    async def scenario() -> None:
        db = DatabaseManager(db_path=tmp_path / "db_detail.db")
        try:
            shutdown_event, healer = _build_components(db)
            store = MarketStateStore()
            resolver = EventClusterResolver()
            workers = _build_workers(
                db, healer, shutdown_event, store=store, resolver=resolver
            )

            harvest_market = _build_market(
                CanonicalMarketType.MONEYLINE,
                [_build_outcome("home", 2.10), _build_outcome("away", 3.40)],
            )
            harvest_event = _build_event(
                "evt-200", "Alpha FC", "Beta United", "2025-06-01T18:00:00Z",
                markets=[harvest_market],
            )
            workers._process_harvest_events([harvest_event])
            canonical_id = store.get_event_id_for_provider(TARGET_DOMAIN, "evt-200")
            assert canonical_id is not None

            totals_market = _build_market(
                CanonicalMarketType.TOTALS,
                [
                    _build_outcome("over", 1.90, line=2.5),
                    _build_outcome("under", 1.95, line=2.5),
                ],
            )
            detail_event = _build_event(
                "evt-200", "Alpha FC", "Beta United", "2025-06-01T18:00:00Z",
                markets=[totals_market],
            )
            workers._ingest_event_markets(detail_event, "evt-200")

            snapshot: Optional[EventSnapshot] = store.get_event_snapshot(canonical_id)
            assert snapshot is not None
            totals = [
                m for m in snapshot.markets
                if m.canonical_type == CanonicalMarketType.TOTALS and m.line == 2.5
            ]
            assert len(totals) == 1
            quotes = {q.label: q for q in totals[0].quotes[TARGET_DOMAIN]}
            assert set(quotes) == {"over", "under"}
            assert quotes["over"].odds_decimal == pytest.approx(1.90)
            assert quotes["under"].odds_decimal == pytest.approx(1.95)
            # Non-destructive merge: harvest moneyline quotes must survive.
            assert any(
                m.canonical_type == CanonicalMarketType.MONEYLINE
                for m in snapshot.markets
            )
        finally:
            db.close()

    asyncio.run(scenario())


# ------------------------------------------------------------------ #
# 5. Defensive detail ingestion for unregistered events
# ------------------------------------------------------------------ #

def test_detail_ingestion_defensively_registers_unknown_event(tmp_path: Any) -> None:
    async def scenario() -> None:
        db = DatabaseManager(db_path=tmp_path / "db_defensive.db")
        try:
            shutdown_event, healer = _build_components(db)
            store = MarketStateStore()
            resolver = EventClusterResolver()
            workers = _build_workers(
                db, healer, shutdown_event, store=store, resolver=resolver
            )

            market = _build_market(
                CanonicalMarketType.MONEYLINE,
                [_build_outcome("home", 1.75), _build_outcome("away", 2.05)],
            )
            ghost_event = _build_event(
                "evt-ghost", "Gamma SC", "Delta AC", "2025-06-02T20:00:00Z",
                markets=[market],
            )
            # Must not raise KeyError despite never being harvested.
            workers._ingest_event_markets(ghost_event, "evt-ghost")

            canonical_id = store.get_event_id_for_provider(TARGET_DOMAIN, "evt-ghost")
            assert canonical_id is not None
            snapshot = store.get_event_snapshot(canonical_id)
            assert snapshot is not None
            assert snapshot.home_team == "Gamma SC"
            assert snapshot.away_team == "Delta AC"
            moneyline = next(
                m for m in snapshot.markets
                if m.canonical_type == CanonicalMarketType.MONEYLINE
            )
            assert {q.label for q in moneyline.quotes[TARGET_DOMAIN]} == {
                "home", "away"
            }
        finally:
            db.close()

    asyncio.run(scenario())


# ------------------------------------------------------------------ #
# 6. Engine signal decoupling
# ------------------------------------------------------------------ #

def _install_signal_recorder(
    loop: asyncio.AbstractEventLoop,
) -> List[Any]:
    """Replace the loop's ``add_signal_handler`` with a recording spy."""
    registered: List[Any] = []

    def _record(sig: Any, callback: Any, *args: Any, **kwargs: Any) -> None:
        registered.append(sig)

    loop.add_signal_handler = _record  # type: ignore[method-assign]
    return registered


def test_engine_start_without_signal_registration(tmp_path: Any) -> None:
    async def scenario() -> None:
        db = DatabaseManager(db_path=tmp_path / "db_engine_signals.db")
        try:
            engine = PipelineEngine(target_domain=TARGET_DOMAIN, db_manager=db)
            # No captured schema fixture is required: a missing executor makes
            # the engine idle until shutdown, which is all this test needs.
            engine.workers.build_replay_executor = lambda: None  # type: ignore[method-assign]

            loop = asyncio.get_running_loop()
            registered = _install_signal_recorder(loop)

            async def _shutdown_soon() -> None:
                await asyncio.sleep(0.05)
                engine._shutdown_event.set()

            shutdown_task = asyncio.create_task(_shutdown_soon())
            await engine.start(register_signals=False)
            await shutdown_task
            assert registered == []
        finally:
            db.close()

    asyncio.run(scenario())


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="ProactorEventLoop does not support add_signal_handler.",
)
def test_engine_start_with_signal_registration_positive_control(
    tmp_path: Any,
) -> None:
    async def scenario() -> None:
        db = DatabaseManager(db_path=tmp_path / "db_engine_signals_ctrl.db")
        try:
            engine = PipelineEngine(target_domain=TARGET_DOMAIN, db_manager=db)
            engine.workers.build_replay_executor = lambda: None  # type: ignore[method-assign]

            loop = asyncio.get_running_loop()
            registered = _install_signal_recorder(loop)

            async def _shutdown_soon() -> None:
                await asyncio.sleep(0.05)
                engine._shutdown_event.set()

            shutdown_task = asyncio.create_task(_shutdown_soon())
            await engine.start(register_signals=True)
            await shutdown_task
            assert set(registered) == {signal.SIGINT, signal.SIGTERM}
        finally:
            db.close()

    asyncio.run(scenario())
