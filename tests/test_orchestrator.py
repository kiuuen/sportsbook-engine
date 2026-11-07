"""Unit tests for :class:`orchestrator.MultiTargetOrchestrator`.

The suite isolates the orchestrator from real infrastructure by patching
``PipelineEngine`` and ``DatabaseManager`` inside the :mod:`orchestrator`
namespace with deterministic test doubles. Store and resolver collaborators
are supplied either as mock sentinels (explicit-sharing scenarios) or as
patched classes (default-singleton scenarios), so no real state store,
resolver, browser session, or SQLite database is ever constructed.

Lifecycle notes mirrored from the real orchestrator contract:
- ``run()`` discovers targets, spawns one task per engine (named
  ``engine:{domain}``), registers signal handlers, and blocks on a shutdown
  event before invoking ``shutdown()``.
- ``shutdown()`` stops engines concurrently, reaps their tasks, clears the
  registries, and closes the discovery database only when the orchestrator
  owns it.
- A crashed engine task is deregistered via a done callback; sibling engines
  are never cancelled.
"""

import asyncio
import contextlib
import pathlib
import unittest.mock
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

import orchestrator as orchestrator_module
from orchestrator import MultiTargetOrchestrator

_POLL_INTERVAL_SECONDS = 0.01
_DEFAULT_TIMEOUT_SECONDS = 5.0
_DEFAULT_DB_PATH = "./data/control_plane.db"


# ---------------------------------------------------------------------- #
# Test doubles
# ---------------------------------------------------------------------- #


class _StubEngine:
    """Deterministic stand-in for :class:`~pipeline.engine.PipelineEngine`.

    Records the collaborator instances and extra keyword arguments it was
    built with, blocks inside ``start()`` on a shutdown event until ``stop()``
    is invoked (mirroring the real engine's cooperative shutdown contract),
    and simulates a fatal crash when ``start_exception`` is set.
    """

    def __init__(
        self,
        target_domain: str,
        db_manager: Any = None,
        store: Any = None,
        resolver: Any = None,
        **extra_kwargs: Any,
    ) -> None:
        self.target_domain = target_domain
        self.db_manager = db_manager
        self.received_store = store
        self.received_resolver = resolver
        self.received_extra_kwargs: Dict[str, Any] = dict(extra_kwargs)
        self._shutdown_event = asyncio.Event()
        self.start_exception: Optional[BaseException] = None
        self.start_count = 0
        self.stop_count = 0
        self.start_completed = False

    async def start(self, register_signals: bool = True) -> None:
        self.start_count += 1
        try:
            if self.start_exception is not None:
                raise self.start_exception
            await self._shutdown_event.wait()
        finally:
            self.start_completed = True

    async def stop(self) -> None:
        self.stop_count += 1
        self._shutdown_event.set()
        await self._shutdown_event.wait()


class _EngineHarness:
    """Builds :class:`_StubEngine` instances and records every creation."""

    def __init__(self) -> None:
        self.engines: List[_StubEngine] = []
        self.failing_domains: Dict[str, BaseException] = {}
        self.unbuildable_domains: Dict[str, BaseException] = {}

    def factory(self, *args: Any, **kwargs: Any) -> _StubEngine:
        domain = str(kwargs.get("target_domain", ""))
        if domain in self.unbuildable_domains:
            raise self.unbuildable_domains[domain]
        engine = _StubEngine(*args, **kwargs)
        if domain in self.failing_domains:
            engine.start_exception = self.failing_domains[domain]
        self.engines.append(engine)
        return engine

    def engine_for(self, domain: str) -> _StubEngine:
        for engine in self.engines:
            if engine.target_domain == domain:
                return engine
        raise AssertionError(f"No stub engine was built for domain '{domain}'.")


def _make_db_factory(
    created: Optional[List[Any]] = None,
    discovery_rows: Optional[List[Dict[str, Any]]] = None,
) -> Callable[..., Any]:
    """Return a ``DatabaseManager`` replacement factory.

    Each call produces a fresh ``MagicMock`` exposing ``db_path``,
    ``get_connection()`` (whose cursor returns ``discovery_rows``), and
    ``close()``. Created instances are appended to ``created`` when supplied.
    """

    def factory(*args: Any, **kwargs: Any) -> Any:
        db = unittest.mock.MagicMock(name="DatabaseManagerInstance")
        db.db_path = pathlib.Path(kwargs.get("db_path", _DEFAULT_DB_PATH))
        connection = unittest.mock.MagicMock(name="sqlite3.Connection")
        connection.execute.return_value.fetchall.return_value = list(
            discovery_rows or []
        )
        db.get_connection.return_value = connection
        if created is not None:
            created.append(db)
        return db

    return factory


class _PatchedCollaborators:
    """Handles to the patched collaborator classes for call-count assertions."""

    def __init__(
        self,
        engine_class: unittest.mock.MagicMock,
        db_class: unittest.mock.MagicMock,
        store_class: unittest.mock.MagicMock,
        resolver_class: unittest.mock.MagicMock,
    ) -> None:
        self.engine_class = engine_class
        self.db_class = db_class
        self.store_class = store_class
        self.resolver_class = resolver_class


@contextlib.contextmanager
def _patched_orchestrator_collaborators(
    harness: _EngineHarness,
    db_factory: Callable[..., Any],
    store_factory: Optional[Callable[..., Any]] = None,
    resolver_factory: Optional[Callable[..., Any]] = None,
):
    """Patch every orchestrator collaborator class in its own namespace.

    Store and resolver classes are always patched so tests never construct
    real singletons; when no explicit factory is given, each construction
    call yields a fresh ``MagicMock`` sentinel via ``return_value``.
    """
    with contextlib.ExitStack() as stack:
        engine_class = stack.enter_context(
            unittest.mock.patch.object(
                orchestrator_module,
                "PipelineEngine",
                side_effect=harness.factory,
            )
        )
        db_class = stack.enter_context(
            unittest.mock.patch.object(
                orchestrator_module, "DatabaseManager", side_effect=db_factory
            )
        )
        if store_factory is None:
            store_class = stack.enter_context(
                unittest.mock.patch.object(orchestrator_module, "MarketStateStore")
            )
            store_class.return_value = unittest.mock.MagicMock(name="auto_store")
        else:
            store_class = stack.enter_context(
                unittest.mock.patch.object(
                    orchestrator_module,
                    "MarketStateStore",
                    side_effect=store_factory,
                )
            )
        if resolver_factory is None:
            resolver_class = stack.enter_context(
                unittest.mock.patch.object(
                    orchestrator_module, "EventClusterResolver"
                )
            )
            resolver_class.return_value = unittest.mock.MagicMock(
                name="auto_resolver"
            )
        else:
            resolver_class = stack.enter_context(
                unittest.mock.patch.object(
                    orchestrator_module,
                    "EventClusterResolver",
                    side_effect=resolver_factory,
                )
            )
        yield _PatchedCollaborators(engine_class, db_class, store_class, resolver_class)


@contextlib.contextmanager
def _suppressed_signal_handlers():
    """Neutralize signal-handler registration on the running event loop.

    The orchestrator registers SIGINT/SIGTERM handlers unconditionally inside
    ``run()``; replacing the loop's ``add_signal_handler`` with a no-op keeps
    tests hermetic and independent of platform signal support.
    """
    loop = asyncio.get_running_loop()
    original = loop.add_signal_handler
    loop.add_signal_handler = lambda *args, **kwargs: None  # type: ignore[method-assign]
    try:
        yield
    finally:
        loop.add_signal_handler = original  # type: ignore[method-assign]


async def _await_condition(
    condition: Callable[[], bool],
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Poll ``condition`` on the running loop until true or timeout expires."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() >= deadline:
            raise AssertionError("Timed out waiting for expected condition.")
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)


async def _run_until_engines_built(
    orchestrator: MultiTargetOrchestrator,
    harness: _EngineHarness,
    expected_engine_count: int,
) -> "asyncio.Task":
    """Dispatch ``run()`` as a task and wait until all engines are spawned."""
    with _suppressed_signal_handlers():
        run_task = asyncio.create_task(orchestrator.run())
    await _await_condition(lambda: len(harness.engines) == expected_engine_count)
    # Let the spawned engine tasks and done callbacks settle.
    await asyncio.sleep(0)
    return run_task


async def _teardown(
    orchestrator: MultiTargetOrchestrator, run_task: "asyncio.Task"
) -> None:
    """Shut the orchestrator down and await ``run()`` completion, hung-proof."""
    await asyncio.wait_for(orchestrator.shutdown(), _DEFAULT_TIMEOUT_SECONDS)
    await asyncio.wait_for(run_task, _DEFAULT_TIMEOUT_SECONDS)


# ---------------------------------------------------------------------- #
# 1. Shared state verification
# ---------------------------------------------------------------------- #


def test_engines_share_explicit_store_and_resolver_instances() -> None:
    """All engines receive the exact same explicitly supplied store/resolver."""
    explicit_domains = ["alpha.example", "beta.example"]
    shared_store = unittest.mock.MagicMock(name="shared_store")
    shared_resolver = unittest.mock.MagicMock(name="shared_resolver")
    harness = _EngineHarness()

    async def scenario() -> None:
        with _patched_orchestrator_collaborators(harness, _make_db_factory()):
            orchestrator = MultiTargetOrchestrator(
                target_domains=explicit_domains,
                db_manager=unittest.mock.MagicMock(name="external_db"),
                store=shared_store,
                resolver=shared_resolver,
                status_interval=0.0,
            )
            run_task = await _run_until_engines_built(
                orchestrator, harness, len(explicit_domains)
            )
            try:
                assert orchestrator.store is shared_store
                assert orchestrator.resolver is shared_resolver
                for domain in explicit_domains:
                    engine = harness.engine_for(domain)
                    assert engine.received_store is shared_store
                    assert engine.received_resolver is shared_resolver
            finally:
                await _teardown(orchestrator, run_task)

    asyncio.run(scenario())


def test_default_store_and_resolver_created_once_and_shared() -> None:
    """When omitted, one default store/resolver is built and shared by all."""
    explicit_domains = ["alpha.example", "beta.example", "gamma.example"]
    harness = _EngineHarness()

    async def scenario() -> None:
        with _patched_orchestrator_collaborators(harness, _make_db_factory()) as mocks:
            orchestrator = MultiTargetOrchestrator(
                target_domains=explicit_domains, status_interval=0.0
            )
            run_task = await _run_until_engines_built(
                orchestrator, harness, len(explicit_domains)
            )
            try:
                # Exactly one default singleton per collaborator was built.
                assert mocks.store_class.call_count == 1
                assert mocks.resolver_class.call_count == 1
                default_store = mocks.store_class.return_value
                default_resolver = mocks.resolver_class.return_value
                assert orchestrator.store is default_store
                assert orchestrator.resolver is default_resolver
                for domain in explicit_domains:
                    engine = harness.engine_for(domain)
                    assert engine.received_store is default_store
                    assert engine.received_resolver is default_resolver
            finally:
                await _teardown(orchestrator, run_task)

    asyncio.run(scenario())


def test_engine_kwargs_forwarded_to_every_engine() -> None:
    """``engine_kwargs`` entries reach every engine constructor verbatim."""
    domains = ["alpha.example", "beta.example"]
    harness = _EngineHarness()
    forwarded = {"harvest_interval": 11.0, "request_timeout": 7.5}

    async def scenario() -> None:
        with _patched_orchestrator_collaborators(harness, _make_db_factory()):
            orchestrator = MultiTargetOrchestrator(
                target_domains=domains,
                db_manager=unittest.mock.MagicMock(name="external_db"),
                status_interval=0.0,
                engine_kwargs=dict(forwarded),
            )
            run_task = await _run_until_engines_built(
                orchestrator, harness, len(domains)
            )
            try:
                for domain in domains:
                    engine = harness.engine_for(domain)
                    for key, value in forwarded.items():
                        assert engine.received_extra_kwargs.get(key) == value
            finally:
                await _teardown(orchestrator, run_task)

    asyncio.run(scenario())


# ---------------------------------------------------------------------- #
# 2. Target discovery
# ---------------------------------------------------------------------- #


def test_explicit_domain_list_is_used_directly() -> None:
    """Explicit domains are orchestrated verbatim without database queries."""
    explicit_domains = ["alpha.example", "beta.example"]
    harness = _EngineHarness()

    async def scenario() -> None:
        external_db = unittest.mock.MagicMock(name="external_db")
        with _patched_orchestrator_collaborators(harness, _make_db_factory()):
            orchestrator = MultiTargetOrchestrator(
                target_domains=explicit_domains,
                db_manager=external_db,
                status_interval=0.0,
            )
            run_task = await _run_until_engines_built(
                orchestrator, harness, len(explicit_domains)
            )
            try:
                orchestrated_domains = sorted(
                    engine.target_domain for engine in harness.engines
                )
                assert orchestrated_domains == sorted(explicit_domains)
                # Discovery must never have touched the caller-owned database.
                external_db.get_connection.assert_not_called()
            finally:
                await _teardown(orchestrator, run_task)
                # A caller-supplied discovery database is never closed.
                external_db.close.assert_not_called()

    asyncio.run(scenario())


def test_database_discovery_fallback_queries_targets_table() -> None:
    """Without explicit domains, targets are read from the control-plane DB."""
    discovered_rows = [
        {"domain": "alpha.example"},
        {"domain": "beta.example"},
    ]
    harness = _EngineHarness()
    created_dbs: List[Any] = []

    async def scenario() -> None:
        with _patched_orchestrator_collaborators(
            harness,
            _make_db_factory(created=created_dbs, discovery_rows=discovered_rows),
        ):
            orchestrator = MultiTargetOrchestrator(status_interval=0.0)
            run_task = await _run_until_engines_built(
                orchestrator, harness, len(discovered_rows)
            )
            try:
                orchestrated_domains = sorted(
                    engine.target_domain for engine in harness.engines
                )
                assert orchestrated_domains == ["alpha.example", "beta.example"]
                # The first constructed database is the orchestrator-owned
                # discovery connection; shutdown must release it.
                assert created_dbs, "Discovery database was never constructed."
                discovery_db = created_dbs[0]
                discovery_db.get_connection.assert_called()
            finally:
                await _teardown(orchestrator, run_task)
                discovery_db = created_dbs[0]
                discovery_db.close.assert_called()

    asyncio.run(scenario())


def test_explicit_empty_domain_aborts_run_without_engines() -> None:
    """An empty explicit domain is rejected and run() exits without engines."""
    harness = _EngineHarness()

    async def scenario() -> None:
        with _patched_orchestrator_collaborators(harness, _make_db_factory()):
            orchestrator = MultiTargetOrchestrator(
                target_domains=["alpha.example", "   "],
                db_manager=unittest.mock.MagicMock(name="external_db"),
                status_interval=0.0,
            )
            with _suppressed_signal_handlers():
                run_task = asyncio.create_task(orchestrator.run())
            await asyncio.wait_for(run_task, _DEFAULT_TIMEOUT_SECONDS)
        assert harness.engines == []
        assert orchestrator.active_targets == ()

    asyncio.run(scenario())


# ---------------------------------------------------------------------- #
# 3. Fault isolation
# ---------------------------------------------------------------------- #


def test_failing_engine_does_not_disturb_sibling_engines() -> None:
    """A crashed engine is deregistered; sibling engines keep running."""
    healthy_domains = ["healthy-one.example", "healthy-two.example"]
    failing_domain = "broken.example"
    harness = _EngineHarness()
    harness.failing_domains[failing_domain] = RuntimeError("fatal engine crash")
    all_domains = healthy_domains + [failing_domain]

    async def scenario() -> None:
        with _patched_orchestrator_collaborators(harness, _make_db_factory()):
            orchestrator = MultiTargetOrchestrator(
                target_domains=all_domains, status_interval=0.0
            )
            run_task = await _run_until_engines_built(
                orchestrator, harness, len(all_domains)
            )
            try:
                # Wait until the done callback has deregistered the crashed
                # engine (this also guarantees its exception was retrieved).
                await _await_condition(
                    lambda: failing_domain not in orchestrator._engines
                )
                # Sibling engines must still be alive and untouched.
                for domain in healthy_domains:
                    engine = harness.engine_for(domain)
                    assert engine.start_count == 1
                    assert engine.stop_count == 0
                    assert not engine.start_completed
                    assert domain in orchestrator._engines
                    sibling_task = orchestrator._engine_tasks[domain]
                    assert not sibling_task.done()
                assert failing_domain not in orchestrator.active_targets
                assert set(orchestrator.active_targets) == set(healthy_domains)
                # The orchestrator as a whole must still be running.
                assert not run_task.done()
            finally:
                await _teardown(orchestrator, run_task)
                for domain in healthy_domains:
                    assert harness.engine_for(domain).start_completed

    asyncio.run(scenario())


def test_unbuildable_engine_is_skipped_without_aborting_run() -> None:
    """An engine whose construction fails is skipped; remaining targets run."""
    domains = ["buildable.example", "unbuildable.example"]
    harness = _EngineHarness()
    harness.unbuildable_domains["unbuildable.example"] = RuntimeError(
        "engine construction failure"
    )

    async def scenario() -> None:
        with _patched_orchestrator_collaborators(harness, _make_db_factory()):
            orchestrator = MultiTargetOrchestrator(
                target_domains=domains, status_interval=0.0
            )
            run_task = await _run_until_engines_built(
                orchestrator, harness, len(domains) - 1
            )
            try:
                assert [e.target_domain for e in harness.engines] == [
                    "buildable.example"
                ]
                assert set(orchestrator.active_targets) == {"buildable.example"}
                assert not run_task.done()
            finally:
                await _teardown(orchestrator, run_task)

    asyncio.run(scenario())


# ---------------------------------------------------------------------- #
# 4. Coordinated graceful shutdown
# ---------------------------------------------------------------------- #


def test_shutdown_signals_all_engines_and_releases_resources() -> None:
    """``shutdown()`` stops every engine, reaps tasks, and closes its own DB."""
    domains = ["alpha.example", "beta.example", "gamma.example"]
    harness = _EngineHarness()
    created_dbs: List[Any] = []

    async def scenario() -> None:
        with _patched_orchestrator_collaborators(
            harness, _make_db_factory(created=created_dbs)
        ):
            orchestrator = MultiTargetOrchestrator(
                target_domains=domains, status_interval=0.0
            )
            run_task = await _run_until_engines_built(
                orchestrator, harness, len(domains)
            )
            assert not run_task.done()
            await _teardown(orchestrator, run_task)
            for domain in domains:
                engine = harness.engine_for(domain)
                assert engine.stop_count == 1
                assert engine.start_completed
            # The orchestrator-owned discovery database (first created) is
            # closed on shutdown; per-engine databases belong to the engines.
            assert len(created_dbs) == len(domains) + 1
            created_dbs[0].close.assert_called()
            assert orchestrator._engine_tasks == {}
            assert orchestrator._engines == {}
            assert orchestrator.active_targets == ()

    asyncio.run(scenario())


def test_shutdown_is_idempotent_after_run_completed() -> None:
    """Calling ``shutdown()`` again on a quiesced orchestrator is a safe no-op."""
    domains = ["alpha.example"]
    harness = _EngineHarness()

    async def scenario() -> None:
        with _patched_orchestrator_collaborators(harness, _make_db_factory()):
            orchestrator = MultiTargetOrchestrator(
                target_domains=domains, status_interval=0.0
            )
            run_task = await _run_until_engines_built(
                orchestrator, harness, len(domains)
            )
            await _teardown(orchestrator, run_task)
            # A second shutdown must complete without hangs or double-stops.
            await asyncio.wait_for(orchestrator.shutdown(), _DEFAULT_TIMEOUT_SECONDS)
            assert harness.engine_for(domains[0]).stop_count == 1

    asyncio.run(scenario())


def test_shutdown_stops_engines_concurrently() -> None:
    """Engine ``stop()`` calls overlap in time instead of running serially."""
    domains = ["alpha.example", "beta.example", "gamma.example"]
    harness = _EngineHarness()
    stop_intervals: List[Tuple[float, float]] = []

    class _SlowStopEngine(_StubEngine):
        """Stub engine whose ``stop()`` yields before completing."""

        async def stop(self) -> None:  # type: ignore[override]
            loop = asyncio.get_running_loop()
            began = loop.time()
            self.stop_count += 1
            self._shutdown_event.set()
            await asyncio.sleep(0.05)
            stop_intervals.append((began, loop.time()))
            self.start_completed = True

    def slow_factory(*args: Any, **kwargs: Any) -> _SlowStopEngine:
        engine = _SlowStopEngine(*args, **kwargs)
        harness.engines.append(engine)
        return engine

    async def scenario() -> None:
        with _patched_orchestrator_collaborators(harness, _make_db_factory()):
            # Replace the harness factory with the slow-stop variant.
            orchestrator = MultiTargetOrchestrator(
                target_domains=domains, status_interval=0.0
            )
            # Re-patch PipelineEngine with the slow-stop factory for this run.
            with unittest.mock.patch.object(
                orchestrator_module, "PipelineEngine", side_effect=slow_factory
            ):
                with _suppressed_signal_handlers():
                    run_task = asyncio.create_task(orchestrator.run())
                await _await_condition(
                    lambda: len(harness.engines) == len(domains)
                )
                await asyncio.sleep(0)
                await asyncio.wait_for(
                    orchestrator.shutdown(), _DEFAULT_TIMEOUT_SECONDS
                )
                await asyncio.wait_for(run_task, _DEFAULT_TIMEOUT_SECONDS)
        assert len(stop_intervals) == len(domains)
        # Every stop must have begun before the last one finished: proof of
        # concurrent (not serial) graceful termination.
        latest_start = max(interval[0] for interval in stop_intervals)
        earliest_end = min(interval[1] for interval in stop_intervals)
        assert latest_start < earliest_end

    asyncio.run(scenario())


# ---------------------------------------------------------------------- #
# 5. Snapshot access
# ---------------------------------------------------------------------- #


def test_snapshot_all_events_delegates_to_store() -> None:
    """``snapshot_all_events()`` returns the shared store's snapshot tuple."""
    expected_snapshots: Tuple[Any, ...] = (
        unittest.mock.MagicMock(name="snapshot_one"),
        unittest.mock.MagicMock(name="snapshot_two"),
    )
    shared_store = unittest.mock.MagicMock(name="shared_store")
    shared_store.snapshot_all.return_value = expected_snapshots

    orchestrator = MultiTargetOrchestrator(
        target_domains=["alpha.example"],
        db_manager=unittest.mock.MagicMock(name="external_db"),
        store=shared_store,
        resolver=unittest.mock.MagicMock(name="shared_resolver"),
    )
    assert orchestrator.snapshot_all_events() == expected_snapshots
    shared_store.snapshot_all.assert_called_once_with()


def test_get_event_snapshot_delegates_to_store() -> None:
    """``get_event_snapshot()`` delegates one-event reads to the shared store."""
    expected_snapshot = unittest.mock.MagicMock(name="event_snapshot")
    shared_store = unittest.mock.MagicMock(name="shared_store")
    shared_store.get_event_snapshot.return_value = expected_snapshot

    orchestrator = MultiTargetOrchestrator(
        target_domains=["alpha.example"],
        db_manager=unittest.mock.MagicMock(name="external_db"),
        store=shared_store,
        resolver=unittest.mock.MagicMock(name="shared_resolver"),
    )
    assert orchestrator.get_event_snapshot("event-1") is expected_snapshot
    shared_store.get_event_snapshot.assert_called_once_with("event-1")


# ---------------------------------------------------------------------- #
# Constructor validation
# ---------------------------------------------------------------------- #


def test_store_construction_failure_raises_runtime_error() -> None:
    """A failing default store construction aborts construction loudly."""

    def failing_store_factory(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("store bootstrap failure")

    with _patched_orchestrator_collaborators(
        _EngineHarness(),
        _make_db_factory(),
        store_factory=failing_store_factory,
    ) as mocks:
        with pytest.raises(RuntimeError, match="MarketStateStore"):
            MultiTargetOrchestrator(
                target_domains=["alpha.example"],
                db_manager=unittest.mock.MagicMock(name="external_db"),
            )
        # The resolver is never reached after the store failure.
        assert mocks.resolver_class.call_count == 0


def test_resolver_construction_failure_raises_runtime_error() -> None:
    """A failing default resolver construction aborts construction loudly."""

    def failing_resolver_factory(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("resolver bootstrap failure")

    with _patched_orchestrator_collaborators(
        _EngineHarness(),
        _make_db_factory(),
        resolver_factory=failing_resolver_factory,
    ) as mocks:
        with pytest.raises(RuntimeError, match="EventClusterResolver"):
            MultiTargetOrchestrator(
                target_domains=["alpha.example"],
                db_manager=unittest.mock.MagicMock(name="external_db"),
            )
        # The store was built successfully before the resolver failure.
        assert mocks.store_class.call_count == 1


def test_negative_status_interval_is_rejected() -> None:
    """A negative ``status_interval`` is rejected at construction time."""
    with pytest.raises(ValueError, match="status_interval"):
        MultiTargetOrchestrator(
            target_domains=["alpha.example"],
            db_manager=unittest.mock.MagicMock(name="external_db"),
            store=unittest.mock.MagicMock(name="shared_store"),
            resolver=unittest.mock.MagicMock(name="shared_resolver"),
            status_interval=-1.0,
        )
