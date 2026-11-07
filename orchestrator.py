"""Central multi-tenant coordinator supervising concurrent ingestion pipelines.

This module hosts :class:`MultiTargetOrchestrator`, the process-level owner of
global signals and of one :class:`~pipeline.engine.PipelineEngine` task per
sportsbook target. All managed engines share the exact same
:class:`~state.MarketStateStore` and :class:`~resolution.EventClusterResolver`
singletons so multi-provider odds converge into one canonical memory space.

Fault isolation guarantees:
- Each engine runs in its own asyncio task with its own ``DatabaseManager``
  connection (``PipelineEngine.start`` closes its manager on exit, so sharing
  one connection across engines is unsafe).
- An engine that dies unexpectedly is logged and deregistered; sibling
  engines are never cancelled or disrupted.
- Browser-session healing is fully delegated to each engine's internal
  :class:`~pipeline.healing.FailureHealer`; the orchestrator never touches
  sessions directly.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sqlite3
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from database_manager import DatabaseManager
from pipeline.engine import PipelineEngine
from resolution import EventClusterResolver
from state import EventSnapshot, MarketStateStore

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "./data/control_plane.db"
DEFAULT_STATUS_INTERVAL = 60.0


class MultiTargetOrchestrator:
    """Supervises one ingestion pipeline engine per sportsbook target.

    Args:
        target_domains: Explicit target domains to supervise. When ``None``,
            all targets stored in the control-plane database are discovered.
        db_path: Path to the SQLite control-plane database.
        db_manager: Optional pre-built :class:`DatabaseManager` used **only**
            for target discovery. The orchestrator closes it on shutdown only
            when it constructed it itself.
        store: Optional shared :class:`MarketStateStore` singleton; a default
            instance is created when omitted.
        resolver: Optional shared :class:`EventClusterResolver` singleton; a
            default instance is created when omitted.
        status_interval: Seconds between periodic status reports. A value
            of ``0`` disables the status reporter entirely.
        engine_kwargs: Optional extra keyword arguments forwarded verbatim to
            every :class:`PipelineEngine` constructor (e.g. tuning intervals).

    Raises:
        RuntimeError: If the shared store or resolver singletons cannot be
            constructed, since the orchestrator cannot function without them.
    """

    def __init__(
        self,
        target_domains: Optional[Sequence[str]] = None,
        db_path: str = DEFAULT_DB_PATH,
        db_manager: Optional[DatabaseManager] = None,
        store: Optional[MarketStateStore] = None,
        resolver: Optional[EventClusterResolver] = None,
        status_interval: float = DEFAULT_STATUS_INTERVAL,
        engine_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if status_interval < 0:
            raise ValueError("status_interval must be non-negative.")
        self._db_path = str(db_path)
        self._explicit_domains: Optional[List[str]] = (
            [str(d) for d in target_domains] if target_domains is not None else None
        )
        self._owns_discovery_db = db_manager is None
        self._discovery_db = db_manager if db_manager is not None else DatabaseManager(db_path=self._db_path)
        self._status_interval = float(status_interval)
        self._engine_kwargs: Dict[str, Any] = dict(engine_kwargs) if engine_kwargs else {}

        if store is not None:
            self.store: MarketStateStore = store
        else:
            try:
                self.store = MarketStateStore()
            except Exception as err:
                raise RuntimeError(f"[Orchestrator] Failed to initialize shared MarketStateStore: {err}") from err
        if resolver is not None:
            self.resolver: EventClusterResolver = resolver
        else:
            try:
                self.resolver = EventClusterResolver()
            except Exception as err:
                raise RuntimeError(f"[Orchestrator] Failed to initialize shared EventClusterResolver: {err}") from err

        self._engines: Dict[str, PipelineEngine] = {}
        self._engine_tasks: Dict[str, "asyncio.Task"] = {}
        self._status_task: Optional["asyncio.Task"] = None
        self._shutdown_event = asyncio.Event()

    # ------------------------------------------------------------------ #
    # Target discovery
    # ------------------------------------------------------------------ #

    def _discover_targets(self) -> List[str]:
        """Resolve the supervised target list (explicit or from the database).

        Returns:
            Deduplicated, order-preserving list of target domains.

        Raises:
            RuntimeError: When database discovery fails at the SQLite level.
        """
        if self._explicit_domains is not None:
            domains: List[str] = []
            seen: set = set()
            for raw in self._explicit_domains:
                normalized = raw.strip()
                if not normalized:
                    raise ValueError("Target domains must be non-empty strings.")
                key = normalized.lower()
                if key not in seen:
                    seen.add(key)
                    domains.append(normalized)
            return domains

        try:
            conn = self._discovery_db.get_connection()
            rows = conn.execute("SELECT domain FROM targets ORDER BY id ASC").fetchall()
        except sqlite3.Error as err:
            raise RuntimeError(f"[Orchestrator] Failed to discover targets from the control plane database: {err}") from err
        return [str(row["domain"]) for row in rows]

    # ------------------------------------------------------------------ #
    # Engine construction and supervision
    # ------------------------------------------------------------------ #

    def _build_engine(self, domain: str) -> PipelineEngine:
        """Build one engine with its own database connection and shared state.

        Each engine receives a dedicated :class:`DatabaseManager` because
        ``PipelineEngine.start`` closes its manager on exit; sharing a single
        connection would let one engine's shutdown break all siblings. The
        shared store/resolver singletons are injected so every provider's
        quotes converge in the same memory space.
        """
        engine_db = DatabaseManager(db_path=self._db_path)
        return PipelineEngine(
            target_domain=domain,
            db_manager=engine_db,
            store=self.store,
            resolver=self.resolver,
            **self._engine_kwargs,
        )

    def _on_engine_done(self, domain: str, task: "asyncio.Task") -> None:
        """Fault-isolation callback: deregister a finished engine task.

        A crashed engine is logged and removed from the active registry;
        sibling engines continue running untouched.
        """
        self._engine_tasks.pop(domain, None)
        self._engines.pop(domain, None)
        if task.cancelled():
            logger.info(f"[Orchestrator] Engine task for '{domain}' was cancelled.")
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                f"[Orchestrator] Engine for '{domain}' terminated unexpectedly "
                f"(isolated; other targets unaffected): {exc}"
            )
        else:
            logger.info(f"[Orchestrator] Engine for '{domain}' exited cleanly.")

    async def run(self) -> None:
        """Discover targets, spawn one engine task per target, and block.

        Registers global SIGINT/SIGTERM handlers (engines run with
        ``register_signals=False`` so they never collide with these), starts
        the optional status reporter, and awaits the shutdown signal before
        performing coordinated graceful termination.
        """
        try:
            domains = self._discover_targets()
        except (RuntimeError, TypeError, ValueError) as err:
            logger.error(f"[Orchestrator] Target discovery failed: {err}")
            return
        if not domains:
            logger.error(
                "[Orchestrator] No targets to supervise; provide domains as "
                "arguments or persist targets in the control plane database."
            )
            return

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._handle_shutdown)
            except (NotImplementedError, RuntimeError) as err:
                logger.debug(f"[Orchestrator] Signal handler for {sig} unavailable: {err}")

        for domain in domains:
            try:
                engine = self._build_engine(domain)
            except Exception as err:
                logger.error(
                    f"[Orchestrator] Failed to build engine for '{domain}' "
                    f"(skipping target): {err}"
                )
                continue
            self._engines[domain] = engine
            task = asyncio.create_task(
                engine.start(register_signals=False), name=f"engine:{domain}"
            )
            task.add_done_callback(lambda t, d=domain: self._on_engine_done(d, t))
            self._engine_tasks[domain] = task

        if self._status_interval > 0:
            self._status_task = asyncio.create_task(
                self._status_loop(), name="orchestrator-status"
            )

        logger.info(
            f"[Orchestrator] Supervising {len(self._engines)} target engine(s): "
            f"{sorted(self._engines)}."
        )
        try:
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            # Hard cancellation (e.g. KeyboardInterrupt on platforms without
            # signal handlers): cancel engine tasks synchronously, no awaits.
            self._shutdown_event.set()
            for task in self._engine_tasks.values():
                if not task.done():
                    task.cancel()
            raise
        else:
            await self.shutdown()

    # ------------------------------------------------------------------ #
    # Lifecycle and signal handling
    # ------------------------------------------------------------------ #

    def _handle_shutdown(self) -> None:
        """Handle a global shutdown signal by setting the shutdown event."""
        logger.info("[Orchestrator] Shutdown signal received; initiating coordinated shutdown.")
        self._shutdown_event.set()

    async def shutdown(self) -> None:
        """Gracefully stop every active engine and release shared resources.

        Engines are stopped concurrently via ``stop()`` (which lets in-flight
        requests complete), their tasks are awaited, and the orchestrator's
        own discovery database connection is closed when it owns it.
        """
        self._shutdown_event.set()
        if self._status_task is not None and not self._status_task.done():
            self._status_task.cancel()
            self._status_task = None

        engines = list(self._engines.values())
        if engines:
            results = await asyncio.gather(
                *(engine.stop() for engine in engines), return_exceptions=True
            )
            for engine, result in zip(engines, results):
                if isinstance(result, Exception):
                    logger.error(
                        f"[Orchestrator] Graceful stop failed for "
                        f"'{engine.target_domain}': {result}"
                    )
        if self._engine_tasks:
            await asyncio.gather(*self._engine_tasks.values(), return_exceptions=True)
            self._engine_tasks.clear()
        self._engines.clear()

        if self._owns_discovery_db:
            self._discovery_db.close()
        logger.info("[Orchestrator] Coordinated shutdown complete.")

    # ------------------------------------------------------------------ #
    # Status reporting and downstream access
    # ------------------------------------------------------------------ #

    async def _status_loop(self) -> None:
        """Periodically log aggregate market-state statistics.

        Sleeps shutdown-aware by waiting on the shutdown event with a timeout,
        so a shutdown signal interrupts the sleep immediately.
        """
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=self._status_interval
                )
                break  # Shutdown event set.
            except asyncio.TimeoutError:
                pass
            self._log_status_snapshot()

    def _log_status_snapshot(self) -> None:
        """Emit one aggregate status report; never raises."""
        try:
            snapshots = self.store.snapshot_all()
        except Exception as err:
            logger.warning(f"[Orchestrator] Status snapshot failed: {err}")
            return
        total_events = len(snapshots)
        total_markets = sum(len(snapshot.markets) for snapshot in snapshots)
        multi_provider_events = sum(
            1 for snapshot in snapshots if len(snapshot.providers) > 1
        )
        cluster_count = self.resolver.cluster_count if self.resolver is not None else 0
        logger.info(
            f"[Orchestrator] Status: active_events={total_events}, "
            f"total_markets={total_markets}, "
            f"multi_provider_overlap={multi_provider_events}, "
            f"resolver_clusters={cluster_count}, "
            f"active_engines={len(self._engines)}."
        )

    def snapshot_all_events(self) -> Tuple[EventSnapshot, ...]:
        """Return frozen snapshots of every live event for downstream consumers.

        Raises:
            RuntimeError: If the shared store is unavailable.
        """
        if self.store is None:
            raise RuntimeError("[Orchestrator] Shared market state store is unavailable.")
        return self.store.snapshot_all()

    def get_event_snapshot(self, event_id: str) -> Optional[EventSnapshot]:
        """Return a frozen snapshot for one canonical event, or ``None``.

        Raises:
            RuntimeError: If the shared store is unavailable.
        """
        if self.store is None:
            raise RuntimeError("[Orchestrator] Shared market state store is unavailable.")
        return self.store.get_event_snapshot(event_id)

    @property
    def active_targets(self) -> Tuple[str, ...]:
        """Domains of engines currently supervised (alive or stopping)."""
        return tuple(self._engines.keys())


# ---------------------------------------------------------------------- #
# CLI entrypoint
# ---------------------------------------------------------------------- #

def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse orchestrator CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-target ingestion pipeline orchestrator."
    )
    parser.add_argument(
        "domains",
        nargs="*",
        help="Target domains to supervise. Omit for interactive selection.",
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"Path to the control-plane SQLite database (default: {DEFAULT_DB_PATH}).",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=DEFAULT_STATUS_INTERVAL,
        help=(
            "Seconds between status reports; 0 disables reporting "
            f"(default: {DEFAULT_STATUS_INTERVAL})."
        ),
    )
    return parser.parse_args(list(argv))


def _select_targets_interactive(db_manager: DatabaseManager) -> Optional[List[str]]:
    """Interactively select one target or all stored targets.

    Returns:
        Selected domains, or ``None`` when selection was aborted or no
        targets exist.
    """
    try:
        conn = db_manager.get_connection()
        rows = conn.execute("SELECT domain FROM targets ORDER BY id ASC").fetchall()
    except sqlite3.Error as err:
        print(f"[!] Failed to query stored targets: {err}", file=sys.stderr)
        return None
    domains: List[str] = [str(row["domain"]) for row in rows]
    if not domains:
        print(
            "[!] No stored targets found in the control plane database.",
            file=sys.stderr,
        )
        return None

    print("[*] Available targets:")
    for idx, name in enumerate(domains, start=1):
        print(f"    {idx}. {name}")
    print("    a. ALL targets")

    try:
        raw = input(f"Select a target [1-{len(domains)}, 'a' for all, Enter=all]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[!] Target selection aborted.", file=sys.stderr)
        return None

    if not raw or raw.lower() in ("a", "all"):
        return domains
    try:
        chosen = int(raw)
    except ValueError:
        print(f"[!] Invalid selection '{raw}'; aborting.", file=sys.stderr)
        return None
    if not 1 <= chosen <= len(domains):
        print(f"[!] Selection out of range: {chosen}", file=sys.stderr)
        return None
    return [domains[chosen - 1]]


def main() -> None:
    """CLI entrypoint for ``python orchestrator.py``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args(sys.argv[1:])
    if args.status_interval < 0:
        print("[!] --status-interval must be non-negative.", file=sys.stderr)
        sys.exit(1)

    cli_db = DatabaseManager(db_path=args.db)
    try:
        if args.domains:
            domains: Optional[List[str]] = [d.strip() for d in args.domains if d.strip()]
            if not domains:
                print("[!] No valid domain arguments provided.", file=sys.stderr)
                sys.exit(1)
        else:
            domains = _select_targets_interactive(cli_db)
            if domains is None:
                sys.exit(1)
    finally:
        cli_db.close()

    orchestrator = MultiTargetOrchestrator(
        target_domains=domains,
        db_path=args.db,
        status_interval=args.status_interval,
    )
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        logger.info("[Orchestrator] Keyboard interrupt received; shutting down.")


if __name__ == "__main__":
    main()
