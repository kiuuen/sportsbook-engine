"""Multi-target orchestration of concurrent ingestion pipelines.

This module hosts :class:`MultiTargetOrchestrator`, the supervisor that runs
one :class:`~pipeline.engine.PipelineEngine` per registered sportsbook target
as an isolated asyncio task. All managed engines share a single
:class:`~state.MarketStateStore` and :class:`~resolution.EventClusterResolver`
so cross-provider data converges into one canonical memory space.

Responsibilities
----------------
- Target discovery from the control-plane SQLite database, or an explicit
  caller-supplied domain list.
- Fault-isolated concurrent execution: an unhandled engine failure is logged
  and never propagates to sibling engines.
- Coordinated graceful shutdown: engines are stopped cooperatively, leftover
  tasks are cancelled, and per-engine database connections are closed.
- Read-only state access for downstream consumers via the ``store`` and
  ``resolver`` properties and :meth:`get_live_snapshots`.
"""

import asyncio
import json
import logging
import pathlib
import signal
import sqlite3
from typing import Dict, List, Optional, Tuple

from database_manager import DatabaseManager
from pipeline.engine import PipelineEngine
from resolution import EventClusterResolver
from state import EventSnapshot, MarketStateStore

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = pathlib.Path("./data/control_plane.db")
_DEFAULT_SCHEMA_DIR = pathlib.Path("./data/schemas")

class MultiTargetOrchestrator:
    """Supervises concurrent :class:`PipelineEngine` instances for many targets.

    Args:
        target_domains: Optional explicit list of target domains to orchestrate.
            When omitted or empty, all targets stored in the control-plane
            database are discovered automatically.
        db_manager: Optional :class:`DatabaseManager` used for target discovery.
            Its database path is reused to build one dedicated connection per
            engine (engines close their own database manager on shutdown, so a
            single shared connection must never be handed to them). When
            omitted, the default control-plane database path is used.
        store: Optional shared :class:`~state.MarketStateStore`. When omitted,
            a shared default instance is created so every managed engine
            converges data into the same memory space.
        resolver: Optional shared :class:`~resolution.EventClusterResolver`.
            When omitted, a shared default instance is created.
        schema_dir: Optional directory containing JSON schema files that are
            synced into the control-plane database on every start(). When
            omitted, ``./data/schemas/`` is used.
    """

    def __init__(
        self,
        target_domains: Optional[List[str]] = None,
        db_manager: Optional[DatabaseManager] = None,
        store: Optional[MarketStateStore] = None,
        resolver: Optional[EventClusterResolver] = None,
        schema_dir: Optional[pathlib.Path] = None,
    ) -> None:
        explicit: List[str] = []
        for domain in target_domains or []:
            if isinstance(domain, str) and domain.strip():
                explicit.append(domain.strip())
        # Deduplicate while preserving caller ordering.
        self._explicit_domains: Optional[List[str]] = (
            list(dict.fromkeys(explicit)) if explicit else None
        )
        self._external_db_manager: Optional[DatabaseManager] = db_manager
        self._db_path: pathlib.Path = (
            pathlib.Path(db_manager.db_path) if db_manager is not None else _DEFAULT_DB_PATH
        )
        self._schema_dir: pathlib.Path = (
            pathlib.Path(schema_dir) if schema_dir is not None else _DEFAULT_SCHEMA_DIR
        )

        if store is not None:
            self._store: Optional[MarketStateStore] = store
        else:
            try:
                self._store = MarketStateStore()
            except Exception as err:
                logger.error(
                    f"[MultiTargetOrchestrator] Failed to initialize shared "
                    f"MarketStateStore: {err}"
                )
                self._store = None

        if resolver is not None:
            self._resolver: Optional[EventClusterResolver] = resolver
        else:
            try:
                self._resolver = EventClusterResolver()
            except Exception as err:
                logger.error(
                    f"[MultiTargetOrchestrator] Failed to initialize shared "
                    f"EventClusterResolver: {err}"
                )
                self._resolver = None

        self._engines: Dict[str, PipelineEngine] = {}
        self._engine_dbs: List[DatabaseManager] = []
        self._tasks: List["asyncio.Task"] = []
        self._stop_task: Optional["asyncio.Task"] = None

    # ------------------------------------------------------------------ #
    # State access & reporting
    # ------------------------------------------------------------------ #

    @property
    def store(self) -> Optional[MarketStateStore]:
        """The shared market state store (may be None on init failure)."""
        return self._store

    @property
    def resolver(self) -> Optional[EventClusterResolver]:
        """The shared event cluster resolver (may be None on init failure)."""
        return self._resolver

    def get_live_snapshots(self) -> Tuple[EventSnapshot, ...]:
        """Return frozen snapshots of every live canonical event in the store.

        Returns an empty tuple when the store is unavailable or the snapshot
        read fails, so reporting consumers never crash on store errors.
        """
        if self._store is None:
            return ()
        try:
            return self._store.snapshot_all()
        except Exception as err:
            logger.error(f"[MultiTargetOrchestrator] Snapshot read failed: {err}")
            return ()

    # ------------------------------------------------------------------ #
    # Target discovery
    # ------------------------------------------------------------------ #

    def _sync_schema_files(self) -> None:
        """Sync every JSON schema file from the schema directory into SQLite.

        Each ``*.json`` file found in the schema directory is parsed and
        persisted via :meth:`DatabaseManager.save_schema_dict`, so newly
        dropped schema files participate in the current run without manual
        intervention. Individual file failures are logged and never abort the
        sync of the remaining files.
        """
        if not self._schema_dir.is_dir():
            logger.info(
                f"[MultiTargetOrchestrator] Schema directory "
                f"'{self._schema_dir}' does not exist; skipping schema sync."
            )
            return

        schema_files = sorted(self._schema_dir.glob("*.json"))
        if not schema_files:
            logger.info(
                f"[MultiTargetOrchestrator] No JSON schema files found in "
                f"'{self._schema_dir}'; skipping schema sync."
            )
            return

        db = self._external_db_manager
        owns_db = False
        if db is None:
            try:
                db = DatabaseManager(db_path=self._db_path)
                owns_db = True
            except Exception as err:
                logger.error(
                    f"[MultiTargetOrchestrator] Failed to open control-plane "
                    f"database for schema sync: {err}"
                )
                return
        try:
            synced = 0
            for schema_path in schema_files:
                try:
                    raw_text = schema_path.read_text(encoding="utf-8")
                    payload = json.loads(raw_text)
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
                    logger.error(
                        f"[MultiTargetOrchestrator] Failed to read or parse "
                        f"schema file '{schema_path.name}': {err}"
                    )
                    continue
                if not isinstance(payload, dict):
                    logger.error(
                        f"[MultiTargetOrchestrator] Schema file "
                        f"'{schema_path.name}' is not a JSON object; skipping."
                    )
                    continue
                try:
                    target_id = db.save_schema_dict(payload)
                    synced += 1
                    logger.info(
                        f"[MultiTargetOrchestrator] Synced schema file "
                        f"'{schema_path.name}' (target_id={target_id})."
                    )
                except Exception as err:
                    logger.error(
                        f"[MultiTargetOrchestrator] Failed to persist schema "
                        f"file '{schema_path.name}': {err}"
                    )
            logger.info(
                f"[MultiTargetOrchestrator] Schema sync complete: "
                f"{synced}/{len(schema_files)} file(s) persisted."
            )
        finally:
            if owns_db:
                db.close()

    def _discover_target_domains(self) -> List[str]:
        """Resolve the orchestrator's target domains.

        Returns the explicit domain list when supplied; otherwise queries the
        ``targets`` table of the control-plane database. Discovery failures
        are logged and yield an empty list.
        """
        if self._explicit_domains:
            return list(self._explicit_domains)

        db = self._external_db_manager
        owns_db = False
        if db is None:
            try:
                db = DatabaseManager(db_path=self._db_path)
                owns_db = True
            except Exception as err:
                logger.error(
                    f"[MultiTargetOrchestrator] Failed to open control-plane "
                    f"database for target discovery: {err}"
                )
                return []
        try:
            try:
                conn = db.get_connection()
                rows = conn.execute(
                    "SELECT domain FROM targets ORDER BY id ASC"
                ).fetchall()
            except sqlite3.Error as err:
                logger.error(
                    f"[MultiTargetOrchestrator] Failed to query stored targets: {err}"
                )
                return []
            return [str(row["domain"]) for row in rows if row["domain"]]
        finally:
            if owns_db:
                db.close()

    # ------------------------------------------------------------------ #
    # Engine lifecycle
    # ------------------------------------------------------------------ #

    def _build_engine(self, domain: str) -> Optional[PipelineEngine]:
        """Build one engine with its own database connection; None on failure."""
        try:
            db = DatabaseManager(db_path=self._db_path)
        except Exception as err:
            logger.error(
                f"[MultiTargetOrchestrator] Failed to open database for target "
                f"'{domain}': {err}"
            )
            return None
        try:
            engine = PipelineEngine(
                target_domain=domain,
                db_manager=db,
                store=self._store,
                resolver=self._resolver,
            )
        except Exception as err:
            logger.error(
                f"[MultiTargetOrchestrator] Failed to build engine for target "
                f"'{domain}': {err}"
            )
            db.close()
            return None
        self._engine_dbs.append(db)
        return engine

    async def _run_engine(self, domain: str, engine: PipelineEngine) -> None:
        """Run one engine with fault isolation.

        Unhandled engine exceptions are logged so sibling engines keep
        running; ``asyncio.CancelledError`` always propagates.
        """
        try:
            await engine.start(register_signals=False)
        except asyncio.CancelledError:
            raise
        except Exception as err:
            logger.error(
                f"[MultiTargetOrchestrator] Engine for target '{domain}' "
                f"terminated with an unhandled error: {err}"
            )

    async def start(self, register_signals: bool = True) -> None:
        """Discover targets and run one engine task per target until shutdown.

        Args:
            register_signals: When True (default), SIGINT/SIGTERM handlers are
                registered on the running loop to trigger coordinated
                shutdown. Set to False when an outer supervisor owns signal
                management.
        """
        if self._tasks:
            logger.warning(
                "[MultiTargetOrchestrator] start() ignored: orchestrator is "
                "already running."
            )
            return

        self._sync_schema_files()
        domains = self._discover_target_domains()
        if not domains:
            logger.warning(
                "[MultiTargetOrchestrator] No targets to orchestrate; "
                "start() exiting immediately."
            )
            return

        if register_signals:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, self._handle_shutdown_signal)
                except (NotImplementedError, RuntimeError) as err:
                    logger.debug(
                        f"[MultiTargetOrchestrator] Signal handler for {sig} "
                        f"unavailable: {err}"
                    )

        for domain in domains:
            engine = self._build_engine(domain)
            if engine is None:
                continue
            self._engines[domain] = engine
            self._tasks.append(
                asyncio.create_task(
                    self._run_engine(domain, engine),
                    name=f"orchestrator:{domain}",
                )
            )

        if not self._tasks:
            logger.warning(
                "[MultiTargetOrchestrator] No engines could be built; idling."
            )
            return

        logger.info(
            f"[MultiTargetOrchestrator] Dispatched {len(self._tasks)} engine "
            f"task(s) for targets: {sorted(self._engines)}."
        )
        # Block until every engine has exited (normally or via stop()).
        # return_exceptions=True guarantees one failure never unravels the rest.
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._engines.clear()
        logger.info("[MultiTargetOrchestrator] All engine tasks have exited.")

    def _handle_shutdown_signal(self) -> None:
        """Handle a shutdown signal by scheduling one coordinated stop."""
        logger.info(
            "[MultiTargetOrchestrator] Shutdown signal received; initiating "
            "coordinated shutdown."
        )
        if self._stop_task is None or self._stop_task.done():
            self._stop_task = asyncio.create_task(self.stop())

    async def stop(self) -> None:
        """Trigger cooperative shutdown across all engines and clean up.

        Engines are stopped cooperatively first (each engine's ``stop()``
        sets its shutdown event and returns immediately), then any leftover
        orchestrator tasks are cancelled, and finally per-engine database
        connections are closed defensively (engines normally close their own
        connection in their shutdown path; the second close is a no-op).
        """
        for domain, engine in list(self._engines.items()):
            try:
                await engine.stop()
            except Exception as err:
                logger.warning(
                    f"[MultiTargetOrchestrator] Cooperative stop failed for "
                    f"target '{domain}': {err}"
                )
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        for db in self._engine_dbs:
            db.close()
        self._engine_dbs.clear()
        self._engines.clear()
        logger.info("[MultiTargetOrchestrator] Coordinated shutdown complete.")
