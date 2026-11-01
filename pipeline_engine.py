import asyncio
import logging
import signal
from typing import Any, Dict, List, Optional, Set, Tuple

from database_manager import DatabaseManager
from replay_executor import ReplayExecutor
from session_manager import SessionManager
from streaming_executor import StreamingExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Endpoint roles that act as tree-traversal roots for harvesting active event IDs.
HARVEST_ROLES: Tuple[str, ...] = ("live_overview", "fixtures_list")
# Endpoint roles that carry per-event market/odds payloads.
DETAIL_ROLES: Tuple[str, ...] = ("odds_detail",)
# Parameter names recognized as active match/event identifiers.
EVENT_ID_PARAMS: Tuple[str, ...] = ("event_id", "match_id", "eventid", "matchid")


class PipelineEngine:
    """Autonomous background ingestion worker performing deep tree-traversal harvesting.

    The engine continuously:
      1. Polls live fixtures/overview endpoints to harvest active match/event IDs.
      2. Polls or streams detailed market/odds endpoints for every active event ID.
      3. Auto-heals expired sessions on HTTP 401/403 via :class:`SessionManager`,
         persisting refreshed credentials to SQLite before resuming.
      4. Applies exponential backoff on rate limits and transient network errors.

    Args:
        target_domain: Domain name of the target to harvest.
        db_manager: Optional shared :class:`DatabaseManager` instance.
        base_profile_dir: Optional base directory for browser profiles.
        port: Chrome remote debugging port used by the session manager.
        harvest_interval: Seconds between event-ID harvesting cycles.
        detail_poll_interval: Base seconds between detail polling cycles.
        base_backoff: Base delay in seconds for exponential backoff.
        max_backoff: Maximum backoff delay in seconds.
    """

    def __init__(
        self,
        target_domain: str,
        db_manager: Optional[DatabaseManager] = None,
        base_profile_dir: Optional[str] = None,
        port: int = 9222,
        harvest_interval: float = 60.0,
        detail_poll_interval: float = 5.0,
        base_backoff: float = 2.0,
        max_backoff: float = 60.0,
    ) -> None:
        self.target_domain = target_domain
        self.db_manager = db_manager if db_manager else DatabaseManager()
        self.session_manager = SessionManager(
            db_manager=self.db_manager,
            base_profile_dir=base_profile_dir,
            port=port,
        )
        self.harvest_interval = max(float(harvest_interval), 1.0)
        self.detail_poll_interval = max(float(detail_poll_interval), 1.0)
        self.base_backoff = max(float(base_backoff), 0.5)
        self.max_backoff = max(float(max_backoff), self.base_backoff)

        self._active_entity_ids: Dict[str, Set[Any]] = {}
        self._shutdown_event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []
        self._session_refresh_lock = asyncio.Lock()
        self._session_generation = 0
        self._endpoint_failures: Dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # Failure classification and backoff helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_http_status(err: BaseException) -> Optional[int]:
        """Extract an HTTP status code from an exception message when present."""
        import re

        match = re.search(r"status code (\d+)", str(err))
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    @classmethod
    def _is_auth_failure(cls, err: BaseException) -> bool:
        """Return True when the error represents an authentication/authorization failure."""
        return cls._extract_http_status(err) in (401, 403)

    @classmethod
    def _is_rate_limit(cls, err: BaseException) -> bool:
        """Return True when the error represents an HTTP 429 rate limit."""
        return cls._extract_http_status(err) == 429

    def _register_backoff(self, endpoint_key: str) -> None:
        """Increment the consecutive failure counter for an endpoint."""
        self._endpoint_failures[endpoint_key] = self._endpoint_failures.get(endpoint_key, 0) + 1

    def _reset_backoff(self, endpoint_key: str) -> None:
        """Clear the failure counter for an endpoint after a success."""
        self._endpoint_failures.pop(endpoint_key, None)

    def _backoff_delay(self, endpoint_key: str) -> float:
        """Compute the current capped exponential backoff delay for an endpoint."""
        failures = self._endpoint_failures.get(endpoint_key, 0)
        return min(self.base_backoff * (2 ** failures), self.max_backoff)

    async def _sleep_or_shutdown(self, delay: float) -> None:
        """Sleep for ``delay`` seconds, returning early when shutdown is requested."""
        try:
            await asyncio.wait_for(self._shutdown_event.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass

    # ------------------------------------------------------------------ #
    # Session auto-healing
    # ------------------------------------------------------------------ #

    async def _handle_session_refresh(self, endpoint_key: str) -> bool:
        """Refresh the target session and bump the generation counter on success."""
        async with self._session_refresh_lock:
            try:
                success, message = await self.session_manager.refresh_session(
                    target_domain=self.target_domain,
                    initial_url=f"https://{self.target_domain}",
                )
            except Exception as err:
                logger.error(
                    f"[PipelineEngine] Session refresh raised for '{endpoint_key}': {err}"
                )
                return False
            if success:
                self._session_generation += 1
                logger.info(
                    f"[PipelineEngine] Session refreshed for '{endpoint_key}'; "
                    f"workers will reload credentials."
                )
                return True
            logger.warning(
                f"[PipelineEngine] Session refresh failed for '{endpoint_key}': {message}"
            )
            return False

    async def _handle_execution_failure(self, endpoint_key: str, err: Exception) -> bool:
        """Classify an execution failure and react appropriately.

        Returns:
            True when the failure was resolved via a session refresh and the
            caller should retry immediately; False otherwise (backoff applied).
        """
        if self._is_auth_failure(err):
            logger.warning(
                f"[PipelineEngine] Auth failure on '{endpoint_key}' "
                f"(HTTP {self._extract_http_status(err)}); triggering session refresh."
            )
            if await self._handle_session_refresh(endpoint_key):
                return True
            self._register_backoff(endpoint_key)
            return False
        if self._is_rate_limit(err):
            logger.warning(f"[PipelineEngine] Rate limit hit on '{endpoint_key}'; backing off.")
        else:
            logger.warning(f"[PipelineEngine] Transient error on '{endpoint_key}': {err}")
        self._register_backoff(endpoint_key)
        return False

    # ------------------------------------------------------------------ #
    # Executor construction and entity harvesting
    # ------------------------------------------------------------------ #

    def _build_replay_executor(self) -> Optional[ReplayExecutor]:
        """Build a fresh ReplayExecutor, returning None on unrecoverable load errors."""
        try:
            return ReplayExecutor(
                schema_source=self.target_domain, db_manager=self.db_manager
            )
        except Exception as err:
            logger.error(
                f"[PipelineEngine] Failed to build ReplayExecutor for "
                f"'{self.target_domain}': {err}"
            )
            return None

    def _update_active_entity_ids(
        self, endpoint_key: str, executor: ReplayExecutor, response: Any
    ) -> None:
        """Update active entity IDs from a response based on provided parameters."""
        if not isinstance(response, dict):
            return
        endpoint_info = executor.endpoints.get(endpoint_key, {})
        provided_params = endpoint_info.get("provides_parameters", [])
        if not isinstance(provided_params, list):
            provided_params = []
        harvested: Dict[str, List[Any]] = {}
        executor._harvest_identifiers(response, set(provided_params), harvested)
        for param in provided_params:
            values = harvested.get(param)
            if values:
                self._active_entity_ids[param] = set(values)

    def _harvest_event_ids(
        self, executor: ReplayExecutor, endpoint_key: str, response: Any
    ) -> int:
        """Harvest event/match identifiers from a harvest-root response."""
        harvested: Dict[str, List[Any]] = {}
        executor._harvest_identifiers(response, set(EVENT_ID_PARAMS), harvested)
        new_count = 0
        for param, values in harvested.items():
            bucket = self._active_entity_ids.setdefault(param, set())
            before = len(bucket)
            bucket.update(values)
            new_count += len(bucket) - before
        return new_count

    @staticmethod
    def _classify_endpoints(
        executor: Optional[ReplayExecutor],
    ) -> Tuple[List[str], List[str]]:
        """Split endpoint keys into harvest roots and per-event detail endpoints."""
        if executor is None:
            return [], []
        harvest_keys: List[str] = []
        detail_keys: List[str] = []
        for key, info in executor.endpoints.items():
            provides = info.get("provides_parameters", [])
            requires = info.get("dynamic_parameters", [])
            provides_event = any(
                str(p).lower() in EVENT_ID_PARAMS for p in provides if isinstance(provides, list)
            )
            requires_event = any(
                str(p).lower() in EVENT_ID_PARAMS for p in requires if isinstance(requires, list)
            )
            role = str(info.get("role", ""))
            if role in HARVEST_ROLES or provides_event:
                harvest_keys.append(key)
            elif role in DETAIL_ROLES or requires_event:
                detail_keys.append(key)
        detail_keys = [key for key in detail_keys if key not in harvest_keys]
        return harvest_keys, detail_keys

    def _snapshot_event_ids(self, required_params: List[str]) -> List[Any]:
        """Return the active event IDs relevant to an endpoint's required params.

        Returns an empty list when an event parameter is required but no IDs
        have been harvested yet, and ``[None]`` when no event parameter is
        required (the endpoint should be executed once per cycle).
        """
        if not isinstance(required_params, list):
            required_params = []
        for param in required_params:
            if str(param).lower() in EVENT_ID_PARAMS:
                ids = self._active_entity_ids.get(param, set())
                if not ids:
                    return []
                return sorted(ids, key=lambda value: str(value))
        return [None]

    @staticmethod
    def _build_params(
        required_params: List[str], event_id: Any
    ) -> Optional[Dict[str, Any]]:
        """Build request parameters mapping event-ID parameters to a harvested ID."""
        if event_id is None:
            return None
        params: Dict[str, Any] = {}
        for param in required_params:
            if isinstance(param, str) and param.lower() in EVENT_ID_PARAMS:
                params[param] = event_id
        return params or None

    # ------------------------------------------------------------------ #
    # Worker loops
    # ------------------------------------------------------------------ #

    async def _harvest_loop(self, harvest_keys: List[str]) -> None:
        """Periodically refresh active event IDs from harvest-root endpoints."""
        local_generation = -1
        executor: Optional[ReplayExecutor] = None
        while not self._shutdown_event.is_set():
            if executor is None or local_generation != self._session_generation:
                executor = self._build_replay_executor()
                local_generation = self._session_generation
                if executor is None:
                    await self._sleep_or_shutdown(self.harvest_interval)
                    continue
            for key in harvest_keys:
                if self._shutdown_event.is_set():
                    break
                try:
                    response = await asyncio.to_thread(executor.execute, key)
                    self._update_active_entity_ids(key, executor, response)
                    new_ids = self._harvest_event_ids(executor, key, response)
                    self._reset_backoff(key)
                    logger.info(
                        f"[PipelineEngine] Harvest cycle on '{key}' discovered "
                        f"{new_ids} new event identifier(s)."
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as err:
                    await self._handle_execution_failure(key, err)
            await self._sleep_or_shutdown(self.harvest_interval)

    async def _detail_loop(
        self, endpoint_key: str, streaming_executor: StreamingExecutor
    ) -> None:
        """Poll or stream a detail endpoint for every active event ID."""
        local_generation = -1
        executor: Optional[ReplayExecutor] = None
        while not self._shutdown_event.is_set():
            if executor is None or local_generation != self._session_generation:
                executor = self._build_replay_executor()
                local_generation = self._session_generation
                if executor is None:
                    await self._sleep_or_shutdown(self.detail_poll_interval)
                    continue

            endpoint_info = executor.endpoints.get(endpoint_key, {})
            protocol = str(endpoint_info.get("protocol", "https")).lower()
            required_params = endpoint_info.get("dynamic_parameters", [])
            if not isinstance(required_params, list):
                required_params = []

            try:
                if protocol == "websocket":
                    try:
                        success = await streaming_executor.probe()
                    except asyncio.CancelledError:
                        raise
                    except Exception as err:
                        logger.warning(
                            f"[PipelineEngine] Streaming handshake disrupted for "
                            f"'{endpoint_key}': {err}"
                        )
                        success = False
                    if success:
                        self._reset_backoff(endpoint_key)
                        logger.info(
                            f"[PipelineEngine] Streaming probe succeeded for '{endpoint_key}'."
                        )
                    else:
                        logger.warning(
                            f"[PipelineEngine] Streaming probe failed for "
                            f"'{endpoint_key}'; will retry with backoff."
                        )
                        self._register_backoff(endpoint_key)
                else:
                    event_ids = self._snapshot_event_ids(required_params)
                    if not event_ids:
                        logger.debug(
                            f"[PipelineEngine] No active event IDs available yet for "
                            f"'{endpoint_key}'; waiting for harvest."
                        )
                    for event_id in event_ids:
                        if self._shutdown_event.is_set():
                            break
                        params = self._build_params(required_params, event_id)
                        try:
                            response = await asyncio.to_thread(
                                executor.execute, endpoint_key, params
                            )
                            self._update_active_entity_ids(endpoint_key, executor, response)
                            self._reset_backoff(endpoint_key)
                            label = str(event_id) if event_id is not None else "static"
                            logger.info(
                                f"[PipelineEngine] Detail poll succeeded for "
                                f"'{endpoint_key}' (event={label})."
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception as err:
                            refreshed = await self._handle_execution_failure(endpoint_key, err)
                            if refreshed:
                                break  # Outer loop rebuilds the executor with new credentials.
                            await self._sleep_or_shutdown(self._backoff_delay(endpoint_key))
                            if self._shutdown_event.is_set():
                                break
            except asyncio.CancelledError:
                raise
            except Exception as err:
                logger.error(
                    f"[PipelineEngine] Unexpected error in detail loop for "
                    f"'{endpoint_key}': {err}"
                )
                self._register_backoff(endpoint_key)

            await self._sleep_or_shutdown(
                self.detail_poll_interval + self._backoff_delay(endpoint_key)
            )

    # ------------------------------------------------------------------ #
    # Lifecycle management
    # ------------------------------------------------------------------ #

    async def _cancel_tasks(self) -> None:
        """Cancel all worker tasks and await their completion."""
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _run_pipeline(self) -> None:
        """Launch harvest and detail worker tasks and wait for shutdown."""
        streaming_executor = StreamingExecutor(
            target_domain=self.target_domain,
            db_manager=self.db_manager,
        )
        executor = self._build_replay_executor()
        harvest_keys, detail_keys = self._classify_endpoints(executor)

        if executor is None:
            logger.error(
                "[PipelineEngine] Cannot start workers without a valid target schema; "
                "engine will idle until shutdown."
            )
        else:
            for key in harvest_keys:
                task = asyncio.create_task(
                    self._harvest_loop([key]), name=f"harvest:{key}"
                )
                self._tasks.append(task)
            for key in detail_keys:
                task = asyncio.create_task(
                    self._detail_loop(key, streaming_executor), name=f"detail:{key}"
                )
                self._tasks.append(task)

        if self._tasks:
            logger.info(
                f"[PipelineEngine] Dispatched {len(self._tasks)} worker task(s) "
                f"({len(harvest_keys)} harvest, {len(detail_keys)} detail)."
            )
        else:
            logger.warning(
                "[PipelineEngine] No harvest or detail endpoints identified; idling."
            )

        await self._shutdown_event.wait()
        await self._cancel_tasks()

    async def start(self) -> None:
        """Start the autonomous ingestion pipeline and block until shutdown."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._handle_shutdown)
            except (NotImplementedError, RuntimeError) as err:
                logger.debug(
                    f"[PipelineEngine] Signal handler for {sig} unavailable: {err}"
                )

        logger.info(
            f"[PipelineEngine] Starting autonomous ingestion pipeline for "
            f"'{self.target_domain}'."
        )
        try:
            await self._run_pipeline()
        except asyncio.CancelledError:
            logger.info("[PipelineEngine] Pipeline engine shutdown initiated.")
        except Exception as err:
            logger.error(f"[PipelineEngine] Pipeline engine encountered an error: {err}")
        finally:
            self.db_manager.close()
            logger.info("[PipelineEngine] Pipeline engine shutdown complete.")

    def _handle_shutdown(self) -> None:
        """Handle a shutdown signal by setting the shutdown event."""
        logger.info("[PipelineEngine] Shutdown signal received; initiating graceful shutdown.")
        self._shutdown_event.set()

    async def stop(self) -> None:
        """Request a graceful shutdown and wait until all workers have stopped."""
        self._handle_shutdown()
        await self._shutdown_event.wait()


if __name__ == "__main__":

    async def main() -> None:
        target_domain = "example.com"
        pipeline_engine = PipelineEngine(target_domain=target_domain)
        try:
            await pipeline_engine.start()
        except KeyboardInterrupt:
            logger.info("[PipelineEngine] Keyboard interrupt received; shutting down.")
            await pipeline_engine.stop()

    asyncio.run(main())
