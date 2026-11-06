"""Executor construction and asynchronous worker loops for the ingestion pipeline.

This module hosts the replay/streaming executor builders, the entity-ID
harvesting and parameter-mapping helpers, and the harvest/detail worker loops
extracted from the legacy pipeline engine, keeping lifecycle orchestration,
resilience handling (see :mod:`pipeline.healing`), and worker logic separated.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from database_manager import DatabaseManager
from pipeline.healing import FailureHealer
from replay_executor import ReplayExecutor
from streaming_executor import StreamingExecutor

logger = logging.getLogger(__name__)

# Endpoint roles that act as tree-traversal roots for harvesting active event IDs.
HARVEST_ROLES: Tuple[str, ...] = ("live_overview", "fixtures_list")
# Endpoint roles that carry per-event market/odds payloads.
DETAIL_ROLES: Tuple[str, ...] = ("odds_detail",)
# Parameter names recognized as active match/event identifiers.
EVENT_ID_PARAMS: Tuple[str, ...] = ("event_id", "match_id", "eventid", "matchid")


class PipelineWorkers:
    """Builds executors and runs harvest/detail worker loops for one target.

    Args:
        target_domain: Domain name of the target to harvest.
        db_manager: Shared :class:`DatabaseManager` instance.
        healer: :class:`~pipeline.healing.FailureHealer` providing backoff
            tracking, session auto-healing, and shutdown-aware sleeps.
        shutdown_event: Engine-wide event signaling graceful shutdown.
        harvest_interval: Seconds between event-ID harvesting cycles.
        detail_poll_interval: Base seconds between detail polling cycles.
        request_timeout: Per-request/per-probe timeout in seconds.
    """

    def __init__(
        self,
        target_domain: str,
        db_manager: DatabaseManager,
        healer: FailureHealer,
        shutdown_event: asyncio.Event,
        harvest_interval: float = 60.0,
        detail_poll_interval: float = 5.0,
        request_timeout: float = 30.0,
    ) -> None:
        self.target_domain = target_domain
        self.db_manager = db_manager
        self.healer = healer
        self._shutdown_event = shutdown_event
        self.harvest_interval = max(float(harvest_interval), 1.0)
        self.detail_poll_interval = max(float(detail_poll_interval), 1.0)
        self.request_timeout = max(float(request_timeout), 1.0)
        self._active_entity_ids: Dict[str, Set[Any]] = {}

    # ------------------------------------------------------------------ #
    # Executor construction
    # ------------------------------------------------------------------ #

    def build_replay_executor(self) -> Optional[ReplayExecutor]:
        """Build a fresh ReplayExecutor, returning None on unrecoverable load errors."""
        try:
            return ReplayExecutor(
                schema_source=self.target_domain, db_manager=self.db_manager
            )
        except Exception as err:
            logger.error(
                f"[PipelineWorkers] Failed to build ReplayExecutor for "
                f"'{self.target_domain}': {err}"
            )
            return None

    def build_streaming_executor(self) -> Optional[StreamingExecutor]:
        """Build a StreamingExecutor, returning None when unavailable.

        Targets without streaming endpoints (HTTP-only) are fully supported:
        construction failures or missing streaming data are logged and the
        engine continues in HTTP-polling mode.
        """
        try:
            return StreamingExecutor(
                target_domain=self.target_domain,
                db_manager=self.db_manager,
            )
        except Exception as err:
            logger.warning(
                f"[PipelineWorkers] StreamingExecutor unavailable for "
                f"'{self.target_domain}' (falling back to HTTP polling): {err}"
            )
            return None

    # ------------------------------------------------------------------ #
    # Entity harvesting and parameter mapping
    # ------------------------------------------------------------------ #

    def update_active_entity_ids(
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

    def harvest_event_ids(
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
    def classify_endpoints(
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

    def snapshot_event_ids(self, required_params: List[str]) -> List[Any]:
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
    def build_params(
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

    async def harvest_loop(self, harvest_keys: List[str]) -> None:
        """Periodically refresh active event IDs from harvest-root endpoints."""
        local_generation = -1
        executor: Optional[ReplayExecutor] = None
        while not self._shutdown_event.is_set():
            if executor is None or local_generation != self.healer.session_generation:
                executor = self.build_replay_executor()
                local_generation = self.healer.session_generation
                if executor is None:
                    await self.healer.sleep_or_shutdown(self.harvest_interval)
                    continue
            for key in harvest_keys:
                if self._shutdown_event.is_set():
                    break
                try:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(executor.execute, key),
                        timeout=self.request_timeout,
                    )
                    self.update_active_entity_ids(key, executor, response)
                    new_ids = self.harvest_event_ids(executor, key, response)
                    self.healer.reset_backoff(key)
                    logger.info(
                        f"[PipelineWorkers] Harvest cycle on '{key}' discovered "
                        f"{new_ids} new event identifier(s)."
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as err:
                    await self.healer.handle_execution_failure(key, err)
            await self.healer.sleep_or_shutdown(self.harvest_interval)

    async def detail_loop(
        self, endpoint_key: str, streaming_executor: Optional[StreamingExecutor]
    ) -> None:
        """Poll or stream a detail endpoint for every active event ID."""
        local_generation = -1
        executor: Optional[ReplayExecutor] = None
        while not self._shutdown_event.is_set():
            if executor is None or local_generation != self.healer.session_generation:
                executor = self.build_replay_executor()
                local_generation = self.healer.session_generation
                if executor is None:
                    await self.healer.sleep_or_shutdown(self.detail_poll_interval)
                    continue

            endpoint_info = executor.endpoints.get(endpoint_key, {})
            protocol = str(endpoint_info.get("protocol", "https")).lower()
            required_params = endpoint_info.get("dynamic_parameters", [])
            if not isinstance(required_params, list):
                required_params = []

            try:
                if protocol == "websocket":
                    if streaming_executor is None:
                        logger.warning(
                            f"[PipelineWorkers] No streaming executor available for "
                            f"'{endpoint_key}'; will retry with backoff."
                        )
                        self.healer.register_backoff(endpoint_key)
                    else:
                        try:
                            success = await asyncio.wait_for(
                                streaming_executor.probe(), timeout=self.request_timeout
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception as err:
                            logger.warning(
                                f"[PipelineWorkers] Streaming handshake disrupted for "
                                f"'{endpoint_key}': {err}"
                            )
                            success = False
                        if success:
                            self.healer.reset_backoff(endpoint_key)
                            logger.info(
                                f"[PipelineWorkers] Streaming probe succeeded for '{endpoint_key}'."
                            )
                        else:
                            logger.warning(
                                f"[PipelineWorkers] Streaming probe failed for "
                                f"'{endpoint_key}'; will retry with backoff."
                            )
                            self.healer.register_backoff(endpoint_key)
                else:
                    event_ids = self.snapshot_event_ids(required_params)
                    if not event_ids:
                        logger.debug(
                            f"[PipelineWorkers] No active event IDs available yet for "
                            f"'{endpoint_key}'; waiting for harvest."
                        )
                    for event_id in event_ids:
                        if self._shutdown_event.is_set():
                            break
                        params = self.build_params(required_params, event_id)
                        try:
                            response = await asyncio.wait_for(
                                asyncio.to_thread(executor.execute, endpoint_key, params),
                                timeout=self.request_timeout,
                            )
                            self.update_active_entity_ids(endpoint_key, executor, response)
                            self.healer.reset_backoff(endpoint_key)
                            label = str(event_id) if event_id is not None else "static"
                            logger.info(
                                f"[PipelineWorkers] Detail poll succeeded for "
                                f"'{endpoint_key}' (event={label})."
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception as err:
                            refreshed = await self.healer.handle_execution_failure(
                                endpoint_key, err
                            )
                            if refreshed:
                                break  # Outer loop rebuilds the executor with new credentials.
                            await self.healer.sleep_or_shutdown(
                                self.healer.backoff_delay(endpoint_key)
                            )
                            if self._shutdown_event.is_set():
                                break
            except asyncio.CancelledError:
                raise
            except Exception as err:
                logger.error(
                    f"[PipelineWorkers] Unexpected error in detail loop for "
                    f"'{endpoint_key}': {err}"
                )
                self.healer.register_backoff(endpoint_key)

            await self.healer.sleep_or_shutdown(
                self.detail_poll_interval + self.healer.backoff_delay(endpoint_key)
            )
