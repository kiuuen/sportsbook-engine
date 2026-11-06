"""Executor construction and asynchronous worker loops for the ingestion pipeline.

This module hosts the replay/streaming executor builders, the entity-ID
harvesting and parameter-mapping helpers, and the harvest/detail worker loops
extracted from the legacy pipeline engine, keeping lifecycle orchestration,
resilience handling (see :mod:`pipeline.healing`), and worker logic separated.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from database_manager import DatabaseManager
from pipeline.healing import FailureHealer
from replay_executor import ReplayExecutor
from streaming_executor import StreamingExecutor
from odds_normalizer import CanonicalEvent, OddsNormalizer

logger = logging.getLogger(__name__)

# Endpoint roles that act as tree-traversal roots for harvesting active event IDs.
HARVEST_ROLES: Tuple[str, ...] = ("live_overview", "fixtures_list")
# Endpoint roles that carry per-event market/odds payloads.
DETAIL_ROLES: Tuple[str, ...] = ("odds_detail",)
# Parameter names recognized as active match/event identifiers.
EVENT_ID_PARAMS: Tuple[str, ...] = ("event_id", "match_id", "eventid", "matchid")
# Consecutive HTTP 400 failures before an endpoint is quarantined for the session.
CLIENT_ERROR_QUARANTINE_THRESHOLD: int = 3

# Pre-normalized event-ID parameter names for exact-match comparisons.
_EVENT_ID_NORMALIZED: frozenset = frozenset(
    _normalize_param_name(p) for p in EVENT_ID_PARAMS
)

def _normalize_param_name(param: Any) -> str:
    """Normalize a parameter name for exact matching against JSON keys.

    Strips placeholder braces and all non-alphanumeric characters, so
    ``{event_id}``, ``EventId``, and ``event-id`` all normalize to
    ``eventid``.
    """
    return re.sub(r"[^a-z0-9]", "", str(param).lower().replace("{", "").replace("}", ""))

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
        self._endpoint_400_counts: Dict[str, int] = {}
        self._quarantined_endpoints: Set[str] = set()
        try:
            self.normalizer = OddsNormalizer()
        except Exception as err:
            logger.error(
                f"[PipelineWorkers] Failed to initialize OddsNormalizer for "
                f"'{self.target_domain}': {err}"
            )
            self.normalizer = None
        else:
            try:
                self.normalizer.load_field_mappings_from_db(
                    self.target_domain, self.db_manager
                )
            except Exception as err:
                logger.warning(
                    f"[PipelineWorkers] Failed to load field mappings for "
                    f"'{self.target_domain}': {err}"
                )

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

    def _normalize_response(self, response: Any, context_label: str) -> None:
        """Normalize a raw API response and log the canonical extraction yield.

        Normalization failures are logged and swallowed so they never crash
        the worker loops.
        """
        if self.normalizer is None:
            return
        try:
            events: List[CanonicalEvent] = self.normalizer.normalize(response)
        except Exception as err:
            logger.warning(
                f"[PipelineWorkers] Normalization failed for {context_label}: {err}"
            )
            return
        market_count = sum(len(event.markets) for event in events)
        odds_count = sum(
            1
            for event in events
            for market in event.markets
            for outcome in market.outcomes
            if outcome.odds_decimal is not None
        )
        logger.info(
            f"[PipelineWorkers] Normalized {context_label}: "
            f"{len(events)} event(s), {market_count} market(s), "
            f"{odds_count} decimal odds value(s)."
        )

    # ------------------------------------------------------------------ #
    # Entity harvesting and parameter mapping
    # ------------------------------------------------------------------ #

    @classmethod
    def _harvest_declared_params(
        cls,
        node: Any,
        declared_map: Dict[str, str],
        harvested: Dict[str, List[Any]],
    ) -> None:
        """Collect values whose JSON keys exactly match schema-declared parameters.

        Args:
            node: Arbitrary JSON node (dict, list, or scalar) to walk.
            declared_map: Mapping of normalized parameter name -> declared name
                taken from the endpoint's ``provides_parameters``.
            harvested: Output accumulator keyed by declared parameter name.

        Unlike the legacy heuristic scraper, this never infers ownership from a
        generic ``*id`` key suffix, so ``sportId`` values can never leak into the
        ``event_id`` bucket. A bare ``id`` key is accepted only when exactly one
        id-like parameter is declared (unambiguous context).
        """
        if isinstance(node, dict):
            bare_id_target: Optional[str] = None
            id_like = [orig for norm, orig in declared_map.items() if norm.endswith("id")]
            if len(id_like) == 1:
                bare_id_target = id_like[0]
            for key, value in node.items():
                normalized = _normalize_param_name(key)
                target = declared_map.get(normalized)
                if target is None and normalized == "id" and bare_id_target is not None:
                    target = bare_id_target
                if (
                    target is not None
                    and isinstance(value, (str, int))
                    and not isinstance(value, bool)
                ):
                    harvested.setdefault(target, []).append(value)
                cls._harvest_declared_params(value, declared_map, harvested)
        elif isinstance(node, list):
            for item in node:
                cls._harvest_declared_params(item, declared_map, harvested)

    def update_active_entity_ids(
        self, endpoint_key: str, executor: ReplayExecutor, response: Any
    ) -> None:
        """Update typed entity-ID buckets from a response using declared parameters.

        Only parameters the endpoint explicitly declares in
        ``provides_parameters`` are harvested, each into its own bucket, so
        ``sport_id`` and ``event_id`` namespaces never cross-contaminate.
        """
        if not isinstance(response, dict):
            return
        endpoint_info = executor.endpoints.get(endpoint_key, {})
        provided_params = endpoint_info.get("provides_parameters", [])
        if not isinstance(provided_params, list):
            provided_params = []
        declared_map = {
            _normalize_param_name(p): str(p)
            for p in provided_params
            if isinstance(p, str) and _normalize_param_name(p)
        }
        if not declared_map:
            return
        harvested: Dict[str, List[Any]] = {}
        self._harvest_declared_params(response, declared_map, harvested)
        for param, values in harvested.items():
            if values:
                self._active_entity_ids.setdefault(param, set()).update(values)

    def harvest_event_ids(
        self, executor: ReplayExecutor, endpoint_key: str, response: Any
    ) -> int:
        """Harvest event/match identifiers declared by the endpoint's schema.

        Only ``provides_parameters`` entries that normalize to a known event-ID
        parameter are collected, using exact key matching. Endpoints that do not
        declare an event-ID parameter (e.g., catalog roots providing ``sport_id``)
        contribute nothing, preventing sport/category IDs from being stored as
        match identifiers.
        """
        endpoint_info = executor.endpoints.get(endpoint_key, {})
        provided = endpoint_info.get("provides_parameters", [])
        if not isinstance(provided, list):
            provided = []
        event_params = [
            str(p)
            for p in provided
            if isinstance(p, str) and _normalize_param_name(p) in _EVENT_ID_NORMALIZED
        ]
        if not event_params:
            return 0
        declared_map = {_normalize_param_name(p): p for p in event_params}
        harvested: Dict[str, List[Any]] = {}
        self._harvest_declared_params(response, declared_map, harvested)
        new_count = 0
        for param, values in harvested.items():
            bucket = self._active_entity_ids.setdefault(param, set())
            before = len(bucket)
            bucket.update(values)
            new_count += len(bucket) - before
        return new_count

    def resolve_endpoint_params(
        self, executor: ReplayExecutor, endpoint_key: str
    ) -> Dict[str, Any]:
        """Build explicit request parameters from typed entity-ID buckets.

        Only schema-declared ``dynamic_parameters`` are populated, each strictly
        from its own bucket. Any parameter still missing is delegated to
        :meth:`ReplayExecutor._resolve_dynamic_params`, which chains through the
        stored dependency graph (declared producers first, then catalog-root
        roles), enabling hierarchical fetches such as
        ``catalog_root -> sport_id -> fixtures_list -> event_id``.
        """
        endpoint_info = executor.endpoints.get(endpoint_key, {})
        required = endpoint_info.get("dynamic_parameters", [])
        if not isinstance(required, list):
            required = []
        explicit: Dict[str, Any] = {}
        for param in required:
            if not isinstance(param, str) or not param:
                continue
            values = self._active_entity_ids.get(param) or self._active_entity_ids.get(
                _normalize_param_name(param)
            )
            if values:
                explicit[param] = sorted(values, key=lambda v: str(v))[0]
        try:
            return executor._resolve_dynamic_params(endpoint_key, explicit)
        except Exception as err:
            logger.warning(
                f"[PipelineWorkers] Dependency-chain resolution failed for "
                f"'{endpoint_key}'; using explicitly resolved params only: {err}"
            )
            return explicit

    @staticmethod
    def order_harvest_keys(
        executor: ReplayExecutor, harvest_keys: List[str]
    ) -> List[str]:
        """Topologically order harvest roots so parameter producers run first.

        A harvest root A precedes root B when A declares a
        ``provides_parameters`` entry that B requires in ``dynamic_parameters``.
        Cycles or unresolved cross-dependencies fall back to appending the
        remaining keys in their original order.
        """
        if len(harvest_keys) < 2:
            return list(harvest_keys)
        provides_map: Dict[str, Set[str]] = {}
        requires_map: Dict[str, Set[str]] = {}
        for key in harvest_keys:
            info = executor.endpoints.get(key, {})
            provides = info.get("provides_parameters", [])
            requires = info.get("dynamic_parameters", [])
            provides_map[key] = {
                _normalize_param_name(p) for p in provides if isinstance(p, str)
            }
            requires_map[key] = {
                _normalize_param_name(p) for p in requires if isinstance(p, str)
            }
        dependencies: Dict[str, Set[str]] = {}
        for key in harvest_keys:
            dependencies[key] = {
                other
                for other in harvest_keys
                if other != key and provides_map.get(other, set()) & requires_map[key]
            }
        ordered: List[str] = []
        pending = list(harvest_keys)
        while pending:
            ready = [k for k in pending if not (dependencies[k] - set(ordered))]
            if not ready:
                ordered.extend(pending)
                break
            for key in ready:
                ordered.append(key)
                pending.remove(key)
        return ordered

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
                _normalize_param_name(p) in _EVENT_ID_NORMALIZED for p in provides if isinstance(provides, list)
            )
            requires_event = any(
                _normalize_param_name(p) in _EVENT_ID_NORMALIZED for p in requires if isinstance(requires, list)
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
            if _normalize_param_name(param) in _EVENT_ID_NORMALIZED:
                ids = self._active_entity_ids.get(param) or self._active_entity_ids.get(
                    _normalize_param_name(param)
                ) or set()
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
            if isinstance(param, str) and _normalize_param_name(param) in _EVENT_ID_NORMALIZED:
                params[param] = event_id
        return params or None

    def _purge_entity_id(self, event_id: Any) -> None:
        """Remove a dead event ID from every event-ID bucket in active rotation.

        Only buckets whose key normalizes to a known event-ID parameter are
        touched, so unrelated namespaces (e.g. ``sport_id``) can never be
        purged by a coincidental value match.
        """
        for param in list(self._active_entity_ids.keys()):
            if _normalize_param_name(param) in _EVENT_ID_NORMALIZED:
                bucket = self._active_entity_ids[param]
                bucket.discard(event_id)
                if not bucket:
                    del self._active_entity_ids[param]

    def _register_client_error(self, endpoint_key: str, status: int) -> bool:
        """Track consecutive HTTP 400 failures and quarantine at the threshold.

        Returns:
            True when the endpoint has just been quarantined.
        """
        if status != 400:
            return False
        count = self._endpoint_400_counts.get(endpoint_key, 0) + 1
        self._endpoint_400_counts[endpoint_key] = count
        if count >= CLIENT_ERROR_QUARANTINE_THRESHOLD:
            self._quarantined_endpoints.add(endpoint_key)
            logger.warning(
                f"[PipelineWorkers] Endpoint '{endpoint_key}' quarantined for the "
                f"remainder of the session after {count} consecutive HTTP 400 errors."
            )
            return True
        return False

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
                ordered_keys = self.order_harvest_keys(executor, harvest_keys)
            for key in ordered_keys:
                if self._shutdown_event.is_set():
                    break
                try:
                    params = self.resolve_endpoint_params(executor, key)
                    response = await asyncio.wait_for(
                        asyncio.to_thread(executor.execute, key, params),
                        timeout=self.request_timeout,
                    )
                    self.update_active_entity_ids(key, executor, response)
                    self._normalize_response(response, f"harvest endpoint '{key}'")
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

            if endpoint_key in self._quarantined_endpoints:
                logger.debug(
                    f"[PipelineWorkers] Skipping quarantined endpoint '{endpoint_key}'."
                )
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
                        params = self.build_params(required_params, event_id) or {}
                        for param in required_params:
                            if (
                                isinstance(param, str)
                                and param
                                and _normalize_param_name(param) not in _EVENT_ID_NORMALIZED
                                and param not in params
                            ):
                                values = self._active_entity_ids.get(param) or self._active_entity_ids.get(
                                    _normalize_param_name(param)
                                ) or set()
                                if values:
                                    params[param] = sorted(values, key=lambda v: str(v))[0]
                        try:
                            response = await asyncio.wait_for(
                                asyncio.to_thread(executor.execute, endpoint_key, params),
                                timeout=self.request_timeout,
                            )
                            self.update_active_entity_ids(endpoint_key, executor, response)
                            self._normalize_response(
                                response,
                                f"detail endpoint '{endpoint_key}' "
                                f"(event={str(event_id) if event_id is not None else 'static'})",
                            )
                            self.healer.reset_backoff(endpoint_key)
                            self._endpoint_400_counts.pop(endpoint_key, None)
                            label = str(event_id) if event_id is not None else "static"
                            logger.info(
                                f"[PipelineWorkers] Detail poll succeeded for "
                                f"'{endpoint_key}' (event={label})."
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception as err:
                            status = self.healer.extract_http_status(err)
                            if status in (400, 404):
                                if event_id is not None:
                                    self._purge_entity_id(event_id)
                                    logger.warning(
                                        f"[PipelineWorkers] Purging stale event ID "
                                        f"'{event_id}' from '{endpoint_key}' after "
                                        f"HTTP {status}."
                                    )
                                if status == 400 and self._register_client_error(
                                    endpoint_key, status
                                ):
                                    break  # Quarantined; outer loop will skip future cycles.
                                continue  # No backoff: advance to the next event ID.
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
