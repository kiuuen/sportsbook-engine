"""Failure classification, exponential backoff tracking, and session auto-healing.

This module isolates the resilience primitives used by the ingestion pipeline
engine so that worker loops remain free of failure-handling concerns.
"""

import asyncio
import logging
import re
from typing import Dict, Optional

from session_manager import SessionManager

logger = logging.getLogger(__name__)


class FailureHealer:
    """Classifies execution failures, applies backoff, and auto-heals sessions.

    The healer tracks consecutive failures per endpoint key to compute capped
    exponential backoff delays, and delegates credential recovery to a
    :class:`~session_manager.SessionManager` on HTTP 401/403 failures.

    Args:
        session_manager: Session manager used to refresh expired credentials.
        target_domain: Domain name of the target being harvested.
        shutdown_event: Engine-wide event signaling graceful shutdown; used to
            abort sleeps early.
        base_backoff: Base delay in seconds for exponential backoff.
        max_backoff: Maximum backoff delay in seconds.
    """

    def __init__(
        self,
        session_manager: SessionManager,
        target_domain: str,
        shutdown_event: asyncio.Event,
        base_backoff: float = 2.0,
        max_backoff: float = 60.0,
    ) -> None:
        self.session_manager = session_manager
        self.target_domain = target_domain
        self._shutdown_event = shutdown_event
        self.base_backoff = max(float(base_backoff), 0.5)
        self.max_backoff = max(float(max_backoff), self.base_backoff)
        self._endpoint_failures: Dict[str, int] = {}
        self._session_refresh_lock = asyncio.Lock()
        self._session_generation = 0

    @property
    def session_generation(self) -> int:
        """Monotonic counter bumped on every successful session refresh."""
        return self._session_generation

    # ------------------------------------------------------------------ #
    # Failure classification and backoff helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_http_status(err: BaseException) -> Optional[int]:
        """Extract an HTTP status code from an exception message when present."""
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

    @classmethod
    def _is_client_error(cls, err: BaseException) -> bool:
        """Return True when the error represents HTTP 400/404.

        These statuses signal a dead entity ID or an invalid endpoint
        template, not a transient network condition, so they must never
        feed the exponential backoff machinery.
        """
        return cls._extract_http_status(err) in (400, 404)

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
                    f"[FailureHealer] Session refresh raised for '{endpoint_key}': {err}"
                )
                return False
            if success:
                self._session_generation += 1
                logger.info(
                    f"[FailureHealer] Session refreshed for '{endpoint_key}'; "
                    f"workers will reload credentials."
                )
                return True
            logger.warning(
                f"[FailureHealer] Session refresh failed for '{endpoint_key}': {message}"
            )
            return False

    async def _handle_execution_failure(self, endpoint_key: str, err: Exception) -> bool:
        """Classify an execution failure and react appropriately.

        Returns:
            True when the failure was resolved via a session refresh and the
            caller should retry immediately; False otherwise. Client errors
            (HTTP 400/404) return False without registering backoff: they
            indicate a dead entity ID or an invalid endpoint template and are
            expected to be handled by the caller (entity purging/quarantine).
        """
        if self._is_auth_failure(err):
            logger.warning(
                f"[FailureHealer] Auth failure on '{endpoint_key}' "
                f"(HTTP {self._extract_http_status(err)}); triggering session refresh."
            )
            if await self._handle_session_refresh(endpoint_key):
                return True
            self._register_backoff(endpoint_key)
            return False
        if self._is_client_error(err):
            logger.warning(
                f"[FailureHealer] Client error on '{endpoint_key}' "
                f"(HTTP {self._extract_http_status(err)}); entity likely dead or "
                f"endpoint template invalid; no backoff registered."
            )
            return False
        if self._is_rate_limit(err):
            logger.warning(f"[FailureHealer] Rate limit hit on '{endpoint_key}'; backing off.")
        else:
            logger.warning(f"[FailureHealer] Transient error on '{endpoint_key}': {err}")
        self._register_backoff(endpoint_key)
        return False

    def register_backoff(self, endpoint_key: str) -> None:
        """Public wrapper: increment the consecutive failure counter for an endpoint."""
        self._register_backoff(endpoint_key)

    def reset_backoff(self, endpoint_key: str) -> None:
        """Public wrapper: clear the failure counter for an endpoint after a success."""
        self._reset_backoff(endpoint_key)

    def backoff_delay(self, endpoint_key: str) -> float:
        """Public wrapper: current capped exponential backoff delay for an endpoint."""
        return self._backoff_delay(endpoint_key)

    async def sleep_or_shutdown(self, delay: float) -> None:
        """Public wrapper: sleep for ``delay`` seconds unless shutdown is requested."""
        await self._sleep_or_shutdown(delay)

    async def handle_execution_failure(self, endpoint_key: str, err: Exception) -> bool:
        """Public wrapper: classify a failure; True when a retry should happen immediately."""
        return await self._handle_execution_failure(endpoint_key, err)

    @classmethod
    def is_client_error(cls, err: BaseException) -> bool:
        """Public wrapper: True when the error represents HTTP 400/404."""
        return cls._is_client_error(err)

    @staticmethod
    def extract_http_status(err: BaseException) -> Optional[int]:
        """Public wrapper: extract an HTTP status code from an error message."""
        return FailureHealer._extract_http_status(err)
