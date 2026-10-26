"""StreamingExecutor – Replay and probe WebSocket streaming endpoints.

This module provides a class to load WebSocket endpoints from the control plane
(SQLite) and optionally from a fallback JSON schema, then probe them by
connecting via ``websockets`` and waiting for the first incoming frame.
Results are logged back into the control plane via :class:`DatabaseManager`.

The implementation follows the conventions in ``CONVENTIONS.md``:
- No placeholders, stubs, or lazy hard‑coding.
- Full type hints and explicit error handling.
- All text is in English.
"""

import asyncio
import json
import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import websockets
from database_manager import DatabaseManager

__all__ = ["StreamingExecutor"]

logger = logging.getLogger(__name__)


class StreamingExecutor:
    """Replay and probe WebSocket streaming endpoints.

    The executor retrieves streaming endpoint definitions and session data from
    the control plane SQLite database. If a target or its data is missing, a
    fallback JSON schema can be supplied. After loading, the :meth:`probe` method
    attempts to connect to each WebSocket, wait for the first frame, and logs
    the outcome back into the database.

    Args:
        target_domain: Domain name of the target to probe.
        db_path: Path to the SQLite control plane database. Ignored if a
            ``db_manager`` is provided.
        db_manager: Optional pre‑configured ``DatabaseManager`` instance.
        fallback_schema_path: Optional path to a JSON file containing a full
            schema (including ``streaming_endpoints`` and ``session``) to use
            when database records are missing.

    Raises:
        ValueError: If required data cannot be located in the database or the
            fallback schema.
        RuntimeError: If the fallback schema cannot be parsed.
    """

    def __init__(
        self,
        target_domain: str,
        db_path: Union[str, pathlib.Path] = "./data/control_plane.db",
        db_manager: Optional[DatabaseManager] = None,
        fallback_schema_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        self.target_domain: str = target_domain
        self.db_path: pathlib.Path = pathlib.Path(db_path)

        # Use provided manager or create a new one
        if db_manager is None:
            self.db_manager: DatabaseManager = DatabaseManager(db_path)
        else:
            self.db_manager = db_manager

        # Load target (create if missing via fallback)
        target_record = self.db_manager.get_target_by_domain(self.target_domain)
        if target_record is None:
            if fallback_schema_path is None:
                raise ValueError(
                    f"Target '{self.target_domain}' not found in database "
                    "and no fallback schema provided."
                )
            schema = self._load_fallback_schema(fallback_schema_path)
            # Insert a minimal target record to obtain an ID
            target_id = self.db_manager.upsert_target(
                domain=self.target_domain,
                provider_family=schema.get("provider_family", "fallback"),
                discovered_at=schema.get(
                    "discovered_at", self.db_manager._utc_now_iso()
                ),
                canonical_pipeline=schema.get("canonical_pipeline", {}),
            )
            logger.info(
                "Created placeholder target record for domain %s with ID %d",
                self.target_domain,
                target_id,
            )
        else:
            target_id = target_record["id"]

        self.target_id: int = target_id

        # Load streaming endpoints
        streaming_endpoints = self.db_manager.get_endpoints_by_target_id(
            self.target_id, category="streaming"
        )
        if not streaming_endpoints and fallback_schema_path is not None:
            schema = self._load_fallback_schema(fallback_schema_path)
            streaming_endpoints = list(schema.get("streaming_endpoints", {}).values())
        if not streaming_endpoints:
            raise ValueError(
                f"No streaming endpoints found for target '{self.target_domain}'."
            )
        self.streaming_endpoints: List[Dict[str, Any]] = streaming_endpoints

        # Load session data (headers and cookies)
        session_record = self.db_manager.get_session_by_target_id(self.target_id)
        if session_record is None and fallback_schema_path is not None:
            schema = self._load_fallback_schema(fallback_schema_path)
            session_data = schema.get("session", {})
        elif session_record is not None:
            session_data = {
                "essential_headers": session_record["essential_headers"],
                "cookies": session_record["cookies"],
            }
        else:
            raise ValueError(
                f"No session data found for target '{self.target_domain}'."
            )
        self.session_headers: Dict[str, str] = session_data.get("essential_headers", {})
        self.session_cookies: Dict[str, str] = session_data.get("cookies", {})

    def _load_fallback_schema(self, schema_path: Union[str, pathlib.Path]) -> Dict[str, Any]:
        """Load and validate a fallback JSON schema.

        Args:
            schema_path: Path to the JSON schema file.

        Returns:
            Parsed JSON dictionary.

        Raises:
            RuntimeError: If the file cannot be read or parsed.
        """
        path = pathlib.Path(schema_path)
        if not path.is_file():
            raise RuntimeError(f"Fallback schema file not found: {path}")
        try:
            with path.open("r", encoding="utf-8") as f:
                schema = json.load(f)
        except Exception as exc:
            raise RuntimeError(f"Failed to parse fallback schema: {exc}") from exc
        # Minimal validation
        if not isinstance(schema, dict):
            raise RuntimeError("Fallback schema must be a JSON object")
        return schema

    async def probe(self, timeout: float = 30.0) -> bool:
        """Probe all streaming endpoints.

        For each endpoint a WebSocket connection is established using the stored
        session headers and the endpoint's handshake headers. The method waits
        for the first incoming frame. The result (success/failure) is logged via
        :meth:`DatabaseManager.log_replay_health`.

        Args:
            timeout: Maximum time (seconds) to wait for a connection and the first
                frame per endpoint.

        Returns:
            ``True`` if any endpoint probe succeeded, ``False`` otherwise.
        """
        overall_success = False

        for endpoint in self.streaming_endpoints:
            endpoint_key = endpoint.get("endpoint_key", "unknown")
            ws_url = endpoint.get("template_url")
            if not ws_url:
                logger.warning(
                    "Streaming endpoint %s lacks a WebSocket URL – skipping", endpoint_key
                )
                continue

            # Combine session headers with endpoint‑specific handshake headers
            handshake_headers = {
                **self.session_headers,
                **endpoint.get("handshake_headers", {}),
            }
            # Convert to list of (name, value) tuples for websockets
            extra_headers = list(handshake_headers.items())

            # Cookies are passed as a list of (name, value) tuples
            cookies = list(self.session_cookies.items())

            success = False
            status_code = 0
            message = ""

            try:
                async with asyncio.timeout(timeout):
                    async with websockets.connect(
                        ws_url,
                        extra_headers=extra_headers,
                        cookies=cookies,
                    ) as ws:
                        try:
                            # Wait for the first frame
                            msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
                            success = True
                            status_code = 0
                            message = (
                                f"Probe successful, received first frame of length {len(msg)}"
                            )
                        except asyncio.TimeoutError:
                            success = False
                            status_code = 408
                            message = "Probe timed out waiting for first frame"
                        except Exception as recv_err:
                            success = False
                            status_code = 500
                            message = f"Probe error while receiving: {recv_err}"
            except Exception as conn_err:
                # Connection or handshake failure
                success = False
                status_code = 500
                message = f"WebSocket connection failed: {conn_err}"

            # Log the outcome regardless of success
            try:
                self.db_manager.log_replay_health(
                    target_id=self.target_id,
                    endpoint_key=endpoint_key,
                    status="SUCCESS" if success else "FAILURE",
                    status_code=status_code,
                    message=message,
                    response_preview="",
                )
            except Exception as log_err:
                logger.error(
                    "Failed to log probe result for endpoint %s: %s",
                    endpoint_key,
                    log_err,
                )

            if success:
                overall_success = True
                # Continue probing other endpoints to log all results

        return overall_success
