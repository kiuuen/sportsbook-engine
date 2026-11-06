"""Core ingestion pipeline engine orchestrating harvest and detail workers.

This module hosts the :class:`PipelineEngine`, the lifecycle owner of the
autonomous background ingestion worker. It wires together session management,
failure healing (see :mod:`pipeline.healing`), and the worker loops
(see :mod:`pipeline.workers`), handling signal-driven graceful shutdown.
"""

import asyncio
import logging
import signal
import sqlite3
import sys
from typing import List, Optional

from database_manager import DatabaseManager
from pipeline.healing import FailureHealer
from pipeline.workers import PipelineWorkers
from session_manager import SessionManager
from streaming_executor import StreamingExecutor

logger = logging.getLogger(__name__)


class PipelineEngine:
    """Autonomous background ingestion worker performing deep tree-traversal harvesting.

    The engine continuously:
      1. Polls live fixtures/overview endpoints to harvest active match/event IDs.
      2. Consumes real-time deltas via :class:`StreamingExecutor` when the
         target defines streaming endpoints; otherwise falls back to
         continuous HTTP polling of detail/feed endpoints via
         :class:`ReplayExecutor`.
      3. Auto-heals expired sessions on HTTP 401/403 via
         :class:`~session_manager.SessionManager`, persisting refreshed
         credentials to SQLite before resuming.
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
        request_timeout: Per-request/per-probe timeout in seconds guarding
            against hung network calls.
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
        request_timeout: float = 30.0,
    ) -> None:
        self.target_domain = target_domain
        self.db_manager = db_manager if db_manager else DatabaseManager()
        # The shutdown event must exist before healers/workers reference it.
        self._shutdown_event = asyncio.Event()
        self.session_manager = SessionManager(
            db_manager=self.db_manager,
            base_profile_dir=base_profile_dir,
            port=port,
        )
        self.healer = FailureHealer(
            session_manager=self.session_manager,
            target_domain=target_domain,
            shutdown_event=self._shutdown_event,
            base_backoff=base_backoff,
            max_backoff=max_backoff,
        )
        self.workers = PipelineWorkers(
            target_domain=target_domain,
            db_manager=self.db_manager,
            healer=self.healer,
            shutdown_event=self._shutdown_event,
            harvest_interval=harvest_interval,
            detail_poll_interval=detail_poll_interval,
            request_timeout=request_timeout,
        )
        self._tasks: List["asyncio.Task"] = []

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
        executor = self.workers.build_replay_executor()
        harvest_keys, detail_keys = self.workers.classify_endpoints(executor)

        streaming_executor: Optional[StreamingExecutor] = None
        has_streaming = any(
            str((executor.endpoints.get(key, {}) or {}).get("protocol", "")).lower()
            == "websocket"
            for key in detail_keys
        ) if executor is not None else False
        if has_streaming:
            streaming_executor = self.workers.build_streaming_executor()

        if executor is None:
            logger.error(
                "[PipelineEngine] Cannot start workers without a valid target schema; "
                "engine will idle until shutdown."
            )
        else:
            for key in harvest_keys:
                task = asyncio.create_task(
                    self.workers.harvest_loop([key]), name=f"harvest:{key}"
                )
                self._tasks.append(task)
            for key in detail_keys:
                task = asyncio.create_task(
                    self.workers.detail_loop(key, streaming_executor),
                    name=f"detail:{key}",
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
        logger.info(
            "[PipelineEngine] Shutdown signal received; initiating graceful shutdown."
        )
        self._shutdown_event.set()

    async def stop(self) -> None:
        """Request a graceful shutdown and wait until all workers have stopped."""
        self._handle_shutdown()
        await self._shutdown_event.wait()


if __name__ == "__main__":

    def _select_target_domain() -> Optional[str]:
        """Resolve the target domain from CLI arguments or interactive selection.

        Returns:
            The selected domain, or None when no valid target could be resolved.
        """
        args = sys.argv[1:]
        if args:
            domain = args[0].strip()
            if domain:
                return domain
            print("[!] Empty domain argument provided.", file=sys.stderr)
            return None

        db_manager = DatabaseManager()
        try:
            try:
                conn = db_manager.get_connection()
                rows = conn.execute(
                    "SELECT domain FROM targets ORDER BY id ASC"
                ).fetchall()
            except sqlite3.Error as err:
                print(f"[!] Failed to query stored targets: {err}", file=sys.stderr)
                return None
            domains: List[str] = [str(row["domain"]) for row in rows]
        finally:
            db_manager.close()

        if not domains:
            print(
                "[!] No stored targets found in the control plane database.",
                file=sys.stderr,
            )
            return None

        print("[*] Available targets:")
        for idx, name in enumerate(domains, start=1):
            print(f"    {idx}. {name}")

        try:
            raw = input(f"Select a target [1-{len(domains)}, default=1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[!] Target selection aborted.", file=sys.stderr)
            return None

        if not raw:
            chosen_index = 1
        else:
            try:
                chosen_index = int(raw)
            except ValueError:
                print(f"[!] Invalid selection '{raw}'; aborting.", file=sys.stderr)
                return None

        if not 1 <= chosen_index <= len(domains):
            print(f"[!] Selection out of range: {chosen_index}", file=sys.stderr)
            return None

        return domains[chosen_index - 1]

    def main() -> None:
        domain = _select_target_domain()
        if domain is None:
            sys.exit(1)
        logger.info(f"[PipelineEngine] Using target domain: {domain}")
        engine = PipelineEngine(target_domain=domain)
        try:
            asyncio.run(engine.start())
        except KeyboardInterrupt:
            logger.info(
                "[PipelineEngine] Keyboard interrupt received; shutting down."
            )

    main()
