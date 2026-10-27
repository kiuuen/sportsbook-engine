import asyncio
import logging
import signal
from typing import Any, Dict, List, Optional, Set, Tuple

from database_manager import DatabaseManager
from replay_executor import ReplayExecutor
from streaming_executor import StreamingExecutor
from session_manager import SessionManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class PipelineEngine:
    """Orchestrate the hierarchical real-time telemetry and data ingestion pipeline."""

    def __init__(
        self,
        target_domain: str,
        db_manager: Optional[DatabaseManager] = None,
        base_profile_dir: Optional[str] = None,
        port: int = 9222,
    ) -> None:
        self.target_domain = target_domain
        self.db_manager = db_manager if db_manager else DatabaseManager()
        self.session_manager = SessionManager(
            db_manager=self.db_manager,
            base_profile_dir=base_profile_dir,
            port=port,
        )
        self._active_entity_ids: Dict[str, Set[Any]] = {}
        self._shutdown_event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []

    async def _handle_session_refresh(self, endpoint_key: str) -> bool:
        """Handle session refresh for a target endpoint."""
        try:
            success, message = await self.session_manager.refresh_session(
                target_domain=self.target_domain,
                initial_url=f"https://{self.target_domain}",
            )
            if success:
                logger.info(f"Session refreshed successfully for endpoint: {endpoint_key}")
                return True
            logger.warning(f"Failed to refresh session for endpoint {endpoint_key}: {message}")
            return False
        except Exception as err:
            logger.error(f"Error refreshing session for endpoint {endpoint_key}: {err}")
            return False

    async def _process_endpoint(
        self,
        endpoint_key: str,
        executor: ReplayExecutor,
        streaming_executor: Optional[StreamingExecutor] = None,
    ) -> None:
        """Process an endpoint using the appropriate executor."""
        while not self._shutdown_event.is_set():
            try:
                if streaming_executor:
                    # Try streaming first
                    success = await streaming_executor.probe()
                    if success:
                        logger.info(f"Streaming probe succeeded for endpoint: {endpoint_key}")
                        continue

                # Fall back to replay
                response = executor.execute(endpoint_key)
                if response:
                    # Update active entity IDs
                    self._update_active_entity_ids(endpoint_key, response)
                    logger.info(f"Replay execution succeeded for endpoint: {endpoint_key}")
            except Exception as err:
                if "HTTP 401" in str(err) or "HTTP 403" in str(err):
                    # Handle session expiration
                    if await self._handle_session_refresh(endpoint_key):
                        # Retry after session refresh
                        continue
                logger.error(f"Error processing endpoint {endpoint_key}: {err}")

            # Wait before retrying
            await asyncio.sleep(5.0)

    def _update_active_entity_ids(self, endpoint_key: str, response: Any) -> None:
        """Update active entity IDs based on the endpoint response."""
        if not isinstance(response, dict):
            return

        # Get the endpoint info to determine which parameters it provides
        executor = ReplayExecutor(schema_source=self.target_domain, db_manager=self.db_manager)
        endpoint_info = executor.endpoints.get(endpoint_key, {})
        provided_params = endpoint_info.get("provides_parameters", [])

        # Harvest entity IDs from the response
        harvested: Dict[str, List[Any]] = {}
        executor._harvest_identifiers(response, set(provided_params), harvested)

        # Update active entity IDs
        for param in provided_params:
            if param in harvested:
                self._active_entity_ids[param] = set(harvested[param])

    async def _traverse_dependency_graph(self) -> None:
        """Traverse the dependency graph to discover and maintain active entity IDs."""
        executor = ReplayExecutor(schema_source=self.target_domain, db_manager=self.db_manager)
        dependency_graph = executor.dependency_graph

        # Start with catalog root endpoints
        catalog_roots = [
            ep_key for ep_key, ep_info in executor.endpoints.items()
            if ep_info.get("role") == "catalog_root"
        ]

        for root_key in catalog_roots:
            try:
                response = executor.execute(root_key)
                self._update_active_entity_ids(root_key, response)
            except Exception as err:
                logger.error(f"Error processing catalog root {root_key}: {err}")

        # Process producers for each active entity ID
        producers = dependency_graph.get("producers", {})
        for param, producer_keys in producers.items():
            if param in self._active_entity_ids:
                for producer_key in producer_keys:
                    try:
                        params = {param: next(iter(self._active_entity_ids[param]))}
                        response = executor.execute(producer_key, params)
                        self._update_active_entity_ids(producer_key, response)
                    except Exception as err:
                        logger.error(f"Error processing producer {producer_key} for param {param}: {err}")

    async def _run_pipeline(self) -> None:
        """Run the pipeline to process all endpoints."""
        replay_executor = ReplayExecutor(schema_source=self.target_domain, db_manager=self.db_manager)
        streaming_executor = StreamingExecutor(
            target_domain=self.target_domain,
            db_manager=self.db_manager,
        )

        # Traverse the dependency graph to discover active entity IDs
        await self._traverse_dependency_graph()

        # Create tasks for each endpoint
        for endpoint_key in replay_executor.endpoints:
            task = asyncio.create_task(
                self._process_endpoint(endpoint_key, replay_executor, streaming_executor)
            )
            self._tasks.append(task)

        # Wait for shutdown
        await self._shutdown_event.wait()

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def start(self) -> None:
        """Start the pipeline engine."""
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_shutdown)

        try:
            await self._run_pipeline()
        except asyncio.CancelledError:
            logger.info("Pipeline engine shutdown initiated.")
        except Exception as err:
            logger.error(f"Pipeline engine encountered an error: {err}")
        finally:
            self.db_manager.close()
            logger.info("Pipeline engine shutdown complete.")

    def _handle_shutdown(self) -> None:
        """Handle shutdown signal."""
        logger.info("Shutdown signal received. Initiating graceful shutdown...")
        self._shutdown_event.set()

    async def stop(self) -> None:
        """Stop the pipeline engine."""
        self._handle_shutdown()
        await self._shutdown_event.wait()

if __name__ == "__main__":
    async def main() -> None:
        target_domain = "example.com"
        pipeline_engine = PipelineEngine(target_domain=target_domain)

        try:
            await pipeline_engine.start()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Shutting down...")
            await pipeline_engine.stop()

    asyncio.run(main())
