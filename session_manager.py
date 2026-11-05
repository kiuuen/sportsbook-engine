import asyncio
import concurrent.futures
import logging
import os
from typing import Dict, Optional, Tuple

from browser_orchestrator import ChromeOrchestrator
from cdp_interceptor import CDPNetworkInterceptor, NetworkEvent
from database_manager import DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class SessionManager:
    """Autonomously refresh expired target sessions in SQLite using headless browser orchestration."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        base_profile_dir: Optional[str] = None,
        port: int = 9222,
    ) -> None:
        self.db_manager = db_manager
        self.base_profile_dir = base_profile_dir
        self.port = port
        self._lock = asyncio.Lock()

    async def refresh_session(
        self,
        target_domain: str,
        initial_url: Optional[str] = None,
        timeout: float = 30.0,
    ) -> Tuple[bool, str]:
        """Refresh session for a target domain using headless browser orchestration.

        Args:
            target_domain: The domain to refresh the session for.
            initial_url: Optional initial URL to navigate to.
            timeout: Timeout in seconds for the browser operations.

        Returns:
            A tuple of (success: bool, message: str) indicating the outcome.
        """
        async with self._lock:
            orchestrator = None
            interceptor = None
            try:
                orchestrator = ChromeOrchestrator(
                    profile_name=target_domain,
                    port=self.port,
                    base_profile_dir=self.base_profile_dir,
                    headless=True,
                )
                orchestrator.start(timeout=timeout, initial_url=initial_url)

                ws_url = orchestrator.get_cdp_url()
                if not ws_url:
                    return False, "Failed to obtain CDP WebSocket URL."

                captured_headers: Dict[str, str] = {}
                captured_cookies: Dict[str, str] = {}

                def on_network_event(event: NetworkEvent) -> None:
                    nonlocal captured_headers
                    if event.method == "GET" and event.url.startswith(f"https://{target_domain}"):
                        captured_headers.update(event.request_headers)

                interceptor = CDPNetworkInterceptor(
                    cdp_ws_url=ws_url,
                    on_event_callback=on_network_event,
                )
                await interceptor.start()

                # Wait for the browser to fully load and capture network events
                await asyncio.sleep(timeout)

                captured_cookies = await interceptor.get_cookies(
                    urls=[f"https://{target_domain}"]
                )
                if not captured_cookies:
                    logging.warning(
                        f"[SessionManager] No session cookies found for {target_domain}."
                    )

                target_record = self.db_manager.get_target_by_domain(target_domain)
                if not target_record:
                    return False, f"Target domain {target_domain} not found in database."

                target_id = target_record["id"]
                self.db_manager.upsert_session(
                    target_id=target_id,
                    essential_headers=captured_headers,
                    cookies=captured_cookies,
                )

                return True, f"Session refreshed successfully for {target_domain}."

            except Exception as err:
                logging.error(f"Error refreshing session for {target_domain}: {err}")
                return False, f"Error refreshing session: {err}"
            finally:
                if interceptor:
                    await interceptor.stop()
                if orchestrator:
                    orchestrator.stop()

    def refresh_session_sync(
        self,
        target_domain: str,
        initial_url: Optional[str] = None,
        timeout: float = 30.0,
    ) -> Tuple[bool, str]:
        """Synchronous wrapper for refresh_session.

        If an event loop is already running in the current thread, the
        coroutine is executed in a separate worker thread with its own event
        loop to avoid ``RuntimeError: asyncio.run() cannot be called from a
        running event loop``. Otherwise, standard ``asyncio.run`` is used.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop in this thread: safe to use asyncio.run directly.
            return asyncio.run(self.refresh_session(target_domain, initial_url, timeout))

        # A loop is already running in this thread: delegate to a worker thread.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                asyncio.run, self.refresh_session(target_domain, initial_url, timeout)
            )
            return future.result()

if __name__ == "__main__":
    async def main() -> None:
        print("[*] Starting SessionManager test...")
        db_path = "./data/test_session_manager.db"
        db_manager = DatabaseManager(db_path=db_path)
        session_manager = SessionManager(db_manager=db_manager)

        target_domain = "example.com"
        initial_url = "https://example.com/login"

        try:
            # Seed a minimal dummy target record so get_target_by_domain succeeds.
            db_manager.upsert_target(
                domain=target_domain,
                provider_family="generic",
                discovered_at="2024-01-01T00:00:00Z",
                canonical_pipeline={"source": "session_manager_test"},
                field_mappings={},
            )
            print(f"[+] Seeded dummy target record for {target_domain}.")

            success, message = await session_manager.refresh_session(
                target_domain=target_domain,
                initial_url=initial_url,
                timeout=10.0,
            )
            print(f"[+] Session refresh result: {success}, {message}")

            # Synchronous example (exercises the running-loop worker-thread path).
            success_sync, message_sync = session_manager.refresh_session_sync(
                target_domain=target_domain,
                initial_url=initial_url,
                timeout=10.0,
            )
            print(f"[+] Synchronous session refresh result: {success_sync}, {message_sync}")
        finally:
            print("[*] Cleaning up test database...")
            try:
                if os.path.exists(db_path):
                    os.remove(db_path)
                    print(f"[+] Removed test database: {db_path}")
            except OSError as cleanup_err:
                print(f"[-] Failed to remove test database {db_path}: {cleanup_err}")

    asyncio.run(main())
