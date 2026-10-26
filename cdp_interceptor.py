import asyncio
import base64
import json
import logging
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import websockets

from browser_orchestrator import ChromeOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class NetworkEvent:
    request_id: str
    url: str
    method: str
    request_headers: Dict[str, str]
    status_code: int
    mime_type: str
    resource_type: str
    response_body: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class CDPNetworkInterceptor:

    def __init__(
        self,
        cdp_ws_url: str,
        on_event_callback: Optional[Callable[[NetworkEvent], None]] = None,
    ) -> None:
        self.cdp_ws_url: str = cdp_ws_url
        self.on_event_callback: Optional[Callable[[NetworkEvent], None]] = on_event_callback
        self.captured_events: List[NetworkEvent] = []

        self._ws: Optional[Any] = None
        self._listen_task: Optional[asyncio.Task[None]] = None
        self._is_listening: bool = False
        self._msg_id: int = 0
        self._pending_commands: Dict[int, asyncio.Future[Dict[str, Any]]] = {}
        self._pending_requests: Dict[str, Dict[str, Any]] = {}

    async def start(self) -> None:
        if self._is_listening:
            return

        try:
            self._ws = await websockets.connect(self.cdp_ws_url, max_size=None)
            self._is_listening = True
            self._listen_task = asyncio.create_task(self._listen_loop())

            # Enable network domain in CDP
            await self._send_command("Network.enable")
            logging.info("[CDP] Network monitoring enabled successfully.")
        except Exception as err:
            self._is_listening = False
            raise RuntimeError(f"Failed to connect or start CDP Interceptor: {err}") from err

    async def stop(self) -> None:
        if not self._is_listening:
            return

        self._is_listening = False
        try:
            if self._ws:
                try:
                    await self._send_command("Network.disable")
                except Exception:
                    pass
                await self._ws.close()
        except Exception as err:
            logging.warning(f"[CDP] Exception during WebSocket closure: {err}")
        finally:
            if self._listen_task and not self._listen_task.done():
                self._listen_task.cancel()
                try:
                    await self._listen_task
                except asyncio.CancelledError:
                    pass
            logging.info("[CDP] Interceptor stopped.")

    def get_captured_events(self) -> List[NetworkEvent]:
        return list(self.captured_events)

    async def _send_command(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self._ws:
            raise RuntimeError("WebSocket is not connected.")

        self._msg_id += 1
        cmd_id = self._msg_id
        payload = {"id": cmd_id, "method": method}
        if params:
            payload["params"] = params

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Dict[str, Any]] = loop.create_future()
        self._pending_commands[cmd_id] = future

        await self._ws.send(json.dumps(payload))
        return await future

    async def _listen_loop(self) -> None:
        if not self._ws:
            return

        try:
            async for raw_msg in self._ws:
                try:
                    data = json.loads(raw_msg)
                except json.JSONDecodeError:
                    continue

                # Handle response to sent commands
                if "id" in data and data["id"] in self._pending_commands:
                    cmd_id = data["id"]
                    future = self._pending_commands.pop(cmd_id)
                    if "error" in data:
                        future.set_exception(
                            RuntimeError(data["error"].get("message", "CDP Command Error"))
                        )
                    else:
                        future.set_result(data.get("result", {}))
                    continue

                # Handle asynchronous CDP events
                if "method" in data:
                    self._handle_cdp_event(data["method"], data.get("params", {}))

        except websockets.ConnectionClosed:
            logging.info("[CDP] WebSocket connection closed.")
        except Exception as err:
            logging.error(f"[CDP] Error in listen loop: {err}")

    @staticmethod
    def _is_api_candidate(url: str, mime_type: str = "", resource_type: str = "") -> bool:
        url_lower = url.lower()
        trackers = (
            "facebook.com",
            "google-analytics.com",
            "googletagmanager.com",
            "doubleclick.net",
            "sentry.io",
            "datadoghq.com",
            "hotjar.com",
            "segment.io",
            "mixpanel.com",
            "analytics",
        )
        if any(tracker in url_lower for tracker in trackers):
            return False

        ignored_extensions = (
            ".css",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".svg",
            ".ico",
            ".woff",
            ".woff2",
            ".ttf",
            ".eot",
            ".mp4",
            ".webp",
        )
        if any(url_lower.endswith(ext) or f"{ext}?" in url_lower for ext in ignored_extensions):
            return False

        if resource_type in ("XHR", "Fetch") or "json" in mime_type.lower():
            return True

        if not mime_type and not resource_type:
            return True

        return False

    def _handle_cdp_event(self, method: str, params: Dict[str, Any]) -> None:
        request_id = params.get("requestId")
        if not request_id:
            return

        if method == "Network.requestWillBeSent":
            request_info = params.get("request", {})
            req_url = request_info.get("url", "")
            req_method = request_info.get("method", "GET")
            self._pending_requests[request_id] = {
                "url": req_url,
                "method": req_method,
                "request_headers": request_info.get("headers", {}),
                "timestamp": params.get("timestamp", time.time()),
            }
            if self._is_api_candidate(req_url):
                print(f"[>>] Request starting: {req_method} {req_url}")

        elif method == "Network.responseReceived":
            response = params.get("response", {})
            if request_id not in self._pending_requests:
                self._pending_requests[request_id] = {
                    "url": response.get("url", ""),
                    "method": "GET",
                    "request_headers": {},
                    "timestamp": params.get("timestamp", time.time()),
                }
            self._pending_requests[request_id].update(
                {
                    "status_code": response.get("status", 0),
                    "mime_type": response.get("mimeType", ""),
                    "resource_type": params.get("type", ""),
                }
            )

        elif method == "Network.loadingFinished":
            if request_id in self._pending_requests:
                req_data = self._pending_requests[request_id]
                asyncio.create_task(self._process_completed_request(request_id, req_data))

        elif method == "Network.loadingFailed":
            self._pending_requests.pop(request_id, None)

    async def _process_completed_request(
        self, request_id: str, req_data: Dict[str, Any]
    ) -> None:
        resource_type = req_data.get("resource_type", "")
        mime_type = req_data.get("mime_type", "").lower()
        req_url = req_data.get("url", "")

        is_structured_data = (
            resource_type in ("XHR", "Fetch")
            or "json" in mime_type
            or "application/javascript" in mime_type
        )

        body: Optional[str] = None
        if is_structured_data and self._is_api_candidate(req_url, mime_type, resource_type):
            try:
                result = await self._send_command(
                    "Network.getResponseBody", {"requestId": request_id}
                )
                raw_body = result.get("body", "")
                if result.get("base64Encoded", False):
                    body = base64.b64decode(raw_body).decode("utf-8", errors="replace")
                else:
                    body = raw_body
            except Exception as err:
                logging.debug(f"[CDP] Could not get response body for {request_id}: {err}")

        event = NetworkEvent(
            request_id=request_id,
            url=req_url,
            method=req_data.get("method", "GET"),
            request_headers=req_data.get("request_headers", {}),
            status_code=req_data.get("status_code", 0),
            mime_type=mime_type,
            resource_type=resource_type,
            response_body=body,
        )

        self.captured_events.append(event)
        self._pending_requests.pop(request_id, None)

        if self._is_api_candidate(req_url, mime_type, resource_type) and body is not None:
            print(
                f"[>>] Intercepted API: {event.method} {event.url} ({event.status_code} / {event.mime_type})"
            )

        if self.on_event_callback:
            try:
                self.on_event_callback(event)
            except Exception as callback_err:
                logging.error(f"[CDP] Error in event callback: {callback_err}")


if __name__ == "__main__":
    async def main() -> None:
        print("[*] Starting Chrome Orchestrator for test...")
        orchestrator = ChromeOrchestrator(
            profile_name="interceptor_test_profile", port=9222, headless=False
        )
        orchestrator.start(timeout=10.0)

        try:
            ws_url = orchestrator.get_cdp_url()
            print(f"[+] CDP WebSocket URL obtained: {ws_url}")

            def on_network_event(event: NetworkEvent) -> None:
                if event.response_body:
                    print(f"\n[>>> CAPTURED CDP EVENT WITH JSON/DATA PAYLOAD <<<]")
                    print(f"URL: {event.url}")
                    print(f"Method: {event.method}")
                    print(f"Status: {event.status_code}")
                    print(f"Resource / MIME Type: {event.resource_type} / {event.mime_type}")
                    print(f"Body (first 200 chars): {event.response_body[:200]}...\n")

            interceptor = CDPNetworkInterceptor(
                cdp_ws_url=ws_url, on_event_callback=on_network_event
            )
            await interceptor.start()

            print("[*] Navigating to test JSON endpoint (httpbin.org/json)...")
            req = urllib.request.Request(
                "http://httpbin.org/json",
                headers={"User-Agent": "Mozilla/5.0"}
            )
            try:
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    print(f"[+] Direct HTTP response received with status {resp.status}")
            except Exception as e:
                print(f"[-] Error in test navigation: {e}")

            # Wait for interceptor to capture and process network events
            await asyncio.sleep(3)

            events = interceptor.get_captured_events()
            print(f"[+] Total captured events: {len(events)}")

            await interceptor.stop()
        finally:
            print("[*] Stopping Chrome Orchestrator...")
            orchestrator.stop()

    asyncio.run(main())
