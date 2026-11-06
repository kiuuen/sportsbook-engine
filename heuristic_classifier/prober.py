"""Standalone automated replay probe for candidate API endpoints.

Replays a captured network event as a headless HTTP request to verify
that the endpoint is independently reproducible outside the browser
session, returning the request headers used and a JSON payload preview.
"""

import json
import urllib.request
from typing import Dict, Tuple

from cdp_interceptor import NetworkEvent


class AutomatedReplayProber:
    """Replays a captured network event via a standalone HTTP request."""

    @staticmethod
    def replay_candidate_endpoint(
        event: NetworkEvent
    ) -> Tuple[bool, str, Dict[str, str], str]:
        """Replay a captured network event as a standalone HTTP request.

        Args:
            event: The captured network event to replay.

        Returns:
            A tuple of (success, message, headers_used, json_preview).
            On success, ``json_preview`` contains the first five lines of
            the pretty-printed JSON response; otherwise it is empty.
        """
        if not event or not event.url:
            return False, "No valid candidate event provided.", {}, ""

        # Filter out HTTP/2 pseudo-headers and hop-by-hop headers
        clean_headers: Dict[str, str] = {
            k: v for k, v in event.request_headers.items()
            if not k.startswith(":") and k.lower() not in ("host", "content-length")
        }

        if "User-Agent" not in clean_headers and "user-agent" not in clean_headers:
            clean_headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

        clean_headers["Accept"] = "application/json, text/plain, */*"

        try:
            req = urllib.request.Request(
                url=event.url,
                headers=clean_headers,
                method=event.method if event.method else "GET",
            )
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                status_code = resp.status
                resp_body = resp.read().decode("utf-8", errors="replace")

                if status_code == 200:
                    try:
                        parsed_json = json.loads(resp_body)
                        formatted_json = json.dumps(parsed_json, indent=2)
                        lines = formatted_json.splitlines()
                        five_line_preview = "\n".join(lines[:5])
                        return (
                            True,
                            "Replay probe succeeded (HTTP 200 + Valid JSON).",
                            clean_headers,
                            five_line_preview,
                        )
                    except json.JSONDecodeError:
                        return (
                            False,
                            "HTTP status 200 returned, but payload was not valid JSON.",
                            clean_headers,
                            "",
                        )
                else:
                    return (
                        False,
                        f"Replay failed with HTTP status code {status_code}.",
                        clean_headers,
                        "",
                    )
        except Exception as err:
            return False, f"Replay probe error: {err}", clean_headers, ""
