import datetime
import json
import logging
import pathlib
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Set

from cdp_interceptor import NetworkEvent


class SchemaExporter:

    ESSENTIAL_HEADER_KEYS: Set[str] = {
        "user-agent",
        "referer",
        "accept",
        "sec-ch-ua",
        "sec-ch-ua-mobile",
        "sec-ch-ua-platform",
        "authorization",
        "x-requested-with",
    }

    def __init__(self, base_output_dir: str = "./data/schemas") -> None:
        self.output_dir: pathlib.Path = pathlib.Path(base_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _extract_domain(target_url: str, events: List[NetworkEvent]) -> str:
        if target_url:
            parsed = urllib.parse.urlparse(target_url)
            host = parsed.netloc.split(":")[0].lower()
            if host:
                return host.replace("www.", "")

        for ev in events:
            if ev.url:
                parsed = urllib.parse.urlparse(ev.url)
                host = parsed.netloc.split(":")[0].lower()
                if host and not any(tracker in host for tracker in ("google.com", "facebook.com", "doubleclick.net")):
                    return host.replace("www.", "")

        return "unknown_target"

    @staticmethod
    def _extract_cookies_from_headers(headers: Dict[str, str]) -> Dict[str, str]:
        cookies: Dict[str, str] = {}
        for k, v in headers.items():
            if k.lower() == "cookie":
                parts = v.split(";")
                for part in parts:
                    if "=" in part:
                        cookie_k, cookie_v = part.strip().split("=", 1)
                        cookies[cookie_k.strip()] = cookie_v.strip()
        return cookies

    @staticmethod
    def _extract_essential_headers(headers: Dict[str, str]) -> Dict[str, str]:
        essential: Dict[str, str] = {}
        for k, v in headers.items():
            if not k.startswith(":") and k.lower() in SchemaExporter.ESSENTIAL_HEADER_KEYS:
                essential[k] = v
        return essential

    @staticmethod
    def _generate_curl(method: str, url: str, headers: Dict[str, str]) -> str:
        header_flags = " ".join([f'-H "{k}: {v}"' for k, v in headers.items()])
        if header_flags:
            return f'curl -X {method} "{url}" {header_flags}'
        return f'curl -X {method} "{url}"'

    @staticmethod
    def _truncate_payload(data: Any, max_list_items: int = 3) -> Any:
        """Recursively truncate large nested arrays and dict objects.

        Args:
            data: JSON serializable value (dict, list, primitive).
            max_list_items: Maximum number of items to retain in any array.

        Returns:
            A deep copy of ``data`` where every array exceeding ``max_list_items``
            is truncated to the first *max_list_items* elements.
        """
        if isinstance(data, list):
            if len(data) > max_list_items:
                truncated = data[:max_list_items]
                return [SchemaExporter._truncate_payload(item, max_list_items) for item in truncated]
            else:
                return [SchemaExporter._truncate_payload(item, max_list_items) for item in data]
        elif isinstance(data, dict):
            return {k: SchemaExporter._truncate_payload(v, max_list_items) for k, v in data.items()}
        else:
            return data

    def export_schema(
        self,
        target_url: str,
        classified_results: List[Any],
        events: List[NetworkEvent],
    ) -> pathlib.Path:
        target_domain = self._extract_domain(target_url, events)
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        session_headers: Dict[str, str] = {}
        session_cookies: Dict[str, str] = {}

        for ev in events:
            if ev.request_headers:
                ess_headers = self._extract_essential_headers(ev.request_headers)
                session_headers.update(ess_headers)
                extracted_cookies = self._extract_cookies_from_headers(ev.request_headers)
                session_cookies.update(extracted_cookies)

        endpoints_dict: Dict[str, Dict[str, Any]] = {}

        for res in classified_results:
            # Strict Odds Filter: exclude zero-odds candidates
            odds_count: int = getattr(res, "odds_count", 0)
            if odds_count == 0:
                continue

            sample_ev: Optional[NetworkEvent] = getattr(res, "sample_event", None)
            if not sample_ev:
                continue

            raw_body: Optional[str] = getattr(sample_ev, "response_body", None)
            if not raw_body:
                continue

            # Strict Structured JSON Filter
            try:
                parsed_payload: Any = json.loads(raw_body)
            except Exception:
                continue

            if not isinstance(parsed_payload, (dict, list)):
                continue

            method = sample_ev.method if sample_ev.method else "GET"
            req_headers = sample_ev.request_headers if sample_ev.request_headers else session_headers

            essential_req_headers = self._extract_essential_headers(req_headers)
            if not essential_req_headers:
                essential_req_headers = session_headers

            template_url = getattr(res, "url_template", getattr(res, "url", ""))
            dynamic_params = sorted(list(set(re.findall(r"\{([^}]+)\}", template_url))))

            curl_cmd = self._generate_curl(method, getattr(res, "url", ""), essential_req_headers)

            # Truncate nested payload arrays (keep max 3 items)
            sanitized_payload: Any = parsed_payload
            try:
                sanitized_payload = self._truncate_payload(parsed_payload, max_list_items=3)

                # Cap max serialized size to ~8 KB to keep schema clean and lightweight
                try:
                    payload_json = json.dumps(sanitized_payload, separators=(',', ':'))
                    max_bytes = 8192
                    if len(payload_json.encode('utf-8')) > max_bytes:
                        for limit in (2, 1):
                            sanitized_payload = self._truncate_payload(parsed_payload, max_list_items=limit)
                            payload_json = json.dumps(sanitized_payload, separators=(',', ':'))
                            if len(payload_json.encode('utf-8')) <= max_bytes:
                                break
                except Exception as json_err:
                    logging.warning(f"Size-check failed for payload: {json_err}")
                    sanitized_payload = {}
            except Exception as trunc_err:
                logging.warning(f"Payload truncation error: {trunc_err}")
                sanitized_payload = parsed_payload

            endpoint_data = {
                "method": method,
                "template_url": template_url,
                "dynamic_parameters": dynamic_params,
                "curl_command": curl_cmd,
                "sample_response": sanitized_payload,
            }
            endpoint_key = f"{method}_{template_url}"
            endpoints_dict[endpoint_key] = endpoint_data

        schema_payload = {
            "target_domain": target_domain,
            "discovered_at": timestamp,
            "session": {
                "essential_headers": session_headers,
                "cookies": session_cookies,
            },
            "endpoints": endpoints_dict,
        }

        output_file = self.output_dir / f"{target_domain}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(schema_payload, f, indent=2)

        return output_file
