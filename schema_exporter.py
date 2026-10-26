import datetime
import json
import logging
import pathlib
import re
import urllib.parse
import urllib.request
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
        "origin",
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
            if k.startswith(":"):
                continue
            k_lower = k.lower()
            if k_lower in SchemaExporter.ESSENTIAL_HEADER_KEYS or k_lower.startswith("x-"):
                essential[k_lower] = v
        return essential

    @staticmethod
    def _generate_curl(method: str, url: str, headers: Dict[str, str]) -> str:
        header_flags = " ".join([
            '-H "' + k.lower() + ": " + v.replace('"', '\\"') + '"'
            for k, v in headers.items()
        ])
        if header_flags:
            return f'curl -X {method} "{url}" {header_flags}'
        return f'curl -X {method} "{url}"'

    @staticmethod
    def _summarize_payload(data: Any, max_depth: int = 3) -> Any:
        """Generate a compact structural summary of a JSON payload.

        Args:
            data: JSON serializable value.
            max_depth: Maximum recursion depth for nested structures.

        Returns:
            A lightweight representation retaining structure with truncated values.
        """
        if max_depth <= 0:
            if isinstance(data, dict):
                return "{...}"
            if isinstance(data, list):
                return "[...]"
            return "..."

        if isinstance(data, dict):
            summary: Dict[str, Any] = {}
            for i, (k, v) in enumerate(data.items()):
                if i >= 8:
                    summary["_truncated_keys"] = f"+{len(data) - 8} keys"
                    break
                summary[k] = SchemaExporter._summarize_payload(v, max_depth - 1)
            return summary
        elif isinstance(data, list):
            if not data:
                return []
            return [SchemaExporter._summarize_payload(data[0], max_depth - 1)]
        elif isinstance(data, str):
            return data[:30] + "..." if len(data) > 30 else data
        else:
            return data

    @staticmethod
    def _verify_replay_candidate(event: NetworkEvent) -> bool:
        if not event or not event.url:
            return False

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
                if resp.status == 200:
                    body = resp.read().decode("utf-8", errors="replace")
                    json.loads(body)
                    return True
        except Exception:
            pass
        return False

    @staticmethod
    def _infer_provider_family(target_domain: str, endpoints: Dict[str, Any]) -> str:
        gateway_hosts: Dict[str, int] = {}
        route_taxonomy_segments: Set[str] = set()
        structural_keys: Set[str] = set()

        for ep_key, ep_info in endpoints.items():
            template_url = ep_info.get("template_url", "")
            parsed_url = urllib.parse.urlparse(template_url)
            netloc = parsed_url.netloc.lower().split(":")[0]
            if netloc:
                gateway_hosts[netloc] = gateway_hosts.get(netloc, 0) + 1

            path_parts = [p.lower() for p in parsed_url.path.split("/") if p and not p.startswith("{")]
            route_taxonomy_segments.update(path_parts[:3])

            sample_resp = ep_info.get("sample_response", {})
            if isinstance(sample_resp, dict):
                structural_keys.update(list(sample_resp.keys())[:5])

        if gateway_hosts:
            primary_gateway = max(gateway_hosts.items(), key=lambda x: x[1])[0]
        else:
            primary_gateway = target_domain

        clean_gateway = primary_gateway.replace(".", "_").replace("-", "_").replace("www_", "")
        taxonomy_sig = "".join(sorted(route_taxonomy_segments)) + "".join(sorted(structural_keys))
        signature_hash = sum(ord(c) for c in taxonomy_sig) % 10000 if taxonomy_sig else 1000

        return f"provider_{clean_gateway}_{signature_hash:04d}"

    @staticmethod
    def _build_canonical_pipeline(endpoints: Dict[str, Dict[str, Any]]) -> Dict[str, Optional[str]]:
        pipeline: Dict[str, Optional[str]] = {
            "sports_catalog": None,
            "live_overview": None,
            "upcoming_fixtures": None,
            "match_detail_metrics": None,
        }

        for ep_key, ep_info in endpoints.items():
            role = ep_info.get("role", "")
            volatility = ep_info.get("volatility", "")
            template_url = ep_info.get("template_url", "").lower()
            dynamic_params = ep_info.get("dynamic_parameters", [])

            if role == "catalog_root" and pipeline["sports_catalog"] is None:
                pipeline["sports_catalog"] = ep_key

            is_live_signal = "live" in template_url or "inplay" in template_url or volatility == "realtime"
            if role == "fixtures_list" and is_live_signal and pipeline["live_overview"] is None:
                pipeline["live_overview"] = ep_key

            if role == "fixtures_list" and not is_live_signal and pipeline["upcoming_fixtures"] is None:
                pipeline["upcoming_fixtures"] = ep_key

            has_match_param = any("event" in p.lower() or "match" in p.lower() or "fixture" in p.lower() for p in dynamic_params)
            if (role == "odds_detail" or has_match_param) and pipeline["match_detail_metrics"] is None:
                pipeline["match_detail_metrics"] = ep_key

        return pipeline

    def export_schema(
        self,
        target_url: str,
        classified_results: List[Any],
        events: List[NetworkEvent],
    ) -> Optional[pathlib.Path]:
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
        auxiliary_endpoints_dict: Dict[str, Dict[str, Any]] = {}

        for res in classified_results:
            sample_ev: Optional[NetworkEvent] = getattr(res, "sample_event", None)
            if not sample_ev:
                continue

            raw_body: Optional[str] = getattr(sample_ev, "response_body", None)
            if not raw_body:
                continue

            try:
                parsed_payload: Any = json.loads(raw_body)
            except Exception:
                continue

            if not isinstance(parsed_payload, (dict, list)):
                continue

            is_auxiliary: bool = getattr(res, "is_auxiliary_widget", False)
            is_viable: bool = getattr(res, "is_viable", False)
            odds_count: int = getattr(res, "odds_count", 0)
            score: float = getattr(res, "score", 0.0)
            role: str = getattr(res, "role", "catalog_root")
            volatility: str = getattr(res, "volatility", "static")

            # Core requirements filter vs auxiliary capture
            if not is_auxiliary:
                if not is_viable or score < 45.0:
                    continue
                is_detailed_or_realtime = (role == "odds_detail" or volatility == "realtime")
                if is_detailed_or_realtime and odds_count < 5:
                    continue
                if not self._verify_replay_candidate(sample_ev):
                    continue

            method = sample_ev.method if sample_ev.method else "GET"
            req_headers = sample_ev.request_headers if sample_ev.request_headers else session_headers
            essential_req_headers = self._extract_essential_headers(req_headers)
            if not essential_req_headers:
                essential_req_headers = session_headers

            template_url = getattr(res, "url_template", getattr(res, "url", ""))
            dynamic_params = sorted(list(set(re.findall(r"\{([^}]+)\}", template_url))))
            curl_cmd = self._generate_curl(method, getattr(res, "url", ""), essential_req_headers)

            sanitized_payload: Any = {}
            try:
                sanitized_payload = self._summarize_payload(parsed_payload)
            except Exception as trunc_err:
                logging.warning(f"Payload summarization error: {trunc_err}")
                sanitized_payload = {"error": "payload_summarization_failed"}

            produced_params: List[str] = getattr(res, "produced_params", [])

            endpoint_data = {
                "role": role,
                "volatility": volatility,
                "method": method,
                "template_url": template_url,
                "dynamic_parameters": dynamic_params,
                "provides_parameters": produced_params,
                "curl_command": curl_cmd,
                "sample_response": sanitized_payload,
            }
            endpoint_key = f"{method}_{template_url}"

            if is_auxiliary:
                auxiliary_endpoints_dict[endpoint_key] = endpoint_data
            else:
                endpoints_dict[endpoint_key] = endpoint_data

        if not endpoints_dict and not auxiliary_endpoints_dict:
            print("[!] Schema export aborted: No valid APIs found for this target.")
            return None

        provider_family = self._infer_provider_family(target_domain, endpoints_dict or auxiliary_endpoints_dict)
        canonical_pipeline = self._build_canonical_pipeline(endpoints_dict)

        producers_map: Dict[str, List[str]] = {}
        consumers_map: Dict[str, List[str]] = {}

        for ep_key, ep_val in endpoints_dict.items():
            for param in ep_val.get("provides_parameters", []):
                producers_map.setdefault(param, []).append(ep_key)
            if ep_val.get("dynamic_parameters"):
                consumers_map[ep_key] = ep_val["dynamic_parameters"]

        schema_payload = {
            "target_domain": target_domain,
            "provider_family": provider_family,
            "discovered_at": timestamp,
            "session": {
                "essential_headers": session_headers,
                "cookies": session_cookies,
            },
            "canonical_pipeline": canonical_pipeline,
            "dependency_graph": {
                "producers": producers_map,
                "consumers": consumers_map,
            },
            "endpoints": endpoints_dict,
            "auxiliary_endpoints": auxiliary_endpoints_dict,
        }

        output_file = self.output_dir / f"{target_domain}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(schema_payload, f, indent=2)

        return output_file
