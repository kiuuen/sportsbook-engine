import datetime
import json
import logging
import pathlib
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Set, Tuple

from cdp_interceptor import NetworkEvent
from heuristic_classifier import ClassificationResult, HeuristicClassifier, RouteTemplateDeducer, AutomatedReplayProber, NoiseAssistant
from browser_orchestrator import ChromeOrchestrator
from database_manager import DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


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

        if event.resource_type == "WebSocket" or event.method in ("WS_RECV", "WS_SENT"):
            return bool(event.response_body and event.response_body.strip())

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

        taxonomy_keywords = ("catalog", "sports", "categories", "taxonomy", "menu", "tree", "navigation", "hierarchy", "structure")

        for ep_key, ep_info in endpoints.items():
            role = ep_info.get("role", "")
            volatility = ep_info.get("volatility", "")
            template_url = ep_info.get("template_url", "").lower()
            dynamic_params = ep_info.get("dynamic_parameters", [])
            sample_resp = ep_info.get("sample_response", {})

            has_taxonomy_url = any(kw in template_url for kw in taxonomy_keywords)
            has_taxonomy_resp = isinstance(sample_resp, dict) and any(
                kw in str(key).lower() for key in sample_resp.keys() for kw in taxonomy_keywords
            )

            if pipeline["sports_catalog"] is None and (role == "catalog_root" or has_taxonomy_url or has_taxonomy_resp):
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

    def build_schema_dict(
        self,
        target_url: str,
        classified_results: List[ClassificationResult],
        events: List[NetworkEvent],
    ) -> Dict[str, Any]:
        """
        Construct the full schema dictionary from captured events and classification results.
        The returned dict matches the structure expected by DatabaseManager.save_schema_dict.
        """
        # Extract target domain
        target_domain = self._extract_domain(target_url, events)

        # Aggregate session headers and cookies
        session_headers: Dict[str, str] = {}
        session_cookies: Dict[str, str] = {}
        for ev in events:
            if ev.request_headers:
                ess = self._extract_essential_headers(ev.request_headers)
                session_headers.update(ess)
                cookies = self._extract_cookies_from_headers(ev.request_headers)
                session_cookies.update(cookies)

        # Build endpoint dictionaries
        endpoints_dict: Dict[str, Dict[str, Any]] = {}
        streaming_endpoints_dict: Dict[str, Dict[str, Any]] = {}
        auxiliary_endpoints_dict: Dict[str, Dict[str, Any]] = {}

        for res in classified_results:
            sample_ev = getattr(res, "sample_event", None)
            if not sample_ev:
                continue
            raw_body = getattr(sample_ev, "response_body", None)
            if not raw_body:
                continue
            try:
                parsed_payload = json.loads(raw_body)
            except Exception:
                continue
            if not isinstance(parsed_payload, (dict, list)):
                continue

            is_auxiliary = getattr(res, "is_auxiliary_widget", False)
            is_viable = getattr(res, "is_viable", False)
            score = getattr(res, "score", 0.0)
            role = getattr(res, "role", "catalog_root")
            volatility = getattr(res, "volatility", "static")
            is_websocket = (
                sample_ev.resource_type == "WebSocket"
                or sample_ev.method in ("WS_RECV", "WS_SENT")
            )

            # Apply same filtering logic as original export_schema
            if not is_auxiliary:
                if not is_viable or score < 45.0:
                    continue
                is_detailed_or_realtime = (role == "odds_detail" or volatility == "realtime")
                if is_detailed_or_realtime and getattr(res, "odds_count", 0) < 5 and not is_websocket:
                    continue
                if not self._verify_replay_candidate(sample_ev):
                    continue

            method = sample_ev.method if sample_ev.method else "GET"
            req_headers = sample_ev.request_headers if sample_ev.request_headers else {}
            essential_req_headers = (
                self._extract_essential_headers(req_headers) if not is_websocket else {}
            )
            template_url = getattr(res, "url_template", getattr(res, "url", ""))
            dynamic_params = sorted(
                list(set(re.findall(r"\{([^}]+)\}", template_url)))
            )
            curl_cmd = (
                self._generate_curl(method, getattr(res, "url", ""), essential_req_headers)
                if not is_websocket
                else ""
            )

            try:
                summarized_payload = self._summarize_payload(parsed_payload)
            except Exception as e:
                logging.warning(f"Payload summarization error: {e}")
                summarized_payload = {"error": "payload_summarization_failed"}

            produced_params = getattr(res, "produced_params", [])

            endpoint_data = {
                "role": role,
                "volatility": volatility,
                "protocol": (
                    "websocket"
                    if is_websocket
                    else ("https" if getattr(sample_ev, "url", "").startswith("https") else "http")
                ),
                "method": method,
                "template_url": template_url,
                "dynamic_parameters": dynamic_params,
                "provides_parameters": produced_params,
                "curl_command": curl_cmd,
                "handshake_headers": essential_req_headers if is_websocket else {},
                "sample_response": summarized_payload,
            }
            endpoint_key = f"{method}_{template_url}"
            if is_websocket:
                streaming_endpoints_dict[endpoint_key] = endpoint_data
            elif is_auxiliary:
                auxiliary_endpoints_dict[endpoint_key] = endpoint_data
            else:
                endpoints_dict[endpoint_key] = endpoint_data

        # Derive provider family and canonical pipeline
        provider_family = self._infer_provider_family(
            target_domain,
            {**endpoints_dict, **streaming_endpoints_dict, **auxiliary_endpoints_dict},
        )
        canonical_pipeline = self._build_canonical_pipeline(endpoints_dict)

        # Build dependency graph
        producers_map: Dict[str, List[str]] = {}
        consumers_map: Dict[str, List[str]] = {}
        for ep_key, ep_val in endpoints_dict.items():
            for param in ep_val.get("provides_parameters", []):
                producers_map.setdefault(param, []).append(ep_key)
            if ep_val.get("dynamic_parameters"):
                consumers_map[ep_key] = ep_val["dynamic_parameters"]

        # Assemble final schema dictionary
        schema_dict: Dict[str, Any] = {
            "target_domain": target_domain,
            "provider_family": provider_family,
            "discovered_at": datetime.datetime.utcnow()
                .replace(tzinfo=datetime.timezone.utc)
                .isoformat(),
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
            "streaming_endpoints": streaming_endpoints_dict,
            "auxiliary_endpoints": auxiliary_endpoints_dict,
        }

        return schema_dict

    def export_schema(
        self,
        target_url: str,
        classified_results: List[ClassificationResult],
        events: List[NetworkEvent],
        db_manager: Optional[DatabaseManager] = None,
    ) -> Tuple[pathlib.Path, Optional[int]]:
        """
        Export the discovered schema to a JSON file and optionally persist it in SQLite.
        Returns a tuple of (json_path, target_id). target_id is None if db_manager is not provided.
        """
        schema_dict = self.build_schema_dict(target_url, classified_results, events)

        target_domain = schema_dict["target_domain"]
        output_file = self.output_dir / f"{target_domain}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(schema_dict, f, indent=2)

        target_id: Optional[int] = None
        if db_manager is not None:
            target_id = db_manager.save_schema_dict(schema_dict)

        return output_file, target_id

if __name__ == "__main__":
    import asyncio

    async def run_live_test_harness() -> None:
        print("\n=======================================================")
        print("  LIVE SPORTSBOOK API HEURISTIC CLASSIFIER HARNESS")
        print("=======================================================\n")

        target_url = input("Enter target sportsbook URL (or press Enter for blank): ").strip()
        if target_url and not target_url.startswith(("http://", "https://")):
            target_url = "https://" + target_url

        classifier = HeuristicClassifier("sportsbook_lexicon.json")
        orchestrator = ChromeOrchestrator(
            profile_name="heuristic_harness_profile", port=9222, headless=False
        )

        print("[*] Launching browser instance...")
        orchestrator.start(timeout=15.0, initial_url=target_url if target_url else None)

        try:
            ws_url = orchestrator.get_cdp_url(target_url=target_url if target_url else None)
            print(f"[+] CDP WebSocket URL acquired: {ws_url}")

            interceptor = CDPNetworkInterceptor(cdp_ws_url=ws_url)
            await interceptor.start()

            print("\n-------------------------------------------------------")
            print("OPERATOR ACTION REQUIRED:")
            print("1. Chrome has opened.")
            print("2. Navigate to any sportsbook site (e.g., live events, markets).")
            print("3. Interact with odds/fixtures on the page.")
            print("4. Return here and press ENTER when ready to evaluate captured traffic.")
            print("-------------------------------------------------------\n")

            await asyncio.to_thread(
                input, "--> Press ENTER to stop capturing and evaluate candidate endpoints... <--\n"
            )

            print("[*] Draining pending in-flight network requests...")
            await asyncio.sleep(2.0)

            events = interceptor.get_captured_events()
            await interceptor.stop()

            print(f"[*] Interception complete. Total network events captured: {len(events)}")
            print("[*] Grouping events by deduced route templates & scoring...\n")

            # Group events by deduced route template
            template_groups: Dict[str, List[NetworkEvent]] = {}
            for ev in events:
                if ev.response_body:
                    tmpl = RouteTemplateDeducer.deduce_template(ev.url)
                    template_groups.setdefault(tmpl, []).append(ev)

            results: List[ClassificationResult] = []
            for tmpl, group_events in template_groups.items():
                call_count = len(group_events)
                timestamps = [e.timestamp for e in group_events]
                time_span = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0
                sample_event = group_events[0]

                res = classifier.score_payload(
                    url=sample_event.url,
                    payload_str=sample_event.response_body or "",
                    call_count=call_count,
                    time_span_seconds=time_span,
                    sample_event=sample_event,
                )
                if res.is_viable or res.is_grey_zone:
                    results.append(res)

            results.sort(
                key=lambda r: (r.score, r.core_betting_matches, r.odds_count, r.call_count),
                reverse=True,
            )

            print("=======================================================")
            print("         SCORED CANDIDATE ENDPOINTS SUMMARY            ")
            print("=======================================================")

            if not results:
                print("[-] No sportsbook API candidate endpoints detected in captured traffic.")
            else:
                for idx, res in enumerate(results[:15], start=1):
                    print(f"\n[{idx}] Score: {res.score}/100 | Calls: {res.call_count} | TimeSpan: {res.time_span_seconds}s | Odds Detected: {res.odds_count}")
                    print(f"    Raw Sample URL: {res.url}")
                    print(f"    Deduced Template: {res.url_template}")
                    if res.matched_keywords:
                        print(f"    Matched Terms: {', '.join(sorted(res.matched_keywords)[:8])}")
                    if res.is_grey_zone:
                        print(f"    [!] GREY ZONE DETECTED - Candidate terms: {res.ambiguous_keys[:5]}")

                # Execute Automated Replay Probe for top candidate with odds_count >= 5
                eligible_candidates = [r for r in results if r.odds_count >= 5]
                if eligible_candidates:
                    top_candidate = eligible_candidates[0]
                    if top_candidate.sample_event:
                        print("\n=======================================================")
                        print("     AUTOMATED STANDALONE REPLAY PROBE (TOP ENDPOINT)   ")
                        print("=======================================================")
                        print(f"[*] Executing headless HTTP replay for: {top_candidate.url}")

                        success, msg, headers_used, json_preview = AutomatedReplayProber.replay_candidate_endpoint(
                            top_candidate.sample_event
                        )

                        print(f"[+] Replay Verification Result: {'SUCCESS' if success else 'FAILED'}")
                        print(f"    Message: {msg}")
                        print(f"    Final Parameterized Template: {top_candidate.url_template}")
                        print("    Minimal Required Headers:")
                        for h_k, h_v in list(headers_used.items())[:6]:
                            print(f"      - {h_k}: {h_v}")

                        if success and json_preview:
                            print("\n    JSON Payload Preview (5 lines):")
                            for line in json_preview.splitlines():
                                print(f"      {line}")
                else:
                    print("\n[-] No candidate endpoint met the minimum odds threshold (odds_count >= 5) for replay probe.")

                # Execute Guarded Post-Discovery Noise Assistant for residual traffic
                NoiseAssistant.process_noise_domains(
                    events=events,
                    classified_results=results,
                    target_url=target_url,
                )

                # Export validated blueprint schema to JSON file
                try:
                    exporter = SchemaExporter()
                    export_path, target_id = exporter.export_schema(
                        target_url=target_url,
                        classified_results=results,
                        events=events,
                    )
                    if export_path is not None:
                        print(f"\n[+] Schema successfully exported to: {export_path.resolve()}")
                except Exception as err:
                    logging.error(f"[Exporter Error] Failed to export schema: {err}", exc_info=True)

            print("\n=======================================================\n")

        finally:
            print("[*] Terminating Chrome orchestrator process...")
            orchestrator.stop()

    asyncio.run(run_live_test_harness())
