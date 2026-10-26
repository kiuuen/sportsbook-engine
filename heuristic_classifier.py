import json
import logging
import math
import os
import pathlib
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from browser_orchestrator import ChromeOrchestrator
from cdp_interceptor import CDPNetworkInterceptor, NetworkEvent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class ClassificationResult:
    url: str
    url_template: str
    score: float
    category_scores: Dict[str, float]
    odds_count: int
    matched_keywords: Set[str] = field(default_factory=set)
    is_grey_zone: bool = False
    ambiguous_keys: List[str] = field(default_factory=list)
    call_count: int = 1
    time_span_seconds: float = 0.0
    sample_event: Optional[NetworkEvent] = None


class LexiconManager:

    GENERIC_BLACKLIST: Set[str] = {
        "id",
        "data",
        "name",
        "type",
        "status",
        "value",
        "items",
        "code",
        "message",
        "result",
        "success",
        "error",
        "key",
        "title",
        "description",
        "info",
        "updated",
        "created",
        "timestamp",
        "version",
    }

    POSITIVE_CATEGORIES: Set[str] = {
        "betting_markets_and_odds",
        "sports_event_metadata",
        "live_state_and_settlement",
        "provider_identifiers_and_abbreviations",
    }

    NOISE_CATEGORIES: Set[str] = {
        "noise_cms_and_layout",
        "noise_account_and_cashier",
        "noise_casino_and_promotions",
        "noise_telemetry_and_tracking",
    }

    def __init__(self, lexicon_path: str = "sportsbook_lexicon.json") -> None:
        self.lexicon_path: pathlib.Path = pathlib.Path(lexicon_path)
        self.positive_keywords: Dict[str, Set[str]] = {}
        self.noise_keywords: Dict[str, Set[str]] = {}
        self.all_positive_terms: Set[str] = set()
        self.all_noise_terms: Set[str] = set()
        self.load_lexicon()

    def load_lexicon(self) -> None:
        if not self.lexicon_path.exists():
            raise FileNotFoundError(f"Lexicon file not found at {self.lexicon_path.resolve()}")

        with open(self.lexicon_path, "r", encoding="utf-8") as f:
            data: Dict[str, List[str]] = json.load(f)

        for category, terms in data.items():
            term_set = {t.lower() for t in terms}
            if category in self.POSITIVE_CATEGORIES:
                self.positive_keywords[category] = term_set
                self.all_positive_terms.update(term_set)
            elif category in self.NOISE_CATEGORIES:
                self.noise_keywords[category] = term_set
                self.all_noise_terms.update(term_set)

    def is_generic_or_blacklisted(self, term: str) -> bool:
        normalized = term.strip().lower()
        return normalized in self.GENERIC_BLACKLIST or len(normalized) <= 2


class RouteTemplateDeducer:

    @staticmethod
    def deduce_template(url: str) -> str:
        parsed = urllib.parse.urlparse(url)
        path_segments = parsed.path.split("/")
        deduced_segments = []

        for seg in path_segments:
            if not seg:
                deduced_segments.append("")
                continue
            # UUID pattern
            if re.match(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$", seg):
                deduced_segments.append("{uuid}")
            # Epoch Timestamp (10 or 13 digits)
            elif re.match(r"^\d{10}(\d{3})?$", seg):
                deduced_segments.append("{timestamp}")
            # Long numeric ID (5+ digits)
            elif re.match(r"^\d{5,}$", seg):
                deduced_segments.append("{id}")
            else:
                deduced_segments.append(seg)

        deduced_path = "/".join(deduced_segments)

        # Parse query params
        query_params = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        deduced_query = []

        for k, vals in query_params.items():
            k_lower = k.lower()
            val = vals[0] if vals else ""

            if "market" in k_lower or "mkt" in k_lower:
                placeholder = f"{k}={{market_id}}"
            elif "event" in k_lower or "match" in k_lower or "fixture" in k_lower:
                placeholder = f"{k}={{event_id}}"
            elif "time" in k_lower or "date" in k_lower or re.match(r"^\d{10}(\d{3})?$", val):
                placeholder = f"{k}={{timestamp}}"
            elif re.match(r"^\d{5,}$", val):
                placeholder = f"{k}={{{k}_id}}"
            else:
                placeholder = f"{k}={val}"

            deduced_query.append(placeholder)

        query_str = "&".join(deduced_query) if deduced_query else ""
        return urllib.parse.urlunparse((
            parsed.scheme,
            parsed.netloc,
            deduced_path,
            parsed.params,
            query_str,
            parsed.fragment
        ))


class HeuristicClassifier:

    def __init__(self, lexicon_path: str = "sportsbook_lexicon.json") -> None:
        self.lexicon_mgr: LexiconManager = LexiconManager(lexicon_path)

    def _inspect_node(
        self,
        node: Any,
        keys_found: List[str],
        numeric_values: List[float],
        array_lengths: List[int],
    ) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                keys_found.append(str(k).lower())
                self._inspect_node(v, keys_found, numeric_values, array_lengths)
        elif isinstance(node, list):
            array_lengths.append(len(node))
            for item in node:
                self._inspect_node(item, keys_found, numeric_values, array_lengths)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            numeric_values.append(float(node))
        elif isinstance(node, str):
            if re.match(r"^-?\d+(\.\d+)?$", node.strip()):
                try:
                    numeric_values.append(float(node.strip()))
                except ValueError:
                    pass

    def _is_decimal_odds(self, val: float) -> bool:
        return 1.01 <= val <= 100.0 and round(val, 2) == val

    def _is_american_odds(self, val: float) -> bool:
        return (val <= -100 or val >= 100) and val == math.floor(val) and abs(val) <= 50000

    def score_payload(
        self,
        url: str,
        payload_str: str,
        call_count: int = 1,
        time_span_seconds: float = 0.0,
        sample_event: Optional[NetworkEvent] = None,
    ) -> ClassificationResult:
        url_template = RouteTemplateDeducer.deduce_template(url)

        if not payload_str or not payload_str.strip():
            return ClassificationResult(
                url=url,
                url_template=url_template,
                score=0.0,
                category_scores={},
                odds_count=0,
                call_count=call_count,
                time_span_seconds=time_span_seconds,
                sample_event=sample_event,
            )

        try:
            data = json.loads(payload_str)
        except Exception:
            return ClassificationResult(
                url=url,
                url_template=url_template,
                score=0.0,
                category_scores={},
                odds_count=0,
                call_count=call_count,
                time_span_seconds=time_span_seconds,
                sample_event=sample_event,
            )

        keys_found: List[str] = []
        numeric_values: List[float] = []
        array_lengths: List[int] = []

        self._inspect_node(data, keys_found, numeric_values, array_lengths)

        if not keys_found and not numeric_values:
            return ClassificationResult(
                url=url,
                url_template=url_template,
                score=0.0,
                category_scores={},
                odds_count=0,
                call_count=call_count,
                time_span_seconds=time_span_seconds,
                sample_event=sample_event,
            )

        matched_keywords: Set[str] = set()
        category_matches: Dict[str, int] = {
            cat: 0 for cat in self.lexicon_mgr.POSITIVE_CATEGORIES | self.lexicon_mgr.NOISE_CATEGORIES
        }

        for key in keys_found:
            tokens = re.split(r"[_\-\s\.]", key)
            for cat, terms in self.lexicon_mgr.positive_keywords.items():
                for term in terms:
                    if term in key or term in tokens:
                        category_matches[cat] += 1
                        matched_keywords.add(term)

            for cat, terms in self.lexicon_mgr.noise_keywords.items():
                for term in terms:
                    if term in key or term in tokens:
                        category_matches[cat] += 1

        decimal_odds_count = sum(1 for v in numeric_values if self._is_decimal_odds(v))
        american_odds_count = sum(1 for v in numeric_values if self._is_american_odds(v))
        total_odds_count = decimal_odds_count + american_odds_count

        significant_arrays = [l for l in array_lengths if l >= 2]
        array_repetition_score = min(len(significant_arrays) * 5.0, 25.0)

        pos_score = sum(
            category_matches[cat] * 12.0 for cat in self.lexicon_mgr.POSITIVE_CATEGORIES
        )
        noise_score = sum(
            category_matches[cat] * 15.0 for cat in self.lexicon_mgr.NOISE_CATEGORIES
        )

        odds_score = min(total_odds_count * 4.0, 35.0)

        # Temporal & Frequency Scoring Boost
        frequency_bonus = min(call_count * 2.5, 12.0)
        temporal_polling_bonus = 10.0 if (call_count >= 2 and time_span_seconds >= 1.0) else 0.0

        raw_score = (
            pos_score
            + odds_score
            + array_repetition_score
            + frequency_bonus
            + temporal_polling_bonus
            - noise_score
        )

        final_score = max(0.0, min(100.0, raw_score))

        has_structural_relevance = total_odds_count >= 3 or len(significant_arrays) >= 2
        low_vocabulary = pos_score < 10.0
        low_noise = noise_score < 10.0

        is_grey_zone = has_structural_relevance and low_vocabulary and low_noise

        ambiguous_keys: List[str] = []
        if is_grey_zone:
            for key in set(keys_found):
                if not self.lexicon_mgr.is_generic_or_blacklisted(key):
                    if not any(
                        key in terms
                        for terms in self.lexicon_mgr.positive_keywords.values()
                    ):
                        ambiguous_keys.append(key)

        cat_score_dict = {
            cat: float(count) for cat, count in category_matches.items() if count > 0
        }

        return ClassificationResult(
            url=url,
            url_template=url_template,
            score=round(final_score, 2),
            category_scores=cat_score_dict,
            odds_count=total_odds_count,
            matched_keywords=matched_keywords,
            is_grey_zone=is_grey_zone,
            ambiguous_keys=ambiguous_keys[:20],
            call_count=call_count,
            time_span_seconds=round(time_span_seconds, 2),
            sample_event=sample_event,
        )


class AutomatedReplayProber:

    @staticmethod
    def replay_candidate_endpoint(
        event: NetworkEvent
    ) -> Tuple[bool, str, Dict[str, str], str]:
        if not event or not event.url:
            return False, "No valid candidate event provided.", {}, ""

        # Filter out HTTP/2 pseudo-headers
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
                        return True, "Replay probe succeeded (HTTP 200 + Valid JSON).", clean_headers, five_line_preview
                    except json.JSONDecodeError:
                        return False, f"HTTP status 200 returned, but payload was not valid JSON.", clean_headers, ""
                else:
                    return False, f"Replay failed with HTTP status code {status_code}.", clean_headers, ""
        except Exception as err:
            return False, f"Replay probe error: {err}", clean_headers, ""


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
                if res.score > 0.0 or res.is_grey_zone:
                    results.append(res)

            results.sort(key=lambda r: r.score, reverse=True)

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

                # Execute Automated Replay Probe for top candidate
                top_candidate = results[0]
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

            print("\n=======================================================\n")

        finally:
            print("[*] Terminating Chrome orchestrator process...")
            orchestrator.stop()

    asyncio.run(run_live_test_harness())
