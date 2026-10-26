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
from schema_exporter import SchemaExporter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class ClassificationResult:
    url: str
    url_template: str
    score: float
    category_scores: Dict[str, float]
    odds_count: int
    matched_keywords: Set[str] = field(default_factory=set)
    core_betting_matches: int = 0
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

    CATEGORY_WEIGHTS: Dict[str, float] = {
        "betting_markets_and_odds": 36.0,
        "sports_event_metadata": 12.0,
        "live_state_and_settlement": 18.0,
        "provider_identifiers_and_abbreviations": 12.0,
    }

    def __init__(
        self,
        lexicon_path: str = "sportsbook_lexicon.json",
        filters_path: str = "network_filters.json",
    ) -> None:
        self.lexicon_mgr: LexiconManager = LexiconManager(lexicon_path)
        self.filters_path: pathlib.Path = pathlib.Path(filters_path)
        self.ignored_url_keywords: List[str] = self._load_ignored_url_keywords()

    def _load_ignored_url_keywords(self) -> List[str]:
        if self.filters_path.exists():
            try:
                with open(self.filters_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return [kw.lower() for kw in data.get("ignored_url_keywords", [])]
            except Exception as e:
                logging.warning(f"Could not load {self.filters_path}: {e}")
        return []

    def _inspect_node(
        self,
        node: Any,
        keys_found: List[str],
        numeric_values: List[float],
        array_lengths: List[int],
        contextual_numerics: Optional[List[Tuple[str, float]]] = None,
        parent_key: str = "",
    ) -> None:
        if contextual_numerics is None:
            contextual_numerics = []

        if isinstance(node, dict):
            for k, v in node.items():
                key_str = str(k).lower()
                keys_found.append(key_str)
                self._inspect_node(
                    v, keys_found, numeric_values, array_lengths, contextual_numerics, parent_key=key_str
                )
        elif isinstance(node, list):
            array_lengths.append(len(node))
            for item in node:
                self._inspect_node(
                    item, keys_found, numeric_values, array_lengths, contextual_numerics, parent_key=parent_key
                )
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            val = float(node)
            numeric_values.append(val)
            if parent_key:
                contextual_numerics.append((parent_key, val))
        elif isinstance(node, str):
            if re.match(r"^-?\d+(\.\d+)?$", node.strip()):
                try:
                    val = float(node.strip())
                    numeric_values.append(val)
                    if parent_key:
                        contextual_numerics.append((parent_key, val))
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
        contextual_numerics: List[Tuple[str, float]] = []

        self._inspect_node(data, keys_found, numeric_values, array_lengths, contextual_numerics)

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
        core_betting_matched_terms: Set[str] = set()
        category_matches: Dict[str, int] = {
            cat: 0 for cat in self.lexicon_mgr.POSITIVE_CATEGORIES | self.lexicon_mgr.NOISE_CATEGORIES
        }

        for key in keys_found:
            tokens = [t for t in re.split(r"[_\-\s\.]", key) if t]
            tokens_set = set(tokens)
            for cat, terms in self.lexicon_mgr.positive_keywords.items():
                for term in terms:
                    if len(term) <= 3:
                        if term in tokens_set:
                            category_matches[cat] += 1
                            matched_keywords.add(term)
                            if cat == "betting_markets_and_odds":
                                core_betting_matched_terms.add(term)
                    else:
                        if term in key or term in tokens_set:
                            category_matches[cat] += 1
                            matched_keywords.add(term)
                            if cat == "betting_markets_and_odds":
                                core_betting_matched_terms.add(term)

            for cat, terms in self.lexicon_mgr.noise_keywords.items():
                for term in terms:
                    if len(term) <= 3:
                        if term in tokens_set:
                            category_matches[cat] += 1
                    else:
                        if term in key or term in tokens_set:
                            category_matches[cat] += 1

        decimal_odds_count = sum(1 for v in numeric_values if self._is_decimal_odds(v))
        american_odds_count = sum(1 for v in numeric_values if self._is_american_odds(v))
        total_odds_count = decimal_odds_count + american_odds_count

        core_terms = self.lexicon_mgr.positive_keywords.get("betting_markets_and_odds", set())
        contextual_odds_count = 0
        for parent_key, val in contextual_numerics:
            if self._is_decimal_odds(val) or self._is_american_odds(val):
                key_tokens = set(re.split(r"[_\-\s\.]", parent_key))
                if any(term in parent_key or term in key_tokens for term in core_terms):
                    contextual_odds_count += 1

        contextual_odds_bonus = min(contextual_odds_count * 3.0, 15.0)

        significant_arrays = [l for l in array_lengths if l >= 2]
        array_repetition_score = min(len(significant_arrays) * 5.0, 25.0)

        pos_score = sum(
            category_matches[cat] * self.CATEGORY_WEIGHTS.get(cat, 12.0)
            for cat in self.lexicon_mgr.POSITIVE_CATEGORIES
        )
        noise_score = sum(
            category_matches[cat] * 15.0 for cat in self.lexicon_mgr.NOISE_CATEGORIES
        )

        odds_score = min(total_odds_count * 4.0, 35.0)

        frequency_bonus = min(call_count * 2.5, 12.0)
        temporal_polling_bonus = 10.0 if (call_count >= 2 and time_span_seconds >= 1.0) else 0.0

        url_lower = url.lower()
        url_path_lower = urllib.parse.urlparse(url).path.lower()
        url_noise_penalty = 60.0 if (
            any(kw in url_lower for kw in self.ignored_url_keywords)
            or "translation" in url_path_lower
        ) else 0.0

        raw_score = (
            pos_score
            + odds_score
            + contextual_odds_bonus
            + array_repetition_score
            + frequency_bonus
            + temporal_polling_bonus
            - noise_score
            - url_noise_penalty
        )

        final_score = max(0.0, min(100.0, raw_score))

        # Mandatory Odds Gate: payloads without odds cannot exceed 40.0
        if total_odds_count == 0:
            final_score = min(final_score, 40.0)

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
            core_betting_matches=len(core_betting_matched_terms),
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


class NoiseAssistant:

    @staticmethod
    def process_noise_domains(
        events: List[NetworkEvent],
        classified_results: List[ClassificationResult],
        target_url: str,
        filters_path: str = "network_filters.json",
    ) -> None:
        target_host = urllib.parse.urlparse(target_url).netloc.lower().split(":")[0] if target_url else ""
        host_parts = [p for p in target_host.split(".") if p]
        root_domain = ".".join(host_parts[-2:]) if len(host_parts) >= 2 else target_host

        # Build set of immune hosts from endpoints with score > 0 or odds_count > 0
        immune_hosts: Set[str] = set()
        for res in classified_results:
            if res.score > 0.0 or res.odds_count > 0:
                host = urllib.parse.urlparse(res.url).netloc.lower().split(":")[0]
                if host:
                    immune_hosts.add(host)

        def is_immune(host: str) -> bool:
            if not host:
                return True
            if host in immune_hosts:
                return True
            if root_domain:
                if host == root_domain or host.endswith("." + root_domain):
                    return True
            if target_host:
                if host == target_host or host.endswith("." + target_host):
                    return True
            return False

        filters_file = pathlib.Path(filters_path)
        existing_ignored: Set[str] = set()
        filters_data: Dict[str, Any] = {}

        if filters_file.exists():
            try:
                with open(filters_file, "r", encoding="utf-8") as f:
                    filters_data = json.load(f)
                existing_ignored = {d.lower() for d in filters_data.get("ignored_domains", [])}
            except Exception as e:
                logging.warning(f"Could not load {filters_path}: {e}")

        domain_counts: Dict[str, int] = {}
        for ev in events:
            host = urllib.parse.urlparse(ev.url).netloc.lower()
            if not host or is_immune(host) or host in existing_ignored:
                continue
            domain_counts[host] = domain_counts.get(host, 0) + 1

        candidate_domains = sorted(domain_counts.keys(), key=lambda d: domain_counts[d], reverse=True)
        if not candidate_domains:
            return

        print("\n=======================================================")
        print("     POST-DISCOVERY NOISE ASSISTANT (TELEMETRY)        ")
        print("=======================================================")
        print("[*] Detected third-party noise/telemetry domains in residual traffic:")
        for domain in candidate_domains:
            print(f"    - {domain} ({domain_counts[domain]} requests)")

        response = input("\nDo you want to append these domains to 'ignored_domains' in network_filters.json? [y/N]: ").strip().lower()
        if response in ("y", "yes"):
            updated_list = filters_data.get("ignored_domains", [])
            for domain in candidate_domains:
                if domain not in updated_list:
                    updated_list.append(domain)
            filters_data["ignored_domains"] = updated_list

            try:
                with open(filters_file, "w", encoding="utf-8") as f:
                    json.dump(filters_data, f, indent=2)
                print(f"[+] Successfully appended {len(candidate_domains)} domains to {filters_path}.")
            except Exception as err:
                print(f"[-] Failed to update {filters_path}: {err}")
        else:
            print("[*] Skipped adding noise domains.")


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
                    saved_path = exporter.export_schema(
                        target_url=target_url,
                        classified_results=results,
                        events=events,
                    )
                    if saved_path is not None:
                        print(f"\n[+] Schema successfully exported to: {saved_path.resolve()}")
                except Exception as err:
                    logging.error(f"[Exporter Error] Failed to export schema: {err}", exc_info=True)

            print("\n=======================================================\n")

        finally:
            print("[*] Terminating Chrome orchestrator process...")
            orchestrator.stop()

    asyncio.run(run_live_test_harness())
