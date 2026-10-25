import json
import logging
import math
import os
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from browser_orchestrator import ChromeOrchestrator
from cdp_interceptor import CDPNetworkInterceptor, NetworkEvent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class ClassificationResult:
    url: str
    score: float
    category_scores: Dict[str, float]
    odds_count: int
    matched_keywords: Set[str] = field(default_factory=set)
    is_grey_zone: bool = False
    ambiguous_keys: List[str] = field(default_factory=list)


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
            # Check string representation of numbers
            if re.match(r"^-?\d+(\.\d+)?$", node.strip()):
                try:
                    numeric_values.append(float(node.strip()))
                except ValueError:
                    pass

    def _is_decimal_odds(self, val: float) -> bool:
        # Standard decimal odds range between 1.01 and 100.0 with typical precision
        return 1.01 <= val <= 100.0 and round(val, 2) == val

    def _is_american_odds(self, val: float) -> bool:
        # American odds are typically integers <= -100 or >= +100
        return (val <= -100 or val >= 100) and val == math.floor(val) and abs(val) <= 50000

    def score_payload(self, url: str, payload_str: str) -> ClassificationResult:
        if not payload_str or not payload_str.strip():
            return ClassificationResult(
                url=url, score=0.0, category_scores={}, odds_count=0
            )

        try:
            data = json.loads(payload_str)
        except Exception:
            # Non-JSON payloads are immediately rejected from domain scoring
            return ClassificationResult(
                url=url, score=0.0, category_scores={}, odds_count=0
            )

        keys_found: List[str] = []
        numeric_values: List[float] = []
        array_lengths: List[int] = []

        self._inspect_node(data, keys_found, numeric_values, array_lengths)

        if not keys_found and not numeric_values:
            return ClassificationResult(
                url=url, score=0.0, category_scores={}, odds_count=0
            )

        matched_keywords: Set[str] = set()
        category_matches: Dict[str, int] = {
            cat: 0 for cat in self.lexicon_mgr.POSITIVE_CATEGORIES | self.lexicon_mgr.NOISE_CATEGORIES
        }

        # Match keys against lexicon
        for key in keys_found:
            # Direct match or substring match for key tokens
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

        # Calculate Odds Heuristics
        decimal_odds_count = sum(1 for v in numeric_values if self._is_decimal_odds(v))
        american_odds_count = sum(1 for v in numeric_values if self._is_american_odds(v))
        total_odds_count = decimal_odds_count + american_odds_count

        # Structural Heuristics (Repeated collections/arrays indicative of markets or events)
        significant_arrays = [l for l in array_lengths if l >= 2]
        array_repetition_score = min(len(significant_arrays) * 5.0, 25.0)

        # Base Lexical Scoring
        pos_score = sum(
            category_matches[cat] * 12.0 for cat in self.lexicon_mgr.POSITIVE_CATEGORIES
        )
        noise_score = sum(
            category_matches[cat] * 15.0 for cat in self.lexicon_mgr.NOISE_CATEGORIES
        )

        odds_score = min(total_odds_count * 4.0, 35.0)

        # Aggregate raw score
        raw_score = pos_score + odds_score + array_repetition_score - noise_score

        # Normalize score between 0.0 and 100.0
        final_score = max(0.0, min(100.0, raw_score))

        # Ambiguity / Grey Zone Detection Logic
        # Grey zone: High numeric odds or repeating arrays, low noise, but low recognized positive vocabulary
        has_structural_relevance = total_odds_count >= 3 or len(significant_arrays) >= 2
        low_vocabulary = pos_score < 10.0
        low_noise = noise_score < 10.0

        is_grey_zone = has_structural_relevance and low_vocabulary and low_noise

        ambiguous_keys: List[str] = []
        if is_grey_zone:
            # Collect unclassified keys for potential expansion
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
            score=round(final_score, 2),
            category_scores=cat_score_dict,
            odds_count=total_odds_count,
            matched_keywords=matched_keywords,
            is_grey_zone=is_grey_zone,
            ambiguous_keys=ambiguous_keys[:20],  # Top candidates
        )

    def propose_terms_for_grey_zone(
        self,
        result: ClassificationResult,
        llm_callback: Optional[Callable[[List[str]], List[str]]] = None,
    ) -> List[str]:
        if not result.is_grey_zone or not result.ambiguous_keys:
            return []

        filtered_candidates = [
            k for k in result.ambiguous_keys
            if not self.lexicon_mgr.is_generic_or_blacklisted(k)
        ]

        if llm_callback and filtered_candidates:
            try:
                proposed = llm_callback(filtered_candidates)
                return [
                    p for p in proposed
                    if not self.lexicon_mgr.is_generic_or_blacklisted(p)
                ]
            except Exception as err:
                logging.error(f"[Classifier] Error executing LLM callback: {err}")
                return []

        return filtered_candidates


if __name__ == "__main__":
    import asyncio

    async def run_live_test_harness() -> None:
        print("\n=======================================================")
        print("  LIVE SPORTSBOOK API HEURISTIC CLASSIFIER HARNESS")
        print("=======================================================\n")

        classifier = HeuristicClassifier("sportsbook_lexicon.json")
        orchestrator = ChromeOrchestrator(
            profile_name="heuristic_harness_profile", port=9222, headless=False
        )

        print("[*] Launching browser instance...")
        orchestrator.start(timeout=15.0)

        try:
            ws_url = orchestrator.get_cdp_url()
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
            print("[*] Running Heuristic Scoring Engine on JSON responses...\n")

            results: List[ClassificationResult] = []
            for event in events:
                if event.response_body:
                    res = classifier.score_payload(event.url, event.response_body)
                    if res.score > 0.0 or res.is_grey_zone:
                        results.append(res)

            # Sort candidate endpoints by relevance score descending
            results.sort(key=lambda r: r.score, reverse=True)

            print("=======================================================")
            print("         SCORED CANDIDATE ENDPOINTS SUMMARY            ")
            print("=======================================================")

            if not results:
                print("[-] No sportsbook API candidate endpoints detected in captured traffic.")
            else:
                for idx, res in enumerate(results[:15], start=1):
                    print(f"\n[{idx}] Score: {res.score}/100 | Odds Detected: {res.odds_count}")
                    print(f"    URL: {res.url[:110]}...")
                    if res.matched_keywords:
                        print(f"    Matched Terms: {', '.join(sorted(res.matched_keywords)[:8])}")
                    if res.is_grey_zone:
                        print(f"    [!] GREY ZONE DETECTED - Candidate terms: {res.ambiguous_keys[:5]}")

            print("\n=======================================================\n")

        finally:
            print("[*] Terminating Chrome orchestrator process...")
            orchestrator.stop()

    asyncio.run(run_live_test_harness())
