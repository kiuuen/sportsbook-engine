"""Heuristic scoring engine for captured network payloads.

This module isolates the payload-scoring logic of the classification
pipeline so that exporters and pipeline consumers can share a single
implementation without importing the legacy monolith.
"""

import json
import logging
import math
import pathlib
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Set, Tuple

from cdp_interceptor import NetworkEvent
from heuristic_classifier.deducer import RouteTemplateDeducer
from heuristic_classifier.lexicon import LexiconManager
from heuristic_classifier.models import ClassificationResult

logger = logging.getLogger(__name__)

class HeuristicClassifier:
    """Scores captured network payloads for sportsbook API relevance.

    Combines lexicon-driven category matching, structural traversal of
    relational JSON payloads, odds-value detection, and URL/template
    signals into a composite 0-100 relevance score, and derives the
    endpoint role, volatility tier, produced identifiers, and canonical
    field mappings for downstream persistence and replay.
    """

    CANONICAL_FIELD_ROLES: Dict[str, Tuple[str, ...]] = {
        "home_team": ("home_team", "home_competitor"),
        "away_team": ("away_team", "away_competitor"),
        "league": ("league", "league_id", "tournament", "tournament_id", "competition", "championship"),
        "sport": ("sport", "sport_id", "sport_key", "sport_name"),
        "event_id": ("event_id", "match_id", "fixture_id", "game_id"),
        "market_id": ("market_id", "market_type", "market_name"),
        "odds_value": ("odds", "price", "prices", "decimal_odds", "over_price", "under_price"),
        "commence_time": ("commence_time", "start_time", "kickoff", "scheduled_at", "event_date"),
    }

    CATEGORY_WEIGHTS: Dict[str, float] = {
        "betting_markets_and_odds": 36.0,
        "sports_event_metadata": 12.0,
        "live_state_and_settlement": 18.0,
        "provider_identifiers_and_abbreviations": 12.0,
    }

    STYLE_KEY_PATTERN: re.Pattern = re.compile(
        r".*(color|margin[-_]?(?:top|bottom|left|right)|padding|border|radius|shadow|font|typography|palette|theme|style|dimension|flex|grid|align|justify).*",
        re.IGNORECASE,
    )
    COLOR_MODEL_PATTERN: re.Pattern = re.compile(
        r"^#(?:[0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$|^rgba?\(|^hsla?\(",
        re.IGNORECASE,
    )
    DIMENSION_PATTERN: re.Pattern = re.compile(
        r"^\d+(\.\d+)?(px|em|rem|vh|vw|pt)$",
        re.IGNORECASE,
    )

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
                logger.warning(f"Could not load {self.filters_path}: {e}")
        return []

    def _match_key_categories(self, key_str: str) -> Tuple[Set[str], Set[str], Set[str]]:
        tokens = [t for t in re.split(r"[_\-\s\.]", key_str) if t]
        tokens_set = set(tokens)
        matched_pos_cats: Set[str] = set()
        matched_noise_cats: Set[str] = set()
        matched_terms: Set[str] = set()

        for cat, terms in self.lexicon_mgr.positive_keywords.items():
            for term in terms:
                if (len(term) <= 3 and term in tokens_set) or (len(term) > 3 and (term in key_str or term in tokens_set)):
                    matched_pos_cats.add(cat)
                    matched_terms.add(term)

        for cat, terms in self.lexicon_mgr.noise_keywords.items():
            for term in terms:
                if (len(term) <= 3 and term in tokens_set) or (len(term) > 3 and (term in key_str or term in tokens_set)):
                    matched_noise_cats.add(cat)

        return matched_pos_cats, matched_noise_cats, matched_terms

    def _traverse_relational_structures(
        self,
        node: Any,
        is_repetitive_context: bool,
        ancestor_positive_categories: Set[str],
        keys_found: List[str],
        structural_odds: List[float],
        array_lengths: List[int],
        contextual_numerics: List[Tuple[str, float]],
        style_tokens_found: List[str],
        parent_key: str = "",
    ) -> None:
        if isinstance(node, dict):
            local_pos_in_keys: Set[str] = set()
            for k in node.keys():
                k_str = str(k).lower()
                keys_found.append(k_str)
                if self.STYLE_KEY_PATTERN.match(k_str):
                    style_tokens_found.append(k_str)
                k_pos, _, _ = self._match_key_categories(k_str)
                local_pos_in_keys.update(k_pos)

            current_pos_categories = ancestor_positive_categories | local_pos_in_keys

            for k, v in node.items():
                k_str = str(k).lower()
                self._traverse_relational_structures(
                    v,
                    is_repetitive_context=is_repetitive_context,
                    ancestor_positive_categories=current_pos_categories,
                    keys_found=keys_found,
                    structural_odds=structural_odds,
                    array_lengths=array_lengths,
                    contextual_numerics=contextual_numerics,
                    style_tokens_found=style_tokens_found,
                    parent_key=k_str,
                )
        elif isinstance(node, list):
            array_len = len(node)
            array_lengths.append(array_len)
            is_homogeneous_repetitive = array_len >= 2 and all(isinstance(item, (dict, list)) for item in node)
            child_repetitive_flag = is_repetitive_context or is_homogeneous_repetitive

            for item in node:
                self._traverse_relational_structures(
                    item,
                    is_repetitive_context=child_repetitive_flag,
                    ancestor_positive_categories=ancestor_positive_categories,
                    keys_found=keys_found,
                    structural_odds=structural_odds,
                    array_lengths=array_lengths,
                    contextual_numerics=contextual_numerics,
                    style_tokens_found=style_tokens_found,
                    parent_key=parent_key,
                )
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            val = float(node)
            if parent_key:
                contextual_numerics.append((parent_key, val))
            if is_repetitive_context and bool(ancestor_positive_categories):
                if self._is_decimal_odds(val) or self._is_american_odds(val):
                    structural_odds.append(val)
        elif isinstance(node, str):
            clean_str = node.strip()
            if self.COLOR_MODEL_PATTERN.match(clean_str) or self.DIMENSION_PATTERN.match(clean_str):
                style_tokens_found.append(clean_str)
            elif parent_key and self.STYLE_KEY_PATTERN.match(parent_key):
                style_tokens_found.append(clean_str)

            if re.match(r"^-?\d+(\.\d+)?$", clean_str):
                try:
                    val = float(clean_str)
                    if parent_key:
                        contextual_numerics.append((parent_key, val))
                    if is_repetitive_context and bool(ancestor_positive_categories):
                        if self._is_decimal_odds(val) or self._is_american_odds(val):
                            structural_odds.append(val)
                except ValueError:
                    pass

    def _is_decimal_odds(self, val: float) -> bool:
        return 1.01 <= val <= 100.0 and round(val, 2) == val

    def _is_american_odds(self, val: float) -> bool:
        return (val <= -100 or val >= 100) and val == math.floor(val) and abs(val) <= 50000

    def _infer_field_mappings(
        self,
        keys_found: List[str],
        contextual_numerics: List[Tuple[str, float]],
    ) -> Dict[str, str]:
        """Infer a lightweight canonical-to-raw field mapping from observed keys.

        Roles are resolved by matching raw keys (or their tokens) against the
        sportsbook lexicon. Unknown or generic keys are ignored; the first
        matching raw key wins for each canonical role.
        """
        mappings: Dict[str, str] = {}
        claimed_keys: Set[str] = set()

        for canonical_role, lexicon_terms in self.CANONICAL_FIELD_ROLES.items():
            for raw_key in keys_found:
                if raw_key in claimed_keys or self.lexicon_mgr.is_generic_or_blacklisted(raw_key):
                    continue
                key_tokens = {t for t in re.split(r"[_\-\s\.]", raw_key) if t}
                if any(term == raw_key or term in key_tokens for term in lexicon_terms):
                    mappings[canonical_role] = raw_key
                    claimed_keys.add(raw_key)
                    break

        # Fallback: infer odds_value from numeric values under odds-related keys
        if "odds_value" not in mappings:
            core_terms = self.lexicon_mgr.positive_keywords.get("betting_markets_and_odds", set())
            for parent_key, val in contextual_numerics:
                if self._is_decimal_odds(val) or self._is_american_odds(val):
                    key_tokens = set(re.split(r"[_\-\s\.]", parent_key))
                    if any(term in parent_key or term in key_tokens for term in core_terms):
                        mappings["odds_value"] = parent_key
                        break

        return mappings

    def _extract_produced_identifiers(self, keys_found: List[str]) -> List[str]:
        identifier_keys: Set[str] = set()
        id_patterns = (r".*_?id$", r".*_?uuid$", r".*_?code$", r".*_?key$", r".*_?slug$")
        for key in keys_found:
            k_clean = key.strip().lower()
            if self.lexicon_mgr.is_generic_or_blacklisted(k_clean):
                continue
            if any(re.match(pattern, k_clean) for pattern in id_patterns):
                identifier_keys.add(k_clean)
        return sorted(list(identifier_keys))

    def _deduce_role_and_volatility(
        self,
        url_template: str,
        total_odds_count: int,
        call_count: int,
        time_span_seconds: float,
        pos_score: float,
        significant_arrays_count: int,
        is_websocket: bool = False,
    ) -> Tuple[str, str]:
        has_path_params = bool(re.search(r"\{[^}]+\}", url_template))

        # New: force fixtures_list for list-oriented paths
        list_keywords = ("livenow", "upcoming", "highlights", "fixtures", "liveevents", "matches")
        if any(kw in url_template.lower() for kw in list_keywords):
            role = "fixtures_list"
        else:
            # Existing role logic
            if total_odds_count >= 10 or (has_path_params and total_odds_count >= 3):
                role = "odds_detail"
            elif significant_arrays_count >= 1 and (total_odds_count >= 1 or pos_score >= 20.0):
                role = "fixtures_list"
            else:
                role = "catalog_root"

        # Volatility tier classification logic
        if is_websocket or (call_count >= 2 and time_span_seconds > 0.0 and (time_span_seconds / call_count) <= 5.0) or role == "odds_detail":
            volatility = "realtime"
        elif role == "fixtures_list" or has_path_params:
            volatility = "semi_volatile"
        else:
            volatility = "static"

        return role, volatility

    def score_payload(
        self,
        url: str,
        payload_str: str,
        call_count: int = 1,
        time_span_seconds: float = 0.0,
        sample_event: Optional[NetworkEvent] = None,
    ) -> ClassificationResult:
        url_template = RouteTemplateDeducer.deduce_template(url)

        is_websocket = (
            sample_event is not None
            and (sample_event.resource_type == "WebSocket" or sample_event.method in ("WS_RECV", "WS_SENT"))
        )

        clean_payload = payload_str.strip()
        json_match = re.search(r"(\[.*\]|\{.*\})", clean_payload, re.DOTALL)
        if json_match:
            clean_payload = json_match.group(1)

        if not clean_payload:
            return ClassificationResult(
                url=url,
                url_template=url_template,
                score=0.0,
                category_scores={},
                odds_count=0,
                call_count=call_count,
                time_span_seconds=time_span_seconds,
                sample_event=sample_event,
                is_viable=False,
            )

        try:
            data = json.loads(clean_payload)
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
                is_viable=False,
            )

        keys_found: List[str] = []
        structural_odds: List[float] = []
        array_lengths: List[int] = []
        contextual_numerics: List[Tuple[str, float]] = []
        style_tokens_found: List[str] = []

        self._traverse_relational_structures(
            node=data,
            is_repetitive_context=False,
            ancestor_positive_categories=set(),
            keys_found=keys_found,
            structural_odds=structural_odds,
            array_lengths=array_lengths,
            contextual_numerics=contextual_numerics,
            style_tokens_found=style_tokens_found,
            parent_key="",
        )

        if not keys_found and not structural_odds:
            return ClassificationResult(
                url=url,
                url_template=url_template,
                score=0.0,
                category_scores={},
                odds_count=0,
                call_count=call_count,
                time_span_seconds=time_span_seconds,
                sample_event=sample_event,
                is_viable=False,
            )

        matched_keywords: Set[str] = set()
        core_betting_matched_terms: Set[str] = set()
        category_matches: Dict[str, int] = {
            cat: 0 for cat in self.lexicon_mgr.POSITIVE_CATEGORIES | self.lexicon_mgr.NOISE_CATEGORIES
        }

        for key in keys_found:
            pos_cats, noise_cats, terms = self._match_key_categories(key)
            for c in pos_cats:
                category_matches[c] += 1
            for c in noise_cats:
                category_matches[c] += 1
            matched_keywords.update(terms)
            if "betting_markets_and_odds" in pos_cats:
                core_betting_matched_terms.update(
                    terms.intersection(self.lexicon_mgr.positive_keywords.get("betting_markets_and_odds", set()))
                )

        total_odds_count = len(structural_odds)

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

        style_token_count = len(style_tokens_found)
        style_noise_penalty = style_token_count * 20.0

        casino_matches = category_matches.get("noise_casino_and_promotions", 0)
        has_sports_context = len(core_betting_matched_terms) > 0 or category_matches.get("sports_event_metadata", 0) > 0
        casino_penalty = (casino_matches * 30.0) if not has_sports_context else 0.0

        raw_score = (
            pos_score
            + odds_score
            + contextual_odds_bonus
            + array_repetition_score
            + frequency_bonus
            + temporal_polling_bonus
            - noise_score
            - url_noise_penalty
            - style_noise_penalty
            - casino_penalty
        )

        final_score = max(0.0, min(100.0, raw_score))

        produced_params = self._extract_produced_identifiers(keys_found)
        field_mappings = self._infer_field_mappings(keys_found, contextual_numerics)
        role, volatility = self._deduce_role_and_volatility(
            url_template=url_template,
            total_odds_count=total_odds_count,
            call_count=call_count,
            time_span_seconds=time_span_seconds,
            pos_score=pos_score,
            significant_arrays_count=len(significant_arrays),
            is_websocket=is_websocket,
        )

        total_pos_matches = sum(category_matches[cat] for cat in self.lexicon_mgr.POSITIVE_CATEGORIES)
        total_noise_matches = sum(category_matches[cat] for cat in self.lexicon_mgr.NOISE_CATEGORIES)
        favorable_signal_ratio = total_pos_matches > (total_noise_matches * 1.5) and pos_score >= noise_score
        is_detailed_or_realtime = (role == "odds_detail" or volatility == "realtime")
        significant_arrays = [l for l in array_lengths if l >= 2]
        has_structural_repetition = len(significant_arrays) >= 1

        # Enforce strict zero-scoring for noise categories or non-target entertainment catalogs
        has_entertainment_signals = any(term in url_lower for term in self.lexicon_mgr.all_noise_terms) or any(
            any(term in key for term in self.lexicon_mgr.all_noise_terms) for key in keys_found
        )

        if total_noise_matches > 0 and (not has_sports_context or noise_score >= pos_score):
            final_score = 0.0

        if has_entertainment_signals and not has_sports_context:
            final_score = 0.0

        # Disqualify payloads dominated by UI/theme styling tokens or generic casino catalogs without sports context
        if style_token_count >= 3 or (style_token_count > 0 and not has_sports_context and total_odds_count == 0):
            final_score = 0.0

        if casino_matches > 0 and not has_sports_context:
            final_score = 0.0

        if total_odds_count == 0:
            is_valid_catalog = (
                has_sports_context
                and favorable_signal_ratio
                and not is_detailed_or_realtime
                and (category_matches.get("sports_event_metadata", 0) > 0 or has_structural_repetition or pos_score >= 12.0)
            )
            if not is_valid_catalog:
                final_score = min(final_score, 40.0)

        # Composite Signal-to-Noise Confidence Validation
        active_pos_categories_count = sum(1 for cat in self.lexicon_mgr.POSITIVE_CATEGORIES if category_matches[cat] > 0)

        has_lexical_diversity = active_pos_categories_count >= 2 or len(core_betting_matched_terms) >= 1

        if is_detailed_or_realtime:
            has_adequate_odds = total_odds_count >= 5
        else:
            has_adequate_odds = total_odds_count >= 5 or (has_sports_context and has_structural_repetition)

        is_composite_confident = (
            final_score >= 45.0
            and has_lexical_diversity
            and has_structural_repetition
            and favorable_signal_ratio
            and has_adequate_odds
            and style_token_count < 3
            and (casino_matches == 0 or has_sports_context)
        )

        # Generic payloads failing composite confidence must not enter grey zone
        has_structural_relevance = total_odds_count >= 3 or len(significant_arrays) >= 2
        low_vocabulary = pos_score < 10.0
        low_noise = noise_score < 10.0
        is_grey_zone = (
            has_structural_relevance
            and low_vocabulary
            and low_noise
            and total_noise_matches == 0
            and total_odds_count >= 3
        )

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

        # Determine auxiliary widget tier dynamically based on content signals
        is_auxiliary_widget = (
            total_odds_count < 2
            and len(core_betting_matched_terms) == 0
            and (noise_score > pos_score or not is_composite_confident)
        )

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
            is_viable=is_composite_confident,
            role=role,
            volatility=volatility,
            field_mappings=field_mappings,
            produced_params=produced_params,
            is_auxiliary_widget=is_auxiliary_widget,
        )
