"""Market semantic classification layer for odds normalization.

Maps raw market labels to canonical market groups (moneyline, totals,
spreads, unknown) using a lexicon-driven classifier. Canonical group
vocabularies are built dynamically from terms present in
``sportsbook_lexicon.json``; only generic token patterns are additive.

This module is isolated to market semantic labeling: it contains no
provider adapters and no normalization orchestration.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from heuristic_classifier import LexiconManager
from odds_normalizer.models import CanonicalMarketType
from odds_normalizer.utils import SafePayloadUtils

logger = logging.getLogger("odds_normalizer")

class MarketClassifier:
    """Lexicon-driven semantic classifier mapping raw market labels to canonical groups.

    Canonical group vocabularies are built dynamically from terms present in
    ``sportsbook_lexicon.json``; only generic token patterns are additive.
    """

    _SEED_CANONICAL_TERMS: Dict[str, Set[str]] = {
        CanonicalMarketType.MONEYLINE: {"moneyline", "draw_no_bet", "outright", "to_qualify", "futures"},
        CanonicalMarketType.TOTALS: {"totals", "over_under", "over_price", "under_price"},
        CanonicalMarketType.SPREADS: {
            "spreads", "spread", "spread_value", "handicap", "handicap_value",
            "asian_handicap", "european_handicap", "line", "lines",
        },
    }

    _TOKEN_HINT_RULES: List[Tuple[re.Pattern, str]] = [
        (re.compile(r"\b(over|under|total|totals)\b", re.IGNORECASE), CanonicalMarketType.TOTALS),
        (re.compile(r"\b(handicap|handicaps|spread|spreads)\b", re.IGNORECASE), CanonicalMarketType.SPREADS),
        (re.compile(r"\b(moneyline|ml|winner|win|1x2|match|draw)\b", re.IGNORECASE), CanonicalMarketType.MONEYLINE),
    ]

    def __init__(self, lexicon_mgr: Optional[LexiconManager]) -> None:
        self.lexicon_mgr = lexicon_mgr
        self.canonical_terms: Dict[str, Set[str]] = {
            canonical: set() for canonical in CanonicalMarketType.ALL if canonical != CanonicalMarketType.UNKNOWN
        }
        self._build_vocabularies()

    def _build_vocabularies(self) -> None:
        """Intersect seed terms with the live lexicon so unknown lexicons degrade gracefully."""
        betting_terms: Set[str] = set()
        if self.lexicon_mgr is not None:
            betting_terms = self.lexicon_mgr.positive_keywords.get("betting_markets_and_odds", set())

        for canonical, seed_terms in self._SEED_CANONICAL_TERMS.items():
            if betting_terms:
                self.canonical_terms[canonical] = {t for t in seed_terms if t in betting_terms}
            else:
                logger.warning("[MarketClassifier] Lexicon betting terms unavailable; using seed vocabulary only.")
                self.canonical_terms[canonical] = set(seed_terms)

    def classify(self, market_label: str) -> Tuple[str, float]:
        """Classify a raw market label into a canonical group with a confidence score."""
        if not market_label:
            return CanonicalMarketType.UNKNOWN, 0.0

        tokens = SafePayloadUtils.tokenize_key(market_label)
        token_set = set(tokens)
        label_lower = market_label.lower()

        best_group = CanonicalMarketType.UNKNOWN
        best_hits = 0
        for canonical, terms in self.canonical_terms.items():
            hits = sum(1 for term in terms if term in token_set or term in label_lower)
            if hits > best_hits:
                best_group, best_hits = canonical, hits

        if best_hits > 0:
            return best_group, min(1.0, best_hits / 3.0)

        for pattern, canonical in self._TOKEN_HINT_RULES:
            if pattern.search(label_lower):
                return canonical, 0.4

        return CanonicalMarketType.UNKNOWN, 0.0
