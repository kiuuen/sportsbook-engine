"""Lexicon management for the heuristic classification pipeline.

This module isolates the sportsbook vocabulary loader so that scoring logic,
adapters, and pipeline consumers can share a single, dependency-light
definition without importing the legacy monolith. Domain-wide vocabulary
lives exclusively in ``sportsbook_lexicon.json``; no keyword sets are
hardcoded here beyond the structural category-name contract.
"""

import json
import logging
import pathlib
from typing import Dict, Set

logger = logging.getLogger(__name__)


class LexiconManager:
    """Loads and serves the sportsbook lexicon from its JSON source file.

    Positive categories contribute scoring evidence, noise categories
    contribute penalties, and the generic blacklist filters out keys that
    carry no domain signal.
    """

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
        """Load and partition lexicon terms into positive and noise buckets.

        Raises:
            FileNotFoundError: If the lexicon file does not exist.
            ValueError: If the lexicon file is not valid JSON or its root
                is not a JSON object.
        """
        if not self.lexicon_path.exists():
            raise FileNotFoundError(
                f"Lexicon file not found at {self.lexicon_path.resolve()}"
            )

        try:
            with open(self.lexicon_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as err:
            raise ValueError(
                f"Lexicon file {self.lexicon_path} is not valid JSON: {err}"
            ) from err

        if not isinstance(data, dict):
            raise ValueError(
                f"Lexicon file {self.lexicon_path} must contain a JSON object "
                "mapping category names to term lists."
            )

        for category, terms in data.items():
            if category == "pagination_scaling":
                continue  # Numeric scaling config consumed by the replay executor.
            if not isinstance(terms, list):
                logger.warning(
                    "Skipping lexicon category '%s': expected a list of terms.",
                    category,
                )
                continue
            term_set = {str(t).lower() for t in terms}
            if category in self.POSITIVE_CATEGORIES:
                self.positive_keywords[category] = term_set
                self.all_positive_terms.update(term_set)
            elif category in self.NOISE_CATEGORIES:
                self.noise_keywords[category] = term_set
                self.all_noise_terms.update(term_set)

    def is_generic_or_blacklisted(self, term: str) -> bool:
        """Return True if the term is a generic key or too short to be meaningful."""
        normalized = term.strip().lower()
        return normalized in self.GENERIC_BLACKLIST or len(normalized) <= 2
