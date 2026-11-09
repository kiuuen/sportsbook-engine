"""Candidate evaluation scanner for promotional text payloads.

Scores arbitrary text chunks (extracted from DOM containers or network
responses) for promotional signal density before costly downstream LLM
extraction is invoked. All domain vocabulary lives exclusively in
``promo_lexicon.json``; this module hardcodes only the structural
category-name contract (``positive_*`` / ``noise_*`` prefixes and the
dedicated casino noise category).
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

__all__ = ["PromoLexiconLoader", "PromotionScanner", "ScanEvaluation"]

logger = logging.getLogger(__name__)


class PromoLexiconLoader:
    """Loads and serves the promotional lexicon from its JSON source file.

    Categories prefixed with ``positive_`` contribute scoring evidence;
    categories prefixed with ``noise_`` contribute penalties. The casino
    category is tracked separately so the scanner can discriminate
    sports-betting promotions from casino/table-game content.

    Raises:
        FileNotFoundError: If the lexicon file does not exist.
        ValueError: If the file is not valid JSON or its root is not an object.
    """

    POSITIVE_PREFIX: str = "positive_"
    NOISE_PREFIX: str = "noise_"
    CASINO_CATEGORY: str = "noise_casino_gaming"

    def __init__(
        self,
        lexicon_path: Union[str, pathlib.Path] = pathlib.Path(__file__)
        .with_name("promo_lexicon.json"),
    ) -> None:
        self.lexicon_path: pathlib.Path = pathlib.Path(lexicon_path)
        self.positive_terms: Dict[str, List[Tuple[str, re.Pattern[str]]]] = {}
        self.noise_terms: Dict[str, List[Tuple[str, re.Pattern[str]]]] = {}
        self.load_lexicon()

    @staticmethod
    def _compile_term(term: str) -> Tuple[str, re.Pattern[str]]:
        """Compile one lexicon term into a word-boundary regex matcher."""
        pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
        return term, pattern

    def load_lexicon(self) -> None:
        """Load and partition lexicon terms into positive and noise buckets."""
        if not self.lexicon_path.exists():
            raise FileNotFoundError(
                f"Promotional lexicon file not found at "
                f"{self.lexicon_path.resolve()}"
            )
        try:
            with open(self.lexicon_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as err:
            raise ValueError(
                f"Promotional lexicon file {self.lexicon_path} is not valid "
                f"JSON: {err}"
            ) from err
        if not isinstance(data, dict):
            raise ValueError(
                f"Promotional lexicon file {self.lexicon_path} must contain a "
                "JSON object mapping category names to term lists."
            )
        for category, terms in data.items():
            if not isinstance(terms, list):
                logger.warning(
                    "Skipping promotional lexicon category '%s': expected a "
                    "list of terms.",
                    category,
                )
                continue
            compiled = [self._compile_term(str(term)) for term in terms]
            if category.startswith(self.POSITIVE_PREFIX):
                self.positive_terms[category] = compiled
            elif category.startswith(self.NOISE_PREFIX):
                self.noise_terms[category] = compiled
            else:
                logger.warning(
                    "Skipping promotional lexicon category '%s': does not "
                    "start with '%s' or '%s'.",
                    category,
                    self.POSITIVE_PREFIX,
                    self.NOISE_PREFIX,
                )


@dataclass
class ScanEvaluation:
    """Structured outcome of evaluating one candidate text payload."""

    relevance_score: float
    matched_categories: Dict[str, List[str]]
    matched_noise_terms: List[str]
    disqualifiers: List[str]
    is_viable: bool
    candidate_window: str
    word_count: int
    source_text_length: int

    def to_dict(self) -> Dict[str, Any]:
        """Export a JSON-safe dictionary representation."""
        return {
            "relevance_score": self.relevance_score,
            "matched_categories": {
                category: list(terms)
                for category, terms in self.matched_categories.items()
            },
            "matched_noise_terms": list(self.matched_noise_terms),
            "disqualifiers": list(self.disqualifiers),
            "is_viable": self.is_viable,
            "candidate_window": self.candidate_window,
            "word_count": self.word_count,
            "source_text_length": self.source_text_length,
        }


class PromotionScanner:
    """Evaluates text chunks for promotional signal density and viability.

    The scanner computes a relevance score from positive promotional
    vocabulary minus noise penalties, detects casino-dominant content,
    and decides whether a payload carries enough operational wagering
    signals to justify downstream LLM extraction.

    Args:
        lexicon: Optional pre-loaded lexicon; a default loader backed by
            ``promo_lexicon.json`` is created when omitted.
        viability_threshold: Minimum relevance score for a viable payload.
        min_positive_categories: Minimum distinct positive categories hit.
        min_positive_terms: Minimum total distinct positive terms matched.
        noise_penalty_factor: Weight applied to each generic noise hit.
        casino_penalty_factor: Weight applied to each casino noise hit.
        window_context_chars: Characters of context padded around matched
            positive spans when isolating the candidate window.
        density_normalizer_words: Word count at which one positive hit
            represents full signal density.
    """

    def __init__(
        self,
        lexicon: Optional[PromoLexiconLoader] = None,
        *,
        viability_threshold: float = 0.35,
        min_positive_categories: int = 2,
        min_positive_terms: int = 3,
        noise_penalty_factor: float = 1.0,
        casino_penalty_factor: float = 1.5,
        window_context_chars: int = 120,
        density_normalizer_words: float = 40.0,
    ) -> None:
        if not isinstance(viability_threshold, (int, float)):
            raise TypeError("viability_threshold must be a number")
        if not 0.0 <= float(viability_threshold) <= 1.0:
            raise ValueError("viability_threshold must be within [0, 1]")
        for name in (
            "min_positive_categories",
            "min_positive_terms",
            "window_context_chars",
        ):
            value = locals()[name]
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "noise_penalty_factor",
            "casino_penalty_factor",
            "density_normalizer_words",
        ):
            value = locals()[name]
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be a positive number")
        self._lexicon = lexicon if lexicon is not None else PromoLexiconLoader()
        if not isinstance(self._lexicon, PromoLexiconLoader):
            raise TypeError("lexicon must be a PromoLexiconLoader instance")
        self._viability_threshold = float(viability_threshold)
        self._min_positive_categories = min_positive_categories
        self._min_positive_terms = min_positive_terms
        self._noise_penalty_factor = float(noise_penalty_factor)
        self._casino_penalty_factor = float(casino_penalty_factor)
        self._window_context_chars = window_context_chars
        self._density_normalizer_words = float(density_normalizer_words)

    @staticmethod
    def _collect_matches(
        normalized_text: str,
        category_terms: Dict[str, List[Tuple[str, re.Pattern[str]]]],
    ) -> Tuple[Dict[str, List[str]], List[Tuple[int, int]]]:
        """Find distinct matched terms per category plus their char spans."""
        matches: Dict[str, List[str]] = {}
        spans: List[Tuple[int, int]] = []
        for category, term_patterns in category_terms.items():
            for term, pattern in term_patterns:
                found = pattern.search(normalized_text)
                if found is None:
                    continue
                matches.setdefault(category, [])
                if term not in matches[category]:
                    matches[category].append(term)
                spans.append((found.start(), found.end()))
        return matches, spans

    @staticmethod
    def _extract_window(
        text: str, spans: List[Tuple[int, int]], context_chars: int
    ) -> str:
        """Isolate the text window covering all positive match spans."""
        if not spans:
            return ""
        start = max(0, min(span[0] for span in spans) - context_chars)
        end = min(len(text), max(span[1] for span in spans) + context_chars)
        return text[start:end].strip()

    def evaluate_text(self, text: Optional[str]) -> ScanEvaluation:
        """Evaluate a single text payload and return a structured outcome.

        Raises:
            TypeError: If ``text`` is neither a string nor ``None``.
        """
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise TypeError("evaluate_text expects a string payload or None")
        normalized = text.lower()
        word_count = len(re.findall(r"[a-z0-9]+", normalized))

        positive_matches, positive_spans = self._collect_matches(
            normalized, self._lexicon.positive_terms
        )
        noise_matches, _ = self._collect_matches(
            normalized, self._lexicon.noise_terms
        )

        positive_weight = float(
            sum(len(terms) for terms in positive_matches.values())
        )
        noise_weight = 0.0
        for category, terms in noise_matches.items():
            factor = (
                self._casino_penalty_factor
                if category == PromoLexiconLoader.CASINO_CATEGORY
                else self._noise_penalty_factor
            )
            noise_weight += len(terms) * factor

        disqualifiers: List[str] = []
        casino_hits = len(noise_matches.get(PromoLexiconLoader.CASINO_CATEGORY, []))
        if casino_hits > 0 and casino_hits >= positive_weight:
            disqualifiers.append("casino_dominant")
        if noise_weight > positive_weight:
            disqualifiers.append("noise_dominant")
        if (
            len(positive_matches) < self._min_positive_categories
            or positive_weight < self._min_positive_terms
        ):
            disqualifiers.append("insufficient_signals")

        total_weight = positive_weight + noise_weight
        purity = positive_weight / total_weight if total_weight > 0 else 0.0
        density = positive_weight / max(
            1.0, word_count / self._density_normalizer_words
        )
        relevance_score = round(
            max(0.0, min(1.0, purity * min(1.0, density))), 4
        )
        is_viable = not disqualifiers and relevance_score >= self._viability_threshold
        matched_noise_terms = [
            term
            for terms in noise_matches.values()
            for term in terms
        ]
        return ScanEvaluation(
            relevance_score=relevance_score,
            matched_categories=positive_matches,
            matched_noise_terms=matched_noise_terms,
            disqualifiers=disqualifiers,
            is_viable=is_viable,
            candidate_window=self._extract_window(
                text, positive_spans, self._window_context_chars
            ),
            word_count=word_count,
            source_text_length=len(text),
        )

    def evaluate_chunks(
        self, chunks: List[str]
    ) -> List[ScanEvaluation]:
        """Evaluate a batch of text chunks, preserving input order.

        Raises:
            TypeError: If ``chunks`` is not a list/tuple of strings.
        """
        if not isinstance(chunks, (list, tuple)):
            raise TypeError("evaluate_chunks expects a list of strings")
        evaluations: List[ScanEvaluation] = []
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, str):
                raise TypeError(
                    f"evaluate_chunks expects strings; item {index} is "
                    f"{type(chunk).__name__}"
                )
            evaluations.append(self.evaluate_text(chunk))
        return evaluations
