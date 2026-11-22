"""Candidate evaluation scanner for promotional text payloads.

Scores arbitrary text chunks (extracted from DOM containers or network
responses) for promotional signal density before costly downstream LLM
extraction is invoked. All domain vocabulary lives exclusively in
``promo_lexicon.json``; this module hardcodes only the structural
category-name contract (``positive_*`` / ``noise_*`` prefixes and the
dedicated casino noise category). Matching is performed on a casefolded,
diacritic-stripped projection of the text while all windows are sliced from
the original text via an index map; multiple occurrences are clustered with
a bounded gap so distant matches (e.g., header navigation vs. footer
disclaimers) never produce a full-document window; `candidate_window` is the
densest cluster (most occurrences, earliest, then shortest on ties) and
`candidate_windows` lists every cluster window in document order.

Raises:
    FileNotFoundError: If the lexicon file does not exist.
    ValueError: If the file is not valid JSON or its root is not an object.
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

__all__ = ["PromoLexiconLoader", "PromotionScanner", "ScanEvaluation"]

logger = logging.getLogger(__name__)

def _normalize_text(text: str) -> str:
    """Casefold and strip combining diacritics from a string.

    Used for lexicon terms (where index alignment is irrelevant).
    """
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text.casefold())
        if not unicodedata.combining(char)
    )

def _normalize_with_index(text: str) -> Tuple[str, List[int]]:
    """Normalize text while recording a per-character index map.

    Each character is casefolded and NFD-decomposed individually; combining
    marks are dropped. Returns the normalized string plus a list mapping
    every normalized-string index to the originating index in ``text``, so
    match spans found in the normalized string can be sliced out of the
    original (accented) source text without corruption.
    """
    normalized_chars: List[str] = []
    index_map: List[int] = []
    for position, char in enumerate(text):
        for folded_char in unicodedata.normalize("NFD", char.casefold()):
            if unicodedata.combining(folded_char):
                continue
            normalized_chars.append(folded_char)
            index_map.append(position)
    return "".join(normalized_chars), index_map

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
        """Compile one lexicon term into a diacritic-insensitive matcher.

        Each letter is rendered as a character class accepting both its
        original (possibly accented) casefolded codepoint and its unaccented
        base. A single pattern therefore matches the raw lexicon term as well
        as the scanner's diacritic-stripped, casefolded text projection.
        Plain characters (and multi-codepoint casefold expansions) are
        emitted as escaped literals and rely on ``re.IGNORECASE`` for case.
        """
        parts: List[str] = []
        for char in term:
            folded = char.casefold()
            base = _normalize_text(char)
            if (
                char.isspace()
                or len(folded) != 1
                or len(base) != 1
                or folded == base
            ):
                parts.append(re.escape(char))
                continue
            parts.append(f"[{re.escape(folded)}{re.escape(base)}]")
        pattern = re.compile(rf"(?<!\w){''.join(parts)}(?!\w)", re.IGNORECASE)
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
    candidate_windows: List[str]
    window_word_count: int
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
            "candidate_windows": list(self.candidate_windows),
            "window_word_count": self.window_word_count,
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
        max_cluster_gap_chars: Maximum gap in characters between positive
            match spans for them to belong to the same candidate cluster.
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
        max_cluster_gap_chars: int = 240,
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
            "max_cluster_gap_chars",
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
        self._max_cluster_gap_chars = max_cluster_gap_chars
        self._density_normalizer_words = float(density_normalizer_words)

    @staticmethod
    def _collect_matches(
        normalized_text: str,
        index_map: List[int],
        category_terms: Dict[str, List[Tuple[str, re.Pattern[str]]]],
    ) -> Tuple[Dict[str, List[str]], List[Tuple[int, int]]]:
        """Find every occurrence of each term per category.

        Returns distinct matched terms per category plus the span of every
        occurrence mapped back to original-text indices via ``index_map``.
        """
        matches: Dict[str, List[str]] = {}
        spans: List[Tuple[int, int]] = []
        for category, term_patterns in category_terms.items():
            for term, pattern in term_patterns:
                occurrences = [
                    (index_map[found.start()], index_map[found.end() - 1] + 1)
                    for found in pattern.finditer(normalized_text)
                ]
                if not occurrences:
                    continue
                matches.setdefault(category, [])
                if term not in matches[category]:
                    matches[category].append(term)
                spans.extend(occurrences)
        return matches, spans

    @staticmethod
    def _cluster_spans(
        spans: List[Tuple[int, int]], max_gap: int
    ) -> List[Tuple[int, int, int]]:
        """Merge sorted match spans into clusters of nearby occurrences.

        Two spans belong to the same cluster when the gap between them is
        at most ``max_gap`` characters. Returns clusters as
        ``(start, end, occurrence_count)`` tuples in ascending order.
        """
        if not spans:
            return []
        ordered = sorted(spans)
        clusters: List[Tuple[int, int, int]] = []
        start, end, count = ordered[0][0], ordered[0][1], 1
        for span_start, span_end in ordered[1:]:
            if span_start - end <= max_gap:
                end = max(end, span_end)
                count += 1
            else:
                clusters.append((start, end, count))
                start, end, count = span_start, span_end, 1
        clusters.append((start, end, count))
        return clusters

    @staticmethod
    def _extract_window(
        text: str, start: int, end: int, context_chars: int
    ) -> str:
        """Slice one bounded window around a single cluster span."""
        window_start = max(0, start - context_chars)
        window_end = min(len(text), end + context_chars)
        return text[window_start:window_end].strip()

    def evaluate_text(self, text: Optional[str]) -> ScanEvaluation:
        """Evaluate a single text payload and return a structured outcome.

        Raises:
            TypeError: If ``text`` is neither a string nor ``None``.
        """
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise TypeError("evaluate_text expects a string payload or None")
        normalized, index_map = _normalize_with_index(text)
        word_count = len(re.findall(r"[^\W_]+", text, flags=re.UNICODE))

        positive_matches, positive_spans = self._collect_matches(
            normalized, index_map, self._lexicon.positive_terms
        )
        noise_matches, _ = self._collect_matches(
            normalized, index_map, self._lexicon.noise_terms
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

        clusters = self._cluster_spans(positive_spans, self._max_cluster_gap_chars)
        candidate_windows = [
            self._extract_window(text, start, end, self._window_context_chars)
            for start, end, _ in clusters
        ]
        if clusters:
            best_start, best_end, _ = max(
                clusters,
                key=lambda cluster: (
                    cluster[2],
                    -cluster[0],
                    -(cluster[1] - cluster[0]),
                ),
            )
            candidate_window = self._extract_window(
                text, best_start, best_end, self._window_context_chars
            )
            window_word_count = len(
                re.findall(
                    r"[^\W_]+", text[best_start:best_end], flags=re.UNICODE
                )
            )
        else:
            candidate_window = ""
            window_word_count = 0

        density = positive_weight / max(
            1.0, window_word_count / self._density_normalizer_words
        )
        total_weight = positive_weight + noise_weight
        purity = positive_weight / total_weight if total_weight > 0 else 0.0
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
            candidate_window=candidate_window,
            candidate_windows=candidate_windows,
            window_word_count=window_word_count,
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
