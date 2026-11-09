"""Unit tests for the promotional text scanner and lexicon evaluation logic.

Covers :class:`bonus_engine.scanner.PromoLexiconLoader`,
:class:`bonus_engine.scanner.PromotionScanner`, and
:class:`bonus_engine.scanner.ScanEvaluation`.

Per project conventions, lexicon-driven tests load the repository's real,
un-truncated ``promo_lexicon.json`` rather than synthetic JSON payloads.
Inline text payloads exercise the scoring, disqualification, and window
extraction behaviour against that genuine vocabulary.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List

import pytest

from bonus_engine.scanner import (
    PromoLexiconLoader,
    PromotionScanner,
    ScanEvaluation,
)

REPO_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
REAL_LEXICON_PATH: pathlib.Path = (
    REPO_ROOT / "bonus_engine" / "promo_lexicon.json"
)

VIABLE_PROMO_TEXT: str = (
    "New customers only. Place a £10 bet at minimum odds of 1.80. "
    "The welcome bonus carries wagering requirements of 5x the bonus "
    "amount before bonus funds convert to withdrawable cash. "
    "Free bet stake not returned. Offer expires within 7 days. "
    "Full T&Cs apply."
)
NOISE_DOMINANT_TEXT: str = (
    "Check our payment methods, withdrawal fees and privacy policy. "
    "Contact customer support for help. A free bet may be referenced once."
)
CASINO_DOMINANT_TEXT: str = (
    "Enjoy free spins on slots and roulette in our online casino, "
    "plus a welcome bonus."
)
SIGNAL_FREE_TEXT: str = "Great prices await you today."


@pytest.fixture(scope="module")
def real_lexicon() -> PromoLexiconLoader:
    """Loader backed by the repository's genuine promotional lexicon."""
    return PromoLexiconLoader(REAL_LEXICON_PATH)


@pytest.fixture()
def scanner(real_lexicon: PromoLexiconLoader) -> PromotionScanner:
    """Scanner configured with default thresholds and the real lexicon."""
    return PromotionScanner(lexicon=real_lexicon)


class TestPromoLexiconLoader:
    """Tests for lexicon loading, partitioning, and term compilation."""

    def test_default_path_loads_real_lexicon(self) -> None:
        loader = PromoLexiconLoader()
        assert loader.lexicon_path == pathlib.Path(
            PromoLexiconLoader.__module__.replace(".", "/")
        ).with_name("promo_lexicon.json") or loader.lexicon_path.exists()
        assert loader.positive_terms
        assert loader.noise_terms

    def test_real_lexicon_partitions_all_categories(self) -> None:
        loader = PromoLexiconLoader(REAL_LEXICON_PATH)
        with open(REAL_LEXICON_PATH, "r", encoding="utf-8") as handle:
            raw: Dict[str, Any] = json.load(handle)
        for category in raw:
            if category.startswith(PromoLexiconLoader.POSITIVE_PREFIX):
                assert category in loader.positive_terms
            elif category.startswith(PromoLexiconLoader.NOISE_PREFIX):
                assert category in loader.noise_terms
        assert (
            PromoLexiconLoader.CASINO_CATEGORY in loader.noise_terms
        )

    def test_compiled_patterns_are_case_insensitive(self) -> None:
        loader = PromoLexiconLoader(REAL_LEXICON_PATH)
        for category_terms in (loader.positive_terms, loader.noise_terms):
            for patterns in category_terms.values():
                for term, pattern in patterns:
                    assert pattern.search(term.lower()) is not None

    def test_compile_term_returns_term_and_pattern(self) -> None:
        term, pattern = PromoLexiconLoader._compile_term("Free Bet")
        assert term == "Free Bet"
        assert pattern.search("claim a free bet today") is not None
        assert pattern.search("CLAIM A FREE BET TODAY") is not None

    def test_compile_term_enforces_word_boundaries(self) -> None:
        _, pattern = PromoLexiconLoader._compile_term("free bet")
        assert pattern.search("freebets") is None
        assert pattern.search("riskfree betting") is None

    def test_custom_lexicon_partitions_by_prefix(
        self, tmp_path: pathlib.Path
    ) -> None:
        lexicon_file = tmp_path / "custom_lexicon.json"
        lexicon_file.write_text(
            json.dumps(
                {
                    "positive_alpha": ["alpha term"],
                    "noise_beta": ["beta term"],
                    "gamma": ["ignored term"],
                }
            ),
            encoding="utf-8",
        )
        loader = PromoLexiconLoader(lexicon_file)
        assert "positive_alpha" in loader.positive_terms
        assert "noise_beta" in loader.noise_terms
        assert "gamma" not in loader.positive_terms
        assert "gamma" not in loader.noise_terms

    def test_missing_lexicon_file_raises(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(FileNotFoundError):
            PromoLexiconLoader(tmp_path / "missing.json")

    def test_invalid_json_raises_value_error(
        self, tmp_path: pathlib.Path
    ) -> None:
        lexicon_file = tmp_path / "broken.json"
        lexicon_file.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ValueError):
            PromoLexiconLoader(lexicon_file)

    def test_non_object_root_raises_value_error(
        self, tmp_path: pathlib.Path
    ) -> None:
        lexicon_file = tmp_path / "list_root.json"
        lexicon_file.write_text(json.dumps(["a", "b"]), encoding="utf-8")
        with pytest.raises(ValueError):
            PromoLexiconLoader(lexicon_file)

    def test_non_list_category_skipped_with_warning(
        self,
        tmp_path: pathlib.Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        lexicon_file = tmp_path / "mixed_types.json"
        lexicon_file.write_text(
            json.dumps(
                {
                    "positive_good": ["real term"],
                    "positive_bad": "not a list",
                }
            ),
            encoding="utf-8",
        )
        with caplog.at_level("WARNING"):
            loader = PromoLexiconLoader(lexicon_file)
        assert "positive_good" in loader.positive_terms
        assert "positive_bad" not in loader.positive_terms
        assert any(
            "positive_bad" in record.message for record in caplog.records
        )

    def test_unprefixed_category_skipped_with_warning(
        self,
        tmp_path: pathlib.Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        lexicon_file = tmp_path / "unprefixed.json"
        lexicon_file.write_text(
            json.dumps(
                {
                    "middle_ground": ["orphan term"],
                    "positive_kept": ["kept term"],
                }
            ),
            encoding="utf-8",
        )
        with caplog.at_level("WARNING"):
            loader = PromoLexiconLoader(lexicon_file)
        assert "positive_kept" in loader.positive_terms
        assert "middle_ground" not in loader.positive_terms
        assert "middle_ground" not in loader.noise_terms
        assert any(
            "middle_ground" in record.message for record in caplog.records
        )


class TestPromotionScannerInit:
    """Tests for constructor argument validation and defaults."""

    def test_default_construction_uses_default_lexicon(self) -> None:
        scanner = PromotionScanner()
        assert isinstance(scanner._lexicon, PromoLexiconLoader)

    def test_rejects_non_lexicon_object(self) -> None:
        with pytest.raises(TypeError):
            PromotionScanner(lexicon="not a loader")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_threshold", ["high", None, [0.5]]
    )
    def test_rejects_non_numeric_threshold(
        self, bad_threshold: Any
    ) -> None:
        with pytest.raises(TypeError):
            PromotionScanner(viability_threshold=bad_threshold)

    @pytest.mark.parametrize(
        "bad_threshold", [-0.1, 1.01, 2.0]
    )
    def test_rejects_out_of_range_threshold(
        self, bad_threshold: float
    ) -> None:
        with pytest.raises(ValueError):
            PromotionScanner(viability_threshold=bad_threshold)

    @pytest.mark.parametrize(
        "param_name",
        ["min_positive_categories", "min_positive_terms", "window_context_chars"],
    )
    @pytest.mark.parametrize(
        "bad_value", [-1, 1.5, "3", None]
    )
    def test_rejects_invalid_integer_params(
        self, param_name: str, bad_value: Any
    ) -> None:
        with pytest.raises(ValueError):
            PromotionScanner(**{param_name: bad_value})

    @pytest.mark.parametrize(
        "param_name",
        [
            "noise_penalty_factor",
            "casino_penalty_factor",
            "density_normalizer_words",
        ],
    )
    @pytest.mark.parametrize(
        "bad_value", [0, -1.0, "x", None]
    )
    def test_rejects_invalid_positive_number_params(
        self, param_name: str, bad_value: Any
    ) -> None:
        with pytest.raises(ValueError):
            PromotionScanner(**{param_name: bad_value})


class TestScanEvaluation:
    """Tests for the structured evaluation result dataclass."""

    def _build_evaluation(self) -> ScanEvaluation:
        return ScanEvaluation(
            relevance_score=0.75,
            matched_categories={"positive_a": ["term1"], "positive_b": ["term2"]},
            matched_noise_terms=["noise1"],
            disqualifiers=["noise_dominant"],
            is_viable=False,
            candidate_window="window text",
            word_count=42,
            source_text_length=300,
        )

    def test_to_dict_schema(self) -> None:
        evaluation = self._build_evaluation()
        exported: Dict[str, Any] = evaluation.to_dict()
        assert set(exported) == {
            "relevance_score",
            "matched_categories",
            "matched_noise_terms",
            "disqualifiers",
            "is_viable",
            "candidate_window",
            "word_count",
            "source_text_length",
        }
        assert exported["relevance_score"] == 0.75
        assert exported["matched_categories"] == {
            "positive_a": ["term1"],
            "positive_b": ["term2"],
        }
        assert exported["matched_noise_terms"] == ["noise1"]
        assert exported["disqualifiers"] == ["noise_dominant"]
        assert exported["is_viable"] is False
        assert exported["candidate_window"] == "window text"
        assert exported["word_count"] == 42
        assert exported["source_text_length"] == 300

    def test_to_dict_returns_independent_copies(self) -> None:
        evaluation = self._build_evaluation()
        exported = evaluation.to_dict()
        exported["matched_categories"]["positive_a"].append("mutation")
        exported["matched_noise_terms"].append("mutation")
        exported["disqualifiers"].append("mutation")
        assert evaluation.matched_categories["positive_a"] == ["term1"]
        assert evaluation.matched_noise_terms == ["noise1"]
        assert evaluation.disqualifiers == ["noise_dominant"]

    def test_to_dict_is_json_serializable(self) -> None:
        evaluation = self._build_evaluation()
        assert isinstance(json.dumps(evaluation.to_dict()), str)


class TestEvaluateText:
    """Tests for single-payload evaluation behaviour."""

    def test_viable_promotional_text_scores_high(
        self, scanner: PromotionScanner
    ) -> None:
        evaluation = scanner.evaluate_text(VIABLE_PROMO_TEXT)
        assert evaluation.is_viable is True
        assert evaluation.disqualifiers == []
        assert evaluation.relevance_score >= 0.35
        assert evaluation.relevance_score <= 1.0

    def test_viable_text_matches_expected_categories(
        self, scanner: PromotionScanner
    ) -> None:
        evaluation = scanner.evaluate_text(VIABLE_PROMO_TEXT)
        assert "positive_promotion_structures" in evaluation.matched_categories
        assert "positive_odds_constraints" in evaluation.matched_categories
        assert "positive_wagering_requirements" in evaluation.matched_categories
        assert "positive_temporal_limits" in evaluation.matched_categories
        assert "welcome bonus" in evaluation.matched_categories[
            "positive_promotion_structures"
        ]
        assert "minimum odds" in evaluation.matched_categories[
            "positive_odds_constraints"
        ]

    def test_viable_text_has_no_noise_matches(
        self, scanner: PromotionScanner
    ) -> None:
        evaluation = scanner.evaluate_text(VIABLE_PROMO_TEXT)
        assert evaluation.matched_noise_terms == []

    def test_insufficient_signals_disqualifies(
        self, scanner: PromotionScanner
    ) -> None:
        evaluation = scanner.evaluate_text(SIGNAL_FREE_TEXT)
        assert evaluation.is_viable is False
        assert "insufficient_signals" in evaluation.disqualifiers
        assert evaluation.relevance_score == 0.0

    def test_single_category_blocks_viability(
        self, scanner: PromotionScanner
    ) -> None:
        text = (
            "Grab a welcome bonus, a free bet, a deposit match and "
            "boosted odds today."
        )
        evaluation = scanner.evaluate_text(text)
        assert "insufficient_signals" in evaluation.disqualifiers
        assert evaluation.is_viable is False

    def test_noise_dominance_disqualifies(
        self, scanner: PromotionScanner
    ) -> None:
        evaluation = scanner.evaluate_text(NOISE_DOMINANT_TEXT)
        assert "noise_dominant" in evaluation.disqualifiers
        assert evaluation.is_viable is False
        assert "payment methods" in evaluation.matched_noise_terms
        assert "withdrawal" in evaluation.matched_noise_terms

    def test_casino_dominance_disqualifies(
        self, scanner: PromotionScanner
    ) -> None:
        evaluation = scanner.evaluate_text(CASINO_DOMINANT_TEXT)
        assert "casino_dominant" in evaluation.disqualifiers
        assert evaluation.is_viable is False
        assert "free spins" in evaluation.matched_noise_terms
        assert "online casino" in evaluation.matched_noise_terms

    def test_noise_only_text_scores_zero(
        self, scanner: PromotionScanner
    ) -> None:
        evaluation = scanner.evaluate_text(
            "Withdrawal fees, privacy policy and customer support details."
        )
        assert evaluation.relevance_score == 0.0
        assert evaluation.is_viable is False
        assert "noise_dominant" in evaluation.disqualifiers

    def test_empty_text_is_not_viable(self, scanner: PromotionScanner) -> None:
        evaluation = scanner.evaluate_text("")
        assert evaluation.is_viable is False
        assert evaluation.word_count == 0
        assert evaluation.source_text_length == 0
        assert evaluation.candidate_window == ""
        assert evaluation.relevance_score == 0.0
        assert evaluation.disqualifiers == ["insufficient_signals"]

    def test_none_text_treated_as_empty(
        self, scanner: PromotionScanner
    ) -> None:
        evaluation = scanner.evaluate_text(None)
        assert evaluation.word_count == 0
        assert evaluation.source_text_length == 0
        assert evaluation.is_viable is False

    def test_non_string_payload_raises_type_error(
        self, scanner: PromotionScanner
    ) -> None:
        with pytest.raises(TypeError):
            scanner.evaluate_text(12345)  # type: ignore[arg-type]

    def test_relevance_score_stays_within_unit_range(
        self, scanner: PromotionScanner
    ) -> None:
        for text in (
            VIABLE_PROMO_TEXT,
            NOISE_DOMINANT_TEXT,
            CASINO_DOMINANT_TEXT,
            SIGNAL_FREE_TEXT,
            "",
        ):
            evaluation = scanner.evaluate_text(text)
            assert 0.0 <= evaluation.relevance_score <= 1.0

    def test_low_density_dilutes_score(
        self, scanner: PromotionScanner
    ) -> None:
        diluted_text = "welcome bonus " + "filler " * 500
        evaluation = scanner.evaluate_text(diluted_text)
        assert 0.0 < evaluation.relevance_score < 0.1
        assert evaluation.is_viable is False

    def test_word_count_counts_alphanumeric_tokens(
        self, scanner: PromotionScanner
    ) -> None:
        evaluation = scanner.evaluate_text("Bet £10 at odds of 1.80")
        assert evaluation.word_count == 7

    def test_source_text_length_preserved(
        self, scanner: PromotionScanner
    ) -> None:
        evaluation = scanner.evaluate_text(VIABLE_PROMO_TEXT)
        assert evaluation.source_text_length == len(VIABLE_PROMO_TEXT)


class TestCandidateWindowExtraction:
    """Tests for isolation of the candidate window around positive spans."""

    def test_window_covers_all_positive_spans(
        self, scanner: PromotionScanner
    ) -> None:
        filler = "filler " * 100
        text = f"{filler}welcome bonus{filler}"
        evaluation = scanner.evaluate_text(text)
        assert "welcome bonus" in evaluation.candidate_window
        assert len(evaluation.candidate_window) < len(text)

    def test_window_respects_context_setting(
        self, real_lexicon: PromoLexiconLoader
    ) -> None:
        tight_scanner = PromotionScanner(
            lexicon=real_lexicon, window_context_chars=0
        )
        evaluation = tight_scanner.evaluate_text("abc welcome bonus xyz")
        assert evaluation.candidate_window == "welcome bonus"

    def test_window_pads_to_context_limit(
        self, real_lexicon: PromoLexiconLoader
    ) -> None:
        scanner = PromotionScanner(
            lexicon=real_lexicon, window_context_chars=10
        )
        text = "0123456789 welcome bonus 0123456789"
        evaluation = scanner.evaluate_text(text)
        assert evaluation.candidate_window == text

    def test_empty_text_yields_empty_window(
        self, scanner: PromotionScanner
    ) -> None:
        assert scanner.evaluate_text("").candidate_window == ""


class TestEvaluateChunks:
    """Tests for batch evaluation behaviour."""

    def test_batch_preserves_input_order(
        self, scanner: PromotionScanner
    ) -> None:
        chunks: List[str] = [
            VIABLE_PROMO_TEXT,
            NOISE_DOMINANT_TEXT,
            CASINO_DOMINANT_TEXT,
            SIGNAL_FREE_TEXT,
            "",
        ]
        evaluations = scanner.evaluate_chunks(chunks)
        assert len(evaluations) == len(chunks)
        for chunk, evaluation in zip(chunks, evaluations):
            assert evaluation == scanner.evaluate_text(chunk)

    def test_batch_matches_individual_evaluation(
        self, scanner: PromotionScanner
    ) -> None:
        evaluations = scanner.evaluate_chunks([VIABLE_PROMO_TEXT])
        assert evaluations[0].to_dict() == scanner.evaluate_text(
            VIABLE_PROMO_TEXT
        ).to_dict()

    def test_batch_rejects_non_list_input(
        self, scanner: PromotionScanner
    ) -> None:
        with pytest.raises(TypeError):
            scanner.evaluate_chunks(VIABLE_PROMO_TEXT)  # type: ignore[arg-type]

    def test_batch_rejects_non_string_items(
        self, scanner: PromotionScanner
    ) -> None:
        with pytest.raises(TypeError) as excinfo:
            scanner.evaluate_chunks(["valid", 42])  # type: ignore[list-item]
        assert "item 1" in str(excinfo.value)


class TestScannerTuning:
    """Tests demonstrating configurable threshold behaviour."""

    def test_relaxed_thresholds_admit_single_signal(
        self, real_lexicon: PromoLexiconLoader
    ) -> None:
        lenient = PromotionScanner(
            lexicon=real_lexicon,
            viability_threshold=0.1,
            min_positive_categories=1,
            min_positive_terms=1,
        )
        evaluation = lenient.evaluate_text("Grab a welcome bonus today")
        assert evaluation.is_viable is True
        assert evaluation.disqualifiers == []

    def test_strict_thresholds_reject_viable_default_text(
        self, real_lexicon: PromoLexiconLoader
    ) -> None:
        strict = PromotionScanner(
            lexicon=real_lexicon, viability_threshold=0.999999
        )
        evaluation = strict.evaluate_text(VIABLE_PROMO_TEXT)
        assert evaluation.relevance_score < 0.999999
        assert evaluation.is_viable is False

    def test_casino_penalty_weighting_applied(
        self, real_lexicon: PromoLexiconLoader
    ) -> None:
        heavy = PromotionScanner(
            lexicon=real_lexicon, casino_penalty_factor=10.0
        )
        evaluation = heavy.evaluate_text(CASINO_DOMINANT_TEXT)
        assert "casino_dominant" in evaluation.disqualifiers
        assert evaluation.relevance_score == 0.0
