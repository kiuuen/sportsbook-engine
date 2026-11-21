"""Unit tests for the bonus extraction pipeline.

The test suite exercises the full extraction workflow from LLM interaction
through invariant validation, including all provider compatibility edge
cases. The test data is synthetic but realistic; all fixtures are
self-contained and no external dependencies are required.
"""

from __future__ import annotations

import json
import time as time_module
import urllib.error
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import patch

import pytest
from pytz import UTC

from bonus_engine.extractor import (
    ExtractionConfig,
    ExtractionError,
    NonRetryableExtractionError,
    PromotionExtractor,
    TransportCallable,
)
from bonus_engine.models import (
    CalculationBase,
    EvidenceCitation,
    MonetaryConstraints,
    Promotion,
    PromotionType,
    SourceProvenance,
    TemporalValidity,
    VerificationState,
    WageringRequirement,
)
from bonus_engine.scanner import ScanEvaluation
from bonus_engine.validator import DiscrepancyCode, InvariantValidator

# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> ExtractionConfig:
    """Default test configuration with a mock API key."""
    return ExtractionConfig(
        api_key="test-key",
        endpoint_url="https://api.test/v1/chat/completions",
        model_identifier="test-model",
    )


def build_wagering(
    *,
    rollover_multiplier: float = 5.0,
    calculation_base: Union[CalculationBase, str] = CalculationBase.BONUS_AMOUNT,
    min_selection_odds: Optional[float] = None,
    max_selection_odds: Optional[float] = None,
    min_accumulator_odds: Optional[float] = None,
    min_accumulator_legs: Optional[int] = None,
    max_stake_contribution_pct: Optional[float] = None,
    completion_deadline_days: Optional[int] = None,
    stake_returned: Optional[bool] = None,
    eligible_market_types: Optional[List[str]] = None,
    excluded_market_types: Optional[List[str]] = None,
) -> WageringRequirement:
    """Build a wagering requirement with optional overrides."""
    return WageringRequirement(
        rollover_multiplier=rollover_multiplier,
        calculation_base=calculation_base,
        min_selection_odds=min_selection_odds,
        max_selection_odds=max_selection_odds,
        min_accumulator_odds=min_accumulator_odds,
        min_accumulator_legs=min_accumulator_legs,
        max_stake_contribution_pct=max_stake_contribution_pct,
        completion_deadline_days=completion_deadline_days,
        stake_returned=stake_returned,
        eligible_market_types=eligible_market_types or [],
        excluded_market_types=excluded_market_types or [],
    )


def build_temporal(
    *,
    valid_from: Optional[datetime] = None,
    valid_until: Optional[datetime] = None,
    wagering_deadline_days: Optional[int] = None,
    timezone_name: str = "UTC",
) -> TemporalValidity:
    """Build temporal constraints with optional overrides."""
    return TemporalValidity(
        valid_from=valid_from,
        valid_until=valid_until,
        wagering_deadline_days=wagering_deadline_days,
        timezone_name=timezone_name,
    )


def build_monetary(
    *,
    min_qualifying_deposit: Optional[float] = None,
    max_qualifying_deposit: Optional[float] = None,
    min_bonus_amount: Optional[float] = None,
    max_bonus_amount: Optional[float] = None,
    max_conversion_amount: Optional[float] = None,
    bonus_percentage: Optional[float] = None,
    excluded_payment_methods: Optional[List[str]] = None,
    currency: Optional[str] = None,
) -> MonetaryConstraints:
    """Build monetary constraints with optional overrides."""
    return MonetaryConstraints(
        min_qualifying_deposit=min_qualifying_deposit,
        max_qualifying_deposit=max_qualifying_deposit,
        min_bonus_amount=min_bonus_amount,
        max_bonus_amount=max_bonus_amount,
        max_conversion_amount=max_conversion_amount,
        bonus_percentage=bonus_percentage,
        excluded_payment_methods=excluded_payment_methods or [],
        currency=currency,
    )


def build_provenance(
    *,
    source_url: str = "https://example.com/promo",
    content_hash: str = "a" * 64,
    captured_at: Optional[datetime] = None,
    extraction_method: str = "automated",
    citations: Optional[List[EvidenceCitation]] = None,
) -> SourceProvenance:
    """Build provenance with optional overrides."""
    return SourceProvenance(
        source_url=source_url,
        content_hash=content_hash,
        captured_at=captured_at or datetime.now(timezone.utc),
        extraction_method=extraction_method,
        citations=citations or [],
    )


def build_promotion(
    *,
    target_domain: str = "example.com",
    promotion_type: Union[PromotionType, str] = PromotionType.DEPOSIT_MATCH,
    title: str = "Welcome Bonus",
    promo_code: Optional[str] = None,
    wagering: Optional[WageringRequirement] = None,
    temporal: Optional[TemporalValidity] = None,
    monetary: Optional[MonetaryConstraints] = None,
    verification_state: Union[VerificationState, str] = VerificationState.EXTRACTED,
    provenance: Optional[SourceProvenance] = None,
    notes: Optional[str] = None,
    cited_fields: Optional[Tuple[str, ...]] = None,
) -> Promotion:
    """Build a promotion with optional overrides and optional evidence citations."""
    if wagering is None:
        wagering = build_wagering()
    if temporal is None:
        temporal = build_temporal()
    if monetary is None:
        monetary = build_monetary()
    if provenance is None:
        provenance = build_provenance()
    if cited_fields is not None:
        provenance.citations = make_citations(cited_fields)
    return Promotion(
        target_domain=target_domain,
        promotion_type=promotion_type,
        title=title,
        promo_code=promo_code,
        wagering=wagering,
        temporal=temporal,
        monetary=monetary,
        verification_state=verification_state,
        provenance=provenance,
        notes=notes,
    )


def build_campaign(
    *,
    promotion_type: Union[PromotionType, str] = PromotionType.DEPOSIT_MATCH,
    title: str = "Welcome Bonus",
    promo_code: Optional[str] = "WELCOME100",
    wagering: Optional[Dict[str, Any]] = None,
    temporal: Optional[Dict[str, Any]] = None,
    monetary: Optional[Dict[str, Any]] = None,
    evidence: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a campaign payload with optional overrides."""
    if wagering is None:
        wagering = {
            "rollover_multiplier": 5.0,
            "calculation_base": "deposit_plus_bonus",
            "min_selection_odds": None,
            "max_selection_odds": None,
            "min_accumulator_odds": None,
            "min_accumulator_legs": None,
            "max_stake_contribution_pct": None,
            "completion_deadline_days": None,
            "stake_returned": None,
            "eligible_market_types": None,
            "excluded_market_types": None,
        }
    if temporal is None:
        temporal = {
            "valid_from": None,
            "valid_until": None,
            "wagering_deadline_days": None,
            "timezone_name": "UTC",
        }
    if monetary is None:
        monetary = {
            "min_qualifying_deposit": None,
            "max_qualifying_deposit": None,
            "min_bonus_amount": None,
            "max_bonus_amount": 100.0,
            "max_conversion_amount": None,
            "bonus_percentage": None,
            "excluded_payment_methods": None,
            "currency": "EUR",
        }
    if evidence is None:
        evidence = make_citations(E2E_CITED_FIELDS)
    serialized_evidence = [
        item.to_dict() if isinstance(item, EvidenceCitation) else item
        for item in (evidence or [])
    ]
    return {
        "promotion_type": promotion_type.value
        if isinstance(promotion_type, PromotionType)
        else promotion_type,
        "title": title,
        "promo_code": promo_code,
        "wagering": wagering,
        "temporal": temporal,
        "monetary": monetary,
        "evidence": serialized_evidence,
    }


def make_citations(field_names: Tuple[str, ...]) -> List[EvidenceCitation]:
    """Build evidence citations for the given field names."""
    return [
        EvidenceCitation(
            field_name=name,
            raw_text=QUOTE_LIBRARY[name],
            locator=f"locator-for-{name}",
        )
        for name in field_names
    ]


def make_source_text_safe(field_names) -> str:
    """Build a source text containing verbatim quotes for the given fields."""
    return " ".join(QUOTE_LIBRARY[name] for name in field_names)


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

QUOTE_LIBRARY = {
    "rollover_multiplier": "wagering requirement of five times the bonus amount",
    "min_selection_odds": "minimum selection odds of 1.5",
    "max_selection_odds": "maximum selection odds of 100.0",
    "min_accumulator_odds": "minimum accumulator odds of 1000.0",
    "min_accumulator_legs": "minimum accumulator legs of 20",
    "max_stake_contribution_pct": "maximum stake contribution of 50%",
    "completion_deadline_days": "completion deadline of 30 days",
    "min_qualifying_deposit": "minimum qualifying deposit of 100.0",
    "max_qualifying_deposit": "maximum qualifying deposit of 1000.0",
    "min_bonus_amount": "minimum bonus amount of 10.0",
    "max_bonus_amount": "maximum bonus amount of 100.0",
    "max_conversion_amount": "maximum conversion amount of 500.0",
    "bonus_percentage": "bonus percentage of 100.0",
}

E2E_CITED_FIELDS = (
    "rollover_multiplier",
    "min_selection_odds",
    "max_selection_odds",
    "min_accumulator_odds",
    "min_accumulator_legs",
    "max_stake_contribution_pct",
    "completion_deadline_days",
    "min_qualifying_deposit",
    "max_qualifying_deposit",
    "min_bonus_amount",
    "max_bonus_amount",
    "max_conversion_amount",
    "bonus_percentage",
)

E2E_SOURCE_TEXT = make_source_text_safe(E2E_CITED_FIELDS)

# ---------------------------------------------------------------------------
# Provider envelope builders
# ---------------------------------------------------------------------------


def openai_envelope(content: str) -> str:
    """Build an OpenAI-style envelope with the given content."""
    return json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": content,
                    }
                }
            ]
        }
    )


def legacy_openai_envelope(content: str) -> str:
    """Build a legacy OpenAI-style envelope with the given content."""
    return json.dumps(
        {
            "choices": [
                {
                    "text": content,
                }
            ]
        }
    )


def anthropic_envelope(content: str) -> str:
    """Build an Anthropic-style envelope with the given content."""
    return json.dumps(
        {
            "content": content,
        }
    )


def mistral_envelope(content: str) -> str:
    """Build a Mistral-style envelope with the given content."""
    return json.dumps(
        {
            "output_text": content,
        }
    )


def generic_envelope(content: str) -> str:
    """Build a generic envelope with the given content."""
    return json.dumps(
        {
            "text": content,
        }
    )


ENVELOPE_BUILDERS = {
    "openai": openai_envelope,
    "legacy_openai": legacy_openai_envelope,
    "anthropic": anthropic_envelope,
    "mistral": mistral_envelope,
    "generic": generic_envelope,
}


# ---------------------------------------------------------------------------
# Transport mocks
# ---------------------------------------------------------------------------


class ScriptedTransport:
    """Transport replaying a script of (status, body) tuples."""

    def __init__(self, script: List[Tuple[int, str]]) -> None:
        self._script = list(script)
        self.calls: List[Dict[str, Any]] = []

    def __call__(
        self, url: str, headers: Dict[str, str], body: bytes, timeout: float
    ) -> Tuple[int, str]:
        payload = json.loads(body.decode("utf-8"))
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "payload": payload,
                "timeout": timeout,
            }
        )
        if len(self._script) > 1:
            status, text = self._script.pop(0)
        else:
            status, text = self._script[0]
        return status, text


class FailingTransport:
    """Transport raising transport-level failures."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self, url: str, headers: Dict[str, str], body: bytes, timeout: float
    ) -> Tuple[int, str]:
        self.calls += 1
        raise urllib.error.URLError("connection failure")


class FlakyThenSuccessTransport:
    """Transport raising transport-level failures before replaying a script."""

    def __init__(self, script: List[Tuple[int, str]], failures: int) -> None:
        self._script = list(script)
        self._remaining_failures = failures
        self.calls = 0

    def __call__(
        self, url: str, headers: Dict[str, str], body: bytes, timeout: float
    ) -> Tuple[int, str]:
        self.calls += 1
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise urllib.error.URLError("transient connection failure")
        if len(self._script) > 1:
            status, text = self._script.pop(0)
        else:
            status, text = self._script[0]
        return status, text


# ---------------------------------------------------------------------------
# InvariantValidator
# ---------------------------------------------------------------------------


class TestInvariantValidatorWagering:
    def test_rollover_below_sports_floor_is_flagged(self):
        promotion = build_promotion(
            wagering=build_wagering(rollover_multiplier=0.5),
            cited_fields=("rollover_multiplier",),
        )
        report = InvariantValidator().validate(promotion)
        assert report.discrepancy_codes == [
            DiscrepancyCode.ROLLOVER_BELOW_SPORTS_FLOOR
        ]

    def test_rollover_just_above_sports_floor_passes(self):
        promotion = build_promotion(
            wagering=build_wagering(rollover_multiplier=1.0),
            cited_fields=("rollover_multiplier",),
        )
        report = InvariantValidator().validate(promotion)
        assert DiscrepancyCode.ROLLOVER_BELOW_SPORTS_FLOOR not in report.discrepancy_codes

    def test_rollover_just_below_sports_ceiling_passes(self):
        promotion = build_promotion(
            wagering=build_wagering(rollover_multiplier=20.0),
            cited_fields=("rollover_multiplier",),
        )
        report = InvariantValidator().validate(promotion)
        assert DiscrepancyCode.ROLLOVER_ABOVE_SPORTS_CEILING not in report.discrepancy_codes

    def test_rollover_above_sports_ceiling_is_flagged(self):
        promotion = build_promotion(
            wagering=build_wagering(rollover_multiplier=21.0),
            cited_fields=("rollover_multiplier",),
        )
        report = InvariantValidator().validate(promotion)
        assert report.discrepancy_codes == [
            DiscrepancyCode.ROLLOVER_ABOVE_SPORTS_CEILING
        ]

    def test_unspecified_calculation_base_with_positive_rollover_is_ambiguous(self):
        promotion = build_promotion(
            wagering=build_wagering(
                rollover_multiplier=5.0,
                calculation_base=CalculationBase.UNSPECIFIED,
            ),
            cited_fields=("rollover_multiplier",),
        )
        report = InvariantValidator().validate(promotion)
        assert report.discrepancy_codes == [
            DiscrepancyCode.CALCULATION_BASE_AMBIGUOUS
        ]

    def test_explicit_calculation_base_with_positive_rollover_is_clean(self):
        for base in (
            CalculationBase.BONUS_AMOUNT,
            CalculationBase.DEPOSIT_PLUS_BONUS,
        ):
            promotion = build_promotion(
                wagering=build_wagering(
                    rollover_multiplier=5.0,
                    calculation_base=base,
                ),
                cited_fields=("rollover_multiplier",),
            )
            report = InvariantValidator().validate(promotion)
            assert DiscrepancyCode.CALCULATION_BASE_AMBIGUOUS not in report.discrepancy_codes

    def test_zero_rollover_with_unspecified_base_is_clean(self):
        promotion = build_promotion(
            wagering=build_wagering(
                rollover_multiplier=0.0,
                calculation_base=CalculationBase.UNSPECIFIED,
            ),
            cited_fields=("rollover_multiplier",),
        )
        report = InvariantValidator().validate(promotion)
        assert DiscrepancyCode.CALCULATION_BASE_AMBIGUOUS not in report.discrepancy_codes

    @pytest.mark.parametrize(
        "odds,flagged",
        [
            (1.0, True),
            (1.01, False),
            (100.0, False),
            (100.1, True),
        ],
    )
    def test_selection_odds_bounds(self, odds, flagged):
        promotion = build_promotion(
            wagering=build_wagering(min_selection_odds=odds),
            cited_fields=("min_selection_odds",),
        )
        report = InvariantValidator().validate(promotion)
        assert (
            DiscrepancyCode.SELECTION_ODDS_IMPLAUSIBLE in report.discrepancy_codes
        ) is flagged

    def test_max_selection_odds_below_min_is_structural_violation(self):
        promotion = build_promotion(
            wagering=build_wagering(
                min_selection_odds=2.0,
                max_selection_odds=1.5,
            ),
            cited_fields=("min_selection_odds", "max_selection_odds"),
        )
        report = InvariantValidator().validate(promotion)
        assert (
            DiscrepancyCode.STRUCTURAL_INVARIANT_VIOLATION
            in report.discrepancy_codes
        )

    def test_min_accumulator_odds_below_min_selection_is_structural_violation(self):
        promotion = build_promotion(
            wagering=build_wagering(
                min_selection_odds=2.0,
                min_accumulator_odds=1.5,
            ),
            cited_fields=("min_selection_odds", "min_accumulator_odds"),
        )
        report = InvariantValidator().validate(promotion)
        assert (
            DiscrepancyCode.STRUCTURAL_INVARIANT_VIOLATION
            in report.discrepancy_codes
        )

    @pytest.mark.parametrize(
        "odds,flagged",
        [
            (1000.0, False),
            (1000.1, True),
        ],
    )
    def test_accumulator_odds_bounds(self, odds, flagged):
        promotion = build_promotion(
            wagering=build_wagering(min_accumulator_odds=odds),
            cited_fields=("min_accumulator_odds",),
        )
        report = InvariantValidator().validate(promotion)
        assert (
            DiscrepancyCode.ACCUMULATOR_CONSTRAINTS_IMPLAUSIBLE
            in report.discrepancy_codes
        ) is flagged

    @pytest.mark.parametrize(
        "legs,flagged",
        [
            (2, False),
            (20, False),
            (21, True),
        ],
    )
    def test_accumulator_bounds(self, legs, flagged):
        promotion = build_promotion(
            wagering=build_wagering(min_accumulator_legs=legs),
            cited_fields=("min_accumulator_legs",),
        )
        report = InvariantValidator().validate(promotion)
        assert (
            DiscrepancyCode.ACCUMULATOR_CONSTRAINTS_IMPLAUSIBLE
            in report.discrepancy_codes
        ) is flagged

        assert (
            DiscrepancyCode.ACCUMULATOR_CONSTRAINTS_IMPLAUSIBLE
            in report.discrepancy_codes
        ) is flagged

    def test_casino_boundary_is_inclusive_at_threshold(self):
        promotion = build_promotion(
            wagering=build_wagering(rollover_multiplier=30.0),
            cited_fields=("rollover_multiplier",),
        )
        report = InvariantValidator().validate(promotion)
        assert report.discrepancy_codes == [DiscrepancyCode.CASINO_STYLE_ROLLOVER]

    def test_rollover_just_below_casino_threshold_hits_ceiling(self):
        promotion = build_promotion(
            wagering=build_wagering(rollover_multiplier=29.0),
            cited_fields=("rollover_multiplier",),
        )
        report = InvariantValidator().validate(promotion)
        assert report.discrepancy_codes == [
            DiscrepancyCode.ROLLOVER_ABOVE_SPORTS_CEILING
        ]


class TestInvariantValidatorTemporal:
    def test_excessive_validity_window_is_flagged(self):
        promotion = build_promotion(
            temporal=build_temporal(
                valid_from=datetime(2024, 1, 1, tzinfo=UTC),
                valid_until=datetime(2025, 1, 2, tzinfo=UTC),
            )
        )
        report = InvariantValidator().validate(promotion)
        assert DiscrepancyCode.TEMPORAL_WINDOW_EXCESSIVE in report.discrepancy_codes

    def test_exact_max_window_passes(self):
        promotion = build_promotion(
            temporal=build_temporal(
                valid_from=datetime(2024, 1, 1, tzinfo=UTC),
                valid_until=datetime(2024, 12, 31, tzinfo=UTC),
            )
        )
        report = InvariantValidator().validate(promotion)
        assert DiscrepancyCode.TEMPORAL_WINDOW_EXCESSIVE not in report.discrepancy_codes

    @pytest.mark.parametrize("deadline", [0, 366])
    def test_implausible_wagering_deadline_is_flagged(self, deadline):
        promotion = build_promotion(
            temporal=build_temporal(wagering_deadline_days=deadline)
        )
        report = InvariantValidator().validate(promotion)
        assert DiscrepancyCode.WAGERING_DEADLINE_IMPLAUSIBLE in report.discrepancy_codes

    def test_boundary_deadlines_pass(self):
        for deadline in (1, 365):
            promotion = build_promotion(
                temporal=build_temporal(wagering_deadline_days=deadline)
            )
            report = InvariantValidator().validate(promotion)
            assert (
                DiscrepancyCode.WAGERING_DEADLINE_IMPLAUSIBLE
                not in report.discrepancy_codes
            )

    def test_valid_until_before_valid_from_is_structural_violation(self):
        promotion = build_promotion(
            temporal=build_temporal(
                valid_from=datetime(2024, 3, 1, tzinfo=UTC),
                valid_until=datetime(2024, 1, 1, tzinfo=UTC),
            )
        )
        report = InvariantValidator().validate(promotion)
        assert (
            DiscrepancyCode.STRUCTURAL_INVARIANT_VIOLATION
            in report.discrepancy_codes
        )


class TestInvariantValidatorMonetary:
    @pytest.mark.parametrize(
        "percentage,flagged",
        [(1.0, False), (100.0, False), (0.5, True), (100.5, True)],
    )
    def test_bonus_percentage_bounds(self, percentage, flagged):
        promotion = build_promotion(
            monetary=MonetaryConstraints(bonus_percentage=percentage),
            cited_fields=("bonus_percentage",),
        )
        report = InvariantValidator().validate(promotion)
        assert (
            DiscrepancyCode.BONUS_PERCENTAGE_IMPLAUSIBLE in report.discrepancy_codes
        ) is flagged


class TestInvariantValidatorEvidence:
    def test_missing_citations_for_numeric_claims_are_flagged(self):
        promotion = build_promotion(
            wagering=build_wagering(min_selection_odds=1.5),
            monetary=MonetaryConstraints(bonus_percentage=100.0),
            cited_fields=("rollover_multiplier",),
        )
        report = InvariantValidator().validate(promotion)
        assert DiscrepancyCode.MISSING_EVIDENCE_CITATION in report.discrepancy_codes
        assert "min_selection_odds" in report.evidence_gaps
        assert "bonus_percentage" in report.evidence_gaps
        assert "rollover_multiplier" not in report.evidence_gaps

    def test_authentic_quotes_pass_across_casing_whitespace_diacritics(self):
        promotion = build_promotion(
            wagering=build_wagering(),
            cited_fields=("rollover_multiplier",),
        )
        mangled = "WAGERING  Requirement of FIVE Times the Bonus Amount"
        promotion.provenance.citations[0].raw_text = mangled
        report = InvariantValidator().validate(promotion, source_text=mangled)
        assert DiscrepancyCode.EVIDENCE_QUOTE_UNVERIFIED not in report.discrepancy_codes

    def test_fabricated_quote_is_flagged(self):
        promotion = build_promotion(
            wagering=build_wagering(),
            cited_fields=("rollover_multiplier",),
        )
        promotion.provenance.citations[0].raw_text = (
            "totally fabricated terms text never present anywhere"
        )
        report = InvariantValidator().validate(
            promotion, source_text=E2E_SOURCE_TEXT
        )
        assert DiscrepancyCode.EVIDENCE_QUOTE_UNVERIFIED in report.discrepancy_codes

    def test_diacritic_variation_passes_tolerant_matching(self):
        promotion = build_promotion(
            wagering=build_wagering(),
            cited_fields=("rollover_multiplier",),
        )
        accented = "wagéríng requirement of five times the bonus amount"
        promotion.provenance.citations[0].raw_text = accented
        report = InvariantValidator().validate(promotion, source_text=accented)
        assert DiscrepancyCode.EVIDENCE_QUOTE_UNVERIFIED not in report.discrepancy_codes

    def test_blank_source_text_skips_quote_verification(self):
        promotion = build_promotion(
            wagering=build_wagering(),
            cited_fields=("rollover_multiplier",),
        )
        promotion.provenance.citations[0].raw_text = "fabricated text"
        report = InvariantValidator().validate(promotion, source_text="   ")
        assert DiscrepancyCode.EVIDENCE_QUOTE_UNVERIFIED not in report.discrepancy_codes

    def test_validate_rejects_non_promotion(self):
        with pytest.raises(TypeError):
            InvariantValidator().validate("not a promotion")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ExtractionConfig
# ---------------------------------------------------------------------------


class TestExtractionConfig:
    def test_valid_configuration_is_accepted(self):
        config = ExtractionConfig(
            api_key="key", endpoint_url="https://api.test", model_identifier="m"
        )
        assert config.timeout_seconds == 60.0
        assert config.max_retries == 3
        assert config.retry_backoff_seconds == 2.0
        assert config.temperature == 0.0

    @pytest.mark.parametrize(
        "field", ["api_key", "endpoint_url", "model_identifier"]
    )
    @pytest.mark.parametrize("bad_value", ["", "   ", None, 123])
    def test_required_strings_are_validated(self, field, bad_value):
        params = {
            "api_key": "key",
            "endpoint_url": "https://api.test",
            "model_identifier": "m",
        }
        params[field] = bad_value
        with pytest.raises((ValueError, TypeError)):
            ExtractionConfig(**params)

    def test_numeric_fields_reject_non_numbers(self):
        with pytest.raises(TypeError):
            ExtractionConfig(
                api_key="k",
                endpoint_url="https://api.test",
                model_identifier="m",
                temperature="hot",
            )

    def test_invalid_ranges_are_rejected(self):
        with pytest.raises(ValueError):
            ExtractionConfig(
                api_key="k",
                endpoint_url="https://api.test",
                model_identifier="m",
                timeout_seconds=0,
            )
        with pytest.raises(ValueError):
            ExtractionConfig(
                api_key="k",
                endpoint_url="https://api.test",
                model_identifier="m",
                retry_backoff_seconds=-1.0,
            )
        with pytest.raises(ValueError):
            ExtractionConfig(
                api_key="k",
                endpoint_url="https://api.test",
                model_identifier="m",
                max_retries=-1,
            )
        with pytest.raises(ValueError):
            ExtractionConfig(
                api_key="k",
                endpoint_url="https://api.test",
                model_identifier="m",
                max_retries=True,
            )

    def test_from_environment_loads_valid_overrides(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("BONUS_LLM_API_KEY", " env-key ")
        monkeypatch.setenv("BONUS_LLM_ENDPOINT_URL", " https://api.test ")
        monkeypatch.setenv("BONUS_LLM_MODEL", " model-x ")
        monkeypatch.setenv("BONUS_LLM_TIMEOUT_SECONDS", "12.5")
        monkeypatch.setenv("BONUS_LLM_MAX_RETRIES", "5")
        monkeypatch.setenv("BONUS_LLM_RETRY_BACKOFF_SECONDS", "0.5")
        monkeypatch.setenv("BONUS_LLM_TEMPERATURE", "0.2")
        config = ExtractionConfig.from_environment()
        assert config.api_key == "env-key"
        assert config.endpoint_url == "https://api.test"
        assert config.model_identifier == "model-x"
        assert config.timeout_seconds == 12.5
        assert config.max_retries == 5
        assert config.retry_backoff_seconds == 0.5
        assert config.temperature == 0.2

    def test_from_environment_applies_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("BONUS_LLM_API_KEY", "k")
        monkeypatch.setenv("BONUS_LLM_ENDPOINT_URL", "https://api.test")
        monkeypatch.setenv("BONUS_LLM_MODEL", "m")
        config = ExtractionConfig.from_environment()
        assert config.timeout_seconds == 60.0
        assert config.max_retries == 3
        assert config.retry_backoff_seconds == 2.0
        assert config.temperature == 0.0

    def test_from_environment_missing_required_variables(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("BONUS_LLM_API_KEY", raising=False)
        monkeypatch.delenv("BONUS_LLM_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("BONUS_LLM_MODEL", raising=False)
        with pytest.raises(ValueError) as excinfo:
            ExtractionConfig.from_environment()
        assert "API_KEY" in str(excinfo.value)

    @pytest.mark.parametrize(
        "name,value",
        [
            ("BONUS_LLM_TIMEOUT_SECONDS", "not-a-number"),
            ("BONUS_LLM_TIMEOUT_SECONDS", "-1"),
            ("BONUS_LLM_MAX_RETRIES", "1.5"),
            ("BONUS_LLM_MAX_RETRIES", "-2"),
            ("BONUS_LLM_TEMPERATURE", "abc"),
        ],
    )
    def test_from_environment_invalid_overrides(
        self, monkeypatch: pytest.MonkeyPatch, name, value
    ):
        monkeypatch.setenv("BONUS_LLM_API_KEY", "k")
        monkeypatch.setenv("BONUS_LLM_ENDPOINT_URL", "https://api.test")
        monkeypatch.setenv("BONUS_LLM_MODEL", "m")
        monkeypatch.setenv(name, value)
        with pytest.raises(ValueError):
            ExtractionConfig.from_environment()


# ---------------------------------------------------------------------------
# PromotionExtractor: end-to-end extraction
# ---------------------------------------------------------------------------


class TestPromotionExtractorEndToEnd:
    def test_compliant_payload_is_extracted_and_validated(
        self, config: ExtractionConfig
    ):
        transport = ScriptedTransport(
            [(200, openai_envelope(json.dumps(build_campaign())))]
        )
        extractor = PromotionExtractor(config, transport=transport)
        promotions = extractor.extract_promotions(
            E2E_SOURCE_TEXT,
            target_domain="example.com",
            source_url="https://example.com/promo",
            raw_content=E2E_SOURCE_TEXT,
        )
        assert len(promotions) == 1
        promotion = promotions[0]
        assert promotion.promotion_type is PromotionType.DEPOSIT_MATCH
        assert promotion.promo_code == "WELCOME100"
        assert promotion.wagering is not None
        assert promotion.wagering.rollover_multiplier == 5.0
        assert (
            promotion.wagering.calculation_base
            is CalculationBase.DEPOSIT_PLUS_BONUS
        )
        assert promotion.monetary is not None
        assert promotion.monetary.max_bonus_amount == 100.0
        assert promotion.monetary.currency == "EUR"
        assert promotion.provenance is not None
        assert promotion.provenance.source_url == "https://example.com/promo"
        assert len(promotion.provenance.citations) == len(E2E_CITED_FIELDS)
        assert (
            promotion.verification_state
            is VerificationState.INVARIANT_VALIDATED
        )

    def test_partial_nullable_wagering_parameters_are_preserved(
        self, config: ExtractionConfig
    ):
        campaign = build_campaign(
            wagering={
                "rollover_multiplier": 3,
                "calculation_base": "bonus_amount",
                "min_selection_odds": None,
                "max_selection_odds": None,
                "min_accumulator_odds": None,
                "min_accumulator_legs": None,
                "max_stake_contribution_pct": None,
                "completion_deadline_days": None,
                "stake_returned": None,
                "eligible_market_types": None,
                "excluded_market_types": None,
            },
            evidence=make_citations(("rollover_multiplier",)),
        )
        transport = ScriptedTransport(
            [(200, openai_envelope(json.dumps(campaign)))]
        )
        extractor = PromotionExtractor(config, transport=transport)
        promotions = extractor.extract_promotions(
            QUOTE_LIBRARY["rollover_multiplier"],
            target_domain="example.com",
            source_url="https://example.com/promo",
        )
        assert len(promotions) == 1
        wagering = promotions[0].wagering
        assert wagering is not None
        assert wagering.rollover_multiplier == 3.0
        assert wagering.min_selection_odds is None
        assert wagering.completion_deadline_days is None
        assert wagering.stake_returned is None
        assert wagering.eligible_market_types == []
        assert wagering.excluded_market_types == []

    def test_unrecognized_calculation_base_is_normalized(
        self, config: ExtractionConfig
    ):
        campaign = build_campaign(
            wagering={
                "rollover_multiplier": 5,
                "calculation_base": "mystery_base",
            },
            evidence=make_citations(("rollover_multiplier",)),
        )
        transport = ScriptedTransport(
            [(200, openai_envelope(json.dumps(campaign)))]
        )
        extractor = PromotionExtractor(config, transport=transport)
        promotions = extractor.extract_promotions(
            QUOTE_LIBRARY["rollover_multiplier"],
            target_domain="example.com",
            source_url="https://example.com/promo",
        )
        assert len(promotions) == 1
        assert promotions[0].wagering is not None
        assert (
            promotions[0].wagering.calculation_base
            is CalculationBase.UNSPECIFIED
        )

    def test_unrecognized_promotion_type_falls_back_to_other(
        self, config: ExtractionConfig
    ):
        campaign = build_campaign(
            promotion_type="mystery_promo",
            evidence=make_citations(("rollover_multiplier",)),
        )
        transport = ScriptedTransport(
            [(200, openai_envelope(json.dumps(campaign)))]
        )
        extractor = PromotionExtractor(config, transport=transport)
        promotions = extractor.extract_promotions(
            QUOTE_LIBRARY["rollover_multiplier"],
            target_domain="example.com",
            source_url="https://example.com/promo",
        )
        assert len(promotions) == 1
        assert promotions[0].promotion_type is PromotionType.OTHER

    def test_invariant_violation_is_flagged_for_review(
        self, config: ExtractionConfig
    ):
        campaign = build_campaign(
            wagering={
                "rollover_multiplier": 40,
                "calculation_base": "bonus_amount",
            },
            evidence=make_citations(("rollover_multiplier",)),
        )
        transport = ScriptedTransport(
            [(200, openai_envelope(json.dumps(campaign)))]
        )
        extractor = PromotionExtractor(config, transport=transport)
        promotions = extractor.extract_promotions(
            QUOTE_LIBRARY["rollover_multiplier"],
            target_domain="example.com",
            source_url="https://example.com/promo",
        )
        assert len(promotions) == 1
        promotion = promotions[0]
        assert promotion.verification_state is VerificationState.EXTRACTED
        assert "flagged_for_review" in (promotion.notes or "")
        assert "casino_style_rollover" in (promotion.notes or "")

    def test_fabricated_quote_is_flagged_for_review(
        self, config: ExtractionConfig
    ):
        campaign = build_campaign(
            evidence=[
                {
                    "field_name": "rollover_multiplier",
                    "raw_text": "invented quote that appears nowhere",
                }
            ]
        )
        transport = ScriptedTransport(
            [(200, openai_envelope(json.dumps(campaign)))]
        )
        extractor = PromotionExtractor(config, transport=transport)
        promotions = extractor.extract_promotions(
            E2E_SOURCE_TEXT,
            target_domain="example.com",
            source_url="https://example.com/promo",
        )
        assert len(promotions) == 1
        promotion = promotions[0]
        assert promotion.verification_state is VerificationState.EXTRACTED
        assert "evidence_quote_unverified" in (promotion.notes or "")

    def test_malformed_campaign_payload_is_skipped(
        self, config: ExtractionConfig
    ):
        campaign = build_campaign()
        del campaign["title"]
        transport = ScriptedTransport(
            [(200, openai_envelope(json.dumps(campaign)))]
        )
        extractor = PromotionExtractor(config, transport=transport)
        promotions = extractor.extract_promotions(
            E2E_SOURCE_TEXT,
            target_domain="example.com",
            source_url="https://example.com/promo",
        )
        assert promotions == []

    def test_multiple_campaigns_are_extracted_in_order(
        self, config: ExtractionConfig
    ):
        first = build_campaign(title="First Offer")
        second = build_campaign(
            title="Second Offer",
            promotion_type="freebet",
            wagering={
                "rollover_multiplier": 1,
                "calculation_base": "freebet_stake",
            },
        )
        transport = ScriptedTransport(
            [(200, openai_envelope(json.dumps([first, second])))]
        )
        extractor = PromotionExtractor(config, transport=transport)
        promotions = extractor.extract_promotions(
            E2E_SOURCE_TEXT,
            target_domain="example.com",
            source_url="https://example.com/promo",
        )
        assert [p.title for p in promotions] == ["First Offer", "Second Offer"]


# ---------------------------------------------------------------------------
# PromotionExtractor: provider compatibility and resilience
# ---------------------------------------------------------------------------


class TestPromotionExtractorProviderCompatibility:
    @pytest.mark.parametrize("envelope_name", sorted(ENVELOPE_BUILDERS))
    def test_all_supported_envelopes_are_parsed(
        self, config: ExtractionConfig, envelope_name
    ):
        transport = ScriptedTransport(
            [
                (
                    200,
                    ENVELOPE_BUILDERS[envelope_name](
                        json.dumps(build_campaign())
                    ),
                )
            ]
        )
        extractor = PromotionExtractor(config, transport=transport)
        promotions = extractor.extract_promotions(
            E2E_SOURCE_TEXT,
            target_domain="example.com",
            source_url="https://example.com/promo",
        )
        assert len(promotions) == 1

    def test_markdown_fenced_json_is_parsed(self, config: ExtractionConfig):
        fenced = "```json\n" + json.dumps(build_campaign()) + "\n```"
        transport = ScriptedTransport([(200, openai_envelope(fenced))])
        extractor = PromotionExtractor(config, transport=transport)
        promotions = extractor.extract_promotions(
            E2E_SOURCE_TEXT,
            target_domain="example.com",
            source_url="https://example.com/promo",
        )
        assert len(promotions) == 1

    def test_response_format_fallback_on_client_error(
        self, config: ExtractionConfig
    ):
        transport = ScriptedTransport(
            [
                (400, json.dumps({"error": "response_format unsupported"})),
                (200, openai_envelope(json.dumps(build_campaign()))),
            ]
        )
        extractor = PromotionExtractor(config, transport=transport)
        promotions = extractor.extract_promotions(
            E2E_SOURCE_TEXT,
            target_domain="example.com",
            source_url="https://example.com/promo",
        )
        assert len(promotions) == 1
        assert len(transport.calls) == 2
        assert "response_format" in transport.calls[0]["payload"]
        assert "response_format" not in transport.calls[1]["payload"]

    def test_response_format_fallback_is_one_time_only(
        self, config: ExtractionConfig
    ):
        transport = ScriptedTransport(
            [
                (400, "bad request"),
                (400, "bad request"),
                (400, "bad request"),
            ]
        )
        extractor = PromotionExtractor(config, transport=transport)
        with pytest.raises(ExtractionError):
            extractor.extract_promotions(
                E2E_SOURCE_TEXT,
                target_domain="example.com",
                source_url="https://example.com/promo",
            )
        assert len(transport.calls) == 2

    def test_retry_then_success_on_transient_failures(
        self, config: ExtractionConfig, monkeypatch: pytest.MonkeyPatch
    ):
        sleeps: List[float] = []
        monkeypatch.setattr(time_module, "sleep", sleeps.append)
        transport = ScriptedTransport(
            [
                (500, "server error"),
                (429, "rate limited"),
                (200, openai_envelope(json.dumps(build_campaign()))),
            ]
        )
        extractor = PromotionExtractor(config, transport=transport)
        promotions = extractor.extract_promotions(
            E2E_SOURCE_TEXT,
            target_domain="example.com",
            source_url="https://example.com/promo",
        )
        assert len(promotions) == 1
        assert len(transport.calls) == 3
        assert len(sleeps) == 2

    def test_transport_failure_retries_then_succeeds(
        self, config: ExtractionConfig, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(time_module, "sleep", lambda seconds: None)
        flaky = FlakyThenSuccessTransport(
            [(200, openai_envelope(json.dumps(build_campaign())))], failures=2
        )
        extractor = PromotionExtractor(config, transport=flaky)
        promotions = extractor.extract_promotions(
            E2E_SOURCE_TEXT,
            target_domain="example.com",
            source_url="https://example.com/promo",
        )
        assert len(promotions) == 1
        assert flaky.calls == 3

    def test_exhausted_retries_degrade_to_empty_list(
        self, config: ExtractionConfig, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(time_module, "sleep", lambda seconds: None)
        transport = ScriptedTransport([(500, "server error")])
        extractor = PromotionExtractor(config, transport=transport)
        promotions = extractor.extract_promotions(
            E2E_SOURCE_TEXT,
            target_domain="example.com",
            source_url="https://example.com/promo",
        )
        assert promotions == []
        assert len(transport.calls) == config.max_retries + 1

    def test_persistent_transport_failure_degrades_to_empty_list(
        self, config: ExtractionConfig, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(time_module, "sleep", lambda seconds: None)
        transport = FailingTransport()
        extractor = PromotionExtractor(config, transport=transport)
        promotions = extractor.extract_promotions(
            E2E_SOURCE_TEXT,
            target_domain="example.com",
            source_url="https://example.com/promo",
        )
        assert promotions == []
        assert transport.calls == config.max_retries + 1

    def test_non_retryable_client_error_raises_immediately(
        self, config: ExtractionConfig
    ):
        transport = ScriptedTransport([(403, "forbidden")])
        extractor = PromotionExtractor(config, transport=transport)
        with pytest.raises(NonRetryableExtractionError):
            extractor.extract_promotions(
                E2E_SOURCE_TEXT,
                target_domain="example.com",
                source_url="https://example.com/promo",
            )
        assert len(transport.calls) == 1

    @pytest.mark.parametrize(
        "body",
        [
            "not json at all",
            json.dumps({"choices": []}),
            json.dumps({"choices": [{"message": {}}]}),
            json.dumps({"unexpected": "shape"}),
            json.dumps([1, 2, 3]),
        ],
    )
    def test_malformed_response_bodies_degrade_safely(
        self, config: ExtractionConfig, body
    ):
        transport = ScriptedTransport([(200, body)])
        extractor = PromotionExtractor(config, transport=transport)
        assert (
            extractor.extract_promotions(
                E2E_SOURCE_TEXT,
                target_domain="example.com",
                source_url="https://example.com/promo",
            )
            == []
        )

    @pytest.mark.parametrize(
        "content",
        [
            "not json",
            "```json\n{broken\n```",
            json.dumps({"promotions": "not-a-list"}),
            json.dumps({"promotions": ["a", 42, None]}),
            json.dumps("just a string"),
        ],
    )
    def test_malformed_assistant_content_degrades_safely(
        self, config: ExtractionConfig, content
    ):
        transport = ScriptedTransport([(200, openai_envelope(content))])
        extractor = PromotionExtractor(config, transport=transport)
        assert (
            extractor.extract_promotions(
                E2E_SOURCE_TEXT,
                target_domain="example.com",
                source_url="https://example.com/promo",
            )
            == []
        )


# ---------------------------------------------------------------------------
# PromotionExtractor: extract_from_evaluations
# ---------------------------------------------------------------------------


class TestExtractFromEvaluations:
    def _evaluation(self, *, viable: bool, window: str) -> ScanEvaluation:
        return ScanEvaluation(
            relevance_score=0.9 if viable else 0.1,
            matched_categories={"positive_welcome": ["welcome bonus"]},
            matched_noise_terms=[],
            disqualifiers=[] if viable else ["insufficient_signals"],
            is_viable=viable,
            candidate_window=window,
            candidate_windows=[window] if window else [],
            window_word_count=len(window.split()),
            word_count=100,
            source_text_length=100,
        )

    def test_viable_evaluations_are_extracted_in_order(
        self, config: ExtractionConfig
    ):
        first = build_campaign(title="First Offer")
        second = build_campaign(title="Second Offer")
        transport = ScriptedTransport(
            [
                (200, openai_envelope(json.dumps(first))),
                (200, openai_envelope(json.dumps(second))),
            ]
        )
        extractor = PromotionExtractor(config, transport=transport)
        evaluations = [
            self._evaluation(viable=True, window="first window text"),
            self._evaluation(viable=False, window="skipped window"),
            self._evaluation(viable=True, window="second window text"),
            self._evaluation(viable=True, window=""),
        ]
        promotions = extractor.extract_from_evaluations(
            evaluations,
            target_domain="example.com",
            source_url="https://example.com/promo",
        )
        assert [p.title for p in promotions] == ["First Offer", "Second Offer"]
        assert len(transport.calls) == 2
        assert transport.calls[0]["payload"]["messages"][1]["content"] == (
            "first window text"
        )
        assert transport.calls[1]["payload"]["messages"][1]["content"] == (
            "second window text"
        )

    def test_non_sequence_input_raises_type_error(
        self, config: ExtractionConfig
    ):
        extractor = PromotionExtractor(config, transport=ScriptedTransport([]))
        with pytest.raises(TypeError):
            extractor.extract_from_evaluations(
                "not a sequence",  # type: ignore[arg-type]
                target_domain="example.com",
                source_url="https://example.com/promo",
            )

    def test_non_evaluation_items_raise_type_error(
        self, config: ExtractionConfig
    ):
        extractor = PromotionExtractor(config, transport=ScriptedTransport([]))
        with pytest.raises(TypeError):
            extractor.extract_from_evaluations(
                [self._evaluation(viable=True, window="ok"), "bogus"],
                target_domain="example.com",
                source_url="https://example.com/promo",
            )


# ---------------------------------------------------------------------------
# PromotionExtractor: defensive argument validation and construction
# ---------------------------------------------------------------------------


class TestPromotionExtractorDefensive:
    def test_constructor_rejects_invalid_config(self):
        with pytest.raises(TypeError):
            PromotionExtractor("not a config")  # type: ignore[arg-type]

    def test_constructor_rejects_invalid_validator(
        self, config: ExtractionConfig
    ):
        with pytest.raises(TypeError):
            PromotionExtractor(config, validator="not a validator")

    def test_constructor_rejects_non_callable_transport(
        self, config: ExtractionConfig
    ):
        with pytest.raises(TypeError):
            PromotionExtractor(config, transport="not callable")

    def test_extract_promotions_rejects_non_string_candidate(
        self, config: ExtractionConfig
    ):
        extractor = PromotionExtractor(config, transport=ScriptedTransport([]))
        with pytest.raises(TypeError):
            extractor.extract_promotions(
                123,  # type: ignore[arg-type]
                target_domain="example.com",
                source_url="https://example.com/promo",
            )

    def test_extract_promotions_rejects_empty_target_domain(
        self, config: ExtractionConfig
    ):
        extractor = PromotionExtractor(config, transport=ScriptedTransport([]))
        with pytest.raises(ValueError):
            extractor.extract_promotions(
                "text", target_domain="   ", source_url="https://example.com"
            )

    def test_extract_promotions_rejects_empty_source_url(
        self, config: ExtractionConfig
    ):
        extractor = PromotionExtractor(config, transport=ScriptedTransport([]))
        with pytest.raises(ValueError):
            extractor.extract_promotions(
                "text", target_domain="example.com", source_url=""
            )

    def test_extract_promotions_rejects_non_string_raw_content(
        self, config: ExtractionConfig
    ):
        extractor = PromotionExtractor(config, transport=ScriptedTransport([]))
        with pytest.raises(TypeError):
            extractor.extract_promotions(
                "text",
                target_domain="example.com",
                source_url="https://example.com",
                raw_content=123,  # type: ignore[arg-type]
            )

    def test_transport_receives_expected_request_shape(
        self, config: ExtractionConfig
    ):
        transport = ScriptedTransport(
            [(200, openai_envelope(json.dumps(build_campaign())))]
        )
        extractor = PromotionExtractor(config, transport=transport)
        extractor.extract_promotions(
            E2E_SOURCE_TEXT,
            target_domain="example.com",
            source_url="https://example.com/promo",
        )
        call = transport.calls[0]
        assert call["url"] == config.endpoint_url
        assert call["headers"]["Authorization"] == "Bearer test-key"
        assert call["headers"]["Content-Type"] == "application/json"
        assert call["payload"]["model"] == config.model_identifier
        assert call["payload"]["temperature"] == config.temperature
        assert call["payload"]["messages"][0]["role"] == "system"
        assert call["payload"]["messages"][1]["content"] == E2E_SOURCE_TEXT
