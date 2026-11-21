"""Deterministic invariant validation for extracted promotional parameters.

Audits candidate ``Promotion`` aggregates against sports-wagering domain
guardrails before lifecycle advancement. Every threshold is a configurable
constructor parameter; this module hardcodes only the canonical model field
names (the structural contract of ``bonus_engine.models``).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .models import (
    CalculationBase,
    MonetaryConstraints,
    Promotion,
    TemporalValidity,
    WageringRequirement,
)

__all__ = ["DiscrepancyCode", "ValidationReport", "InvariantValidator"]

_WHITESPACE_RUN_PATTERN = re.compile(r"\s+")


def _normalize_for_matching(text: str) -> str:
    """Casefold, strip diacritics, and collapse whitespace for tolerant matching."""
    decomposed = "".join(
        char
        for char in unicodedata.normalize("NFD", text.casefold())
        if not unicodedata.combining(char)
    )
    return _WHITESPACE_RUN_PATTERN.sub(" ", decomposed).strip()

class DiscrepancyCode(str, Enum):
    """Machine-readable discrepancy codes emitted by invariant validation."""

    ROLLOVER_BELOW_SPORTS_FLOOR = "rollover_below_sports_floor"
    ROLLOVER_ABOVE_SPORTS_CEILING = "rollover_above_sports_ceiling"
    CASINO_STYLE_ROLLOVER = "casino_style_rollover"
    CALCULATION_BASE_AMBIGUOUS = "calculation_base_ambiguous"
    SELECTION_ODDS_IMPLAUSIBLE = "selection_odds_implausible"
    ACCUMULATOR_CONSTRAINTS_IMPLAUSIBLE = "accumulator_constraints_implausible"
    TEMPORAL_WINDOW_EXCESSIVE = "temporal_window_excessive"
    WAGERING_DEADLINE_IMPLAUSIBLE = "wagering_deadline_implausible"
    MISSING_EVIDENCE_CITATION = "missing_evidence_citation"
    STRUCTURAL_INVARIANT_VIOLATION = "structural_invariant_violation"
    EVIDENCE_QUOTE_UNVERIFIED = "evidence_quote_unverified"
    BONUS_PERCENTAGE_IMPLAUSIBLE = "bonus_percentage_implausible"

@dataclass
class ValidationReport:
    """Structured outcome of one invariant validation pass."""

    is_compliant: bool
    discrepancy_codes: List[DiscrepancyCode]
    violations: List[str]
    evidence_gaps: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Export a JSON-safe dictionary representation."""
        return {
            "is_compliant": self.is_compliant,
            "discrepancy_codes": [code.value for code in self.discrepancy_codes],
            "violations": list(self.violations),
            "evidence_gaps": list(self.evidence_gaps),
        }

class InvariantValidator:
    """Audits promotional parameters against sports-wagering domain guardrails.

    Args:
        min_sports_rollover: Lowest feasible sports rollover multiplier.
        max_sports_rollover: Highest feasible sports rollover multiplier.
        casino_rollover_threshold: Rollover at or above which the requirement
            is classified as casino-style rather than sports wagering.
        min_plausible_selection_odds: Lowest plausible decimal odds per selection.
        max_plausible_selection_odds: Highest plausible decimal odds per selection.
        max_plausible_accumulator_odds: Highest plausible accumulator odds floor.
        max_plausible_accumulator_legs: Highest plausible accumulator leg count.
        max_temporal_window_days: Longest plausible validity window in days.
        max_wagering_deadline_days: Longest plausible wagering deadline in days.
        min_plausible_bonus_percentage: Lowest plausible bonus match percentage.
        max_plausible_bonus_percentage: Highest plausible bonus match percentage.
    """

    _WAGERING_NUMERIC_FIELDS: Tuple[str, ...] = (
        "rollover_multiplier",
        "min_selection_odds",
        "max_selection_odds",
        "min_accumulator_odds",
        "min_accumulator_legs",
        "max_stake_contribution_pct",
        "completion_deadline_days",
    )
    _MONETARY_NUMERIC_FIELDS: Tuple[str, ...] = (
        "min_qualifying_deposit",
        "max_qualifying_deposit",
        "min_bonus_amount",
        "max_bonus_amount",
        "max_conversion_amount",
        "bonus_percentage",
    )

    def __init__(
        self,
        *,
        min_sports_rollover: float = 1.0,
        max_sports_rollover: float = 20.0,
        casino_rollover_threshold: float = 30.0,
        min_plausible_selection_odds: float = 1.01,
        max_plausible_selection_odds: float = 100.0,
        max_plausible_accumulator_odds: float = 1000.0,
        max_plausible_accumulator_legs: int = 20,
        max_temporal_window_days: int = 365,
        max_wagering_deadline_days: int = 365,
        min_plausible_bonus_percentage: float = 1.0,
        max_plausible_bonus_percentage: float = 100.0,
    ) -> None:
        numeric_params = (
            ("min_sports_rollover", min_sports_rollover),
            ("max_sports_rollover", max_sports_rollover),
            ("casino_rollover_threshold", casino_rollover_threshold),
            ("min_plausible_selection_odds", min_plausible_selection_odds),
            ("max_plausible_selection_odds", max_plausible_selection_odds),
            ("max_plausible_accumulator_odds", max_plausible_accumulator_odds),
            ("min_plausible_bonus_percentage", min_plausible_bonus_percentage),
            ("max_plausible_bonus_percentage", max_plausible_bonus_percentage),
        )
        for name, value in numeric_params:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in (
            ("max_plausible_accumulator_legs", max_plausible_accumulator_legs),
            ("max_temporal_window_days", max_temporal_window_days),
            ("max_wagering_deadline_days", max_wagering_deadline_days),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max_sports_rollover < min_sports_rollover:
            raise ValueError(
                "max_sports_rollover must be >= min_sports_rollover"
            )
        if casino_rollover_threshold < max_sports_rollover:
            raise ValueError(
                "casino_rollover_threshold must be >= max_sports_rollover"
            )
        if max_plausible_bonus_percentage < min_plausible_bonus_percentage:
            raise ValueError(
                "max_plausible_bonus_percentage must be >= "
                "min_plausible_bonus_percentage"
            )
        self._min_sports_rollover = float(min_sports_rollover)
        self._max_sports_rollover = float(max_sports_rollover)
        self._casino_rollover_threshold = float(casino_rollover_threshold)
        self._min_plausible_selection_odds = float(min_plausible_selection_odds)
        self._max_plausible_selection_odds = float(max_plausible_selection_odds)
        self._max_plausible_accumulator_odds = float(max_plausible_accumulator_odds)
        self._max_plausible_accumulator_legs = max_plausible_accumulator_legs
        self._max_temporal_window_days = max_temporal_window_days
        self._max_wagering_deadline_days = max_wagering_deadline_days
        self._min_plausible_bonus_percentage = float(min_plausible_bonus_percentage)
        self._max_plausible_bonus_percentage = float(max_plausible_bonus_percentage)

    def validate(
        self,
        promotion: Promotion,
        *,
        source_text: Optional[str] = None,
    ) -> ValidationReport:
        """Run all invariant audits and return a structured report.

        When ``source_text`` is provided, every evidence citation's quoted
        excerpt is verified to appear (whitespace/case/diacritic-tolerant)
        in the source text; fabricated quotes are flagged.

        Raises:
            TypeError: If ``promotion`` is not a ``Promotion`` instance.
        """
        if not isinstance(promotion, Promotion):
            raise TypeError("validate expects a Promotion instance")
        codes: List[DiscrepancyCode] = []
        violations: List[str] = []
        evidence_gaps: List[str] = []

        structural = promotion.validate_invariants()
        if structural:
            codes.append(DiscrepancyCode.STRUCTURAL_INVARIANT_VIOLATION)
            violations.extend(structural)

        if promotion.wagering is not None:
            self._audit_wagering(promotion, promotion.wagering, codes, violations)
        if promotion.temporal is not None:
            self._audit_temporal(promotion.temporal, codes, violations)
        if promotion.monetary is not None:
            self._audit_monetary(promotion.monetary, codes, violations)
        self._audit_evidence(
            promotion, codes, violations, evidence_gaps, source_text
        )

        unique_codes = list(dict.fromkeys(codes))
        return ValidationReport(
            is_compliant=not unique_codes,
            discrepancy_codes=unique_codes,
            violations=violations,
            evidence_gaps=evidence_gaps,
        )

    def _audit_wagering(
        self,
        promotion: Promotion,
        wagering: WageringRequirement,
        codes: List[DiscrepancyCode],
        violations: List[str],
    ) -> None:
        """Enforce rollover, calculation-base, and odds guardrails."""
        if (
            wagering.rollover_multiplier is not None
            and wagering.rollover_multiplier < self._min_sports_rollover
        ):
            codes.append(DiscrepancyCode.ROLLOVER_BELOW_SPORTS_FLOOR)
            violations.append(
                f"rollover_multiplier {wagering.rollover_multiplier} is below "
                f"the sports wagering floor {self._min_sports_rollover}"
            )
        elif (
            wagering.rollover_multiplier is not None
            and wagering.rollover_multiplier >= self._casino_rollover_threshold
        ):
            codes.append(DiscrepancyCode.CASINO_STYLE_ROLLOVER)
            violations.append(
                f"rollover_multiplier {wagering.rollover_multiplier} meets or "
                f"exceeds the casino-style threshold "
                f"{self._casino_rollover_threshold}"
            )
        elif (
            wagering.rollover_multiplier is not None
            and wagering.rollover_multiplier > self._max_sports_rollover
        ):
            codes.append(DiscrepancyCode.ROLLOVER_ABOVE_SPORTS_CEILING)
            violations.append(
                f"rollover_multiplier {wagering.rollover_multiplier} exceeds "
                f"the feasible sports ceiling {self._max_sports_rollover}"
            )

        if (
            wagering.rollover_multiplier is not None
            and wagering.calculation_base == CalculationBase.UNSPECIFIED
        ):
            codes.append(DiscrepancyCode.CALCULATION_BASE_AMBIGUOUS)
            violations.append(
                "a positive rollover requirement lacks an explicit "
                "calculation base; rollover scope is ambiguous"
            )

        if (
            wagering.min_selection_odds is not None
            and not (
                self._min_plausible_selection_odds
                <= wagering.min_selection_odds
                <= self._max_plausible_selection_odds
            )
        ):
            codes.append(DiscrepancyCode.SELECTION_ODDS_IMPLAUSIBLE)
            violations.append(
                f"min_selection_odds {wagering.min_selection_odds} is outside "
                f"the plausible range "
                f"[{self._min_plausible_selection_odds}, "
                f"{self._max_plausible_selection_odds}]"
            )
        if (
            wagering.max_selection_odds is not None
            and wagering.max_selection_odds > self._max_plausible_selection_odds
        ):
            codes.append(DiscrepancyCode.SELECTION_ODDS_IMPLAUSIBLE)
            violations.append(
                f"max_selection_odds {wagering.max_selection_odds} exceeds "
                f"the plausible ceiling {self._max_plausible_selection_odds}"
            )
        if (
            wagering.min_accumulator_odds is not None
            and wagering.min_accumulator_odds > self._max_plausible_accumulator_odds
        ):
            codes.append(DiscrepancyCode.ACCUMULATOR_CONSTRAINTS_IMPLAUSIBLE)
            violations.append(
                f"min_accumulator_odds {wagering.min_accumulator_odds} exceeds "
                f"the plausible ceiling {self._max_plausible_accumulator_odds}"
            )
        if (
            wagering.min_accumulator_legs is not None
            and wagering.min_accumulator_legs > self._max_plausible_accumulator_legs
        ):
            codes.append(DiscrepancyCode.ACCUMULATOR_CONSTRAINTS_IMPLAUSIBLE)
            violations.append(
                f"min_accumulator_legs {wagering.min_accumulator_legs} exceeds "
                f"the plausible ceiling {self._max_plausible_accumulator_legs}"
            )

    def _audit_temporal(
        self,
        temporal: TemporalValidity,
        codes: List[DiscrepancyCode],
        violations: List[str],
    ) -> None:
        """Enforce realistic temporal validity windows."""
        if temporal.valid_from is not None and temporal.valid_until is not None:
            span_days = (
                temporal.valid_until - temporal.valid_from
            ).total_seconds() / 86400.0
            if span_days > self._max_temporal_window_days:
                codes.append(DiscrepancyCode.TEMPORAL_WINDOW_EXCESSIVE)
                violations.append(
                    f"temporal validity window spans {span_days:.1f} days, "
                    f"exceeding the plausible maximum "
                    f"{self._max_temporal_window_days} days"
                )
        if temporal.wagering_deadline_days is not None and (
            temporal.wagering_deadline_days < 1
            or temporal.wagering_deadline_days > self._max_wagering_deadline_days
        ):
            codes.append(DiscrepancyCode.WAGERING_DEADLINE_IMPLAUSIBLE)
            violations.append(
                f"wagering_deadline_days "
                f"{temporal.wagering_deadline_days} is outside the plausible "
                f"range [1, {self._max_wagering_deadline_days}]"
            )

    def _audit_monetary(
        self,
        monetary: MonetaryConstraints,
        codes: List[DiscrepancyCode],
        violations: List[str],
    ) -> None:
        """Enforce plausibility bounds on monetary valuation metrics."""
        if (
            monetary.bonus_percentage is not None
            and not (
                self._min_plausible_bonus_percentage
                <= monetary.bonus_percentage
                <= self._max_plausible_bonus_percentage
            )
        ):
            codes.append(DiscrepancyCode.BONUS_PERCENTAGE_IMPLAUSIBLE)
            violations.append(
                f"bonus_percentage {monetary.bonus_percentage} is outside the "
                f"plausible range [{self._min_plausible_bonus_percentage}, "
                f"{self._max_plausible_bonus_percentage}]"
            )

    def _audit_evidence(
        self,
        promotion: Promotion,
        codes: List[DiscrepancyCode],
        violations: List[str],
        evidence_gaps: List[str],
        source_text: Optional[str],
    ) -> None:
        """Require citations for numeric claims and verify quote authenticity."""
        citations = (
            promotion.provenance.citations if promotion.provenance else []
        )
        cited_fields = {
            citation.field_name
            for citation in citations
            if isinstance(citation.field_name, str)
        }
        for field_name, value in self._iter_numeric_claims(promotion):
            if value is None:
                continue
            if field_name not in cited_fields:
                evidence_gaps.append(field_name)
        if evidence_gaps:
            codes.append(DiscrepancyCode.MISSING_EVIDENCE_CITATION)
            violations.append(
                "Numerical claims lack evidence citations: "
                + ", ".join(evidence_gaps)
            )

        if source_text is None or not isinstance(source_text, str) \
                or not source_text.strip():
            return
        normalized_source = _normalize_for_matching(source_text)
        unverified: List[str] = []
        for citation in citations:
            raw_text = citation.raw_text
            if not isinstance(raw_text, str) or not raw_text.strip():
                unverified.append(citation.field_name)
                continue
            if _normalize_for_matching(raw_text) not in normalized_source:
                unverified.append(citation.field_name)
        if unverified:
            codes.append(DiscrepancyCode.EVIDENCE_QUOTE_UNVERIFIED)
            violations.append(
                "Cited excerpts not found in source text: "
                + ", ".join(sorted(set(unverified)))
            )

    def _iter_numeric_claims(
        self, promotion: Promotion
    ) -> Iterator[Tuple[str, Any]]:
        """Yield ``(canonical_field_name, value)`` for present numeric claims."""
        wagering = promotion.wagering
        if wagering is not None:
            for name in self._WAGERING_NUMERIC_FIELDS:
                yield name, getattr(wagering, name)
        monetary = promotion.monetary
        if monetary is not None:
            for name in self._MONETARY_NUMERIC_FIELDS:
                yield name, getattr(monetary, name)
