"""Canonical domain models for sports betting promotional campaigns.

Target-agnostic representations of promotions extracted from bookmaker
domains. This module deliberately contains no target-specific vocabulary:
raw-term matching belongs to ``sportsbook_lexicon.json`` and per-target
``field_mappings`` stored in SQLite.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from odds_normalizer.utils import SafePayloadUtils

__all__ = [
    "PromotionType",
    "CalculationBase",
    "VerificationState",
    "TemporalValidity",
    "WageringRequirement",
    "MonetaryConstraints",
    "EvidenceCitation",
    "SourceProvenance",
    "Promotion",
]

_E = TypeVar("_E", bound=Enum)


class PromotionType(str, Enum):
    """Canonical promotion categories supported across bookmaker targets."""

    WELCOME = "welcome"
    DEPOSIT_MATCH = "deposit_match"
    FREEBET = "freebet"
    RELOAD = "reload"
    CASHBACK = "cashback"
    ODDS_BOOST = "odds_boost"
    RISK_FREE_BET = "risk_free_bet"
    OTHER = "other"


class CalculationBase(str, Enum):
    """Canonical bases on which a rollover requirement may be computed."""

    BONUS_AMOUNT = "bonus_amount"
    DEPOSIT_PLUS_BONUS = "deposit_plus_bonus"
    FREEBET_STAKE = "freebet_stake"
    WINNINGS_ONLY = "winnings_only"
    STAKE_PLUS_WINNINGS = "stake_plus_winnings"


class VerificationState(str, Enum):
    """Promotional lifecycle verification states."""

    EXTRACTED = "extracted"
    INVARIANT_VALIDATED = "invariant_validated"
    OPERATOR_CONFIRMED = "operator_confirmed"
    REJECTED = "rejected"


_ALLOWED_TRANSITIONS: Dict[VerificationState, frozenset] = {
    VerificationState.EXTRACTED: frozenset(
        {VerificationState.INVARIANT_VALIDATED, VerificationState.REJECTED}
    ),
    VerificationState.INVARIANT_VALIDATED: frozenset(
        {VerificationState.OPERATOR_CONFIRMED, VerificationState.REJECTED}
    ),
    VerificationState.OPERATOR_CONFIRMED: frozenset(),
    VerificationState.REJECTED: frozenset(),
}


def _parse_datetime(value: Any) -> Optional[datetime]:
    """Defensively parse an ISO-8601 datetime, returning ``None`` on failure."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except (ValueError, AttributeError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _serialize_datetime(value: Optional[datetime]) -> Optional[str]:
    """Serialize a datetime to ISO-8601, or ``None`` when absent."""
    return value.isoformat() if isinstance(value, datetime) else None


def _coerce_enum(
    value: Any, enum_cls: Type[_E], default: Optional[_E] = None
) -> Optional[_E]:
    """Defensively coerce a raw value into an enum member."""
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value.strip().lower())
        except ValueError:
            if default is not None:
                return default
            raise ValueError(
                f"Unknown {enum_cls.__name__} value: {value!r}"
            ) from None
    if default is not None:
        return default
    raise TypeError(f"Cannot coerce {value!r} to {enum_cls.__name__}")


def _coerce_opt_float(value: Any) -> Optional[float]:
    """Coerce to float or ``None``, never raising on garbage input."""
    if value is None:
        return None
    return SafePayloadUtils.coerce_float(value)


def _coerce_opt_int(value: Any) -> Optional[int]:
    """Coerce to int or ``None``, never raising on garbage input."""
    coerced = _coerce_opt_float(value)
    if coerced is None:
        return None
    return int(coerced)


def _coerce_str_list(value: Any) -> List[str]:
    """Coerce an arbitrary payload fragment into a list of strings."""
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in (SafePayloadUtils.coerce_str(v) for v in value) if item]


@dataclass
class EvidenceCitation:
    """Maps one extracted canonical value back to the raw promotional terms."""

    field_name: str
    raw_text: str
    locator: Optional[str] = None
    captured_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if not isinstance(self.field_name, str) or not self.field_name.strip():
            raise ValueError("EvidenceCitation.field_name must be a non-empty string")
        if not isinstance(self.raw_text, str):
            raise TypeError("EvidenceCitation.raw_text must be a string")
        self.locator = SafePayloadUtils.coerce_str(self.locator)
        self.captured_at = _parse_datetime(self.captured_at)

    def to_dict(self) -> Dict[str, Any]:
        """Export a JSON-safe dictionary representation."""
        return {
            "field_name": self.field_name,
            "raw_text": self.raw_text,
            "locator": self.locator,
            "captured_at": _serialize_datetime(self.captured_at),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceCitation":
        """Defensively rebuild a citation from a dictionary payload."""
        if not isinstance(data, dict):
            raise TypeError("EvidenceCitation.from_dict expects a dictionary")
        field_name = SafePayloadUtils.coerce_str(data.get("field_name"))
        raw_text = data.get("raw_text")
        if field_name is None:
            raise ValueError("EvidenceCitation requires a non-empty 'field_name'")
        if not isinstance(raw_text, str):
            raise ValueError("EvidenceCitation requires a string 'raw_text'")
        return cls(
            field_name=field_name,
            raw_text=raw_text,
            locator=SafePayloadUtils.coerce_str(data.get("locator")),
            captured_at=_parse_datetime(data.get("captured_at")),
        )


@dataclass
class SourceProvenance:
    """Source attribution and auditability metadata for an extraction."""

    source_url: str
    content_hash: str
    captured_at: datetime
    extraction_method: str = "automated"
    citations: List[EvidenceCitation] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.source_url, str) or not self.source_url.strip():
            raise ValueError("SourceProvenance.source_url must be a non-empty string")
        if not isinstance(self.content_hash, str) or not self.content_hash.strip():
            raise ValueError("SourceProvenance.content_hash must be a non-empty string")
        parsed_captured = _parse_datetime(self.captured_at)
        if parsed_captured is None:
            raise ValueError("SourceProvenance.captured_at must be a valid datetime")
        self.captured_at = parsed_captured
        method = SafePayloadUtils.coerce_str(self.extraction_method)
        self.extraction_method = method if method is not None else "automated"
        if not isinstance(self.citations, list):
            raise TypeError("SourceProvenance.citations must be a list")

    @staticmethod
    def compute_content_hash(raw_content: Union[str, bytes]) -> str:
        """Compute the SHA-256 change-detection hash of raw captured content."""
        if isinstance(raw_content, str):
            raw_content = raw_content.encode("utf-8")
        if not isinstance(raw_content, bytes):
            raise TypeError("raw_content must be a string or bytes")
        return hashlib.sha256(raw_content).hexdigest()

    def add_citation(self, citation: EvidenceCitation) -> None:
        """Append an evidence citation after defensive type checking."""
        if not isinstance(citation, EvidenceCitation):
            raise TypeError("citation must be an EvidenceCitation instance")
        self.citations.append(citation)

    def to_dict(self) -> Dict[str, Any]:
        """Export a JSON-safe dictionary representation."""
        return {
            "source_url": self.source_url,
            "content_hash": self.content_hash,
            "captured_at": _serialize_datetime(self.captured_at),
            "extraction_method": self.extraction_method,
            "citations": [citation.to_dict() for citation in self.citations],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceProvenance":
        """Defensively rebuild provenance from a dictionary payload."""
        if not isinstance(data, dict):
            raise TypeError("SourceProvenance.from_dict expects a dictionary")
        source_url = SafePayloadUtils.coerce_str(data.get("source_url"))
        content_hash = SafePayloadUtils.coerce_str(data.get("content_hash"))
        captured_at = _parse_datetime(data.get("captured_at"))
        if source_url is None or content_hash is None or captured_at is None:
            raise ValueError(
                "SourceProvenance requires 'source_url', 'content_hash', 'captured_at'"
            )
        raw_citations = data.get("citations", [])
        citations = (
            [EvidenceCitation.from_dict(item) for item in raw_citations]
            if isinstance(raw_citations, list)
            else []
        )
        return cls(
            source_url=source_url,
            content_hash=content_hash,
            captured_at=captured_at,
            extraction_method=SafePayloadUtils.coerce_str(
                data.get("extraction_method")
            )
            or "automated",
            citations=citations,
        )


@dataclass
class TemporalValidity:
    """Temporal validity constraints for a promotion."""

    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    wagering_deadline_days: Optional[int] = None
    timezone_name: str = "UTC"

    def __post_init__(self) -> None:
        self.valid_from = _parse_datetime(self.valid_from)
        self.valid_until = _parse_datetime(self.valid_until)
        self.wagering_deadline_days = _coerce_opt_int(self.wagering_deadline_days)
        tz_name = SafePayloadUtils.coerce_str(self.timezone_name)
        self.timezone_name = tz_name if tz_name is not None else "UTC"
        if self.wagering_deadline_days is not None and self.wagering_deadline_days < 0:
            raise ValueError("wagering_deadline_days must be non-negative")

    def validate_invariants(self) -> List[str]:
        """Return human-readable violations of temporal invariants."""
        violations: List[str] = []
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_until < self.valid_from
        ):
            violations.append("valid_until precedes valid_from")
        return violations

    def to_dict(self) -> Dict[str, Any]:
        """Export a JSON-safe dictionary representation."""
        return {
            "valid_from": _serialize_datetime(self.valid_from),
            "valid_until": _serialize_datetime(self.valid_until),
            "wagering_deadline_days": self.wagering_deadline_days,
            "timezone_name": self.timezone_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalValidity":
        """Defensively rebuild temporal constraints from a dictionary."""
        if not isinstance(data, dict):
            raise TypeError("TemporalValidity.from_dict expects a dictionary")
        return cls(
            valid_from=_parse_datetime(data.get("valid_from")),
            valid_until=_parse_datetime(data.get("valid_until")),
            wagering_deadline_days=_coerce_opt_int(data.get("wagering_deadline_days")),
            timezone_name=SafePayloadUtils.coerce_str(data.get("timezone_name"))
            or "UTC",
        )


@dataclass
class WageringRequirement:
    """Mathematical parameters required to evaluate wagering feasibility."""

    rollover_multiplier: float
    calculation_base: CalculationBase = CalculationBase.BONUS_AMOUNT
    min_selection_odds: Optional[float] = None
    max_selection_odds: Optional[float] = None
    min_accumulator_odds: Optional[float] = None
    min_accumulator_legs: Optional[int] = None
    max_stake_contribution_pct: Optional[float] = None
    completion_deadline_days: Optional[int] = None
    eligible_market_types: List[str] = field(default_factory=list)
    excluded_market_types: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        rollover = _coerce_opt_float(self.rollover_multiplier)
        if rollover is None or rollover < 0:
            raise ValueError("rollover_multiplier must be a non-negative number")
        self.rollover_multiplier = rollover
        self.calculation_base = (
            _coerce_enum(self.calculation_base, CalculationBase)
            or CalculationBase.BONUS_AMOUNT
        )
        self.min_selection_odds = _coerce_opt_float(self.min_selection_odds)
        self.max_selection_odds = _coerce_opt_float(self.max_selection_odds)
        self.min_accumulator_odds = _coerce_opt_float(self.min_accumulator_odds)
        self.min_accumulator_legs = _coerce_opt_int(self.min_accumulator_legs)
        self.max_stake_contribution_pct = _coerce_opt_float(
            self.max_stake_contribution_pct
        )
        self.completion_deadline_days = _coerce_opt_int(self.completion_deadline_days)
        self.eligible_market_types = _coerce_str_list(self.eligible_market_types)
        self.excluded_market_types = _coerce_str_list(self.excluded_market_types)

    def validate_invariants(self) -> List[str]:
        """Return human-readable violations of wagering invariants."""
        violations: List[str] = []
        if self.min_selection_odds is not None and self.min_selection_odds < 1.0:
            violations.append("min_selection_odds must be >= 1.0 in decimal format")
        if (
            self.min_selection_odds is not None
            and self.max_selection_odds is not None
            and self.max_selection_odds < self.min_selection_odds
        ):
            violations.append("max_selection_odds precedes min_selection_odds")
        if (
            self.min_selection_odds is not None
            and self.min_accumulator_odds is not None
            and self.min_accumulator_odds < self.min_selection_odds
        ):
            violations.append(
                "min_accumulator_odds must be >= min_selection_odds"
            )
        if self.min_accumulator_legs is not None and self.min_accumulator_legs < 2:
            violations.append("min_accumulator_legs must be >= 2")
        if (
            self.max_stake_contribution_pct is not None
            and not 0.0 <= self.max_stake_contribution_pct <= 100.0
        ):
            violations.append("max_stake_contribution_pct must be within [0, 100]")
        if (
            self.completion_deadline_days is not None
            and self.completion_deadline_days < 0
        ):
            violations.append("completion_deadline_days must be non-negative")
        return violations

    def to_dict(self) -> Dict[str, Any]:
        """Export a JSON-safe dictionary representation."""
        return {
            "rollover_multiplier": self.rollover_multiplier,
            "calculation_base": self.calculation_base.value,
            "min_selection_odds": self.min_selection_odds,
            "max_selection_odds": self.max_selection_odds,
            "min_accumulator_odds": self.min_accumulator_odds,
            "min_accumulator_legs": self.min_accumulator_legs,
            "max_stake_contribution_pct": self.max_stake_contribution_pct,
            "completion_deadline_days": self.completion_deadline_days,
            "eligible_market_types": list(self.eligible_market_types),
            "excluded_market_types": list(self.excluded_market_types),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WageringRequirement":
        """Defensively rebuild a wagering requirement from a dictionary."""
        if not isinstance(data, dict):
            raise TypeError("WageringRequirement.from_dict expects a dictionary")
        rollover = _coerce_opt_float(data.get("rollover_multiplier"))
        if rollover is None:
            raise ValueError("WageringRequirement requires 'rollover_multiplier'")
        return cls(
            rollover_multiplier=rollover,
            calculation_base=_coerce_enum(
                data.get("calculation_base"), CalculationBase
            )
            or CalculationBase.BONUS_AMOUNT,
            min_selection_odds=_coerce_opt_float(data.get("min_selection_odds")),
            max_selection_odds=_coerce_opt_float(data.get("max_selection_odds")),
            min_accumulator_odds=_coerce_opt_float(data.get("min_accumulator_odds")),
            min_accumulator_legs=_coerce_opt_int(data.get("min_accumulator_legs")),
            max_stake_contribution_pct=_coerce_opt_float(
                data.get("max_stake_contribution_pct")
            ),
            completion_deadline_days=_coerce_opt_int(
                data.get("completion_deadline_days")
            ),
            eligible_market_types=_coerce_str_list(data.get("eligible_market_types")),
            excluded_market_types=_coerce_str_list(data.get("excluded_market_types")),
        )


@dataclass
class MonetaryConstraints:
    """Monetary limits and qualifying deposit requirements."""

    min_qualifying_deposit: Optional[float] = None
    max_qualifying_deposit: Optional[float] = None
    min_bonus_amount: Optional[float] = None
    max_bonus_amount: Optional[float] = None
    max_conversion_amount: Optional[float] = None
    currency: Optional[str] = None

    def __post_init__(self) -> None:
        self.min_qualifying_deposit = _coerce_opt_float(
            self.min_qualifying_deposit
        )
        self.max_qualifying_deposit = _coerce_opt_float(
            self.max_qualifying_deposit
        )
        self.min_bonus_amount = _coerce_opt_float(self.min_bonus_amount)
        self.max_bonus_amount = _coerce_opt_float(self.max_bonus_amount)
        self.max_conversion_amount = _coerce_opt_float(self.max_conversion_amount)
        self.currency = SafePayloadUtils.coerce_str(self.currency)
        for label, value in (
            ("min_qualifying_deposit", self.min_qualifying_deposit),
            ("max_qualifying_deposit", self.max_qualifying_deposit),
            ("min_bonus_amount", self.min_bonus_amount),
            ("max_bonus_amount", self.max_bonus_amount),
            ("max_conversion_amount", self.max_conversion_amount),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{label} must be non-negative")

    def validate_invariants(self) -> List[str]:
        """Return human-readable violations of monetary invariants."""
        violations: List[str] = []
        if (
            self.min_qualifying_deposit is not None
            and self.max_qualifying_deposit is not None
            and self.max_qualifying_deposit < self.min_qualifying_deposit
        ):
            violations.append(
                "max_qualifying_deposit precedes min_qualifying_deposit"
            )
        if (
            self.min_bonus_amount is not None
            and self.max_bonus_amount is not None
            and self.max_bonus_amount < self.min_bonus_amount
        ):
            violations.append("max_bonus_amount precedes min_bonus_amount")
        return violations

    def to_dict(self) -> Dict[str, Any]:
        """Export a JSON-safe dictionary representation."""
        return {
            "min_qualifying_deposit": self.min_qualifying_deposit,
            "max_qualifying_deposit": self.max_qualifying_deposit,
            "min_bonus_amount": self.min_bonus_amount,
            "max_bonus_amount": self.max_bonus_amount,
            "max_conversion_amount": self.max_conversion_amount,
            "currency": self.currency,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MonetaryConstraints":
        """Defensively rebuild monetary constraints from a dictionary."""
        if not isinstance(data, dict):
            raise TypeError("MonetaryConstraints.from_dict expects a dictionary")
        return cls(
            min_qualifying_deposit=_coerce_opt_float(
                data.get("min_qualifying_deposit")
            ),
            max_qualifying_deposit=_coerce_opt_float(
                data.get("max_qualifying_deposit")
            ),
            min_bonus_amount=_coerce_opt_float(data.get("min_bonus_amount")),
            max_bonus_amount=_coerce_opt_float(data.get("max_bonus_amount")),
            max_conversion_amount=_coerce_opt_float(data.get("max_conversion_amount")),
            currency=SafePayloadUtils.coerce_str(data.get("currency")),
        )


@dataclass
class Promotion:
    """Aggregate root representing one promotion on a bookmaker domain.

    A single target domain carries a 1-to-N collection of ``Promotion``
    instances (welcome incentives, deposit matches, freebets, reloads, ...).
    """

    target_domain: str
    promotion_type: PromotionType
    title: str
    promotion_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    wagering: Optional[WageringRequirement] = None
    temporal: Optional[TemporalValidity] = None
    monetary: Optional[MonetaryConstraints] = None
    verification_state: VerificationState = VerificationState.EXTRACTED
    provenance: Optional[SourceProvenance] = None
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.target_domain, str) or not self.target_domain.strip():
            raise ValueError("Promotion.target_domain must be a non-empty string")
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("Promotion.title must be a non-empty string")
        if not isinstance(self.promotion_id, str) or not self.promotion_id.strip():
            raise ValueError("Promotion.promotion_id must be a non-empty string")
        self.promotion_type = _coerce_enum(
            self.promotion_type, PromotionType, PromotionType.OTHER
        )
        self.verification_state = _coerce_enum(
            self.verification_state, VerificationState, VerificationState.EXTRACTED
        )
        for label, value in (
            ("wagering", self.wagering),
            ("temporal", self.temporal),
            ("monetary", self.monetary),
            ("provenance", self.provenance),
        ):
            expected = {
                "wagering": WageringRequirement,
                "temporal": TemporalValidity,
                "monetary": MonetaryConstraints,
                "provenance": SourceProvenance,
            }[label]
            if value is not None and not isinstance(value, expected):
                raise TypeError(f"Promotion.{label} must be a {expected.__name__}")
        self.notes = SafePayloadUtils.coerce_str(self.notes)

    def validate_invariants(self) -> List[str]:
        """Aggregate invariant violations across all nested structures."""
        violations: List[str] = []
        if self.wagering is not None:
            violations.extend(self.wagering.validate_invariants())
        if self.temporal is not None:
            violations.extend(self.temporal.validate_invariants())
        if self.monetary is not None:
            violations.extend(self.monetary.validate_invariants())
        return violations

    def transition_to(self, next_state: VerificationState) -> None:
        """Advance the lifecycle state, enforcing the allowed transition map."""
        target = _coerce_enum(next_state, VerificationState)
        if target is None:
            raise ValueError("transition_to requires a VerificationState")
        allowed = _ALLOWED_TRANSITIONS.get(self.verification_state, frozenset())
        if target not in allowed:
            raise ValueError(
                f"Illegal lifecycle transition from "
                f"{self.verification_state.value!r} to {target.value!r}"
            )
        self.verification_state = target

    def fingerprint(self) -> str:
        """Stable SHA-256 fingerprint of the canonical serialized model."""
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Export a JSON-safe dictionary representation."""
        return {
            "promotion_id": self.promotion_id,
            "target_domain": self.target_domain,
            "promotion_type": self.promotion_type.value,
            "title": self.title,
            "wagering": self.wagering.to_dict() if self.wagering else None,
            "temporal": self.temporal.to_dict() if self.temporal else None,
            "monetary": self.monetary.to_dict() if self.monetary else None,
            "verification_state": self.verification_state.value,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Promotion":
        """Defensively rebuild a promotion from a dictionary payload."""
        if not isinstance(data, dict):
            raise TypeError("Promotion.from_dict expects a dictionary")
        target_domain = SafePayloadUtils.coerce_str(data.get("target_domain"))
        title = SafePayloadUtils.coerce_str(data.get("title"))
        if target_domain is None or title is None:
            raise ValueError("Promotion requires 'target_domain' and 'title'")
        promotion_type = _coerce_enum(data.get("promotion_type"), PromotionType)
        if promotion_type is None:
            raise ValueError("Promotion requires a valid 'promotion_type'")
        raw_wagering = data.get("wagering")
        raw_temporal = data.get("temporal")
        raw_monetary = data.get("monetary")
        raw_provenance = data.get("provenance")
        return cls(
            target_domain=target_domain,
            promotion_type=promotion_type,
            title=title,
            promotion_id=SafePayloadUtils.coerce_str(data.get("promotion_id"))
            or uuid.uuid4().hex,
            wagering=WageringRequirement.from_dict(raw_wagering)
            if isinstance(raw_wagering, dict)
            else None,
            temporal=TemporalValidity.from_dict(raw_temporal)
            if isinstance(raw_temporal, dict)
            else None,
            monetary=MonetaryConstraints.from_dict(raw_monetary)
            if isinstance(raw_monetary, dict)
            else None,
            verification_state=_coerce_enum(
                data.get("verification_state"),
                VerificationState,
                VerificationState.EXTRACTED,
            ),
            provenance=SourceProvenance.from_dict(raw_provenance)
            if isinstance(raw_provenance, dict)
            else None,
            notes=SafePayloadUtils.coerce_str(data.get("notes")),
        )
