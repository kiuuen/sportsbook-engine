"""Public interface for the bonus_engine package."""

from bonus_engine.database import BonusDatabaseManager
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

__all__ = [
    "BonusDatabaseManager",
    "CalculationBase",
    "EvidenceCitation",
    "MonetaryConstraints",
    "Promotion",
    "PromotionType",
    "SourceProvenance",
    "TemporalValidity",
    "VerificationState",
    "WageringRequirement",
]
