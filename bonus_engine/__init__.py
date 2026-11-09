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
from bonus_engine.scanner import (
    PromoLexiconLoader,
    PromotionScanner,
    ScanEvaluation,
)

__all__ = [
    "BonusDatabaseManager",
    "CalculationBase",
    "EvidenceCitation",
    "MonetaryConstraints",
    "PromoLexiconLoader",
    "Promotion",
    "PromotionScanner",
    "PromotionType",
    "ScanEvaluation",
    "SourceProvenance",
    "TemporalValidity",
    "VerificationState",
    "WageringRequirement",
]
