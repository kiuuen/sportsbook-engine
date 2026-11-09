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
from bonus_engine.extractor import ExtractionConfig, PromotionExtractor
from bonus_engine.validator import (
    DiscrepancyCode,
    InvariantValidator,
    ValidationReport,
)

__all__ = [
    "BonusDatabaseManager",
    "CalculationBase",
    "DiscrepancyCode",
    "EvidenceCitation",
    "ExtractionConfig",
    "InvariantValidator",
    "MonetaryConstraints",
    "PromoLexiconLoader",
    "Promotion",
    "PromotionExtractor",
    "PromotionScanner",
    "PromotionType",
    "ScanEvaluation",
    "SourceProvenance",
    "TemporalValidity",
    "ValidationReport",
    "VerificationState",
    "WageringRequirement",
]
