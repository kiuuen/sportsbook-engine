"""Canonical entity model layer for odds normalization.

Defines the data structures (events → markets → outcomes with decimal odds)
extracted from the legacy normalizer for reuse across adapters and consumers.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

class CanonicalMarketType:
    """Namespace of canonical market group identifiers."""

    MONEYLINE = "moneyline"
    TOTALS = "totals"
    SPREADS = "spreads"
    UNKNOWN = "unknown"

    ALL: Tuple[str, ...] = (MONEYLINE, TOTALS, SPREADS, UNKNOWN)

@dataclass
class CanonicalOutcome:
    """A single normalized selection with decimal odds."""

    label: str
    odds_decimal: Optional[float]
    odds_raw: Any = None
    odds_format: str = "unknown"
    line: Optional[float] = None
    provider_outcome_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class CanonicalMarket:
    """A normalized market belonging to a canonical group."""

    canonical_type: str
    raw_label: str
    market_id: Optional[str] = None
    classification_confidence: float = 0.0
    outcomes: List[CanonicalOutcome] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data

@dataclass
class CanonicalEvent:
    """A normalized event with its normalized markets."""

    event_id: Optional[str]
    home_team: Optional[str]
    away_team: Optional[str]
    start_time: Optional[str]
    raw_label: Optional[str]
    source_provider: str = "unknown"
    league: Optional[str] = None
    sport: Optional[str] = None
    markets: List[CanonicalMarket] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
