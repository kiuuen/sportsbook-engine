"""Canonical data structures for the heuristic classification pipeline.

This module isolates the classification result contract so that scoring
logic, exporters, and pipeline consumers can share a single, dependency-
light definition without importing the legacy monolith.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from cdp_interceptor import NetworkEvent


@dataclass
class ClassificationResult:
    """Outcome of scoring a single captured network endpoint.

    Aggregates the lexical/structural evidence extracted from a response
    payload together with the deduced route template, role, volatility tier,
    and learned field mappings for downstream persistence and replay.
    """

    url: str
    url_template: str
    score: float
    category_scores: Dict[str, float]
    odds_count: int
    matched_keywords: Set[str] = field(default_factory=set)
    core_betting_matches: int = 0
    is_grey_zone: bool = False
    ambiguous_keys: List[str] = field(default_factory=list)
    call_count: int = 1
    time_span_seconds: float = 0.0
    sample_event: Optional[NetworkEvent] = None
    is_viable: bool = False
    role: str = "catalog_root"
    volatility: str = "static"
    produced_params: List[str] = field(default_factory=list)
    is_auxiliary_widget: bool = False
    field_mappings: Dict[str, str] = field(default_factory=dict)
