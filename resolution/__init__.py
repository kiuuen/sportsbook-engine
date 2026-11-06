"""Entity resolution package for canonical event identity across feeds.

Public API:
    normalize_team_name: Canonical normalization of raw team names.
    extract_category_markers: Extract semantic category markers from a team name.
    generate_canonical_event_id: Deterministic cross-feed event IDs.
    compare_events: Score and decide whether two events match.
    MatchDecision: Outcome of a cross-feed event comparison.
"""

from resolution.cleaner import extract_category_markers, normalize_team_name
from resolution.matcher import (
    MatchDecision,
    compare_events,
    generate_canonical_event_id,
)

__all__ = [
    "normalize_team_name",
    "extract_category_markers",
    "generate_canonical_event_id",
    "compare_events",
    "MatchDecision",
]
