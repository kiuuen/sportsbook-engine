"""Entity resolution package for canonical event identity across feeds.

Public API:
    normalize_team_name: Canonical normalization of raw team names.
    generate_canonical_event_id: Deterministic cross-feed event IDs.
"""

from resolution.cleaner import normalize_team_name
from resolution.matcher import generate_canonical_event_id

__all__ = ["normalize_team_name", "generate_canonical_event_id"]
