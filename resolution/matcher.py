"""Deterministic canonical event ID generation for entity resolution.

Combines order-independent team names, the event calendar date, and an
optional sport discriminator into a stable UUIDv5 identifier so the same
real-world event resolves to the same ID across heterogeneous feeds.
"""

import datetime
import uuid
from typing import Optional

from resolution.cleaner import normalize_team_name

# Fixed, application-owned namespace (RFC 4122 "random" reserved slot).
# This constant must never change, or previously issued canonical IDs
# would be invalidated.
_CANONICAL_EVENT_NAMESPACE = uuid.UUID("6f1d3b4a-9c2e-4f7d-8a51-3b0c9d2e7f60")

_DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y", "%d-%m-%Y")


def _extract_date_component(start_time: Optional[str]) -> str:
    """Extract the ``YYYY-MM-DD`` calendar date from a raw timestamp.

    Tries ISO-8601 first (covers most feed formats including trailing
    time/zone components), then a small set of common legacy layouts.

    Args:
        start_time: Raw start timestamp string, e.g. ``2024-05-01T18:30:00Z``.

    Returns:
        The date as ``YYYY-MM-DD``, or an empty string when unparseable/absent.
    """
    if start_time is None:
        return ""

    text = str(start_time).strip()
    if not text:
        return ""

    # ISO-8601 handles the majority of feed formats, including offsets.
    try:
        parsed = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.date().isoformat()
    except ValueError:
        pass

    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.datetime.strptime(text, fmt)
            return parsed.date().isoformat()
        except ValueError:
            continue

    return ""


def generate_canonical_event_id(
    home_team: Optional[str],
    away_team: Optional[str],
    start_time: Optional[str],
    sport: Optional[str] = None,
) -> str:
    """Generate a deterministic canonical event ID.

    Team names are normalized and sorted alphabetically so that
    discrepancies in home/away assignment between bookmakers do not
    produce different IDs. The sorted names, the calendar date, and the
    optional sport are combined into a UUIDv5 identifier.

    Args:
        home_team: Raw home team name from the source feed.
        away_team: Raw away team name from the source feed.
        start_time: Raw start timestamp; only the calendar date is used.
        sport: Optional sport discriminator to prevent cross-sport collisions.

    Returns:
        A deterministic UUIDv5 string.

    Raises:
        ValueError: If both team names normalize to empty strings, since
            an event without any team identity cannot be resolved safely.
    """
    normalized_home = normalize_team_name(home_team)
    normalized_away = normalize_team_name(away_team)

    if not normalized_home and not normalized_away:
        raise ValueError(
            "Cannot generate canonical event ID: both team names are empty "
            "after normalization."
        )

    sorted_teams = sorted([normalized_home, normalized_away])
    date_component = _extract_date_component(start_time)
    sport_component = str(sport).strip().lower() if sport is not None else ""

    identity_parts = [sorted_teams[0], sorted_teams[1], date_component]
    if sport_component:
        identity_parts.append(sport_component)

    identity_source = "|".join(identity_parts)
    return str(uuid.uuid5(_CANONICAL_EVENT_NAMESPACE, identity_source))
