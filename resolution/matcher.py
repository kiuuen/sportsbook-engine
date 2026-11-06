"""Deterministic canonical event ID generation for entity resolution.

Combines order-independent team names, the event calendar date, and an
optional sport discriminator into a stable UUIDv5 identifier so the same
real-world event resolves to the same ID across heterogeneous feeds.
"""

import datetime
import uuid
from dataclasses import dataclass
from typing import Optional, Tuple
from difflib import SequenceMatcher

from resolution.cleaner import extract_category_markers, normalize_team_name

# Fixed, application-owned namespace (RFC 4122 "random" reserved slot).
# This constant must never change, or previously issued canonical IDs
# would be invalidated.
_CANONICAL_EVENT_NAMESPACE = uuid.UUID("6f1d3b4a-9c2e-4f7d-8a51-3b0c9d2e7f60")

_DATE_FORMATS = ("%Y-%m-%d", "%Y/%m-%d", "%d.%m.%Y", "%d-%m-%Y")

# Temporal matching window: deltas within the match window score 1.0,
# scores decay linearly to 0.0 at the max window boundary.
_TEMPORAL_MATCH_WINDOW_HOURS = 6.0
_TEMPORAL_MAX_WINDOW_HOURS = 48.0

# Minimum aligned name-token similarity for a match decision.
_NAME_MATCH_THRESHOLD = 0.5

# Minimum per-token fuzzy similarity (or prefix relationship) for two
# tokens to be considered the same word.
_TOKEN_FUZZY_THRESHOLD = 0.8

def parse_start_time_utc(start_time: Optional[str]) -> Optional[datetime.datetime]:
    """Parse a raw start timestamp into an aware UTC datetime.

    Accepts ISO-8601 (with ``Z`` or numeric offsets), a small set of
    common legacy date layouts (interpreted as UTC), and numeric epoch
    values (seconds or milliseconds). Naive datetimes are assumed UTC.

    Returns ``None`` when the value is absent or unparseable.
    """
    if start_time is None:
        return None

    if isinstance(start_time, (int, float)) and not isinstance(start_time, bool):
        timestamp = float(start_time)
        if timestamp > 1e11:  # heuristic: milliseconds
            timestamp /= 1000.0
        try:
            return datetime.datetime.fromtimestamp(
                timestamp, tz=datetime.timezone.utc
            )
        except (OverflowError, OSError, ValueError):
            return None

    text = str(start_time).strip()
    if not text:
        return None

    parsed = None
    try:
        parsed = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        for fmt in _DATE_FORMATS:
            try:
                parsed = datetime.datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
    if parsed is None:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.astimezone(datetime.timezone.utc)


def _extract_date_component(start_time: Optional[str]) -> str:
    """Extract the ``YYYY-MM-DD`` calendar date from a raw timestamp.

    Tries ISO-8601 first (covers most feed formats including trailing
    time/zone components), then a small set of common legacy layouts.

    Args:
        start_time: Raw start timestamp string, e.g. ``2024-05-01T18:30:00Z``.

    Returns:
        The date as ``YYYY-MM-DD``, or an empty string when unparseable/absent.
    """
    parsed = parse_start_time_utc(start_time)
    return parsed.date().isoformat() if parsed else ""


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


@dataclass(frozen=True)
class MatchDecision:
    """Outcome of a cross-feed event comparison."""

    is_match: bool
    confidence: float
    name_similarity: float
    temporal_score: Optional[float]  # None when temporal proximity is unknown
    category_conflict: bool
    reason: str

def _tokens_match(a: str, b: str) -> bool:
    """Decide whether two tokens represent the same word.

    Accepts exact equality, a shared prefix of at least three characters
    (covers truncated forms such as "man" / "manchester"), and minor
    spelling differences via SequenceMatcher ratio.
    """
    if a == b:
        return True
    if len(a) >= 3 and len(b) >= 3 and (a.startswith(b) or b.startswith(a)):
        return True
    return SequenceMatcher(None, a, b).ratio() >= _TOKEN_FUZZY_THRESHOLD


def _token_similarity(a: str, b: str) -> float:
    """Fuzzy Jaccard similarity between the tokens of two normalized names.

    Tokens are paired greedily one-to-one using :func:`_tokens_match`,
    so partial matches, shared roots, and minor misspellings contribute
    similarity instead of collapsing the score to zero.
    """
    if not a or not b:
        return 0.0
    tokens_a, tokens_b = a.split(" "), b.split(" ")
    if not tokens_a or not tokens_b:
        return 0.0

    matched = 0
    used_b: set = set()
    for token_a in tokens_a:
        for index, token_b in enumerate(tokens_b):
            if index in used_b:
                continue
            if _tokens_match(token_a, token_b):
                used_b.add(index)
                matched += 1
                break
    return matched / (len(tokens_a) + len(tokens_b) - matched)

def _name_alignment_score(
    home_a: str, away_a: str, home_b: str, away_b: str
) -> float:
    """Best average token similarity over direct and swapped pairings.

    Bookmakers disagree on home/away assignment, so both alignments are
    evaluated and the better one wins.
    """
    direct = 0.5 * (
        _token_similarity(home_a, home_b) + _token_similarity(away_a, away_b)
    )
    swapped = 0.5 * (
        _token_similarity(home_a, away_b) + _token_similarity(away_a, home_b)
    )
    return max(direct, swapped)


def _temporal_score(
    start_a: Optional[str], start_b: Optional[str]
) -> Optional[float]:
    """Score temporal proximity between two raw timestamps in [0, 1].

    Returns ``None`` when either timestamp is absent/unparseable, so the
    caller can degrade gracefully instead of rejecting the candidate.
    """
    parsed_a = parse_start_time_utc(start_a)
    parsed_b = parse_start_time_utc(start_b)
    if parsed_a is None or parsed_b is None:
        return None

    delta_hours = abs((parsed_a - parsed_b).total_seconds()) / 3600.0
    if delta_hours <= _TEMPORAL_MATCH_WINDOW_HOURS:
        return 1.0
    if delta_hours >= _TEMPORAL_MAX_WINDOW_HOURS:
        return 0.0
    span = _TEMPORAL_MAX_WINDOW_HOURS - _TEMPORAL_MATCH_WINDOW_HOURS
    return 1.0 - (delta_hours - _TEMPORAL_MATCH_WINDOW_HOURS) / span


def compare_events(
    home_a: Optional[str],
    away_a: Optional[str],
    start_a: Optional[str],
    home_b: Optional[str],
    away_b: Optional[str],
    start_b: Optional[str],
    sport_a: Optional[str] = None,
    sport_b: Optional[str] = None,
) -> MatchDecision:
    """Score two feed-specific event records and decide whether they match.

    Decision logic, in order:
      1. Both records need at least one usable team name.
      2. Conflicting semantic category markers (youth/reserve/gender)
         veto the match outright, regardless of similarity.
      3. Known, conflicting sports veto the match.
      4. Otherwise the decision combines aligned name-token similarity
         with temporal proximity; unknown timestamps reduce confidence
         but do not veto.
    """
    norm_home_a, norm_away_a = normalize_team_name(home_a), normalize_team_name(away_a)
    norm_home_b, norm_away_b = normalize_team_name(home_b), normalize_team_name(away_b)

    if not (norm_home_a or norm_away_a) or not (norm_home_b or norm_away_b):
        return MatchDecision(
            is_match=False,
            confidence=0.0,
            name_similarity=0.0,
            temporal_score=None,
            category_conflict=False,
            reason="insufficient team identity on at least one side",
        )

    markers_a = extract_category_markers(home_a) | extract_category_markers(away_a)
    markers_b = extract_category_markers(home_b) | extract_category_markers(away_b)
    similarity = _name_alignment_score(
        norm_home_a, norm_away_a, norm_home_b, norm_away_b
    )
    temporal = _temporal_score(start_a, start_b)

    if markers_a != markers_b:
        return MatchDecision(
            is_match=False,
            confidence=0.0,
            name_similarity=similarity,
            temporal_score=temporal,
            category_conflict=True,
            reason=f"category discriminator conflict: {sorted(markers_a)} vs {sorted(markers_b)}",
        )

    sport_a_norm = str(sport_a).strip().lower() if sport_a else ""
    sport_b_norm = str(sport_b).strip().lower() if sport_b else ""
    if sport_a_norm and sport_b_norm and sport_a_norm != sport_b_norm:
        return MatchDecision(
            is_match=False,
            confidence=0.0,
            name_similarity=similarity,
            temporal_score=temporal,
            category_conflict=False,
            reason=f"sport mismatch: {sport_a_norm} vs {sport_b_norm}",
        )

    if temporal is None:
        confidence = similarity * 0.85
        is_match = similarity >= _NAME_MATCH_THRESHOLD
        reason = "temporal proximity unknown; decided on name similarity alone"
    else:
        confidence = 0.6 * similarity + 0.4 * temporal
        is_match = similarity >= _NAME_MATCH_THRESHOLD and temporal > 0.0
        reason = "name similarity and temporal proximity combined"

    return MatchDecision(
        is_match=is_match,
        confidence=round(confidence, 4),
        name_similarity=round(similarity, 4),
        temporal_score=None if temporal is None else round(temporal, 4),
        category_conflict=False,
        reason=reason,
    )
