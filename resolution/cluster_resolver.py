"""In-memory active entity resolution for provider fixtures.

Maintains a bounded catalog of active canonical event clusters and resolves
incoming provider fixtures to cluster identifiers. Previously resolved
fixtures are served from an O(1) cache; new fixtures are compared only
against a date/sport-partitioned candidate set using the deterministic
scoring logic from :mod:`resolution.matcher`, never against the full
catalog. Eviction purges concluded or expired clusters together with all
of their lookup entries so memory usage stays bounded.
"""

import datetime
import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from resolution.cleaner import normalize_team_name
from resolution.matcher import (
    compare_events,
    generate_canonical_event_id,
    parse_start_time_utc,
)

_UNKNOWN_BUCKET = ""

# Bucket key type: (calendar date "YYYY-MM-DD" or "" when unknown, sport or "").
_BucketKey = Tuple[str, str]


@dataclass
class EventCluster:
    """An active canonical event cluster tracked by the resolver."""

    cluster_id: str
    home_team: str
    away_team: str
    start_time: Optional[str]
    sport: Optional[str]
    created_at: datetime.datetime
    last_seen_at: datetime.datetime
    fixture_keys: Set[str] = field(default_factory=set)


class EventClusterResolver:
    """Resolves provider fixtures to canonical cluster identifiers.

    Args:
        name_match_threshold: Minimum aligned name-token similarity required
            to accept a candidate cluster (mirrors the matcher's scale).
        temporal_match_window_hours: Configurable mirror of the matcher's
            full-proximity window; used for documentation and index probing.
        temporal_max_window_hours: Maximum temporal separation still eligible
            for matching; drives the date-bucket probe radius.
        cluster_ttl_hours: Clusters not touched within this window, or whose
            start time lies further in the past, are purged by
            :meth:`evict_expired`.
        max_active_clusters: Hard memory bound; the least-recently-seen
            clusters are evicted when the bound would be exceeded.
        max_candidates_per_resolution: Upper bound on comparisons performed
            for a single resolution, including the no-date fallback scan.

    Raises:
        ValueError: If any constructor argument is outside its valid range.
    """

    def __init__(
        self,
        name_match_threshold: float = 0.5,
        temporal_match_window_hours: float = 6.0,
        temporal_max_window_hours: float = 48.0,
        cluster_ttl_hours: float = 72.0,
        max_active_clusters: int = 10000,
        max_candidates_per_resolution: int = 64,
    ) -> None:
        if not 0.0 <= name_match_threshold <= 1.0:
            raise ValueError("name_match_threshold must be within [0, 1].")
        if temporal_match_window_hours <= 0.0:
            raise ValueError("temporal_match_window_hours must be positive.")
        if temporal_max_window_hours < temporal_match_window_hours:
            raise ValueError(
                "temporal_max_window_hours must be >= temporal_match_window_hours."
            )
        if cluster_ttl_hours <= 0.0:
            raise ValueError("cluster_ttl_hours must be positive.")
        if max_active_clusters < 1:
            raise ValueError("max_active_clusters must be at least 1.")
        if max_candidates_per_resolution < 1:
            raise ValueError("max_candidates_per_resolution must be at least 1.")

        self._name_match_threshold = float(name_match_threshold)
        self._temporal_match_window_hours = float(temporal_match_window_hours)
        self._temporal_max_window_hours = float(temporal_max_window_hours)
        self._cluster_ttl_seconds = float(cluster_ttl_hours) * 3600.0
        self._max_active_clusters = int(max_active_clusters)
        self._max_candidates = int(max_candidates_per_resolution)

        self._clusters: Dict[str, EventCluster] = {}
        self._fixture_to_cluster: Dict[str, str] = {}
        self._index: Dict[_BucketKey, Set[str]] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def resolve_fixture(
        self,
        provider: Optional[str],
        fixture_id: Optional[str],
        home_team: Optional[str],
        away_team: Optional[str],
        start_time: Optional[str],
        sport: Optional[str] = None,
        now: Optional[datetime.datetime] = None,
    ) -> str:
        """Resolve a provider fixture to a canonical cluster identifier.

        Previously resolved fixture identities return their cluster in O(1).
        Otherwise the fixture is compared against indexed candidate clusters
        only; a satisfying candidate wins, else a new cluster is created.

        Args:
            provider: Provider identifier (used for the fast-lookup key).
            fixture_id: Provider-specific fixture identifier, if available.
            home_team: Raw home team name from the feed.
            away_team: Raw away team name from the feed.
            start_time: Raw start timestamp from the feed.
            sport: Optional sport discriminator.
            now: Optional clock override (UTC-aware); defaults to current time.

        Returns:
            The canonical cluster identifier.

        Raises:
            ValueError: If both team names are empty after normalization.
        """
        if not (normalize_or_empty(home_team) or normalize_or_empty(away_team)):
            raise ValueError(
                "Cannot resolve fixture: both team names are empty after "
                "normalization."
            )

        current_time = now if now is not None else _utc_now()
        fixture_key = self._fixture_key(
            provider, fixture_id, home_team, away_team, start_time, sport
        )

        cached_cluster_id = self._fixture_to_cluster.get(fixture_key)
        if cached_cluster_id is not None:
            cluster = self._clusters.get(cached_cluster_id)
            if cluster is not None:
                cluster.last_seen_at = current_time
                return cached_cluster_id
            # Stale pointer to an evicted cluster; fall through to re-resolution.
            self._fixture_to_cluster.pop(fixture_key, None)

        matched_cluster_id = self._find_matching_cluster(
            home_team, away_team, start_time, sport
        )
        if matched_cluster_id is None:
            matched_cluster_id = self._create_or_get_cluster(
                home_team, away_team, start_time, sport, current_time
            )

        self._attach(matched_cluster_id, fixture_key, current_time)
        return matched_cluster_id

    def conclude_cluster(self, cluster_id: str) -> bool:
        """Purge a concluded cluster and all of its lookup entries.

        Returns:
            ``True`` when the cluster existed and was removed.
        """
        return self._detach(cluster_id)

    def evict_expired(
        self, now: Optional[datetime.datetime] = None
    ) -> int:
        """Purge expired clusters and their lookup entries.

        A cluster is expired when it has not been touched within the TTL or
        when its parsed start time lies further in the past than the TTL.

        Returns:
            The number of clusters evicted.
        """
        current_time = now if now is not None else _utc_now()
        expired_ids = [
            cluster_id
            for cluster_id, cluster in self._clusters.items()
            if self._is_expired(cluster, current_time)
        ]
        for cluster_id in expired_ids:
            self._detach(cluster_id)
        return len(expired_ids)

    def get_cluster(self, cluster_id: str) -> Optional[EventCluster]:
        """Return a defensive snapshot copy of a cluster, if present."""
        cluster = self._clusters.get(cluster_id)
        if cluster is None:
            return None
        return EventCluster(
            cluster_id=cluster.cluster_id,
            home_team=cluster.home_team,
            away_team=cluster.away_team,
            start_time=cluster.start_time,
            sport=cluster.sport,
            created_at=cluster.created_at,
            last_seen_at=cluster.last_seen_at,
            fixture_keys=set(cluster.fixture_keys),
        )

    def fixtures_for_cluster(self, cluster_id: str) -> Tuple[str, ...]:
        """Return the provider fixture keys currently bound to a cluster."""
        cluster = self._clusters.get(cluster_id)
        return tuple(sorted(cluster.fixture_keys)) if cluster else ()

    @property
    def cluster_count(self) -> int:
        """Number of currently active clusters."""
        return len(self._clusters)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _find_matching_cluster(
        self,
        home_team: Optional[str],
        away_team: Optional[str],
        start_time: Optional[str],
        sport: Optional[str],
    ) -> Optional[str]:
        """Return the best-scoring satisfying candidate cluster, if any."""
        date_bucket = self._date_bucket(start_time)
        sport_bucket = self._sport_bucket(sport)

        best_cluster_id: Optional[str] = None
        best_confidence = -1.0
        comparisons = 0

        for candidate_id in self._candidate_ids(date_bucket, sport_bucket):
            if comparisons >= self._max_candidates:
                break
            cluster = self._clusters.get(candidate_id)
            if cluster is None:
                continue
            comparisons += 1
            decision = compare_events(
                home_team,
                away_team,
                start_time,
                cluster.home_team,
                cluster.away_team,
                cluster.start_time,
                sport,
                cluster.sport,
            )
            if (
                not decision.is_match
                or decision.category_conflict
                or decision.name_similarity < self._name_match_threshold
                or (
                    decision.temporal_score is not None
                    and decision.temporal_score <= 0.0
                )
            ):
                continue
            if decision.confidence > best_confidence:
                best_confidence = decision.confidence
                best_cluster_id = candidate_id

        return best_cluster_id

    def _candidate_ids(
        self, date_bucket: str, sport_bucket: str
    ) -> List[str]:
        """Collect candidate cluster ids from relevant index buckets only.

        Probes the fixture's sport bucket plus the unknown-sport bucket for
        each calendar date within the temporal probe radius. When the date
        is unknown, falls back to a capped scan over all buckets.
        """
        candidates: List[str] = []
        seen: Set[str] = set()

        if not date_bucket:
            for bucket_ids in self._index.values():
                for cluster_id in bucket_ids:
                    if cluster_id not in seen:
                        seen.add(cluster_id)
                        candidates.append(cluster_id)
                        if len(candidates) >= self._max_candidates:
                            return candidates
            return candidates

        probe_radius_days = int(
            math.ceil(self._temporal_max_window_hours / 24.0)
        )
        base_date = datetime.date.fromisoformat(date_bucket)
        sport_buckets = (
            (sport_bucket, _UNKNOWN_BUCKET)
            if sport_bucket
            else self._sport_buckets_for_date(date_bucket)
        )
        for day_offset in range(-probe_radius_days, probe_radius_days + 1):
            probed_date = base_date + datetime.timedelta(days=day_offset)
            for bucket_sport in sport_buckets:
                bucket_ids = self._index.get(
                    (probed_date.isoformat(), bucket_sport)
                )
                if not bucket_ids:
                    continue
                for cluster_id in bucket_ids:
                    if cluster_id not in seen:
                        seen.add(cluster_id)
                        candidates.append(cluster_id)
        return candidates

    def _sport_buckets_for_date(self, date_bucket: str) -> Tuple[str, ...]:
        """Sport buckets present in the index for a given calendar date."""
        return tuple(
            sport
            for (bucket_date, sport) in self._index
            if bucket_date == date_bucket
        )

    def _create_or_get_cluster(
        self,
        home_team: Optional[str],
        away_team: Optional[str],
        start_time: Optional[str],
        sport: Optional[str],
        current_time: datetime.datetime,
    ) -> str:
        """Create a new cluster (enforcing the memory bound) or return the
        existing cluster sharing the same deterministic canonical ID."""
        cluster_id = generate_canonical_event_id(
            home_team, away_team, start_time, sport
        )
        if cluster_id in self._clusters:
            # Same canonical identity (identical normalized teams, date and
            # sport); the candidate comparison above already vetted it.
            return cluster_id

        self._enforce_capacity()
        cluster = EventCluster(
            cluster_id=cluster_id,
            home_team=str(home_team) if home_team is not None else "",
            away_team=str(away_team) if away_team is not None else "",
            start_time=start_time,
            sport=sport,
            created_at=current_time,
            last_seen_at=current_time,
        )
        self._clusters[cluster_id] = cluster
        bucket = (self._date_bucket(start_time), self._sport_bucket(sport))
        self._index.setdefault(bucket, set()).add(cluster_id)
        return cluster_id

    def _enforce_capacity(self) -> None:
        """Evict least-recently-seen clusters until below the memory bound."""
        while len(self._clusters) >= self._max_active_clusters:
            oldest_id = min(
                self._clusters,
                key=lambda cid: self._clusters[cid].last_seen_at,
            )
            self._detach(oldest_id)

    def _attach(
        self,
        cluster_id: str,
        fixture_key: str,
        current_time: datetime.datetime,
    ) -> None:
        """Bind a fixture key to a cluster bidirectionally."""
        cluster = self._clusters.get(cluster_id)
        if cluster is None:
            raise KeyError(f"Unknown cluster id: {cluster_id}")
        cluster.fixture_keys.add(fixture_key)
        cluster.last_seen_at = current_time
        self._fixture_to_cluster[fixture_key] = cluster_id

    def _detach(self, cluster_id: str) -> bool:
        """Remove a cluster, its index entries, and its fixture bindings."""
        cluster = self._clusters.pop(cluster_id, None)
        if cluster is None:
            return False
        for fixture_key in cluster.fixture_keys:
            if self._fixture_to_cluster.get(fixture_key) == cluster_id:
                self._fixture_to_cluster.pop(fixture_key, None)
        bucket = (self._date_bucket(cluster.start_time), self._sport_bucket(cluster.sport))
        bucket_ids = self._index.get(bucket)
        if bucket_ids is not None:
            bucket_ids.discard(cluster_id)
            if not bucket_ids:
                self._index.pop(bucket, None)
        return True

    def _is_expired(
        self, cluster: EventCluster, current_time: datetime.datetime
    ) -> bool:
        """Decide whether a cluster has exceeded its TTL."""
        if (current_time - cluster.last_seen_at).total_seconds() > self._cluster_ttl_seconds:
            return True
        parsed_start = parse_start_time_utc(cluster.start_time)
        return (
            parsed_start is not None
            and (current_time - parsed_start).total_seconds() > self._cluster_ttl_seconds
        )

    @staticmethod
    def _fixture_key(
        provider: Optional[str],
        fixture_id: Optional[str],
        home_team: Optional[str],
        away_team: Optional[str],
        start_time: Optional[str],
        sport: Optional[str],
    ) -> str:
        """Build the O(1) lookup key for a provider fixture.

        Uses ``provider::fixture_id`` when both are available; otherwise
        derives a deterministic content hash from the raw fields.
        """
        provider_text = str(provider).strip() if provider is not None else ""
        fixture_text = str(fixture_id).strip() if fixture_id is not None else ""
        if provider_text and fixture_text:
            return f"{provider_text}::{fixture_text}"

        content = "|".join(
            "" if value is None else str(value)
            for value in (provider, fixture_id, home_team, away_team, start_time, sport)
        )
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return f"content::{digest}"

    @staticmethod
    def _sport_bucket(sport: Optional[str]) -> str:
        """Normalized sport bucket key (empty string when unknown)."""
        return str(sport).strip().lower() if sport is not None else ""

    @staticmethod
    def _date_bucket(start_time: Optional[str]) -> str:
        """Calendar-date bucket key (empty string when unparseable)."""
        parsed = parse_start_time_utc(start_time)
        return parsed.date().isoformat() if parsed else _UNKNOWN_BUCKET


def normalize_or_empty(name: Optional[str]) -> str:
    """Normalize a team name, degrading to an empty string on failure."""
    try:
        return normalize_team_name(name)
    except (TypeError, ValueError):
        return ""


def _utc_now() -> datetime.datetime:
    """Current aware UTC timestamp."""
    return datetime.datetime.now(datetime.timezone.utc)
