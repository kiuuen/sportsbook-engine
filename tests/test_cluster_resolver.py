"""Unit tests for :class:`resolution.cluster_resolver.EventClusterResolver`.

Covers multi-provider convergence, the O(1) identity fast path, sport/date
partition scoping, category qualifier isolation, eviction with reverse-index
pruning, and constructor parameter configurability. All time-dependent
behavior is driven through the injectable ``now`` parameter for full
determinism; no synthetic JSON payloads are used.
"""

import datetime
from typing import Any, Callable, List, Tuple

import pytest

from resolution import cluster_resolver
from resolution.cluster_resolver import EventClusterResolver

_UTC = datetime.timezone.utc

_KICKOFF = "2024-05-01T18:30:00Z"
_BEFORE_KICKOFF = datetime.datetime(2024, 5, 1, 12, 0, tzinfo=_UTC)
_AFTER_KICKOFF_PLUS_GRACE = datetime.datetime(2024, 5, 2, 1, 0, tzinfo=_UTC)


def _resolve(
    resolver: EventClusterResolver,
    provider_domain: str,
    provider_event_id: str,
    home_team: str,
    away_team: str,
    start_time: str,
    sport: str,
    now: datetime.datetime,
) -> str:
    """Call :meth:`EventClusterResolver.resolve` with explicit arguments."""
    return resolver.resolve(
        provider_domain,
        provider_event_id,
        home_team,
        away_team,
        start_time,
        sport,
        now=now,
    )


class TestMultiProviderConvergence:
    """Providers reporting the same fixture must converge on one cluster."""

    def test_lexical_variations_converge_to_single_cluster(self) -> None:
        resolver = EventClusterResolver()
        cluster_a = _resolve(
            resolver, "provider_a", "evt-1",
            "Wolves", "Arsenal", _KICKOFF, "football", _BEFORE_KICKOFF,
        )
        cluster_b = _resolve(
            resolver, "provider_b", "998877",
            "Wolverhampton", "Arsenal FC", _KICKOFF, "football", _BEFORE_KICKOFF,
        )

        assert cluster_a == cluster_b
        assert resolver.cluster_count == 1
        assert set(
            resolver.provider_identities_for_cluster(cluster_a)
        ) == {("provider_a", "evt-1"), ("provider_b", "998877")}

    def test_converged_cluster_snapshot_holds_both_identities(self) -> None:
        resolver = EventClusterResolver()
        cluster_id = _resolve(
            resolver, "feed_one", "A-1",
            "Inter Milan", "Napoli", _KICKOFF, "football", _BEFORE_KICKOFF,
        )
        _resolve(
            resolver, "feed_two", "B-2",
            "Internazionale", "Napoli", _KICKOFF, "football", _BEFORE_KICKOFF,
        )

        snapshot = resolver.get_cluster(cluster_id)
        assert snapshot is not None
        assert snapshot.provider_identities == {
            ("feed_one", "A-1"),
            ("feed_two", "B-2"),
        }


class TestCachingFastPath:
    """Known provider identities must short-circuit candidate comparison."""

    def test_repeat_identity_skips_candidate_comparisons(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_compare: Callable[..., Any] = cluster_resolver.compare_events
        comparison_args: List[Tuple[Any, ...]] = []

        def counting_compare(*args: Any, **kwargs: Any) -> Any:
            comparison_args.append(args)
            return real_compare(*args, **kwargs)

        monkeypatch.setattr(cluster_resolver, "compare_events", counting_compare)

        resolver = EventClusterResolver()
        first = _resolve(
            resolver, "p", "e1",
            "Bayern Munich", "Dortmund", _KICKOFF, "football", _BEFORE_KICKOFF,
        )
        assert len(comparison_args) == 1

        second = _resolve(
            resolver, "p", "e1",
            "Bayern Munich", "Dortmund", _KICKOFF, "football", _BEFORE_KICKOFF,
        )

        assert second == first
        assert len(comparison_args) == 1

    def test_fast_path_refreshes_last_seen_timestamp(self) -> None:
        resolver = EventClusterResolver()
        cluster_id = _resolve(
            resolver, "p", "e1",
            "Bayern Munich", "Dortmund", _KICKOFF, "football", _BEFORE_KICKOFF,
        )
        later = _BEFORE_KICKOFF + datetime.timedelta(hours=2)
        _resolve(
            resolver, "p", "e1",
            "Bayern Munich", "Dortmund", _KICKOFF, "football", later,
        )

        snapshot = resolver.get_cluster(cluster_id)
        assert snapshot is not None
        assert snapshot.last_seen_at == later


class TestPartitionScoping:
    """Different sports or far-apart dates must never merge clusters."""

    def test_different_sports_form_separate_clusters(self) -> None:
        resolver = EventClusterResolver()
        cluster_football = _resolve(
            resolver, "p", "f1",
            "Alpha FC", "Beta United", _KICKOFF, "football", _BEFORE_KICKOFF,
        )
        cluster_basketball = _resolve(
            resolver, "p", "b1",
            "Alpha FC", "Beta United", _KICKOFF, "basketball", _BEFORE_KICKOFF,
        )

        assert cluster_football != cluster_basketball
        assert resolver.cluster_count == 2

    def test_dates_beyond_temporal_window_form_separate_clusters(self) -> None:
        resolver = EventClusterResolver()
        cluster_day_one = _resolve(
            resolver, "p", "d1",
            "Alpha FC", "Beta United", _KICKOFF, "football", _BEFORE_KICKOFF,
        )
        far_later_kickoff = "2024-05-04T18:30:00Z"  # 72h later: beyond 48h max.
        cluster_day_four = _resolve(
            resolver, "q", "d2",
            "Alpha FC", "Beta United", far_later_kickoff, "football",
            _BEFORE_KICKOFF,
        )

        assert cluster_day_one != cluster_day_four
        assert resolver.cluster_count == 2


class TestCategoryQualifierIsolation:
    """Conflicting youth/gender qualifiers must veto cluster merging."""

    def test_u21_fixture_isolated_from_main_team(self) -> None:
        resolver = EventClusterResolver()
        main_cluster = _resolve(
            resolver, "p", "m1",
            "Brazil", "Chile", _KICKOFF, "football", _BEFORE_KICKOFF,
        )
        youth_cluster = _resolve(
            resolver, "q", "y1",
            "Brazil U21", "Chile U21", _KICKOFF, "football", _BEFORE_KICKOFF,
        )

        assert main_cluster != youth_cluster
        assert resolver.cluster_count == 2

    def test_women_fixture_isolated_from_mens_team(self) -> None:
        resolver = EventClusterResolver()
        mens_cluster = _resolve(
            resolver, "p", "m1",
            "Arsenal", "Chelsea", _KICKOFF, "football", _BEFORE_KICKOFF,
        )
        womens_cluster = _resolve(
            resolver, "q", "w1",
            "Arsenal Women", "Chelsea Women", _KICKOFF, "football",
            _BEFORE_KICKOFF,
        )

        assert mens_cluster != womens_cluster
        assert resolver.cluster_count == 2


class TestEvictionAndReverseIndexPruning:
    """Eviction must purge expired clusters and their reverse lookups."""

    def test_evict_purges_expired_cluster_and_reverse_entries(self) -> None:
        resolver = EventClusterResolver(retention_grace_hours=6.0)
        cluster_id = _resolve(
            resolver, "p", "e1",
            "Norwich", "Leeds", _KICKOFF, "football", _BEFORE_KICKOFF,
        )
        assert resolver.get_cluster(cluster_id) is not None

        evicted = resolver.evict(now=_AFTER_KICKOFF_PLUS_GRACE)

        assert evicted == 1
        assert resolver.cluster_count == 0
        assert resolver.get_cluster(cluster_id) is None
        assert resolver.provider_identities_for_cluster(cluster_id) == ()

    def test_purged_identity_re_resolves_to_fresh_cluster(self) -> None:
        resolver = EventClusterResolver(retention_grace_hours=6.0)
        original = _resolve(
            resolver, "p", "e1",
            "Norwich", "Leeds", _KICKOFF, "football", _BEFORE_KICKOFF,
        )
        resolver.evict(now=_AFTER_KICKOFF_PLUS_GRACE)

        fresh = _resolve(
            resolver, "p", "e1",
            "Norwich", "Leeds", _KICKOFF, "football", _AFTER_KICKOFF_PLUS_GRACE,
        )

        assert fresh == original
        assert resolver.provider_identities_for_cluster(fresh) == (("p", "e1"),)

    def test_evict_retains_clusters_without_parseable_kickoff(self) -> None:
        resolver = EventClusterResolver()
        _resolve(
            resolver, "p", "e1",
            "Alpha", "Beta", "", "football", _BEFORE_KICKOFF,
        )

        evicted = resolver.evict(now=datetime.datetime(2025, 1, 1, tzinfo=_UTC))

        assert evicted == 0
        assert resolver.cluster_count == 1


class TestParameterConfigurability:
    """Constructor parameters must alter matching and pruning boundaries."""

    def test_high_confidence_threshold_blocks_lexical_merge(self) -> None:
        strict = EventClusterResolver(confidence_threshold=0.99)
        cluster_a = _resolve(
            strict, "p", "1",
            "Wolves", "Arsenal", _KICKOFF, "football", _BEFORE_KICKOFF,
        )
        cluster_b = _resolve(
            strict, "q", "2",
            "Wolverhampton", "Arsenal FC", _KICKOFF, "football", _BEFORE_KICKOFF,
        )

        assert cluster_a != cluster_b
        assert strict.cluster_count == 2

    def test_lenient_confidence_threshold_enables_weak_merge(self) -> None:
        kickoff = "2024-05-01T18:30:00Z"
        lenient = EventClusterResolver(confidence_threshold=0.5)
        cluster_a = _resolve(
            lenient, "p", "1",
            "Paris SG", "Lyon", kickoff, "football", _BEFORE_KICKOFF,
        )
        cluster_b = _resolve(
            lenient, "q", "2",
            "Paris Saint Germain", "Lyon", kickoff, "football", _BEFORE_KICKOFF,
        )

        assert cluster_a == cluster_b
        assert lenient.cluster_count == 1

    def test_default_threshold_rejects_weak_merge(self) -> None:
        kickoff = "2024-05-01T18:30:00Z"
        default = EventClusterResolver()
        cluster_a = _resolve(
            default, "p", "1",
            "Paris SG", "Lyon", kickoff, "football", _BEFORE_KICKOFF,
        )
        cluster_b = _resolve(
            default, "q", "2",
            "Paris Saint Germain", "Lyon", kickoff, "football", _BEFORE_KICKOFF,
        )

        assert cluster_a != cluster_b
        assert default.cluster_count == 2

    def test_retention_grace_controls_eviction_boundary(self) -> None:
        kickoff = "2024-05-01T18:30:00Z"
        one_hour_past_kickoff = datetime.datetime(2024, 5, 1, 19, 30, tzinfo=_UTC)

        short_grace = EventClusterResolver(retention_grace_hours=0.5)
        _resolve(
            short_grace, "p", "e1",
            "Norwich", "Leeds", kickoff, "football", _BEFORE_KICKOFF,
        )
        assert short_grace.evict(now=one_hour_past_kickoff) == 1
        assert short_grace.cluster_count == 0

        long_grace = EventClusterResolver(retention_grace_hours=6.0)
        _resolve(
            long_grace, "p", "e1",
            "Norwich", "Leeds", kickoff, "football", _BEFORE_KICKOFF,
        )
        assert long_grace.evict(now=one_hour_past_kickoff) == 0
        assert long_grace.cluster_count == 1

    def test_invalid_constructor_arguments_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            EventClusterResolver(confidence_threshold=1.5)
        with pytest.raises(ValueError):
            EventClusterResolver(temporal_match_window_hours=0.0)
        with pytest.raises(ValueError):
            EventClusterResolver(retention_grace_hours=-1.0)
        with pytest.raises(ValueError):
            EventClusterResolver(max_active_clusters=0)
        with pytest.raises(ValueError):
            EventClusterResolver(max_candidates_per_resolution=0)
