"""Unit tests for :mod:`resolution.matcher`.

Covers symmetric competitor alignment, sub-word/prefix similarity,
qualifier and sport vetoes, temporal proximity scoring, and deterministic
canonical event ID generation. The module under test operates on scalar
strings rather than raw provider payloads, so no JSON fixtures are needed.
"""

import uuid

import pytest

from resolution.matcher import (
    _name_alignment_score,
    _tokens_match,
    compare_events,
    generate_canonical_event_id,
)

_KICKOFF = "2024-05-01T18:30:00Z"


class TestSymmetricCompetitorAlignment:
    def test_direct_pairing_matches(self) -> None:
        decision = compare_events(
            "Lakers", "Celtics", _KICKOFF, "Lakers", "Celtics", _KICKOFF
        )
        assert decision.is_match is True
        assert decision.name_similarity >= 0.9

    def test_inverted_pairing_matches(self) -> None:
        decision = compare_events(
            "Lakers", "Celtics", _KICKOFF, "Celtics", "Lakers", _KICKOFF
        )
        assert decision.is_match is True
        assert decision.name_similarity >= 0.9

    def test_alignment_orientation_resolves_to_best_pairing(self) -> None:
        inverted = _name_alignment_score(
            "paris saint germain", "marseille",
            "marseille", "paris saint germain",
        )
        misaligned = _name_alignment_score(
            "paris saint germain", "marseille",
            "paris saint germain", "lyon",
        )
        assert inverted == pytest.approx(1.0)
        assert misaligned < inverted


class TestSubWordAndPrefixSimilarity:
    def test_plural_root_prefix_token_match(self) -> None:
        assert _tokens_match("wolves", "wolverhampton") is True

    def test_single_letter_initial_token_match(self) -> None:
        assert _tokens_match("n", "novak") is True

    def test_abbreviated_team_name_resolves(self) -> None:
        decision = compare_events(
            "Wolves", "Fulham", _KICKOFF,
            "Wolverhampton Wanderers", "Fulham", _KICKOFF,
        )
        assert decision.is_match is True
        assert decision.name_similarity >= 0.5

    def test_surname_with_initial_resolves(self) -> None:
        decision = compare_events(
            "Djokovic N.", "Alcaraz C.", _KICKOFF,
            "Novak Djokovic", "Alcaraz C.", _KICKOFF,
        )
        assert decision.is_match is True
        assert decision.name_similarity >= 0.9


class TestQualifierVeto:
    def test_youth_qualifier_conflict_vetoes_match(self) -> None:
        decision = compare_events(
            "Arsenal", "Chelsea", _KICKOFF, "Arsenal U21", "Chelsea", _KICKOFF
        )
        assert decision.is_match is False
        assert decision.category_conflict is True
        assert decision.confidence == 0.0
        assert decision.name_similarity == pytest.approx(0.75)

    def test_gender_qualifier_conflict_vetoes_match(self) -> None:
        decision = compare_events(
            "Chelsea Women", "Arsenal", _KICKOFF, "Chelsea", "Arsenal", _KICKOFF
        )
        assert decision.is_match is False
        assert decision.category_conflict is True
        assert decision.name_similarity >= 0.7

    def test_consistent_qualifiers_do_not_veto(self) -> None:
        decision = compare_events(
            "Arsenal U21", "Chelsea", _KICKOFF,
            "Arsenal U21", "Chelsea", _KICKOFF,
        )
        assert decision.is_match is True
        assert decision.category_conflict is False


class TestSportMismatchVeto:
    def test_conflicting_sports_veto_match(self) -> None:
        decision = compare_events(
            "Real Madrid", "Barcelona", _KICKOFF,
            "Real Madrid", "Barcelona", _KICKOFF,
            sport_a="soccer", sport_b="basketball",
        )
        assert decision.is_match is False
        assert decision.category_conflict is False
        assert "sport mismatch" in decision.reason

    def test_absent_sport_on_one_side_does_not_veto(self) -> None:
        decision = compare_events(
            "Real Madrid", "Barcelona", _KICKOFF,
            "Real Madrid", "Barcelona", _KICKOFF,
            sport_a="soccer", sport_b=None,
        )
        assert decision.is_match is True


class TestTemporalProximityScoring:
    def test_identical_kickoffs_score_full(self) -> None:
        decision = compare_events(
            "Arsenal", "Chelsea", _KICKOFF, "Arsenal", "Chelsea", _KICKOFF
        )
        assert decision.temporal_score == pytest.approx(1.0)
        assert decision.is_match is True

    def test_kickoffs_within_match_window_score_full(self) -> None:
        decision = compare_events(
            "Arsenal", "Chelsea", "2024-05-01T18:00:00Z",
            "Arsenal", "Chelsea", "2024-05-01T20:00:00Z",
        )
        assert decision.temporal_score == pytest.approx(1.0)
        assert decision.is_match is True

    def test_kickoffs_in_decay_band_score_partially(self) -> None:
        decision = compare_events(
            "Arsenal", "Chelsea", "2024-05-01T18:00:00Z",
            "Arsenal", "Chelsea", "2024-05-02T18:00:00Z",
        )
        assert decision.temporal_score == pytest.approx(1.0 - 18.0 / 42.0)
        assert decision.is_match is True

    def test_kickoffs_beyond_max_window_fail_match(self) -> None:
        decision = compare_events(
            "Arsenal", "Chelsea", "2024-05-01T18:00:00Z",
            "Arsenal", "Chelsea", "2024-05-04T18:00:00Z",
        )
        assert decision.temporal_score == pytest.approx(0.0)
        assert decision.is_match is False

    def test_missing_timestamps_degrade_gracefully(self) -> None:
        decision = compare_events(
            "Arsenal", "Chelsea", None, "Arsenal", "Chelsea", None
        )
        assert decision.temporal_score is None
        assert decision.is_match is True
        assert decision.confidence == pytest.approx(0.85)

    def test_unparseable_timestamps_degrade_gracefully(self) -> None:
        decision = compare_events(
            "Arsenal", "Chelsea", "not-a-timestamp",
            "Arsenal", "Chelsea", "also-bad",
        )
        assert decision.temporal_score is None
        assert decision.is_match is True


class TestCanonicalEventIdGeneration:
    def test_home_away_swap_yields_identical_uuidv5(self) -> None:
        first = generate_canonical_event_id(
            "Real Madrid", "Barcelona", _KICKOFF, "soccer"
        )
        second = generate_canonical_event_id(
            "Barcelona", "Real Madrid", _KICKOFF, "soccer"
        )
        assert first == second
        assert uuid.UUID(first).version == 5

    def test_repeated_calls_are_deterministic(self) -> None:
        assert generate_canonical_event_id(
            "Arsenal", "Chelsea", _KICKOFF
        ) == generate_canonical_event_id("Arsenal", "Chelsea", _KICKOFF)

    def test_sport_discriminator_changes_identity(self) -> None:
        soccer_id = generate_canonical_event_id(
            "Real Madrid", "Barcelona", _KICKOFF, "soccer"
        )
        basketball_id = generate_canonical_event_id(
            "Real Madrid", "Barcelona", _KICKOFF, "basketball"
        )
        assert soccer_id != basketball_id

    def test_both_team_names_empty_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            generate_canonical_event_id("!!!", None, _KICKOFF)
        with pytest.raises(ValueError):
            generate_canonical_event_id(None, None, _KICKOFF)
