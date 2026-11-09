"""Unit and integration tests for bonus_engine.models and bonus_engine.database.

All database tests run against transient SQLite stores under ``tmp_path`` to
guarantee full isolation. Model tests exercise canonical domain objects
directly; no synthetic raw-payload JSON fixtures are invented.
"""

from __future__ import annotations

import datetime
import itertools
import json
import uuid
from typing import Any, Callable, Dict, Optional

import pytest

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

TARGET_DOMAIN = "tests-bonus-target.example"


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------

def _build_promotion(**overrides: Any) -> Promotion:
    """Build a fully populated, invariant-clean Promotion with sane defaults."""
    defaults: Dict[str, Any] = {
        "target_domain": TARGET_DOMAIN,
        "promotion_type": PromotionType.WELCOME,
        "title": "Welcome Offer",
        "wagering": WageringRequirement(
            rollover_multiplier=5.0,
            calculation_base=CalculationBase.DEPOSIT_PLUS_BONUS,
            min_selection_odds=1.5,
            max_selection_odds=10.0,
            completion_deadline_days=30,
        ),
        "temporal": TemporalValidity(
            valid_from="2024-01-01T00:00:00+00:00",
            valid_until="2024-12-31T23:59:59+00:00",
            wagering_deadline_days=30,
        ),
        "monetary": MonetaryConstraints(
            min_qualifying_deposit=10.0,
            max_bonus_amount=100.0,
            currency="EUR",
        ),
    }
    defaults.update(overrides)
    return Promotion(**defaults)


@pytest.fixture()
def db_manager(tmp_path: Any) -> BonusDatabaseManager:
    """Yield an isolated BonusDatabaseManager backed by a transient store."""
    manager = BonusDatabaseManager(db_path=tmp_path / "bonus_engine.db")
    yield manager
    manager.close()


@pytest.fixture()
def promotion_factory() -> Callable[..., Promotion]:
    """Factory producing uniquely identified, valid Promotion instances."""
    counter = itertools.count()

    def _factory(**overrides: Any) -> Promotion:
        overrides.setdefault("title", f"Promotion {next(counter)}")
        overrides.setdefault("promotion_id", uuid.uuid4().hex)
        return _build_promotion(**overrides)

    return _factory


# ---------------------------------------------------------------------------
# Model integrity: enums, coercion helpers, and nested structures
# ---------------------------------------------------------------------------

class TestEnumDefinitions:
    def test_promotion_type_values(self) -> None:
        assert PromotionType.WELCOME.value == "welcome"
        assert PromotionType.FREEBET.value == "freebet"
        assert PromotionType.OTHER.value == "other"

    def test_calculation_base_values(self) -> None:
        assert CalculationBase.BONUS_AMOUNT.value == "bonus_amount"
        assert CalculationBase.STAKE_PLUS_WINNINGS.value == "stake_plus_winnings"

    def test_verification_state_values(self) -> None:
        assert VerificationState.EXTRACTED.value == "extracted"
        assert VerificationState.OPERATOR_CONFIRMED.value == "operator_confirmed"


class TestTemporalValidity:
    def test_datetime_passthrough_gets_utc_tz(self) -> None:
        naive = datetime.datetime(2024, 5, 1, 12, 0, 0)
        temporal = TemporalValidity(valid_from=naive)
        assert temporal.valid_from is not None
        assert temporal.valid_from.tzinfo is not None

    def test_iso_string_parsing(self) -> None:
        temporal = TemporalValidity(valid_from="2024-01-01T00:00:00+00:00")
        assert temporal.valid_from is not None
        assert temporal.valid_from.year == 2024

    def test_invalid_string_yields_none(self) -> None:
        temporal = TemporalValidity(valid_from="not-a-date")
        assert temporal.valid_from is None

    def test_negative_deadline_rejected(self) -> None:
        with pytest.raises(ValueError):
            TemporalValidity(wagering_deadline_days=-1)

    def test_garbage_deadline_coerced_to_none(self) -> None:
        temporal = TemporalValidity(wagering_deadline_days="garbage")
        assert temporal.wagering_deadline_days is None

    def test_invariant_violation_reported(self) -> None:
        temporal = TemporalValidity(
            valid_from="2024-06-01T00:00:00+00:00",
            valid_until="2024-01-01T00:00:00+00:00",
        )
        assert "valid_until precedes valid_from" in temporal.validate_invariants()

    def test_roundtrip_serialization(self) -> None:
        original = TemporalValidity(
            valid_from="2024-01-01T00:00:00+00:00",
            valid_until="2024-12-31T23:59:59+00:00",
            wagering_deadline_days=14,
            timezone_name="Europe/Berlin",
        )
        rebuilt = TemporalValidity.from_dict(original.to_dict())
        assert rebuilt.to_dict() == original.to_dict()


class TestWageringRequirement:
    def test_defaults_applied(self) -> None:
        requirement = WageringRequirement(rollover_multiplier="3")
        assert requirement.rollover_multiplier == 3.0
        assert requirement.calculation_base == CalculationBase.BONUS_AMOUNT

    def test_missing_rollover_rejected(self) -> None:
        with pytest.raises(ValueError):
            WageringRequirement(rollover_multiplier=None)

    def test_negative_rollover_rejected(self) -> None:
        with pytest.raises(ValueError):
            WageringRequirement(rollover_multiplier=-1.0)

    def test_min_odds_below_one_flagged(self) -> None:
        requirement = WageringRequirement(rollover_multiplier=1.0, min_selection_odds=0.5)
        assert any("min_selection_odds" in v for v in requirement.validate_invariants())

    def test_max_odds_below_min_flagged(self) -> None:
        requirement = WageringRequirement(
            rollover_multiplier=1.0, min_selection_odds=2.0, max_selection_odds=1.5
        )
        assert any("max_selection_odds" in v for v in requirement.validate_invariants())

    def test_accumulator_odds_below_selection_flagged(self) -> None:
        requirement = WageringRequirement(
            rollover_multiplier=1.0, min_selection_odds=2.0, min_accumulator_odds=1.5
        )
        assert any("min_accumulator_odds" in v for v in requirement.validate_invariants())

    def test_accumulator_legs_below_two_flagged(self) -> None:
        requirement = WageringRequirement(rollover_multiplier=1.0, min_accumulator_legs=1)
        assert any("min_accumulator_legs" in v for v in requirement.validate_invariants())

    def test_stake_contribution_out_of_range_flagged(self) -> None:
        requirement = WageringRequirement(rollover_multiplier=1.0, max_stake_contribution_pct=150.0)
        assert any("max_stake_contribution_pct" in v for v in requirement.validate_invariants())

    def test_negative_deadline_flagged(self) -> None:
        requirement = WageringRequirement(rollover_multiplier=1.0, completion_deadline_days=-5)
        assert any("completion_deadline_days" in v for v in requirement.validate_invariants())

    def test_clean_requirement_has_no_violations(self) -> None:
        requirement = WageringRequirement(
            rollover_multiplier=4.0,
            min_selection_odds=1.8,
            max_selection_odds=9.0,
            min_accumulator_odds=2.0,
            min_accumulator_legs=3,
            max_stake_contribution_pct=50.0,
            completion_deadline_days=7,
        )
        assert requirement.validate_invariants() == []

    def test_roundtrip_serialization(self) -> None:
        original = WageringRequirement(
            rollover_multiplier=6.0,
            calculation_base=CalculationBase.FREEBET_STAKE,
            eligible_market_types=["match_result"],
        )
        rebuilt = WageringRequirement.from_dict(original.to_dict())
        assert rebuilt.to_dict() == original.to_dict()

    def test_from_dict_requires_rollover(self) -> None:
        with pytest.raises(ValueError):
            WageringRequirement.from_dict({})


class TestMonetaryConstraints:
    def test_negative_amount_rejected(self) -> None:
        with pytest.raises(ValueError):
            MonetaryConstraints(min_bonus_amount=-10.0)

    def test_inverted_deposit_bounds_flagged(self) -> None:
        constraints = MonetaryConstraints(min_qualifying_deposit=50.0, max_qualifying_deposit=10.0)
        assert any("qualifying_deposit" in v for v in constraints.validate_invariants())

    def test_inverted_bonus_bounds_flagged(self) -> None:
        constraints = MonetaryConstraints(min_bonus_amount=100.0, max_bonus_amount=20.0)
        assert any("bonus_amount" in v for v in constraints.validate_invariants())

    def test_garbage_coerced_to_none(self) -> None:
        constraints = MonetaryConstraints(min_bonus_amount=object())
        assert constraints.min_bonus_amount is None

    def test_roundtrip_serialization(self) -> None:
        original = MonetaryConstraints(
            min_qualifying_deposit=5.0, max_conversion_amount=250.0, currency="USD"
        )
        rebuilt = MonetaryConstraints.from_dict(original.to_dict())
        assert rebuilt.to_dict() == original.to_dict()


class TestEvidenceCitation:
    def test_empty_field_name_rejected(self) -> None:
        with pytest.raises(ValueError):
            EvidenceCitation(field_name="   ", raw_text="some terms")

    def test_non_string_raw_text_rejected(self) -> None:
        with pytest.raises(TypeError):
            EvidenceCitation(field_name="rollover", raw_text=123)

    def test_roundtrip_serialization(self) -> None:
        original = EvidenceCitation(
            field_name="rollover_multiplier",
            raw_text="5x bonus amount",
            locator="terms#section-2",
            captured_at="2024-03-01T10:00:00+00:00",
        )
        rebuilt = EvidenceCitation.from_dict(original.to_dict())
        assert rebuilt.field_name == original.field_name
        assert rebuilt.raw_text == original.raw_text
        assert rebuilt.locator == original.locator

    def test_from_dict_requires_field_name(self) -> None:
        with pytest.raises(ValueError):
            EvidenceCitation.from_dict({"raw_text": "text"})

    def test_from_dict_rejects_non_dict(self) -> None:
        with pytest.raises(TypeError):
            EvidenceCitation.from_dict(["not", "a", "dict"])


class TestSourceProvenance:
    def test_content_hash_matches_across_str_and_bytes(self) -> None:
        as_str = SourceProvenance.compute_content_hash("identical content")
        as_bytes = SourceProvenance.compute_content_hash(b"identical content")
        assert as_str == as_bytes
        assert len(as_str) == 64

    def test_content_hash_rejects_invalid_type(self) -> None:
        with pytest.raises(TypeError):
            SourceProvenance.compute_content_hash(12345)

    def test_missing_required_fields_rejected(self) -> None:
        with pytest.raises(ValueError):
            SourceProvenance(
                source_url="https://tests-bonus-target.example/promos",
                content_hash="",
                captured_at=datetime.datetime.now(datetime.timezone.utc),
            )

    def test_invalid_captured_at_rejected(self) -> None:
        with pytest.raises(ValueError):
            SourceProvenance(
                source_url="https://tests-bonus-target.example/promos",
                content_hash="abc",
                captured_at="not-a-date",
            )

    def test_add_citation_type_checked(self) -> None:
        provenance = SourceProvenance(
            source_url="https://tests-bonus-target.example/promos",
            content_hash="abc",
            captured_at=datetime.datetime.now(datetime.timezone.utc),
        )
        with pytest.raises(TypeError):
            provenance.add_citation({"field_name": "not-a-citation"})  # type: ignore[arg-type]

    def test_roundtrip_serialization_with_citations(self) -> None:
        original = SourceProvenance(
            source_url="https://tests-bonus-target.example/promos",
            content_hash="abc123",
            captured_at="2024-02-01T08:00:00+00:00",
            extraction_method="manual",
            citations=[
                EvidenceCitation(field_name="title", raw_text="Deposit Match 100%")
            ],
        )
        rebuilt = SourceProvenance.from_dict(original.to_dict())
        assert rebuilt.to_dict() == original.to_dict()


# ---------------------------------------------------------------------------
# Model integrity: Promotion aggregate
# ---------------------------------------------------------------------------

class TestPromotionAggregate:
    def test_defaults_generated(self) -> None:
        promotion = _build_promotion()
        assert promotion.promotion_id
        assert promotion.verification_state == VerificationState.EXTRACTED
        assert promotion.notes is None

    def test_blank_required_strings_rejected(self) -> None:
        with pytest.raises(ValueError):
            _build_promotion(title="   ")
        with pytest.raises(ValueError):
            _build_promotion(target_domain="")

    def test_invalid_nested_type_rejected(self) -> None:
        with pytest.raises(TypeError):
            _build_promotion(wagering={"rollover_multiplier": 5})

    def test_unknown_promotion_type_string_rejected(self) -> None:
        with pytest.raises(ValueError):
            _build_promotion(promotion_type="mystery-bonus")

    def test_string_promotion_type_coerced(self) -> None:
        promotion = _build_promotion(promotion_type="cashback")
        assert promotion.promotion_type == PromotionType.CASHBACK

    def test_to_dict_from_dict_roundtrip(self) -> None:
        original = _build_promotion(notes="Captured from promo landing page")
        rebuilt = Promotion.from_dict(original.to_dict())
        assert rebuilt.to_dict() == original.to_dict()
        assert rebuilt.fingerprint() == original.fingerprint()

    def test_from_dict_missing_required_fields_rejected(self) -> None:
        with pytest.raises(ValueError):
            Promotion.from_dict({"title": "Only a title"})

    def test_from_dict_rejects_non_dict(self) -> None:
        with pytest.raises(TypeError):
            Promotion.from_dict("not-a-dict")  # type: ignore[arg-type]

    def test_fingerprint_changes_on_content_change(self) -> None:
        first = _build_promotion()
        second = _build_promotion(title="Changed Title")
        assert first.fingerprint() != second.fingerprint()

    def test_fingerprint_stable_across_rebuilds(self) -> None:
        original = _build_promotion()
        rebuilt = Promotion.from_dict(original.to_dict())
        assert rebuilt.fingerprint() == original.fingerprint()

    def test_invariant_violations_aggregated_from_nested_structures(self) -> None:
        promotion = _build_promotion(
            wagering=WageringRequirement(rollover_multiplier=1.0, min_selection_odds=0.5),
            monetary=MonetaryConstraints(min_bonus_amount=100.0, max_bonus_amount=1.0),
        )
        violations = promotion.validate_invariants()
        assert any("min_selection_odds" in v for v in violations)
        assert any("bonus_amount" in v for v in violations)


class TestLifecycleTransitions:
    def test_legal_forward_transition(self) -> None:
        promotion = _build_promotion()
        promotion.transition_to(VerificationState.INVARIANT_VALIDATED)
        assert promotion.verification_state == VerificationState.INVARIANT_VALIDATED

    def test_full_lifecycle_path(self) -> None:
        promotion = _build_promotion()
        promotion.transition_to(VerificationState.INVARIANT_VALIDATED)
        promotion.transition_to(VerificationState.OPERATOR_CONFIRMED)
        assert promotion.verification_state == VerificationState.OPERATOR_CONFIRMED

    def test_direct_extracted_to_confirmed_rejected(self) -> None:
        promotion = _build_promotion()
        with pytest.raises(ValueError):
            promotion.transition_to(VerificationState.OPERATOR_CONFIRMED)

    def test_terminal_state_has_no_outgoing_transitions(self) -> None:
        promotion = _build_promotion()
        promotion.transition_to(VerificationState.REJECTED)
        with pytest.raises(ValueError):
            promotion.transition_to(VerificationState.EXTRACTED)

    def test_invalid_transition_target_rejected(self) -> None:
        promotion = _build_promotion()
        with pytest.raises(ValueError):
            promotion.transition_to("bogus-state")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Database lifecycle and idempotency
# ---------------------------------------------------------------------------

class TestDatabaseLifecycle:
    def test_repeated_instantiation_is_idempotent(
        self, tmp_path: Any, promotion_factory: Callable[..., Promotion]
    ) -> None:
        db_path = tmp_path / "bonus_engine.db"
        first = BonusDatabaseManager(db_path=db_path)
        first.ensure_target(TARGET_DOMAIN)
        first.save_promotion(TARGET_DOMAIN, promotion_factory())
        first.close()

        second = BonusDatabaseManager(db_path=db_path)
        try:
            promotions = second.get_promotions_by_domain(TARGET_DOMAIN)
            assert len(promotions) == 1
        finally:
            second.close()

    def test_connection_reopened_after_close(self, db_manager: BonusDatabaseManager) -> None:
        db_manager.close()
        conn = db_manager.get_connection()
        assert conn is not None
        conn.execute("SELECT 1").fetchone()

    def test_context_manager_closes_on_exit(self, tmp_path: Any) -> None:
        with BonusDatabaseManager(db_path=tmp_path / "ctx.db") as manager:
            manager.ensure_target(TARGET_DOMAIN)
        assert manager._conn is None

    def test_ensure_target_normalizes_and_is_stable(self, db_manager: BonusDatabaseManager) -> None:
        first_id = db_manager.ensure_target("  Example-Target.EXAMPLE ")
        second_id = db_manager.ensure_target("example-target.example")
        assert first_id == second_id

    def test_ensure_target_rejects_blank_domain(self, db_manager: BonusDatabaseManager) -> None:
        with pytest.raises(ValueError):
            db_manager.ensure_target("   ")


# ---------------------------------------------------------------------------
# Content fingerprinting and change detection
# ---------------------------------------------------------------------------

class TestSourceVersioning:
    URL = "https://tests-bonus-target.example/promotional-terms"

    def test_first_capture_reported_as_changed(self, db_manager: BonusDatabaseManager) -> None:
        source_id, changed = db_manager.record_source(TARGET_DOMAIN, self.URL, "v1 content")
        assert changed is True
        assert source_id > 0

    def test_identical_content_is_unchanged(self, db_manager: BonusDatabaseManager) -> None:
        first_id, _ = db_manager.record_source(TARGET_DOMAIN, self.URL, "v1 content")
        second_id, changed = db_manager.record_source(TARGET_DOMAIN, self.URL, "v1 content")
        assert changed is False
        assert second_id == first_id

    def test_altered_content_creates_new_version(self, db_manager: BonusDatabaseManager) -> None:
        first_id, _ = db_manager.record_source(TARGET_DOMAIN, self.URL, "v1 content")
        second_id, changed = db_manager.record_source(TARGET_DOMAIN, self.URL, "v2 content")
        assert changed is True
        assert second_id != first_id

        versions = db_manager.get_source_versions(TARGET_DOMAIN, self.URL)
        assert len(versions) == 2
        assert versions[0]["version"] == 2
        assert versions[1]["version"] == 1

    def test_bytes_content_decoded_and_deduplicated(
        self, db_manager: BonusDatabaseManager
    ) -> None:
        db_manager.record_source(TARGET_DOMAIN, self.URL, b"raw bytes")
        _, changed = db_manager.record_source(TARGET_DOMAIN, self.URL, "raw bytes")
        assert changed is False

    def test_invalid_raw_content_type_rejected(self, db_manager: BonusDatabaseManager) -> None:
        with pytest.raises(TypeError):
            db_manager.record_source(TARGET_DOMAIN, self.URL, 12345)  # type: ignore[arg-type]

    def test_blank_source_url_rejected(self, db_manager: BonusDatabaseManager) -> None:
        with pytest.raises(ValueError):
            db_manager.record_source(TARGET_DOMAIN, "  ", "content")

    def test_explicit_captured_at_persisted(self, db_manager: BonusDatabaseManager) -> None:
        captured = datetime.datetime(2024, 4, 1, 9, 30, tzinfo=datetime.timezone.utc)
        db_manager.record_source(TARGET_DOMAIN, self.URL, "content", captured_at=captured)
        latest = db_manager.get_latest_source(TARGET_DOMAIN, self.URL)
        assert latest is not None
        assert latest["captured_at"].startswith("2024-04-01T09:30")

    def test_has_content_changed_transitions(self, db_manager: BonusDatabaseManager) -> None:
        assert db_manager.has_content_changed(TARGET_DOMAIN, self.URL, "original") is True
        db_manager.record_source(TARGET_DOMAIN, self.URL, "original")
        assert db_manager.has_content_changed(TARGET_DOMAIN, self.URL, "original") is False
        assert db_manager.has_content_changed(TARGET_DOMAIN, self.URL, "modified") is True

    def test_versions_scoped_per_source_url(self, db_manager: BonusDatabaseManager) -> None:
        db_manager.record_source(TARGET_DOMAIN, f"{self.URL}/a", "shared content")
        source_id_b, changed_b = db_manager.record_source(
            TARGET_DOMAIN, f"{self.URL}/b", "shared content"
        )
        assert changed_b is True
        assert source_id_b > 0


# ---------------------------------------------------------------------------
# Relationship and multiplicity constraints
# ---------------------------------------------------------------------------

class TestPromotionMultiplicity:
    def test_multiple_promotions_coexist_without_collision(
        self, db_manager: BonusDatabaseManager, promotion_factory: Callable[..., Promotion]
    ) -> None:
        saved_ids = [
            db_manager.save_promotion(TARGET_DOMAIN, promotion_factory())
            for _ in range(5)
        ]
        assert len(set(saved_ids)) == 5
        stored = db_manager.get_promotions_by_domain(TARGET_DOMAIN)
        assert {p.promotion_id for p in stored} == set(saved_ids)

    def test_upsert_updates_existing_promotion_without_duplicates(
        self, db_manager: BonusDatabaseManager, promotion_factory: Callable[..., Promotion]
    ) -> None:
        promotion = promotion_factory()
        db_manager.save_promotion(TARGET_DOMAIN, promotion)
        promotion.title = "Updated Title"
        returned_id = db_manager.save_promotion(TARGET_DOMAIN, promotion)

        assert returned_id == promotion.promotion_id
        stored = db_manager.get_promotions_by_domain(TARGET_DOMAIN)
        assert len(stored) == 1
        assert stored[0].title == "Updated Title"

    def test_promotions_isolated_across_domains(
        self, db_manager: BonusDatabaseManager, promotion_factory: Callable[..., Promotion]
    ) -> None:
        db_manager.save_promotion("domain-one.example", promotion_factory())
        db_manager.save_promotion("domain-two.example", promotion_factory())
        assert len(db_manager.get_promotions_by_domain("domain-one.example")) == 1
        assert len(db_manager.get_promotions_by_domain("domain-two.example")) == 1

    def test_save_promotion_rejects_non_promotion(self, db_manager: BonusDatabaseManager) -> None:
        with pytest.raises(TypeError):
            db_manager.save_promotion(TARGET_DOMAIN, {"title": "not-a-promotion"})  # type: ignore[arg-type]

    def test_invariant_violations_logged_but_persisted(
        self, db_manager: BonusDatabaseManager, promotion_factory: Callable[..., Promotion]
    ) -> None:
        promotion = promotion_factory(
            wagering=WageringRequirement(rollover_multiplier=1.0, min_selection_odds=0.5)
        )
        saved_id = db_manager.save_promotion(TARGET_DOMAIN, promotion)
        fetched = db_manager.get_promotion(saved_id)
        assert fetched is not None
        assert fetched.validate_invariants() != []


# ---------------------------------------------------------------------------
# Lifecycle management and filtering
# ---------------------------------------------------------------------------

class TestPromotionQueries:
    def test_get_promotion_unknown_returns_none(self, db_manager: BonusDatabaseManager) -> None:
        assert db_manager.get_promotion(uuid.uuid4().hex) is None

    def test_get_promotion_rejects_blank_id(self, db_manager: BonusDatabaseManager) -> None:
        with pytest.raises(ValueError):
            db_manager.get_promotion("  ")

    def test_filter_by_domain_and_state(
        self, db_manager: BonusDatabaseManager, promotion_factory: Callable[..., Promotion]
    ) -> None:
        extracted = promotion_factory()
        validated = promotion_factory()
        db_manager.save_promotion(TARGET_DOMAIN, extracted)
        db_manager.save_promotion(TARGET_DOMAIN, validated)
        db_manager.transition_promotion(
            validated.promotion_id, VerificationState.INVARIANT_VALIDATED
        )

        all_promotions = db_manager.get_promotions_by_domain(TARGET_DOMAIN)
        assert len(all_promotions) == 2
        extracted_only = db_manager.get_promotions_by_domain(
            TARGET_DOMAIN, state=VerificationState.EXTRACTED
        )
        assert [p.promotion_id for p in extracted_only] == [extracted.promotion_id]

    def test_filter_by_state_across_domains(
        self, db_manager: BonusDatabaseManager, promotion_factory: Callable[..., Promotion]
    ) -> None:
        confirmed = promotion_factory()
        db_manager.save_promotion(TARGET_DOMAIN, confirmed)
        db_manager.transition_promotion(
            confirmed.promotion_id, VerificationState.INVARIANT_VALIDATED
        )
        db_manager.transition_promotion(
            confirmed.promotion_id, VerificationState.OPERATOR_CONFIRMED
        )
        db_manager.save_promotion("other-domain.example", promotion_factory())

        confirmed_only = db_manager.get_promotions_by_state(VerificationState.OPERATOR_CONFIRMED)
        assert [p.promotion_id for p in confirmed_only] == [confirmed.promotion_id]

    def test_filter_by_provenance(
        self, db_manager: BonusDatabaseManager, promotion_factory: Callable[..., Promotion]
    ) -> None:
        url = "https://tests-bonus-target.example/promos"
        promotion = promotion_factory(
            provenance=SourceProvenance(
                source_url=url,
                content_hash="hash-value",
                captured_at=datetime.datetime.now(datetime.timezone.utc),
                extraction_method="manual-review",
            )
        )
        db_manager.save_promotion(TARGET_DOMAIN, promotion)
        db_manager.save_promotion(TARGET_DOMAIN, promotion_factory())

        by_url = db_manager.get_promotions_by_provenance(source_url=url)
        assert [p.promotion_id for p in by_url] == [promotion.promotion_id]
        by_method = db_manager.get_promotions_by_provenance(extraction_method="manual-review")
        assert [p.promotion_id for p in by_method] == [promotion.promotion_id]

    def test_provenance_filter_requires_at_least_one_filter(
        self, db_manager: BonusDatabaseManager
    ) -> None:
        with pytest.raises(ValueError):
            db_manager.get_promotions_by_provenance()

    def test_state_filter_accepts_raw_string(
        self, db_manager: BonusDatabaseManager, promotion_factory: Callable[..., Promotion]
    ) -> None:
        db_manager.save_promotion(TARGET_DOMAIN, promotion_factory())
        extracted = db_manager.get_promotions_by_domain(
            TARGET_DOMAIN, state=VerificationState.EXTRACTED  # type: ignore[arg-type]
        )
        assert len(extracted) == 1


class TestAtomicTransitions:
    def test_legal_transition_persisted(
        self, db_manager: BonusDatabaseManager, promotion_factory: Callable[..., Promotion]
    ) -> None:
        promotion = promotion_factory()
        db_manager.save_promotion(TARGET_DOMAIN, promotion)
        advanced = db_manager.transition_promotion(
            promotion.promotion_id, VerificationState.INVARIANT_VALIDATED
        )
        assert advanced.verification_state == VerificationState.INVARIANT_VALIDATED
        reloaded = db_manager.get_promotion(promotion.promotion_id)
        assert reloaded is not None
        assert reloaded.verification_state == VerificationState.INVARIANT_VALIDATED

    def test_illegal_transition_rolls_back(
        self, db_manager: BonusDatabaseManager, promotion_factory: Callable[..., Promotion]
    ) -> None:
        promotion = promotion_factory()
        db_manager.save_promotion(TARGET_DOMAIN, promotion)
        with pytest.raises(ValueError):
            db_manager.transition_promotion(
                promotion.promotion_id, VerificationState.OPERATOR_CONFIRMED
            )
        reloaded = db_manager.get_promotion(promotion.promotion_id)
        assert reloaded is not None
        assert reloaded.verification_state == VerificationState.EXTRACTED

    def test_unknown_promotion_raises_key_error(
        self, db_manager: BonusDatabaseManager
    ) -> None:
        with pytest.raises(KeyError):
            db_manager.transition_promotion(uuid.uuid4().hex, VerificationState.REJECTED)

    def test_blank_promotion_id_rejected(self, db_manager: BonusDatabaseManager) -> None:
        with pytest.raises(ValueError):
            db_manager.transition_promotion("  ", VerificationState.REJECTED)


class TestOperatorAdjustments:
    def test_adjustment_routes_through_validation_to_confirmed(
        self, db_manager: BonusDatabaseManager, promotion_factory: Callable[..., Promotion]
    ) -> None:
        promotion = promotion_factory()
        db_manager.save_promotion(TARGET_DOMAIN, promotion)
        adjusted = db_manager.apply_operator_adjustment(
            promotion.promotion_id,
            {"title": "Operator Corrected Title"},
            operator_note="Title corrected during review",
        )
        assert adjusted.title == "Operator Corrected Title"
        assert adjusted.notes == "Title corrected during review"
        assert adjusted.verification_state == VerificationState.OPERATOR_CONFIRMED

        reloaded = db_manager.get_promotion(promotion.promotion_id)
        assert reloaded is not None
        assert reloaded.verification_state == VerificationState.OPERATOR_CONFIRMED
        assert reloaded.title == "Operator Corrected Title"

    def test_adjustment_from_validated_state(
        self, db_manager: BonusDatabaseManager, promotion_factory: Callable[..., Promotion]
    ) -> None:
        promotion = promotion_factory()
        db_manager.save_promotion(TARGET_DOMAIN, promotion)
        db_manager.transition_promotion(
            promotion.promotion_id, VerificationState.INVARIANT_VALIDATED
        )
        adjusted = db_manager.apply_operator_adjustment(
            promotion.promotion_id, {"notes": "Confirmed after review"}
        )
        assert adjusted.verification_state == VerificationState.OPERATOR_CONFIRMED

    def test_unknown_adjustment_keys_rejected_and_rolled_back(
        self, db_manager: BonusDatabaseManager, promotion_factory: Callable[..., Promotion]
    ) -> None:
        promotion = promotion_factory()
        db_manager.save_promotion(TARGET_DOMAIN, promotion)
        with pytest.raises(ValueError):
            db_manager.apply_operator_adjustment(
                promotion.promotion_id, {"nonexistent_field": "value"}
            )
        reloaded = db_manager.get_promotion(promotion.promotion_id)
        assert reloaded is not None
        assert reloaded.verification_state == VerificationState.EXTRACTED

    def test_empty_updates_rejected(self, db_manager: BonusDatabaseManager) -> None:
        with pytest.raises(ValueError):
            db_manager.apply_operator_adjustment(uuid.uuid4().hex, {})

    def test_unknown_promotion_raises_key_error(
        self, db_manager: BonusDatabaseManager
    ) -> None:
        with pytest.raises(KeyError):
            db_manager.apply_operator_adjustment(uuid.uuid4().hex, {"title": "New Title"})

    def test_invalid_adjustment_values_rejected(
        self, db_manager: BonusDatabaseManager, promotion_factory: Callable[..., Promotion]
    ) -> None:
        promotion = promotion_factory()
        db_manager.save_promotion(TARGET_DOMAIN, promotion)
        with pytest.raises(ValueError):
            db_manager.apply_operator_adjustment(
                promotion.promotion_id, {"promotion_type": "bogus-type"}
            )


# ---------------------------------------------------------------------------
# Defensive handling: corrupted payloads
# ---------------------------------------------------------------------------

class TestCorruptedPayloads:
    def test_corrupted_promotion_json_raises_value_error(
        self, db_manager: BonusDatabaseManager
    ) -> None:
        target_id = db_manager.ensure_target(TARGET_DOMAIN)
        conn = db_manager.get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO promotions (
                    target_id, promotion_id, promotion_type, title,
                    verification_state, promotion_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    target_id,
                    "corrupted-promotion-id",
                    "welcome",
                    "Corrupted Record",
                    "extracted",
                    "{this is not valid json",
                    db_manager._utc_now_iso(),
                    db_manager._utc_now_iso(),
                ),
            )
        with pytest.raises(ValueError):
            db_manager.get_promotion("corrupted-promotion-id")

    def test_non_dict_promotion_json_raises_value_error(
        self, db_manager: BonusDatabaseManager
    ) -> None:
        target_id = db_manager.ensure_target(TARGET_DOMAIN)
        conn = db_manager.get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO promotions (
                    target_id, promotion_id, promotion_type, title,
                    verification_state, promotion_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    target_id,
                    "list-promotion-id",
                    "welcome",
                    "List Payload Record",
                    "extracted",
                    json.dumps(["not", "a", "dict"]),
                    db_manager._utc_now_iso(),
                    db_manager._utc_now_iso(),
                ),
            )
        with pytest.raises(ValueError):
            db_manager.get_promotion("list-promotion-id")
