"""Tests for the bonus-engine CLI subcommands.

The handlers are invoked directly with hand-built ``argparse.Namespace``
objects so no process is spawned and no real terminal is required. Every
database is a transient ``BonusDatabaseManager`` under ``tmp_path``; the
``bonus-poll`` handler is driven with a real ``BonusAcquisitionPipeline``
whose HTTP fetcher is a routing transport bound to real captured fixtures
and whose LLM extractor is a real ``PromotionExtractor`` with an injected
transport. No mocks, stubs, or synthetic payloads are used.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pytest

from bonus_engine import cli
from bonus_engine.acquisition import BonusAcquisitionPipeline
from bonus_engine.database import BonusDatabaseManager
from bonus_engine.extractor import ExtractionConfig, PromotionExtractor
from bonus_engine.fetcher import BonusFetcher, HTMLSanitizer
from bonus_engine.models import (
    Promotion,
    PromotionType,
    TargetAcquisitionState,
    VerificationState,
)
from bonus_engine.scanner import PromotionScanner

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "bonus_engine"


def load_fixture(name: str) -> str:
    """Load one raw captured fixture verbatim from disk."""
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


CALIENTE_TERMINOS_HTML = load_fixture("caliente_terminos_y_condiciones.html")

TERMINOS_URL = "https://www.caliente.mx/mas/ayuda/terminos-y-condiciones/"

# The real Caliente terms page is genuinely promo-dense; its full-document scan
# is only gated out by boilerplate noise volume, so the same relaxed scanner
# parameters used by the acquisition tests are reused here to exercise the
# viable extraction branch end to end.
VIABLE_SCANNER = PromotionScanner(viability_threshold=0.30, noise_penalty_factor=0.2)


class RoutingTransport:
    """Fetch transport mapping URL substrings to ``(status, body)`` tuples."""

    def __init__(self, routes: Dict[str, Tuple[int, str]]) -> None:
        self.routes: Dict[str, Tuple[int, str]] = dict(routes)
        self.calls: List[str] = []

    def __call__(
        self, url: str, headers: Dict[str, str], timeout: float
    ) -> Tuple[int, str]:
        self.calls.append(url)
        for substring, (status, body) in self.routes.items():
            if substring in url:
                return status, body
        return 404, ""


def build_llm_response(source_text: str) -> str:
    """Build an LLM envelope with one citable and one fabricated campaign."""
    marker = source_text.lower().index("apuestas gratis")
    quote = source_text[marker : marker + 120]
    p1 = {
        "promotion_type": "freebet",
        "title": "Apuestas Gratis",
        "evidence": [{"field_name": "title", "raw_text": quote}],
    }
    p2 = {
        "promotion_type": "freebet",
        "title": "Promo Inventada",
        "evidence": [
            {
                "field_name": "title",
                "raw_text": "premio garantizado de 9999 pesos imposible xyzzy",
            }
        ],
    }
    return json.dumps({"content": json.dumps({"promotions": [p1, p2]})})


def make_llm_transport(
    source_text: Optional[str] = None,
) -> Tuple[
    Callable[[str, Dict[str, str], bytes, float], Tuple[int, str]],
    List[Dict[str, str]],
]:
    """Return a lazy LLM transport plus a shared call-record list."""
    calls: List[Dict[str, str]] = []
    payload = build_llm_response(source_text) if source_text else None

    def transport(
        url: str, headers: Dict[str, str], body: bytes, timeout: float
    ) -> Tuple[int, str]:
        calls.append({"url": url, "headers": headers, "timeout": timeout})
        if payload is None:
            return 404, ""
        return 200, payload

    return transport, calls


def make_pipeline(
    db: BonusDatabaseManager,
    routes: Dict[str, Tuple[int, str]],
    *,
    llm_transport: Callable[
        [str, Dict[str, str], bytes, float], Tuple[int, str]
    ],
) -> BonusAcquisitionPipeline:
    """Construct a real pipeline wired to a routing fetcher and LLM transport."""
    fetcher = BonusFetcher(transport=RoutingTransport(routes))
    config = ExtractionConfig(
        api_key="test",
        endpoint_url="http://llm.test/v1/chat/completions",
        model_identifier="test-model",
    )
    extractor = PromotionExtractor(config, transport=llm_transport)
    return BonusAcquisitionPipeline(db, fetcher, VIABLE_SCANNER, extractor)


def clean_terminos_text() -> str:
    """Return the sanitized clean text of the real terms fixture."""
    return HTMLSanitizer().sanitize(CALIENTE_TERMINOS_HTML)


def seed_promotion(
    db: BonusDatabaseManager,
    domain: str,
    title: str,
    state: VerificationState,
    promotion_id: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    """Persist a minimal promotion in a chosen lifecycle state."""
    promotion = Promotion(
        target_domain=domain,
        promotion_type=PromotionType.WELCOME,
        title=title,
        promotion_id=promotion_id,
        verification_state=state,
        notes=notes,
    )
    db.save_promotion(domain, promotion)
    return promotion.promotion_id


def scripted_input(responses: List[str]) -> Callable[[str], str]:
    """Return an ``input_fn`` replaying ``responses`` then raising EOFError."""
    queue: List[str] = list(responses)

    def _input(_prompt: str) -> str:
        if not queue:
            raise EOFError
        return queue.pop(0)

    return _input


@pytest.fixture
def db(tmp_path: Path) -> BonusDatabaseManager:
    """A fresh isolated bonus database per test."""
    return BonusDatabaseManager(str(tmp_path / "bonus_engine_cli.db"))


# ---------------------------------------------------------------------------
# 1 & 2. bonus-target-add
# ---------------------------------------------------------------------------


def test_target_add_registers_pinned_url(
    db: BonusDatabaseManager, capsys: pytest.CaptureFixture[str]
) -> None:
    args = argparse.Namespace(domains=["caliente.mx"], url=TERMINOS_URL, db=str(db.db_path))

    return_code = cli.cmd_bonus_target_add(args, db=db)

    assert return_code == 0
    registered = db.get_registered_source_urls("caliente.mx")
    assert len(registered) == 1
    assert registered[0]["source_url"] == TERMINOS_URL
    assert registered[0]["pinned"] is True
    output = capsys.readouterr().out
    assert "caliente.mx" in output
    assert TERMINOS_URL in output


def test_target_add_without_url_registers_bare_target(
    db: BonusDatabaseManager, capsys: pytest.CaptureFixture[str]
) -> None:
    args = argparse.Namespace(domains=["rivalo.mx"], url=None, db=str(db.db_path))

    return_code = cli.cmd_bonus_target_add(args, db=db)

    assert return_code == 0
    assert db.get_registered_source_urls("rivalo.mx") == []
    assert db.get_target_acquisition_state("rivalo.mx") is (
        TargetAcquisitionState.DISCOVERED
    )
    assert "rivalo.mx" in capsys.readouterr().out


def test_target_add_reports_value_error(
    db: BonusDatabaseManager, capsys: pytest.CaptureFixture[str]
) -> None:
    args = argparse.Namespace(domains=[""], url=None, db=str(db.db_path))

    return_code = cli.cmd_bonus_target_add(args, db=db)

    assert return_code == 1
    capsys.readouterr()


# ---------------------------------------------------------------------------
# 3. bonus-audit
# ---------------------------------------------------------------------------


def test_audit_reports_state_counts(
    db: BonusDatabaseManager, capsys: pytest.CaptureFixture[str]
) -> None:
    db.ensure_target("needs.mx", TargetAcquisitionState.NEEDS_BONUS_URL)
    seed_promotion(
        db, "audit.mx", "Validated", VerificationState.INVARIANT_VALIDATED
    )
    seed_promotion(db, "audit.mx", "Review", VerificationState.MANUAL_REVIEW)
    seed_promotion(db, "audit.mx", "Confirmed", VerificationState.OPERATOR_CONFIRMED)
    seed_promotion(db, "audit.mx", "Rejected", VerificationState.REJECTED)
    args = argparse.Namespace(db=str(db.db_path))

    return_code = cli.cmd_bonus_audit(args, db=db)

    assert return_code == 0
    output = capsys.readouterr().out
    assert "NEEDS_BONUS_URL: 1" in output
    assert "MANUAL_REVIEW: 1" in output
    assert "OPERATOR_CONFIRMED: 1" in output
    assert "REJECTED: 1" in output
    assert "INVARIANT_VALIDATED: 1" in output


# ---------------------------------------------------------------------------
# 4 & 5. bonus-poll
# ---------------------------------------------------------------------------


def test_poll_with_injected_pipeline_extracts_promotions(
    db: BonusDatabaseManager, capsys: pytest.CaptureFixture[str]
) -> None:
    db.register_source_url("caliente.mx", TERMINOS_URL)
    llm_transport, calls = make_llm_transport(clean_terminos_text())
    pipeline = make_pipeline(
        db,
        {"terminos-y-condiciones": (200, CALIENTE_TERMINOS_HTML)},
        llm_transport=llm_transport,
    )
    args = argparse.Namespace(domain="caliente.mx", db=str(db.db_path))

    return_code = cli.cmd_bonus_poll(args, pipeline=pipeline)

    assert return_code == 0
    output = capsys.readouterr().out
    assert "caliente.mx" in output
    match = re.search(r"promotions_extracted=(\d+)", output)
    assert match is not None
    assert int(match.group(1)) >= 1
    assert len(calls) >= 2


def test_poll_with_no_active_targets_reports_empty(
    db: BonusDatabaseManager, capsys: pytest.CaptureFixture[str]
) -> None:
    llm_transport, _ = make_llm_transport()
    pipeline = make_pipeline(db, {}, llm_transport=llm_transport)
    args = argparse.Namespace(domain=None, db=str(db.db_path))

    return_code = cli.cmd_bonus_poll(args, pipeline=pipeline)

    assert return_code == 0
    assert "No active targets to poll." in capsys.readouterr().out


def test_poll_pipeline_construction_failure_returns_error(
    db: BonusDatabaseManager,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for variable in ("BONUS_LLM_API_KEY", "BONUS_LLM_ENDPOINT_URL", "BONUS_LLM_MODEL"):
        monkeypatch.delenv(variable, raising=False)
    args = argparse.Namespace(domain=None, db=str(db.db_path))

    return_code = cli.cmd_bonus_poll(args)

    assert return_code == 1
    capsys.readouterr()


# ---------------------------------------------------------------------------
# 6. bonus-review
# ---------------------------------------------------------------------------


def test_review_confirms_promotion(
    db: BonusDatabaseManager, capsys: pytest.CaptureFixture[str]
) -> None:
    promotion_id = seed_promotion(
        db,
        "review.mx",
        "Needs Review",
        VerificationState.MANUAL_REVIEW,
        promotion_id="review-confirm-1",
        notes="flagged_for_review [citation_not_found]",
    )
    args = argparse.Namespace(domain=None, db=str(db.db_path))

    return_code = cli.cmd_bonus_review(
        args, db=db, input_fn=scripted_input(["c"])
    )

    assert return_code == 0
    persisted = db.get_promotion(promotion_id)
    assert persisted is not None
    assert persisted.verification_state is VerificationState.OPERATOR_CONFIRMED
    output = capsys.readouterr().out
    assert "review.mx" in output
    assert "citation_not_found" in output


def test_review_rejects_promotion(
    db: BonusDatabaseManager, capsys: pytest.CaptureFixture[str]
) -> None:
    promotion_id = seed_promotion(
        db,
        "review.mx",
        "Needs Review",
        VerificationState.MANUAL_REVIEW,
        promotion_id="review-reject-1",
    )
    args = argparse.Namespace(domain="review.mx", db=str(db.db_path))

    return_code = cli.cmd_bonus_review(
        args, db=db, input_fn=scripted_input(["r"])
    )

    assert return_code == 0
    persisted = db.get_promotion(promotion_id)
    assert persisted is not None
    assert persisted.verification_state is VerificationState.REJECTED
    capsys.readouterr()


def test_review_quit_leaves_promotion_unchanged(
    db: BonusDatabaseManager, capsys: pytest.CaptureFixture[str]
) -> None:
    promotion_id = seed_promotion(
        db,
        "review.mx",
        "Needs Review",
        VerificationState.MANUAL_REVIEW,
        promotion_id="review-quit-1",
    )
    args = argparse.Namespace(domain=None, db=str(db.db_path))

    return_code = cli.cmd_bonus_review(args, db=db, input_fn=scripted_input([]))

    assert return_code == 0
    persisted = db.get_promotion(promotion_id)
    assert persisted is not None
    assert persisted.verification_state is VerificationState.MANUAL_REVIEW
    capsys.readouterr()


def test_review_without_pending_promotions_reports_empty(
    db: BonusDatabaseManager, capsys: pytest.CaptureFixture[str]
) -> None:
    args = argparse.Namespace(domain=None, db=str(db.db_path))

    return_code = cli.cmd_bonus_review(
        args, db=db, input_fn=scripted_input(["c"])
    )

    assert return_code == 0
    assert "No promotions awaiting manual review." in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 7. register_bonus_subcommands
# ---------------------------------------------------------------------------


def test_register_bonus_subcommands_parses_arguments() -> None:
    parser = argparse.ArgumentParser(prog="sportsbook_engine_test")
    subparsers = parser.add_subparsers(dest="command")

    cli.register_bonus_subcommands(subparsers)

    assert parser.parse_args(["bonus-audit"]).command == "bonus-audit"
    parsed = parser.parse_args(
        ["bonus-target-add", "x.mx", "--url", "https://x.mx/bonos"]
    )
    assert parsed.command == "bonus-target-add"
    assert parsed.domains == ["x.mx"]
    assert parsed.url == "https://x.mx/bonos"
    assert parser.parse_args(["bonus-poll"]).domain is None
    assert parser.parse_args(["bonus-review", "--domain", "y.mx"]).domain == "y.mx"
    assert set(cli.BONUS_COMMAND_HANDLERS) == {
        "bonus-target-add",
        "bonus-poll",
        "bonus-audit",
        "bonus-review",
    }


# ---------------------------------------------------------------------------
# build_pipeline configuration contract
# ---------------------------------------------------------------------------


def test_build_pipeline_requires_llm_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for variable in ("BONUS_LLM_API_KEY", "BONUS_LLM_ENDPOINT_URL", "BONUS_LLM_MODEL"):
        monkeypatch.delenv(variable, raising=False)

    with pytest.raises(RuntimeError):
        cli.build_pipeline(str(tmp_path / "bonus_engine_cli.db"))


def test_build_pipeline_constructs_with_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BONUS_LLM_API_KEY", "test-key")
    monkeypatch.setenv("BONUS_LLM_ENDPOINT_URL", "http://llm.test/v1/chat/completions")
    monkeypatch.setenv("BONUS_LLM_MODEL", "test-model")

    pipeline = cli.build_pipeline(str(tmp_path / "bonus_engine_cli.db"))

    assert isinstance(pipeline, BonusAcquisitionPipeline)
    pipeline._db.close()


# ---------------------------------------------------------------------------
# main dispatcher integration
# ---------------------------------------------------------------------------


def test_interactive_menu_exposes_bonus_commands() -> None:
    import main

    menu_values = set(main._MENU_CHOICES.values())
    assert {"bonus-poll", "bonus-audit", "bonus-review"} <= menu_values
    assert "bonus-target-add" not in menu_values
    assert main._build_parser().parse_args(["bonus-audit"]).command == "bonus-audit"


def test_main_dispatches_bonus_audit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import main

    return_code = main.main(["bonus-audit", "--db", str(tmp_path / "dispatched.db")])

    assert return_code == 0
    assert "BONUS ENGINE AUDIT" in capsys.readouterr().out
