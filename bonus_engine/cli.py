"""Command-line subcommands for the bonus engine ingestion subsystem.

Operator-facing subcommands manage bonus target domains, run acquisition
polls, report audit counts across target acquisition and promotion
verification states, and drive interactive manual review of promotions
flagged by the ``InvariantValidator``.

Every handler builds its own ``BonusDatabaseManager`` or
``BonusAcquisitionPipeline`` from ``args.db`` unless one is explicitly
injected, and closes only the resources it opened itself. The handlers take
their dependencies through keyword-only injection points so tests can drive
them against transient databases and routed transports without any network
or LLM endpoint access.
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
from typing import Callable, Dict, Optional, Tuple

from bonus_engine.acquisition import BonusAcquisitionPipeline, TargetProcessingResult
from bonus_engine.database import BonusDatabaseManager
from bonus_engine.extractor import ExtractionConfig, PromotionExtractor
from bonus_engine.fetcher import BonusFetcher
from bonus_engine.models import TargetAcquisitionState, VerificationState
from bonus_engine.scanner import PromotionScanner

__all__ = [
    "BONUS_COMMAND_HANDLERS",
    "DEFAULT_DB_PATH",
    "build_pipeline",
    "cmd_bonus_audit",
    "cmd_bonus_poll",
    "cmd_bonus_review",
    "cmd_bonus_target_add",
    "register_bonus_subcommands",
]

logger = logging.getLogger(__name__)

#: Default bonus-engine SQLite database, overridable via ``BONUS_ENGINE_DB``.
DEFAULT_DB_PATH: str = os.environ.get("BONUS_ENGINE_DB", "./data/bonus_engine.db")

#: Promotion states surfaced by the ``bonus-audit`` summary, in lifecycle order.
_AUDIT_PROMOTION_STATES: Tuple[VerificationState, ...] = (
    VerificationState.INVARIANT_VALIDATED,
    VerificationState.MANUAL_REVIEW,
    VerificationState.OPERATOR_CONFIRMED,
    VerificationState.REJECTED,
)

_LLM_CONFIG_HINT: str = (
    "Bonus LLM extraction is not configured. Set BONUS_LLM_API_KEY, "
    "BONUS_LLM_ENDPOINT_URL, and BONUS_LLM_MODEL (optionally "
    "BONUS_LLM_TIMEOUT_SECONDS, BONUS_LLM_MAX_RETRIES, "
    "BONUS_LLM_RETRY_BACKOFF_SECONDS, BONUS_LLM_TEMPERATURE). "
    "Underlying error: {error}"
)

_REVIEW_PROMPT: str = "[c]onfirm / [r]eject / [s]kip / [q]uit: "


def _resolve_db_path(args: argparse.Namespace) -> str:
    """Resolve the bonus database path from parsed arguments or the default."""
    candidate = getattr(args, "db", None)
    if isinstance(candidate, str) and candidate.strip():
        return candidate
    return DEFAULT_DB_PATH


def _load_extraction_config() -> ExtractionConfig:
    """Build the LLM extraction config from the environment.

    Raises:
        RuntimeError: If the ``BONUS_LLM_*`` environment is incomplete, with
            an actionable message describing the required variables.
    """
    try:
        return ExtractionConfig.from_environment()
    except (ValueError, TypeError) as err:
        raise RuntimeError(_LLM_CONFIG_HINT.format(error=err)) from err


def build_pipeline(db_path: str) -> BonusAcquisitionPipeline:
    """Construct a network- and LLM-backed acquisition pipeline.

    Args:
        db_path: Path to the bonus-engine SQLite database.

    Raises:
        ValueError: If ``db_path`` is not a non-empty string.
        RuntimeError: If the ``BONUS_LLM_*`` extraction environment is unset.
    """
    if not isinstance(db_path, str) or not db_path.strip():
        raise ValueError("db_path must be a non-empty string")
    config = _load_extraction_config()
    db = BonusDatabaseManager(db_path)
    try:
        pipeline = BonusAcquisitionPipeline(
            db,
            BonusFetcher(),
            PromotionScanner(),
            PromotionExtractor(config),
        )
    except Exception:
        db.close()
        raise
    return pipeline


def cmd_bonus_target_add(
    args: argparse.Namespace, *, db: Optional[BonusDatabaseManager] = None
) -> int:
    """Register one or more target domains with an optional pinned URL."""
    owned_db: Optional[BonusDatabaseManager] = None
    try:
        if db is None:
            owned_db = BonusDatabaseManager(_resolve_db_path(args))
            db = owned_db
        domains = list(getattr(args, "domains", None) or [])
        if not domains:
            logger.error("[bonus-target-add] At least one target domain is required.")
            return 1
        pinned_url = getattr(args, "url", None)
        has_pinned = isinstance(pinned_url, str) and bool(pinned_url.strip())
        for domain in domains:
            db.ensure_target(domain)
            if has_pinned:
                db.register_source_url(domain, pinned_url, pinned=True)
                print(f"[+] Registered target {domain} (pinned: {pinned_url})")
            else:
                print(f"[+] Registered target {domain}")
        return 0
    except (ValueError, sqlite3.Error) as err:
        logger.error("[bonus-target-add] Registration failed: %s", err)
        return 1
    finally:
        if owned_db is not None:
            owned_db.close()


def _format_result_line(result: TargetProcessingResult) -> str:
    """Render a one-line per-domain acquisition summary."""
    return (
        f"[*] {result.domain}: candidates={result.candidates} "
        f"processed={result.processed} "
        f"promotions_extracted={result.promotions_extracted} "
        f"needs_bonus_url={result.needs_bonus_url}"
    )


def cmd_bonus_poll(
    args: argparse.Namespace, *, pipeline: Optional[BonusAcquisitionPipeline] = None
) -> int:
    """Run the acquisition pipeline over one domain or every active target."""
    db_path = _resolve_db_path(args)
    owned_db: Optional[BonusDatabaseManager] = None
    try:
        if pipeline is None:
            try:
                pipeline = build_pipeline(db_path)
            except (RuntimeError, ValueError, TypeError, sqlite3.Error) as err:
                logger.error(
                    "[bonus-poll] Failed to construct acquisition pipeline: %s", err
                )
                return 1

        domain = getattr(args, "domain", None)
        if isinstance(domain, str) and domain.strip():
            domains = [domain.strip()]
        else:
            owned_db = BonusDatabaseManager(db_path)
            domains = owned_db.get_targets_by_acquisition_state(
                TargetAcquisitionState.ACTIVE
            )

        if not domains:
            print("[-] No active targets to poll.")
            return 0

        for result in pipeline.poll(domains):
            print(_format_result_line(result))
        return 0
    except (ValueError, sqlite3.Error) as err:
        logger.error("[bonus-poll] Poll failed: %s", err)
        return 1
    finally:
        if owned_db is not None:
            owned_db.close()


def cmd_bonus_audit(
    args: argparse.Namespace, *, db: Optional[BonusDatabaseManager] = None
) -> int:
    """Print target counts by acquisition state and promotion counts by state."""
    owned_db: Optional[BonusDatabaseManager] = None
    try:
        if db is None:
            owned_db = BonusDatabaseManager(_resolve_db_path(args))
            db = owned_db
        print("\n=======================================================")
        print("                BONUS ENGINE AUDIT                     ")
        print("=======================================================")
        print("    Targets by acquisition state:")
        for state in TargetAcquisitionState:
            count = len(db.get_targets_by_acquisition_state(state))
            print(f"        {state.name}: {count}")
        print("    Promotions by verification state:")
        for state in _AUDIT_PROMOTION_STATES:
            count = len(db.get_promotions_by_state(state))
            print(f"        {state.name}: {count}")
        print()
        return 0
    except (ValueError, sqlite3.Error) as err:
        logger.error("[bonus-audit] Audit failed: %s", err)
        return 1
    finally:
        if owned_db is not None:
            owned_db.close()


def cmd_bonus_review(
    args: argparse.Namespace,
    *,
    db: Optional[BonusDatabaseManager] = None,
    input_fn: Callable[[str], str] = input,
) -> int:
    """Interactively confirm or reject promotions awaiting manual review."""
    owned_db: Optional[BonusDatabaseManager] = None
    reviewed = 0
    confirmed = 0
    rejected = 0
    try:
        if db is None:
            owned_db = BonusDatabaseManager(_resolve_db_path(args))
            db = owned_db
        domain = getattr(args, "domain", None)
        if isinstance(domain, str) and domain.strip():
            promotions = db.get_promotions_by_domain(
                domain.strip(), VerificationState.MANUAL_REVIEW
            )
        else:
            promotions = db.get_promotions_by_state(VerificationState.MANUAL_REVIEW)

        if not promotions:
            print("[-] No promotions awaiting manual review.")
            return 0

        quit_requested = False
        for promotion in promotions:
            print("-------------------------------------------------------")
            print(f"    Domain:       {promotion.target_domain}")
            print(f"    Promotion ID: {promotion.promotion_id}")
            print(f"    Title:        {promotion.title}")
            provenance = promotion.provenance
            source_url = getattr(provenance, "source_url", None)
            if source_url:
                print(f"    Source URL:   {source_url}")
            if promotion.notes:
                print(f"    Discrepancy:  {promotion.notes}")

            try:
                answer = input_fn(_REVIEW_PROMPT)
            except (EOFError, KeyboardInterrupt):
                print()
                logger.info("[bonus-review] Operator ended the review session.")
                quit_requested = True
                break

            decision = (answer or "").strip().lower()
            if decision.startswith("c"):
                db.transition_promotion(
                    promotion.promotion_id, VerificationState.OPERATOR_CONFIRMED
                )
                reviewed += 1
                confirmed += 1
                print(f"[+] Confirmed promotion {promotion.promotion_id}")
            elif decision.startswith("r"):
                db.transition_promotion(
                    promotion.promotion_id, VerificationState.REJECTED
                )
                reviewed += 1
                rejected += 1
                print(f"[-] Rejected promotion {promotion.promotion_id}")
            elif decision.startswith("s"):
                reviewed += 1
                print(f"[~] Skipped promotion {promotion.promotion_id}")
            elif decision.startswith("q"):
                logger.info("[bonus-review] Operator quit the review session.")
                quit_requested = True
                break
            else:
                print(f"[!] Unrecognized response {answer!r}; skipping.")
        if quit_requested:
            logger.info("[bonus-review] Review session stopped early.")

        print("=======================================================")
        print(
            f"    Reviewed: {reviewed} | Confirmed: {confirmed} | "
            f"Rejected: {rejected}"
        )
        return 0
    except (ValueError, KeyError, sqlite3.Error) as err:
        logger.error("[bonus-review] Review failed: %s", err)
        return 1
    finally:
        if owned_db is not None:
            owned_db.close()


def _add_db_argument(parser: argparse.ArgumentParser) -> None:
    """Attach the shared ``--db`` option to a bonus subcommand parser."""
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"Bonus-engine SQLite database path (default: {DEFAULT_DB_PATH}).",
    )


def register_bonus_subcommands(subparsers: "argparse._SubParsersAction") -> None:
    """Register the four bonus-engine subcommands on ``subparsers``."""
    target_add = subparsers.add_parser(
        "bonus-target-add",
        help="Register bonus target domain(s) with an optional pinned URL.",
    )
    target_add.add_argument(
        "domains",
        nargs="+",
        help="One or more target domains to register.",
    )
    target_add.add_argument(
        "--url",
        default=None,
        help="Pinned promotional source URL registered for every domain.",
    )
    _add_db_argument(target_add)

    poll = subparsers.add_parser(
        "bonus-poll",
        help="Run the acquisition pipeline over one or every active target.",
    )
    poll.add_argument(
        "--domain",
        default=None,
        help="Poll only this single domain instead of all active targets.",
    )
    _add_db_argument(poll)

    audit = subparsers.add_parser(
        "bonus-audit",
        help="Report target and promotion counts grouped by lifecycle state.",
    )
    _add_db_argument(audit)

    review = subparsers.add_parser(
        "bonus-review",
        help="Interactively confirm or reject promotions awaiting manual review.",
    )
    review.add_argument(
        "--domain",
        default=None,
        help="Review only promotions for this domain.",
    )
    _add_db_argument(review)


BONUS_COMMAND_HANDLERS: Dict[str, Callable[[argparse.Namespace], int]] = {
    "bonus-target-add": cmd_bonus_target_add,
    "bonus-poll": cmd_bonus_poll,
    "bonus-audit": cmd_bonus_audit,
    "bonus-review": cmd_bonus_review,
}
