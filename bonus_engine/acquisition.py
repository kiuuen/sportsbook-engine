"""Acquisition pipeline orchestrator for the bonus engine.

Coordinates the full content acquisition lifecycle for a monitored sportsbook
domain: Level-1 URL discovery (or an already pinned registry), Level-0
diff-watch hashing, scanner viability filtering, LLM-backed extraction, and
persistence updates. The pipeline never performs network I/O itself; all HTTP
and LLM interactions flow through injected ``BonusFetcher`` and
``PromotionExtractor`` dependencies so tests can route responses without
touching the network.

Targets whose discovery yields no usable promotional URLs, or whose every
candidate is unreachable or insufficiently explicit, transition to
``TargetAcquisitionState.NEEDS_BONUS_URL`` for manual operator pinning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from bonus_engine.database import BonusDatabaseManager
from bonus_engine.discovery import TargetDiscovery
from bonus_engine.extractor import PromotionExtractor
from bonus_engine.fetcher import BonusFetcher, HTMLSanitizer
from bonus_engine.models import TargetAcquisitionState
from bonus_engine.scanner import PromotionScanner

__all__ = ["BonusAcquisitionPipeline", "TargetProcessingResult"]

logger = logging.getLogger(__name__)


@dataclass
class TargetProcessingResult:
    """Aggregate outcome of one ``process_target`` invocation for a domain."""

    domain: str
    candidates: int = 0
    fetched: int = 0
    processed: int = 0
    not_viable: int = 0
    skipped_unchanged: int = 0
    skipped_insufficient: int = 0
    unreachable: int = 0
    promotions_extracted: int = 0
    needs_bonus_url: bool = False
    promotion_ids: List[str] = field(default_factory=list)


class BonusAcquisitionPipeline:
    """Orchestrate discovery, diff-watching, scanning, and extraction.

    Args:
        db: Persistence layer used for targets, source registry, and promotions.
        fetcher: HTTP fetcher used for candidate retrieval and discovery.
        scanner: Evaluates candidate text for promotional viability.
        extractor: Runs LLM extraction and lifecycle routing per viable window.
        discovery: Level-1 discovery; defaults to ``TargetDiscovery(fetcher)``.
        sanitizer: HTML sanitation/diff-hashing; defaults to ``HTMLSanitizer()``.

    Raises:
        TypeError: If any injected dependency is not of the expected type.
    """

    def __init__(
        self,
        db: BonusDatabaseManager,
        fetcher: BonusFetcher,
        scanner: PromotionScanner,
        extractor: PromotionExtractor,
        *,
        discovery: Optional[TargetDiscovery] = None,
        sanitizer: Optional[HTMLSanitizer] = None,
    ) -> None:
        if not isinstance(db, BonusDatabaseManager):
            raise TypeError("db must be a BonusDatabaseManager instance")
        if not isinstance(fetcher, BonusFetcher):
            raise TypeError("fetcher must be a BonusFetcher instance")
        if not isinstance(scanner, PromotionScanner):
            raise TypeError("scanner must be a PromotionScanner instance")
        if not isinstance(extractor, PromotionExtractor):
            raise TypeError("extractor must be a PromotionExtractor instance")
        if discovery is not None and not isinstance(discovery, TargetDiscovery):
            raise TypeError("discovery must be a TargetDiscovery instance or None")
        if sanitizer is not None and not isinstance(sanitizer, HTMLSanitizer):
            raise TypeError("sanitizer must be an HTMLSanitizer instance or None")
        self._db = db
        self._fetcher = fetcher
        self._scanner = scanner
        self._extractor = extractor
        self._discovery = discovery if discovery is not None else TargetDiscovery(fetcher)
        self._sanitizer = sanitizer if sanitizer is not None else HTMLSanitizer()

    def process_target(self, domain: str) -> TargetProcessingResult:
        """Acquire, scan, and extract promotions for one target domain.

        Raises:
            ValueError: If ``domain`` is blank.
        """
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError("domain must be a non-empty string")
        normalized = domain.strip()
        self._db.ensure_target(normalized)

        candidates: List[str] = []
        last_hashes: Dict[str, Optional[str]] = {}

        registered = self._db.get_registered_source_urls(normalized)
        if registered:
            candidates = [row["source_url"] for row in registered]
            last_hashes = {
                row["source_url"]: row["last_content_hash"] for row in registered
            }
        else:
            discovered = self._discovery.discover(normalized)
            if not discovered:
                self._db.set_target_acquisition_state(
                    normalized, TargetAcquisitionState.NEEDS_BONUS_URL
                )
                return TargetProcessingResult(
                    domain=normalized, candidates=0, needs_bonus_url=True
                )
            for url in discovered:
                self._db.register_source_url(normalized, url, pinned=False)
            candidates = list(discovered)
            last_hashes = {}

        result = TargetProcessingResult(domain=normalized, candidates=len(candidates))
        seen_promotion_ids: List[str] = []

        for url in candidates:
            res = self._fetcher.fetch(url)
            if not res.ok:
                result.unreachable += 1
                self._db.record_source_check(normalized, url, content_hash=None)
                logger.warning("Unreachable bonus source for %s: %s", normalized, url)
                continue

            clean = self._sanitizer.sanitize(res.html)
            if self._sanitizer.is_insufficient(clean):
                result.skipped_insufficient += 1
                self._db.record_source_check(normalized, url, content_hash=None)
                logger.warning(
                    "Insufficient bonus content for %s (%s): clean text < %d chars",
                    normalized,
                    url,
                    self._sanitizer.MIN_CONTENT_CHARS,
                )
                continue

            new_hash = self._sanitizer.clean_text_hash(clean)
            result.fetched += 1

            if last_hashes.get(url) == new_hash:
                result.skipped_unchanged += 1
                self._db.record_source_check(normalized, url, content_hash=new_hash)
                continue

            evaluation = self._scanner.evaluate_text(clean)
            if not evaluation.is_viable:
                result.not_viable += 1
                self._db.record_source_check(normalized, url, content_hash=new_hash)
                continue

            promotions = self._extractor.extract_from_evaluations(
                [evaluation],
                target_domain=normalized,
                source_url=url,
                raw_content=clean,
            )
            source_id, _changed = self._db.record_source(
                normalized, url, raw_content=clean, extraction_method="automated"
            )
            for promotion in promotions:
                pid = self._db.save_promotion(normalized, promotion, source_id=source_id)
                result.promotions_extracted += 1
                if pid not in seen_promotion_ids:
                    seen_promotion_ids.append(pid)
            result.processed += 1
            self._db.record_source_check(normalized, url, content_hash=new_hash)

        result.promotion_ids = seen_promotion_ids

        if (
            result.candidates > 0
            and result.processed == 0
            and result.not_viable == 0
            and result.skipped_unchanged == 0
        ):
            self._db.set_target_acquisition_state(
                normalized, TargetAcquisitionState.NEEDS_BONUS_URL
            )
            result.needs_bonus_url = True
        elif result.processed > 0:
            current = self._db.get_target_acquisition_state(normalized)
            if current in (
                TargetAcquisitionState.DISCOVERED,
                TargetAcquisitionState.NEEDS_BONUS_URL,
            ):
                self._db.set_target_acquisition_state(
                    normalized, TargetAcquisitionState.ACTIVE
                )
        return result

    def poll(self, domains: Sequence[str]) -> List[TargetProcessingResult]:
        """Process each target domain in order and return one result each.

        Raises:
            TypeError: If ``domains`` is not a non-string sequence.
            ValueError: If any domain is blank.
        """
        if isinstance(domains, (str, bytes)):
            raise TypeError("domains must be a sequence of domain strings")
        results: List[TargetProcessingResult] = []
        for domain in domains:
            results.append(self.process_target(domain))
        return results
