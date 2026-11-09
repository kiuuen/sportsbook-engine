"""LLM-assisted extraction coordinator for promotional campaigns.

Consumes distilled candidate text produced by ``PromotionScanner``, invokes
an environment-configured LLM endpoint, defensively parses structured
campaign payloads into ``bonus_engine.models`` aggregates, and routes each
candidate through ``bonus_engine.validator`` to assign the initial lifecycle
state. All interaction failures are contained and logged; no unhandled
crashes escape the public extraction methods.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .models import (
    CalculationBase,
    EvidenceCitation,
    Promotion,
    PromotionType,
    SourceProvenance,
    VerificationState,
)
from .scanner import ScanEvaluation
from .validator import InvariantValidator

__all__ = ["ExtractionConfig", "ExtractionError", "PromotionExtractor"]

logger = logging.getLogger(__name__)

TransportCallable = Callable[[str, Dict[str, str], bytes, float], Tuple[int, str]]


class ExtractionError(RuntimeError):
    """Raised when the LLM interaction cannot produce a usable response."""


@dataclass
class ExtractionConfig:
    """Environment-driven configuration for LLM interaction.

    Attributes:
        api_key: Bearer credential sent with each request.
        endpoint_url: Fully-qualified chat-completion endpoint URL.
        model_identifier: Model identifier requested at the endpoint.
        timeout_seconds: Per-attempt network timeout in seconds.
        max_retries: Maximum retry attempts for transient failures.
        retry_backoff_seconds: Base delay for exponential retry backoff.
        temperature: Sampling temperature for deterministic extraction.
    """

    api_key: str
    endpoint_url: str
    model_identifier: str
    timeout_seconds: float = 60.0
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    temperature: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("api_key", self.api_key),
            ("endpoint_url", self.endpoint_url),
            ("model_identifier", self.model_identifier),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"ExtractionConfig.{name} must be a non-empty string")
        for name, value in (
            ("timeout_seconds", self.timeout_seconds),
            ("retry_backoff_seconds", self.retry_backoff_seconds),
            ("temperature", self.temperature),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"ExtractionConfig.{name} must be a number")
        if self.timeout_seconds <= 0 or self.retry_backoff_seconds < 0:
            raise ValueError(
                "ExtractionConfig.timeout_seconds must be positive and "
                "retry_backoff_seconds must be non-negative"
            )
        if not isinstance(self.max_retries, int) or isinstance(self.max_retries, bool) \
                or self.max_retries < 0:
            raise ValueError("ExtractionConfig.max_retries must be a non-negative integer")

    @classmethod
    def from_environment(cls, prefix: str = "BONUS_LLM_") -> "ExtractionConfig":
        """Build a configuration from ``<PREFIX>API_KEY``, ``<PREFIX>ENDPOINT_URL``,
        ``<PREFIX>MODEL``, and optional numeric overrides.

        Raises:
            ValueError: If a required variable is missing or an override is invalid.
        """
        api_key = os.environ.get(f"{prefix}API_KEY")
        endpoint_url = os.environ.get(f"{prefix}ENDPOINT_URL")
        model_identifier = os.environ.get(f"{prefix}MODEL")
        missing = [
            name
            for name, value in (
                ("API_KEY", api_key),
                ("ENDPOINT_URL", endpoint_url),
                ("MODEL", model_identifier),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )
        return cls(
            api_key=api_key.strip(),
            endpoint_url=endpoint_url.strip(),
            model_identifier=model_identifier.strip(),
            timeout_seconds=cls._coerce_env_float(
                os.environ.get(f"{prefix}TIMEOUT_SECONDS"), 60.0
            ),
            max_retries=cls._coerce_env_int(
                os.environ.get(f"{prefix}MAX_RETRIES"), 3
            ),
            retry_backoff_seconds=cls._coerce_env_float(
                os.environ.get(f"{prefix}RETRY_BACKOFF_SECONDS"), 2.0
            ),
            temperature=cls._coerce_env_float(
                os.environ.get(f"{prefix}TEMPERATURE"), 0.0
            ),
        )

    @staticmethod
    def _coerce_env_float(raw: Optional[str], default: float) -> float:
        """Coerce an environment string to a positive float, or the default."""
        if raw is None or not raw.strip():
            return default
        try:
            value = float(raw.strip())
        except ValueError:
            raise ValueError(f"Environment numeric override is not a number: {raw!r}") from None
        if value <= 0:
            raise ValueError(f"Environment numeric override must be positive: {raw!r}")
        return value

    @staticmethod
    def _coerce_env_int(raw: Optional[str], default: int) -> int:
        """Coerce an environment string to a non-negative int, or the default."""
        if raw is None or not raw.strip():
            return default
        try:
            value = int(raw.strip())
        except ValueError:
            raise ValueError(f"Environment integer override is not an integer: {raw!r}") from None
        if value < 0:
            raise ValueError(f"Environment integer override must be non-negative: {raw!r}")
        return value


class PromotionExtractor:
    """Coordinates LLM extraction, model construction, and invariant routing.

    Args:
        config: Environment-driven LLM interaction configuration.
        validator: Optional invariant validator; a default is created when omitted.
        transport: Optional injectable transport callable
            ``(url, headers, body_bytes, timeout) -> (status_code, body_text)``;
            a stdlib ``urllib`` implementation is used when omitted.
    """

    _FENCE_PATTERN = re.compile(r"^```[\w-]*\s*|\s*```$", re.MULTILINE)
    _CAMPAIGN_COLLECTION_KEYS: Tuple[str, ...] = ("promotions", "campaigns")

    def __init__(
        self,
        config: ExtractionConfig,
        *,
        validator: Optional[InvariantValidator] = None,
        transport: Optional[TransportCallable] = None,
    ) -> None:
        if not isinstance(config, ExtractionConfig):
            raise TypeError("config must be an ExtractionConfig instance")
        self._config = config
        self._validator = validator if validator is not None else InvariantValidator()
        if not isinstance(self._validator, InvariantValidator):
            raise TypeError("validator must be an InvariantValidator instance")
        self._transport = transport if transport is not None else self._default_transport
        if not callable(self._transport):
            raise TypeError("transport must be a callable")

    @staticmethod
    def _default_transport(
        url: str, headers: Dict[str, str], body: bytes, timeout: float
    ) -> Tuple[int, str]:
        """POST the request via ``urllib``, returning ``(status, body_text)``.

        HTTP error responses are returned as normal results so retry logic can
        inspect the status code; transport-level failures propagate to the caller.
        """
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as err:
            return err.code, err.read().decode("utf-8", errors="replace")

    def _build_messages(self, candidate_text: str) -> List[Dict[str, str]]:
        """Build the chat messages enforcing the canonical JSON output schema."""
        promotion_types = ", ".join(item.value for item in PromotionType)
        calculation_bases = ", ".join(item.value for item in CalculationBase)
        schema = {
            "promotions": [
                {
                    "promotion_type": f"<one of: {promotion_types}>",
                    "title": "<promotion title>",
                    "wagering": {
                        "rollover_multiplier": "<number or null>",
                        "calculation_base": f"<one of: {calculation_bases} or null>",
                        "min_selection_odds": "<number or null>",
                        "max_selection_odds": "<number or null>",
                        "min_accumulator_odds": "<number or null>",
                        "min_accumulator_legs": "<integer or null>",
                        "max_stake_contribution_pct": "<number or null>",
                        "completion_deadline_days": "<integer or null>",
                    },
                    "temporal": {
                        "valid_from": "<ISO-8601 datetime or null>",
                        "valid_until": "<ISO-8601 datetime or null>",
                        "wagering_deadline_days": "<integer or null>",
                    },
                    "monetary": {
                        "min_qualifying_deposit": "<number or null>",
                        "max_qualifying_deposit": "<number or null>",
                        "min_bonus_amount": "<number or null>",
                        "max_bonus_amount": "<number or null>",
                        "max_conversion_amount": "<number or null>",
                        "currency": "<ISO-4217 code or null>",
                    },
                    "notes": "<free-text clarification or null>",
                    "evidence": [
                        {
                            "field_name": "<canonical field the value came from>",
                            "raw_text": "<verbatim promotional terms text>",
                            "locator": "<optional locator or null>",
                        }
                    ],
                }
            ]
        }
        system_prompt = (
            "You are a promotional-terms extraction engine for sports betting "
            "campaigns. Respond with a single JSON object matching exactly this "
            "schema:\n"
            f"{json.dumps(schema, indent=2)}\n"
            "Rules: use null for any value not stated in the text; never invent "
            "values; every numeric value must have a matching 'evidence' entry "
            "whose 'field_name' equals the canonical field name and whose "
            "'raw_text' quotes the source terms verbatim."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": candidate_text},
        ]

    def _invoke_llm(self, candidate_text: str) -> str:
        """Invoke the configured LLM endpoint with retries and backoff.

        Raises:
            ExtractionError: On exhausted retries or a non-retryable client error.
        """
        payload = {
            "model": self._config.model_identifier,
            "temperature": self._config.temperature,
            "messages": self._build_messages(candidate_text),
            "response_format": {"type": "json_object"},
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._config.api_key}",
        }
        last_error = "no attempts executed"
        for attempt in range(self._config.max_retries + 1):
            try:
                status, response_text = self._transport(
                    self._config.endpoint_url,
                    headers,
                    body,
                    self._config.timeout_seconds,
                )
            except (urllib.error.URLError, TimeoutError, OSError) as err:
                last_error = f"transport failure: {err}"
                logger.warning(
                    "LLM transport failure on attempt %d: %s", attempt + 1, err
                )
                status = 0
            if status == 200:
                return response_text
            last_error = f"HTTP status {status}"
            if status == 429 or 500 <= status < 600 or status == 0:
                if attempt < self._config.max_retries:
                    delay = self._config.retry_backoff_seconds * (2 ** attempt)
                    logger.warning(
                        "Retryable LLM failure (%s); retrying in %.1fs",
                        last_error,
                        delay,
                    )
                    time.sleep(delay)
                    continue
            else:
                raise ExtractionError(
                    f"Non-retryable LLM failure ({last_error})"
                )
        raise ExtractionError(f"LLM extraction failed after retries: {last_error}")

    def _extract_message_content(self, response_text: str) -> Optional[str]:
        """Defensively locate the assistant message content in a response body."""
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as err:
            logger.warning("LLM response body is not valid JSON: %s", err)
            return None
        if not isinstance(data, dict):
            return None
        choices = data.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
        content = data.get("content")
        return content if isinstance(content, str) else None

    def _parse_campaign_payload(self, content: str) -> List[Dict[str, Any]]:
        """Parse the assistant content into a list of campaign dictionaries."""
        cleaned = self._FENCE_PATTERN.sub("", content.strip()).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as err:
            logger.warning("LLM output is not valid JSON: %s", err)
            return []
        campaigns: List[Any]
        if isinstance(data, list):
            campaigns = data
        elif isinstance(data, dict):
            campaigns = next(
                (
                    data[key]
                    for key in self._CAMPAIGN_COLLECTION_KEYS
                    if isinstance(data.get(key), list)
                ),
                [data],
            )
        else:
            logger.warning("LLM output root is neither an object nor an array")
            return []
        valid = [item for item in campaigns if isinstance(item, dict)]
        if len(valid) != len(campaigns):
            logger.warning("Dropped %d non-object campaign entries", len(campaigns) - len(valid))
        return valid

    def _build_promotion(
        self,
        campaign: Dict[str, Any],
        target_domain: str,
        source_url: str,
        content_hash: str,
    ) -> Optional[Promotion]:
        """Construct one ``Promotion`` from a campaign payload, or ``None``."""
        payload = dict(campaign)
        raw_evidence = payload.pop("evidence", None)
        payload["target_domain"] = target_domain
        try:
            promotion = Promotion.from_dict(payload)
        except (ValueError, TypeError) as err:
            logger.warning("Skipping malformed campaign payload: %s", err)
            return None
        citations: List[EvidenceCitation] = []
        if isinstance(raw_evidence, list):
            for item in raw_evidence:
                try:
                    citations.append(EvidenceCitation.from_dict(item))
                except (ValueError, TypeError) as err:
                    logger.debug("Skipping malformed evidence citation: %s", err)
        try:
            promotion.provenance = SourceProvenance(
                source_url=source_url,
                content_hash=content_hash,
                captured_at=datetime.now(timezone.utc),
                citations=citations,
            )
        except (ValueError, TypeError) as err:
            logger.warning("Skipping campaign with invalid provenance: %s", err)
            return None
        return promotion

    def _apply_lifecycle(self, promotion: Promotion) -> None:
        """Route a parsed promotion to its initial lifecycle state."""
        report = self._validator.validate(promotion)
        if report.is_compliant:
            try:
                promotion.transition_to(VerificationState.INVARIANT_VALIDATED)
            except ValueError as err:
                logger.warning("Lifecycle transition failed: %s", err)
            return
        codes = ", ".join(code.value for code in report.discrepancy_codes)
        flag = f"flagged_for_review [{codes}]"
        promotion.notes = (
            f"{promotion.notes} | {flag}" if promotion.notes else flag
        )
        logger.info(
            "Promotion flagged for review (%s): %s",
            codes,
            "; ".join(report.violations),
        )

    def extract_promotions(
        self,
        candidate_text: str,
        *,
        target_domain: str,
        source_url: str,
        raw_content: Optional[str] = None,
    ) -> List[Promotion]:
        """Extract, validate, and lifecycle-route promotions from candidate text.

        Raises:
            TypeError: If arguments are not strings.
            ValueError: If ``target_domain`` or ``source_url`` is empty.
        """
        if not isinstance(candidate_text, str):
            raise TypeError("candidate_text must be a string")
        if not isinstance(target_domain, str) or not target_domain.strip():
            raise ValueError("target_domain must be a non-empty string")
        if not isinstance(source_url, str) or not source_url.strip():
            raise ValueError("source_url must be a non-empty string")
        if raw_content is not None and not isinstance(raw_content, str):
            raise TypeError("raw_content must be a string or None")
        content_hash = SourceProvenance.compute_content_hash(
            raw_content if raw_content is not None else candidate_text
        )
        try:
            response_text = self._invoke_llm(candidate_text)
        except ExtractionError as err:
            logger.error("LLM extraction aborted: %s", err)
            return []
        content = self._extract_message_content(response_text)
        if content is None:
            logger.error("LLM response contained no usable message content")
            return []
        promotions: List[Promotion] = []
        for campaign in self._parse_campaign_payload(content):
            promotion = self._build_promotion(
                campaign, target_domain, source_url, content_hash
            )
            if promotion is None:
                continue
            self._apply_lifecycle(promotion)
            promotions.append(promotion)
        return promotions

    def extract_from_evaluations(
        self,
        evaluations: Sequence[ScanEvaluation],
        *,
        target_domain: str,
        source_url: str,
        raw_content: Optional[str] = None,
    ) -> List[Promotion]:
        """Extract promotions from viable scanner evaluations, preserving order.

        Raises:
            TypeError: If ``evaluations`` is not a sequence of ``ScanEvaluation``.
        """
        if not isinstance(evaluations, (list, tuple)):
            raise TypeError("evaluations must be a list of ScanEvaluation instances")
        promotions: List[Promotion] = []
        for index, evaluation in enumerate(evaluations):
            if not isinstance(evaluation, ScanEvaluation):
                raise TypeError(
                    f"evaluations must contain ScanEvaluation instances; "
                    f"item {index} is {type(evaluation).__name__}"
                )
            if not evaluation.is_viable or not evaluation.candidate_window:
                continue
            promotions.extend(
                self.extract_promotions(
                    evaluation.candidate_window,
                    target_domain=target_domain,
                    source_url=source_url,
                    raw_content=raw_content,
                )
            )
        return promotions
