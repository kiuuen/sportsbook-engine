"""HTTP acquisition and HTML sanitization for promotional ingestion.

Provides a browser-impersonating HTTP fetcher built on ``curl_cffi`` and a
deterministic HTML sanitizer that reduces arbitrary markup to clean plain
text. The sanitizer strips script, style, noscript, header, footer, and nav
elements along with their inner content, removes comments and remaining tags,
unescapes entities, and normalizes whitespace so that a stable SHA-256 digest
can be computed over content that excludes dynamic markup.

A transport callable is injectable so tests never touch the network; the same
contract is used by ``bonus_engine.extractor``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Callable, Dict, List, Optional, Tuple

__all__ = ["BonusFetcher", "FetchResult", "HTMLSanitizer"]

DEFAULT_USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

FetchTransport = Callable[[str, Dict[str, str], float], Tuple[int, str]]

BLOCK_TAGS: frozenset[str] = frozenset(
    {"script", "style", "noscript", "header", "footer", "nav"}
)


@dataclass
class FetchResult:
    """Outcome of a single HTTP fetch attempt.

    Attributes:
        url: The requested URL, echoed verbatim.
        status_code: HTTP status code, or ``0`` for a transport-level failure.
        html: The decoded response body; empty string on failure.
        ok: ``True`` when the status is a 2xx/3xx and the body has content.
        error: A human-readable failure description, or ``None`` when ``ok``.
    """

    url: str
    status_code: int
    html: str
    ok: bool
    error: Optional[str] = None


class BonusFetcher:
    """Fetch promotional pages with browser-impersonating HTTP requests.

    Args:
        user_agent: Value sent in the ``User-Agent`` request header.
        timeout_seconds: Per-request network timeout in seconds.
        impersonate: ``curl_cffi`` browser profile used for TLS impersonation.
        transport: Optional injectable transport callable
            ``(url, headers, timeout) -> (status_code, body_text)``; a
            ``curl_cffi`` implementation is used when omitted. Transport-level
            failures are surfaced as ``(0, "")`` rather than raised.
    """

    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout_seconds: float = 30.0,
        impersonate: str = "chrome",
        transport: Optional[FetchTransport] = None,
    ) -> None:
        if not isinstance(user_agent, str) or not user_agent.strip():
            raise ValueError("user_agent must be a non-empty string")
        if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool):
            raise TypeError("timeout_seconds must be a number")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not isinstance(impersonate, str) or not impersonate.strip():
            raise ValueError("impersonate must be a non-empty string")
        self._user_agent = user_agent
        self._timeout_seconds = float(timeout_seconds)
        self._impersonate = impersonate
        if transport is None:
            self._transport: FetchTransport = self._default_transport
        else:
            if not callable(transport):
                raise TypeError("transport must be a callable")
            self._transport = transport

    def _default_transport(
        self, url: str, headers: Dict[str, str], timeout: float
    ) -> Tuple[int, str]:
        """GET the URL via ``curl_cffi``, returning ``(status_code, body_text)``.

        Any network or ``curl_cffi`` exception is caught and surfaced as
        ``(0, "")`` so callers can treat it as a transport-level failure
        without handling exceptions.
        """
        from curl_cffi import requests

        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=timeout,
                impersonate=self._impersonate,
            )
        except Exception:
            return 0, ""
        return response.status_code, response.text

    def fetch(self, url: str) -> FetchResult:
        """Fetch ``url`` and classify the outcome.

        Network and HTTP failures are returned as an unsuccessful
        ``FetchResult`` and never raised; only invalid input raises.

        Raises:
            ValueError: If ``url`` is not a non-empty string.
        """
        if not isinstance(url, str) or not url.strip():
            raise ValueError("url must be a non-empty string")
        headers: Dict[str, str] = {"User-Agent": self._user_agent}
        try:
            status_code, html = self._transport(url, headers, self._timeout_seconds)
        except Exception:
            status_code, html = 0, ""
        if not isinstance(status_code, int):
            status_code = 0
        if not isinstance(html, str):
            html = ""
        ok = 200 <= status_code < 400 and bool(html.strip())
        if status_code == 0:
            error: Optional[str] = "network error"
        elif ok:
            error = None
        else:
            error = f"HTTP {status_code}"
        return FetchResult(
            url=url,
            status_code=status_code,
            html=html,
            ok=ok,
            error=error,
        )


class _TagStripper(HTMLParser):
    """Collect character data outside a set of suppressed block elements.

    Nesting depth of suppressed elements is tracked so that the entire element,
    including inner content of nested suppressed elements, contributes no text.
    Comments and markup are ignored; character and entity references are
    converted to their unicode equivalents by the base parser.
    """

    def __init__(self, block_tags: frozenset[str]) -> None:
        super().__init__(convert_charrefs=True)
        self._block_tags = block_tags
        self._block_depth = 0
        self.text_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag in self._block_tags:
            self._block_depth += 1

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag in self._block_tags:
            self._block_depth = max(self._block_depth - 1, 0)

    def handle_endtag(self, tag: str) -> None:
        if tag in self._block_tags and self._block_depth > 0:
            self._block_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._block_depth == 0:
            self.text_parts.append(data)


class HTMLSanitizer:
    """Reduce arbitrary HTML to clean, deterministic plain text.

    Removal is exhaustive: suppressed block elements lose both their tags and
    their inner content, comments are dropped, all remaining tags are discarded,
    entities are unescaped, and whitespace runs collapse to a single space.
    Malformed markup never raises; it degrades to the text that can be parsed.
    """

    BLOCK_TAGS: frozenset[str] = BLOCK_TAGS
    MIN_CONTENT_CHARS: int = 100
    _WHITESPACE = re.compile(r"\s+")

    def sanitize(self, html: str) -> str:
        """Return the clean plain text of ``html`` with markup removed."""
        if not isinstance(html, str):
            return ""
        stripper = _TagStripper(self.BLOCK_TAGS)
        try:
            stripper.feed(html)
            stripper.close()
        except Exception:
            return self._WHITESPACE.sub(" ", "".join(stripper.text_parts)).strip()
        return self._WHITESPACE.sub(" ", "".join(stripper.text_parts)).strip()

    def clean_text_hash(self, text: str) -> str:
        """Return the SHA-256 hex digest of ``text`` encoded as UTF-8."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def hash_html(self, html: str) -> str:
        """Return the SHA-256 hex digest of the sanitized ``html``."""
        return self.clean_text_hash(self.sanitize(html))

    def is_insufficient(self, clean_text: str) -> bool:
        """Return ``True`` when ``clean_text`` holds fewer than the minimum chars."""
        return len(clean_text.strip()) < self.MIN_CONTENT_CHARS
