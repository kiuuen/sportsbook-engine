"""Level-1 discovery of promotional URLs for a sportsbook domain.

Implements sitemap parsing and root-anchor link discovery so a target domain
yields a bounded, deduplicated set of candidate promotion URLs. All promotional
path-keyword vocabulary and the robots ``Sitemap:`` directive come from
``bonus_engine/discovery_patterns.json`` rather than Python literals, keeping
domain knowledge out of code.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urljoin, urlparse

from bonus_engine.fetcher import BonusFetcher

__all__ = ["DiscoveryPatterns", "TargetDiscovery"]

_LOC_TAG_RE = re.compile(r"<loc>(.*?)</loc>", re.IGNORECASE | re.DOTALL)
_ANCHOR_HREF_RE = re.compile(
    r'<a[^>]*\bhref\s*=\s*(["\'])(.*?)\1', re.IGNORECASE | re.DOTALL
)
_HTTP_SCHEMES = frozenset({"http", "https"})


class DiscoveryPatterns:
    """Loaded promotional discovery patterns sourced from JSON.

    Attributes:
        path_keywords: Case-insensitive substrings that mark a URL path as
            promotional.
        robots_sitemap_directive: The ``robots.txt`` directive prefix that
            declares a sitemap URL.
        max_candidates_per_domain: Ceiling on returned candidate URLs per
            domain.
    """

    DEFAULT_PATH = Path(__file__).with_name("discovery_patterns.json")

    def __init__(self, path: Optional[Path] = None) -> None:
        source = path if path is not None else self.DEFAULT_PATH
        if not isinstance(source, Path):
            raise TypeError("path must be a Path or None")
        if not source.is_file():
            raise FileNotFoundError(f"discovery patterns not found: {source}")
        try:
            with source.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid discovery patterns JSON in {source}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("discovery patterns JSON must be an object")
        raw_keywords = payload.get("path_keywords")
        if not isinstance(raw_keywords, list) or not all(
            isinstance(item, str) and item.strip() for item in raw_keywords
        ):
            raise ValueError("path_keywords must be a list of non-empty strings")
        directive = payload.get("robots_sitemap_directive")
        if not isinstance(directive, str) or not directive.strip():
            raise ValueError("robots_sitemap_directive must be a non-empty string")
        maximum = payload.get("max_candidates_per_domain")
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
            raise ValueError("max_candidates_per_domain must be a positive integer")
        self.path_keywords: Tuple[str, ...] = tuple(raw_keywords)
        self.robots_sitemap_directive: str = directive
        self.max_candidates_per_domain: int = maximum

    def matches_path(self, url: str) -> bool:
        """Return ``True`` when any path keyword occurs in ``url``'s path.

        Matching is case-insensitive and operates on the path component only.
        """
        if not isinstance(url, str) or not url.strip():
            return False
        path = urlparse(url).path.lower()
        return any(keyword in path for keyword in self.path_keywords)


class TargetDiscovery:
    """Discover promotional candidate URLs for a domain via sitemaps and links.

    Args:
        fetcher: The injected ``BonusFetcher`` used for all HTTP acquisition.
        patterns: Loaded discovery patterns; defaults to the shipped JSON.
    """

    def __init__(
        self,
        fetcher: BonusFetcher,
        *,
        patterns: Optional[DiscoveryPatterns] = None,
    ) -> None:
        if not isinstance(fetcher, BonusFetcher):
            raise TypeError("fetcher must be a BonusFetcher")
        self._fetcher = fetcher
        self._patterns = patterns if patterns is not None else DiscoveryPatterns()

    @staticmethod
    def _host(domain: str) -> str:
        """Reduce ``domain`` to a lower-cased host without scheme or path."""
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError("domain must be a non-empty string")
        candidate = domain.strip()
        if "//" not in candidate:
            candidate = f"//{candidate}"
        parsed = urlparse(candidate)
        host = parsed.netloc
        if not host:
            raise ValueError(f"domain does not contain a host: {domain}")
        return host.lower()

    @staticmethod
    def _host_key(host: str) -> str:
        """Return a www-insensitive comparison key for ``host``."""
        if host.lower().startswith("www."):
            return host.lower()[len("www."):]
        return host.lower()

    def extract_robots_sitemaps(self, domain: str, robots_text: str) -> List[str]:
        """Return the sitemap URLs declared on each ``robots.txt`` line.

        Scanning is case-insensitive for the directive; lines that carry the
        directive contribute the trimmed URL that follows. De-duplicates while
        preserving order and returns ``[]`` when none are found.
        """
        if not isinstance(robots_text, str):
            raise TypeError("robots_text must be a string")
        directive = self._patterns.robots_sitemap_directive.lower()
        seen: set[str] = set()
        declared: List[str] = []
        for raw_line in robots_text.splitlines():
            line = raw_line.strip()
            lower = line.lower()
            if not lower.startswith(directive):
                continue
            candidate = line[len(self._patterns.robots_sitemap_directive):].strip()
            if not candidate:
                continue
            if candidate not in seen:
                seen.add(candidate)
                declared.append(candidate)
        return declared

    def fetch_sitemap(self, domain: str) -> List[str]:
        """Collect promotional sitemap URLs for ``domain`` (uncapped).

        Candidate sources are the robots-declared sitemap URLs (only when the
        robots fetch is ok) plus the default ``/sitemap.xml``, fetched in order.
        Every ``<loc>`` value of an ok sitemap is retained when its path matches
        the promotional vocabulary; results are de-duplicated preserving order.
        """
        host = self._host(domain)
        candidates: List[str] = []
        robots_url = f"https://{host}/robots.txt"
        robots_result = self._fetcher.fetch(robots_url)
        if robots_result.ok:
            for declared in self.extract_robots_sitemaps(
                host, robots_result.html
            ):
                if declared not in candidates:
                    candidates.append(declared)
        default_url = f"https://{host}/sitemap.xml"
        if default_url not in candidates:
            candidates.append(default_url)

        locs: List[str] = []
        for sitemap_url in candidates:
            result = self._fetcher.fetch(sitemap_url)
            if not result.ok:
                continue
            for match in _LOC_TAG_RE.findall(result.html):
                loc = match.strip()
                if loc and self._patterns.matches_path(loc) and loc not in locs:
                    locs.append(loc)
        return locs

    def extract_root_links(self, domain: str, html: str) -> List[str]:
        """Return internal promotional anchors resolved against the domain.

        Only http/https internal links whose netloc matches the target (case and
        www-insensitive) and whose path matches the promotional vocabulary are
        kept. De-duplicates preserving order.
        """
        if not isinstance(html, str):
            raise TypeError("html must be a string")
        host = self._host(domain)
        host_key = self._host_key(host)
        base = f"https://{host}/"
        links: List[str] = []
        for match in _ANCHOR_HREF_RE.findall(html):
            href = match[1]
            resolved = urljoin(base, href)
            parsed = urlparse(resolved)
            if parsed.scheme not in _HTTP_SCHEMES:
                continue
            if self._host_key(parsed.netloc) != host_key:
                continue
            if not self._patterns.matches_path(resolved):
                continue
            if resolved not in links:
                links.append(resolved)
        return links

    def discover(
        self, domain: str, *, homepage_html: Optional[str] = None
    ) -> List[str]:
        """Combine sitemap and root-link discovery, capped per domain.

        Sitemap candidates come first, then root-anchor candidates, all
        de-duplicated and capped at ``max_candidates_per_domain``. A target
        without discoverable promo URLs yields ``[]``.
        """
        results = self.fetch_sitemap(domain)
        if homepage_html is None:
            host = self._host(domain)
            frontpage = self._fetcher.fetch(f"https://{host}/")
            body: str = frontpage.html if frontpage.ok else ""
        else:
            if not isinstance(homepage_html, str):
                raise TypeError("homepage_html must be a string or None")
            body = homepage_html
        results.extend(self.extract_root_links(domain, body))
        de_duplicated: List[str] = []
        for url in results:
            if url not in de_duplicated:
                de_duplicated.append(url)
        return de_duplicated[: self._patterns.max_candidates_per_domain]
