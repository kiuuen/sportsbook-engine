"""Unit tests for the Level-1 sitemap and root-link discovery scraper.

All fixtures are loaded verbatim from the real captured HTML/XML under
``tests/fixtures/bonus_engine``. Fetching is exercised exclusively through an
injected routing transport so the suite never touches the network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from bonus_engine.discovery import DiscoveryPatterns, TargetDiscovery
from bonus_engine.fetcher import BonusFetcher

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "bonus_engine"

PLAYDOIT_ROBOTS_PATH = FIXTURE_DIR / "playdoit_robots.txt"
CALIENTE_ROBOTS_PATH = FIXTURE_DIR / "caliente_robots.txt"
CODERE_ROBOTS_PATH = FIXTURE_DIR / "codere_robots.txt"
PLAYDOIT_SITEMAP_PATH = FIXTURE_DIR / "playdoit_sitemap.xml"
CALIENTE_FRONTPAGE_PATH = FIXTURE_DIR / "caliente_frontpage.html"
PLAYDOIT_FRONTPAGE_PATH = FIXTURE_DIR / "playdoit_frontpage.html"


def _read_fixture(path: Path) -> str:
    """Return the fixture at ``path`` decoded as UTF-8."""
    return path.read_text(encoding="utf-8")


def _routed_fetcher(routes: Dict[str, Tuple[int, str]]) -> BonusFetcher:
    """Build a ``BonusFetcher`` whose transport routes by URL substring."""

    def transport(
        url: str, headers: Dict[str, str], timeout: float
    ) -> Tuple[int, str]:
        for frag, (status, body) in routes.items():
            if frag in url:
                return status, body
        return 404, ""

    return BonusFetcher(transport=transport)


CALIENTE_ROBOTS_TEXT = _read_fixture(CALIENTE_ROBOTS_PATH)
CODERE_ROBOTS_TEXT = _read_fixture(CODERE_ROBOTS_PATH)
PLAYDOIT_ROBOTS_TEXT = _read_fixture(PLAYDOIT_ROBOTS_PATH)
PLAYDOIT_SITEMAP_TEXT = _read_fixture(PLAYDOIT_SITEMAP_PATH)
CALIENTE_FRONTPAGE_TEXT = _read_fixture(CALIENTE_FRONTPAGE_PATH)
PLAYDOIT_FRONTPAGE_TEXT = _read_fixture(PLAYDOIT_FRONTPAGE_PATH)

CALIENTE_SITEMAP_URL = "https://www.caliente.mx/sitemap.xml"
CALIENTE_TERMINOS_URL = "https://www.caliente.mx/mas/ayuda/terminos-y-condiciones/"
PLAYDOIT_PROMOCIONES_URL = "https://www.site.com/promociones"
PLAYDOIT_FREEBET_URL = "https://www.site.com/promociones/modals/freebet"
EXTERNAL_PROMO_URL = "http://promo.winner.com/casino/docs/tst.pdf"


def test_extract_robots_sitemaps_caliente() -> None:
    """Caliente's robots file declares exactly its main sitemap."""
    discovery = TargetDiscovery(_routed_fetcher({}))

    sitemaps = discovery.extract_robots_sitemaps("caliente.mx", CALIENTE_ROBOTS_TEXT)

    assert sitemaps == [CALIENTE_SITEMAP_URL]


def test_extract_robots_sitemaps_codere_multiple() -> None:
    """Codere's robots file declares two distinct sitemap URLs in order."""
    discovery = TargetDiscovery(_routed_fetcher({}))

    sitemaps = discovery.extract_robots_sitemaps("codere.mx", CODERE_ROBOTS_TEXT)

    assert sitemaps == [
        "https://www.codere.mx/library/sitemap.xml",
        "https://www.codere.mx/ai-sitemap.xml",
    ]


def test_extract_robots_sitemaps_playdoit_404_html() -> None:
    """A Cloudflare 404 HTML body carries no sitemap directive."""
    discovery = TargetDiscovery(_routed_fetcher({}))

    sitemaps = discovery.extract_robots_sitemaps("playdoit.mx", PLAYDOIT_ROBOTS_TEXT)

    assert sitemaps == []


def test_fetch_sitemap_playdoit_uncapped_and_matching() -> None:
    """Sitemap locs are collected uncapped and match the promo vocabulary."""
    fetcher = _routed_fetcher(
        {
            "robots.txt": (404, PLAYDOIT_ROBOTS_TEXT),
            "sitemap.xml": (200, PLAYDOIT_SITEMAP_TEXT),
        }
    )
    discovery = TargetDiscovery(fetcher)

    urls = discovery.fetch_sitemap("playdoit.mx")

    assert urls
    assert len(urls) > 10
    assert PLAYDOIT_PROMOCIONES_URL in urls
    assert PLAYDOIT_FREEBET_URL in urls
    patterns = DiscoveryPatterns()
    assert all(patterns.matches_path(url) for url in urls)
    assert len(urls) == len(set(urls))


def test_extract_root_links_caliente_internal_only() -> None:
    """Only the internal promo anchor survives; external promo is excluded."""
    discovery = TargetDiscovery(_routed_fetcher({}))

    links = discovery.extract_root_links("caliente.mx", CALIENTE_FRONTPAGE_TEXT)

    assert links == [CALIENTE_TERMINOS_URL]
    assert EXTERNAL_PROMO_URL not in links


def test_extract_root_links_playdoit_spa_returns_empty() -> None:
    """The Playdoit SPA shell has no anchors, so no links are found."""
    discovery = TargetDiscovery(_routed_fetcher({}))

    links = discovery.extract_root_links("playdoit.mx", PLAYDOIT_FRONTPAGE_TEXT)

    assert links == []


def test_discover_caps_at_max_candidates_per_domain() -> None:
    """Discovery combines sitemap and home links, capped at the configured max."""
    fetcher = _routed_fetcher(
        {
            "robots.txt": (404, PLAYDOIT_ROBOTS_TEXT),
            "sitemap.xml": (200, PLAYDOIT_SITEMAP_TEXT),
        }
    )
    discovery = TargetDiscovery(fetcher)

    urls = discovery.discover("playdoit.mx", homepage_html=PLAYDOIT_FRONTPAGE_TEXT)

    assert len(urls) == discovery._patterns.max_candidates_per_domain
    assert len(urls) == 10
    assert len(urls) == len(set(urls))
    patterns = DiscoveryPatterns()
    assert all(patterns.matches_path(url) for url in urls)


def test_discover_returns_empty_when_no_routes() -> None:
    """No reachable sitemap and an anchor-less shell yield no candidates."""
    discovery = TargetDiscovery(_routed_fetcher({}))

    urls = discovery.discover("playdoit.mx", homepage_html=PLAYDOIT_FRONTPAGE_TEXT)

    assert urls == []


def test_discovery_patterns_loads_json() -> None:
    """The shipped patterns load from JSON with the documented vocabulary."""
    patterns = DiscoveryPatterns()

    assert patterns.matches_path("https://x.mx/promociones") is True
    assert patterns.matches_path("https://x.mx/casino") is False
    assert patterns.max_candidates_per_domain == 10


def test_extract_root_links_accepts_www_and_full_url_domain() -> None:
    """Domain inputs with scheme and www resolve the same internal link."""
    discovery = TargetDiscovery(_routed_fetcher({}))

    via_full = discovery.extract_root_links(
        "https://www.caliente.mx/", CALIENTE_FRONTPAGE_TEXT
    )
    via_www = discovery.extract_root_links("www.caliente.mx", CALIENTE_FRONTPAGE_TEXT)

    assert via_full == [CALIENTE_TERMINOS_URL]
    assert via_www == [CALIENTE_TERMINOS_URL]
