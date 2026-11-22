"""Unit tests for the bonus HTTP fetcher and HTML sanitizer.

All page fixtures are loaded verbatim from the real captured HTML/XML under
``tests/fixtures/bonus_engine``. Fetching is exercised exclusively through an
injected transport so the suite never touches the network; the default
``curl_cffi`` transport is validated by forcing its dependency to raise.

Note on fixture content: the captured Playdoit pages are single-page
application shells whose only visible character data is the document title,
so their sanitized text is identical and shorter than ``MIN_CONTENT_CHARS``.
The sitemap fixture carries real ``<loc>`` URL text and is used as the
content-rich counterpart where the task requires a distinct hash and a
non-insufficient verdict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pytest

from bonus_engine.fetcher import (
    BLOCK_TAGS,
    BonusFetcher,
    FetchResult,
    FetchTransport,
    HTMLSanitizer,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "bonus_engine"
FRONTPAGE_PATH = FIXTURE_DIR / "playdoit_frontpage.html"
PROMOCIONES_PATH = FIXTURE_DIR / "playdoit_promociones.html"
TERMINOS_PATH = FIXTURE_DIR / "playdoit_terminos.html"
SITEMAP_PATH = FIXTURE_DIR / "playdoit_sitemap.xml"

# Distinctive tokens that occur only inside suppressed element content in the
# real Playdoit shell: a script identifier, a CSS declaration, and a DOM API.
SCRIPT_ONLY_TOKEN = "google_tags_first_party"
STYLE_ONLY_TOKEN = "background:#fff"
SCRIPT_DOM_TOKEN = "document.createElement"
# Visible character data emitted from the real ``<title>`` element.
VISIBLE_TEXT_TOKEN = "Playdoit.mx"


def _read_fixture(path: Path) -> str:
    """Return the fixture at ``path`` decoded as UTF-8."""
    return path.read_text(encoding="utf-8")


def _transport_returning(body: str) -> FetchTransport:
    """Build a transport that always reports HTTP 200 with ``body``."""
    return lambda url, headers, timeout: (200, body)


def _transport_status(status_code: int) -> FetchTransport:
    """Build a transport that always reports ``status_code`` with no body."""
    return lambda url, headers, timeout: (status_code, "")


# ---------------------------------------------------------------------------
# BonusFetcher
# ---------------------------------------------------------------------------


def test_fetch_returns_ok_result_for_real_fixture() -> None:
    """An injected 200 transport yields an ok result carrying the real HTML."""
    html = _read_fixture(FRONTPAGE_PATH)
    fetcher = BonusFetcher(transport=_transport_returning(html))

    result = fetcher.fetch("https://www.playdoit.mx/")

    assert isinstance(result, FetchResult)
    assert result.url == "https://www.playdoit.mx/"
    assert result.status_code == 200
    assert result.html == html
    assert result.ok is True
    assert result.error is None


def test_fetch_reports_http_error_status() -> None:
    """A 404 transport yields a non-ok result with an HTTP error description."""
    fetcher = BonusFetcher(transport=_transport_status(404))

    result = fetcher.fetch("https://www.playdoit.mx/promociones")

    assert result.ok is False
    assert result.status_code == 404
    assert result.error == "HTTP 404"


def test_fetch_reports_network_error_for_zero_status() -> None:
    """A transport-level failure (status 0) yields the network error label."""
    fetcher = BonusFetcher(transport=_transport_status(0))

    result = fetcher.fetch("https://www.playdoit.mx/")

    assert result.ok is False
    assert result.status_code == 0
    assert result.error == "network error"


def test_fetch_treats_empty_body_as_not_ok() -> None:
    """A 200 response with only whitespace is not considered ok."""
    fetcher = BonusFetcher(transport=_transport_returning("   \n  "))

    result = fetcher.fetch("https://www.playdoit.mx/")

    assert result.ok is False
    assert result.error == "HTTP 200"


def test_fetch_contains_transport_exception() -> None:
    """A raising transport is contained and surfaces as a network error."""

    def transport(url: str, headers: Dict[str, str], timeout: float) -> Tuple[int, str]:
        raise OSError("connection reset")

    fetcher = BonusFetcher(transport=transport)

    result = fetcher.fetch("https://www.playdoit.mx/")

    assert result.ok is False
    assert result.error == "network error"


@pytest.mark.parametrize("bad_url", ["", "   ", "\t\n"])
def test_fetch_rejects_empty_or_blank_url(bad_url: str) -> None:
    """Empty and blank URLs are invalid input and raise ``ValueError``."""
    fetcher = BonusFetcher(transport=_transport_status(200))

    with pytest.raises(ValueError):
        fetcher.fetch(bad_url)


def test_fetch_rejects_non_string_url() -> None:
    """A non-string URL is invalid input and raises ``ValueError``."""
    fetcher = BonusFetcher(transport=_transport_status(200))

    with pytest.raises(ValueError):
        fetcher.fetch(None)  # type: ignore[arg-type]


def test_default_transport_surfaces_curl_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default curl_cffi transport maps a raised exception to (0, '')."""
    import curl_cffi.requests as curl_requests

    def raising_get(*args: object, **kwargs: object) -> object:
        raise RuntimeError("curl failure")

    monkeypatch.setattr(curl_requests, "get", raising_get)

    fetcher = BonusFetcher()
    result = fetcher.fetch("https://www.playdoit.mx/")

    assert result.status_code == 0
    assert result.ok is False
    assert result.error == "network error"


def test_default_transport_sends_user_agent_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default transport forwards the configured User-Agent header."""
    import curl_cffi.requests as curl_requests

    captured: Dict[str, Dict[str, str]] = {}

    class _Response:
        status_code = 200
        text = "<html><body>ok</body></html>"

    def fake_get(url: str, headers: Dict[str, str], timeout: float, impersonate: str) -> object:
        captured["headers"] = headers
        captured["impersonate"] = impersonate  # type: ignore[assignment]
        return _Response()

    monkeypatch.setattr(curl_requests, "get", fake_get)

    fetcher = BonusFetcher(user_agent="TestAgent/1.0", impersonate="firefox")
    result = fetcher.fetch("https://www.playdoit.mx/")

    assert result.ok is True
    assert captured["headers"]["User-Agent"] == "TestAgent/1.0"
    assert captured["impersonate"] == "firefox"  # type: ignore[index]


# ---------------------------------------------------------------------------
# HTMLSanitizer.sanitize
# ---------------------------------------------------------------------------


def test_sanitize_real_frontpage_strips_markup_and_block_content() -> None:
    """Suppressed element content and all tags are removed from the shell."""
    html = _read_fixture(FRONTPAGE_PATH)
    sanitizer = HTMLSanitizer()

    clean = sanitizer.sanitize(html)

    assert "<" not in clean
    assert ">" not in clean
    assert SCRIPT_ONLY_TOKEN not in clean
    assert SCRIPT_DOM_TOKEN not in clean
    assert STYLE_ONLY_TOKEN not in clean
    assert VISIBLE_TEXT_TOKEN in clean


def test_sanitize_block_tags_constant_is_complete() -> None:
    """The module exposes the documented suppressed tag set."""
    assert BLOCK_TAGS == frozenset(
        {"script", "style", "noscript", "header", "footer", "nav"}
    )


def test_sanitize_removes_nested_block_content() -> None:
    """Nested suppressed elements contribute no character data."""
    html = (
        "<body><nav>menu<footer>deep <script>var secret=1;</script>tail"
        "</footer></nav><p>visible</p></body>"
    )

    clean = HTMLSanitizer().sanitize(html)

    assert clean == "visible"


def test_sanitize_strips_comments_and_keeps_surrounding_text() -> None:
    """HTML comments are removed while surrounding text survives."""
    html = '<div>Antes <!-- oculto secretamente --> Después</div>'

    clean = HTMLSanitizer().sanitize(html)

    assert "oculto" not in clean
    assert "<!--" not in clean
    assert "Antes" in clean
    assert "Después" in clean


def test_sanitize_unescapes_entities_and_collapses_whitespace() -> None:
    """Entities are decoded and whitespace runs collapse to single spaces."""
    html = "<p>Hola&nbsp;&amp;\n\t  mundo &#33;</p>"

    clean = HTMLSanitizer().sanitize(html)

    assert clean == "Hola & mundo !"


def test_sanitize_never_raises_on_malformed_markup() -> None:
    """Malformed markup degrades gracefully without raising."""
    html = "<p>hola<div><b>bold</b></div><i>italic</p><span>tail</div>"

    clean = HTMLSanitizer().sanitize(html)

    assert isinstance(clean, str)
    assert "hola" in clean
    assert "tail" in clean


# ---------------------------------------------------------------------------
# HTMLSanitizer hashing
# ---------------------------------------------------------------------------


def test_hash_html_is_stable_and_matches_clean_text_hash() -> None:
    """Hashing is deterministic and equals hashing the sanitized text."""
    html = _read_fixture(FRONTPAGE_PATH)
    sanitizer = HTMLSanitizer()

    first = sanitizer.hash_html(html)
    second = sanitizer.hash_html(html)

    assert first == second
    assert first == sanitizer.clean_text_hash(sanitizer.sanitize(html))


def test_hash_html_differs_across_distinct_real_fixtures() -> None:
    """Two different captured fixtures produce different clean-text hashes."""
    sanitizer = HTMLSanitizer()

    frontpage_hash = sanitizer.hash_html(_read_fixture(FRONTPAGE_PATH))
    sitemap_hash = sanitizer.hash_html(_read_fixture(SITEMAP_PATH))

    assert frontpage_hash != sitemap_hash


def test_hash_html_ignores_dynamic_script_token() -> None:
    """A token injected into a removed <script> does not change the hash."""
    html = _read_fixture(FRONTPAGE_PATH)
    sanitizer = HTMLSanitizer()
    baseline = sanitizer.hash_html(html)

    mutated = html.replace("<script>", "<script>DYNAMIC_TOKEN_XYZ", 1)

    assert mutated != html
    assert sanitizer.hash_html(mutated) == baseline


def test_hash_html_changes_when_visible_text_changes() -> None:
    """A change to visible text does change the hash."""
    html = _read_fixture(FRONTPAGE_PATH)
    sanitizer = HTMLSanitizer()
    baseline = sanitizer.hash_html(html)

    mutated = html.replace(
        f"<title>{VISIBLE_TEXT_TOKEN}</title>",
        f"<title>{VISIBLE_TEXT_TOKEN} ACTUALIZADO</title>",
        1,
    )

    assert mutated != html
    assert sanitizer.hash_html(mutated) != baseline


# ---------------------------------------------------------------------------
# HTMLSanitizer.is_insufficient
# ---------------------------------------------------------------------------


def test_is_insufficient_true_for_short_js_shell() -> None:
    """A JS-only shell sanitizes below the minimum and is insufficient."""
    shell = (
        "<html><head><script>var app=1;</script></head>"
        '<body class="initializing"><noscript>Active Javascript</noscript>'
        '<div id="root"></div></body></html>'
    )
    sanitizer = HTMLSanitizer()

    clean = sanitizer.sanitize(shell)

    assert sanitizer.is_insufficient(clean) is True


def test_is_insufficient_true_for_real_frontpage_shell() -> None:
    """The real Playdoit shell exposes only its title, so it is insufficient."""
    sanitizer = HTMLSanitizer()

    clean = sanitizer.sanitize(_read_fixture(FRONTPAGE_PATH))

    assert len(clean.strip()) < sanitizer.MIN_CONTENT_CHARS
    assert sanitizer.is_insufficient(clean) is True


def test_is_insufficient_false_for_content_rich_fixture() -> None:
    """The captured sitemap exposes real URL text above the minimum length."""
    sanitizer = HTMLSanitizer()

    clean = sanitizer.sanitize(_read_fixture(SITEMAP_PATH))

    assert len(clean.strip()) >= sanitizer.MIN_CONTENT_CHARS
    assert sanitizer.is_insufficient(clean) is False
